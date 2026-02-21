"""
VLIW packet optimization for the Hexagon JIT backend.

Hexagon instructions are grouped into packets of 1-4 instructions that
execute simultaneously. Parse bits [15:14] of each instruction indicate
packet boundaries:
  0b00 = duplex (two 16-bit sub-instructions -- never used here)
  0b01 = not end of packet (more instructions follow in this packet)
  0b10 = endloop0 (end of hardware loop0; also not-end-of-packet)
  0b11 = end of packet (last instruction)

By default, the JIT emits single-instruction packets (all parse bits = 0b11).
The PacketOptimizer post-pass identifies independent adjacent instructions
and bundles them into multi-instruction packets for improved IPC.

Hexagon packet rules:
- Up to 4 instructions per packet
- At most 1 branch/jump instruction (must be last in packet)
- At most 2 memory operations (loads/stores)
- No WAW (write-after-write) conflicts for same register
- No RAW (read-after-write) within packet -- reads use old values
  (exception: .new operands can read values written in same packet)
- Slot restrictions: loads on slot 0-1, ALU on slot 0-3, branches on slot 1

For simplicity, this initial implementation uses greedy sequential bundling
without slot assignment. Instructions are bundled when they have no data
dependencies and obey the structural constraints.
"""

from rpython.jit.backend.hexagon.arch import (
    INST_SIZE,
    PARSE_BITS_MASK, PARSE_BITS_SHIFT,
    PARSE_NOT_END, PARSE_END_PACKET,
    PARSE_ENDLOOP0, PARSE_ENDLOOP1,
)

# Instruction metadata flags
FLAG_BRANCH    = 0x01   # instruction is a branch/jump/call
FLAG_MEM_LOAD  = 0x02   # instruction is a memory load
FLAG_MEM_STORE = 0x04   # instruction is a memory store
FLAG_BARRIER   = 0x08   # can't be reordered (trap, barrier, etc.)


def _list_to_bitmask(regs):
    """Convert a list of register numbers to an integer bitmask."""
    mask = 0
    for reg in regs:
        mask |= (1 << reg)
    return mask


class InstrMeta(object):
    """Metadata for a single instruction in the packet optimizer."""
    _immutable_ = True

    def __init__(self, pos, reads_mask, writes_mask, flags=0):
        self.pos = pos              # byte offset in instruction buffer
        self.reads = reads_mask     # bitmask of register numbers read
        self.writes = writes_mask   # bitmask of register numbers written
        self.flags = flags

    def is_branch(self):
        return bool(self.flags & FLAG_BRANCH)

    def is_mem_load(self):
        return bool(self.flags & FLAG_MEM_LOAD)

    def is_mem_store(self):
        return bool(self.flags & FLAG_MEM_STORE)

    def is_barrier(self):
        return bool(self.flags & FLAG_BARRIER)


class PacketOptimizer(object):
    """Optimizes instruction packets for the Hexagon VLIW pipeline.

    Usage:
        optimizer = PacketOptimizer()
        # After emitting each instruction, annotate it:
        optimizer.annotate(pos, reads={0, 1}, writes={2})
        ...
        # At control flow boundaries (branches, labels):
        optimizer.flush(mc)

    Non-annotated instructions in the buffer are left as single-instruction
    packets. Only annotated instructions participate in bundling.
    """
    MAX_PACKET_SIZE = 4   # Hexagon packets have up to 4 instructions
    MAX_MEM_PER_PACKET = 2  # at most 2 memory ops per packet

    def __init__(self):
        self.pending = []       # list of InstrMeta
        self.total_packets = 0  # stats: packets produced
        self.total_instrs = 0   # stats: instructions processed
        self.bundled_instrs = 0  # stats: instructions that were bundled

    def annotate(self, pos, reads, writes, flags=0):
        """Record metadata for the instruction at buffer position `pos`.

        reads/writes are lists of register numbers (ints).
        """
        self.pending.append(InstrMeta(pos,
                                       _list_to_bitmask(reads),
                                       _list_to_bitmask(writes),
                                       flags))

    def barrier(self):
        """Insert a packet boundary -- forces a flush of the current packet."""
        if self.pending:
            self.pending.append(
                InstrMeta(-1, 0, 0, FLAG_BARRIER))

    def flush(self, mc):
        """Process pending instructions and set parse bits for optimal packets.

        mc: the InstrBuilder whose _buf will be patched.
        """
        if not self.pending:
            return

        packets = self._schedule(self.pending)
        self._apply_parse_bits(mc, packets)
        self.pending = []

    def _has_dependency(self, a, b):
        """Check if instructions a and b conflict within a packet.

        In Hexagon VLIW, instructions in the same packet execute
        simultaneously. All reads happen before all writes (register
        scoreboarding). So:
        - RAW (read-after-write) = conflict (b reads what a writes)
        - WAW (write-after-write) = conflict (both write same reg)
        - WAR (write-after-read) = OK in VLIW (reads happen first)
        """
        # RAW: b reads what a writes
        if a.writes & b.reads:
            return True
        # WAW: both write the same register
        if a.writes & b.writes:
            return True
        return False

    def _check_structural(self, packet, instr):
        """Check structural constraints for adding instr to packet."""
        if len(packet) >= self.MAX_PACKET_SIZE:
            return False

        # Only one branch per packet
        if instr.is_branch():
            for existing in packet:
                if existing.is_branch():
                    return False

        # At most 2 memory ops per packet
        if instr.is_mem_load() or instr.is_mem_store():
            mem_count = 0
            for existing in packet:
                if existing.is_mem_load() or existing.is_mem_store():
                    mem_count += 1
            if mem_count >= self.MAX_MEM_PER_PACKET:
                return False

        # Memory ordering: don't bundle potentially aliasing memory ops
        if instr.is_mem_store():
            for existing in packet:
                if existing.is_mem_store() or existing.is_mem_load():
                    return False
        if instr.is_mem_load():
            for existing in packet:
                if existing.is_mem_store():
                    return False

        return True

    def _can_add_to_packet(self, packet, instr):
        """Check if instr can be added to the current packet."""
        # Barriers can't be bundled
        if instr.is_barrier():
            return False

        if not self._check_structural(packet, instr):
            return False

        for existing in packet:
            if self._has_dependency(existing, instr):
                return False
        return True

    def _schedule(self, instrs):
        """Group instructions into packets greedily.

        Returns a list of packets, where each packet is a list of InstrMeta.
        Instructions within a packet are kept in original order.
        """
        packets = []
        current_packet = []

        for instr in instrs:
            if instr.is_barrier():
                # Barrier forces a packet boundary
                if current_packet:
                    packets.append(current_packet)
                    current_packet = []
                continue

            if self._can_add_to_packet(current_packet, instr):
                current_packet.append(instr)
            else:
                if current_packet:
                    packets.append(current_packet)
                current_packet = [instr]

        if current_packet:
            packets.append(current_packet)

        # Update stats
        for packet in packets:
            self.total_packets += 1
            self.total_instrs += len(packet)
            if len(packet) > 1:
                self.bundled_instrs += len(packet)

        return packets

    def _apply_parse_bits(self, mc, packets):
        """Patch parse bits in the instruction buffer to reflect packets.

        Within each packet:
        - All instructions except the last get parse bits 0b01 (not end)
        - The last instruction gets parse bits 0b11 (end of packet)
        """
        for packet in packets:
            for i, instr in enumerate(packet):
                if instr.pos < 0:
                    continue  # skip sentinel entries
                idx = instr.pos // INST_SIZE
                word = mc._buf[idx]
                old_word = word
                # Clear existing parse bits
                word &= ~PARSE_BITS_MASK
                if i == len(packet) - 1:
                    # Last instruction: end of packet
                    word |= (PARSE_END_PACKET << PARSE_BITS_SHIFT)
                else:
                    # Not last: more instructions follow in this packet
                    word |= (PARSE_NOT_END << PARSE_BITS_SHIFT)
                mc._buf[idx] = word
                if 16 <= idx <= 27:
                    from rpython.rlib.debug import debug_print
                    debug_print("HEXDBG parse_bits idx=", idx,
                                " old=", old_word, " new=", word)

    def get_stats(self):
        """Return optimization statistics for debugging."""
        return {
            'total_packets': self.total_packets,
            'total_instrs': self.total_instrs,
            'bundled_instrs': self.bundled_instrs,
        }
