"""
Hexagon code builder for the JIT backend.

Provides AbstractHexagonBuilder (base with pseudo-instructions),
InstrBuilder (main code emitter with block management), and
OverwritingBuilder (for patching existing code).
"""

from rpython.jit.backend.hexagon.arch import (
    WORD, INST_SIZE,
    PARSE_END_PACKET, PARSE_BITS_SHIFT, PARSE_BITS_MASK,
    PARSE_ENDLOOP0, PARSE_ENDLOOP1,
    SINT16_IMM_MIN, SINT16_IMM_MAX,
    ABI_STACK_ALIGN,
)
from rpython.jit.backend.hexagon import registers as r
from rpython.jit.backend.hexagon.instruction_builder import gen_all_instr_assemblers
from rpython.jit.backend.hexagon.vector_ext import HVXInstructionMixin
from rpython.rlib.objectmodel import we_are_translated
from rpython.rlib.rarithmetic import intmask


class AbstractHexagonBuilder(HVXInstructionMixin):
    """Base class for Hexagon instruction emission.

    Inherits HVX vector instruction methods from HVXInstructionMixin.
    Subclasses must implement write32().
    """

    def write32(self, value):
        """Write a 32-bit instruction word to the code buffer."""
        raise NotImplementedError

    # -----------------------------------------------------------------------
    # Pseudo-instructions (convenience wrappers)
    # -----------------------------------------------------------------------

    def NOP(self):
        """Emit a NOP (R0 = R0, no side effects)."""
        self.TFR(r.r0.value, r.r0.value)

    def MV(self, rd, rs):
        """Move register: Rd = Rs."""
        self.TFR(rd, rs)

    def NEG(self, rd, rs):
        """Negate: Rd = -Rs = sub(#0, Rs).
        Implemented as Rd = sub(Rd_zero, Rs) via ADDI then SUB,
        or more efficiently: Rd = sub(0, Rs) which is SUB(rd, r0, rs)
        if we keep r0=0... but Hexagon has no zero register.
        Use: Rd = add(Rs, #0) then Rd = sub(Rd, Rs) ... no.
        Simplest: Rd = sub(Rd_tmp, Rs) where Rd_tmp = 0.
        Or: Rd = #0; Rd = sub(Rd, Rs).
        """
        # Two-instruction sequence: Rd = #0; Rd = sub(Rd, Rs)
        self.TFRSI(rd, 0)
        self.SUB(rd, rd, rs)

    def NOT(self, rd, rs):
        """Bitwise NOT: Rd = ~Rs = xor(Rs, #-1).
        Since we don't have XORI with 16-bit imm that covers -1,
        use: Rd = #-1; Rd = xor(Rd, Rs).
        """
        self.TFRSI(rd, -1)
        self.XOR(rd, rd, rs)

    def ZXTB(self, rd, rs):
        """Zero-extend byte: Rd = zxtb(Rs) = and(Rs, #0xFF)."""
        self.ANDI(rd, rs, 0xFF)

    def DFMUL(self, rdd, rss, rtt):
        """Double-precision float multiply (multi-instruction sequence).

        Hexagon has no single dfmpy instruction. Double-float multiply is
        decomposed into partial products:
          Rdd  = dfmpyll(Rss, Rtt)      // low * low
          Rdd += dfmpylh(Rss, Rtt)      // accumulate low * high
          Rdd += dfmpylh(Rtt, Rss)      // accumulate high * low
          Rdd += dfmpyhh(Rss, Rtt)      // accumulate high * high

        For full IEEE compliance with denormalized inputs, dfmpyfix should
        be applied first, but we skip that for now (matches -ffast-math).
        """
        self.DFMPYLL(rdd, rss, rtt)
        self.DFMPYLH_ACC(rdd, rss, rtt)
        self.DFMPYLH_ACC(rdd, rtt, rss)
        self.DFMPYHH_ACC(rdd, rss, rtt)

    def LI(self, rd, imm):
        """Load immediate (pseudo): load a 32-bit integer into Rd.
        Uses 1-2 instructions depending on the value range.
        """
        self.gen_load_int(rd, imm)

    # -----------------------------------------------------------------------
    # Immediate loading
    # -----------------------------------------------------------------------

    def gen_load_int(self, rd, imm):
        """Load a 32-bit integer constant into register rd.

        Strategy:
        - If imm fits in s16 [-32768, 32767]: single TFRSI
        - Otherwise: TFRSI low half, then set high half via shift+or sequence

        Uses a scratch register for the high half.  If rd happens to be
        scratch1, we use scratch2 instead to avoid clobbering rd.
        """
        if SINT16_IMM_MIN <= imm <= SINT16_IMM_MAX:
            self.TFRSI(rd, imm)
        else:
            lo = imm & 0xFFFF
            hi = (imm >> 16) & 0xFFFF
            # Pick a scratch register that differs from rd
            if rd == r.scratch1.value:
                scratch = r.scratch2.value
            else:
                scratch = r.scratch1.value
            # Load sign-extended low 16 bits
            if lo >= 0x8000:
                self.TFRSI(rd, intmask(lo | (~0xFFFF)))
            else:
                self.TFRSI(rd, lo)
            if hi == 0:
                if lo >= 0x8000:
                    # lo was sign-extended, giving wrong high bits.
                    # Zero-extend to keep only the low 16 bits.
                    self.ZXTH(rd, rd)
            elif lo >= 0x8000:
                # rd has sign-extended lo (wrong high bits).
                # scratch = hi << 16; rd = zxth(rd) | scratch
                self.TFRSI(scratch, intmask(hi | (~0xFFFF)) if hi >= 0x8000 else hi)
                self.S2_ASL_I_R(scratch, scratch, 16)
                self.ZXTH(rd, rd)
                self.OR(rd, rd, scratch)
            else:
                # lo is positive, rd has correct low half with zeros in high
                self.TFRSI(scratch, intmask(hi | (~0xFFFF)) if hi >= 0x8000 else hi)
                self.S2_ASL_I_R(scratch, scratch, 16)
                self.OR(rd, rd, scratch)

    def gen_load_int_pair(self, rdd_even, imm64):
        """Load a 64-bit integer constant into a register pair.

        rdd_even is the even register number of the pair.
        """
        lo32 = imm64 & 0xFFFFFFFF
        hi32 = (imm64 >> 32) & 0xFFFFFFFF
        # Load low word into even register, high word into odd register
        self.gen_load_int(rdd_even, intmask(lo32))
        self.gen_load_int(rdd_even + 1, intmask(hi32))

    # -----------------------------------------------------------------------
    # Memory access helpers
    # -----------------------------------------------------------------------

    def load_word(self, rd, rs, offset):
        """Rd = memw(Rs + #offset)"""
        self.LDW(rd, rs, offset)

    def store_word(self, rs, rt, offset):
        """memw(Rs + #offset) = Rt"""
        self.STW(rs, rt, offset)

    def load_double(self, rdd, rs, offset):
        """Rdd = memd(Rs + #offset)"""
        self.LDD(rdd, rs, offset)

    def store_double(self, rs, rtt, offset):
        """memd(Rs + #offset) = Rtt"""
        self.STD(rs, rtt, offset)

    # -----------------------------------------------------------------------
    # Frame access helpers (FP-relative)
    # -----------------------------------------------------------------------

    def load_from_jitframe(self, rd, offset):
        """Load a word from the JITFRAME at fp+offset."""
        self.LDW(rd, r.fp.value, offset)

    def store_to_jitframe(self, rt, offset):
        """Store a word to the JITFRAME at fp+offset."""
        self.STW(r.fp.value, rt, offset)

    def load_pair_from_jitframe(self, rdd, offset):
        """Load a double from the JITFRAME at fp+offset."""
        self.LDD(rdd, r.fp.value, offset)

    def store_pair_to_jitframe(self, rtt, offset):
        """Store a double to the JITFRAME at fp+offset."""
        self.STD(r.fp.value, rtt, offset)

    # -----------------------------------------------------------------------
    # Stack operations
    # -----------------------------------------------------------------------

    def push_reg(self, reg):
        """Push a register onto the stack (SP-relative)."""
        self.ADDI(r.sp.value, r.sp.value, -WORD)
        self.STW(r.sp.value, reg, 0)

    def pop_reg(self, reg):
        """Pop a register from the stack."""
        self.LDW(reg, r.sp.value, 0)
        self.ADDI(r.sp.value, r.sp.value, WORD)

    # -----------------------------------------------------------------------
    # Branch helpers
    # -----------------------------------------------------------------------

    def JUMP(self, offset):
        """Unconditional PC-relative jump."""
        self.J_IMM(offset)

    def CALL(self, offset):
        """Unconditional PC-relative call."""
        self.CALL_IMM(offset)

    def TRAP0(self, imm8=0):
        """Emit trap instruction (useful for guard stubs).
        trap0(#imm8)
        Encoding: 0b01010100_00000000_pp_00000000_IIIIIIII
        where I = imm8
        """
        bits = (0b01010100 << 24) | (0b11 << 14) | (imm8 & 0xFF)
        self.write32(bits)

    def BARRIER(self):
        """Memory barrier instruction: barrier.
        Encoding: 0b10100000_00000000_pp_00000000_00000000
        """
        self.write32((0b10100000 << 24) | (0b11 << 14))

    def ISYNC(self):
        """Instruction sync barrier."""
        self.write32((0b01010111 << 24) | (0b11 << 14))

    def ICINVA(self, rs):
        """Instruction cache invalidate address: icinva(Rs)."""
        # Simplified encoding; primarily used for cache management
        self.write32((0b10100110 << 24) | (0b11 << 14) | (int(rs) << 16))

    def DCCLEANA(self, rs):
        """Data cache clean address: dccleana(Rs)."""
        self.write32((0b10100110 << 24) | (0b11 << 14) |
                     (int(rs) << 16) | (1 << 5))


# Add all instruction assembler methods to the base class
gen_all_instr_assemblers(AbstractHexagonBuilder)


class OverwritingBuilder(AbstractHexagonBuilder):
    """Builder that overwrites existing code in-place.

    Used for patching jump targets, guard resolution, etc.
    """

    def __init__(self, mc, start, max_bytes):
        self._mc = mc
        self._start = start
        self._pos = 0
        self._max_bytes = max_bytes

    def write32(self, value):
        assert self._pos + 4 <= self._max_bytes
        pos = self._start + self._pos
        self._mc.overwrite32(pos, value)
        self._pos += 4

    def currpos(self):
        return self._start + self._pos

    def get_relative_pos(self, break_basic_block=True):
        return self._start + self._pos


class InstrBuilder(AbstractHexagonBuilder):
    """Main instruction builder with block management and constant pool support.

    Emits instructions into an internal buffer, which can later be copied
    to executable memory.
    """

    def __init__(self):
        self._buf = []
        self._pos = 0
        self.ops_offset = {}
        self.rawstart = 0

    def write32(self, value):
        """Append a 32-bit word to the instruction buffer."""
        from rpython.rlib.debug import debug_print
        idx = len(self._buf)
        masked = value & 0xFFFFFFFF
        self._buf.append(masked)
        self._pos += INST_SIZE
        if 16 <= idx <= 27:
            debug_print("HEXDBG write32 idx=", idx, " val=", masked,
                        " raw=", value)

    def overwrite32(self, pos, value):
        """Overwrite a 32-bit word at a specific position in the buffer."""
        from rpython.rlib.debug import debug_print
        idx = pos // INST_SIZE
        assert 0 <= idx < len(self._buf)
        old = self._buf[idx]
        masked = value & 0xFFFFFFFF
        self._buf[idx] = masked
        if 16 <= idx <= 27:
            debug_print("HEXDBG overwrite32 idx=", idx, " old=", old,
                        " new=", masked, " raw=", value)

    def get_relative_pos(self, break_basic_block=True):
        """Return the current position in the code buffer (byte offset)."""
        return self._pos

    def mark_op(self, op):
        """Record the position of an IR operation in the code."""
        self.ops_offset[op] = self.get_relative_pos()

    def copy_to_raw_memory(self, addr):
        """Copy the instruction buffer to raw memory at addr.

        Used after code generation to place the machine code in
        executable memory.
        """
        from rpython.rlib.rmmap import enter_assembler_writing, leave_assembler_writing
        from rpython.rtyper.lltypesystem import rffi, lltype

        enter_assembler_writing()
        try:
            p = rffi.cast(rffi.UINTP, addr)
            for i in range(len(self._buf)):
                p[i] = rffi.cast(rffi.UINT, self._buf[i])
        finally:
            leave_assembler_writing()

    def absolute_addr(self):
        """Return the absolute address of the start of materialized code."""
        return self.rawstart

    def get_code_size(self):
        """Return the total size of generated code in bytes."""
        return self._pos

    def materialize(self, cpu, allblocks):
        """Allocate executable memory and copy code into it.

        Returns the raw start address of the executable code.
        """
        size = self.get_relative_pos()
        malloced = cpu.asmmemmgr.malloc(size, size)
        allblocks.append(malloced)
        rawstart = malloced[0]
        self.copy_to_raw_memory(rawstart)
        return rawstart

    def clear(self):
        """Reset the builder for reuse."""
        self._buf = []
        self._pos = 0
        self.ops_offset = {}

    # -----------------------------------------------------------------------
    # Hardware loop support
    # -----------------------------------------------------------------------

    def set_endloop0(self, pos=-1):
        """Set parse bits to endloop0 (0b10) on the instruction at pos.

        If pos=-1, patches the most recently emitted instruction.
        endloop0 marks the last instruction of a hardware loop0 body.
        """
        if pos < 0:
            idx = len(self._buf) - 1
        else:
            idx = pos // INST_SIZE
        if 0 <= idx < len(self._buf):
            word = self._buf[idx]
            word &= ~PARSE_BITS_MASK
            word |= (PARSE_ENDLOOP0 << PARSE_BITS_SHIFT)
            self._buf[idx] = word

    def set_endloop1(self, pos=-1):
        """Set parse bits to endloop1 (0b01) on the instruction at pos.

        If pos=-1, patches the most recently emitted instruction.
        endloop1 marks the last instruction of a hardware loop1 body.
        """
        if pos < 0:
            idx = len(self._buf) - 1
        else:
            idx = pos // INST_SIZE
        if 0 <= idx < len(self._buf):
            word = self._buf[idx]
            word &= ~PARSE_BITS_MASK
            word |= (PARSE_ENDLOOP1 << PARSE_BITS_SHIFT)
            self._buf[idx] = word

    # -----------------------------------------------------------------------
    # Constant pool support
    # -----------------------------------------------------------------------

    def emit_pending_constants(self):
        """Emit any pending constant pool entries at the current position.

        For Hexagon, large constants are loaded via multi-instruction
        sequences rather than PC-relative loads, so this is simpler
        than architectures with explicit constant pools.
        """
        # Align to 4-byte boundary (should already be aligned)
        while self._pos % INST_SIZE != 0:
            self._pos += 1

    # -----------------------------------------------------------------------
    # Raw data emission (for constant pools, GC tables, etc.)
    # -----------------------------------------------------------------------

    def write_data32(self, value):
        """Write a 32-bit data word (not an instruction)."""
        self._buf.append(value & 0xFFFFFFFF)
        self._pos += 4

    def write_data64(self, value):
        """Write a 64-bit data word."""
        self.write_data32(value & 0xFFFFFFFF)
        self.write_data32((value >> 32) & 0xFFFFFFFF)

    # -----------------------------------------------------------------------
    # Location-based helpers (for regalloc)
    # -----------------------------------------------------------------------

    def regalloc_mov(self, src, dst):
        """Move a value between any two locations.

        Used by the register allocator for spills, reloads, and reg-reg moves.
        """
        from rpython.jit.backend.hexagon.locations import (
            RegisterLocation, RegisterPairLocation, ImmLocation, StackLocation)

        if src.is_core_reg() and dst.is_core_reg():
            self.MV(dst.value, src.value)
        elif src.is_core_reg() and dst.is_stack():
            self.store_to_jitframe(src.value, dst.value)
        elif src.is_stack() and dst.is_core_reg():
            self.load_from_jitframe(dst.value, src.value)
        elif src.is_imm() and dst.is_core_reg():
            self.gen_load_int(dst.value, src.value)
        elif src.is_reg_pair() and dst.is_reg_pair():
            # Move pair: move both words
            if src.value != dst.value:
                self.MV(dst.even_reg(), src.even_reg())
                self.MV(dst.odd_reg(), src.odd_reg())
        elif src.is_reg_pair() and dst.is_stack():
            # Store pair to jitframe
            self.store_pair_to_jitframe(src.value, dst.value)
        elif src.is_stack() and dst.is_reg_pair():
            # Load pair from jitframe
            self.load_pair_from_jitframe(dst.value, src.value)
        elif src.is_imm() and dst.is_stack():
            # Load immediate to scratch, then store
            self.gen_load_int(r.scratch1.value, src.value)
            self.store_to_jitframe(r.scratch1.value, dst.value)
        elif src.is_stack() and dst.is_stack():
            # Stack-to-stack: go through scratch register
            if src.is_float():
                self.load_pair_from_jitframe(r.d14.value, src.value)
                self.store_pair_to_jitframe(r.d14.value, dst.value)
            else:
                self.load_from_jitframe(r.scratch1.value, src.value)
                self.store_to_jitframe(r.scratch1.value, dst.value)
        else:
            raise AssertionError("unsupported regalloc_mov: %s -> %s" %
                                 (src, dst))

    def push_locations(self, locs):
        """Save locations to the stack (for cycle resolution in remap)."""
        for loc in locs:
            if loc.is_core_reg():
                self.push_reg(loc.value)
            elif loc.is_reg_pair():
                self.push_reg(loc.odd_reg())
                self.push_reg(loc.even_reg())
            elif loc.is_stack():
                # Load from jitframe, push to stack
                if loc.is_float():
                    self.load_pair_from_jitframe(r.d14.value, loc.value)
                    self.push_reg(r.r15.value)  # high word
                    self.push_reg(r.r14.value)  # low word
                else:
                    self.load_from_jitframe(r.scratch1.value, loc.value)
                    self.push_reg(r.scratch1.value)

    def pop_locations(self, locs):
        """Restore locations from the stack (reverse order of push)."""
        for i in range(len(locs) - 1, -1, -1):
            loc = locs[i]
            if loc.is_core_reg():
                self.pop_reg(loc.value)
            elif loc.is_reg_pair():
                self.pop_reg(loc.even_reg())
                self.pop_reg(loc.odd_reg())
            elif loc.is_stack():
                if loc.is_float():
                    self.pop_reg(r.r14.value)
                    self.pop_reg(r.r15.value)
                    self.store_pair_to_jitframe(r.d14.value, loc.value)
                else:
                    self.pop_reg(r.scratch1.value)
                    self.store_to_jitframe(r.scratch1.value, loc.value)

    def mov_loc_to_raw_stack(self, loc, sp_offset):
        """Move a location's value to a raw stack position (for C calls)."""
        if loc.is_core_reg():
            self.STW(r.sp.value, loc.value, sp_offset)
        elif loc.is_reg_pair():
            self.STD(r.sp.value, loc.value, sp_offset)
        elif loc.is_stack():
            if loc.is_float():
                self.load_pair_from_jitframe(r.d14.value, loc.value)
                self.STD(r.sp.value, r.d14.value, sp_offset)
            else:
                self.load_from_jitframe(r.scratch1.value, loc.value)
                self.STW(r.sp.value, r.scratch1.value, sp_offset)
        elif loc.is_imm():
            self.gen_load_int(r.scratch1.value, loc.value)
            self.STW(r.sp.value, r.scratch1.value, sp_offset)
