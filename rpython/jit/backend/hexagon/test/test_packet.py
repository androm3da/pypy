"""
Tests for the VLIW packet optimizer.

These tests verify dependency analysis, structural constraint checking,
greedy scheduling, parse bit patching, and assembler integration helpers.
"""

from rpython.jit.backend.hexagon.packet import (
    InstrMeta, PacketOptimizer,
    FLAG_BRANCH, FLAG_MEM_LOAD, FLAG_MEM_STORE, FLAG_BARRIER,
    PARSE_ENDLOOP0, PARSE_ENDLOOP1,
)
from rpython.jit.backend.hexagon.arch import (
    INST_SIZE, PARSE_BITS_MASK, PARSE_BITS_SHIFT,
    PARSE_NOT_END, PARSE_END_PACKET,
)
from rpython.jit.backend.hexagon.assembler import (
    _is_control_flow_op, _extract_op_reg_info,
    _BINARY_ALU_OPS, _MOVE_OPS,
    _GC_LOAD_OPS, _GC_STORE_OPS, _GETFIELD_OPS, _SETFIELD_OPS,
)
from rpython.jit.metainterp.resoperation import rop


class FakeMC(object):
    """Fake InstrBuilder for testing parse bit patching."""

    def __init__(self, n_instrs):
        # Start with all instructions as single-instruction packets (0b11)
        self._buf = [(PARSE_END_PACKET << PARSE_BITS_SHIFT)] * n_instrs

    def get_parse_bits(self, idx):
        return (self._buf[idx] >> PARSE_BITS_SHIFT) & 0x3


def make_instr(pos, reads=None, writes=None, flags=0):
    """Helper to create InstrMeta for tests."""
    return InstrMeta(
        pos,
        frozenset(reads or []),
        frozenset(writes or []),
        flags,
    )


class TestInstrMeta(object):

    def test_is_branch(self):
        m = make_instr(0, flags=FLAG_BRANCH)
        assert m.is_branch()
        assert not m.is_mem_load()

    def test_is_mem_load(self):
        m = make_instr(0, flags=FLAG_MEM_LOAD)
        assert m.is_mem_load()
        assert not m.is_branch()

    def test_is_mem_store(self):
        m = make_instr(0, flags=FLAG_MEM_STORE)
        assert m.is_mem_store()

    def test_is_barrier(self):
        m = make_instr(0, flags=FLAG_BARRIER)
        assert m.is_barrier()

    def test_no_flags(self):
        m = make_instr(0)
        assert not m.is_branch()
        assert not m.is_mem_load()
        assert not m.is_mem_store()
        assert not m.is_barrier()


class TestDependencyAnalysis(object):

    def setup_method(self, meth=None):
        self.opt = PacketOptimizer()

    def test_no_dependency_independent(self):
        a = make_instr(0, reads=[1], writes=[2])
        b = make_instr(4, reads=[3], writes=[4])
        assert not self.opt._has_dependency(a, b)

    def test_raw_dependency(self):
        """RAW: b reads what a writes."""
        a = make_instr(0, reads=[1], writes=[2])
        b = make_instr(4, reads=[2], writes=[3])
        assert self.opt._has_dependency(a, b)

    def test_waw_dependency(self):
        """WAW: both write the same register."""
        a = make_instr(0, reads=[1], writes=[2])
        b = make_instr(4, reads=[3], writes=[2])
        assert self.opt._has_dependency(a, b)

    def test_war_no_dependency(self):
        """WAR: write-after-read is OK in VLIW (reads happen first)."""
        a = make_instr(0, reads=[2], writes=[1])
        b = make_instr(4, reads=[3], writes=[2])
        assert not self.opt._has_dependency(a, b)

    def test_no_dependency_disjoint_regs(self):
        a = make_instr(0, reads=[0, 1], writes=[5])
        b = make_instr(4, reads=[2, 3], writes=[6])
        assert not self.opt._has_dependency(a, b)

    def test_raw_multiple_regs(self):
        a = make_instr(0, writes=[5, 6])
        b = make_instr(4, reads=[6, 7])
        assert self.opt._has_dependency(a, b)

    def test_waw_multiple_regs(self):
        a = make_instr(0, writes=[5, 6])
        b = make_instr(4, writes=[6, 7])
        assert self.opt._has_dependency(a, b)


class TestStructuralConstraints(object):

    def setup_method(self, meth=None):
        self.opt = PacketOptimizer()

    def test_empty_packet_allows_anything(self):
        instr = make_instr(0)
        assert self.opt._check_structural([], instr)

    def test_max_packet_size(self):
        packet = [make_instr(i * 4) for i in range(4)]
        instr = make_instr(16)
        assert not self.opt._check_structural(packet, instr)

    def test_one_branch_allowed(self):
        packet = [make_instr(0)]
        branch = make_instr(4, flags=FLAG_BRANCH)
        assert self.opt._check_structural(packet, branch)

    def test_two_branches_not_allowed(self):
        packet = [make_instr(0, flags=FLAG_BRANCH)]
        branch2 = make_instr(4, flags=FLAG_BRANCH)
        assert not self.opt._check_structural(packet, branch2)

    def test_two_loads_allowed(self):
        packet = [make_instr(0, flags=FLAG_MEM_LOAD)]
        load2 = make_instr(4, flags=FLAG_MEM_LOAD)
        # Structural check only checks count (<=2), but _check_structural also
        # checks aliasing: load+load is fine since loads don't alias.
        # Wait, looking at the code: a store prevents bundling with any other
        # mem op, but loads can bundle with other loads.
        assert self.opt._check_structural(packet, load2)

    def test_store_blocks_other_mem(self):
        """A store can't bundle with other memory ops (aliasing concern)."""
        packet = [make_instr(0, flags=FLAG_MEM_LOAD)]
        store = make_instr(4, flags=FLAG_MEM_STORE)
        assert not self.opt._check_structural(packet, store)

    def test_store_blocks_other_store(self):
        packet = [make_instr(0, flags=FLAG_MEM_STORE)]
        store2 = make_instr(4, flags=FLAG_MEM_STORE)
        assert not self.opt._check_structural(packet, store2)

    def test_load_blocked_by_store(self):
        packet = [make_instr(0, flags=FLAG_MEM_STORE)]
        load = make_instr(4, flags=FLAG_MEM_LOAD)
        assert not self.opt._check_structural(packet, load)

    def test_three_mem_ops_not_allowed(self):
        """At most 2 memory ops per packet."""
        packet = [
            make_instr(0, flags=FLAG_MEM_LOAD),
            make_instr(4, flags=FLAG_MEM_LOAD),
        ]
        load3 = make_instr(8, flags=FLAG_MEM_LOAD)
        assert not self.opt._check_structural(packet, load3)


class TestCanAddToPacket(object):

    def setup_method(self, meth=None):
        self.opt = PacketOptimizer()

    def test_barrier_not_bundleable(self):
        barrier = make_instr(0, flags=FLAG_BARRIER)
        assert not self.opt._can_add_to_packet([], barrier)

    def test_independent_can_add(self):
        packet = [make_instr(0, reads=[1], writes=[2])]
        instr = make_instr(4, reads=[3], writes=[4])
        assert self.opt._can_add_to_packet(packet, instr)

    def test_dependent_cannot_add(self):
        packet = [make_instr(0, reads=[1], writes=[2])]
        instr = make_instr(4, reads=[2], writes=[3])
        assert not self.opt._can_add_to_packet(packet, instr)


class TestScheduling(object):

    def setup_method(self, meth=None):
        self.opt = PacketOptimizer()

    def test_single_instruction(self):
        instrs = [make_instr(0, reads=[1], writes=[2])]
        packets = self.opt._schedule(instrs)
        assert len(packets) == 1
        assert len(packets[0]) == 1

    def test_two_independent_bundled(self):
        instrs = [
            make_instr(0, reads=[1], writes=[2]),
            make_instr(4, reads=[3], writes=[4]),
        ]
        packets = self.opt._schedule(instrs)
        assert len(packets) == 1
        assert len(packets[0]) == 2

    def test_two_dependent_separate(self):
        instrs = [
            make_instr(0, reads=[1], writes=[2]),
            make_instr(4, reads=[2], writes=[3]),
        ]
        packets = self.opt._schedule(instrs)
        assert len(packets) == 2
        assert len(packets[0]) == 1
        assert len(packets[1]) == 1

    def test_barrier_forces_split(self):
        instrs = [
            make_instr(0, reads=[1], writes=[2]),
            make_instr(-1, flags=FLAG_BARRIER),
            make_instr(8, reads=[3], writes=[4]),
        ]
        packets = self.opt._schedule(instrs)
        assert len(packets) == 2

    def test_four_independent_one_packet(self):
        instrs = [
            make_instr(0, reads=[0], writes=[4]),
            make_instr(4, reads=[1], writes=[5]),
            make_instr(8, reads=[2], writes=[6]),
            make_instr(12, reads=[3], writes=[7]),
        ]
        packets = self.opt._schedule(instrs)
        assert len(packets) == 1
        assert len(packets[0]) == 4

    def test_five_instructions_overflow_packet(self):
        instrs = [
            make_instr(0, reads=[0], writes=[5]),
            make_instr(4, reads=[1], writes=[6]),
            make_instr(8, reads=[2], writes=[7]),
            make_instr(12, reads=[3], writes=[8]),
            make_instr(16, reads=[4], writes=[9]),
        ]
        packets = self.opt._schedule(instrs)
        assert len(packets) == 2
        assert len(packets[0]) == 4
        assert len(packets[1]) == 1

    def test_chain_of_dependencies(self):
        """a writes R2, b reads R2 and writes R3, c reads R3."""
        instrs = [
            make_instr(0, reads=[0], writes=[2]),
            make_instr(4, reads=[2], writes=[3]),
            make_instr(8, reads=[3], writes=[4]),
        ]
        packets = self.opt._schedule(instrs)
        assert len(packets) == 3

    def test_stats_tracking(self):
        opt = PacketOptimizer()
        instrs = [
            make_instr(0, reads=[0], writes=[2]),
            make_instr(4, reads=[1], writes=[3]),
        ]
        opt._schedule(instrs)
        stats = opt.get_stats()
        assert stats['total_packets'] == 1
        assert stats['total_instrs'] == 2
        assert stats['bundled_instrs'] == 2

    def test_stats_single_instruction_not_bundled(self):
        opt = PacketOptimizer()
        instrs = [make_instr(0, reads=[0], writes=[1])]
        opt._schedule(instrs)
        stats = opt.get_stats()
        assert stats['total_packets'] == 1
        assert stats['total_instrs'] == 1
        assert stats['bundled_instrs'] == 0

    def test_original_order_preserved(self):
        """Instructions within a packet keep their original order."""
        instrs = [
            make_instr(0, reads=[0], writes=[4]),
            make_instr(4, reads=[1], writes=[5]),
            make_instr(8, reads=[2], writes=[6]),
        ]
        packets = self.opt._schedule(instrs)
        assert len(packets) == 1
        assert packets[0][0].pos == 0
        assert packets[0][1].pos == 4
        assert packets[0][2].pos == 8


class TestParseBitPatching(object):

    def setup_method(self, meth=None):
        self.opt = PacketOptimizer()

    def test_single_instruction_end_packet(self):
        mc = FakeMC(1)
        packets = [[make_instr(0)]]
        self.opt._apply_parse_bits(mc, packets)
        assert mc.get_parse_bits(0) == PARSE_END_PACKET

    def test_two_instruction_packet(self):
        mc = FakeMC(2)
        packets = [[make_instr(0), make_instr(4)]]
        self.opt._apply_parse_bits(mc, packets)
        assert mc.get_parse_bits(0) == PARSE_NOT_END
        assert mc.get_parse_bits(1) == PARSE_END_PACKET

    def test_three_instruction_packet(self):
        mc = FakeMC(3)
        packets = [[make_instr(0), make_instr(4), make_instr(8)]]
        self.opt._apply_parse_bits(mc, packets)
        assert mc.get_parse_bits(0) == PARSE_NOT_END
        assert mc.get_parse_bits(1) == PARSE_NOT_END
        assert mc.get_parse_bits(2) == PARSE_END_PACKET

    def test_two_separate_packets(self):
        mc = FakeMC(3)
        packets = [
            [make_instr(0), make_instr(4)],
            [make_instr(8)],
        ]
        self.opt._apply_parse_bits(mc, packets)
        assert mc.get_parse_bits(0) == PARSE_NOT_END
        assert mc.get_parse_bits(1) == PARSE_END_PACKET
        assert mc.get_parse_bits(2) == PARSE_END_PACKET

    def test_sentinel_entries_skipped(self):
        mc = FakeMC(2)
        packets = [
            [make_instr(0), make_instr(-1, flags=FLAG_BARRIER), make_instr(4)],
        ]
        self.opt._apply_parse_bits(mc, packets)
        # Sentinel (pos=-1) should be skipped, word at pos=0 and pos=4 patched
        assert mc.get_parse_bits(0) == PARSE_NOT_END
        assert mc.get_parse_bits(1) == PARSE_END_PACKET


class TestFlush(object):

    def setup_method(self, meth=None):
        self.opt = PacketOptimizer()

    def test_flush_empty(self):
        mc = FakeMC(0)
        self.opt.flush(mc)  # should not crash

    def test_annotate_and_flush(self):
        mc = FakeMC(2)
        self.opt.annotate(0, reads=[0], writes=[2])
        self.opt.annotate(4, reads=[1], writes=[3])
        self.opt.flush(mc)
        # Two independent instructions should be bundled
        assert mc.get_parse_bits(0) == PARSE_NOT_END
        assert mc.get_parse_bits(1) == PARSE_END_PACKET
        # Pending list should be cleared
        assert len(self.opt.pending) == 0

    def test_annotate_dependent_flush(self):
        mc = FakeMC(2)
        self.opt.annotate(0, reads=[0], writes=[2])
        self.opt.annotate(4, reads=[2], writes=[3])
        self.opt.flush(mc)
        # Dependent instructions should be in separate packets
        assert mc.get_parse_bits(0) == PARSE_END_PACKET
        assert mc.get_parse_bits(1) == PARSE_END_PACKET

    def test_barrier_annotate_flush(self):
        mc = FakeMC(3)
        self.opt.annotate(0, reads=[0], writes=[4])
        self.opt.annotate(4, reads=[1], writes=[5])
        self.opt.barrier()
        self.opt.annotate(8, reads=[2], writes=[6])
        self.opt.flush(mc)
        # First two should be bundled, third separate after barrier
        assert mc.get_parse_bits(0) == PARSE_NOT_END
        assert mc.get_parse_bits(1) == PARSE_END_PACKET
        assert mc.get_parse_bits(2) == PARSE_END_PACKET


# ---------------------------------------------------------------------------
# Tests for assembler integration helpers
# ---------------------------------------------------------------------------

class FakeRegLoc(object):
    """Minimal mock for a register location."""
    def __init__(self, value):
        self.value = value

    def is_imm(self):
        return False

    def is_core_reg(self):
        return True


class FakeImmLoc(object):
    """Minimal mock for an immediate location."""
    def __init__(self, value):
        self.value = value

    def is_imm(self):
        return True

    def is_core_reg(self):
        return False


class TestControlFlowClassification(object):

    def test_jump_is_control_flow(self):
        assert _is_control_flow_op(rop.JUMP)

    def test_finish_is_control_flow(self):
        assert _is_control_flow_op(rop.FINISH)

    def test_label_is_control_flow(self):
        assert _is_control_flow_op(rop.LABEL)

    def test_guard_true_is_control_flow(self):
        assert _is_control_flow_op(rop.GUARD_TRUE)

    def test_guard_false_is_control_flow(self):
        assert _is_control_flow_op(rop.GUARD_FALSE)

    def test_guard_no_exception_is_control_flow(self):
        assert _is_control_flow_op(rop.GUARD_NO_EXCEPTION)

    def test_guard_value_is_control_flow(self):
        assert _is_control_flow_op(rop.GUARD_VALUE)

    def test_int_add_not_control_flow(self):
        assert not _is_control_flow_op(rop.INT_ADD)

    def test_int_sub_not_control_flow(self):
        assert not _is_control_flow_op(rop.INT_SUB)

    def test_same_as_i_not_control_flow(self):
        assert not _is_control_flow_op(rop.SAME_AS_I)

    def test_getfield_not_control_flow(self):
        assert not _is_control_flow_op(rop.GETFIELD_GC_I)

    def test_setfield_not_control_flow(self):
        assert not _is_control_flow_op(rop.SETFIELD_GC)

    def test_gc_load_not_control_flow(self):
        assert not _is_control_flow_op(rop.GC_LOAD_I)

    def test_gc_store_not_control_flow(self):
        assert not _is_control_flow_op(rop.GC_STORE)


class TestExtractOpRegInfo(object):

    def test_int_add_registers(self):
        arglocs = [FakeRegLoc(1), FakeRegLoc(2), FakeRegLoc(3)]
        info = _extract_op_reg_info(rop.INT_ADD, arglocs)
        assert info is not None
        reads, writes, flags = info
        assert 1 in reads
        assert 2 in reads
        assert writes == [3]
        assert flags == 0

    def test_int_add_immediate(self):
        arglocs = [FakeRegLoc(5), FakeImmLoc(42), FakeRegLoc(6)]
        info = _extract_op_reg_info(rop.INT_ADD, arglocs)
        assert info is not None
        reads, writes, flags = info
        assert reads == [5]
        assert writes == [6]
        assert flags == 0

    def test_int_sub_registers(self):
        arglocs = [FakeRegLoc(3), FakeRegLoc(4), FakeRegLoc(5)]
        info = _extract_op_reg_info(rop.INT_SUB, arglocs)
        assert info is not None
        reads, writes, flags = info
        assert 3 in reads
        assert 4 in reads
        assert writes == [5]

    def test_int_mul(self):
        arglocs = [FakeRegLoc(0), FakeRegLoc(1), FakeRegLoc(2)]
        info = _extract_op_reg_info(rop.INT_MUL, arglocs)
        assert info is not None
        reads, writes, flags = info
        assert 0 in reads
        assert 1 in reads
        assert writes == [2]

    def test_int_and(self):
        arglocs = [FakeRegLoc(7), FakeRegLoc(8), FakeRegLoc(9)]
        info = _extract_op_reg_info(rop.INT_AND, arglocs)
        assert info is not None
        reads, writes, flags = info
        assert 7 in reads and 8 in reads
        assert writes == [9]

    def test_int_or_immediate(self):
        arglocs = [FakeRegLoc(2), FakeImmLoc(0xFF), FakeRegLoc(3)]
        info = _extract_op_reg_info(rop.INT_OR, arglocs)
        assert info is not None
        reads, writes, flags = info
        assert reads == [2]
        assert writes == [3]

    def test_shift_ops(self):
        for opnum in [rop.INT_LSHIFT, rop.INT_RSHIFT, rop.UINT_RSHIFT]:
            arglocs = [FakeRegLoc(4), FakeImmLoc(3), FakeRegLoc(5)]
            info = _extract_op_reg_info(opnum, arglocs)
            assert info is not None
            reads, writes, flags = info
            assert reads == [4]
            assert writes == [5]
            assert flags == 0

    def test_same_as_i_register(self):
        arglocs = [FakeRegLoc(10), FakeRegLoc(11)]
        info = _extract_op_reg_info(rop.SAME_AS_I, arglocs)
        assert info is not None
        reads, writes, flags = info
        assert reads == [10]
        assert writes == [11]

    def test_same_as_i_immediate(self):
        arglocs = [FakeImmLoc(99), FakeRegLoc(7)]
        info = _extract_op_reg_info(rop.SAME_AS_I, arglocs)
        assert info is not None
        reads, writes, flags = info
        assert reads == []
        assert writes == [7]

    def test_gc_load_i(self):
        """gc_load_i arglocs: [base, ofs, res, nsize]."""
        arglocs = [FakeRegLoc(2), FakeImmLoc(16), FakeRegLoc(5), FakeImmLoc(4)]
        info = _extract_op_reg_info(rop.GC_LOAD_I, arglocs)
        assert info is not None
        reads, writes, flags = info
        assert reads == [2]
        assert writes == [5]
        assert flags == FLAG_MEM_LOAD

    def test_gc_load_r(self):
        """gc_load_r arglocs: [base, ofs, res, nsize]."""
        arglocs = [FakeRegLoc(3), FakeImmLoc(8), FakeRegLoc(6), FakeImmLoc(4)]
        info = _extract_op_reg_info(rop.GC_LOAD_R, arglocs)
        assert info is not None
        reads, writes, flags = info
        assert reads == [3]
        assert writes == [6]
        assert flags == FLAG_MEM_LOAD

    def test_gc_load_f(self):
        """gc_load_f arglocs: [base, ofs, res_pair, nsize]."""
        arglocs = [FakeRegLoc(4), FakeImmLoc(0), FakeRegLoc(16), FakeImmLoc(8)]
        info = _extract_op_reg_info(rop.GC_LOAD_F, arglocs)
        assert info is not None
        reads, writes, flags = info
        assert reads == [4]
        assert writes == [16]
        assert flags == FLAG_MEM_LOAD

    def test_gc_store(self):
        """gc_store arglocs: [value, base, ofs, size]."""
        arglocs = [FakeRegLoc(7), FakeRegLoc(1), FakeImmLoc(8), FakeImmLoc(4)]
        info = _extract_op_reg_info(rop.GC_STORE, arglocs)
        assert info is not None
        reads, writes, flags = info
        assert 7 in reads  # value
        assert 1 in reads  # base
        assert writes == []
        assert flags == FLAG_MEM_STORE

    def test_gc_load_two_independent_can_bundle(self):
        """Two independent gc_loads should be bundleable (no RAW/WAW)."""
        from rpython.jit.backend.hexagon.packet import PacketOptimizer
        opt = PacketOptimizer()
        # gc_load_i from base+4 -> R5
        info1 = _extract_op_reg_info(rop.GC_LOAD_I,
            [FakeRegLoc(2), FakeImmLoc(4), FakeRegLoc(5), FakeImmLoc(4)])
        # gc_load_i from base+8 -> R6
        info2 = _extract_op_reg_info(rop.GC_LOAD_I,
            [FakeRegLoc(2), FakeImmLoc(8), FakeRegLoc(6), FakeImmLoc(4)])
        assert info1 is not None and info2 is not None
        opt.annotate(0, info1[0], info1[1], info1[2])
        opt.annotate(4, info2[0], info2[1], info2[2])
        packets = opt._schedule(opt.pending)
        # Two loads from same base, different offsets -> 1 packet
        assert len(packets) == 1
        assert len(packets[0]) == 2

    def test_gc_store_blocks_gc_load_in_packet(self):
        """gc_store + gc_load blocked by structural memory ordering."""
        from rpython.jit.backend.hexagon.packet import PacketOptimizer
        opt = PacketOptimizer()
        info_store = _extract_op_reg_info(rop.GC_STORE,
            [FakeRegLoc(5), FakeRegLoc(1), FakeImmLoc(8), FakeImmLoc(4)])
        info_load = _extract_op_reg_info(rop.GC_LOAD_I,
            [FakeRegLoc(1), FakeImmLoc(4), FakeRegLoc(6), FakeImmLoc(4)])
        assert info_store is not None and info_load is not None
        opt.annotate(0, info_store[0], info_store[1], info_store[2])
        opt.annotate(4, info_load[0], info_load[1], info_load[2])
        packets = opt._schedule(opt.pending)
        # Store + load blocked by structural constraint
        assert len(packets) == 2

    def test_legacy_getfield_gc_i_load(self):
        """Legacy getfield_gc_i arglocs: [base, ofs, res]."""
        arglocs = [FakeRegLoc(2), FakeImmLoc(16), FakeRegLoc(5)]
        info = _extract_op_reg_info(rop.GETFIELD_GC_I, arglocs)
        assert info is not None
        reads, writes, flags = info
        assert reads == [2]
        assert writes == [5]
        assert flags == FLAG_MEM_LOAD

    def test_legacy_setfield_gc_store(self):
        """Legacy setfield_gc arglocs: [base, ofs, val]."""
        arglocs = [FakeRegLoc(1), FakeImmLoc(8), FakeRegLoc(3)]
        info = _extract_op_reg_info(rop.SETFIELD_GC, arglocs)
        assert info is not None
        reads, writes, flags = info
        assert 1 in reads
        assert 3 in reads
        assert writes == []
        assert flags == FLAG_MEM_STORE

    def test_unknown_op_returns_sentinel(self):
        arglocs = [FakeRegLoc(0)]
        reads, writes, flags = _extract_op_reg_info(rop.GUARD_TRUE, arglocs)
        assert flags == -1

    def test_call_returns_sentinel(self):
        arglocs = [FakeRegLoc(0), FakeImmLoc(4), FakeImmLoc(0)]
        reads, writes, flags = _extract_op_reg_info(rop.CALL_N, arglocs)
        assert flags == -1

    def test_int_xor_registers(self):
        arglocs = [FakeRegLoc(1), FakeRegLoc(2), FakeRegLoc(3)]
        info = _extract_op_reg_info(rop.INT_XOR, arglocs)
        assert info is not None
        reads, writes, flags = info
        assert 1 in reads
        assert 2 in reads
        assert writes == [3]

    def test_binary_alu_ops_set(self):
        """All expected ops are in the binary ALU set."""
        expected = [rop.INT_ADD, rop.INT_SUB, rop.INT_MUL, rop.INT_AND,
                    rop.INT_OR, rop.INT_XOR, rop.INT_LSHIFT, rop.INT_RSHIFT,
                    rop.UINT_RSHIFT]
        for opnum in expected:
            assert opnum in _BINARY_ALU_OPS

    def test_move_ops_set(self):
        expected = [rop.SAME_AS_I, rop.SAME_AS_R, rop.CAST_PTR_TO_INT,
                    rop.CAST_INT_TO_PTR]
        for opnum in expected:
            assert opnum in _MOVE_OPS

    def test_gc_load_ops_set(self):
        expected = [rop.GC_LOAD_I, rop.GC_LOAD_R, rop.GC_LOAD_F]
        for opnum in expected:
            assert opnum in _GC_LOAD_OPS

    def test_gc_store_ops_set(self):
        assert rop.GC_STORE in _GC_STORE_OPS
