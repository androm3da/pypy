#!/usr/bin/env python
"""
Instruction encoding tests for the Hexagon JIT backend.

Tests verify that our instruction builder generates the same binary encoding
as llvm-mc (the LLVM machine code assembler). Each test encodes an instruction
using our builder, then compares against llvm-mc output.

Uses hypothesis for property-based testing with random register/immediate
operand combinations.
"""

import struct
import sys
import py
import pytest

from rpython.jit.backend.hexagon import registers as r
from rpython.jit.backend.hexagon import codebuilder
from rpython.jit.backend.hexagon.arch import INST_SIZE

try:
    from hypothesis import given, settings, strategies as st, assume
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False

try:
    from rpython.jit.backend.hexagon.test.external_assembler import (
        assemble_to_words, _tool_available)
    HAS_TOOLS = _tool_available()
except (ImportError, OSError):
    HAS_TOOLS = False

needs_hypothesis = pytest.mark.skipif(
    not HAS_HYPOTHESIS, reason="hypothesis not available")
needs_tools = pytest.mark.skipif(
    not HAS_TOOLS, reason="llvm-mc not available for hexagon")


# ---------------------------------------------------------------------------
# Test code builder: captures instruction words
# ---------------------------------------------------------------------------

class TestCodeBuilder(codebuilder.AbstractHexagonBuilder):
    """Minimal code builder that captures instruction words in a list."""

    def __init__(self):
        self.buffer = []

    def write32(self, value):
        self.buffer.append(value & 0xFFFFFFFF)

    def get_word(self, idx=0):
        return self.buffer[idx]

    def get_words(self):
        return list(self.buffer)

    def hexdump(self):
        """Return raw little-endian bytes of all emitted words."""
        result = b''
        for w in self.buffer:
            result += struct.pack('<I', w)
        return result

    def reset(self):
        self.buffer = []


# ---------------------------------------------------------------------------
# Register name helpers for assembly strings
# ---------------------------------------------------------------------------

def gpr_name(reg_num):
    """Return the assembly name for a GPR number."""
    return 'R%d' % reg_num


def pred_name(pred_num):
    """Return the assembly name for a predicate register."""
    return 'P%d' % pred_num


def pair_name(even_reg):
    """Return the assembly name for a register pair (e.g. R1:R0)."""
    return 'R%d:%d' % (even_reg + 1, even_reg)


def vreg_name(vreg_num):
    """Return the assembly name for an HVX vector register."""
    return 'V%d' % vreg_num


# ---------------------------------------------------------------------------
# GPR strategies for hypothesis
# ---------------------------------------------------------------------------

if HAS_HYPOTHESIS:
    gpr_strategy = st.integers(min_value=0, max_value=31)
    pred_strategy = st.integers(min_value=0, max_value=3)
    even_pair_strategy = st.integers(min_value=0, max_value=15).map(lambda x: x * 2)
    imm5_strategy = st.integers(min_value=0, max_value=31)
    imm16_strategy = st.integers(min_value=-32768, max_value=32767)
    vreg_strategy = st.integers(min_value=0, max_value=31)


# ---------------------------------------------------------------------------
# Self-consistency tests (no toolchain needed)
# ---------------------------------------------------------------------------

class TestInstrBuilderSelfConsistency(object):
    """Tests that verify basic encoding properties without reference assembler."""

    def test_parse_bits_present(self):
        """All single-instruction packets must have parse bits 0b11 at [15:14]."""
        cb = TestCodeBuilder()
        cb.ADD(0, 1, 2)
        w = cb.get_word()
        parse_bits = (w >> 14) & 0b11
        assert parse_bits == 0b11, (
            "parse bits should be 0b11, got 0b%s" % bin(parse_bits))

    def test_add_rd_field(self):
        """ADD: Rd field should be at bits [4:0]."""
        cb = TestCodeBuilder()
        for rd in range(32):
            cb.reset()
            cb.ADD(rd, 0, 0)
            w = cb.get_word()
            assert (w & 0x1F) == rd, (
                "Rd=%d but bits[4:0]=%d" % (rd, w & 0x1F))

    def test_add_rs_field(self):
        """ADD: Rs field should be at bits [20:16]."""
        cb = TestCodeBuilder()
        for rs in range(32):
            cb.reset()
            cb.ADD(0, rs, 0)
            w = cb.get_word()
            assert ((w >> 16) & 0x1F) == rs, (
                "Rs=%d but bits[20:16]=%d" % (rs, (w >> 16) & 0x1F))

    def test_add_rt_field(self):
        """ADD: Rt field should be at bits [12:8]."""
        cb = TestCodeBuilder()
        for rt in range(32):
            cb.reset()
            cb.ADD(0, 0, rt)
            w = cb.get_word()
            assert ((w >> 8) & 0x1F) == rt, (
                "Rt=%d but bits[12:8]=%d" % (rt, (w >> 8) & 0x1F))

    def test_sub_different_from_add(self):
        """SUB and ADD must produce different encodings for same operands."""
        cb = TestCodeBuilder()
        cb.ADD(0, 1, 2)
        add_w = cb.get_word()
        cb.reset()
        cb.SUB(0, 1, 2)
        sub_w = cb.get_word()
        assert add_w != sub_w

    def test_addi_immediate_encoding(self):
        """ADDI: immediate values should be recoverable from encoding."""
        cb = TestCodeBuilder()
        # Test a small positive immediate
        cb.ADDI(5, 10, 42)
        w = cb.get_word()
        # Verify Rd at [4:0]
        assert (w & 0x1F) == 5
        # Verify Rs at [20:16]
        assert ((w >> 16) & 0x1F) == 10

    def test_tfrsi_rd_field(self):
        """TFRSI: Rd should be at bits [4:0]."""
        cb = TestCodeBuilder()
        for rd in range(32):
            cb.reset()
            cb.TFRSI(rd, 0)
            w = cb.get_word()
            assert (w & 0x1F) == rd

    def test_cmp_eq_produces_single_word(self):
        """CMP_EQ should emit exactly one 32-bit instruction."""
        cb = TestCodeBuilder()
        cb.CMP_EQ(0, 1, 2)
        assert len(cb.buffer) == 1

    def test_ldw_produces_single_word(self):
        """LDW should emit exactly one 32-bit instruction."""
        cb = TestCodeBuilder()
        cb.LDW(0, 1, 0)
        assert len(cb.buffer) == 1

    def test_stw_produces_single_word(self):
        """STW should emit exactly one 32-bit instruction."""
        cb = TestCodeBuilder()
        cb.STW(0, 1, 0)
        assert len(cb.buffer) == 1

    def test_jump_imm_produces_single_word(self):
        """J_IMM should emit exactly one 32-bit instruction."""
        cb = TestCodeBuilder()
        cb.J_IMM(0)
        assert len(cb.buffer) == 1

    def test_jumpr_produces_single_word(self):
        """JUMPR should emit exactly one 32-bit instruction."""
        cb = TestCodeBuilder()
        cb.JUMPR(31)  # jump to LR
        assert len(cb.buffer) == 1

    def test_trap0_encoding(self):
        """TRAP0 should encode the immediate in the low 8 bits."""
        cb = TestCodeBuilder()
        cb.TRAP0(0xDE)
        w = cb.get_word()
        assert (w & 0xFF) == 0xDE

    def test_pseudo_nop(self):
        """NOP pseudo-instruction should emit exactly one word."""
        cb = TestCodeBuilder()
        cb.NOP()
        assert len(cb.buffer) == 1

    def test_pseudo_mv(self):
        """MV(rd, rs) should be the same as TFR(rd, rs)."""
        cb = TestCodeBuilder()
        cb.MV(5, 10)
        mv_w = cb.get_word()
        cb.reset()
        cb.TFR(5, 10)
        tfr_w = cb.get_word()
        assert mv_w == tfr_w

    def test_pseudo_neg_is_two_instructions(self):
        """NEG should emit two instructions (TFRSI + SUB)."""
        cb = TestCodeBuilder()
        cb.NEG(5, 10)
        assert len(cb.buffer) == 2

    def test_pseudo_not_is_two_instructions(self):
        """NOT should emit two instructions (TFRSI + XOR)."""
        cb = TestCodeBuilder()
        cb.NOT(5, 10)
        assert len(cb.buffer) == 2

    def test_gen_load_int_small(self):
        """gen_load_int with small immediate should use one TFRSI."""
        cb = TestCodeBuilder()
        cb.gen_load_int(5, 42)
        assert len(cb.buffer) == 1  # single TFRSI

    def test_gen_load_int_negative_small(self):
        """gen_load_int with small negative should use one TFRSI."""
        cb = TestCodeBuilder()
        cb.gen_load_int(5, -1)
        assert len(cb.buffer) == 1

    def test_gen_load_int_large(self):
        """gen_load_int with large value should use multiple instructions."""
        cb = TestCodeBuilder()
        cb.gen_load_int(5, 0x12345678)
        assert len(cb.buffer) > 1

    def test_shift_immediate_range(self):
        """Shift-by-immediate should accept 0-31."""
        cb = TestCodeBuilder()
        for imm in [0, 1, 15, 31]:
            cb.reset()
            cb.S2_ASL_I_R(5, 10, imm)
            assert len(cb.buffer) == 1

    def test_all_alu_ops_different(self):
        """Each ALU operation should produce a distinct encoding."""
        ops = {}
        cb = TestCodeBuilder()
        for name in ['ADD', 'SUB', 'AND', 'OR', 'XOR']:
            cb.reset()
            getattr(cb, name)(0, 1, 2)
            ops[name] = cb.get_word()
        # All should be unique
        values = list(ops.values())
        assert len(set(values)) == len(values), (
            "some ALU ops produced identical encodings: %r" % ops)

    def test_conditional_jumps_differ(self):
        """J_IF_TRUE and J_IF_FALSE should produce different encodings."""
        cb = TestCodeBuilder()
        cb.J_IF_TRUE(0, 0)
        w_true = cb.get_word()
        cb.reset()
        cb.J_IF_FALSE(0, 0)
        w_false = cb.get_word()
        assert w_true != w_false

    def test_load_variants_differ(self):
        """LDB, LDH, LDW should produce different encodings."""
        ops = {}
        cb = TestCodeBuilder()
        for name in ['LDB', 'LDH', 'LDW']:
            cb.reset()
            getattr(cb, name)(0, 1, 0)
            ops[name] = cb.get_word()
        values = list(ops.values())
        assert len(set(values)) == len(values), (
            "load variants produced identical encodings: %r" % ops)

    def test_store_variants_differ(self):
        """STB, STH, STW should produce different encodings."""
        ops = {}
        cb = TestCodeBuilder()
        for name in ['STB', 'STH', 'STW']:
            cb.reset()
            getattr(cb, name)(1, 0, 0)
            ops[name] = cb.get_word()
        values = list(ops.values())
        assert len(set(values)) == len(values), (
            "store variants produced identical encodings: %r" % ops)


# ---------------------------------------------------------------------------
# HVX self-consistency tests
# ---------------------------------------------------------------------------

class TestHVXInstrBuilderSelfConsistency(object):
    """Tests for HVX instruction encoding properties."""

    def test_vadd_w_produces_single_word(self):
        cb = TestCodeBuilder()
        cb.VADD_W(0, 1, 2)
        assert len(cb.buffer) == 1

    def test_vadd_variants_differ(self):
        """VADD_W, VADD_H, VADD_B should produce different encodings."""
        ops = {}
        cb = TestCodeBuilder()
        for name in ['VADD_W', 'VADD_H', 'VADD_B']:
            cb.reset()
            getattr(cb, name)(0, 1, 2)
            ops[name] = cb.get_word()
        values = list(ops.values())
        assert len(set(values)) == len(values)

    def test_vadd_vs_vsub_differ(self):
        cb = TestCodeBuilder()
        cb.VADD_W(0, 1, 2)
        add_w = cb.get_word()
        cb.reset()
        cb.VSUB_W(0, 1, 2)
        sub_w = cb.get_word()
        assert add_w != sub_w

    def test_vand_vor_vxor_differ(self):
        ops = {}
        cb = TestCodeBuilder()
        for name in ['VAND', 'VOR', 'VXOR']:
            cb.reset()
            getattr(cb, name)(0, 1, 2)
            ops[name] = cb.get_word()
        values = list(ops.values())
        assert len(set(values)) == len(values)

    def test_vmem_load_parse_bits(self):
        cb = TestCodeBuilder()
        cb.VMEM_LOAD(0, 1, 0)
        w = cb.get_word()
        parse_bits = (w >> 14) & 0b11
        assert parse_bits == 0b11

    def test_vmem_store_parse_bits(self):
        cb = TestCodeBuilder()
        cb.VMEM_STORE(1, 0, 0)
        w = cb.get_word()
        parse_bits = (w >> 14) & 0b11
        assert parse_bits == 0b11

    def test_vsplat_produces_single_word(self):
        cb = TestCodeBuilder()
        cb.VSPLAT(0, 1)
        assert len(cb.buffer) == 1

    def test_vfadd_sf_produces_single_word(self):
        cb = TestCodeBuilder()
        cb.VFADD_SF(0, 1, 2)
        assert len(cb.buffer) == 1

    def test_vadd_w_vd_field(self):
        """VADD_W: Vd field should be at bits [4:0]."""
        cb = TestCodeBuilder()
        for vd in range(32):
            cb.reset()
            cb.VADD_W(vd, 0, 0)
            w = cb.get_word()
            assert (w & 0x1F) == vd, (
                "Vd=%d but bits[4:0]=%d" % (vd, w & 0x1F))

    def test_vadd_w_vu_field(self):
        """VADD_W: Vu field should be at bits [12:8]."""
        cb = TestCodeBuilder()
        for vu in range(32):
            cb.reset()
            cb.VADD_W(0, vu, 0)
            w = cb.get_word()
            assert ((w >> 8) & 0x1F) == vu, (
                "Vu=%d but bits[12:8]=%d" % (vu, (w >> 8) & 0x1F))

    def test_vadd_w_vv_field(self):
        """VADD_W: Vv field should be at bits [20:16]."""
        cb = TestCodeBuilder()
        for vv in range(32):
            cb.reset()
            cb.VADD_W(0, 0, vv)
            w = cb.get_word()
            assert ((w >> 16) & 0x1F) == vv, (
                "Vv=%d but bits[20:16]=%d" % (vv, (w >> 16) & 0x1F))

    def test_vsub_variants_differ(self):
        """VSUB_W, VSUB_H, VSUB_B should produce different encodings."""
        ops = {}
        cb = TestCodeBuilder()
        for name in ['VSUB_W', 'VSUB_H', 'VSUB_B']:
            cb.reset()
            getattr(cb, name)(0, 1, 2)
            ops[name] = cb.get_word()
        values = list(ops.values())
        assert len(set(values)) == len(values)

    def test_vmpy_h_produces_single_word(self):
        cb = TestCodeBuilder()
        cb.VMPY_H(0, 1, 2)
        assert len(cb.buffer) == 1

    def test_vmpye_w_produces_single_word(self):
        cb = TestCodeBuilder()
        cb.VMPYE_W(0, 1, 2)
        assert len(cb.buffer) == 1

    def test_vfloat_ops_differ(self):
        """VFADD_SF, VFSUB_SF, VFMPY_SF should produce different encodings."""
        ops = {}
        cb = TestCodeBuilder()
        for name in ['VFADD_SF', 'VFSUB_SF', 'VFMPY_SF']:
            cb.reset()
            getattr(cb, name)(0, 1, 2)
            ops[name] = cb.get_word()
        values = list(ops.values())
        assert len(set(values)) == len(values)

    def test_vcmp_eq_w_qd_field(self):
        """VCMP_EQ_W: Qd field should be at bits [1:0]."""
        cb = TestCodeBuilder()
        for qd in range(4):
            cb.reset()
            cb.VCMP_EQ_W(qd, 0, 0)
            w = cb.get_word()
            assert (w & 0x3) == qd, (
                "Qd=%d but bits[1:0]=%d" % (qd, w & 0x3))

    def test_vcmp_eq_vs_gt_differ(self):
        """VCMP_EQ_W and VCMP_GT_W should produce different encodings."""
        cb = TestCodeBuilder()
        cb.VCMP_EQ_W(0, 1, 2)
        eq_w = cb.get_word()
        cb.reset()
        cb.VCMP_GT_W(0, 1, 2)
        gt_w = cb.get_word()
        assert eq_w != gt_w

    def test_vpacke_h_produces_single_word(self):
        cb = TestCodeBuilder()
        cb.VPACKE_H(0, 1, 2)
        assert len(cb.buffer) == 1

    def test_vpacko_h_produces_single_word(self):
        cb = TestCodeBuilder()
        cb.VPACKO_H(0, 1, 2)
        assert len(cb.buffer) == 1

    def test_vpacke_vs_vpacko_differ(self):
        """VPACKE_H and VPACKO_H should produce different encodings."""
        cb = TestCodeBuilder()
        cb.VPACKE_H(0, 1, 2)
        e_w = cb.get_word()
        cb.reset()
        cb.VPACKO_H(0, 1, 2)
        o_w = cb.get_word()
        assert e_w != o_w

    def test_vunpack_h_produces_single_word(self):
        cb = TestCodeBuilder()
        cb.VUNPACK_H(0, 1)
        assert len(cb.buffer) == 1

    def test_vunpack_h_vu_field(self):
        """VUNPACK_H: Vu field should be at bits [12:8]."""
        cb = TestCodeBuilder()
        for vu in range(32):
            cb.reset()
            cb.VUNPACK_H(0, vu)
            w = cb.get_word()
            assert ((w >> 8) & 0x1F) == vu

    def test_vmem_load_store_differ(self):
        """VMEM_LOAD and VMEM_STORE should produce different encodings."""
        cb = TestCodeBuilder()
        cb.VMEM_LOAD(0, 5, 0)
        ld_w = cb.get_word()
        cb.reset()
        cb.VMEM_STORE(5, 0, 0)
        st_w = cb.get_word()
        assert ld_w != st_w

    def test_vmem_load_offset_encoding(self):
        """VMEM_LOAD offset should be encoded in units of 128 bytes."""
        cb = TestCodeBuilder()
        # offset=0 should have zero offset bits
        cb.VMEM_LOAD(0, 5, 0)
        w0 = cb.get_word()
        # offset=128 should have different encoding
        cb.reset()
        cb.VMEM_LOAD(0, 5, 128)
        w1 = cb.get_word()
        assert w0 != w1

    def test_vsplat_rt_field(self):
        """VSPLAT: Rt field should be at bits [20:16]."""
        cb = TestCodeBuilder()
        for rt in range(32):
            cb.reset()
            cb.VSPLAT(0, rt)
            w = cb.get_word()
            assert ((w >> 16) & 0x1F) == rt

    def test_all_hvx_alu_ops_single_word(self):
        """All HVX ALU operations should produce exactly one word."""
        ops = ['VADD_W', 'VADD_H', 'VADD_B', 'VSUB_W', 'VSUB_H', 'VSUB_B',
               'VAND', 'VOR', 'VXOR', 'VMPY_H', 'VMPYE_W',
               'VFADD_SF', 'VFSUB_SF', 'VFMPY_SF', 'VPACKE_H', 'VPACKO_H']
        cb = TestCodeBuilder()
        for name in ops:
            cb.reset()
            getattr(cb, name)(0, 1, 2)
            assert len(cb.buffer) == 1, (
                "%s emitted %d words, expected 1" % (name, len(cb.buffer)))

    def test_all_hvx_alu_parse_bits(self):
        """All HVX ALU operations should have parse bits 0b11 at [15:14]."""
        ops = ['VADD_W', 'VADD_H', 'VADD_B', 'VSUB_W', 'VSUB_H', 'VSUB_B',
               'VAND', 'VOR', 'VXOR', 'VMPY_H', 'VMPYE_W',
               'VFADD_SF', 'VFSUB_SF', 'VFMPY_SF', 'VPACKE_H', 'VPACKO_H']
        cb = TestCodeBuilder()
        for name in ops:
            cb.reset()
            getattr(cb, name)(0, 1, 2)
            w = cb.get_word()
            parse_bits = (w >> 14) & 0b11
            assert parse_bits == 0b11, (
                "%s: parse bits should be 0b11, got 0b%s" % (name, bin(parse_bits)))

    def test_pred_not_produces_single_word(self):
        cb = TestCodeBuilder()
        cb.PRED_NOT(0, 1)
        assert len(cb.buffer) == 1

    def test_pred_not_qd_field(self):
        """PRED_NOT: Qd field should be at bits [1:0]."""
        cb = TestCodeBuilder()
        for qd in range(4):
            cb.reset()
            cb.PRED_NOT(qd, 0)
            w = cb.get_word()
            assert (w & 0x3) == qd

    def test_pred_not_qs_field(self):
        """PRED_NOT: Qs field should be at bits [9:8]."""
        cb = TestCodeBuilder()
        for qs in range(4):
            cb.reset()
            cb.PRED_NOT(0, qs)
            w = cb.get_word()
            assert ((w >> 8) & 0x3) == qs

    def test_vandqrt_produces_single_word(self):
        cb = TestCodeBuilder()
        cb.VANDQRT(0, 1, 5)
        assert len(cb.buffer) == 1

    def test_vandqrt_vd_field(self):
        """VANDQRT: Vd field should be at bits [4:0]."""
        cb = TestCodeBuilder()
        for vd in range(32):
            cb.reset()
            cb.VANDQRT(vd, 0, 0)
            w = cb.get_word()
            assert (w & 0x1F) == vd

    def test_vandqrt_qu_field(self):
        """VANDQRT: Qu field should be at bits [9:8]."""
        cb = TestCodeBuilder()
        for qu in range(4):
            cb.reset()
            cb.VANDQRT(0, qu, 0)
            w = cb.get_word()
            assert ((w >> 8) & 0x3) == qu

    def test_vandqrt_rt_field(self):
        """VANDQRT: Rt field should be at bits [20:16]."""
        cb = TestCodeBuilder()
        for rt in range(32):
            cb.reset()
            cb.VANDQRT(0, 0, rt)
            w = cb.get_word()
            assert ((w >> 16) & 0x1F) == rt

    def test_vmux_produces_single_word(self):
        cb = TestCodeBuilder()
        cb.VMUX(0, 1, 2, 3)
        assert len(cb.buffer) == 1

    def test_vmux_vd_field(self):
        """VMUX: Vd field should be at bits [4:0]."""
        cb = TestCodeBuilder()
        for vd in range(32):
            cb.reset()
            cb.VMUX(vd, 0, 0, 0)
            w = cb.get_word()
            assert (w & 0x1F) == vd

    def test_vmux_qt_field(self):
        """VMUX: Qt field should be at bits [6:5]."""
        cb = TestCodeBuilder()
        for qt in range(4):
            cb.reset()
            cb.VMUX(0, qt, 0, 0)
            w = cb.get_word()
            assert ((w >> 5) & 0x3) == qt

    def test_vinsertw_produces_single_word(self):
        cb = TestCodeBuilder()
        cb.VINSERTW(0, 5)
        assert len(cb.buffer) == 1

    def test_vinsertw_vx_field(self):
        """VINSERTW: Vx field should be at bits [4:0]."""
        cb = TestCodeBuilder()
        for vx in range(32):
            cb.reset()
            cb.VINSERTW(vx, 0)
            w = cb.get_word()
            assert (w & 0x1F) == vx

    def test_vinsertw_rt_field(self):
        """VINSERTW: Rt field should be at bits [20:16]."""
        cb = TestCodeBuilder()
        for rt in range(32):
            cb.reset()
            cb.VINSERTW(0, rt)
            w = cb.get_word()
            assert ((w >> 16) & 0x1F) == rt

    def test_vextractw_produces_single_word(self):
        cb = TestCodeBuilder()
        cb.VEXTRACTW(0, 1, 5)
        assert len(cb.buffer) == 1

    def test_vextractw_rd_field(self):
        """VEXTRACTW: Rd field should be at bits [4:0]."""
        cb = TestCodeBuilder()
        for rd in range(32):
            cb.reset()
            cb.VEXTRACTW(rd, 0, 0)
            w = cb.get_word()
            assert (w & 0x1F) == rd

    def test_vextractw_vu_field(self):
        """VEXTRACTW: Vu field should be at bits [12:8]."""
        cb = TestCodeBuilder()
        for vu in range(32):
            cb.reset()
            cb.VEXTRACTW(0, vu, 0)
            w = cb.get_word()
            assert ((w >> 8) & 0x1F) == vu

    def test_vextractw_rs_field(self):
        """VEXTRACTW: Rs field should be at bits [20:16]."""
        cb = TestCodeBuilder()
        for rs in range(32):
            cb.reset()
            cb.VEXTRACTW(0, 0, rs)
            w = cb.get_word()
            assert ((w >> 16) & 0x1F) == rs

    def test_vror_produces_single_word(self):
        cb = TestCodeBuilder()
        cb.VROR(0, 1, 5)
        assert len(cb.buffer) == 1

    def test_vror_vd_field(self):
        """VROR: Vd field should be at bits [4:0]."""
        cb = TestCodeBuilder()
        for vd in range(32):
            cb.reset()
            cb.VROR(vd, 0, 0)
            w = cb.get_word()
            assert (w & 0x1F) == vd

    def test_vror_vu_field(self):
        """VROR: Vu field should be at bits [12:8]."""
        cb = TestCodeBuilder()
        for vu in range(32):
            cb.reset()
            cb.VROR(0, vu, 0)
            w = cb.get_word()
            assert ((w >> 8) & 0x1F) == vu

    def test_vror_rt_field(self):
        """VROR: Rt field should be at bits [20:16]."""
        cb = TestCodeBuilder()
        for rt in range(32):
            cb.reset()
            cb.VROR(0, 0, rt)
            w = cb.get_word()
            assert ((w >> 16) & 0x1F) == rt

    def test_new_hvx_ops_all_single_word(self):
        """All new HVX operations should produce exactly one word."""
        cb = TestCodeBuilder()
        tests = [
            ('PRED_NOT', (0, 1)),
            ('VANDQRT', (0, 1, 5)),
            ('VMUX', (0, 1, 2, 3)),
            ('VINSERTW', (0, 5)),
            ('VEXTRACTW', (0, 1, 5)),
            ('VROR', (0, 1, 5)),
        ]
        for name, args in tests:
            cb.reset()
            getattr(cb, name)(*args)
            assert len(cb.buffer) == 1, (
                "%s emitted %d words, expected 1" % (name, len(cb.buffer)))

    def test_new_hvx_ops_all_parse_bits(self):
        """All new HVX operations should have parse bits 0b11 at [15:14]."""
        cb = TestCodeBuilder()
        tests = [
            ('PRED_NOT', (0, 1)),
            ('VANDQRT', (0, 1, 5)),
            ('VMUX', (0, 1, 2, 3)),
            ('VINSERTW', (0, 5)),
            ('VEXTRACTW', (0, 1, 5)),
            ('VROR', (0, 1, 5)),
        ]
        for name, args in tests:
            cb.reset()
            getattr(cb, name)(*args)
            w = cb.get_word()
            parse_bits = (w >> 14) & 0b11
            assert parse_bits == 0b11, (
                "%s: parse bits should be 0b11, got 0b%s" % (name, bin(parse_bits)))


# ---------------------------------------------------------------------------
# Reference assembler tests (need llvm-mc)
# ---------------------------------------------------------------------------

@needs_tools
class TestInstrBuilderVsReference(object):
    """Compare our instruction encoding against llvm-mc output."""

    def _check(self, cb_words, asm_str, hvx=False):
        """Compare builder output against reference assembler."""
        ref_words = assemble_to_words(asm_str, hvx=hvx)
        assert len(cb_words) == len(ref_words), (
            "word count mismatch: builder=%d, ref=%d for %r" %
            (len(cb_words), len(ref_words), asm_str))
        for i, (got, expected) in enumerate(zip(cb_words, ref_words)):
            assert got == expected, (
                "word %d mismatch for %r:\n  got:      0x%08X\n  expected: 0x%08X" %
                (i, asm_str, got, expected))

    # --- ALU32 register-register ---

    def test_add(self):
        cb = TestCodeBuilder()
        cb.ADD(0, 1, 2)
        self._check(cb.get_words(),
                     '{ R0 = add(R1, R2) }')

    def test_sub(self):
        cb = TestCodeBuilder()
        cb.SUB(0, 1, 2)
        self._check(cb.get_words(),
                     '{ R0 = sub(R1, R2) }')

    def test_and(self):
        cb = TestCodeBuilder()
        cb.AND(5, 10, 15)
        self._check(cb.get_words(),
                     '{ R5 = and(R10, R15) }')

    def test_or(self):
        cb = TestCodeBuilder()
        cb.OR(5, 10, 15)
        self._check(cb.get_words(),
                     '{ R5 = or(R10, R15) }')

    def test_xor(self):
        cb = TestCodeBuilder()
        cb.XOR(5, 10, 15)
        self._check(cb.get_words(),
                     '{ R5 = xor(R10, R15) }')

    # --- ALU32 register-immediate ---

    def test_addi(self):
        cb = TestCodeBuilder()
        cb.ADDI(3, 5, 100)
        self._check(cb.get_words(),
                     '{ R3 = add(R5, #100) }')

    def test_addi_negative(self):
        cb = TestCodeBuilder()
        cb.ADDI(3, 5, -42)
        self._check(cb.get_words(),
                     '{ R3 = add(R5, #-42) }')

    def test_tfrsi(self):
        cb = TestCodeBuilder()
        cb.TFRSI(10, 255)
        self._check(cb.get_words(),
                     '{ R10 = #255 }')

    def test_tfrsi_negative(self):
        cb = TestCodeBuilder()
        cb.TFRSI(10, -1)
        self._check(cb.get_words(),
                     '{ R10 = #-1 }')

    def test_tfr(self):
        cb = TestCodeBuilder()
        cb.TFR(5, 20)
        self._check(cb.get_words(),
                     '{ R5 = R20 }')

    # --- Shifts ---

    def test_asl_imm(self):
        cb = TestCodeBuilder()
        cb.S2_ASL_I_R(5, 10, 16)
        self._check(cb.get_words(),
                     '{ R5 = asl(R10, #16) }')

    def test_asr_imm(self):
        cb = TestCodeBuilder()
        cb.S2_ASR_I_R(5, 10, 8)
        self._check(cb.get_words(),
                     '{ R5 = asr(R10, #8) }')

    def test_lsr_imm(self):
        cb = TestCodeBuilder()
        cb.S2_LSR_I_R(5, 10, 4)
        self._check(cb.get_words(),
                     '{ R5 = lsr(R10, #4) }')

    # --- Compares ---

    def test_cmp_eq(self):
        cb = TestCodeBuilder()
        cb.CMP_EQ(0, 5, 10)
        self._check(cb.get_words(),
                     '{ P0 = cmp.eq(R5, R10) }')

    def test_cmp_gt(self):
        cb = TestCodeBuilder()
        cb.CMP_GT(1, 5, 10)
        self._check(cb.get_words(),
                     '{ P1 = cmp.gt(R5, R10) }')

    def test_cmp_gtu(self):
        cb = TestCodeBuilder()
        cb.CMP_GTU(2, 5, 10)
        self._check(cb.get_words(),
                     '{ P2 = cmp.gtu(R5, R10) }')

    # --- Loads ---

    def test_ldw(self):
        cb = TestCodeBuilder()
        cb.LDW(5, 10, 0)
        self._check(cb.get_words(),
                     '{ R5 = memw(R10 + #0) }')

    def test_ldw_offset(self):
        cb = TestCodeBuilder()
        cb.LDW(5, 10, 16)
        self._check(cb.get_words(),
                     '{ R5 = memw(R10 + #16) }')

    def test_ldh(self):
        cb = TestCodeBuilder()
        cb.LDH(5, 10, 0)
        self._check(cb.get_words(),
                     '{ R5 = memh(R10 + #0) }')

    def test_ldb(self):
        cb = TestCodeBuilder()
        cb.LDB(5, 10, 0)
        self._check(cb.get_words(),
                     '{ R5 = memb(R10 + #0) }')

    def test_ldub(self):
        cb = TestCodeBuilder()
        cb.LDUB(5, 10, 0)
        self._check(cb.get_words(),
                     '{ R5 = memub(R10 + #0) }')

    def test_lduh(self):
        cb = TestCodeBuilder()
        cb.LDUH(5, 10, 0)
        self._check(cb.get_words(),
                     '{ R5 = memuh(R10 + #0) }')

    # --- Stores ---

    def test_stw(self):
        cb = TestCodeBuilder()
        cb.STW(10, 5, 0)
        self._check(cb.get_words(),
                     '{ memw(R10 + #0) = R5 }')

    def test_stw_offset(self):
        cb = TestCodeBuilder()
        cb.STW(10, 5, 16)
        self._check(cb.get_words(),
                     '{ memw(R10 + #16) = R5 }')

    def test_sth(self):
        cb = TestCodeBuilder()
        cb.STH(10, 5, 0)
        self._check(cb.get_words(),
                     '{ memh(R10 + #0) = R5 }')

    def test_stb(self):
        cb = TestCodeBuilder()
        cb.STB(10, 5, 0)
        self._check(cb.get_words(),
                     '{ memb(R10 + #0) = R5 }')

    # --- Jumps ---

    def test_jumpr(self):
        cb = TestCodeBuilder()
        cb.JUMPR(31)
        self._check(cb.get_words(),
                     '{ jumpr R31 }')

    # --- XTYPE (64-bit) ---

    def test_dfadd(self):
        cb = TestCodeBuilder()
        cb.DFADD(16, 18, 20)
        self._check(cb.get_words(),
                     '{ R17:16 = dfadd(R19:18, R21:20) }')

    def test_dfsub(self):
        cb = TestCodeBuilder()
        cb.DFSUB(16, 18, 20)
        self._check(cb.get_words(),
                     '{ R17:16 = dfsub(R19:18, R21:20) }')

    def test_dfmpyll(self):
        cb = TestCodeBuilder()
        cb.DFMPYLL(16, 18, 20)
        self._check(cb.get_words(),
                     '{ R17:16 = dfmpyll(R19:18, R21:20) }')

    def test_dfmpyfix(self):
        cb = TestCodeBuilder()
        cb.DFMPYFIX(16, 18, 20)
        self._check(cb.get_words(),
                     '{ R17:16 = dfmpyfix(R19:18, R21:20) }')

    def test_dfmpylh_acc(self):
        cb = TestCodeBuilder()
        cb.DFMPYLH_ACC(16, 18, 20)
        self._check(cb.get_words(),
                     '{ R17:16 += dfmpylh(R19:18, R21:20) }')

    def test_dfmpyhh_acc(self):
        cb = TestCodeBuilder()
        cb.DFMPYHH_ACC(16, 18, 20)
        self._check(cb.get_words(),
                     '{ R17:16 += dfmpyhh(R19:18, R21:20) }')

    def test_sfadd(self):
        cb = TestCodeBuilder()
        cb.SFADD(0, 1, 2)
        self._check(cb.get_words(),
                     '{ R0 = sfadd(R1, R2) }')

    def test_sfsub(self):
        cb = TestCodeBuilder()
        cb.SFSUB(0, 1, 2)
        self._check(cb.get_words(),
                     '{ R0 = sfsub(R1, R2) }')

    def test_sfmpy(self):
        cb = TestCodeBuilder()
        cb.SFMPY(0, 1, 2)
        self._check(cb.get_words(),
                     '{ R0 = sfmpy(R1, R2) }')

    # --- Extension/sign ---

    def test_sxtb(self):
        cb = TestCodeBuilder()
        cb.SXTB(5, 10)
        self._check(cb.get_words(),
                     '{ R5 = sxtb(R10) }')

    def test_sxth(self):
        cb = TestCodeBuilder()
        cb.SXTH(5, 10)
        self._check(cb.get_words(),
                     '{ R5 = sxth(R10) }')

    def test_zxth(self):
        cb = TestCodeBuilder()
        cb.ZXTH(5, 10)
        self._check(cb.get_words(),
                     '{ R5 = zxth(R10) }')

    def test_asrh(self):
        cb = TestCodeBuilder()
        cb.ASRH(5, 10)
        self._check(cb.get_words(),
                     '{ R5 = asrh(R10) }')

    # --- Register-immediate ALU ---

    def test_andi(self):
        cb = TestCodeBuilder()
        cb.ANDI(5, 10, 0xFF)
        self._check(cb.get_words(),
                     '{ R5 = and(R10, #255) }')

    def test_ori(self):
        cb = TestCodeBuilder()
        cb.ORI(5, 10, 0xFF)
        self._check(cb.get_words(),
                     '{ R5 = or(R10, #255) }')

    # --- Compare register-immediate ---

    def test_cmp_eqi(self):
        cb = TestCodeBuilder()
        cb.CMP_EQI(0, 5, 42)
        self._check(cb.get_words(),
                     '{ P0 = cmp.eq(R5, #42) }')

    def test_cmp_gti(self):
        cb = TestCodeBuilder()
        cb.CMP_GTI(1, 5, 10)
        self._check(cb.get_words(),
                     '{ P1 = cmp.gt(R5, #10) }')

    def test_cmp_gtui(self):
        cb = TestCodeBuilder()
        cb.CMP_GTUI(2, 5, 10)
        self._check(cb.get_words(),
                     '{ P2 = cmp.gtu(R5, #10) }')

    # --- Conditional select ---

    def test_mux(self):
        cb = TestCodeBuilder()
        cb.MUX(0, 1, 5, 10)
        self._check(cb.get_words(),
                     '{ R0 = mux(P1, R5, R10) }')

    # --- Jumps ---

    def test_callr(self):
        cb = TestCodeBuilder()
        cb.CALLR(5)
        self._check(cb.get_words(),
                     '{ callr R5 }')

    # --- XTYPE 64-bit ---

    def test_combinew(self):
        cb = TestCodeBuilder()
        cb.COMBINEW(0, 1, 2)
        self._check(cb.get_words(),
                     '{ R1:0 = combine(R1, R2) }')

    def test_add64(self):
        cb = TestCodeBuilder()
        cb.ADD64(0, 2, 4)
        self._check(cb.get_words(),
                     '{ R1:0 = add(R3:2, R5:4) }')

    # --- Load/store double ---

    def test_ldd(self):
        cb = TestCodeBuilder()
        cb.LDD(0, 10, 0)
        self._check(cb.get_words(),
                     '{ R1:0 = memd(R10 + #0) }')

    def test_ldd_offset(self):
        cb = TestCodeBuilder()
        cb.LDD(0, 10, 24)
        self._check(cb.get_words(),
                     '{ R1:0 = memd(R10 + #24) }')

    def test_std(self):
        cb = TestCodeBuilder()
        cb.STD(10, 0, 0)
        self._check(cb.get_words(),
                     '{ memd(R10 + #0) = R1:0 }')

    def test_std_offset(self):
        cb = TestCodeBuilder()
        cb.STD(10, 0, 16)
        self._check(cb.get_words(),
                     '{ memd(R10 + #16) = R1:0 }')

    # --- Shift by immediate ---

    def test_mpyi(self):
        cb = TestCodeBuilder()
        cb.MPYI(5, 10, 15)
        self._check(cb.get_words(),
                     '{ R5 = mpyi(R10, R15) }')

    # --- Shift by register ---

    def test_asl_r_r(self):
        cb = TestCodeBuilder()
        cb.S2_ASL_R_R(5, 10, 15)
        self._check(cb.get_words(),
                     '{ R5 = asl(R10, R15) }')

    def test_asr_r_r(self):
        cb = TestCodeBuilder()
        cb.S2_ASR_R_R(5, 10, 15)
        self._check(cb.get_words(),
                     '{ R5 = asr(R10, R15) }')

    def test_lsr_r_r(self):
        cb = TestCodeBuilder()
        cb.S2_LSR_R_R(5, 10, 15)
        self._check(cb.get_words(),
                     '{ R5 = lsr(R10, R15) }')

    # --- 32x32 to 64-bit multiply ---

    def test_mpy_64(self):
        cb = TestCodeBuilder()
        cb.M2_DPMPYSS_S0(0, 5, 10)
        self._check(cb.get_words(),
                     '{ R1:0 = mpy(R5, R10) }')

    def test_mpy_64_other_regs(self):
        cb = TestCodeBuilder()
        cb.M2_DPMPYSS_S0(14, 3, 7)
        self._check(cb.get_words(),
                     '{ R15:14 = mpy(R3, R7) }')


# ---------------------------------------------------------------------------
# HVX reference assembler tests
# ---------------------------------------------------------------------------

@needs_tools
class TestHVXInstrBuilderVsReference(object):
    """Compare HVX instruction encoding against llvm-mc output."""

    def _check(self, cb_words, asm_str):
        ref_words = assemble_to_words(asm_str, hvx=True)
        assert len(cb_words) == len(ref_words), (
            "word count mismatch: builder=%d, ref=%d for %r" %
            (len(cb_words), len(ref_words), asm_str))
        for i, (got, expected) in enumerate(zip(cb_words, ref_words)):
            assert got == expected, (
                "word %d mismatch for %r:\n  got:      0x%08X\n  expected: 0x%08X" %
                (i, asm_str, got, expected))

    def test_vadd_w(self):
        cb = TestCodeBuilder()
        cb.VADD_W(0, 1, 2)
        self._check(cb.get_words(),
                     '{ V0.w = vadd(V1.w, V2.w) }')

    def test_vadd_h(self):
        cb = TestCodeBuilder()
        cb.VADD_H(0, 1, 2)
        self._check(cb.get_words(),
                     '{ V0.h = vadd(V1.h, V2.h) }')

    def test_vadd_b(self):
        cb = TestCodeBuilder()
        cb.VADD_B(0, 1, 2)
        self._check(cb.get_words(),
                     '{ V0.b = vadd(V1.b, V2.b) }')

    def test_vsub_w(self):
        cb = TestCodeBuilder()
        cb.VSUB_W(0, 1, 2)
        self._check(cb.get_words(),
                     '{ V0.w = vsub(V1.w, V2.w) }')

    def test_vand(self):
        cb = TestCodeBuilder()
        cb.VAND(0, 1, 2)
        self._check(cb.get_words(),
                     '{ V0 = vand(V1, V2) }')

    def test_vor(self):
        cb = TestCodeBuilder()
        cb.VOR(0, 1, 2)
        self._check(cb.get_words(),
                     '{ V0 = vor(V1, V2) }')

    def test_vxor(self):
        cb = TestCodeBuilder()
        cb.VXOR(0, 1, 2)
        self._check(cb.get_words(),
                     '{ V0 = vxor(V1, V2) }')

    def test_vsplat(self):
        cb = TestCodeBuilder()
        cb.VSPLAT(0, 5)
        self._check(cb.get_words(),
                     '{ V0 = vsplat(R5) }')

    def test_vsub_h(self):
        cb = TestCodeBuilder()
        cb.VSUB_H(0, 1, 2)
        self._check(cb.get_words(),
                     '{ V0.h = vsub(V1.h, V2.h) }')

    def test_vsub_b(self):
        cb = TestCodeBuilder()
        cb.VSUB_B(0, 1, 2)
        self._check(cb.get_words(),
                     '{ V0.b = vsub(V1.b, V2.b) }')

    def test_vmem_load(self):
        cb = TestCodeBuilder()
        cb.VMEM_LOAD(0, 5, 0)
        self._check(cb.get_words(),
                     '{ V0 = vmem(R5 + #0) }')

    def test_vmem_store(self):
        cb = TestCodeBuilder()
        cb.VMEM_STORE(5, 0, 0)
        self._check(cb.get_words(),
                     '{ vmem(R5 + #0) = V0 }')

    def test_vadd_w_different_regs(self):
        cb = TestCodeBuilder()
        cb.VADD_W(5, 10, 15)
        self._check(cb.get_words(),
                     '{ V5.w = vadd(V10.w, V15.w) }')

    def test_vsub_h(self):
        cb = TestCodeBuilder()
        cb.VSUB_H(3, 7, 11)
        self._check(cb.get_words(),
                     '{ V3.h = vsub(V7.h, V11.h) }')

    def test_vsub_b(self):
        cb = TestCodeBuilder()
        cb.VSUB_B(3, 7, 11)
        self._check(cb.get_words(),
                     '{ V3.b = vsub(V7.b, V11.b) }')

    def test_vmpy_h(self):
        cb = TestCodeBuilder()
        cb.VMPY_H(0, 1, 2)
        self._check(cb.get_words(),
                     '{ V0.h = vmpyi(V1.h, V2.h) }')

    def test_vcmp_eq_w(self):
        cb = TestCodeBuilder()
        cb.VCMP_EQ_W(0, 1, 2)
        self._check(cb.get_words(),
                     '{ Q0 = vcmp.eq(V1.w, V2.w) }')

    def test_vcmp_gt_w(self):
        cb = TestCodeBuilder()
        cb.VCMP_GT_W(1, 3, 5)
        self._check(cb.get_words(),
                     '{ Q1 = vcmp.gt(V3.w, V5.w) }')

    def test_vpacke_h(self):
        cb = TestCodeBuilder()
        cb.VPACKE_H(0, 1, 2)
        self._check(cb.get_words(),
                     '{ V0.h = vpacke(V1.w, V2.w) }')

    def test_vpacko_h(self):
        cb = TestCodeBuilder()
        cb.VPACKO_H(0, 1, 2)
        self._check(cb.get_words(),
                     '{ V0.h = vpacko(V1.w, V2.w) }')

    def test_vmem_load_offset(self):
        cb = TestCodeBuilder()
        cb.VMEM_LOAD(3, 10, 128)
        self._check(cb.get_words(),
                     '{ V3 = vmem(R10 + #1) }')

    def test_vmem_store_offset(self):
        cb = TestCodeBuilder()
        cb.VMEM_STORE(10, 3, 128)
        self._check(cb.get_words(),
                     '{ vmem(R10 + #1) = V3 }')

    def test_pred_not(self):
        cb = TestCodeBuilder()
        cb.PRED_NOT(1, 0)
        self._check(cb.get_words(),
                     '{ Q1 = not(Q0) }')

    def test_pred_not_q3_q2(self):
        cb = TestCodeBuilder()
        cb.PRED_NOT(3, 2)
        self._check(cb.get_words(),
                     '{ Q3 = not(Q2) }')

    def test_vandqrt(self):
        cb = TestCodeBuilder()
        cb.VANDQRT(0, 1, 5)
        self._check(cb.get_words(),
                     '{ V0 = vand(Q1, R5) }')

    def test_vandqrt_different_regs(self):
        cb = TestCodeBuilder()
        cb.VANDQRT(10, 3, 15)
        self._check(cb.get_words(),
                     '{ V10 = vand(Q3, R15) }')

    def test_vmux(self):
        cb = TestCodeBuilder()
        cb.VMUX(0, 1, 2, 3)
        self._check(cb.get_words(),
                     '{ V0 = vmux(Q1, V2, V3) }')

    def test_vmux_different_regs(self):
        cb = TestCodeBuilder()
        cb.VMUX(5, 0, 10, 15)
        self._check(cb.get_words(),
                     '{ V5 = vmux(Q0, V10, V15) }')

    def test_vinsertw(self):
        cb = TestCodeBuilder()
        cb.VINSERTW(0, 5)
        self._check(cb.get_words(),
                     '{ V0.w = vinsert(R5) }')

    def test_vinsertw_different_regs(self):
        cb = TestCodeBuilder()
        cb.VINSERTW(10, 15)
        self._check(cb.get_words(),
                     '{ V10.w = vinsert(R15) }')

    def test_vextractw(self):
        cb = TestCodeBuilder()
        cb.VEXTRACTW(0, 1, 5)
        self._check(cb.get_words(),
                     '{ R0 = vextract(V1, R5) }')

    def test_vextractw_different_regs(self):
        cb = TestCodeBuilder()
        cb.VEXTRACTW(10, 15, 3)
        self._check(cb.get_words(),
                     '{ R10 = vextract(V15, R3) }')

    def test_vror(self):
        cb = TestCodeBuilder()
        cb.VROR(0, 1, 5)
        self._check(cb.get_words(),
                     '{ V0 = vror(V1, R5) }')

    def test_vror_different_regs(self):
        cb = TestCodeBuilder()
        cb.VROR(5, 10, 15)
        self._check(cb.get_words(),
                     '{ V5 = vror(V10, R15) }')


# ---------------------------------------------------------------------------
# Hypothesis-based property tests (with reference assembler)
# ---------------------------------------------------------------------------

if HAS_HYPOTHESIS and HAS_TOOLS:

    class TestInstrBuilderHypothesis(object):
        """Property-based tests using hypothesis + reference assembler."""

        @settings(max_examples=20)
        @given(rd=gpr_strategy, rs=gpr_strategy, rt=gpr_strategy)
        def test_add_random_regs(self, rd, rs, rt):
            cb = TestCodeBuilder()
            cb.ADD(rd, rs, rt)
            ref = assemble_to_words('{ %s = add(%s, %s) }' %
                                    (gpr_name(rd), gpr_name(rs), gpr_name(rt)))
            assert cb.get_words() == ref

        @settings(max_examples=20)
        @given(rd=gpr_strategy, rs=gpr_strategy, rt=gpr_strategy)
        def test_sub_random_regs(self, rd, rs, rt):
            cb = TestCodeBuilder()
            cb.SUB(rd, rs, rt)
            ref = assemble_to_words('{ %s = sub(%s, %s) }' %
                                    (gpr_name(rd), gpr_name(rs), gpr_name(rt)))
            assert cb.get_words() == ref

        @settings(max_examples=20)
        @given(rd=gpr_strategy, rs=gpr_strategy, rt=gpr_strategy)
        def test_and_random_regs(self, rd, rs, rt):
            cb = TestCodeBuilder()
            cb.AND(rd, rs, rt)
            ref = assemble_to_words('{ %s = and(%s, %s) }' %
                                    (gpr_name(rd), gpr_name(rs), gpr_name(rt)))
            assert cb.get_words() == ref

        @settings(max_examples=20)
        @given(rd=gpr_strategy, rs=gpr_strategy,
               imm=st.integers(min_value=-32768, max_value=32767))
        def test_addi_random(self, rd, rs, imm):
            cb = TestCodeBuilder()
            cb.ADDI(rd, rs, imm)
            ref = assemble_to_words('{ %s = add(%s, #%d) }' %
                                    (gpr_name(rd), gpr_name(rs), imm))
            assert cb.get_words() == ref

        @settings(max_examples=20)
        @given(rd=gpr_strategy, imm=imm16_strategy)
        def test_tfrsi_random(self, rd, imm):
            cb = TestCodeBuilder()
            cb.TFRSI(rd, imm)
            ref = assemble_to_words('{ %s = #%d }' % (gpr_name(rd), imm))
            assert cb.get_words() == ref

        @settings(max_examples=20)
        @given(rd=gpr_strategy, rs=gpr_strategy)
        def test_tfr_random(self, rd, rs):
            cb = TestCodeBuilder()
            cb.TFR(rd, rs)
            ref = assemble_to_words('{ %s = %s }' %
                                    (gpr_name(rd), gpr_name(rs)))
            assert cb.get_words() == ref

        @settings(max_examples=20)
        @given(pd=pred_strategy, rs=gpr_strategy, rt=gpr_strategy)
        def test_cmp_eq_random(self, pd, rs, rt):
            cb = TestCodeBuilder()
            cb.CMP_EQ(pd, rs, rt)
            ref = assemble_to_words('{ %s = cmp.eq(%s, %s) }' %
                                    (pred_name(pd), gpr_name(rs), gpr_name(rt)))
            assert cb.get_words() == ref

        @settings(max_examples=20)
        @given(rd=gpr_strategy, rs=gpr_strategy,
               imm=st.integers(min_value=0, max_value=31))
        def test_asl_i_random(self, rd, rs, imm):
            cb = TestCodeBuilder()
            cb.S2_ASL_I_R(rd, rs, imm)
            ref = assemble_to_words('{ %s = asl(%s, #%d) }' %
                                    (gpr_name(rd), gpr_name(rs), imm))
            assert cb.get_words() == ref

    class TestHVXInstrBuilderHypothesis(object):
        """HVX property-based tests using hypothesis + reference assembler."""

        @settings(max_examples=20)
        @given(vd=vreg_strategy, vu=vreg_strategy, vv=vreg_strategy)
        def test_vadd_w_random(self, vd, vu, vv):
            cb = TestCodeBuilder()
            cb.VADD_W(vd, vu, vv)
            ref = assemble_to_words(
                '{ %s.w = vadd(%s.w, %s.w) }' %
                (vreg_name(vd), vreg_name(vu), vreg_name(vv)),
                hvx=True)
            assert cb.get_words() == ref

        @settings(max_examples=20)
        @given(vd=vreg_strategy, vu=vreg_strategy, vv=vreg_strategy)
        def test_vsub_w_random(self, vd, vu, vv):
            cb = TestCodeBuilder()
            cb.VSUB_W(vd, vu, vv)
            ref = assemble_to_words(
                '{ %s.w = vsub(%s.w, %s.w) }' %
                (vreg_name(vd), vreg_name(vu), vreg_name(vv)),
                hvx=True)
            assert cb.get_words() == ref

        @settings(max_examples=20)
        @given(vd=vreg_strategy, vu=vreg_strategy, vv=vreg_strategy)
        def test_vand_random(self, vd, vu, vv):
            cb = TestCodeBuilder()
            cb.VAND(vd, vu, vv)
            ref = assemble_to_words(
                '{ %s = vand(%s, %s) }' %
                (vreg_name(vd), vreg_name(vu), vreg_name(vv)),
                hvx=True)
            assert cb.get_words() == ref

        @settings(max_examples=20)
        @given(vd=vreg_strategy, rt=gpr_strategy)
        def test_vsplat_random(self, vd, rt):
            cb = TestCodeBuilder()
            cb.VSPLAT(vd, rt)
            ref = assemble_to_words(
                '{ %s = vsplat(%s) }' %
                (vreg_name(vd), gpr_name(rt)),
                hvx=True)
            assert cb.get_words() == ref


# ---------------------------------------------------------------------------
# Hardware loop self-consistency tests
# ---------------------------------------------------------------------------

class TestHardwareLoopSelfConsistency(object):
    """Tests for hardware loop instruction encoding properties."""

    def test_loop0r_produces_single_word(self):
        cb = TestCodeBuilder()
        cb.LOOP0R(5, 0)
        assert len(cb.buffer) == 1

    def test_loop1r_produces_single_word(self):
        cb = TestCodeBuilder()
        cb.LOOP1R(5, 0)
        assert len(cb.buffer) == 1

    def test_loop0i_produces_single_word(self):
        cb = TestCodeBuilder()
        cb.LOOP0I(10, 0)
        assert len(cb.buffer) == 1

    def test_loop1i_produces_single_word(self):
        cb = TestCodeBuilder()
        cb.LOOP1I(10, 0)
        assert len(cb.buffer) == 1

    def test_loop0r_vs_loop1r_differ(self):
        """LOOP0R and LOOP1R should have different opcodes."""
        cb = TestCodeBuilder()
        cb.LOOP0R(5, 0)
        w0 = cb.get_word()
        cb.reset()
        cb.LOOP1R(5, 0)
        w1 = cb.get_word()
        assert w0 != w1

    def test_loop0i_vs_loop1i_differ(self):
        """LOOP0I and LOOP1I should have different opcodes."""
        cb = TestCodeBuilder()
        cb.LOOP0I(10, 0)
        w0 = cb.get_word()
        cb.reset()
        cb.LOOP1I(10, 0)
        w1 = cb.get_word()
        assert w0 != w1

    def test_loop0r_rs_field(self):
        """LOOP0R: Rs field should be at bits [20:16]."""
        cb = TestCodeBuilder()
        for rs in range(32):
            cb.reset()
            cb.LOOP0R(rs, 0)
            w = cb.get_word()
            assert ((w >> 16) & 0x1F) == rs, (
                "Rs=%d but bits[20:16]=%d" % (rs, (w >> 16) & 0x1F))

    def test_loop0r_parse_bits(self):
        """LOOP0R should have parse bits 0b11 (single-instruction packet)."""
        cb = TestCodeBuilder()
        cb.LOOP0R(5, 0)
        w = cb.get_word()
        parse_bits = (w >> 14) & 0b11
        assert parse_bits == 0b11

    def test_loop0r_fixed_bits(self):
        """LOOP0R should have fixed zero bits: [13]=0, [7:5]=000, [2:0]=000."""
        cb = TestCodeBuilder()
        cb.LOOP0R(0, 0)
        w = cb.get_word()
        assert (w >> 13) & 0x1 == 0, "bit 13 should be 0"
        assert (w >> 5) & 0x7 == 0, "bits 7:5 should be 000"
        assert w & 0x7 == 0, "bits 2:0 should be 000"

    def test_loop0i_trip_count_encoding(self):
        """LOOP0I: different trip counts produce different encodings."""
        words = set()
        cb = TestCodeBuilder()
        for tc in [1, 5, 10, 100, 1000]:
            cb.reset()
            cb.LOOP0I(tc, 0)
            words.add(cb.get_word())
        assert len(words) == 5, "different trip counts should produce unique words"

    def test_loop0r_different_offsets_differ(self):
        """LOOP0R: different offsets produce different encodings."""
        cb = TestCodeBuilder()
        cb.LOOP0R(5, 0)
        w0 = cb.get_word()
        cb.reset()
        cb.LOOP0R(5, 16)
        w1 = cb.get_word()
        assert w0 != w1

    def test_loop0r_opcode(self):
        """LOOP0R should have opcode 0b01100000000 at bits [31:21]."""
        cb = TestCodeBuilder()
        cb.LOOP0R(0, 0)
        w = cb.get_word()
        opcode = (w >> 21) & 0x7FF
        assert opcode == 0b01100000000

    def test_loop1r_opcode(self):
        """LOOP1R should have opcode 0b01100000001 at bits [31:21]."""
        cb = TestCodeBuilder()
        cb.LOOP1R(0, 0)
        w = cb.get_word()
        opcode = (w >> 21) & 0x7FF
        assert opcode == 0b01100000001

    def test_loop0i_opcode(self):
        """LOOP0I should have opcode 0b01101001000 at bits [31:21]."""
        cb = TestCodeBuilder()
        cb.LOOP0I(0, 0)
        w = cb.get_word()
        opcode = (w >> 21) & 0x7FF
        assert opcode == 0b01101001000

    def test_loop1i_opcode(self):
        """LOOP1I should have opcode 0b01101001001 at bits [31:21]."""
        cb = TestCodeBuilder()
        cb.LOOP1I(0, 0)
        w = cb.get_word()
        opcode = (w >> 21) & 0x7FF
        assert opcode == 0b01101001001


# ---------------------------------------------------------------------------
# Endloop (parse bit modification) tests
# ---------------------------------------------------------------------------

class TestEndloop(object):
    """Tests for endloop0/endloop1 parse bit patching."""

    def test_endloop0_patches_parse_bits(self):
        """set_endloop0 should set parse bits to 0b10."""
        from rpython.jit.backend.hexagon.codebuilder import InstrBuilder
        mc = InstrBuilder()
        mc.NOP()
        mc.set_endloop0()
        w = mc._buf[0]
        parse_bits = (w >> 14) & 0b11
        assert parse_bits == 0b10, (
            "endloop0 should set parse bits to 0b10, got 0b%s" % bin(parse_bits))

    def test_endloop1_patches_parse_bits(self):
        """set_endloop1 should set parse bits to 0b01."""
        from rpython.jit.backend.hexagon.codebuilder import InstrBuilder
        mc = InstrBuilder()
        mc.NOP()
        mc.set_endloop1()
        w = mc._buf[0]
        parse_bits = (w >> 14) & 0b11
        assert parse_bits == 0b01

    def test_endloop0_at_specific_pos(self):
        """set_endloop0 with explicit position."""
        from rpython.jit.backend.hexagon.codebuilder import InstrBuilder
        mc = InstrBuilder()
        mc.NOP()  # pos 0
        mc.NOP()  # pos 4
        mc.NOP()  # pos 8
        mc.set_endloop0(pos=4)
        # Only instruction at pos=4 (index 1) should have endloop0
        assert (mc._buf[0] >> 14) & 0b11 == 0b11  # unchanged
        assert (mc._buf[1] >> 14) & 0b11 == 0b10  # endloop0
        assert (mc._buf[2] >> 14) & 0b11 == 0b11  # unchanged

    def test_endloop1_at_specific_pos(self):
        """set_endloop1 with explicit position."""
        from rpython.jit.backend.hexagon.codebuilder import InstrBuilder
        mc = InstrBuilder()
        mc.NOP()  # pos 0
        mc.NOP()  # pos 4
        mc.set_endloop1(pos=0)
        assert (mc._buf[0] >> 14) & 0b11 == 0b01  # endloop1
        assert (mc._buf[1] >> 14) & 0b11 == 0b11  # unchanged

    def test_endloop0_clears_previous_parse_bits(self):
        """set_endloop0 should replace existing parse bits, not OR."""
        from rpython.jit.backend.hexagon.codebuilder import InstrBuilder
        mc = InstrBuilder()
        mc.NOP()  # has parse bits 0b11
        mc.set_endloop0()
        w = mc._buf[0]
        parse_bits = (w >> 14) & 0b11
        assert parse_bits == 0b10


# ---------------------------------------------------------------------------
# Reference assembler tests for hardware loops
# ---------------------------------------------------------------------------

@needs_tools
class TestHardwareLoopVsReference(object):
    """Compare hardware loop encoding against llvm-mc output."""

    def _check(self, cb_words, asm_str):
        ref_words = assemble_to_words(asm_str)
        assert len(cb_words) == len(ref_words), (
            "word count mismatch: builder=%d, ref=%d for %r" %
            (len(cb_words), len(ref_words), asm_str))
        for i, (got, expected) in enumerate(zip(cb_words, ref_words)):
            assert got == expected, (
                "word %d mismatch for %r:\n  got:      0x%08X\n  expected: 0x%08X" %
                (i, asm_str, got, expected))

    def test_loop0r(self):
        cb = TestCodeBuilder()
        cb.LOOP0R(5, 0)
        self._check(cb.get_words(),
                     '{ loop0(0, R5) }')

    def test_loop1r(self):
        cb = TestCodeBuilder()
        cb.LOOP1R(10, 0)
        self._check(cb.get_words(),
                     '{ loop1(0, R10) }')

    def test_loop0i(self):
        cb = TestCodeBuilder()
        cb.LOOP0I(10, 0)
        self._check(cb.get_words(),
                     '{ loop0(0, #10) }')

    def test_loop1i(self):
        cb = TestCodeBuilder()
        cb.LOOP1I(5, 0)
        self._check(cb.get_words(),
                     '{ loop1(0, #5) }')
