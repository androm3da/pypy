"""
HVX (Hexagon Vector Extensions) support for the Hexagon JIT backend.

HVX provides 128-byte (1024-bit) vector registers (V0-V31), vector pairs
(V1:V0, V3:V2, etc.), and vector predicate registers (Q0-Q3). Operations
include integer and floating-point vector arithmetic, loads/stores,
shuffles, and predicated operations.

Element sizes: byte (b), halfword (h), word (w)
Float sizes: qf16 (half), qf32 (single)

This module provides:
- HexagonVectorExt: VectorExt subclass for HVX configuration
- VectorAssemblerMixin: emit_op_vec_* methods for vector operations
- VectorRegallocMixin: prepare_op_vec_* methods for register allocation
- HVXInstructionMixin: low-level instruction encoding
"""

from rpython.jit.backend.hexagon import registers as r
from rpython.jit.backend.hexagon.arch import (
    INST_SIZE, WORD, PARSE_END_PACKET, PARSE_BITS_SHIFT,
)
from rpython.jit.backend.hexagon.detect import detect_hvx, detect_hvx_length
from rpython.jit.backend.hexagon.locations import ImmLocation
from rpython.jit.backend.llsupport.vector_ext import VectorExt
from rpython.jit.metainterp.resoperation import rop

PARSE_BITS = PARSE_END_PACKET << PARSE_BITS_SHIFT

# HVX vector register size in bytes (128 for V68+)
HVX_VEC_SIZE = 128


# ---------------------------------------------------------------------------
# HVX instruction encoding helpers
# ---------------------------------------------------------------------------

class HVXInstructionMixin(object):
    _mixin_ = True
    """HVX instruction encoding methods added to the code builder.

    HVX instructions use ICLASS 0b0001 (CVI classes) with various subtypes.
    Register fields: Vu@[12:8], Vv@[20:16] or [7:3], Vd@[4:0]
    """

    # -----------------------------------------------------------------------
    # Vector ALU: Vd = vadd/vsub/vand/vor/vxor(Vu, Vv).type
    # Encoding: CVI_VA class
    # -----------------------------------------------------------------------

    def _hvx_alu_vvv(self, vd, vu, vv, opcode_31_21, subop_7_5):
        """Common encoding for 3-register HVX ALU (Enc_45364e).
        Fields: opcode@[31:21], Vv@[20:16], 0@[13], Vu@[12:8],
                subop@[7:5], Vd@[4:0]
        Note: Vv is at [20:16], Vu is at [12:8] (per LLVM tablegen).
        """
        bits = (opcode_31_21 << 21) | (subop_7_5 << 5)
        self.write32(bits | PARSE_BITS |
                     (int(vv) << 16) | (int(vu) << 8) | int(vd))

    def VADD_W(self, vd, vu, vv):
        """Vd.w = vadd(Vu.w, Vv.w) --word-element vector add."""
        self._hvx_alu_vvv(vd, vu, vv, 0b00011100010, 0b000)

    def VADD_H(self, vd, vu, vv):
        """Vd.h = vadd(Vu.h, Vv.h) --halfword-element vector add."""
        self._hvx_alu_vvv(vd, vu, vv, 0b00011111101, 0b111)

    def VADD_B(self, vd, vu, vv):
        """Vd.b = vadd(Vu.b, Vv.b) --byte-element vector add."""
        self._hvx_alu_vvv(vd, vu, vv, 0b00011111101, 0b110)

    def VSUB_W(self, vd, vu, vv):
        """Vd.w = vsub(Vu.w, Vv.w) --word-element vector sub."""
        self._hvx_alu_vvv(vd, vu, vv, 0b00011100010, 0b111)

    def VSUB_H(self, vd, vu, vv):
        """Vd.h = vsub(Vu.h, Vv.h)."""
        self._hvx_alu_vvv(vd, vu, vv, 0b00011100010, 0b110)

    def VSUB_B(self, vd, vu, vv):
        """Vd.b = vsub(Vu.b, Vv.b)."""
        self._hvx_alu_vvv(vd, vu, vv, 0b00011100010, 0b101)

    def VAND(self, vd, vu, vv):
        """Vd = vand(Vu, Vv) --bitwise AND."""
        self._hvx_alu_vvv(vd, vu, vv, 0b00011100001, 0b101)

    def VOR(self, vd, vu, vv):
        """Vd = vor(Vu, Vv) --bitwise OR."""
        self._hvx_alu_vvv(vd, vu, vv, 0b00011100001, 0b110)

    def VXOR(self, vd, vu, vv):
        """Vd = vxor(Vu, Vv) --bitwise XOR."""
        self._hvx_alu_vvv(vd, vu, vv, 0b00011100001, 0b111)

    # -----------------------------------------------------------------------
    # Vector multiply: Vdd = vmpy(Vu, Vv).type  (widens to pair)
    # -----------------------------------------------------------------------

    def VMPY_H(self, vd, vu, vv):
        """Vd.h = vmpyi(Vu.h, Vv.h) --integer multiply halfwords.
        Uses Enc_45364e: Vv@[20:16], Vu@[12:8], Vd@[4:0]
        """
        self._hvx_alu_vvv(vd, vu, vv, 0b00011100001, 0b100)

    def VMPYE_W(self, vd, vu, vv):
        """Vd.w = vmpye(Vu.w, Vv.uh) --even-element multiply words."""
        self._hvx_alu_vvv(vd, vu, vv, 0b00011111111, 0b101)

    # -----------------------------------------------------------------------
    # Vector float operations (qf32)
    # -----------------------------------------------------------------------

    def VFADD_SF(self, vd, vu, vv):
        """Vd.qf32 = vadd(Vu.sf, Vv.sf) --single-precision vector add (QFloat).
        Enc_45364e: Vv@[20:16], Vu@[12:8], Vd@[4:0]
        Inst{13}=1, Inst{31:21}=0b00011111101, Inst{7:5}=0b001
        """
        bits = (0b00011111101 << 21) | (1 << 13) | (0b001 << 5)
        self.write32(bits | PARSE_BITS |
                     (int(vv) << 16) | (int(vu) << 8) | int(vd))

    def VFSUB_SF(self, vd, vu, vv):
        """Vd.qf32 = vsub(Vu.sf, Vv.sf) --single-precision vector sub (QFloat).
        Inst{13}=1, Inst{31:21}=0b00011111101, Inst{7:5}=0b100
        """
        bits = (0b00011111101 << 21) | (1 << 13) | (0b100 << 5)
        self.write32(bits | PARSE_BITS |
                     (int(vv) << 16) | (int(vu) << 8) | int(vd))

    def VFMPY_SF(self, vd, vu, vv):
        """Vd.sf = vmpy(Vu.sf, Vv.sf) --single-precision vector multiply (IEEE).
        Inst{13}=1, Inst{31:21}=0b00011111100, Inst{7:5}=0b001
        """
        bits = (0b00011111100 << 21) | (1 << 13) | (0b001 << 5)
        self.write32(bits | PARSE_BITS |
                     (int(vv) << 16) | (int(vu) << 8) | int(vd))

    # -----------------------------------------------------------------------
    # Vector load/store
    # -----------------------------------------------------------------------

    def VMEM_LOAD(self, vd, rt, offset):
        """Vd = vmem(Rt + #offset) --aligned vector load (128 bytes).
        V6_vL32b_ai: Inst{31:21}=0b00101000000, Inst{12:11}=0b00,
        Inst{7:5}=0b000, Rt@[20:16], Ii{3}@[13], Ii{2:0}@[10:8], Vd@[4:0]
        """
        off = (offset >> 7) & 0xF  # 4-bit offset in units of 128 bytes
        off_hi = (off >> 3) & 0x1   # bit 3 at [13]
        off_lo = off & 0x7          # bits 2:0 at [10:8]
        bits = (0b00101000000 << 21)
        self.write32(bits | PARSE_BITS |
                     (int(rt) << 16) | (off_hi << 13) |
                     (off_lo << 8) | int(vd))

    def VMEM_STORE(self, rt, vs, offset):
        """vmem(Rt + #offset) = Vs --aligned vector store.
        V6_vS32b_ai: Inst{31:21}=0b00101000001, Inst{12:11}=0b00,
        Inst{7:5}=0b000, Rt@[20:16], Ii{3}@[13], Ii{2:0}@[10:8], Vs@[4:0]
        """
        off = (offset >> 7) & 0xF  # 4-bit offset in units of 128 bytes
        off_hi = (off >> 3) & 0x1   # bit 3 at [13]
        off_lo = off & 0x7          # bits 2:0 at [10:8]
        bits = (0b00101000001 << 21)
        self.write32(bits | PARSE_BITS |
                     (int(rt) << 16) | (off_hi << 13) |
                     (off_lo << 8) | int(vs))

    # -----------------------------------------------------------------------
    # Vector broadcast (splat)
    # -----------------------------------------------------------------------

    def VSPLAT(self, vd, rt):
        """Vd = vsplat(Rt) --broadcast scalar to all vector elements (word).
        Encoding (Enc_a5ed8a): opcode@[31:21]=0b00011001101,
        Rt@[20:16], fixed@[13:5]=0b000000001, Vd@[4:0]
        """
        bits = (0b00011001101 << 21) | (0b000000001 << 5)
        self.write32(bits | PARSE_BITS | (int(rt) << 16) | int(vd))

    # -----------------------------------------------------------------------
    # Vector compare
    # -----------------------------------------------------------------------

    def VCMP_EQ_W(self, qd, vu, vv):
        """Qd = vcmp.eq(Vu.w, Vv.w) --word-element equality compare.
        Enc_95441f: Vv@[20:16], Vu@[12:8], Qd@[1:0]
        Inst{31:21}=0b00011111100, Inst{7:2}=0b000010
        """
        bits = (0b00011111100 << 21) | (0b000010 << 2)
        self.write32(bits | PARSE_BITS |
                     (int(vv) << 16) | (int(vu) << 8) | int(qd))

    def VCMP_GT_W(self, qd, vu, vv):
        """Qd = vcmp.gt(Vu.w, Vv.w) --word-element greater-than compare.
        Enc_95441f: Vv@[20:16], Vu@[12:8], Qd@[1:0]
        Inst{31:21}=0b00011111100, Inst{7:2}=0b000110
        """
        bits = (0b00011111100 << 21) | (0b000110 << 2)
        self.write32(bits | PARSE_BITS |
                     (int(vv) << 16) | (int(vu) << 8) | int(qd))

    # -----------------------------------------------------------------------
    # Vector pack/unpack
    # -----------------------------------------------------------------------

    def VPACKE_H(self, vd, vu, vv):
        """Vd.h = vpacke(Vu.w, Vv.w) --pack even halfwords.
        Enc_45364e: Vv@[20:16], Vu@[12:8], Vd@[4:0]
        Inst{31:21}=0b00011111110, Inst{7:5}=0b011
        """
        self._hvx_alu_vvv(vd, vu, vv, 0b00011111110, 0b011)

    def VPACKO_H(self, vd, vu, vv):
        """Vd.h = vpacko(Vu.w, Vv.w) --pack odd halfwords.
        Enc_45364e: Inst{31:21}=0b00011111111, Inst{7:5}=0b010
        """
        self._hvx_alu_vvv(vd, vu, vv, 0b00011111111, 0b010)

    def VUNPACK_H(self, vdd, vu):
        """Vdd.w = vunpack(Vu.h) --unpack halfwords to words.
        Enc_dd766a: Vu@[12:8], Vdd@[4:0]
        Inst{31:16}=0b0001111000000001, Inst{7:5}=0b011
        """
        bits = (0b0001111000000001 << 16) | (0b011 << 5)
        self.write32(bits | PARSE_BITS | (int(vu) << 8) | int(vdd))

    # -----------------------------------------------------------------------
    # Vector predicate operations
    # -----------------------------------------------------------------------

    def PRED_NOT(self, qd, qs):
        """Qd = not(Qs) --invert vector predicate.
        Enc_bfbf03: Qs@[9:8], Qd@[1:0]
        Inst{31:16}=0b0001111000000011, Inst{7:2}=0b000010
        """
        bits = (0b0001111000000011 << 16) | (0b000010 << 2)
        self.write32(bits | PARSE_BITS | (int(qs) << 8) | int(qd))

    def VANDQRT(self, vd, qu, rt):
        """Vd = vand(Qu, Rt) --replicate predicate using scalar mask.
        Enc_7b7ba8: Rt@[20:16], Qu@[9:8], Vd@[4:0]
        Inst{31:21}=0b00011001101, Inst{7:5}=0b101
        """
        bits = (0b00011001101 << 21) | (0b101 << 5)
        self.write32(bits | PARSE_BITS |
                     (int(rt) << 16) | (int(qu) << 8) | int(vd))

    def VMUX(self, vd, qt, vu, vv):
        """Vd = vmux(Qt, Vu, Vv) --select elements by predicate.
        Enc_31db33: Qt@[6:5], Vu@[12:8], Vv@[20:16], Vd@[4:0]
        Inst{31:21}=0b00011110111, Inst{13}=1, Inst{7}=0
        """
        bits = (0b00011110111 << 21) | (1 << 13)
        self.write32(bits | PARSE_BITS |
                     (int(vv) << 16) | (int(vu) << 8) |
                     (int(qt) << 5) | int(vd))

    # -----------------------------------------------------------------------
    # Vector element insert/extract
    # -----------------------------------------------------------------------

    def VINSERTW(self, vx, rt):
        """Vx.w = vinsert(Rt) --insert scalar word at element 0.
        Enc_569cfe: Rt@[20:16], Vx@[4:0]
        Inst{31:21}=0b00011001101, Inst{13:5}=0b100000001
        """
        bits = (0b00011001101 << 21) | (0b100000001 << 5)
        self.write32(bits | PARSE_BITS | (int(rt) << 16) | int(vx))

    def VEXTRACTW(self, rd, vu, rs):
        """Rd = vextract(Vu, Rs) --extract word at byte offset Rs.
        Enc_50e578: Vu@[12:8], Rs@[20:16], Rd@[4:0]
        Inst{31:21}=0b10010010000, Inst{13}=0, Inst{7:5}=0b001
        """
        bits = (0b10010010000 << 21) | (0b001 << 5)
        self.write32(bits | PARSE_BITS |
                     (int(rs) << 16) | (int(vu) << 8) | int(rd))

    def VROR(self, vd, vu, rt):
        """Vd = vror(Vu, Rt) --rotate vector right by Rt bytes.
        Enc_b087ac: Vu@[12:8], Rt@[20:16], Vd@[4:0]
        Inst{31:21}=0b00011001011, Inst{13}=0, Inst{7:5}=0b001
        """
        bits = (0b00011001011 << 21) | (0b001 << 5)
        self.write32(bits | PARSE_BITS |
                     (int(rt) << 16) | (int(vu) << 8) | int(vd))


# ---------------------------------------------------------------------------
# Vector extension configuration
# ---------------------------------------------------------------------------

class HexagonVectorExt(VectorExt):
    """HVX vector extension configuration, extending llsupport's VectorExt."""

    should_align_unroll = True

    def setup_once(self, asm):
        """Detect and configure HVX support."""
        if detect_hvx():
            hvx_len = detect_hvx_length()
            self.enable(hvx_len, accum=True)
        self._setup = True


# ---------------------------------------------------------------------------
# Vector operation emission mixin
# ---------------------------------------------------------------------------

class VectorAssemblerMixin(object):
    """Mixin for emitting HVX vector operations.

    Mixed into AssemblerHexagon. Methods are named emit_op_vec_* to
    match the dispatch table convention (rop.VEC_INT_ADD -> emit_op_vec_int_add).

    Arglocs convention for binary ops: [arg0_loc, arg1_loc, res_loc, itemsize_imm]
    Arglocs convention for unary ops:  [arg0_loc, res_loc, itemsize_imm]
    """
    _mixin_ = True

    # --- Integer arithmetic ---

    def emit_op_vec_int_add(self, op, arglocs):
        vu, vv, vd, itemsize_loc = arglocs
        size = itemsize_loc.value
        if size == 4:
            self.mc.VADD_W(vd.value, vu.value, vv.value)
        elif size == 2:
            self.mc.VADD_H(vd.value, vu.value, vv.value)
        elif size == 1:
            self.mc.VADD_B(vd.value, vu.value, vv.value)

    def emit_op_vec_int_sub(self, op, arglocs):
        vu, vv, vd, itemsize_loc = arglocs
        size = itemsize_loc.value
        if size == 4:
            self.mc.VSUB_W(vd.value, vu.value, vv.value)
        elif size == 2:
            self.mc.VSUB_H(vd.value, vu.value, vv.value)
        elif size == 1:
            self.mc.VSUB_B(vd.value, vu.value, vv.value)

    def emit_op_vec_int_mul(self, op, arglocs):
        vu, vv, vd, itemsize_loc = arglocs
        size = itemsize_loc.value
        if size == 2:
            self.mc.VMPY_H(vd.value, vu.value, vv.value)
        elif size == 4:
            self.mc.VMPYE_W(vd.value, vu.value, vv.value)

    def emit_op_vec_int_and(self, op, arglocs):
        vu, vv, vd = arglocs[0], arglocs[1], arglocs[2]
        self.mc.VAND(vd.value, vu.value, vv.value)

    def emit_op_vec_int_or(self, op, arglocs):
        vu, vv, vd = arglocs[0], arglocs[1], arglocs[2]
        self.mc.VOR(vd.value, vu.value, vv.value)

    def emit_op_vec_int_xor(self, op, arglocs):
        vu, vv, vd = arglocs[0], arglocs[1], arglocs[2]
        self.mc.VXOR(vd.value, vu.value, vv.value)

    # --- Float arithmetic ---

    def emit_op_vec_float_add(self, op, arglocs):
        vu, vv, vd = arglocs[0], arglocs[1], arglocs[2]
        self.mc.VFADD_SF(vd.value, vu.value, vv.value)

    def emit_op_vec_float_sub(self, op, arglocs):
        vu, vv, vd = arglocs[0], arglocs[1], arglocs[2]
        self.mc.VFSUB_SF(vd.value, vu.value, vv.value)

    def emit_op_vec_float_mul(self, op, arglocs):
        vu, vv, vd = arglocs[0], arglocs[1], arglocs[2]
        self.mc.VFMPY_SF(vd.value, vu.value, vv.value)

    # --- Memory operations ---

    def emit_op_vec_load_i(self, op, arglocs):
        """VEC_LOAD_I: Vd = vmem(base + index*scale + ofs)
        arglocs: [base_loc, ofs_loc, res_loc, scale_loc]
        """
        base_loc, ofs_loc, res_loc, scale_loc = arglocs
        # Compute effective address into scratch1
        if ofs_loc.is_imm() and ofs_loc.value == 0:
            self.mc.VMEM_LOAD(res_loc.value, base_loc.value, 0)
        else:
            # base + ofs -> scratch1, then load from scratch1
            if ofs_loc.is_imm():
                self.mc.ADDI(r.scratch1.value, base_loc.value, ofs_loc.value)
            else:
                self.mc.ADD(r.scratch1.value, base_loc.value, ofs_loc.value)
            self.mc.VMEM_LOAD(res_loc.value, r.scratch1.value, 0)

    emit_op_vec_load_f = emit_op_vec_load_i

    def emit_op_vec_store(self, op, arglocs):
        """VEC_STORE: vmem(base + index*scale + ofs) = Vs
        arglocs: [base_loc, ofs_loc, value_loc, scale_loc]
        """
        base_loc, ofs_loc, value_loc, scale_loc = arglocs
        if ofs_loc.is_imm() and ofs_loc.value == 0:
            self.mc.VMEM_STORE(base_loc.value, value_loc.value, 0)
        else:
            if ofs_loc.is_imm():
                self.mc.ADDI(r.scratch1.value, base_loc.value, ofs_loc.value)
            else:
                self.mc.ADD(r.scratch1.value, base_loc.value, ofs_loc.value)
            self.mc.VMEM_STORE(r.scratch1.value, value_loc.value, 0)

    # --- Broadcast (expand) ---

    def emit_op_vec_expand_i(self, op, arglocs):
        """VEC_EXPAND_I: Vd = vsplat(Rt) -- broadcast scalar to all lanes."""
        src_loc, res_loc = arglocs[0], arglocs[1]
        if src_loc.is_core_reg():
            self.mc.VSPLAT(res_loc.value, src_loc.value)
        else:
            # Need to load into a GPR first
            if src_loc.is_imm():
                self.mc.gen_load_int(r.scratch1.value, src_loc.value)
            elif src_loc.is_stack():
                self.mc.load_from_jitframe(r.scratch1.value, src_loc.value)
            self.mc.VSPLAT(res_loc.value, r.scratch1.value)

    emit_op_vec_expand_f = emit_op_vec_expand_i

    # --- Comparisons ---

    def emit_op_vec_int_eq(self, op, arglocs):
        """VEC_INT_EQ: element-wise equality -> mask vector.

        Produces a vector where each word is 0xFFFFFFFF (equal) or
        0x00000000 (not equal), matching SSE PCMPEQ semantics.

        Sequence:
          Q0 = vcmp.eq(Vu.w, Vv.w)   // predicate comparison
          scratch1 = #-1              // all-ones mask
          Vd = vand(Q0, scratch1)     // predicate -> vector mask
        """
        vu, vv, vd, itemsize_loc = arglocs
        size = itemsize_loc.value
        if size == 4:
            self.mc.VCMP_EQ_W(r.q0.value, vu.value, vv.value)
        else:
            # TODO: add VCMP_EQ_H/VCMP_EQ_B when needed
            raise NotImplementedError("vec_int_eq: size %d not yet supported" % size)
        self.mc.gen_load_int(r.scratch1.value, -1)
        self.mc.VANDQRT(vd.value, r.q0.value, r.scratch1.value)

    def emit_op_vec_int_ne(self, op, arglocs):
        """VEC_INT_NE: element-wise not-equal -> mask vector.

        Sequence:
          Q0 = vcmp.eq(Vu.w, Vv.w)   // predicate comparison (equal)
          Q1 = not(Q0)                // invert -> not-equal
          scratch1 = #-1
          Vd = vand(Q1, scratch1)     // predicate -> vector mask
        """
        vu, vv, vd, itemsize_loc = arglocs
        size = itemsize_loc.value
        if size == 4:
            self.mc.VCMP_EQ_W(r.q0.value, vu.value, vv.value)
        else:
            raise NotImplementedError("vec_int_ne: size %d not yet supported" % size)
        self.mc.PRED_NOT(r.q1.value, r.q0.value)
        self.mc.gen_load_int(r.scratch1.value, -1)
        self.mc.VANDQRT(vd.value, r.q1.value, r.scratch1.value)

    # --- Pack/Unpack ---

    def emit_op_vec_pack_i(self, op, arglocs):
        """VEC_PACK_I: Insert element(s) into a vector.

        arglocs: [resultloc, sourceloc, residx_imm, srcidx_imm,
                  count_imm, size_imm]

        For word-size (4) with count=1:
          If residx == 0, use VINSERTW directly.
          Otherwise, use VROR to rotate target position to element 0,
          VINSERTW, then VROR back.

        When source is a vector, extract the element first using VEXTRACTW.
        """
        resultloc, sourceloc, residxloc, srcidxloc, countloc, sizeloc = arglocs
        size = sizeloc.value
        residx = residxloc.value
        srcidx = srcidxloc.value
        count = countloc.value
        si = srcidx
        ri = residx
        k = count
        while k > 0:
            if size == 4:
                # Get the scalar value into scratch1
                if sourceloc.is_vector_reg():
                    byte_off = si * 4
                    self.mc.gen_load_int(r.scratch1.value, byte_off)
                    self.mc.VEXTRACTW(r.scratch2.value,
                                      sourceloc.value, r.scratch1.value)
                    src_gpr = r.scratch2.value
                else:
                    # Source is a scalar GPR
                    src_gpr = sourceloc.value
                # Insert into result vector at position ri
                if ri == 0:
                    self.mc.VINSERTW(resultloc.value, src_gpr)
                else:
                    # Rotate so element ri is at position 0
                    byte_off = ri * 4
                    self.mc.gen_load_int(r.scratch1.value, byte_off)
                    self.mc.VROR(resultloc.value, resultloc.value,
                                r.scratch1.value)
                    self.mc.VINSERTW(resultloc.value, src_gpr)
                    # Rotate back
                    rot_back = HVX_VEC_SIZE - byte_off
                    self.mc.gen_load_int(r.scratch1.value, rot_back)
                    self.mc.VROR(resultloc.value, resultloc.value,
                                r.scratch1.value)
            else:
                raise NotImplementedError(
                    "vec_pack_i: element size %d not yet supported" % size)
            si += 1
            ri += 1
            k -= 1

    emit_op_vec_pack_f = emit_op_vec_pack_i

    def emit_op_vec_unpack_i(self, op, arglocs):
        """VEC_UNPACK_I: Extract element(s) from a vector.

        arglocs: [resultloc, sourceloc, residx_imm, srcidx_imm,
                  count_imm, size_imm]

        For count=1: extract a single scalar from the source vector.
        Uses VEXTRACTW for word-size elements.
        """
        resultloc, sourceloc, residxloc, srcidxloc, countloc, sizeloc = arglocs
        size = sizeloc.value
        srcidx = srcidxloc.value
        residx = residxloc.value
        count = countloc.value
        if resultloc.is_vector_reg():
            # Vector-to-vector: extract a sub-range into another vector
            # For count elements starting at srcidx, pack into result
            # starting at residx
            si = srcidx
            ri = residx
            k = count
            while k > 0:
                if size == 4:
                    byte_off = si * 4
                    self.mc.gen_load_int(r.scratch1.value, byte_off)
                    self.mc.VEXTRACTW(r.scratch2.value,
                                      sourceloc.value, r.scratch1.value)
                    if ri == 0:
                        self.mc.VINSERTW(resultloc.value, r.scratch2.value)
                    else:
                        byte_off = ri * 4
                        self.mc.gen_load_int(r.scratch1.value, byte_off)
                        self.mc.VROR(resultloc.value, resultloc.value,
                                    r.scratch1.value)
                        self.mc.VINSERTW(resultloc.value, r.scratch2.value)
                        rot_back = HVX_VEC_SIZE - byte_off
                        self.mc.gen_load_int(r.scratch1.value, rot_back)
                        self.mc.VROR(resultloc.value, resultloc.value,
                                    r.scratch1.value)
                else:
                    raise NotImplementedError(
                        "vec_unpack_i: element size %d not yet supported" %
                        size)
                si += 1
                ri += 1
                k -= 1
        else:
            # Vector-to-scalar: extract a single element
            if size == 4:
                byte_off = srcidx * 4
                self.mc.gen_load_int(r.scratch1.value, byte_off)
                self.mc.VEXTRACTW(resultloc.value,
                                  sourceloc.value, r.scratch1.value)
            else:
                raise NotImplementedError(
                    "vec_unpack_i: element size %d not yet supported" % size)

    emit_op_vec_unpack_f = emit_op_vec_unpack_i


# ---------------------------------------------------------------------------
# Vector register allocation mixin
# ---------------------------------------------------------------------------

class VectorRegallocMixin(object):
    """Mixin for HVX vector register allocation.

    Mixed into the Regalloc class. Methods are named prepare_op_vec_*
    to match the dispatch convention.
    """
    _mixin_ = True

    def _prepare_vec_binary(self, op):
        """Common prepare for binary vector ops (add, sub, mul, etc.).
        Returns [arg0_loc, arg1_loc, res_loc, itemsize_imm].
        """
        from rpython.jit.metainterp.optimizeopt.schedule import forwarded_vecinfo
        a0 = op.getarg(0)
        a1 = op.getarg(1)
        l0 = self.ensure_in_hvx_reg(a0)
        l1 = self.ensure_in_hvx_reg(a1)
        self.possibly_free_vars_for_op(op)
        res = self.hvxrm.force_allocate_reg(op)
        vecinfo = forwarded_vecinfo(op)
        return [l0, l1, res, ImmLocation(vecinfo.bytesize)]

    def _prepare_vec_binary_logic(self, op):
        """Prepare for bitwise vector ops (and, or, xor) - no itemsize needed."""
        a0 = op.getarg(0)
        a1 = op.getarg(1)
        l0 = self.ensure_in_hvx_reg(a0)
        l1 = self.ensure_in_hvx_reg(a1)
        self.possibly_free_vars_for_op(op)
        res = self.hvxrm.force_allocate_reg(op)
        return [l0, l1, res]

    def prepare_op_vec_int_add(self, op):
        return self._prepare_vec_binary(op)

    def prepare_op_vec_int_sub(self, op):
        return self._prepare_vec_binary(op)

    def prepare_op_vec_int_mul(self, op):
        return self._prepare_vec_binary(op)

    def prepare_op_vec_int_and(self, op):
        return self._prepare_vec_binary_logic(op)

    def prepare_op_vec_int_or(self, op):
        return self._prepare_vec_binary_logic(op)

    def prepare_op_vec_int_xor(self, op):
        return self._prepare_vec_binary_logic(op)

    def prepare_op_vec_float_add(self, op):
        return self._prepare_vec_binary_logic(op)

    def prepare_op_vec_float_sub(self, op):
        return self._prepare_vec_binary_logic(op)

    def prepare_op_vec_float_mul(self, op):
        return self._prepare_vec_binary_logic(op)

    def prepare_op_vec_load_i(self, op):
        """VEC_LOAD_I: base, index, scale, ofs (from descr).
        Returns [base_loc, ofs_loc, res_loc, scale_loc].
        """
        base = op.getarg(0)
        index = op.getarg(1)
        base_loc = self.rm.make_sure_var_in_reg(base, op.getarglist())
        # The vectorizer produces ops with args: base, index, scale, ofs
        scale = op.getarg(2).getint()
        ofs = op.getarg(3).getint()
        if index.is_constant():
            ofs_val = index.getint() * scale + ofs
            ofs_loc = ImmLocation(ofs_val)
        else:
            # Need to compute index * scale + ofs at runtime
            index_loc = self.rm.make_sure_var_in_reg(index, op.getarglist())
            if scale != 1:
                self.assembler.mc.gen_load_int(r.scratch2.value, scale)
                self.assembler.mc.MPYI(r.scratch2.value,
                                       index_loc.value, r.scratch2.value)
                index_loc = r.scratch2
            if ofs != 0:
                self.assembler.mc.ADDI(r.scratch2.value,
                                       index_loc.value, ofs)
                index_loc = r.scratch2
            ofs_loc = index_loc
        self.possibly_free_vars_for_op(op)
        res = self.hvxrm.force_allocate_reg(op)
        return [base_loc, ofs_loc, res, ImmLocation(scale)]

    prepare_op_vec_load_f = prepare_op_vec_load_i

    def prepare_op_vec_store(self, op):
        """VEC_STORE: base, index, value, scale, ofs.
        Returns [base_loc, ofs_loc, value_loc, scale_loc].
        """
        base = op.getarg(0)
        index = op.getarg(1)
        value = op.getarg(2)
        base_loc = self.rm.make_sure_var_in_reg(base, op.getarglist())
        value_loc = self.ensure_in_hvx_reg(value)
        scale = op.getarg(3).getint()
        ofs = op.getarg(4).getint()
        if index.is_constant():
            ofs_val = index.getint() * scale + ofs
            ofs_loc = ImmLocation(ofs_val)
        else:
            index_loc = self.rm.make_sure_var_in_reg(index, op.getarglist())
            if scale != 1:
                self.assembler.mc.gen_load_int(r.scratch2.value, scale)
                self.assembler.mc.MPYI(r.scratch2.value,
                                       index_loc.value, r.scratch2.value)
                index_loc = r.scratch2
            if ofs != 0:
                self.assembler.mc.ADDI(r.scratch2.value,
                                       index_loc.value, ofs)
                index_loc = r.scratch2
            ofs_loc = index_loc
        return [base_loc, ofs_loc, value_loc, ImmLocation(scale)]

    def prepare_op_vec_expand_i(self, op):
        """VEC_EXPAND_I: broadcast scalar to vector.
        Returns [src_loc, res_loc].
        """
        a0 = op.getarg(0)
        src_loc = self.loc(a0)
        self.possibly_free_vars_for_op(op)
        res = self.hvxrm.force_allocate_reg(op)
        return [src_loc, res]

    prepare_op_vec_expand_f = prepare_op_vec_expand_i

    def prepare_op_vec_int_eq(self, op):
        return self._prepare_vec_binary(op)

    def prepare_op_vec_int_ne(self, op):
        return self._prepare_vec_binary(op)

    def prepare_op_vec_pack_i(self, op):
        """VEC_PACK_I: vec, source, index, count
        Returns [resloc, srcloc, residx_imm, srcidx_imm, count_imm, size_imm].
        """
        from rpython.jit.metainterp.history import ConstInt
        from rpython.jit.metainterp.optimizeopt.schedule import forwarded_vecinfo
        from rpython.jit.metainterp.resoperation import VectorOp
        args = op.getarglist()
        arg0 = op.getarg(0)  # existing vector
        arg1 = op.getarg(1)  # source (scalar or vector)
        index = op.getarg(2)
        count = op.getarg(3)
        assert isinstance(index, ConstInt)
        assert isinstance(count, ConstInt)
        # Result must be initialized from arg0 (force into same reg)
        resloc = self.hvxrm.force_result_in_reg(op, arg0, args)
        # Source: scalar in GPR or vector in HVX reg
        vecinfo1 = forwarded_vecinfo(arg1)
        if isinstance(arg1, VectorOp) or vecinfo1.count > 1:
            srcloc = self.ensure_in_hvx_reg(arg1)
            srcidx = 0
        else:
            srcloc = self.rm.make_sure_var_in_reg(arg1, args)
            srcidx = 0
        vecinfo = forwarded_vecinfo(op)
        return [resloc, srcloc, ImmLocation(index.value),
                ImmLocation(srcidx), ImmLocation(count.value),
                ImmLocation(vecinfo.bytesize)]

    prepare_op_vec_pack_f = prepare_op_vec_pack_i

    def prepare_op_vec_unpack_i(self, op):
        """VEC_UNPACK_I: vec, index, count
        Returns [resloc, srcloc, residx_imm, srcidx_imm, count_imm, size_imm].
        """
        from rpython.jit.metainterp.history import ConstInt
        from rpython.jit.metainterp.optimizeopt.schedule import forwarded_vecinfo
        from rpython.jit.metainterp.resoperation import VectorOp
        args = op.getarglist()
        arg0 = op.getarg(0)  # source vector
        index = op.getarg(1)
        count = op.getarg(2)
        assert isinstance(index, ConstInt)
        assert isinstance(count, ConstInt)
        srcloc = self.ensure_in_hvx_reg(arg0)
        if isinstance(op, VectorOp) and op.is_vector():
            # Result is a vector: force into same reg as source
            resloc = self.hvxrm.force_result_in_reg(op, arg0, args)
            vecinfo = forwarded_vecinfo(op)
            size = vecinfo.bytesize
        else:
            # Result is a scalar GPR
            self.possibly_free_vars_for_op(op)
            resloc = self.rm.force_allocate_reg(op)
            vecinfo0 = forwarded_vecinfo(arg0)
            size = vecinfo0.bytesize
        residx = 0
        return [resloc, srcloc, ImmLocation(residx),
                ImmLocation(index.value), ImmLocation(count.value),
                ImmLocation(size)]

    prepare_op_vec_unpack_f = prepare_op_vec_unpack_i

    # --- Helper ---

    def ensure_in_hvx_reg(self, var):
        """Ensure var is in an HVX vector register, spilling if needed."""
        loc = self.hvxrm.loc(var)
        if loc is not None and loc.is_vector_reg():
            return loc
        # Need to allocate a vector register
        loc = self.hvxrm.force_allocate_reg(var)
        return loc
