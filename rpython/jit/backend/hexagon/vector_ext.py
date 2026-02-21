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
- VectorAssemblerMixin: emit methods for vector operations
- HVX instruction encoding
"""

from rpython.jit.backend.hexagon import registers as r
from rpython.jit.backend.hexagon.arch import INST_SIZE, PARSE_END_PACKET, PARSE_BITS_SHIFT
from rpython.jit.backend.hexagon.detect import detect_hvx, detect_hvx_length

PARSE_BITS = PARSE_END_PACKET << PARSE_BITS_SHIFT


# ---------------------------------------------------------------------------
# HVX instruction encoding helpers
# ---------------------------------------------------------------------------

class HVXInstructionMixin(object):
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


# ---------------------------------------------------------------------------
# Vector extension configuration
# ---------------------------------------------------------------------------

class HexagonVectorExt(object):
    """HVX vector extension configuration and state."""

    def __init__(self):
        self._setup = False
        self.vec_size = 0    # vector length in bytes (64 or 128)
        self.enabled = False
        self.accum = False

    def setup_once(self, asm):
        """Detect and configure HVX support."""
        if detect_hvx():
            hvx_len = detect_hvx_length()
            self.vec_size = hvx_len
            self.enabled = True
            self.accum = True
        self._setup = True

    def is_enabled(self):
        return self.enabled

    def get_vec_size(self):
        """Return the HVX vector register size in bytes."""
        return self.vec_size


# ---------------------------------------------------------------------------
# Vector register allocator support
# ---------------------------------------------------------------------------

class HVXRegisterManager(object):
    """Manages HVX vector register allocation: V0-V15."""
    all_regs = r.allocatable_hvx_regs
    FORBID_TEMP_BOXES = True

    def __init__(self):
        self.free_regs = list(self.all_regs)
        self.allocated = {}

    def allocate_reg(self, var):
        if self.free_regs:
            reg = self.free_regs.pop(0)
            self.allocated[var] = reg
            return reg
        return None

    def free_reg(self, var):
        if var in self.allocated:
            reg = self.allocated.pop(var)
            self.free_regs.append(reg)

    def get_reg(self, var):
        return self.allocated.get(var, None)


# ---------------------------------------------------------------------------
# Vector operation emission mixin
# ---------------------------------------------------------------------------

class VectorAssemblerMixin(object):
    """Mixin for emitting HVX vector operations.

    Added to the main assembler class when HVX is enabled.
    """

    def emit_vec_int_add(self, op, arglocs):
        """VEC_INT_ADD: Vd = vadd(Vu, Vv).w"""
        vu, vv, vd = arglocs
        size = op.getdescr().get_item_size()
        if size == 4:
            self.mc.VADD_W(vd.value, vu.value, vv.value)
        elif size == 2:
            self.mc.VADD_H(vd.value, vu.value, vv.value)
        elif size == 1:
            self.mc.VADD_B(vd.value, vu.value, vv.value)

    def emit_vec_int_sub(self, op, arglocs):
        vu, vv, vd = arglocs
        size = op.getdescr().get_item_size()
        if size == 4:
            self.mc.VSUB_W(vd.value, vu.value, vv.value)
        elif size == 2:
            self.mc.VSUB_H(vd.value, vu.value, vv.value)
        elif size == 1:
            self.mc.VSUB_B(vd.value, vu.value, vv.value)

    def emit_vec_int_and(self, op, arglocs):
        vu, vv, vd = arglocs
        self.mc.VAND(vd.value, vu.value, vv.value)

    def emit_vec_int_or(self, op, arglocs):
        vu, vv, vd = arglocs
        self.mc.VOR(vd.value, vu.value, vv.value)

    def emit_vec_int_xor(self, op, arglocs):
        vu, vv, vd = arglocs
        self.mc.VXOR(vd.value, vu.value, vv.value)

    def emit_vec_float_add(self, op, arglocs):
        """VEC_FLOAT_ADD: Vd.sf = vadd(Vu.sf, Vv.sf)"""
        vu, vv, vd = arglocs
        self.mc.VFADD_SF(vd.value, vu.value, vv.value)

    def emit_vec_float_sub(self, op, arglocs):
        vu, vv, vd = arglocs
        self.mc.VFSUB_SF(vd.value, vu.value, vv.value)

    def emit_vec_float_mul(self, op, arglocs):
        vu, vv, vd = arglocs
        self.mc.VFMPY_SF(vd.value, vu.value, vv.value)

    def emit_vec_load(self, op, arglocs):
        """VEC_LOAD: Vd = vmem(Rt + #offset)"""
        rt, vd = arglocs[:2]
        offset = arglocs[2].value if len(arglocs) > 2 else 0
        self.mc.VMEM_LOAD(vd.value, rt.value, offset)

    def emit_vec_store(self, op, arglocs):
        """VEC_STORE: vmem(Rt + #offset) = Vs"""
        rt, vs = arglocs[:2]
        offset = arglocs[2].value if len(arglocs) > 2 else 0
        self.mc.VMEM_STORE(rt.value, vs.value, offset)

    def emit_vec_expand(self, op, arglocs):
        """VEC_EXPAND: Vd = vsplat(Rt) --broadcast scalar."""
        rt, vd = arglocs
        self.mc.VSPLAT(vd.value, rt.value)
