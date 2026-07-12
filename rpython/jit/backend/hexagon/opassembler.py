"""
Operation-level assembly for the Hexagon JIT backend.

Each emit_op_* method takes (op, arglocs) and emits machine instructions
to the code builder (self.mc).
"""

from rpython.jit.backend.hexagon import registers as r
from rpython.jit.backend.hexagon.arch import WORD, INST_SIZE
from rpython.jit.backend.hexagon.codebuilder import OverwritingBuilder
from rpython.jit.backend.llsupport.assembler import BaseAssembler, GuardToken
from rpython.jit.backend.llsupport.descr import (
    FieldDescr, SizeDescr, CallDescr, unpack_fielddescr,
)
from rpython.jit.metainterp.history import FLOAT
from rpython.jit.metainterp.resoperation import rop
from rpython.rlib.objectmodel import we_are_translated
from rpython.rtyper.lltypesystem import lltype, rffi


class ResOpAssembler(BaseAssembler):
    """Emit machine instructions for each IR operation."""

    # -------------------------------------------------------------------
    # Integer arithmetic
    # -------------------------------------------------------------------

    def emit_op_int_add(self, op, arglocs):
        l0, l1, res = arglocs
        if l1.is_imm():
            self.mc.ADDI(res.value, l0.value, l1.value)
        else:
            self.mc.ADD(res.value, l0.value, l1.value)

    emit_op_nursery_ptr_increment = emit_op_int_add

    def emit_op_int_sub(self, op, arglocs):
        l0, l1, res = arglocs
        if l1.is_imm():
            neg = -l1.value
            if -32768 <= neg <= 32767:
                self.mc.ADDI(res.value, l0.value, neg)
            else:
                # Negated value doesn't fit in s16 (e.g. l1=-32768 -> neg=32768)
                self.mc.gen_load_int(r.scratch1.value, l1.value)
                self.mc.SUB(res.value, l0.value, r.scratch1.value)
        else:
            self.mc.SUB(res.value, l0.value, l1.value)

    def emit_op_int_mul(self, op, arglocs):
        l0, l1, res = arglocs
        self.mc.MPYI(res.value, l0.value, l1.value)

    def emit_op_int_and(self, op, arglocs):
        l0, l1, res = arglocs
        if l1.is_imm():
            self.mc.ANDI(res.value, l0.value, l1.value)
        else:
            self.mc.AND(res.value, l0.value, l1.value)

    def emit_op_int_or(self, op, arglocs):
        l0, l1, res = arglocs
        if l1.is_imm():
            self.mc.ORI(res.value, l0.value, l1.value)
        else:
            self.mc.OR(res.value, l0.value, l1.value)

    def emit_op_int_xor(self, op, arglocs):
        l0, l1, res = arglocs
        if l1.is_imm():
            # No XORI instruction; use scratch register
            self.mc.gen_load_int(r.scratch1.value, l1.value)
            self.mc.XOR(res.value, l0.value, r.scratch1.value)
        else:
            self.mc.XOR(res.value, l0.value, l1.value)

    def emit_op_int_neg(self, op, arglocs):
        l0, res = arglocs
        self.mc.NEG(res.value, l0.value)

    def emit_op_int_invert(self, op, arglocs):
        l0, res = arglocs
        self.mc.NOT(res.value, l0.value)

    # -------------------------------------------------------------------
    # Shifts
    # -------------------------------------------------------------------

    def emit_op_int_lshift(self, op, arglocs):
        l0, l1, res = arglocs
        if l1.is_imm():
            self.mc.S2_ASL_I_R(res.value, l0.value, l1.value)
        else:
            self.mc.S2_ASL_R_R(res.value, l0.value, l1.value)

    def emit_op_int_rshift(self, op, arglocs):
        l0, l1, res = arglocs
        if l1.is_imm():
            self.mc.S2_ASR_I_R(res.value, l0.value, l1.value)
        else:
            self.mc.S2_ASR_R_R(res.value, l0.value, l1.value)

    def emit_op_uint_rshift(self, op, arglocs):
        l0, l1, res = arglocs
        if l1.is_imm():
            self.mc.S2_LSR_I_R(res.value, l0.value, l1.value)
        else:
            self.mc.S2_LSR_R_R(res.value, l0.value, l1.value)

    # -------------------------------------------------------------------
    # Integer comparisons
    # -------------------------------------------------------------------

    def _emit_int_cmp(self, l0, l1, res, cmp_mnemonic, invert=False):
        """Emit: Pd = cmp.xx(l0, l1); res = mux(Pd, #1, #0)."""
        pd = r.scratch_pred.value
        # If the first operand is an immediate, load it into a scratch register
        if l0.is_imm():
            self.mc.gen_load_int(r.scratch1.value, l0.value)
            l0_val = r.scratch1.value
        else:
            l0_val = l0.value
        if l1.is_imm():
            # Use immediate comparison instructions
            if cmp_mnemonic == 'eq':
                self.mc.CMP_EQI(pd, l0_val, l1.value)
            elif cmp_mnemonic == 'gt':
                self.mc.CMP_GTI(pd, l0_val, l1.value)
            elif cmp_mnemonic == 'gtu':
                self.mc.CMP_GTUI(pd, l0_val, l1.value)
        else:
            if cmp_mnemonic == 'eq':
                self.mc.CMP_EQ(pd, l0_val, l1.value)
            elif cmp_mnemonic == 'gt':
                self.mc.CMP_GT(pd, l0_val, l1.value)
            elif cmp_mnemonic == 'gtu':
                self.mc.CMP_GTU(pd, l0_val, l1.value)
        # Convert predicate to integer: Rd = mux(Pd, #1, #0)
        self.mc.gen_load_int(r.scratch1.value, 1)
        self.mc.gen_load_int(r.scratch2.value, 0)
        if invert:
            self.mc.MUX(res.value, pd, r.scratch2.value, r.scratch1.value)
        else:
            self.mc.MUX(res.value, pd, r.scratch1.value, r.scratch2.value)

    def emit_op_int_lt(self, op, arglocs):
        l0, l1, res = arglocs
        # l0 < l1 <==> l1 > l0
        self._emit_int_cmp(l1, l0, res, 'gt')

    def emit_op_int_le(self, op, arglocs):
        l0, l1, res = arglocs
        # l0 <= l1 <==> !(l0 > l1)
        self._emit_int_cmp(l0, l1, res, 'gt', invert=True)

    def emit_op_int_eq(self, op, arglocs):
        l0, l1, res = arglocs
        self._emit_int_cmp(l0, l1, res, 'eq')

    def emit_op_int_ne(self, op, arglocs):
        l0, l1, res = arglocs
        self._emit_int_cmp(l0, l1, res, 'eq', invert=True)

    def emit_op_int_gt(self, op, arglocs):
        l0, l1, res = arglocs
        self._emit_int_cmp(l0, l1, res, 'gt')

    def emit_op_int_ge(self, op, arglocs):
        l0, l1, res = arglocs
        # l0 >= l1 <==> !(l1 > l0)
        self._emit_int_cmp(l1, l0, res, 'gt', invert=True)

    def emit_op_uint_lt(self, op, arglocs):
        l0, l1, res = arglocs
        self._emit_int_cmp(l1, l0, res, 'gtu')

    def emit_op_uint_le(self, op, arglocs):
        l0, l1, res = arglocs
        self._emit_int_cmp(l0, l1, res, 'gtu', invert=True)

    def emit_op_uint_gt(self, op, arglocs):
        l0, l1, res = arglocs
        self._emit_int_cmp(l0, l1, res, 'gtu')

    def emit_op_uint_ge(self, op, arglocs):
        l0, l1, res = arglocs
        self._emit_int_cmp(l1, l0, res, 'gtu', invert=True)

    emit_op_ptr_eq = emit_op_int_eq
    emit_op_ptr_ne = emit_op_int_ne
    emit_op_instance_ptr_eq = emit_op_int_eq
    emit_op_instance_ptr_ne = emit_op_int_ne

    # -------------------------------------------------------------------
    # Boolean operations
    # -------------------------------------------------------------------

    def emit_op_int_is_true(self, op, arglocs):
        l0, res = arglocs
        pd = r.scratch_pred.value
        self.mc.CMP_EQI(pd, l0.value, 0)  # Pd = (l0 == 0)
        self.mc.gen_load_int(r.scratch1.value, 0)
        self.mc.gen_load_int(r.scratch2.value, 1)
        self.mc.MUX(res.value, pd, r.scratch1.value, r.scratch2.value)

    def emit_op_int_is_zero(self, op, arglocs):
        l0, res = arglocs
        pd = r.scratch_pred.value
        self.mc.CMP_EQI(pd, l0.value, 0)
        self.mc.gen_load_int(r.scratch1.value, 1)
        self.mc.gen_load_int(r.scratch2.value, 0)
        self.mc.MUX(res.value, pd, r.scratch1.value, r.scratch2.value)

    def emit_op_int_signext(self, op, arglocs):
        l0, l1, res = arglocs
        assert l1.is_imm()
        num_bytes = l1.value
        if num_bytes == 1:
            self.mc.SXTB(res.value, l0.value)
        elif num_bytes == 2:
            self.mc.SXTH(res.value, l0.value)
        elif num_bytes == 4:
            # 32-bit on 32-bit arch: no-op, just move
            if l0.value != res.value:
                self.mc.MV(res.value, l0.value)
        else:
            raise AssertionError("unexpected int_signext bytes: %d" % num_bytes)

    def emit_op_uint_mul_high(self, op, arglocs):
        l0, l1, res = arglocs
        # Unsigned 32x32->64 multiply into the scratch pair, take the
        # high word.  res is written last, so it may alias l0/l1.
        d14_even = r.d14.value
        d14_odd = d14_even + 1
        self.mc.M2_DPMPYUU_S0(d14_even, l0.value, l1.value)
        self.mc.MV(res.value, d14_odd)

    def emit_op_int_force_ge_zero(self, op, arglocs):
        l0, res = arglocs
        pd = r.scratch_pred.value
        self.mc.CMP_GTI(pd, l0.value, -1)  # Pd = (l0 > -1), i.e. l0 >= 0
        self.mc.gen_load_int(r.scratch1.value, 0)
        self.mc.MUX(res.value, pd, l0.value, r.scratch1.value)

    # -------------------------------------------------------------------
    # Float operations (using register pairs for doubles)
    # -------------------------------------------------------------------

    def emit_op_float_add(self, op, arglocs):
        l0, l1, res = arglocs
        self.mc.DFADD(res.value, l0.value, l1.value)

    def emit_op_float_sub(self, op, arglocs):
        l0, l1, res = arglocs
        self.mc.DFSUB(res.value, l0.value, l1.value)

    def emit_op_float_mul(self, op, arglocs):
        l0, l1, tmp, res = arglocs
        self.mc.DFMUL(res.value, l0.value, l1.value, tmp.value)

    def emit_op_float_truediv(self, op, arglocs):
        l0, l1, res = arglocs
        # Hexagon has no hardware double-float division.
        # Emit a call to __hexagon_divdf3(dividend, divisor) -> double.
        # ABI: doubles in R1:R0 (arg1) and R3:R2 (arg2), result in R1:R0.
        #
        # Move args to ABI positions, call helper, move result.
        # Using scratch registers and d14 for temporaries.
        #
        # 1. Move dividend (l0 pair) to R1:R0
        if l0.value != r.d0.value:
            self.mc.MV(r.r0.value, l0.even_reg())
            self.mc.MV(r.r1.value, l0.odd_reg())
        # 2. Move divisor (l1 pair) to R3:R2
        if l1.value != r.d2.value:
            self.mc.MV(r.r2.value, l1.even_reg())
            self.mc.MV(r.r3.value, l1.odd_reg())
        # 3. Call __hexagon_divdf3
        divdf3_addr = self.float_div_addr
        self.mc.gen_load_int(r.scratch1.value, divdf3_addr)
        self.mc.CALLR(r.scratch1.value)
        # 4. Move result from R1:R0 to destination pair
        if res.value != r.d0.value:
            self.mc.MV(res.even_reg(), r.r0.value)
            self.mc.MV(res.odd_reg(), r.r1.value)

    def emit_op_float_neg(self, op, arglocs):
        l0, res = arglocs
        # Toggle sign bit of the high word (bit 31 of odd register)
        # Rdd = dfneg(Rss) ->toggle bit 63 of the 64-bit value
        # Implemented as: high_word ^= 0x80000000
        self.mc.MV(res.even_reg(), l0.even_reg())
        self.mc.gen_load_int(r.scratch1.value, 0x80000000)
        self.mc.XOR(res.odd_reg(), l0.odd_reg(), r.scratch1.value)

    def emit_op_float_abs(self, op, arglocs):
        l0, res = arglocs
        # Clear sign bit of the high word
        self.mc.MV(res.even_reg(), l0.even_reg())
        self.mc.gen_load_int(r.scratch1.value, 0x7FFFFFFF)
        self.mc.AND(res.odd_reg(), l0.odd_reg(), r.scratch1.value)

    def emit_op_convert_float_bytes_to_longlong(self, op, arglocs):
        l0, res = arglocs
        # On 32-bit Hexagon, floats are register pairs and longlong is also
        # a pair. This is a no-op bitwise reinterpretation.
        if l0.value != res.value:
            self.mc.MV(res.even_reg(), l0.even_reg())
            self.mc.MV(res.odd_reg(), l0.odd_reg())

    def emit_op_convert_longlong_bytes_to_float(self, op, arglocs):
        l0, res = arglocs
        if l0.value != res.value:
            self.mc.MV(res.even_reg(), l0.even_reg())
            self.mc.MV(res.odd_reg(), l0.odd_reg())

    # -------------------------------------------------------------------
    # Float comparisons
    # -------------------------------------------------------------------

    def _emit_float_cmp(self, l0, l1, res, cmp_type, invert=False):
        pd = r.scratch_pred.value
        if cmp_type == 'eq':
            self.mc.DFCMP_EQ(pd, l0.value, l1.value)
        elif cmp_type == 'gt':
            self.mc.DFCMP_GT(pd, l0.value, l1.value)
        self.mc.gen_load_int(r.scratch1.value, 1)
        self.mc.gen_load_int(r.scratch2.value, 0)
        if invert:
            self.mc.MUX(res.value, pd, r.scratch2.value, r.scratch1.value)
        else:
            self.mc.MUX(res.value, pd, r.scratch1.value, r.scratch2.value)

    def emit_op_float_lt(self, op, arglocs):
        l0, l1, res = arglocs
        self._emit_float_cmp(l1, l0, res, 'gt')

    def emit_op_float_le(self, op, arglocs):
        l0, l1, res = arglocs
        self._emit_float_cmp(l0, l1, res, 'gt', invert=True)

    def emit_op_float_eq(self, op, arglocs):
        l0, l1, res = arglocs
        self._emit_float_cmp(l0, l1, res, 'eq')

    def emit_op_float_ne(self, op, arglocs):
        l0, l1, res = arglocs
        self._emit_float_cmp(l0, l1, res, 'eq', invert=True)

    def emit_op_float_gt(self, op, arglocs):
        l0, l1, res = arglocs
        self._emit_float_cmp(l0, l1, res, 'gt')

    def emit_op_float_ge(self, op, arglocs):
        l0, l1, res = arglocs
        self._emit_float_cmp(l1, l0, res, 'gt', invert=True)

    # -------------------------------------------------------------------
    # Conversions
    # -------------------------------------------------------------------

    def emit_op_cast_float_to_int(self, op, arglocs):
        l0, res = arglocs
        self.mc.CONV_DF2W(res.value, l0.value)

    def emit_op_cast_int_to_float(self, op, arglocs):
        l0, res = arglocs
        self.mc.CONV_W2DF(res.value, l0.value)

    def emit_op_cast_float_to_singlefloat(self, op, arglocs):
        l0, res = arglocs
        # Convert double (pair) to single-precision float (word)
        self.mc.CONV_DF2SF(res.value, l0.value)

    def emit_op_cast_singlefloat_to_float(self, op, arglocs):
        l0, res = arglocs
        # Convert single-precision float (word) to double (pair)
        self.mc.CONV_SF2DF(res.value, l0.value)

    # -------------------------------------------------------------------
    # Memory operations
    # -------------------------------------------------------------------

    def emit_op_getfield_gc_i(self, op, arglocs):
        base, ofs, res = arglocs
        ofs_val, fieldsize, signed = unpack_fielddescr(op.getdescr())
        if fieldsize == 4:
            self.mc.LDW(res.value, base.value, ofs.value)
        elif fieldsize == 2:
            if signed:
                self.mc.LDH(res.value, base.value, ofs.value)
            else:
                self.mc.LDUH(res.value, base.value, ofs.value)
        elif fieldsize == 1:
            if signed:
                self.mc.LDB(res.value, base.value, ofs.value)
            else:
                self.mc.LDUB(res.value, base.value, ofs.value)

    emit_op_getfield_gc_r = emit_op_getfield_gc_i

    def emit_op_getfield_gc_f(self, op, arglocs):
        base, ofs, res = arglocs
        self.mc.LDD(res.value, base.value, ofs.value)

    def emit_op_setfield_gc(self, op, arglocs):
        base, ofs, val = arglocs
        ofs_val, fieldsize, signed = unpack_fielddescr(op.getdescr())
        if val.is_float() or fieldsize == 8:
            self.mc.STD(base.value, val.value, ofs.value)
        elif fieldsize == 4:
            self.mc.STW(base.value, val.value, ofs.value)
        elif fieldsize == 2:
            self.mc.STH(base.value, val.value, ofs.value)
        elif fieldsize == 1:
            self.mc.STB(base.value, val.value, ofs.value)

    # -------------------------------------------------------------------
    # Calls
    # -------------------------------------------------------------------

    def _emit_call(self, op, arglocs):
        from rpython.jit.backend.hexagon.callbuilder import HexagonCallBuilder
        is_call_release_gil = rop.is_call_release_gil(op.getopnum())

        descr = op.getdescr()
        assert isinstance(descr, CallDescr)

        # arglocs = [resloc, size, sign, funcloc, args...]
        # For release_gil: [resloc, size, sign, save_err, funcloc, args...]
        func_index = 3 + is_call_release_gil
        funcloc = arglocs[func_index]

        cb = HexagonCallBuilder(self, funcloc, arglocs[func_index + 1:],
                                arglocs[0],
                                descr.get_result_type(),
                                arglocs[1].value)  # ressize

        cb.callconv = descr.get_call_conv()
        cb.argtypes = descr.get_arg_types()
        cb.restype = descr.get_result_type()
        cb.ressize = arglocs[1].value
        cb.ressign = arglocs[2].value

        if is_call_release_gil:
            save_err_loc = arglocs[3]
            assert save_err_loc.is_imm()
            cb.emit_call_release_gil(save_err_loc.value)
        else:
            effectinfo = descr.get_extra_info()
            if effectinfo is None or effectinfo.check_can_collect():
                cb.emit()
            else:
                cb.emit_no_collect()

    emit_op_call_i = _emit_call
    emit_op_call_r = _emit_call
    emit_op_call_f = _emit_call
    emit_op_call_n = _emit_call
    emit_op_call_may_force_i = _emit_call
    emit_op_call_may_force_r = _emit_call
    emit_op_call_may_force_f = _emit_call
    emit_op_call_may_force_n = _emit_call
    emit_op_call_release_gil_i = _emit_call
    emit_op_call_release_gil_r = _emit_call
    emit_op_call_release_gil_f = _emit_call
    emit_op_call_release_gil_n = _emit_call

    # -------------------------------------------------------------------
    # GC load/store operations
    # -------------------------------------------------------------------

    def _normalize_mem_access(self, base_loc, ofs_loc):
        """Return (base_reg, imm_offset) for memory access.

        If ofs_loc is a register, add it to base and return (combined, 0).
        If ofs_loc is an immediate, return (base, imm).
        """
        if ofs_loc.is_core_reg():
            self.mc.ADD(r.scratch2.value, base_loc.value, ofs_loc.value)
            return r.scratch2.value, 0
        else:
            return base_loc.value, ofs_loc.value

    def _write_to_mem(self, value_loc, base_loc, ofs_loc, scale):
        base_reg, ofs = self._normalize_mem_access(base_loc, ofs_loc)
        if scale == 3:  # 8 bytes (double)
            self.mc.STD(base_reg, value_loc.value, ofs)
        elif scale == 2:  # 4 bytes (word)
            self.mc.STW(base_reg, value_loc.value, ofs)
        elif scale == 1:  # 2 bytes (halfword)
            self.mc.STH(base_reg, value_loc.value, ofs)
        elif scale == 0:  # 1 byte
            self.mc.STB(base_reg, value_loc.value, ofs)

    def _load_from_mem(self, res_loc, base_loc, ofs_loc, scale, signed):
        base_reg, ofs = self._normalize_mem_access(base_loc, ofs_loc)
        if scale == 3:  # 8 bytes (double)
            self.mc.LDD(res_loc.value, base_reg, ofs)
        elif scale == 2:  # 4 bytes (word)
            self.mc.LDW(res_loc.value, base_reg, ofs)
        elif scale == 1:  # 2 bytes (halfword)
            if signed:
                self.mc.LDH(res_loc.value, base_reg, ofs)
            else:
                self.mc.LDUH(res_loc.value, base_reg, ofs)
        elif scale == 0:  # 1 byte
            if signed:
                self.mc.LDB(res_loc.value, base_reg, ofs)
            else:
                self.mc.LDUB(res_loc.value, base_reg, ofs)

    def emit_op_gc_store(self, op, arglocs):
        value_loc, base_loc, ofs_loc, size_loc = arglocs
        scale = _get_scale(size_loc.value)
        self._write_to_mem(value_loc, base_loc, ofs_loc, scale)

    def _emit_gc_indexed_full_offset(self, index_loc, ofs_loc):
        """Compute full_offset = index + ofs for indexed GC operations."""
        assert index_loc.is_core_reg()
        if ofs_loc.is_imm():
            if ofs_loc.value == 0:
                return index_loc
            self.mc.ADDI(r.scratch1.value, index_loc.value, ofs_loc.value)
        else:
            self.mc.ADD(r.scratch1.value, index_loc.value, ofs_loc.value)
        return r.scratch1

    def emit_op_gc_store_indexed(self, op, arglocs):
        value_loc, base_loc, index_loc, ofs_loc, size_loc = arglocs
        full_ofs_loc = self._emit_gc_indexed_full_offset(index_loc, ofs_loc)
        scale = _get_scale(size_loc.value)
        self._write_to_mem(value_loc, base_loc, full_ofs_loc, scale)

    def _emit_op_gc_load(self, op, arglocs):
        base_loc, ofs_loc, res_loc, nsize_loc = arglocs
        nsize = nsize_loc.value
        signed = (nsize < 0)
        scale = _get_scale(abs(nsize))
        self._load_from_mem(res_loc, base_loc, ofs_loc, scale, signed)

    emit_op_gc_load_i = _emit_op_gc_load
    emit_op_gc_load_r = _emit_op_gc_load
    emit_op_gc_load_f = _emit_op_gc_load

    def _emit_op_gc_load_indexed(self, op, arglocs):
        base_loc, index_loc, ofs_loc, res_loc, nsize_loc = arglocs
        nsize = nsize_loc.value
        signed = (nsize < 0)
        full_ofs_loc = self._emit_gc_indexed_full_offset(index_loc, ofs_loc)
        scale = _get_scale(abs(nsize))
        self._load_from_mem(res_loc, base_loc, full_ofs_loc, scale, signed)

    emit_op_gc_load_indexed_i = _emit_op_gc_load_indexed
    emit_op_gc_load_indexed_r = _emit_op_gc_load_indexed
    emit_op_gc_load_indexed_f = _emit_op_gc_load_indexed

    # -------------------------------------------------------------------
    # Write barrier
    # -------------------------------------------------------------------

    def emit_op_cond_call_gc_wb(self, op, arglocs):
        self._write_barrier_fastpath(self.mc, op.getdescr(), arglocs)

    def emit_op_cond_call_gc_wb_array(self, op, arglocs):
        self._write_barrier_fastpath(self.mc, op.getdescr(), arglocs,
                                     array=True)

    def _write_barrier_fastpath(self, mc, descr, arglocs, array=False,
                                is_frame=False):
        if we_are_translated():
            cls = self.cpu.gc_ll_descr.has_write_barrier_class()
            assert cls is not None and isinstance(descr, cls)

        card_marking = False
        mask = descr.jit_wb_if_flag_singlebyte
        if array and descr.jit_wb_cards_set != 0:
            assert (descr.jit_wb_cards_set_byteofs ==
                    descr.jit_wb_if_flag_byteofs)
            assert descr.jit_wb_cards_set_singlebyte == -0x80
            card_marking = True
            mask = descr.jit_wb_if_flag_singlebyte | -0x80

        loc_base = arglocs[0]
        # Load the flag byte from the object
        mc.LDUB(r.scratch1.value, loc_base.value,
                descr.jit_wb_if_flag_byteofs)
        # Test against mask
        mc.gen_load_int(r.scratch2.value, mask & 0xFF)
        mc.AND(r.scratch1.value, r.scratch1.value, r.scratch2.value)
        pd = r.scratch_pred.value
        mc.CMP_EQI(pd, r.scratch1.value, 0)
        # If zero, nothing to do -> jump past the slowpath call
        jz_location = mc.get_relative_pos()
        mc.TRAP0(0xDE)  # placeholder, patched to: if (P0) jump past

        if card_marking:
            # Check GCFLAG_CARDS_SET bit
            mc.LDUB(r.scratch1.value, loc_base.value,
                    descr.jit_wb_if_flag_byteofs)
            mc.gen_load_int(r.scratch2.value, 0x80)
            mc.AND(r.scratch1.value, r.scratch1.value, r.scratch2.value)
            mc.CMP_EQI(pd, r.scratch1.value, 0)
            js_location = mc.get_relative_pos()
            mc.TRAP0(0xDE)  # placeholder
        else:
            js_location = 0

        # Slow path: call the write barrier helper
        helper_num = 1 if card_marking else 0
        if self.wb_slowpath[helper_num] == 0:
            assert not we_are_translated()
            self.cpu.gc_ll_descr.write_barrier_descr = descr
            self._build_wb_slowpath(card_marking)
            assert self.wb_slowpath[helper_num] != 0

        # Pass object address in R0
        if loc_base is not r.r0:
            mc.ADDI(r.sp.value, r.sp.value, -2 * WORD)
            mc.STW(r.sp.value, r.r0.value, 0)
            mc.MV(r.r0.value, loc_base.value)
        mc.gen_load_int(r.scratch1.value, self.wb_slowpath[helper_num])
        mc.CALLR(r.scratch1.value)
        if loc_base is not r.r0:
            mc.LDW(r.r0.value, r.sp.value, 0)
            mc.ADDI(r.sp.value, r.sp.value, 2 * WORD)

        if card_marking:
            jns_location = mc.get_relative_pos()
            mc.TRAP0(0xDE)  # placeholder

            # Patch js_location: jump here if GCFLAG_CARDS_SET was set
            offset = mc.get_relative_pos() - js_location
            pmc = OverwritingBuilder(mc, js_location, INST_SIZE)
            pmc.J_IF_FALSE(pd, offset)  # jump if NOT zero (cards set)

            # Card marking: set the card bit directly
            loc_index = arglocs[1]
            assert loc_index.is_core_reg()
            # byte_ofs = ~(index >> card_page_shift)
            mc.S2_ASR_I_R(r.scratch1.value, loc_index.value,
                          descr.jit_wb_card_page_shift)
            mc.NOT(r.scratch1.value, r.scratch1.value)
            # bit_index = (index >> card_page_shift) & 7
            mc.S2_LSR_I_R(r.scratch2.value, loc_index.value,
                          descr.jit_wb_card_page_shift)
            mc.ANDI(r.scratch2.value, r.scratch2.value, 7)
            # bit_mask = 1 << bit_index
            # Reuse loc_index as temporary (no longer needed after shifts above)
            mc.TFRSI(loc_index.value, 1)
            mc.S2_ASL_R_R(loc_index.value, loc_index.value,
                          r.scratch2.value)
            # Load current byte, OR in the bit, store back
            mc.ADD(r.scratch1.value, loc_base.value, r.scratch1.value)
            mc.LDUB(r.scratch2.value, r.scratch1.value, 0)
            mc.OR(r.scratch2.value, r.scratch2.value, loc_index.value)
            mc.STB(r.scratch1.value, r.scratch2.value, 0)

            # Patch jns_location
            offset = mc.get_relative_pos() - jns_location
            pmc = OverwritingBuilder(mc, jns_location, INST_SIZE)
            pmc.J_IF_TRUE(pd, offset)

        # Patch jz_location: jump here if flags were zero
        offset = mc.get_relative_pos() - jz_location
        pmc = OverwritingBuilder(mc, jz_location, INST_SIZE)
        pmc.J_IF_TRUE(pd, offset)  # jump if P0=1 (flag was zero)

    # -------------------------------------------------------------------
    # Nursery allocation
    # -------------------------------------------------------------------

    def emit_op_call_malloc_nursery(self, op, arglocs):
        gc_ll_descr = self.cpu.gc_ll_descr
        size = op.getarg(0).getint()
        self.malloc_cond(
            gc_ll_descr.get_nursery_free_addr(),
            gc_ll_descr.get_nursery_top_addr(),
            size)

    def emit_op_call_malloc_nursery_varsize(self, op, arglocs):
        gc_ll_descr = self.cpu.gc_ll_descr
        length_loc = arglocs[0]
        # Get the kind, item size and GC type ID from the op
        arraydescr = op.getdescr()
        kind = op.getarg(0).getint()
        itemsize = op.getarg(1).getint()
        maxlength = (gc_ll_descr.max_size_of_young_obj - WORD * 2) // itemsize
        # r2 is clobbered by the slowpath argument setup (reserved in
        # prepare), so refs must not be tracked there either
        gcmap = self._regalloc.get_gcmap([r.r0, r.r1, r.r2])
        self.malloc_cond_varsize(
            kind,
            gc_ll_descr.get_nursery_free_addr(),
            gc_ll_descr.get_nursery_top_addr(),
            length_loc, gcmap, arraydescr, itemsize, maxlength)

    def emit_op_call_malloc_nursery_varsize_frame(self, op, arglocs):
        gc_ll_descr = self.cpu.gc_ll_descr
        size_loc = arglocs[0]
        self.malloc_cond_varsize_frame(
            gc_ll_descr.get_nursery_free_addr(),
            gc_ll_descr.get_nursery_top_addr(),
            size_loc)

    def malloc_cond(self, nursery_free_adr, nursery_top_adr, size):
        assert size & (WORD - 1) == 0
        mc = self.mc
        # r0 = nursery_free (result ptr)
        mc.gen_load_int(r.scratch1.value, nursery_free_adr)
        mc.LDW(r.r0.value, r.scratch1.value, 0)
        # r1 = nursery_free + size (new free pointer)
        mc.ADDI(r.r1.value, r.r0.value, size)
        # scratch1 = nursery_top
        mc.gen_load_int(r.scratch1.value, nursery_top_adr)
        mc.LDW(r.scratch1.value, r.scratch1.value, 0)
        # Compare: is r1 <= nursery_top?
        pd = r.scratch_pred.value
        mc.CMP_GTU(pd, r.r1.value, r.scratch1.value)
        # If r1 > nursery_top, we need the slow path
        jmp_pos = mc.get_relative_pos()
        mc.TRAP0(0xDE)  # placeholder: if (!P0) jump past (allocation ok)

        # Slow path: call the malloc_slowpath *stub* (never the raw GC
        # function: the stub saves/restores all registers to the
        # jitframe, checks for failure, reloads the possibly-moved
        # frame, and returns R0 = result, R1 = up-to-date nursery_free
        # -- which the store below relies on).  Stub entry convention:
        # R0 = nursery_free, R1 = nursery_free + size (already set).
        gcmap = self._regalloc.get_gcmap([r.r0, r.r1])
        self.push_gcmap(mc, gcmap)
        mc.gen_load_int(r.scratch1.value, self.malloc_slowpath)
        mc.CALLR(r.scratch1.value)
        self.pop_gcmap(mc)

        # Update nursery_free = r1
        end_pos = mc.get_relative_pos()
        mc.gen_load_int(r.scratch1.value, nursery_free_adr)
        mc.STW(r.scratch1.value, r.r1.value, 0)

        # Patch the conditional jump (skip slow path when allocation succeeds)
        # P0=1 means r1 > top (need slow path), P0=0 means ok
        offset = end_pos - jmp_pos
        pmc = OverwritingBuilder(mc, jmp_pos, INST_SIZE)
        pmc.J_IF_FALSE(pd, offset)  # jump past slow path when P0=0 (fits)

    def malloc_cond_varsize_frame(self, nursery_free_adr, nursery_top_adr,
                                  size_loc):
        mc = self.mc
        if size_loc is r.r0:
            mc.MV(r.r1.value, r.r0.value)
            size_loc = r.r1
        mc.gen_load_int(r.scratch1.value, nursery_free_adr)
        mc.LDW(r.r0.value, r.scratch1.value, 0)
        mc.ADD(r.r1.value, r.r0.value, size_loc.value)
        mc.gen_load_int(r.scratch1.value, nursery_top_adr)
        mc.LDW(r.scratch1.value, r.scratch1.value, 0)
        pd = r.scratch_pred.value
        mc.CMP_GTU(pd, r.r1.value, r.scratch1.value)
        jmp_pos = mc.get_relative_pos()
        mc.TRAP0(0xDE)  # placeholder

        # Slow path: via the register-saving stub, like malloc_cond
        # (R0 = nursery_free, R1 = nursery_free + size already set;
        # the stub returns R1 = up-to-date nursery_free)
        gcmap = self._regalloc.get_gcmap([r.r0, r.r1])
        self.push_gcmap(mc, gcmap)
        mc.gen_load_int(r.scratch1.value, self.malloc_slowpath)
        mc.CALLR(r.scratch1.value)
        self.pop_gcmap(mc)

        end_pos = mc.get_relative_pos()
        mc.gen_load_int(r.scratch1.value, nursery_free_adr)
        mc.STW(r.scratch1.value, r.r1.value, 0)

        offset = end_pos - jmp_pos
        pmc = OverwritingBuilder(mc, jmp_pos, INST_SIZE)
        pmc.J_IF_FALSE(pd, offset)

    def malloc_cond_varsize(self, kind, nursery_free_adr, nursery_top_adr,
                            length_loc, gcmap, arraydescr, itemsize,
                            maxlength):
        from rpython.jit.backend.llsupport import rewrite
        from rpython.jit.backend.llsupport.descr import ArrayDescr
        assert isinstance(arraydescr, ArrayDescr)
        mc = self.mc
        pd = r.scratch_pred.value
        # Check if length > maxlength (needs slow path)
        mc.gen_load_int(r.scratch1.value, maxlength)
        mc.CMP_GTU(pd, length_loc.value, r.scratch1.value)
        jmp_too_big = mc.get_relative_pos()
        mc.TRAP0(0xDE)  # placeholder: if (P0) jump slow_path

        # Fast path: try nursery allocation
        # total_size = basesize + length * itemsize, aligned
        if itemsize == 1:
            mc.MV(r.scratch2.value, length_loc.value)
        elif itemsize == 2:
            mc.S2_ASL_I_R(r.scratch2.value, length_loc.value, 1)
        elif itemsize == 4:
            mc.S2_ASL_I_R(r.scratch2.value, length_loc.value, 2)
        elif itemsize == 8:
            mc.S2_ASL_I_R(r.scratch2.value, length_loc.value, 3)
        else:
            mc.gen_load_int(r.scratch1.value, itemsize)
            mc.MPYI(r.scratch2.value, length_loc.value, r.scratch1.value)
        # Add the header size (basesize differs per kind: e.g. strings
        # also have a hash field before the length)
        mc.ADDI(r.scratch2.value, r.scratch2.value, arraydescr.basesize)
        # Align to WORD
        mc.ADDI(r.scratch2.value, r.scratch2.value, WORD - 1)
        mc.gen_load_int(r.scratch1.value, ~(WORD - 1))
        mc.AND(r.scratch2.value, r.scratch2.value, r.scratch1.value)
        # r0 = nursery_free
        mc.gen_load_int(r.scratch1.value, nursery_free_adr)
        mc.LDW(r.r0.value, r.scratch1.value, 0)
        # r1 = nursery_free + total_size
        mc.ADD(r.r1.value, r.r0.value, r.scratch2.value)
        # Compare against nursery_top
        mc.gen_load_int(r.scratch1.value, nursery_top_adr)
        mc.LDW(r.scratch1.value, r.scratch1.value, 0)
        mc.CMP_GTU(pd, r.r1.value, r.scratch1.value)
        jmp_fits = mc.get_relative_pos()
        mc.TRAP0(0xDE)  # placeholder: if (P0) jump slow_path

        # Fast path: update nursery_free and skip slow path
        mc.gen_load_int(r.scratch1.value, nursery_free_adr)
        mc.STW(r.scratch1.value, r.r1.value, 0)
        jmp_done = mc.get_relative_pos()
        mc.TRAP0(0xDE)  # placeholder: jump past slow path

        # Slow path
        slow_pos = mc.get_relative_pos()
        # Patch both jumps to here
        offset = slow_pos - jmp_too_big
        pmc = OverwritingBuilder(mc, jmp_too_big, INST_SIZE)
        pmc.J_IF_TRUE(pd, offset)
        offset = slow_pos - jmp_fits
        pmc = OverwritingBuilder(mc, jmp_fits, INST_SIZE)
        pmc.J_IF_TRUE(pd, offset)

        self.push_gcmap(mc, gcmap)
        # Call the register-saving slowpath stub for the right kind
        if kind == rewrite.FLAG_ARRAY:
            # var stub convention: R0 = itemsize, R1 = tid, R2 = length
            if length_loc.value != r.r2.value:
                mc.MV(r.r2.value, length_loc.value)
            mc.gen_load_int(r.r1.value, arraydescr.tid)
            mc.gen_load_int(r.r0.value, itemsize)
            addr = self.malloc_slowpath_varsize
        else:
            # str/unicode stub convention: R0 = length
            if length_loc.value != r.r0.value:
                mc.MV(r.r0.value, length_loc.value)
            if kind == rewrite.FLAG_STR:
                addr = self.malloc_slowpath_str
            else:
                assert kind == rewrite.FLAG_UNICODE
                addr = self.malloc_slowpath_unicode
        mc.gen_load_int(r.scratch1.value, addr)
        mc.CALLR(r.scratch1.value)
        self.pop_gcmap(mc)

        end_pos = mc.get_relative_pos()
        # Patch jmp_done to skip to here
        offset = end_pos - jmp_done
        pmc = OverwritingBuilder(mc, jmp_done, INST_SIZE)
        pmc.JUMP(offset)

    # -------------------------------------------------------------------
    # Same-as / moves
    # -------------------------------------------------------------------

    def emit_op_same_as_i(self, op, arglocs):
        l0, res = arglocs
        if l0.is_imm():
            self.mc.gen_load_int(res.value, l0.value)
        elif l0.value != res.value:
            self.mc.MV(res.value, l0.value)

    emit_op_same_as_r = emit_op_same_as_i
    emit_op_cast_ptr_to_int = emit_op_same_as_i
    emit_op_cast_int_to_ptr = emit_op_same_as_i

    def emit_op_same_as_f(self, op, arglocs):
        l0, res = arglocs
        if l0.value != res.value:
            # Move register pair
            self.mc.MV(res.even_reg(), l0.even_reg())
            self.mc.MV(res.odd_reg(), l0.odd_reg())

    # -------------------------------------------------------------------
    # Guards
    #
    # Convention: before each guard, P0 (scratch_pred) is set by a compare.
    # The token stores `fail_on_pred_true`:
    #   True  -> guard fails when P0=1 -> patched to: if (P0) jump fail
    #   False -> guard fails when P0=0 -> patched to: if (!P0) jump fail
    # -------------------------------------------------------------------

    def emit_op_guard_true(self, op, arglocs):
        l0 = arglocs[0]
        guard_arglocs = arglocs[1:]
        pd = r.scratch_pred.value
        # P0 = (l0 == 0); guard fails when l0==0, i.e. when P0=1
        self.mc.CMP_EQI(pd, l0.value, 0)
        self._emit_guard(op, guard_arglocs, fail_on_pred_true=True)

    def emit_op_guard_false(self, op, arglocs):
        l0 = arglocs[0]
        guard_arglocs = arglocs[1:]
        pd = r.scratch_pred.value
        # P0 = (l0 == 0); guard fails when l0!=0, i.e. when P0=0
        self.mc.CMP_EQI(pd, l0.value, 0)
        self._emit_guard(op, guard_arglocs, fail_on_pred_true=False)

    emit_op_guard_nonnull = emit_op_guard_true
    emit_op_guard_isnull = emit_op_guard_false

    def _emit_guard(self, op, guard_arglocs, fail_on_pred_true):
        """Emit a guard placeholder (TRAP0), to be patched later.

        P0 must already be set before calling this.
        guard_arglocs: [frame_depth_imm, fail_loc0, fail_loc1, ...]
        fail_on_pred_true: if True, guard fails (jumps to stub) when P0=1.
                           if False, guard fails when P0=0.
        """
        token = self._build_guard_token(op, guard_arglocs[0].value,
                                        guard_arglocs[1:],
                                        self.mc.get_relative_pos())
        token.pos_jump_offset = self.mc.get_relative_pos()
        token.fail_on_pred_true = fail_on_pred_true
        self.mc.TRAP0(0xDE)  # placeholder, will be patched
        self.pending_guards.append(token)

    def _build_guard_token(self, op, frame_depth, fail_locs, offset):
        """Create a GuardToken with all required info for recovery."""
        from rpython.jit.backend.llsupport.gcmap import allocate_gcmap
        from rpython.jit.backend.hexagon.arch import JITFRAME_FIXED_SIZE
        descr = op.getdescr()
        gcmap = allocate_gcmap(self, frame_depth, JITFRAME_FIXED_SIZE)
        faildescrindex = self.get_gcref_from_faildescr(descr)
        return GuardToken(self.cpu, gcmap, descr,
                          failargs=op.getfailargs(), fail_locs=fail_locs,
                          guard_opnum=op.getopnum(),
                          frame_depth=frame_depth,
                          faildescrindex=faildescrindex)

    def emit_op_guard_value(self, op, arglocs):
        l0, l1 = arglocs[0], arglocs[1]
        guard_arglocs = arglocs[2:]
        pd = r.scratch_pred.value
        # P0 = (l0 == l1); guard fails when not equal, i.e. P0=0
        self.mc.CMP_EQ(pd, l0.value, l1.value)
        self._emit_guard(op, guard_arglocs, fail_on_pred_true=False)

    def emit_op_guard_no_exception(self, op, arglocs):
        # Check if pos_exception() is NULL (no exception pending)
        pd = r.scratch_pred.value
        self.mc.gen_load_int(r.scratch1.value, self.cpu.pos_exception())
        self.mc.LDW(r.scratch1.value, r.scratch1.value, 0)
        self.mc.CMP_EQI(pd, r.scratch1.value, 0)
        # P0=1 if no exception (pass), P0=0 if exception (fail)
        self._emit_guard(op, arglocs, fail_on_pred_true=False)

    def emit_op_guard_not_invalidated(self, op, arglocs):
        # Record the current code position for later invalidation.
        # When invalidated, the instruction at this position is overwritten
        # with a jump to the guard failure stub.
        mc = self.mc
        pos = mc.get_relative_pos()
        token = self._build_guard_token(op, arglocs[0].value,
                                        arglocs[1:], pos)
        token.pos_jump_offset = pos
        token.fail_on_pred_true = True
        mc.NOP()  # placeholder; overwritten with jump on invalidation
        self.pending_guards.append(token)

    def emit_op_guard_not_forced(self, op, arglocs):
        # Check if jf_descr was set (indicating a forced frame)
        ofs = self.cpu.get_ofs_of_frame_field('jf_descr')
        self.mc.load_from_jitframe(r.scratch1.value, ofs)
        pd = r.scratch_pred.value
        self.mc.CMP_EQI(pd, r.scratch1.value, 0)
        # If jf_descr != 0, the frame was forced -> fail
        self._emit_guard(op, arglocs, fail_on_pred_true=False)

    emit_op_guard_not_forced_2 = emit_op_guard_not_forced

    def emit_guard_op_guard_not_forced(self, call_op, guard_op, arglocs,
                                       num_arglocs):
        """Handle fused call_may_force/call_assembler + guard_not_forced."""
        guard_arglocs = arglocs[num_arglocs:]

        if rop.is_call_assembler(call_op.getopnum()):
            if num_arglocs == 3:
                [argloc, vloc, result_loc] = arglocs[:3]
            else:
                [argloc, result_loc] = arglocs[:2]
                vloc = self.imm(0)
            self._store_force_index(guard_op)
            # tmploc = r.r0 (call returns jitframe in R0)
            self.call_assembler(call_op, argloc, vloc, result_loc,
                                tmploc=r.r0)
        else:
            assert num_arglocs == call_op.numargs() + 3
            call_op_arglocs = arglocs[0:num_arglocs]
            self._store_force_index(guard_op)
            self._emit_call(call_op, call_op_arglocs)

        # Implement guard_not_forced:
        #   if frame.jf_descr != 0: goto guard_handler
        ofs = self.cpu.get_ofs_of_frame_field('jf_descr')
        self.mc.load_from_jitframe(r.scratch1.value, ofs)
        pd = r.scratch_pred.value
        self.mc.CMP_EQI(pd, r.scratch1.value, 0)
        # If jf_descr != 0, the frame was forced -> fail
        self._emit_guard(guard_op, guard_arglocs, fail_on_pred_true=False)

    # -------------------------------------------------------------------
    # Conditional calls
    # -------------------------------------------------------------------

    def _emit_op_cond_call(self, op, arglocs):
        """Emit COND_CALL and COND_CALL_VALUE operations.

        cond_call(cond, func, *args):
            if cond != 0: func(*args)

        res = cond_call_value(cond, func, *args):
            res = cond or func(*args)
        """
        cond_loc = arglocs[0]
        if len(arglocs) == 2:
            res_loc = arglocs[1]     # cond_call_value
        else:
            res_loc = None           # cond_call

        # Skip res_loc in gcmap (see x86.regalloc for why)
        gcmap = self._regalloc.get_gcmap([res_loc])

        # Test condition: P0 = (cond == 0)
        pd = r.scratch_pred.value
        self.mc.CMP_EQI(pd, cond_loc.value, 0)

        # Conditional branch placeholder (will be overwritten with offset)
        cond_branch_addr = self.mc.get_relative_pos()
        self.mc.TRAP0(0xDE)  # placeholder

        self.push_gcmap(self.mc, gcmap)

        # Load callee function address into scratch1
        callee_func_addr = rffi.cast(lltype.Signed, op.getarg(1).getint())
        self.mc.gen_load_int(r.scratch1.value, callee_func_addr)

        # Determine which slowpath variant to use
        callee_only = False
        floats = False
        if self._regalloc is not None:
            for reg in self._regalloc.rm.reg_bindings.values():
                if reg not in self._regalloc.rm.save_around_call_regs:
                    break
            else:
                callee_only = True
            if len(self._regalloc.fprm.reg_bindings):
                floats = True

        # Jump to cond_call trampoline
        trampoline_addr = self.cond_call_slowpath[floats * 2 + callee_only]
        assert trampoline_addr != 0
        self.mc.gen_load_int(r.scratch2.value, trampoline_addr)
        self.mc.CALLR(r.scratch2.value)

        # Move result from scratch1 (R14) to result location
        if res_loc is not None:
            self.mc.MV(res_loc.value, r.scratch1.value)

        self.pop_gcmap(self.mc)

        # Overwrite the conditional branch placeholder
        branch_dest_addr = self.mc.get_relative_pos()
        branch_offset = branch_dest_addr - cond_branch_addr
        pmc = OverwritingBuilder(self.mc, cond_branch_addr, INST_SIZE)
        if res_loc is None:
            # cond_call: skip the call when cond == 0 (P0=1 means skip)
            pmc.J_IF_TRUE(pd, branch_offset)
        else:
            # cond_call_value: skip when cond != 0 (P0=0 means skip)
            pmc.J_IF_FALSE(pd, branch_offset)

    emit_op_cond_call = _emit_op_cond_call
    emit_op_cond_call_value_i = _emit_op_cond_call
    emit_op_cond_call_value_r = _emit_op_cond_call

    # -------------------------------------------------------------------
    # Call assembler helpers (used by base class call_assembler method)
    # -------------------------------------------------------------------

    def _call_assembler_emit_call(self, addr, argloc, tmploc):
        """Emit the call to the target compiled loop."""
        assert tmploc is r.r0
        self.simple_call(addr, [argloc], result_loc=tmploc)

    def _call_assembler_check_descr(self, expected_descr, tmploc):
        """Check if the returned frame's descriptor matches expected.

        Emits compare + predicate set, then a placeholder conditional branch.
        Returns the position of the placeholder to be patched later.
        """
        ofs = self.cpu.get_ofs_of_frame_field('jf_descr')
        self.mc.LDW(r.scratch1.value, tmploc.value, ofs)

        # Compare: scratch1 == expected_descr? Set P0
        self.mc.gen_load_int(r.scratch2.value, expected_descr)
        pd = r.scratch_pred.value
        self.mc.CMP_EQ(pd, r.scratch1.value, r.scratch2.value)

        # Placeholder: will be patched with J_IF_TRUE to fast path
        pos = self.mc.get_relative_pos()
        self.mc.TRAP0(0xDE)  # placeholder
        return pos

    def _call_assembler_emit_helper_call(self, addr, arglocs, resloc):
        """Call the assembler helper function (slow path)."""
        self.simple_call(addr, arglocs, result_loc=resloc)

    def _call_assembler_patch_je(self, result_loc, jmp_location):
        """Patch the conditional branch from check_descr and emit jump-to-end.

        Returns position of the jump-to-end placeholder.
        """
        # Placeholder: unconditional J to end (patched by _patch_jmp)
        pos = self.mc.get_relative_pos()
        self.mc.TRAP0(0xDE)  # placeholder

        # Patch check_descr placeholder: if P0=1 (match), jump here
        currpos = self.mc.get_relative_pos()
        offset = currpos - jmp_location
        pmc = OverwritingBuilder(self.mc, jmp_location, INST_SIZE)
        pmc.J_IF_TRUE(r.scratch_pred.value, offset)

        return pos

    def _call_assembler_load_result(self, op, result_loc):
        """Load the return value from the returned jitframe (fast path)."""
        if op.type != 'v':
            kind = op.type
            descr = self.cpu.getarraydescr_for_frame(kind)
            ofs = self.cpu.unpack_arraydescr(descr)
            if kind == FLOAT:
                assert result_loc.is_float()
                self.mc.LDD(result_loc.value, r.r0.value, ofs)
            else:
                assert result_loc.is_core_reg()
                self.mc.LDW(result_loc.value, r.r0.value, ofs)

    def _call_assembler_patch_jmp(self, jmp_location):
        """Patch the jump-to-end placeholder."""
        currpos = self.mc.get_relative_pos()
        offset = currpos - jmp_location
        pmc = OverwritingBuilder(self.mc, jmp_location, INST_SIZE)
        pmc.JUMP(offset)

    def emit_op_guard_no_overflow(self, op, arglocs):
        # P0 was set by the preceding overflow op:
        # P0=1 means no overflow (pass), P0=0 means overflow (fail)
        self._emit_guard(op, arglocs, fail_on_pred_true=False)

    def emit_op_guard_overflow(self, op, arglocs):
        # P0 was set by the preceding overflow op:
        # P0=1 means no overflow (fail), P0=0 means overflow (pass)
        self._emit_guard(op, arglocs, fail_on_pred_true=True)

    def emit_op_guard_class(self, op, arglocs):
        obj_loc, expected_class = arglocs[0], arglocs[1]
        guard_arglocs = arglocs[2:]
        pd = r.scratch_pred.value
        offset = self.cpu.vtable_offset
        if offset is not None:
            # Load class pointer from object
            self.mc.LDW(r.scratch1.value, obj_loc.value, offset)
            self.mc.gen_load_int(r.scratch2.value, expected_class.value)
            self.mc.CMP_EQ(pd, r.scratch1.value, r.scratch2.value)
        else:
            # GC removes type pointers; use typeid comparison
            self._cmp_guard_gc_type(obj_loc, expected_class.value)
        self._emit_guard(op, guard_arglocs, fail_on_pred_true=False)

    def _cmp_guard_gc_type(self, obj_loc, expected_typeid):
        """Compare object's GC type id against expected_typeid, set P0."""
        pd = r.scratch_pred.value
        # Load typeid from object header (offset 0, half word)
        self.mc.LDUH(r.scratch1.value, obj_loc.value, 0)
        self.mc.gen_load_int(r.scratch2.value, expected_typeid)
        self.mc.CMP_EQ(pd, r.scratch1.value, r.scratch2.value)

    def emit_op_guard_nonnull_class(self, op, arglocs):
        obj_loc, expected_class = arglocs[0], arglocs[1]
        guard_arglocs = arglocs[2:]
        pd = r.scratch_pred.value
        # First check: is obj null? (P0=1 if null -> fail)
        self.mc.CMP_EQI(pd, obj_loc.value, 0)
        # If null, jump to guard failure
        null_pos = self.mc.get_relative_pos()
        self.mc.TRAP0(0xDE)  # placeholder: if (P0) jump fail

        # Second check: class match
        offset = self.cpu.vtable_offset
        if offset is not None:
            self.mc.LDW(r.scratch1.value, obj_loc.value, offset)
            self.mc.gen_load_int(r.scratch2.value, expected_class.value)
            self.mc.CMP_EQ(pd, r.scratch1.value, r.scratch2.value)
        else:
            self._cmp_guard_gc_type(obj_loc, expected_class.value)
        # P0=1 means match (pass), P0=0 means mismatch (fail)
        # Emit guard that fails when P0=0
        self._emit_guard(op, guard_arglocs, fail_on_pred_true=False)

        # Patch the null check to jump to the guard's recovery stub
        # (which is the last pending guard)
        token = self.pending_guards[-1]
        null_target = token.pos_jump_offset  # the TRAP0 from _emit_guard
        # Actually we need to jump to the same place. Patch null check
        # to jump to the guard fail stub position.
        # Since both checks fail to the same place, just patch to
        # the token's TRAP0 position (which will itself be patched later).
        # For now, patch null to jump past class check to the guard TRAP0.
        offset = token.pos_jump_offset - null_pos
        pmc = OverwritingBuilder(self.mc, null_pos, INST_SIZE)
        pmc.J_IF_TRUE(pd, offset)  # if null, jump to guard's TRAP0

    def emit_op_guard_gc_type(self, op, arglocs):
        obj_loc, expected_typeid = arglocs[0], arglocs[1]
        guard_arglocs = arglocs[2:]
        self._cmp_guard_gc_type(obj_loc, expected_typeid.value)
        self._emit_guard(op, guard_arglocs, fail_on_pred_true=False)

    def emit_op_guard_is_object(self, op, arglocs):
        obj_loc = arglocs[0]
        guard_arglocs = arglocs[1:]
        pd = r.scratch_pred.value
        gc_ll_descr = self.cpu.gc_ll_descr
        # Load typeid from GC header (half-word at offset 0)
        self.mc.LDUH(r.scratch1.value, obj_loc.value, 0)
        # Look up infobits in the type info table
        base_type_info, shift_by, sizeof_ti = (
            gc_ll_descr.get_translated_info_for_typeinfo())
        infobits_offset, IS_OBJECT_FLAG = (
            gc_ll_descr.get_translated_info_for_guard_is_object())
        # scratch2 = base_type_info + infobits_offset
        self.mc.gen_load_int(r.scratch2.value, base_type_info + infobits_offset)
        # scratch1 = typeid << shift_by (byte offset into type info array)
        if shift_by > 0:
            self.mc.S2_ASL_I_R(r.scratch1.value, r.scratch1.value, shift_by)
        # scratch1 = base + infobits_offset + (typeid << shift_by)
        self.mc.ADD(r.scratch1.value, r.scratch1.value, r.scratch2.value)
        # Load the infobits byte
        self.mc.LDUB(r.scratch1.value, r.scratch1.value, 0)
        # Test IS_OBJECT_FLAG
        self.mc.ANDI(r.scratch1.value, r.scratch1.value, IS_OBJECT_FLAG)
        # P0 = (scratch1 != 0) -- i.e. flag is set
        self.mc.CMP_EQI(pd, r.scratch1.value, 0)
        # Guard passes when P0=0 (flag IS set, so scratch1 != 0)
        # fail_on_pred_true=True: fail if P0=1 (flag not set)
        self._emit_guard(op, guard_arglocs, fail_on_pred_true=True)

    def emit_op_guard_subclass(self, op, arglocs):
        from rpython.rtyper.lltypesystem.lloperation import llop
        from rpython.rtyper.rclass import OBJECT_VTABLE
        obj_loc, expected_class = arglocs[0], arglocs[1]
        guard_arglocs = arglocs[2:]
        pd = r.scratch_pred.value
        offset = self.cpu.vtable_offset
        offset2 = self.cpu.subclassrange_min_offset
        if offset is not None:
            # Read vtable pointer, then subclassrange_min from vtable
            self.mc.LDW(r.scratch1.value, obj_loc.value, offset)
            self.mc.LDW(r.scratch1.value, r.scratch1.value, offset2)
        else:
            # Read typeid, look up subclassrange_min from type info table
            gc_ll_descr = self.cpu.gc_ll_descr
            self.mc.LDUH(r.scratch1.value, obj_loc.value, 0)
            base_type_info, shift_by, sizeof_ti = (
                gc_ll_descr.get_translated_info_for_typeinfo())
            self.mc.gen_load_int(r.scratch2.value,
                                 base_type_info + sizeof_ti + offset2)
            if shift_by > 0:
                self.mc.S2_ASL_I_R(r.scratch1.value, r.scratch1.value,
                                   shift_by)
            self.mc.ADD(r.scratch1.value, r.scratch1.value, r.scratch2.value)
            self.mc.LDW(r.scratch1.value, r.scratch1.value, 0)
        # Get the subclass range bounds from the expected class vtable
        from rpython.rtyper import rclass
        vtable_ptr = rffi.cast(rclass.CLASSTYPE, expected_class.value)
        check_min = vtable_ptr.subclassrange_min
        check_max = vtable_ptr.subclassrange_max
        assert check_max > check_min
        check_diff = check_max - check_min - 1
        # Unsigned range check: (subclassrange_min - check_min) <= check_diff
        self.mc.gen_load_int(r.scratch2.value, check_min)
        self.mc.SUB(r.scratch1.value, r.scratch1.value, r.scratch2.value)
        self.mc.gen_load_int(r.scratch2.value, check_diff)
        self.mc.CMP_GTU(pd, r.scratch1.value, r.scratch2.value)
        # P0=1 if scratch1 > check_diff (unsigned) => out of range => FAIL
        self._emit_guard(op, guard_arglocs, fail_on_pred_true=True)

    def emit_op_guard_exception(self, op, arglocs):
        expected_class, res = arglocs[0], arglocs[1]
        guard_arglocs = arglocs[2:]
        pd = r.scratch_pred.value
        # Load the current exception type
        self.mc.gen_load_int(r.scratch1.value, self.cpu.pos_exception())
        self.mc.LDW(r.scratch1.value, r.scratch1.value, 0)
        # Compare with expected
        self.mc.CMP_EQ(pd, r.scratch1.value, expected_class.value)
        # P0=1 if match (pass), P0=0 if mismatch (fail)
        self._emit_guard(op, guard_arglocs, fail_on_pred_true=False)
        # On pass: load exception value into result, then clear exception
        if res is not None:
            self.mc.gen_load_int(r.scratch1.value, self.cpu.pos_exc_value())
            self.mc.LDW(res.value, r.scratch1.value, 0)
            # Clear exception
            self.mc.gen_load_int(r.scratch2.value, 0)
            self.mc.STW(r.scratch1.value, r.scratch2.value, 0)
            self.mc.gen_load_int(r.scratch1.value, self.cpu.pos_exception())
            self.mc.STW(r.scratch1.value, r.scratch2.value, 0)

    # -------------------------------------------------------------------
    # Exception handling
    # -------------------------------------------------------------------

    def emit_op_save_exc_class(self, op, arglocs):
        res = arglocs[0]
        self.mc.gen_load_int(res.value, self.cpu.pos_exception())
        self.mc.LDW(res.value, res.value, 0)

    def emit_op_save_exception(self, op, arglocs):
        res = arglocs[0]
        mc = self.mc
        # Load exc value into result, then clear both exc globals
        mc.gen_load_int(r.scratch1.value, self.cpu.pos_exc_value())
        mc.LDW(res.value, r.scratch1.value, 0)
        # Clear exception
        mc.gen_load_int(r.scratch2.value, 0)
        mc.STW(r.scratch1.value, r.scratch2.value, 0)
        mc.gen_load_int(r.scratch1.value, self.cpu.pos_exception())
        mc.STW(r.scratch1.value, r.scratch2.value, 0)

    def emit_op_restore_exception(self, op, arglocs):
        exc_tp_loc, exc_val_loc = arglocs
        self.mc.gen_load_int(r.scratch1.value, self.cpu.pos_exc_value())
        self.mc.STW(r.scratch1.value, exc_val_loc.value, 0)
        self.mc.gen_load_int(r.scratch1.value, self.cpu.pos_exception())
        self.mc.STW(r.scratch1.value, exc_tp_loc.value, 0)

    # -------------------------------------------------------------------
    # Force token and memory error check
    # -------------------------------------------------------------------

    def emit_op_force_token(self, op, arglocs):
        res = arglocs[0]
        self.mc.MV(res.value, r.fp.value)

    def emit_op_check_memory_error(self, op, arglocs):
        loc = arglocs[0]
        self.propagate_memoryerror_if_reg_is_null(loc)

    def propagate_memoryerror_if_reg_is_null(self, loc):
        pd = r.scratch_pred.value
        self.mc.CMP_EQI(pd, loc.value, 0)
        # If loc == 0, jump to propagate_exception_path
        jmp_pos = self.mc.get_relative_pos()
        self.mc.TRAP0(0xDE)  # placeholder
        # P0=1 when loc==0 -> propagate exception
        self.mc.gen_load_int(r.scratch1.value, self.propagate_exception_path)
        self.mc.JUMPR(r.scratch1.value)
        # Patch: if P0=0 (non-null), skip the exception path
        ok_pos = self.mc.get_relative_pos()
        offset = ok_pos - jmp_pos
        pmc = OverwritingBuilder(self.mc, jmp_pos, INST_SIZE)
        pmc.J_IF_FALSE(pd, offset)  # skip propagation when P0=0 (not null)

    # -------------------------------------------------------------------
    # GC table loading
    # -------------------------------------------------------------------

    def emit_op_load_from_gc_table(self, op, arglocs):
        res = arglocs[0]
        index = op.getarg(0).getint()
        self.load_from_gc_table(res.value, index)

    def load_from_gc_table(self, reg, index):
        """Load a GC table entry into a register.

        The GC table is at the start of the code buffer.  Its absolute
        address (gc_table_addr) is only known after materialization, so
        we emit a fixed-size placeholder here and record the position
        for post-patching in _patch_gc_table_loads().

        Uses scratch2 (R15) for the address to avoid the gen_load_int
        self-clobber issue when scratch1 is used as both rd and
        internal temporary.
        """
        pos = self.mc.get_relative_pos()
        self._gc_table_load_patches.append((pos, reg, index))
        # Emit placeholder NOPs (patched later with gen_load_int)
        for i in range(self._GC_LOAD_INT_MAX):
            self.mc.NOP()
        # Load the value from the address in scratch2
        self.mc.LDW(reg, r.scratch2.value, 0)

    # -------------------------------------------------------------------
    # Array operations
    # -------------------------------------------------------------------

    def emit_op_zero_array(self, op, arglocs):
        if not arglocs:
            return
        base_loc, start_loc, size_loc = arglocs
        from rpython.jit.backend.llsupport.descr import unpack_arraydescr
        item_size, base_ofs, _ = unpack_arraydescr(op.getdescr())
        # Compute destination address
        dstaddr = r.scratch1.value
        if start_loc.is_imm():
            ofs = base_ofs + start_loc.value
            self.mc.ADDI(dstaddr, base_loc.value, ofs)
        else:
            self.mc.ADD(dstaddr, base_loc.value, start_loc.value)
            self.mc.ADDI(dstaddr, dstaddr, base_ofs)
        # Zero out the memory
        if size_loc.is_imm():
            total = size_loc.value
            if total <= 32:
                # Inline zero with store instructions
                self.mc.gen_load_int(r.scratch2.value, 0)
                dst_i = 0
                while total >= 4:
                    self.mc.STW(dstaddr, r.scratch2.value, dst_i)
                    dst_i += 4
                    total -= 4
                while total >= 2:
                    self.mc.STH(dstaddr, r.scratch2.value, dst_i)
                    dst_i += 2
                    total -= 2
                while total >= 1:
                    self.mc.STB(dstaddr, r.scratch2.value, dst_i)
                    dst_i += 1
                    total -= 1
                return
            else:
                self.mc.gen_load_int(r.scratch2.value, total)
                size_reg = r.scratch2.value
        else:
            size_reg = size_loc.value
        # Call memset(dst, 0, size) for large arrays (the regalloc
        # spilled caller-saved regs in prepare_op_zero_array).
        # Set R2 first: size_reg may be R0/R1; dstaddr is scratch1.
        if size_reg != r.r2.value:
            self.mc.MV(r.r2.value, size_reg)
        self.mc.gen_load_int(r.r1.value, 0)
        self.mc.MV(r.r0.value, dstaddr)
        self.mc.gen_load_int(r.scratch1.value, self.memset_addr)
        self.mc.CALLR(r.scratch1.value)

    # -------------------------------------------------------------------
    # Debug and portal frame operations (mostly no-ops)
    # -------------------------------------------------------------------

    def emit_op_jit_debug(self, op, arglocs):
        pass

    def emit_op_enter_portal_frame(self, op, arglocs):
        self.enter_portal_frame(op)

    def emit_op_leave_portal_frame(self, op, arglocs):
        self.leave_portal_frame(op)

    def emit_op_keepalive(self, op, arglocs):
        pass

    def emit_op_increment_debug_counter(self, op, arglocs):
        base_loc = arglocs[0]
        self.mc.LDW(r.scratch1.value, base_loc.value, 0)
        self.mc.ADDI(r.scratch1.value, r.scratch1.value, 1)
        self.mc.STW(base_loc.value, r.scratch1.value, 0)

    # -------------------------------------------------------------------
    # Jump / Finish / Label
    # -------------------------------------------------------------------

    def emit_op_jump(self, op, arglocs):
        from rpython.jit.metainterp.history import TargetToken
        descr = op.getdescr()
        assert isinstance(descr, TargetToken)
        target = descr._ll_loop_code
        if descr in self.target_tokens_currently_compiling:
            # Same compilation unit: both positions are buffer-relative
            offset = target - self.mc.get_relative_pos()
            self.mc.JUMP(offset)
        else:
            # Cross-compilation (bridge -> loop): target is absolute
            self.mc.gen_load_int(r.scratch1.value, target)
            self.mc.JUMPR(r.scratch1.value)

    def emit_op_finish(self, op, arglocs):
        from rpython.jit.metainterp.history import REF as BOX_REF
        from rpython.rlib.rarithmetic import r_uint

        base_ofs = self.cpu.get_baseofs_of_frame_field()
        if arglocs:
            loc = arglocs[0]
            if loc.is_core_reg():
                self.mc.store_to_jitframe(loc.value, base_ofs)
            elif loc.is_reg_pair():
                self.mc.store_pair_to_jitframe(loc.value, base_ofs)
            elif loc.is_imm():
                self.mc.gen_load_int(r.scratch1.value, loc.value)
                self.mc.store_to_jitframe(r.scratch1.value, base_ofs)
            elif loc.is_imm_float():
                # loc.addr points to the 8-byte constant
                self.mc.gen_load_int(r.scratch1.value, loc.addr)
                self.mc.load_double(r.d14.value, r.scratch1.value, 0)
                self.mc.store_pair_to_jitframe(r.d14.value, base_ofs)
            elif loc.is_stack():
                if loc.is_float():
                    self.mc.load_pair_from_jitframe(r.d14.value, loc.value)
                    self.mc.store_pair_to_jitframe(r.d14.value, base_ofs)
                else:
                    self.mc.load_from_jitframe(r.scratch1.value, loc.value)
                    self.mc.store_to_jitframe(r.scratch1.value, base_ofs)
            else:
                raise AssertionError("emit_op_finish: unsupported loc")

        # Store descr via GC ref table
        faildescrindex = self.get_gcref_from_faildescr(op.getdescr())
        self.load_from_gc_table(r.scratch1.value, faildescrindex)
        ofs = self.cpu.get_ofs_of_frame_field('jf_descr')
        self.mc.store_to_jitframe(r.scratch1.value, ofs)

        # Push gcmap so GC knows which slots are alive
        if op.numargs() > 0 and op.getarg(0).type == BOX_REF:
            if self._finish_gcmap:
                self._finish_gcmap[0] |= r_uint(1) << 0
                gcmap = self._finish_gcmap
            else:
                gcmap = self.gcmap_for_finish
            self.push_gcmap(self.mc, gcmap)
        elif self._finish_gcmap:
            gcmap = self._finish_gcmap
            self.push_gcmap(self.mc, gcmap)
        else:
            self.pop_gcmap(self.mc)

        # Return jitframe pointer
        self.mc.MV(r.r0.value, r.fp.value)
        self.gen_func_epilog()

    def emit_op_label(self, op, arglocs):
        # Labels are just position markers; nothing to emit
        pass

    # -------------------------------------------------------------------
    # Overflow operations
    #
    # Each sets P0 (scratch_pred) to 1 if NO overflow occurred, 0 if overflow.
    # The immediately following guard_no_overflow / guard_overflow reads P0.
    # -------------------------------------------------------------------

    def emit_op_int_add_ovf(self, op, arglocs):
        l0, l1, res = arglocs
        pd = r.scratch_pred.value
        s1 = r.scratch1.value
        s2 = r.scratch2.value
        # Compute result
        if l1.is_imm():
            self.mc.ADDI(res.value, l0.value, l1.value)
            # To detect overflow with XOR trick, we need l1 in a register
            self.mc.gen_load_int(s2, l1.value)
            l1_val = s2
        else:
            self.mc.ADD(res.value, l0.value, l1.value)
            l1_val = l1.value
        # Detect signed overflow:
        #   overflow iff (l0 ^ l1) has MSB=0 (same sign) AND
        #                (l0 ^ res) has MSB=1 (sign changed)
        # s1 = xor(l0, l1);  s2 = xor(l0, res)
        # s1 = ~s1;  s1 = s1 & s2  -> MSB=1 iff overflow
        self.mc.XOR(s1, l0.value, l1_val)
        self.mc.XOR(s2, l0.value, res.value)
        self.mc.NOT(s1, s1)
        self.mc.AND(s1, s1, s2)
        # P0 = (s1 > -1) -> P0=1 if s1>=0 (no overflow)
        self.mc.CMP_GTI(pd, s1, -1)

    def emit_op_int_sub_ovf(self, op, arglocs):
        l0, l1, res = arglocs
        pd = r.scratch_pred.value
        s1 = r.scratch1.value
        s2 = r.scratch2.value
        # Compute result
        if l1.is_imm():
            neg = -l1.value
            if -32768 <= neg <= 32767:
                self.mc.ADDI(res.value, l0.value, neg)
            else:
                self.mc.gen_load_int(s1, l1.value)
                self.mc.SUB(res.value, l0.value, s1)
            self.mc.gen_load_int(s2, l1.value)
            l1_val = s2
        else:
            self.mc.SUB(res.value, l0.value, l1.value)
            l1_val = l1.value
        # Detect signed overflow for subtraction:
        #   overflow iff (l0 ^ l1) has MSB=1 (different sign) AND
        #                (l0 ^ res) has MSB=1 (sign changed)
        # s1 = xor(l0, l1);  s2 = xor(l0, res)
        # s1 = s1 & s2  -> MSB=1 iff overflow
        self.mc.XOR(s1, l0.value, l1_val)
        self.mc.XOR(s2, l0.value, res.value)
        self.mc.AND(s1, s1, s2)
        # P0 = (s1 > -1) -> P0=1 if s1>=0 (no overflow)
        self.mc.CMP_GTI(pd, s1, -1)

    def emit_op_int_mul_ovf(self, op, arglocs):
        l0, l1, res = arglocs
        pd = r.scratch_pred.value
        s1 = r.scratch1.value
        s2 = r.scratch2.value
        # Use 32x32->64 signed multiply: Rdd = mpy(Rs, Rt)
        # D14 (R15:R14) is our scratch pair
        d14_even = r.d14.value   # R14 = low word
        d14_odd = d14_even + 1   # R15 = high word
        self.mc.M2_DPMPYSS_S0(d14_even, l0.value, l1.value)
        # Low 32 bits of result -> res
        self.mc.MV(res.value, d14_even)
        # Overflow if high word != sign extension of low word
        # s1 = high word (R15), s2 = asr(low, #31)
        self.mc.MV(s1, d14_odd)
        self.mc.S2_ASR_I_R(s2, d14_even, 31)
        # P0 = (s1 == s2) -> P0=1 if no overflow
        self.mc.CMP_EQ(pd, s1, s2)


def _get_scale(size):
    if size == 1:
        return 0
    elif size == 2:
        return 1
    elif size == 4:
        return 2
    elif size == 8:
        return 3
    else:
        raise AssertionError("unsupported size: %d" % size)


# ---------------------------------------------------------------------------
# Build the dispatch table mapping rop numbers to emit methods
# ---------------------------------------------------------------------------
def not_implemented_op(self, op, arglocs):
    raise NotImplementedError("operation %s not implemented" % op.getopname())


asm_operations = [not_implemented_op] * (rop._LAST + 1)

import itertools as _itertools
from rpython.jit.backend.hexagon.vector_ext import VectorAssemblerMixin as _VecMixin

for _name, _value in _itertools.chain(
        ResOpAssembler.__dict__.iteritems(),
        _VecMixin.__dict__.iteritems()):
    if _name.startswith('emit_op_'):
        opname = _name[len('emit_op_'):]
        num = getattr(rop, opname.upper(), None)
        if num is not None:
            asm_operations[num] = _value
