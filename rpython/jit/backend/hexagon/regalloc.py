"""
Register allocation for the Hexagon JIT backend.

Manages integer GPRs (R0-R13, R24, R25) and float register pairs
(D16, D18, D20, D22). For each IR operation, provides a prepare_op_*()
method that allocates registers and returns a list of locations.
"""

from rpython.jit.backend.hexagon import registers as r
from rpython.jit.backend.hexagon.arch import (
    WORD, DOUBLE_WORD, JITFRAME_FIXED_SIZE, SINT16_IMM_MIN, SINT16_IMM_MAX,
)
from rpython.jit.backend.hexagon.locations import (
    ImmLocation, RegisterLocation, RegisterPairLocation, StackLocation,
    ConstFloatLoc, get_fp_offset,
)
from rpython.jit.backend.hexagon.vector_ext import VectorRegallocMixin
from rpython.jit.backend.llsupport.jump import remap_frame_layout_mixed
from rpython.jit.backend.llsupport.descr import unpack_fielddescr
from rpython.jit.backend.llsupport.gcmap import allocate_gcmap
from rpython.jit.backend.llsupport.regalloc import (
    BaseRegalloc, RegisterManager, FrameManager, TempVar,
    compute_vars_longevity,
)
from rpython.jit.codewriter import longlong
from rpython.jit.metainterp.history import (
    ConstInt, ConstPtr, ConstFloat, Const,
    INT, REF, FLOAT,
)
from rpython.jit.metainterp.resoperation import rop
from rpython.rlib.rarithmetic import r_uint


class TempInt(TempVar):
    type = INT

    def __repr__(self):
        return "<TempInt at %s>" % (id(self),)


class TempPtr(TempVar):
    type = REF

    def __repr__(self):
        return "<TempPtr at %s>" % (id(self),)


class TempFloat(TempVar):
    type = FLOAT

    def __repr__(self):
        return "<TempFloat at %s>" % (id(self),)


# -----------------------------------------------------------------------
# Frame manager: maps IR boxes to JITFRAME stack positions
# -----------------------------------------------------------------------

class HexagonFrameManager(FrameManager):
    def __init__(self, base_ofs):
        FrameManager.__init__(self)
        self.base_ofs = base_ofs

    def frame_pos(self, loc, box_type):
        """Return the JITFRAME offset for a given position and type."""
        if box_type == FLOAT:
            # Float values need 2 words (register pair)
            return StackLocation(loc, get_fp_offset(self.base_ofs, loc), type=FLOAT)
        else:
            return StackLocation(loc, get_fp_offset(self.base_ofs, loc), type=INT)

    @staticmethod
    def frame_size(type):
        if type == FLOAT:
            return 2  # 64-bit = 2 x 32-bit slots
        return 1

    @staticmethod
    def get_loc_index(loc):
        assert loc.is_stack()
        return loc.position


# -----------------------------------------------------------------------
# Register managers
# -----------------------------------------------------------------------

class CoreRegisterManager(RegisterManager):
    """Manages integer GPR allocation: R0-R13, R24, R25."""
    all_regs = r.allocatable_registers
    box_types = [INT, REF]
    save_around_call_regs = r.caller_saved_registers
    FORBID_TEMP_BOXES = True

    def call_result_location(self, v):
        return r.r0

    def convert_to_imm(self, c):
        from rpython.rtyper.lltypesystem import lltype, rffi as _rffi
        if isinstance(c, ConstInt):
            val = _rffi.cast(lltype.Signed, c.value)
            return ImmLocation(val)
        else:
            assert isinstance(c, ConstPtr)
            return ImmLocation(_rffi.cast(lltype.Signed, c.value))

    def get_scratch_reg(self, type=INT, forbidden_vars=[], selected_reg=None):
        if selected_reg is not None:
            return selected_reg
        # Use R14 (scratch1) as the default scratch
        return r.scratch1


class FloatRegisterManager(RegisterManager):
    """Manages float register pair allocation: D16, D18, D20, D22."""
    all_regs = r.allocatable_float_pairs
    box_types = [FLOAT]
    save_around_call_regs = []  # All float pairs are callee-saved
    FORBID_TEMP_BOXES = True

    def call_result_location(self, v):
        # Float return value is in R1:R0 (D0), but we copy to a managed pair
        return r.d16

    def convert_to_imm(self, c):
        from rpython.rtyper.lltypesystem import rffi as _rffi
        assert isinstance(c, ConstFloat)
        adr = self.assembler.datablockwrapper.malloc_aligned(8, 8)
        x = c.getfloatstorage()
        _rffi.cast(_rffi.CArrayPtr(longlong.FLOATSTORAGE), adr)[0] = x
        return ConstFloatLoc(adr)

    def get_scratch_reg(self, type=FLOAT, forbidden_vars=[], selected_reg=None):
        if selected_reg is not None:
            return selected_reg
        # Use D14 (R15:R14) as float scratch
        return r.d14


class HVXRegisterManager(RegisterManager):
    """Manages HVX vector register allocation: V0-V15."""
    all_regs = r.allocatable_hvx_regs
    box_types = [INT, FLOAT]  # Vector ops produce INT or FLOAT typed results
    save_around_call_regs = r.allocatable_hvx_regs  # All caller-saved
    FORBID_TEMP_BOXES = True

    def call_result_location(self, v):
        return r.v0

    def get_scratch_reg(self, type=INT, forbidden_vars=[], selected_reg=None):
        if selected_reg is not None:
            return selected_reg
        # Use V16 as scratch (not in allocatable set)
        return r.v16


# -----------------------------------------------------------------------
# Helper functions for checking immediates
# -----------------------------------------------------------------------

def check_imm16(arg):
    """Check if arg is a ConstInt that fits in a 16-bit signed immediate."""
    if isinstance(arg, ConstInt):
        v = arg.getint()
        if SINT16_IMM_MIN <= v <= SINT16_IMM_MAX:
            return True
    return False


def check_imm10(arg):
    """Check if arg is a ConstInt that fits in a 10-bit signed immediate."""
    if isinstance(arg, ConstInt):
        v = arg.getint()
        if -(1 << 9) <= v <= (1 << 9) - 1:
            return True
    return False


def check_imm5(arg):
    """Check if arg is a ConstInt that fits in a 5-bit unsigned immediate."""
    if isinstance(arg, ConstInt):
        v = arg.getint()
        if 0 <= v <= 31:
            return True
    return False


# -----------------------------------------------------------------------
# Main register allocator
# -----------------------------------------------------------------------

class Regalloc(BaseRegalloc, VectorRegallocMixin):

    def __init__(self, assembler):
        self.assembler = assembler
        self.cpu = assembler.cpu
        self.frame_manager = None
        self.jump_target_descr = None

    def position(self):
        return self.rm.position

    def next_instruction(self):
        self.rm.next_instruction()
        self.fprm.next_instruction()

    def _prepare(self, inputargs, operations, allgcrefs):
        cpu = self.assembler.cpu
        self.fm = HexagonFrameManager(cpu.get_baseofs_of_frame_field())
        self.frame_manager = self.fm
        # Apply GC rewrites: transforms gc operations, inlines malloc, etc.
        operations = cpu.gc_ll_descr.rewrite_assembler(cpu, operations,
                                                       allgcrefs)
        longevity = compute_vars_longevity(inputargs, operations)
        self.longevity = longevity
        asm = self.assembler
        self.rm = CoreRegisterManager(
            self.longevity, self.fm, asm)
        self.fprm = FloatRegisterManager(
            self.longevity, self.fm, asm)
        self.hvxrm = HVXRegisterManager(
            self.longevity, self.fm, asm)
        return operations

    def prepare_bridge(self, inputargs, arglocs, operations, allgcrefs,
                       frame_info):
        operations = self._prepare(inputargs, operations, allgcrefs)
        self._update_bindings(arglocs, inputargs)
        return operations

    def _update_bindings(self, locs, inputargs):
        """Bind boxes to their locations (registers or stack slots).

        Called at bridge entry to set up the register state matching
        the guard's fail_locs.
        """
        used = {}
        for i in range(len(locs)):
            loc = locs[i]
            if loc is None:
                loc = r.fp
            arg = inputargs[i]
            if loc.is_vector_reg():
                self.hvxrm.reg_bindings[arg] = loc
                used[loc] = None
            elif loc.is_core_reg():
                self.rm.reg_bindings[arg] = loc
                used[loc] = None
            elif loc.is_reg_pair():
                self.fprm.reg_bindings[arg] = loc
                used[loc] = None
            else:
                assert loc.is_stack()
                self.fm.bind(arg, loc)
        # Collect the free registers
        self.rm.free_regs = []
        for reg in self.rm.all_regs:
            if reg not in used:
                self.rm.free_regs.append(reg)
        self.fprm.free_regs = []
        for reg in self.fprm.all_regs:
            if reg not in used:
                self.fprm.free_regs.append(reg)
        self.hvxrm.free_regs = []
        for reg in self.hvxrm.all_regs:
            if reg not in used:
                self.hvxrm.free_regs.append(reg)
        self.possibly_free_vars(list(inputargs))
        self.fm.finish_binding()

    def loc(self, v):
        """Return the location of a variable (register or stack)."""
        if v.is_vector():
            return self.hvxrm.loc(v)
        elif v.type == FLOAT:
            return self.fprm.loc(v)
        else:
            return self.rm.loc(v)

    def force_allocate_reg(self, v, forbidden_vars=[], selected_reg=None,
                           need_lower_byte=False):
        if v.is_vector():
            return self.hvxrm.force_allocate_reg(v, forbidden_vars,
                                                  selected_reg)
        elif v.type == FLOAT:
            return self.fprm.force_allocate_reg(v, forbidden_vars,
                                                 selected_reg)
        else:
            return self.rm.force_allocate_reg(v, forbidden_vars,
                                               selected_reg)

    def force_allocate_reg_or_cc(self, v):
        return self.force_allocate_reg(v)

    def make_sure_var_in_reg(self, var, forbidden_vars=[],
                             selected_reg=None, need_lower_byte=False):
        if var.is_vector():
            return self.hvxrm.make_sure_var_in_reg(var, forbidden_vars,
                                                    selected_reg,
                                                    need_lower_byte)
        elif var.type == FLOAT:
            return self.fprm.make_sure_var_in_reg(var, forbidden_vars,
                                                  selected_reg,
                                                  need_lower_byte)
        else:
            return self.rm.make_sure_var_in_reg(var, forbidden_vars,
                                                selected_reg,
                                                need_lower_byte)

    def possibly_free_var(self, v):
        if v.is_vector():
            self.hvxrm.possibly_free_var(v)
        elif v.type == FLOAT:
            self.fprm.possibly_free_var(v)
        else:
            self.rm.possibly_free_var(v)

    def possibly_free_vars(self, vars):
        for v in vars:
            if v is not None:
                self.possibly_free_var(v)

    def possibly_free_vars_for_op(self, op):
        for i in range(op.numargs()):
            self.possibly_free_var(op.getarg(i))

    def force_spill_var(self, v):
        if v.is_vector():
            self.hvxrm.force_spill_var(v)
        elif v.type == FLOAT:
            self.fprm.force_spill_var(v)
        else:
            self.rm.force_spill_var(v)

    def before_call(self, save_all_regs=False):
        """Save caller-saved registers before a C call."""
        self.rm.before_call(save_all_regs=save_all_regs)
        self.fprm.before_call(save_all_regs=save_all_regs)
        self.hvxrm.before_call(save_all_regs=save_all_regs)

    def after_call(self, v):
        """Allocate the result of a call."""
        if v.type == FLOAT:
            return self.fprm.after_call(v)
        else:
            return self.rm.after_call(v)

    def get_gcmap(self, forbidden_regs=[], noregs=False):
        """Build a GC map bitmap indicating which JITFRAME slots hold GC refs."""
        frame_depth = self.fm.get_frame_depth()
        gcmap = allocate_gcmap(self.assembler,
                               frame_depth, JITFRAME_FIXED_SIZE)
        for box, loc in self.rm.reg_bindings.iteritems():
            if loc in forbidden_regs:
                continue
            if box.type == REF and self.rm.is_still_alive(box):
                assert not noregs
                assert loc.is_core_reg()
                val = self.cpu.all_reg_indexes[loc.value]
                gcmap[val // WORD // 8] |= r_uint(1) << (val % (WORD * 8))
        for box, loc in self.fm.bindings_iteritems():
            if box.type == REF and self.rm.is_still_alive(box):
                assert loc.is_stack()
                val = loc.position + JITFRAME_FIXED_SIZE
                gcmap[val // WORD // 8] |= r_uint(1) << (val % (WORD * 8))
        return gcmap

    def get_final_frame_depth(self):
        return self.fm.get_frame_depth()

    # -------------------------------------------------------------------
    # prepare_op_* methods: allocate registers for each IR operation
    # -------------------------------------------------------------------

    # --- Integer binary operations ---

    def _prepare_int_binary_op(self, op):
        """Generic binary operation: res = op(arg0, arg1)."""
        boxes = op.getarglist()
        a0 = boxes[0]
        a1 = boxes[1]
        l0 = self.make_sure_var_in_reg(a0, boxes)
        if check_imm16(a1):
            l1 = ImmLocation(a1.getint())
        else:
            l1 = self.make_sure_var_in_reg(a1, boxes)
        self.possibly_free_vars_for_op(op)
        res = self.force_allocate_reg(op)
        return [l0, l1, res]

    def _prepare_int_binary_op_no_imm(self, op):
        """Binary operation that doesn't support immediates."""
        boxes = op.getarglist()
        a0 = boxes[0]
        a1 = boxes[1]
        l0 = self.make_sure_var_in_reg(a0, boxes)
        l1 = self.make_sure_var_in_reg(a1, boxes)
        self.possibly_free_vars_for_op(op)
        res = self.force_allocate_reg(op)
        return [l0, l1, res]

    def _prepare_int_unary_op(self, op):
        """Unary operation: res = op(arg0)."""
        a0 = op.getarg(0)
        l0 = self.make_sure_var_in_reg(a0)
        self.possibly_free_vars_for_op(op)
        res = self.force_allocate_reg(op)
        return [l0, res]

    prepare_op_int_add = _prepare_int_binary_op
    prepare_op_int_sub = _prepare_int_binary_op
    prepare_op_int_mul = _prepare_int_binary_op_no_imm
    prepare_op_int_and = _prepare_int_binary_op
    prepare_op_int_or = _prepare_int_binary_op
    prepare_op_int_xor = _prepare_int_binary_op
    prepare_op_int_lshift = _prepare_int_binary_op
    prepare_op_int_rshift = _prepare_int_binary_op
    prepare_op_uint_rshift = _prepare_int_binary_op
    prepare_op_int_neg = _prepare_int_unary_op
    prepare_op_int_invert = _prepare_int_unary_op
    prepare_op_uint_mul_high = _prepare_int_binary_op_no_imm

    def prepare_op_int_signext(self, op):
        a0 = op.getarg(0)
        a1 = op.getarg(1)
        l0 = self.make_sure_var_in_reg(a0)
        l1 = ImmLocation(a1.getint())
        self.possibly_free_vars_for_op(op)
        res = self.force_allocate_reg(op)
        return [l0, l1, res]

    prepare_op_nursery_ptr_increment = _prepare_int_binary_op

    # --- Integer comparisons ---

    def _prepare_int_cmp(self, op):
        """Integer comparison: res = cmp(arg0, arg1)."""
        boxes = op.getarglist()
        a0 = boxes[0]
        a1 = boxes[1]
        l0 = self.make_sure_var_in_reg(a0, boxes)
        l1 = self.make_sure_var_in_reg(a1, boxes)
        self.possibly_free_vars_for_op(op)
        res = self.force_allocate_reg(op)
        return [l0, l1, res]

    prepare_op_int_lt = _prepare_int_cmp
    prepare_op_int_le = _prepare_int_cmp
    prepare_op_int_eq = _prepare_int_cmp
    prepare_op_int_ne = _prepare_int_cmp
    prepare_op_int_gt = _prepare_int_cmp
    prepare_op_int_ge = _prepare_int_cmp
    prepare_op_uint_lt = _prepare_int_cmp
    prepare_op_uint_le = _prepare_int_cmp
    prepare_op_uint_gt = _prepare_int_cmp
    prepare_op_uint_ge = _prepare_int_cmp

    prepare_op_ptr_eq = _prepare_int_cmp
    prepare_op_ptr_ne = _prepare_int_cmp
    prepare_op_instance_ptr_eq = _prepare_int_cmp
    prepare_op_instance_ptr_ne = _prepare_int_cmp

    # --- Float operations ---

    def _prepare_float_binary_op(self, op):
        """Float binary operation: res = op(arg0, arg1)."""
        a0 = op.getarg(0)
        a1 = op.getarg(1)
        l0 = self.fprm.loc(a0)
        l1 = self.fprm.loc(a1)
        self.possibly_free_vars_for_op(op)
        res = self.fprm.force_allocate_reg(op)
        return [l0, l1, res]

    def _prepare_float_unary_op(self, op):
        """Float unary operation: res = op(arg0)."""
        a0 = op.getarg(0)
        l0 = self.fprm.loc(a0)
        self.possibly_free_vars_for_op(op)
        res = self.fprm.force_allocate_reg(op)
        return [l0, res]

    prepare_op_float_add = _prepare_float_binary_op
    prepare_op_float_sub = _prepare_float_binary_op
    prepare_op_float_mul = _prepare_float_binary_op
    prepare_op_float_truediv = _prepare_float_binary_op
    prepare_op_float_neg = _prepare_float_unary_op
    prepare_op_float_abs = _prepare_float_unary_op

    # --- Float comparisons ---

    def _prepare_float_cmp(self, op):
        a0 = op.getarg(0)
        a1 = op.getarg(1)
        l0 = self.fprm.loc(a0)
        l1 = self.fprm.loc(a1)
        self.possibly_free_vars_for_op(op)
        res = self.rm.force_allocate_reg(op)  # result is an int (0 or 1)
        return [l0, l1, res]

    prepare_op_float_lt = _prepare_float_cmp
    prepare_op_float_le = _prepare_float_cmp
    prepare_op_float_eq = _prepare_float_cmp
    prepare_op_float_ne = _prepare_float_cmp
    prepare_op_float_gt = _prepare_float_cmp
    prepare_op_float_ge = _prepare_float_cmp

    # --- Conversions ---

    def prepare_op_cast_float_to_int(self, op):
        a0 = op.getarg(0)
        l0 = self.fprm.loc(a0)
        self.possibly_free_vars_for_op(op)
        res = self.rm.force_allocate_reg(op)
        return [l0, res]

    def prepare_op_cast_int_to_float(self, op):
        a0 = op.getarg(0)
        l0 = self.rm.loc(a0)
        self.possibly_free_vars_for_op(op)
        res = self.fprm.force_allocate_reg(op)
        return [l0, res]

    prepare_op_cast_float_to_singlefloat = prepare_op_cast_float_to_int
    prepare_op_cast_singlefloat_to_float = prepare_op_cast_int_to_float

    def prepare_op_convert_float_bytes_to_longlong(self, op):
        a0 = op.getarg(0)
        l0 = self.fprm.loc(a0)
        self.possibly_free_vars_for_op(op)
        res = self.fprm.force_allocate_reg(op)
        return [l0, res]

    def prepare_op_convert_longlong_bytes_to_float(self, op):
        a0 = op.getarg(0)
        l0 = self.fprm.loc(a0)
        self.possibly_free_vars_for_op(op)
        res = self.fprm.force_allocate_reg(op)
        return [l0, res]

    # --- Guards ---
    #
    # NOTE: guard prepare methods must NOT call possibly_free_vars_for_op,
    # because _prepare_guard_arglocs needs to look up failargs which may
    # share boxes with the guard's own args. The caller (_walk_operations)
    # calls possibly_free_vars_for_op after emit.

    def _prepare_guard_arglocs(self, op):
        """Build the guard portion of arglocs: [frame_depth, fail_loc0, ...]
        where fail_locs are the current locations of the failargs."""
        arglocs = [None] * (len(op.getfailargs()) + 1)
        arglocs[0] = ImmLocation(self.fm.get_frame_depth())
        failargs = op.getfailargs()
        for i in range(len(failargs)):
            if failargs[i]:
                arglocs[i + 1] = self.loc(failargs[i])
        return arglocs

    def prepare_op_guard_true(self, op):
        a0 = op.getarg(0)
        l0 = self.make_sure_var_in_reg(a0)
        return [l0] + self._prepare_guard_arglocs(op)

    prepare_op_guard_false = prepare_op_guard_true
    prepare_op_guard_nonnull = prepare_op_guard_true
    prepare_op_guard_isnull = prepare_op_guard_true

    def prepare_op_guard_value(self, op):
        boxes = op.getarglist()
        a0 = boxes[0]
        a1 = boxes[1]
        l0 = self.make_sure_var_in_reg(a0, boxes)
        l1 = self.make_sure_var_in_reg(a1, boxes)
        return [l0, l1] + self._prepare_guard_arglocs(op)

    def prepare_op_guard_class(self, op):
        a0 = op.getarg(0)
        a1 = op.getarg(1)
        l0 = self.rm.make_sure_var_in_reg(a0)
        l1 = self.loc(a1)
        return [l0, l1] + self._prepare_guard_arglocs(op)

    def prepare_op_guard_nonnull_class(self, op):
        a0 = op.getarg(0)
        a1 = op.getarg(1)
        l0 = self.rm.make_sure_var_in_reg(a0)
        l1 = self.loc(a1)
        return [l0, l1] + self._prepare_guard_arglocs(op)

    def prepare_op_guard_gc_type(self, op):
        a0 = op.getarg(0)
        a1 = op.getarg(1)
        l0 = self.rm.make_sure_var_in_reg(a0)
        l1 = ImmLocation(a1.getint())
        return [l0, l1] + self._prepare_guard_arglocs(op)

    def prepare_op_guard_is_object(self, op):
        a0 = op.getarg(0)
        l0 = self.rm.make_sure_var_in_reg(a0)
        return [l0] + self._prepare_guard_arglocs(op)

    def prepare_op_guard_subclass(self, op):
        a0 = op.getarg(0)
        a1 = op.getarg(1)
        l0 = self.rm.make_sure_var_in_reg(a0)
        l1 = ImmLocation(a1.getint())
        return [l0, l1] + self._prepare_guard_arglocs(op)

    def prepare_op_guard_no_exception(self, op):
        return self._prepare_guard_arglocs(op)

    prepare_op_guard_not_invalidated = prepare_op_guard_no_exception

    def prepare_op_guard_no_overflow(self, op):
        return self._prepare_guard_arglocs(op)

    prepare_op_guard_overflow = prepare_op_guard_no_overflow

    def prepare_op_guard_not_forced(self, op):
        return self._prepare_guard_arglocs(op)

    prepare_op_guard_not_forced_2 = prepare_op_guard_not_forced

    def prepare_op_guard_exception(self, op):
        a0 = op.getarg(0)
        l0 = self.make_sure_var_in_reg(a0)
        res = self.force_allocate_reg(op)
        return [l0, res] + self._prepare_guard_arglocs(op)

    # --- Memory operations ---

    def prepare_op_getfield_gc_i(self, op):
        a0 = op.getarg(0)
        l0 = self.make_sure_var_in_reg(a0)
        self.possibly_free_vars_for_op(op)
        res = self.rm.force_allocate_reg(op)
        ofs, _, _ = unpack_fielddescr(op.getdescr())
        return [l0, ImmLocation(ofs), res]

    prepare_op_getfield_gc_r = prepare_op_getfield_gc_i

    def prepare_op_getfield_gc_f(self, op):
        a0 = op.getarg(0)
        l0 = self.make_sure_var_in_reg(a0)
        self.possibly_free_vars_for_op(op)
        res = self.fprm.force_allocate_reg(op)
        ofs, _, _ = unpack_fielddescr(op.getdescr())
        return [l0, ImmLocation(ofs), res]

    def prepare_op_setfield_gc(self, op):
        boxes = op.getarglist()
        a0 = boxes[0]
        a1 = boxes[1]
        l0 = self.make_sure_var_in_reg(a0, boxes)
        l1 = self.make_sure_var_in_reg(a1, boxes)
        ofs, _, _ = unpack_fielddescr(op.getdescr())
        return [l0, ImmLocation(ofs), l1]

    # --- GC load/store operations ---
    # Note: getarrayitem_gc, setarrayitem_gc, raw_load, raw_store,
    # strgetitem, strsetitem, etc. are all rewritten by the GC rewrite
    # pass into gc_load/gc_store variants before reaching the backend.

    def prepare_op_gc_store(self, op):
        base_loc = self.rm.make_sure_var_in_reg(op.getarg(0), op.getarglist())
        ofs = op.getarg(1).getint()
        value_loc = self.make_sure_var_in_reg(op.getarg(2), op.getarglist())
        size = op.getarg(3).getint()
        ofs_loc = ImmLocation(ofs)
        return [value_loc, base_loc, ofs_loc, ImmLocation(size)]

    def _prepare_op_gc_load(self, op):
        a0 = op.getarg(0)
        ofs = op.getarg(1).getint()
        nsize = op.getarg(2).getint()
        base_loc = self.rm.make_sure_var_in_reg(a0)
        ofs_loc = ImmLocation(ofs)
        self.possibly_free_vars_for_op(op)
        if abs(nsize) <= 4:
            res_loc = self.rm.force_allocate_reg(op)
        else:
            res_loc = self.fprm.force_allocate_reg(op)
        return [base_loc, ofs_loc, res_loc, ImmLocation(nsize)]

    prepare_op_gc_load_i = _prepare_op_gc_load
    prepare_op_gc_load_r = _prepare_op_gc_load
    prepare_op_gc_load_f = _prepare_op_gc_load

    def prepare_op_gc_store_indexed(self, op):
        # args: base, index, value, scale, ofs, size
        boxes = op.getarglist()
        base_loc = self.rm.make_sure_var_in_reg(boxes[0], boxes)
        index_loc = self.rm.make_sure_var_in_reg(boxes[1], boxes)
        value_loc = self.rm.make_sure_var_in_reg(boxes[2], boxes)
        assert boxes[3].getint() == 1  # scale must be 1
        ofs_loc = ImmLocation(boxes[4].getint())
        size_loc = ImmLocation(boxes[5].getint())
        return [value_loc, base_loc, index_loc, ofs_loc, size_loc]

    def _prepare_op_gc_load_indexed(self, op):
        # args: base, index, scale, offset, nsize
        boxes = op.getarglist()
        base_loc = self.rm.make_sure_var_in_reg(boxes[0], boxes)
        index_loc = self.rm.make_sure_var_in_reg(boxes[1], boxes)
        assert boxes[2].getint() == 1  # scale must be 1
        ofs_loc = ImmLocation(boxes[3].getint())
        nsize = boxes[4].getint()
        self.possibly_free_vars_for_op(op)
        if abs(nsize) <= 4:
            res_loc = self.rm.force_allocate_reg(op)
        else:
            res_loc = self.fprm.force_allocate_reg(op)
        return [base_loc, index_loc, ofs_loc, res_loc, ImmLocation(nsize)]

    prepare_op_gc_load_indexed_i = _prepare_op_gc_load_indexed
    prepare_op_gc_load_indexed_r = _prepare_op_gc_load_indexed
    prepare_op_gc_load_indexed_f = _prepare_op_gc_load_indexed

    # --- Write barrier ---

    def prepare_op_cond_call_gc_wb(self, op):
        N = op.numargs()
        args = op.getarglist()
        arglocs = [self.rm.make_sure_var_in_reg(op.getarg(i), args)
                    for i in range(N)]
        return arglocs

    def prepare_op_cond_call_gc_wb_array(self, op):
        # For array write barrier, all args (including the index) must be
        # in registers because the card marking code mutates them.
        args = op.getarglist()
        arglocs = []
        for i in range(op.numargs()):
            arg = op.getarg(i)
            loc = self.rm.make_sure_var_in_reg(arg, args)
            if not loc.is_core_reg():
                # Constant: load into a temp register
                tmp = TempInt()
                reg = self.rm.force_allocate_reg(tmp)
                self.assembler.regalloc_mov(loc, reg)
                self.rm.possibly_free_var(tmp)
                loc = reg
            arglocs.append(loc)
        return arglocs

    # --- Nursery allocation ---

    def prepare_op_call_malloc_nursery(self, op):
        self.rm.force_allocate_reg(op, selected_reg=r.r0)
        t = TempInt()
        self.rm.force_allocate_reg(t, selected_reg=r.r1)
        return []

    def prepare_op_call_malloc_nursery_varsize_frame(self, op):
        a0 = op.getarg(0)
        size_loc = self.make_sure_var_in_reg(a0)
        self.possibly_free_vars_for_op(op)
        self.rm.force_allocate_reg(op, selected_reg=r.r0)
        t = TempInt()
        self.rm.force_allocate_reg(t, selected_reg=r.r1)
        return [size_loc]

    def prepare_op_call_malloc_nursery_varsize(self, op):
        a0 = op.getarg(0)  # length
        length_loc = self.make_sure_var_in_reg(a0)
        self.possibly_free_vars_for_op(op)
        self.rm.force_allocate_reg(op, selected_reg=r.r0)
        t = TempInt()
        self.rm.force_allocate_reg(t, selected_reg=r.r1)
        return [length_loc]

    # --- Exception handling ---

    def prepare_op_save_exception(self, op):
        res = self.force_allocate_reg(op)
        return [res]

    prepare_op_save_exc_class = prepare_op_save_exception

    def prepare_op_restore_exception(self, op):
        boxes = op.getarglist()
        exc_tp_loc = self.rm.make_sure_var_in_reg(boxes[0], boxes)
        exc_val_loc = self.rm.make_sure_var_in_reg(boxes[1], boxes)
        return [exc_tp_loc, exc_val_loc]

    # --- Force token and memory error check ---

    def prepare_op_force_token(self, op):
        res = self.force_allocate_reg(op)
        return [res]

    def prepare_op_check_memory_error(self, op):
        l0 = self.rm.make_sure_var_in_reg(op.getarg(0))
        return [l0]

    # --- GC table ---

    def prepare_op_load_from_gc_table(self, op):
        res = self.force_allocate_reg(op)
        return [res]

    # --- Array operations ---

    def prepare_op_zero_array(self, op):
        return []

    # --- Debug and portal frames ---

    def prepare_op_jit_debug(self, op):
        return []

    def prepare_op_enter_portal_frame(self, op):
        return []

    def prepare_op_leave_portal_frame(self, op):
        return []

    def prepare_op_keepalive(self, op):
        return []

    def prepare_op_increment_debug_counter(self, op):
        a0 = op.getarg(0)
        base_loc = self.rm.make_sure_var_in_reg(a0)
        return [base_loc]

    # --- Jump / Finish / Label ---

    def prepare_op_jump(self, op):
        assert self.jump_target_descr is None
        descr = op.getdescr()
        assert isinstance(descr, TargetToken)
        self.jump_target_descr = descr
        arglocs = descr._hexagon_arglocs

        # Remap frame layout: move values from current locations to target
        # Separate core and float locations
        src_core = []
        dst_core = []
        src_float = []
        dst_float = []
        for i in range(op.numargs()):
            box = op.getarg(i)
            src_loc = self.loc(box)
            dst_loc = arglocs[i]
            if box.type == FLOAT:
                src_float.append(src_loc)
                dst_float.append(dst_loc)
            else:
                src_core.append(src_loc)
                dst_core.append(dst_loc)

        remap_frame_layout_mixed(self.assembler,
                                 src_core, dst_core, r.scratch1,
                                 src_float, dst_float, r.scratch2, WORD)
        return []

    def prepare_op_finish(self, op):
        locs = []
        for i in range(op.numargs()):
            locs.append(self.loc(op.getarg(i)))
        return locs

    def prepare_op_label(self, op):
        descr = op.getdescr()
        assert isinstance(descr, TargetToken)
        inputargs = op.getarglist()
        arglocs = [None] * len(inputargs)

        # Spill boxes that aren't used anymore in the loop body
        position = self.rm.position
        for arg in inputargs:
            assert not isinstance(arg, Const)
            if self.longevity[arg].is_last_real_use_before(position):
                self.force_spill_var(arg)

        # Record current locations for each arg
        for i in range(len(inputargs)):
            arg = inputargs[i]
            loc = self.loc(arg)
            arglocs[i] = loc
            if loc.is_core_reg() or loc.is_reg_pair():
                self.fm.mark_as_free(arg)

        # Store arch-specific location info on the target token
        descr._hexagon_arglocs = arglocs
        descr._hexagon_clt = self.assembler.current_clt
        descr._ll_loop_code = self.assembler.mc.get_relative_pos()
        self.assembler.target_tokens_currently_compiling[descr] = None
        return arglocs

    # --- Same-as (no-op moves) ---

    def prepare_op_same_as_i(self, op):
        a0 = op.getarg(0)
        l0 = self.make_sure_var_in_reg(a0)
        self.possibly_free_vars_for_op(op)
        res = self.force_allocate_reg(op)
        return [l0, res]

    prepare_op_same_as_r = prepare_op_same_as_i
    prepare_op_cast_ptr_to_int = prepare_op_same_as_i
    prepare_op_cast_int_to_ptr = prepare_op_same_as_i

    def prepare_op_same_as_f(self, op):
        a0 = op.getarg(0)
        l0 = self.fprm.loc(a0)
        self.possibly_free_vars_for_op(op)
        res = self.fprm.force_allocate_reg(op)
        return [l0, res]

    # --- Calls ---

    def _prepare_op_call(self, op, save_all_regs=False, first_arg_index=1):
        from rpython.jit.backend.llsupport.descr import CallDescr
        calldescr = op.getdescr()
        assert isinstance(calldescr, CallDescr)
        assert len(calldescr.arg_classes) == op.numargs() - first_arg_index

        # arglocs format: [resloc, size, sign, funcloc, arg0, arg1, ...]
        locs = [None] * (op.numargs() + 3)
        for i in range(op.numargs()):
            locs[i + 3] = self.loc(op.getarg(i))

        size = calldescr.get_result_size()
        sign = calldescr.is_result_signed()
        locs[1] = ImmLocation(size)
        locs[2] = ImmLocation(sign)

        effectinfo = calldescr.get_extra_info()
        if save_all_regs:
            gc_level = 2
        elif effectinfo is None or effectinfo.check_can_collect():
            gc_level = 1
        else:
            gc_level = 0

        # Allocate result register and possibly save regs
        if gc_level == 2:
            self.rm.before_call(save_all_regs=True)
            self.fprm.before_call(save_all_regs=True)
        elif gc_level == 1:
            self.fprm.before_call()
            # With shadow stack GC, spill all GC ref registers so the
            # GC can find them via the jitframe (noregs mode).
            if self.cpu.gc_ll_descr.gcrootmap:
                self.rm.before_call(save_all_regs=2)
            else:
                self.rm.before_call()
        self.possibly_free_vars_for_op(op)
        if op.type == FLOAT:
            resloc = self.fprm.after_call(op)
        elif op.type != 'v':
            resloc = self.rm.after_call(op)
        else:
            resloc = None
        locs[0] = resloc
        return locs

    def prepare_op_call_i(self, op):
        return self._prepare_op_call(op)
    prepare_op_call_r = prepare_op_call_i
    prepare_op_call_f = prepare_op_call_i
    prepare_op_call_n = prepare_op_call_i

    # --- Int operations with overflow ---
    # For add_ovf/sub_ovf, the result register must NOT alias the inputs
    # because the software overflow detection reads the original input
    # values AFTER the add/sub has written the result.

    def _prepare_int_binary_op_ovf(self, op):
        boxes = op.getarglist()
        a0 = boxes[0]
        a1 = boxes[1]
        l0 = self.make_sure_var_in_reg(a0, boxes)
        if check_imm16(a1):
            l1 = ImmLocation(a1.getint())
        else:
            l1 = self.make_sure_var_in_reg(a1, boxes)
        # Allocate result BEFORE freeing inputs so that l0/l1 registers
        # stay reserved and res gets a distinct register.
        res = self.force_allocate_reg(op)
        self.possibly_free_vars_for_op(op)
        return [l0, l1, res]

    prepare_op_int_add_ovf = _prepare_int_binary_op_ovf
    prepare_op_int_sub_ovf = _prepare_int_binary_op_ovf
    prepare_op_int_mul_ovf = _prepare_int_binary_op_no_imm

    # --- Boolean / bit operations ---

    prepare_op_int_is_true = _prepare_int_unary_op
    prepare_op_int_is_zero = _prepare_int_unary_op
    prepare_op_int_force_ge_zero = _prepare_int_unary_op

    # --- Conditional calls ---

    def _prepare_op_cond_call(self, op):
        assert 2 <= op.numargs() <= 4 + 2

        func_addr = op.getarg(1)
        assert isinstance(func_addr, Const)

        # Move function arguments to argument registers
        allocated_arg_vars = []
        for i in range(2, op.numargs()):
            reg = r.argument_regs[i - 2]
            arg = op.getarg(i)
            assert arg.type != FLOAT
            self.make_sure_var_in_reg(arg, allocated_arg_vars,
                                      selected_reg=reg)
            allocated_arg_vars.append(arg)

        # Move condition to a register
        argloc = self.make_sure_var_in_reg(op.getarg(0), allocated_arg_vars)

        if op.type == 'v':
            # Plain COND_CALL: call when cond != 0
            return [argloc]
        else:
            # COND_CALL_VALUE_I/R: call when cond == 0, return result
            args = op.getarglist()
            resloc = self.rm.force_result_in_reg(op, args[0],
                                                 forbidden_vars=args[2:])
            return [argloc, resloc]

    prepare_op_cond_call = _prepare_op_cond_call
    prepare_op_cond_call_value_i = _prepare_op_cond_call
    prepare_op_cond_call_value_r = _prepare_op_cond_call

    # --- Paired guard/op dispatch ---

    def prepare_guard_op_guard_not_forced(self, op, guard_op):
        """Prepare the paired call + guard_not_forced operations.

        Returns (arglocs, num_arglocs) where arglocs contains
        the call op's locs followed by the guard op's guard arglocs.
        """
        if rop.is_call_release_gil(op.getopnum()):
            arglocs = self._prepare_op_call(op, save_all_regs=True,
                                            first_arg_index=2)
        elif rop.is_call_assembler(op.getopnum()):
            locs = self.locs_for_call_assembler(op)
            resloc = self._call(op)
            arglocs = locs + [resloc]
        else:
            assert rop.is_call_may_force(op.getopnum())
            arglocs = self._prepare_op_call(op, save_all_regs=True)
        num_arglocs = len(arglocs)
        guard_arglocs = self._prepare_guard_arglocs(guard_op)
        return arglocs + guard_arglocs, num_arglocs

    def _call(self, op):
        """Spill regs for a call that may force, allocate result."""
        self.fprm.before_call(save_all_regs=True)
        save_all = 2  # can force
        if self.cpu.gc_ll_descr.gcrootmap:
            save_all = 2
        self.rm.before_call(save_all_regs=save_all)
        self.possibly_free_vars_for_op(op)
        if op.type == FLOAT:
            return self.fprm.after_call(op)
        elif op.type != 'v':
            return self.rm.after_call(op)
        else:
            return None

    # NOTE: call_may_force and call_release_gil are NOT in the dispatch table.
    # They are always handled through prepare_guard_op_guard_not_forced.
    # Also, call_assembler is handled the same way.


# Needed for prepare_op_jump
from rpython.jit.metainterp.compile import TargetToken


# Build dispatch tables (indexed by opnum) at module level.
# RPython doesn't support 3-arg getattr, so we pre-build these tables
# for the assembler's _walk_operations.

def _notimplemented_prepare(self, op):
    from rpython.rtyper.lltypesystem.lloperation import llop
    from rpython.rtyper.lltypesystem import lltype
    llop.debug_print(lltype.Void,
                     "[Hexagon/regalloc] %s not implemented" % op.getopname())
    raise NotImplementedError(op)

prepare_operations = [_notimplemented_prepare] * (rop._LAST + 1)

for _key, _value in rop.__dict__.items():
    _key = _key.lower()
    if _key.startswith('_'):
        continue
    _methname = 'prepare_op_%s' % _key
    if hasattr(Regalloc, _methname):
        _func = getattr(Regalloc, _methname)
        if hasattr(_func, 'im_func'):
            _func = _func.im_func
        prepare_operations[_value] = _func
