"""
Calling convention implementation for the Hexagon JIT backend.

Hexagon ABI:
- Arguments: R0-R5 (6 registers), then stack (8-byte aligned)
- Float arguments: passed in register pairs (R1:R0, R3:R2, R5:R4)
- Return value: R0 (32-bit int) or R1:R0 (64-bit/float)
- Caller-saved: R0-R15
- Callee-saved: R16-R27
- Stack pointer: R29 (SP), 8-byte aligned
- Link register: R31 (LR)
"""

from rpython.jit.backend.llsupport.callbuilder import AbstractCallBuilder
from rpython.jit.backend.llsupport.jump import remap_frame_layout
from rpython.jit.backend.llsupport import llerrno
from rpython.jit.backend.hexagon import registers as r
from rpython.jit.backend.hexagon.arch import WORD, DOUBLE_WORD, ABI_STACK_ALIGN, INST_SIZE
from rpython.jit.backend.hexagon.codebuilder import OverwritingBuilder
from rpython.jit.metainterp.history import FLOAT
from rpython.rlib.objectmodel import we_are_translated
from rpython.rtyper.lltypesystem import rffi


class HexagonCallBuilder(AbstractCallBuilder):

    def __init__(self, assembler, fnloc, arglocs, resloc, restype, ressize):
        AbstractCallBuilder.__init__(self, assembler, fnloc, arglocs,
                                     resloc, restype, ressize)
        self.current_sp = 0

    def prepare_arguments(self):
        arglocs = self.arglocs

        non_float_locs = []
        non_float_regs = []
        float_locs = []
        float_regs = []
        stack_locs = []

        # Hexagon uses R0-R5 for integer arguments
        # Float arguments use register pairs: D0=R1:R0, D2=R3:R2, D4=R5:R4
        free_regs = r.argument_regs[:]  # copy
        free_regs.reverse()  # [R5,R4,R3,R2,R1,R0]
        free_float_pairs = [r.d4, r.d2, r.d0]  # pairs for float args

        stack_adj_offset = 0
        for arg in arglocs:
            if arg.type == FLOAT:
                if free_float_pairs:
                    float_locs.append(arg)
                    float_regs.append(free_float_pairs.pop())
                    # Remove the corresponding integer regs from free list
                    # (float pairs consume integer reg slots)
                    pair = float_regs[-1]
                    if r.registers[pair.even_reg()] in free_regs:
                        free_regs.remove(r.registers[pair.even_reg()])
                    if r.registers[pair.odd_reg()] in free_regs:
                        free_regs.remove(r.registers[pair.odd_reg()])
                else:
                    stack_adj_offset += DOUBLE_WORD
                    stack_locs.append(arg)
            else:
                if free_regs:
                    non_float_locs.append(arg)
                    non_float_regs.append(free_regs.pop())
                else:
                    stack_adj_offset += WORD
                    stack_locs.append(arg)

        if stack_locs:
            # Align stack adjustment
            stack_adj_offset = ((stack_adj_offset + ABI_STACK_ALIGN - 1)
                                // ABI_STACK_ALIGN * ABI_STACK_ALIGN)
            self.mc.ADDI(r.sp.value, r.sp.value, -stack_adj_offset)
            self.current_sp = stack_adj_offset

            # Spill overflow arguments to stack
            sp_offset = 0
            for loc in stack_locs:
                self.asm.mov_loc_to_raw_stack(loc, sp_offset)
                sp_offset += DOUBLE_WORD if loc.type == FLOAT else WORD

        # Load the function address into scratch register
        if self.fnloc.is_core_reg():
            self.mc.MV(r.scratch1.value, self.fnloc.value)
        elif self.fnloc.is_imm():
            self.mc.gen_load_int(r.scratch1.value, self.fnloc.value)
        else:
            assert self.fnloc.is_stack()
            self.mc.load_from_jitframe(r.scratch1.value, self.fnloc.value)
        self.fnloc = r.scratch1

        # Move argument values to argument registers
        scratch_reg = r.scratch2
        remap_frame_layout(self.asm, non_float_locs, non_float_regs,
                           scratch_reg)

    def push_gcmap(self):
        noregs = self.asm.cpu.gc_ll_descr.is_shadow_stack()
        gcmap = self.asm._regalloc.get_gcmap([r.r0], noregs=noregs)
        self.asm.push_gcmap(self.mc, gcmap)

    def pop_gcmap(self):
        self.asm._reload_frame_if_necessary(self.mc)
        self.asm.pop_gcmap(self.mc)

    def emit_raw_call(self):
        assert self.fnloc is r.scratch1
        self.mc.CALLR(self.fnloc.value)

    def restore_stack_pointer(self):
        if self.current_sp == 0:
            return
        self.mc.ADDI(r.sp.value, r.sp.value, self.current_sp)
        self.current_sp = 0

    def load_result(self):
        resloc = self.resloc
        if resloc is not None and resloc.is_core_reg():
            self._ensure_result_bit_extension(resloc, self.ressize,
                                              self.ressign)

    def _ensure_result_bit_extension(self, resloc, size, signed):
        if size == WORD:
            return
        if size == 2:
            if signed:
                self.mc.SXTH(resloc.value, resloc.value)
            else:
                self.mc.ZXTH(resloc.value, resloc.value)
        elif size == 1:
            if signed:
                self.mc.SXTB(resloc.value, resloc.value)
            else:
                self.mc.ZXTB(resloc.value, resloc.value)

    def call_releasegil_addr_and_move_real_arguments(self, fastgil):
        """Release the GIL before calling a C function."""
        assert self.is_call_release_gil
        assert not self.asm._is_asmgcc()

        gcrootmap = self.asm.cpu.gc_ll_descr.gcrootmap
        if gcrootmap:
            rst = gcrootmap.get_root_stack_top_addr()
            self.mc.gen_load_int(r.scratch1.value, rst)
            self.mc.LDW(r.shadow_old.value, r.scratch1.value, 0)

        # Save the thread ID and release the GIL
        self.mc.gen_load_int(r.scratch1.value, fastgil)
        self.mc.LDW(r.thread_id.value, r.scratch1.value, 0)
        # Store 0 to release the GIL
        self.mc.TFRSI(r.scratch2.value, 0)
        self.mc.STW(r.scratch1.value, r.scratch2.value, 0)

        if not we_are_translated():
            # For testing: mark that we shouldn't access jfp
            self.mc.ADDI(r.fp.value, r.fp.value, 1)

    def move_real_result_and_call_reacqgil_addr(self, fastgil):
        """Reacquire the GIL after a C call returns."""
        # Try to reacquire: load fastgil, check if 0 (free)
        self.mc.gen_load_int(r.scratch1.value, fastgil)
        self.mc.LDW(r.scratch2.value, r.scratch1.value, 0)

        pd = r.scratch_pred.value
        self.mc.CMP_EQI(pd, r.scratch2.value, 0)
        # If fastgil was NOT free, jump to slow path
        patch_slow = self.mc.get_relative_pos()
        self.mc.TRAP0(0xDE)  # placeholder: J_IF_FALSE -> slow path

        # Fast path: fastgil was 0 (free) -- store our thread_id to claim it
        # scratch1 still holds &fastgil
        self.mc.STW(r.scratch1.value, r.thread_id.value, 0)
        # Jump past slow path
        patch_end = self.mc.get_relative_pos()
        self.mc.TRAP0(0xDE)  # placeholder: unconditional jump past slow path

        # Patch the conditional: if NOT free, jump to slow path
        slow_pos = self.mc.get_relative_pos()
        offset_to_slow = slow_pos - patch_slow
        pmc = OverwritingBuilder(self.mc, patch_slow, INST_SIZE)
        pmc.J_IF_FALSE(pd, offset_to_slow)

        # Slow path: call reacqgil
        # Save result register into callee-saved regs before reacqgil call
        reg = self.resloc
        if reg is not None:
            if reg.is_core_reg():
                self.mc.MV(r.r24.value, reg.value)
            elif reg.is_reg_pair():
                # Float result in caller-saved pair (e.g. R1:R0) - save to
                # callee-saved R24, R25 which survive the call
                self.mc.MV(r.r24.value, reg.even_reg())
                self.mc.MV(r.r25.value, reg.odd_reg())

        self.mc.gen_load_int(r.scratch1.value, self.asm.reacqgil_addr)
        self.mc.CALLR(r.scratch1.value)

        # Restore result register
        if reg is not None:
            if reg.is_core_reg():
                self.mc.MV(reg.value, r.r24.value)
            elif reg.is_reg_pair():
                self.mc.MV(reg.even_reg(), r.r24.value)
                self.mc.MV(reg.odd_reg(), r.r25.value)

        # End label: patch the fast-path unconditional jump
        end_pos = self.mc.get_relative_pos()
        offset_to_end = end_pos - patch_end
        pmc = OverwritingBuilder(self.mc, patch_end, INST_SIZE)
        pmc.JUMP(offset_to_end)

        if not we_are_translated():
            self.mc.ADDI(r.fp.value, r.fp.value, -1)

    def write_real_errno(self, save_err):
        """Write saved errno to the real errno location before a C call."""
        if save_err & rffi.RFFI_READSAVED_ERRNO:
            if save_err & rffi.RFFI_ALT_ERRNO:
                rpy_errno = llerrno.get_alt_errno_offset(self.asm.cpu)
            else:
                rpy_errno = llerrno.get_rpy_errno_offset(self.asm.cpu)
            p_errno = llerrno.get_p_errno_offset(self.asm.cpu)
            # Load threadlocal base -> scratch1
            self._load_threadlocal(r.scratch1.value)
            # Load p_errno pointer
            self.mc.LDW(r.scratch2.value, r.scratch1.value, p_errno)
            # Load saved rpy_errno value
            self.mc.LDW(r.scratch1.value, r.scratch1.value, rpy_errno)
            # Store to real errno: *p_errno = rpy_errno_val
            self.mc.STW(r.scratch2.value, r.scratch1.value, 0)
        elif save_err & rffi.RFFI_ZERO_ERRNO_BEFORE:
            p_errno = llerrno.get_p_errno_offset(self.asm.cpu)
            self._load_threadlocal(r.scratch1.value)
            self.mc.LDW(r.scratch2.value, r.scratch1.value, p_errno)
            self.mc.gen_load_int(r.scratch1.value, 0)
            self.mc.STW(r.scratch2.value, r.scratch1.value, 0)

    def read_real_errno(self, save_err):
        """Read the real C errno after a call and save it to RPython."""
        if save_err & rffi.RFFI_SAVE_ERRNO:
            if save_err & rffi.RFFI_ALT_ERRNO:
                rpy_errno = llerrno.get_alt_errno_offset(self.asm.cpu)
            else:
                rpy_errno = llerrno.get_rpy_errno_offset(self.asm.cpu)
            p_errno = llerrno.get_p_errno_offset(self.asm.cpu)
            # Load threadlocal base -> scratch1
            self._load_threadlocal(r.scratch1.value)
            # Load p_errno pointer
            self.mc.LDW(r.scratch2.value, r.scratch1.value, p_errno)
            # Load real errno: scratch2 = *p_errno
            self.mc.LDW(r.scratch2.value, r.scratch2.value, 0)
            # Store to rpy_errno: *(tl_base + rpy_errno) = scratch2
            self._load_threadlocal(r.scratch1.value)
            self.mc.ADDI(r.scratch1.value, r.scratch1.value, rpy_errno)
            self.mc.STW(r.scratch1.value, r.scratch2.value, 0)

    def _load_threadlocal(self, destreg):
        """Load the threadlocal struct base address into destreg."""
        # The threadlocal address is saved on the stack during call_release_gil
        self.mc.LDW(destreg, r.sp.value, self.asm.saved_threadlocal_addr)
