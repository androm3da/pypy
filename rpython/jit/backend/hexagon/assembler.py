"""
Main assembler for the Hexagon JIT backend.

Orchestrates loop/bridge compilation, prologue/epilogue generation,
guard handling, and GC integration.
"""

from rpython.jit.backend.hexagon import registers as r
from rpython.jit.backend.hexagon.arch import (
    WORD, DOUBLE_WORD, INST_SIZE, JITFRAME_FIXED_SIZE,
)
from rpython.jit.backend.hexagon.codebuilder import InstrBuilder, OverwritingBuilder
from rpython.jit.backend.hexagon.opassembler import ResOpAssembler, asm_operations
from rpython.jit.backend.hexagon.regalloc import prepare_operations
from rpython.jit.backend.hexagon.packet import (
    PacketOptimizer, FLAG_BRANCH, FLAG_MEM_LOAD, FLAG_MEM_STORE, FLAG_BARRIER,
)
from rpython.jit.backend.hexagon.regalloc import Regalloc
from rpython.jit.backend.hexagon.vector_ext import VectorAssemblerMixin
from rpython.jit.backend.llsupport import jitframe, rewrite
from rpython.jit.backend.llsupport.assembler import BaseAssembler
from rpython.jit.backend.llsupport.asmmemmgr import MachineDataBlockWrapper
from rpython.jit.backend.model import CompiledLoopToken
from rpython.jit.metainterp.history import FLOAT
from rpython.jit.metainterp.resoperation import rop
from rpython.rlib.jit import AsmInfo
from rpython.rlib import rgc, rmmap
from rpython.rlib.objectmodel import we_are_translated
from rpython.rlib.rjitlog import rjitlog as jl
from rpython.rtyper.lltypesystem import lltype, rffi, llmemory
from rpython.rtyper.annlowlevel import cast_instance_to_gcref


class AssemblerHexagon(ResOpAssembler, VectorAssemblerMixin):

    def __init__(self, cpu, translate_support_code=False):
        ResOpAssembler.__init__(self, cpu, translate_support_code)
        self.mc = None
        self.pending_guards = []
        self._regalloc = None
        self.current_clt = None
        self.propagate_exception_path = 0
        self.stack_check_slowpath = 0
        self.failure_recovery_code = [0, 0, 0, 0]
        self.wb_slowpath = [0, 0, 0, 0, 0]  # 5 slots: no-cards, cards, no-cards+float, cards+float, frame
        self.saved_threadlocal_addr = 0
        self.gc_table_addr = 0
        self._frame_realloc_slowpath = 0
        self.float_div_addr = 0
        # Register allocation statistics
        self.num_moves_calls = 0
        self.num_moves_jump = 0
        self.num_spills = 0
        self.num_spills_to_existing = 0
        self.num_reloads = 0

    def setup_once(self):
        # Initialize GC and memory support from base class
        rmmap.enter_assembler_writing()
        gc_ll_descr = self.cpu.gc_ll_descr
        gc_ll_descr.initialize()
        if hasattr(gc_ll_descr, 'minimal_size_in_nursery'):
            self.gc_minimal_size_in_nursery = gc_ll_descr.minimal_size_in_nursery
        else:
            self.gc_minimal_size_in_nursery = 0
        if hasattr(gc_ll_descr, 'gcheaderbuilder'):
            self.gc_size_of_header = gc_ll_descr.gcheaderbuilder.size_gc_header
        else:
            self.gc_size_of_header = WORD
        from rpython.jit.backend.llsupport.assembler import memcpy_fn, memset_fn
        self.memcpy_addr = rffi.cast(lltype.Signed, memcpy_fn)
        self.memset_addr = rffi.cast(lltype.Signed, memset_fn)
        from rpython.jit.backend.hexagon.runner import _divdf3_func
        self.float_div_addr = rffi.cast(lltype.Signed, _divdf3_func)
        # Build failure recovery stubs (4 variants: exc x withfloats)
        self._build_failure_recovery(exc=False, withfloats=False)
        self._build_failure_recovery(exc=True, withfloats=False)
        if self.cpu.supports_floats:
            self._build_failure_recovery(exc=False, withfloats=True)
            self._build_failure_recovery(exc=True, withfloats=True)
        # Build other helper stubs
        self._build_propagate_exception_path()
        self._build_wb_slowpath(False)
        self._build_wb_slowpath(True)
        self._build_wb_slowpath(False, for_frame=True)
        # Build cond_call slowpaths (4 variants: floats x callee_only)
        lst = [0, 0, 0, 0]
        lst[0] = self._build_cond_call_slowpath(False, False)
        lst[1] = self._build_cond_call_slowpath(False, True)
        if self.cpu.supports_floats:
            lst[2] = self._build_cond_call_slowpath(True, False)
            lst[3] = self._build_cond_call_slowpath(True, True)
        self.cond_call_slowpath = lst
        self._build_stack_check_slowpath()
        self.build_frame_realloc_slowpath()
        # Build malloc slowpaths for different allocation kinds
        if gc_ll_descr.get_malloc_slowpath_addr is not None:
            self.malloc_slowpath = self._build_malloc_slowpath(kind='fixed')
            self.malloc_slowpath_varsize = self._build_malloc_slowpath(
                kind='var')
        if hasattr(gc_ll_descr, 'malloc_str'):
            self.malloc_slowpath_str = self._build_malloc_slowpath(kind='str')
        else:
            self.malloc_slowpath_str = None
        if hasattr(gc_ll_descr, 'malloc_unicode'):
            self.malloc_slowpath_unicode = self._build_malloc_slowpath(
                kind='unicode')
        else:
            self.malloc_slowpath_unicode = None
        self._build_release_gil(gc_ll_descr.gcrootmap)
        # Allocate a gcmap for FINISH operations (only slot 0 = R0 is live)
        from rpython.rlib.rarithmetic import r_uint
        self.gcmap_for_finish = lltype.malloc(jitframe.GCMAP, 1,
                                              flavor='raw',
                                              track_allocation=False)
        self.gcmap_for_finish[0] = r_uint(1)
        rmmap.leave_assembler_writing()

    def finish_once(self):
        pass

    def setup(self, looptoken):
        BaseAssembler.setup(self, looptoken)
        assert self.memcpy_addr != 0, 'setup_once() not called?'
        self.current_clt = looptoken.compiled_loop_token
        self.mc = InstrBuilder()
        self.pending_guards = []
        self.packet_opt = PacketOptimizer()
        self.target_tokens_currently_compiling = {}
        self.frame_depth_to_patch = []
        self._gc_table_load_patches = []
        # Set up data block wrapper for allocating data alongside code
        allblocks = self.get_asmmemmgr_blocks(looptoken)
        self.datablockwrapper = MachineDataBlockWrapper(
            self.cpu.asmmemmgr, allblocks)

    def teardown(self):
        self.current_clt = None
        self._regalloc = None
        self.mc = None
        self.pending_guards = None
        self.packet_opt = None

    # -------------------------------------------------------------------
    # Loop assembly
    # -------------------------------------------------------------------

    @rgc.no_release_gil
    def assemble_loop(self, jd_id, unique_id, logger, loopname, inputargs,
                      operations, looptoken, log):
        rmmap.enter_assembler_writing()
        try:
            return self._assemble_loop(jd_id, unique_id, logger, loopname,
                                       inputargs, operations, looptoken, log)
        finally:
            rmmap.leave_assembler_writing()

    def _assemble_loop(self, jd_id, unique_id, logger, loopname, inputargs,
                       operations, looptoken, log):
        clt = CompiledLoopToken(self.cpu, looptoken.number)
        clt._debug_nbargs = len(inputargs)
        looptoken.compiled_loop_token = clt
        self.setup(looptoken)

        frame_info = self.datablockwrapper.malloc_aligned(
            jitframe.JITFRAMEINFO_SIZE, alignment=WORD)
        clt.frame_info = rffi.cast(jitframe.JITFRAMEINFOPTR, frame_info)
        clt.frame_info.clear()

        self._regalloc = Regalloc(self)
        allgcrefs = []
        operations = self._regalloc._prepare(inputargs, operations, allgcrefs)
        self._regalloc._set_initial_bindings(inputargs, looptoken)
        self._regalloc.possibly_free_vars(list(inputargs))
        self.setup_gcrefs_list(allgcrefs)

        # Reserve space for GC ref table at start of code
        gcref_table_size = len(allgcrefs) * WORD
        if gcref_table_size > 0:
            gcref_table_size = (gcref_table_size + 7) & ~7  # align to 8
            for i in range(gcref_table_size // INST_SIZE):
                self.mc.write32(0)

        # Function prologue
        function_pos = self.mc.get_relative_pos()
        self._call_header()

        # Emit the loop body
        loop_head = self.mc.get_relative_pos()
        looptoken._ll_loop_code = loop_head
        self._walk_operations(inputargs, operations)

        # Emit guard failure stubs
        size_excluding_failure_stuff = self.mc.get_relative_pos()
        self._emit_pending_guards()

        # Calculate frame depth (variable slots only from regalloc)
        frame_depth = self._regalloc.get_final_frame_depth()
        jump_target_descr = self._regalloc.jump_target_descr
        if jump_target_descr is not None:
            tgt_depth = jump_target_descr._hexagon_clt.frame_info.jfi_frame_depth
            target_frame_depth = tgt_depth - JITFRAME_FIXED_SIZE
            frame_depth = max(frame_depth, target_frame_depth)
        # Add fixed register spill slots to get the total frame depth
        frame_depth += JITFRAME_FIXED_SIZE

        # Finalize and copy code to executable memory
        rawstart = self._materialize_loop(looptoken, allgcrefs)
        looptoken._ll_function_addr = rawstart + function_pos

        # Update frame depth
        baseofs = self.cpu.get_baseofs_of_frame_field()
        clt.frame_info.update_frame_depth(baseofs, frame_depth)

        ops_offset = self.mc.ops_offset

        if logger:
            log = logger.log_trace(jl.MARK_TRACE_ASM, None, self.mc)
            log.write(inputargs, operations, ops_offset=ops_offset)

        self.teardown()

        return AsmInfo(ops_offset, rawstart + loop_head,
                       size_excluding_failure_stuff - loop_head)

    # -------------------------------------------------------------------
    # Bridge assembly
    # -------------------------------------------------------------------

    @rgc.no_release_gil
    def assemble_bridge(self, logger, faildescr, inputargs, operations,
                        original_loop_token, log):
        rmmap.enter_assembler_writing()
        try:
            return self._assemble_bridge(logger, faildescr, inputargs,
                                         operations, original_loop_token, log)
        finally:
            rmmap.leave_assembler_writing()

    def _assemble_bridge(self, logger, faildescr, inputargs, operations,
                         original_loop_token, log):
        self.setup(original_loop_token)
        clt = self.current_clt

        self._regalloc = Regalloc(self)
        allgcrefs = []
        # Rebuild register locations from the fail description
        arglocs = self.rebuild_faillocs_from_descr(faildescr, inputargs)
        operations = self._regalloc.prepare_bridge(
            inputargs, arglocs, operations, allgcrefs, clt.frame_info)
        self.setup_gcrefs_list(allgcrefs)

        # Reserve GC ref table
        gcref_table_size = len(allgcrefs) * WORD
        if gcref_table_size > 0:
            gcref_table_size = (gcref_table_size + 7) & ~7
            for i in range(gcref_table_size // INST_SIZE):
                self.mc.write32(0)

        # No prologue for bridges -- we resume from a guard failure
        startpos = self.mc.get_relative_pos()

        # Emit the bridge body
        self._walk_operations(inputargs, operations)

        # Emit guard failure stubs
        codeendpos = self.mc.get_relative_pos()
        self._emit_pending_guards()

        # Calculate frame depth (variable slots only from regalloc)
        bridge_frame_depth = self._regalloc.get_final_frame_depth()
        jump_target_descr = self._regalloc.jump_target_descr
        if jump_target_descr is not None:
            tgt_depth = jump_target_descr._hexagon_clt.frame_info.jfi_frame_depth
            target_frame_depth = tgt_depth - JITFRAME_FIXED_SIZE
            bridge_frame_depth = max(bridge_frame_depth, target_frame_depth)
        # Add fixed register spill slots to get the total frame depth
        bridge_frame_depth += JITFRAME_FIXED_SIZE

        # Update frame depth (max of existing and bridge)
        frame_depth = max(clt.frame_info.jfi_frame_depth, bridge_frame_depth)
        baseofs = self.cpu.get_baseofs_of_frame_field()
        clt.frame_info.update_frame_depth(baseofs, frame_depth)

        # Finalize and copy code to executable memory
        rawstart = self._materialize_loop(original_loop_token, allgcrefs)

        # Patch the original guard to jump to this bridge.
        # Use rawstart + startpos to skip the GC ref table at the
        # start of the bridge code buffer.
        self._patch_guard_to_bridge(faildescr, rawstart + startpos)

        ops_offset = self.mc.ops_offset

        if logger:
            log = logger.log_trace(jl.MARK_TRACE_ASM, None, self.mc)
            log.write(inputargs, operations, ops_offset=ops_offset)

        self.teardown()

        return AsmInfo(ops_offset, rawstart + startpos,
                       codeendpos - startpos)

    # -------------------------------------------------------------------
    # Operation walking
    # -------------------------------------------------------------------

    def _walk_operations(self, inputargs, operations):
        regalloc = self._regalloc
        regalloc.operations = operations
        popt = self.packet_opt
        while regalloc.position() < len(operations) - 1:
            regalloc.next_instruction()
            i = regalloc.position()
            op = operations[i]
            opnum = op.getopnum()

            # Mark operation position
            self.mc.mark_op(op)

            if rop.has_no_side_effect(opnum) and op not in regalloc.longevity:
                # Dead code: result unused and no side effects
                regalloc.possibly_free_vars_for_op(op)
                continue

            if (rop.is_call_may_force(opnum) or
                    rop.is_call_release_gil(opnum) or
                    rop.is_call_assembler(opnum)):
                # Calls are complex: flush before, barrier after
                popt.flush(self.mc)
                # These ops are always paired with guard_not_forced
                guard_op = operations[i + 1]
                guard_num = guard_op.getopnum()
                assert guard_num in (rop.GUARD_NOT_FORCED,
                                     rop.GUARD_NOT_FORCED_2)
                arglocs, num_arglocs = \
                    regalloc.prepare_guard_op_guard_not_forced(op, guard_op)
                if arglocs is not None:
                    self.emit_guard_op_guard_not_forced(
                        op, guard_op, arglocs, num_arglocs)
                regalloc.next_instruction()  # advance past the guard
                regalloc.possibly_free_vars_for_op(op)
                if guard_op.is_guard():
                    regalloc.possibly_free_vars(guard_op.getfailargs())
                regalloc.possibly_free_vars_for_op(guard_op)
                regalloc.possibly_free_var(guard_op)
                continue

            # Flush before control flow operations (guards, jumps, finish)
            if _is_control_flow_op(opnum):
                popt.flush(self.mc)

            # Track positions around prepare phase (may emit spills/loads)
            pos_pre_prepare = self.mc.get_relative_pos()

            # Normal operation dispatch via pre-built table
            arglocs = prepare_operations[opnum](regalloc, op)

            pos_post_prepare = self.mc.get_relative_pos()

            # If prepare phase emitted instructions (regalloc spills/loads),
            # insert a packet barrier
            if pos_post_prepare > pos_pre_prepare:
                popt.barrier()

            # Track emit phase position
            pos_pre_emit = self.mc.get_relative_pos()

            # Emit the operation (arglocs=None means skip emission)
            if arglocs is not None:
                emit_func = asm_operations[opnum]
                emit_func(self, op, arglocs)

            pos_post_emit = self.mc.get_relative_pos()

            # Annotate for VLIW packet bundling
            if not _is_control_flow_op(opnum):
                n_emitted = (pos_post_emit - pos_pre_emit) // INST_SIZE
                if n_emitted == 1:
                    reads, writes, flags = _extract_op_reg_info(opnum, arglocs)
                    if flags >= 0:
                        popt.annotate(pos_pre_emit, reads, writes, flags)
                    else:
                        popt.barrier()
                elif n_emitted > 1:
                    popt.barrier()

            regalloc.possibly_free_vars_for_op(op)
            if rop.is_guard(opnum):
                regalloc.possibly_free_vars(op.getfailargs())
            regalloc.possibly_free_var(op)

        # Flush any remaining pending packet at end of trace
        popt.flush(self.mc)

    # -------------------------------------------------------------------
    # Prologue / Epilogue
    # -------------------------------------------------------------------

    def _call_header(self):
        """Emit function prologue.

        - Save LR and callee-saved registers to the stack
        - Set FP = JITFRAME pointer (passed in R0)
        - Set up the stack frame
        """
        # Save LR on stack
        self.mc.ADDI(r.sp.value, r.sp.value, -self._get_frame_size())

        # Save callee-saved registers
        offset = 0
        for reg in r.callee_saved_to_spill:
            self.mc.STW(r.sp.value, reg.value, offset)
            offset += WORD

        # Set FP = R0 (JITFRAME pointer passed as first argument)
        self.mc.MV(r.fp.value, r.r0.value)

        # Save LR
        self.mc.STW(r.sp.value, r.lr.value, offset)

    def gen_func_epilog(self):
        """Emit function epilogue to self.mc."""
        self._gen_func_epilog_mc(self.mc)

    def _gen_func_epilog_mc(self, mc):
        """Emit function epilogue using the given mc builder.

        - Restore callee-saved registers
        - Restore LR
        - Return (jumpr LR)
        """
        # Restore callee-saved registers
        offset = 0
        for reg in r.callee_saved_to_spill:
            mc.LDW(reg.value, r.sp.value, offset)
            offset += WORD

        # Restore LR
        mc.LDW(r.lr.value, r.sp.value, offset)

        # Deallocate stack frame
        mc.ADDI(r.sp.value, r.sp.value, self._get_frame_size())

        # Return
        mc.JUMPR(r.lr.value)

    def _get_frame_size(self):
        """Calculate the stack frame size for callee-saved registers + LR."""
        n = len(r.callee_saved_to_spill) + 1  # +1 for LR
        size = n * WORD
        # Align to 8 bytes
        if size % 8 != 0:
            size += 8 - (size % 8)
        return size

    # -------------------------------------------------------------------
    # Guard handling
    # -------------------------------------------------------------------

    def _emit_pending_guards(self):
        """Emit all pending guard failure stubs and patch the guard sites."""
        for token in self.pending_guards:
            token.pos_recovery_stub = self.generate_quick_failure(token)

        # Now patch the guard sites to jump to their stubs
        for token in self.pending_guards:
            self._patch_guard_site(token)

    def generate_quick_failure(self, guardtok):
        """Generate a guard recovery stub.

        Each guard failure stub:
        1. Stores the guard's faildescrindex into jf_descr
        2. Stores the gcmap into jf_gcmap
        3. Jumps to the shared failure_recovery_code
        """
        startpos = self.mc.get_relative_pos()
        faildescrindex, target = self.store_info_on_descr(startpos, guardtok)

        # Store faildescrindex into jf_descr via the GC ref table
        self.load_from_gc_table(r.scratch1.value, faildescrindex)
        ofs = self.cpu.get_ofs_of_frame_field('jf_descr')
        self.mc.store_to_jitframe(r.scratch1.value, ofs)

        # Store gcmap into jf_gcmap
        self.push_gcmap(self.mc, guardtok.gcmap)

        # Jump to the shared failure recovery code
        assert target != 0
        self.mc.gen_load_int(r.scratch1.value, target)
        self.mc.JUMPR(r.scratch1.value)

        return startpos

    def process_pending_guards(self, block_start):
        """After code is materialized, record absolute recovery stub
        addresses into faildescrs for later bridge patching."""
        for token in self.pending_guards:
            descr = token.faildescr
            # Store absolute address of recovery stub so bridges can patch it
            descr.adr_jump_offset = block_start + token.pos_recovery_stub

    def _patch_guard_site(self, token):
        """Replace the TRAP0 at the guard site with a conditional jump.

        P0 was set before the guard. The token stores fail_on_pred_true:
          True  -> fail when P0=1 -> if (P0) jump fail_stub
          False -> fail when P0=0 -> if (!P0) jump fail_stub
        """
        jump_pos = token.pos_jump_offset
        target_pos = token.pos_recovery_stub
        offset = target_pos - jump_pos

        mc = OverwritingBuilder(self.mc, jump_pos, INST_SIZE)
        if token.fail_on_pred_true:
            mc.J_IF_TRUE(r.scratch_pred.value, offset)
        else:
            mc.J_IF_FALSE(r.scratch_pred.value, offset)

    def _push_all_regs_to_jitframe_mc(self, mc, callee_only=False):
        """Save registers to the JITFRAME using a given mc builder."""
        base_ofs = self.cpu.get_baseofs_of_frame_field()
        regs = r.callee_saved_to_spill if callee_only else r.allocatable_registers
        for reg in regs:
            idx = self.cpu.all_reg_indexes[reg.value]
            if idx >= 0:
                mc.store_to_jitframe(reg.value, base_ofs + idx * WORD)
        if not callee_only:
            # Float pairs are stored after integer regs in the jitframe.
            # Align to DOUBLE_WORD since STD requires 8-byte aligned offsets.
            float_base = base_ofs + len(self.cpu.gen_regs) * WORD
            float_base = (float_base + (DOUBLE_WORD - 1)) & ~(DOUBLE_WORD - 1)
            for i, pair in enumerate(r.allocatable_float_pairs):
                mc.store_pair_to_jitframe(pair.value,
                                          float_base + i * DOUBLE_WORD)

    def _pop_all_regs_from_jitframe_mc(self, mc, callee_only=False):
        """Restore registers from the JITFRAME using a given mc builder."""
        base_ofs = self.cpu.get_baseofs_of_frame_field()
        regs = r.callee_saved_to_spill if callee_only else r.allocatable_registers
        for reg in regs:
            idx = self.cpu.all_reg_indexes[reg.value]
            if idx >= 0:
                mc.load_from_jitframe(reg.value, base_ofs + idx * WORD)
        if not callee_only:
            float_base = base_ofs + len(self.cpu.gen_regs) * WORD
            float_base = (float_base + (DOUBLE_WORD - 1)) & ~(DOUBLE_WORD - 1)
            for i, pair in enumerate(r.allocatable_float_pairs):
                mc.load_pair_from_jitframe(pair.value,
                                           float_base + i * DOUBLE_WORD)

    def _save_all_regs_to_jitframe(self):
        """Save all managed registers to the JITFRAME."""
        self._push_all_regs_to_jitframe_mc(self.mc, callee_only=False)

    def _load_all_regs_from_jitframe(self):
        """Restore all managed registers from the JITFRAME."""
        self._pop_all_regs_from_jitframe_mc(self.mc, callee_only=False)

    # -------------------------------------------------------------------
    # Helper stubs (built during setup_once, shared across compilations)
    # -------------------------------------------------------------------

    def _build_propagate_exception_path(self):
        """Build a stub that propagates exceptions from C calls.

        Stores the exception into jf_guard_exc, sets jf_descr to
        propagate_exception_descr, moves fp to r0, and returns.
        """
        mc = InstrBuilder()
        # Store exception value into jf_guard_exc
        # (exception is in pos_exc_value global; load + store)
        if self.cpu.pos_exc_value():
            mc.gen_load_int(r.scratch1.value, self.cpu.pos_exc_value())
            mc.LDW(r.scratch1.value, r.scratch1.value, 0)
            ofs = self.cpu.get_ofs_of_frame_field('jf_guard_exc')
            mc.store_to_jitframe(r.scratch1.value, ofs)
            # Clear exception globals
            mc.gen_load_int(r.scratch1.value, self.cpu.pos_exc_value())
            mc.gen_load_int(r.scratch2.value, 0)
            mc.STW(r.scratch1.value, r.scratch2.value, 0)
            mc.gen_load_int(r.scratch1.value, self.cpu.pos_exception())
            mc.STW(r.scratch1.value, r.scratch2.value, 0)
        # Store propagate_exception_descr into jf_descr
        propagate_descr = rffi.cast(lltype.Signed,
            cast_instance_to_gcref(self.cpu.propagate_exception_descr))
        ofs = self.cpu.get_ofs_of_frame_field('jf_descr')
        mc.gen_load_int(r.scratch1.value, propagate_descr)
        mc.store_to_jitframe(r.scratch1.value, ofs)
        # Return jitframe pointer in R0
        mc.MV(r.r0.value, r.fp.value)
        mc.JUMPR(r.lr.value)
        # Materialize and store
        rawstart = self._materialize_helper(mc)
        self.propagate_exception_path = rawstart

    def _build_stack_check_slowpath(self):
        """Build a stub for C stack overflow checking.

        Called at the start of each compiled loop. Checks if SP is
        below the stack limit and calls the slowpath function if so.
        Saves/restores argument registers around the call.
        """
        _, _, slowpathaddr = self.cpu.insert_stack_check()
        if slowpathaddr == 0 or not self.cpu.propagate_exception_descr:
            return  # no stack check (tests or non-translated)

        mc = InstrBuilder()
        # Save LR and argument registers
        n_arg_regs = len(r.argument_regs)
        frame_size = (n_arg_regs + 2) * WORD  # +1 for LR, +1 for alignment
        if frame_size % 8 != 0:
            frame_size += 4
        mc.ADDI(r.sp.value, r.sp.value, -frame_size)
        mc.STW(r.sp.value, r.lr.value, 0)
        for i, reg in enumerate(r.argument_regs):
            mc.STW(r.sp.value, reg.value, (i + 1) * WORD)

        # Pass SP as argument to the stack check function
        mc.MV(r.r0.value, r.sp.value)
        mc.gen_load_int(r.scratch1.value, slowpathaddr)
        mc.CALLR(r.scratch1.value)

        # Check for exception
        mc.gen_load_int(r.scratch1.value, self.cpu.pos_exception())
        mc.LDW(r.scratch1.value, r.scratch1.value, 0)
        pd = r.scratch_pred.value
        mc.CMP_EQI(pd, r.scratch1.value, 0)
        # If no exception (P0=1), jump to restore+return
        jmp_ok = mc.get_relative_pos()
        mc.TRAP0(0xDE)  # placeholder

        # Exception path: restore SP and jump to propagate_exception
        mc.ADDI(r.sp.value, r.sp.value, frame_size)
        mc.gen_load_int(r.scratch1.value, self.propagate_exception_path)
        mc.JUMPR(r.scratch1.value)

        # Normal path: restore registers and return
        normal_pos = mc.get_relative_pos()
        offset = normal_pos - jmp_ok
        pmc = OverwritingBuilder(mc, jmp_ok, INST_SIZE)
        pmc.J_IF_TRUE(pd, offset)

        for i, reg in enumerate(r.argument_regs):
            mc.LDW(reg.value, r.sp.value, (i + 1) * WORD)
        mc.LDW(r.lr.value, r.sp.value, 0)
        mc.ADDI(r.sp.value, r.sp.value, frame_size)
        mc.JUMPR(r.lr.value)

        rawstart = self._materialize_helper(mc)
        self.stack_check_slowpath = rawstart

    def _build_wb_slowpath(self, withcards, for_frame=False):
        """Build write barrier slow path.

        Called when the inline write barrier check detects that a full
        barrier is needed. The object pointer is in R0.
        Saves callee-saved regs to jitframe, calls the WB function,
        restores and returns.
        """
        descr = self.cpu.gc_ll_descr.write_barrier_descr
        if descr is None:
            return
        if not withcards:
            func = descr.get_write_barrier_fn(self.cpu)
        else:
            if descr.jit_wb_cards_set == 0:
                return
            func = descr.get_write_barrier_from_array_fn(self.cpu)
            if func == 0:
                return

        mc = InstrBuilder()
        # Save LR
        mc.ADDI(r.sp.value, r.sp.value, -2 * WORD)
        mc.STW(r.sp.value, r.lr.value, 0)

        save_size = 0
        if not for_frame:
            # Save callee-saved regs to jitframe
            self._push_all_regs_to_jitframe_mc(mc, callee_only=True)
        else:
            # For frame WB, save caller-saved regs on stack
            n_save = len(r.caller_saved_registers)
            save_size = n_save * WORD
            if save_size % 8 != 0:
                save_size += 4
            mc.ADDI(r.sp.value, r.sp.value, -save_size)
            for i, reg in enumerate(r.caller_saved_registers):
                mc.STW(r.sp.value, reg.value, i * WORD)

        # Call the write barrier function (R0 already has the object ptr)
        mc.gen_load_int(r.scratch1.value, func)
        mc.CALLR(r.scratch1.value)

        if not for_frame:
            self._pop_all_regs_from_jitframe_mc(mc, callee_only=True)
        else:
            for i, reg in enumerate(r.caller_saved_registers):
                mc.LDW(reg.value, r.sp.value, i * WORD)
            mc.ADDI(r.sp.value, r.sp.value, save_size)

        if withcards:
            # Re-check the flag for the caller's conditional
            mc.LDUB(r.scratch1.value, r.r0.value,
                    descr.jit_wb_if_flag_byteofs)
            mc.gen_load_int(r.scratch2.value, 0x80)
            mc.AND(r.scratch1.value, r.scratch1.value, r.scratch2.value)

        # Restore LR and return
        mc.LDW(r.lr.value, r.sp.value, 0)
        mc.ADDI(r.sp.value, r.sp.value, 2 * WORD)
        mc.JUMPR(r.lr.value)

        rawstart = self._materialize_helper(mc)
        if for_frame:
            self.wb_slowpath[4] = rawstart
        else:
            self.wb_slowpath[1 if withcards else 0] = rawstart

    def _build_cond_call_slowpath(self, supports_floats, callee_only):
        """Build a general call slowpath trampoline for cond_call.

        The callee function address comes in R14 (scratch1).
        The return value is stored in R14 (scratch1).
        Saves/restores registers to jitframe around the call.
        """
        mc = InstrBuilder()
        # Save LR
        mc.ADDI(r.sp.value, r.sp.value, -2 * WORD)
        mc.STW(r.sp.value, r.lr.value, 0)

        # Save registers to JITFRAME (skip scratch regs and fp)
        self._push_all_regs_to_jitframe_mc(mc, callee_only=callee_only)

        # Call the function (address is in scratch1/R14)
        mc.CALLR(r.scratch1.value)

        # Move return value to scratch1 (R14) for the caller
        mc.MV(r.scratch1.value, r.r0.value)

        # Reload frame if GC may have moved it
        self._reload_frame_if_necessary(mc)

        # Restore registers from JITFRAME
        self._pop_all_regs_from_jitframe_mc(mc, callee_only=callee_only)

        # Restore LR and return
        mc.LDW(r.lr.value, r.sp.value, 0)
        mc.ADDI(r.sp.value, r.sp.value, 2 * WORD)
        mc.JUMPR(r.lr.value)

        return self._materialize_helper(mc)

    def _build_failure_recovery(self, exc, withfloats=False):
        """Build a shared failure recovery stub.

        There are 4 variants: exc (0/1) x withfloats (0/1).
        Each saves all registers to the JITFRAME, optionally handles
        exception state, then returns the JITFRAME pointer to the caller.

        The stub is jumped to from generate_quick_failure() after the
        per-guard code has stored jf_descr and jf_gcmap.
        """
        mc = InstrBuilder()

        # Save all managed registers to JITFRAME.
        # _push_all_regs_to_jitframe_mc always saves float pairs when
        # callee_only=False, so the withfloats flag here just controls
        # which recovery code we select at guard emission time.
        self._push_all_regs_to_jitframe_mc(mc, callee_only=False)

        if exc:
            # Move exception from pos_exc_value to jf_guard_exc, then clear
            mc.gen_load_int(r.scratch1.value, self.cpu.pos_exc_value())
            mc.LDW(r.scratch2.value, r.scratch1.value, 0)
            ofs = self.cpu.get_ofs_of_frame_field('jf_guard_exc')
            mc.store_to_jitframe(r.scratch2.value, ofs)
            # Clear exception globals
            mc.gen_load_int(r.scratch2.value, 0)
            mc.STW(r.scratch1.value, r.scratch2.value, 0)
            mc.gen_load_int(r.scratch1.value, self.cpu.pos_exception())
            mc.STW(r.scratch1.value, r.scratch2.value, 0)

        # Return JITFRAME pointer in R0, then full function epilogue
        mc.MV(r.r0.value, r.fp.value)
        self._gen_func_epilog_mc(mc)

        rawstart = self._materialize_helper(mc)
        self.failure_recovery_code[exc + 2 * withfloats] = rawstart

    def build_frame_realloc_slowpath(self):
        """Build frame reallocation slowpath.

        Called when the jitframe is too small for the needed stack depth.
        At entry:
        - R14 (scratch1) holds the new frame size
        - FP (R30) holds the old jitframe pointer
        - All managed registers are live and need to be preserved
        """
        mc = InstrBuilder()

        # 1. Save all managed registers to the old jitframe
        self._push_all_regs_to_jitframe_mc(mc, callee_only=False)

        # 2. Save and reset any pending exception using callee-saved regs
        #    Use R24/R25 (callee-saved, allocatable) to hold exc state
        exc_val_reg = r.r24
        exc_tp_reg = r.r25
        self._store_and_reset_exception(mc, exc_val_reg, exc_tp_reg)

        # 3. Call realloc_frame(old_jitframe, new_size)
        #    R0 = old jitframe, R1 = new size
        mc.MV(r.r0.value, r.fp.value)
        mc.MV(r.r1.value, r.scratch1.value)
        func = rffi.cast(lltype.Signed, self.cpu.realloc_frame)
        mc.gen_load_int(r.scratch1.value, func)
        mc.CALLR(r.scratch1.value)

        # 4. Update FP to point to the new jitframe (returned in R0)
        mc.MV(r.fp.value, r.r0.value)

        # 5. Restore exception state
        self._restore_exception(mc, exc_val_reg, exc_tp_reg)

        # 6. Update shadow stack if needed
        gcrootmap = self.cpu.gc_ll_descr.gcrootmap
        if gcrootmap and gcrootmap.is_shadow_stack:
            rst = gcrootmap.get_root_stack_top_addr()
            mc.gen_load_int(r.scratch1.value, rst)
            mc.LDW(r.scratch1.value, r.scratch1.value, 0)
            mc.ADDI(r.scratch1.value, r.scratch1.value, -WORD)
            mc.STW(r.scratch1.value, r.fp.value, 0)

        # 7. Clear jf_gcmap
        jf_gcmap_ofs = self.cpu.get_ofs_of_frame_field('jf_gcmap')
        mc.gen_load_int(r.scratch1.value, 0)
        mc.store_to_jitframe(r.scratch1.value, jf_gcmap_ofs)

        # 8. Restore all managed registers from the new jitframe
        self._pop_all_regs_from_jitframe_mc(mc, callee_only=False)

        # 9. Return
        mc.JUMPR(r.lr.value)

        rawstart = self._materialize_helper(mc)
        self._frame_realloc_slowpath = rawstart

    def _build_malloc_slowpath(self, kind):
        """Build malloc slowpath for nursery allocation.

        kind: 'fixed', 'str', 'unicode', or 'var'

        At entry for 'fixed':
            R0 = nursery_free (result pointer)
            R1 = nursery_free + size (new free pointer)
            gcmap is pushed by caller
        For 'str'/'unicode':
            R0 = length
        For 'var':
            R0 = itemsize, R1 = type_id, R2 = length
        """
        mc = InstrBuilder()

        # Save LR
        mc.ADDI(r.sp.value, r.sp.value, -2 * WORD)
        mc.STW(r.sp.value, r.lr.value, 0)

        # Save all managed registers to jitframe
        # Skip the args and fp
        self._push_all_regs_to_jitframe_mc(mc, callee_only=False)

        # Determine which malloc function to call
        gc_ll_descr = self.cpu.gc_ll_descr
        if kind == 'fixed':
            addr = gc_ll_descr.get_malloc_slowpath_addr()
            # Compute size = R1 - R0
            mc.SUB(r.r0.value, r.r1.value, r.r0.value)
            if hasattr(gc_ll_descr, 'passes_frame'):
                mc.MV(r.r1.value, r.fp.value)
        elif kind == 'str':
            addr = gc_ll_descr.get_malloc_fn_addr('malloc_str')
        elif kind == 'unicode':
            addr = gc_ll_descr.get_malloc_fn_addr('malloc_unicode')
        else:
            assert kind == 'var'
            addr = gc_ll_descr.get_malloc_slowpath_array_addr()

        # Store gcmap (from caller, already on jf_gcmap)
        # -- caller already pushed gcmap, nothing extra needed here --

        # Call the allocation function
        mc.gen_load_int(r.scratch1.value, rffi.cast(lltype.Signed, addr))
        mc.CALLR(r.scratch1.value)

        # Check for allocation failure (R0 == 0)
        pd = r.scratch_pred.value
        mc.CMP_EQI(pd, r.r0.value, 0)
        jmp_ok = mc.get_relative_pos()
        mc.TRAP0(0xDE)  # placeholder: if (P0) jump to failure

        # Failure path: propagate memory error
        mc.gen_load_int(r.scratch1.value, self.propagate_exception_path)
        mc.JUMPR(r.scratch1.value)

        # Success path
        ok_pos = mc.get_relative_pos()
        offset = ok_pos - jmp_ok
        pmc = OverwritingBuilder(mc, jmp_ok, INST_SIZE)
        pmc.J_IF_FALSE(pd, offset)  # skip failure when P0=0 (R0 != 0)

        # Reload frame if GC may have moved it
        self._reload_frame_if_necessary(mc)

        # Restore all managed registers from jitframe
        self._pop_all_regs_from_jitframe_mc(mc, callee_only=False)

        # Reload nursery_free_adr into R1 (for 'fixed' kind)
        if kind == 'fixed':
            nursery_free_adr = gc_ll_descr.get_nursery_free_addr()
            mc.gen_load_int(r.r1.value, nursery_free_adr)
            mc.LDW(r.r1.value, r.r1.value, 0)

        # Clear jf_gcmap
        jf_gcmap_ofs = self.cpu.get_ofs_of_frame_field('jf_gcmap')
        mc.gen_load_int(r.scratch1.value, 0)
        mc.store_to_jitframe(r.scratch1.value, jf_gcmap_ofs)

        # Restore LR and return
        mc.LDW(r.lr.value, r.sp.value, 0)
        mc.ADDI(r.sp.value, r.sp.value, 2 * WORD)
        mc.JUMPR(r.lr.value)

        return self._materialize_helper(mc)

    def _materialize_helper(self, mc):
        """Copy a helper InstrBuilder to executable memory, return address."""
        size = mc.get_code_size()
        if size == 0:
            return 0
        return mc.materialize(self.cpu, [])

    # -------------------------------------------------------------------
    # Code finalization
    # -------------------------------------------------------------------

    def _materialize_loop(self, looptoken, allgcrefs):
        """Finalize and copy machine code to executable memory.

        Handles datablockwrapper finalization, code copy, guard address
        recording, and GC ref table setup.
        """
        # Finalize data blocks
        self.datablockwrapper.done()
        self.datablockwrapper = None

        # Allocate executable memory
        allblocks = self.get_asmmemmgr_blocks(looptoken)
        size = self.mc.get_code_size()
        if size == 0:
            self.teardown_gcrefs_list()
            return 0
        alloc = self.cpu.asmmemmgr.malloc(size, size)
        allblocks.append(alloc)
        rawstart = alloc[0]

        # Set gc_table_addr BEFORE patching (GC table is at buffer start)
        gcref_table_size = len(allgcrefs) * WORD
        if gcref_table_size > 0:
            self.gc_table_addr = rawstart

        # Patch GC table load placeholders with actual addresses
        self._patch_gc_table_loads()

        # DEBUG: dump buffer around prologue/first-ops boundary
        _gc_words = gcref_table_size // INST_SIZE
        self._debug_dump_buf("BEFORE_COPY", _gc_words)

        # Copy patched code to executable memory
        self.mc.copy_to_raw_memory(rawstart)
        self.mc.rawstart = rawstart

        # DEBUG: read back from memory to verify copy
        self._debug_dump_mem("AFTER_COPY", rawstart, _gc_words,
                             min(20, len(self.mc._buf) - _gc_words))

        # Record absolute recovery stub addresses for bridge patching
        self.process_pending_guards(rawstart)

        # Create GC ref tracer
        if gcref_table_size > 0:
            tracer = self.cpu.gc_ll_descr.make_gcref_tracer(
                rawstart, allgcrefs)
            gcreftracers = self.get_asmmemmgr_gcreftracers(looptoken)
            gcreftracers.append(tracer)
        self.teardown_gcrefs_list()

        return rawstart

    # Maximum number of instructions gen_load_int can emit.
    # Worst case: TFRSI rd + TFRSI scratch + ASL + ZXTH + OR = 5
    _GC_LOAD_INT_MAX = 5

    def _patch_gc_table_loads(self):
        """Patch GC table load placeholders with actual absolute addresses.

        Called after rawstart is known but before copy_to_raw_memory.
        Overwrites the NOP placeholders emitted by load_from_gc_table
        with actual gen_load_int sequences.
        """
        for pos, reg, index in self._gc_table_load_patches:
            addr = self.gc_table_addr + index * WORD
            ob = OverwritingBuilder(self.mc, pos,
                                    self._GC_LOAD_INT_MAX * INST_SIZE)
            ob.gen_load_int(r.scratch2.value, addr)
            # Pad remaining slots with NOPs
            while ob._pos < self._GC_LOAD_INT_MAX * INST_SIZE:
                ob.NOP()

    # -------------------------------------------------------------------
    # Guard-to-bridge patching
    # -------------------------------------------------------------------

    def _patch_guard_to_bridge(self, faildescr, bridge_addr):
        """Patch a guard's failure recovery stub to jump to a bridge.

        The guard's recovery stub currently saves regs and returns.
        We overwrite the entry of that stub with a jump to the bridge.
        """
        # faildescr.adr_jump_offset was set during process_pending_guards
        # to point to the recovery stub's address in executable memory.
        patch_addr = faildescr.adr_jump_offset
        if patch_addr == 0:
            return
        # Write a JUMP instruction at the patch address that branches
        # to the bridge code.
        rmmap.enter_assembler_writing()
        try:
            mc = InstrBuilder()
            # Load bridge address and jump to it.
            # Since bridges may be far away, use register-indirect jump.
            mc.gen_load_int(r.scratch1.value, bridge_addr)
            mc.JUMPR(r.scratch1.value)
            mc.copy_to_raw_memory(patch_addr)
        finally:
            rmmap.leave_assembler_writing()
        faildescr.adr_jump_offset = 0  # prevent double-patching

    # -------------------------------------------------------------------
    # Redirect call assembler
    # -------------------------------------------------------------------

    def redirect_call_assembler(self, oldlooptoken, newlooptoken):
        """Redirect a CALL_ASSEMBLER to a new loop.

        Patches the entry point of the old loop to jump to the new one.
        """
        oldadr = oldlooptoken._ll_function_addr
        target = newlooptoken._ll_function_addr
        # Update frame info
        baseofs = self.cpu.get_baseofs_of_frame_field()
        newlooptoken.compiled_loop_token.update_frame_info(
            oldlooptoken.compiled_loop_token, baseofs)
        # Overwrite old entry with jump to new
        rmmap.enter_assembler_writing()
        try:
            mc = InstrBuilder()
            mc.gen_load_int(r.scratch1.value, target)
            mc.JUMPR(r.scratch1.value)
            mc.copy_to_raw_memory(oldadr)
        finally:
            rmmap.leave_assembler_writing()

    # -------------------------------------------------------------------
    # Debug helpers
    # -------------------------------------------------------------------

    def _debug_dump_buf(self, label, start_idx):
        from rpython.rlib.debug import debug_print
        end = min(start_idx + 20, len(self.mc._buf))
        debug_print("HEXDBG ", label, ": buf_len=", len(self.mc._buf),
                    " start=", start_idx)
        i = start_idx
        while i < end:
            debug_print("HEXDBG   buf[", i, "] = ", self.mc._buf[i])
            i += 1

    def _debug_dump_mem(self, label, addr, start_idx, count):
        from rpython.rlib.debug import debug_print
        p = rffi.cast(rffi.UINTP, addr)
        debug_print("HEXDBG ", label, ": addr=", addr,
                    " start=", start_idx, " count=", count)
        i = start_idx
        while i < start_idx + count:
            debug_print("HEXDBG   mem[", i, "] = ",
                        rffi.cast(rffi.SIGNED, p[i]))
            i += 1

    # -------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------

    def imm(self, value):
        from rpython.jit.backend.hexagon.locations import ImmLocation
        return ImmLocation(value)

    def new_stack_loc(self, i, tp):
        """Create a StackLocation at position i of type tp.

        Called by rebuild_faillocs_from_descr() in llsupport.
        The position i is relative to the variable area (after JITFRAME_FIXED_SIZE).
        """
        from rpython.jit.backend.hexagon.locations import StackLocation, get_fp_offset
        base_ofs = self.cpu.get_baseofs_of_frame_field()
        return StackLocation(i, get_fp_offset(base_ofs, i), type=tp)

    # -------------------------------------------------------------------
    # Location movement (used by remap_frame_layout in jump.py)
    # -------------------------------------------------------------------

    def regalloc_mov(self, prev_loc, loc):
        """Move a value from prev_loc to loc."""
        self.mc.regalloc_mov(prev_loc, loc)
    mov_loc_loc = regalloc_mov

    def regalloc_push(self, loc, already_pushed):
        """Push loc to the stack. already_pushed counts how many
        slots have already been pushed (used for offset computation)."""
        offset = -DOUBLE_WORD * (already_pushed + 1)
        if loc.is_core_reg():
            self.mc.STW(r.sp.value, loc.value, offset)
        elif loc.is_reg_pair():
            self.mc.STD(r.sp.value, loc.value, offset)
        elif loc.is_stack():
            if loc.type == FLOAT:
                self.mc.load_pair_from_jitframe(r.d14.value, loc.value)
                self.mc.STD(r.sp.value, r.d14.value, offset)
            else:
                self.mc.load_from_jitframe(r.scratch1.value, loc.value)
                self.mc.STW(r.sp.value, r.scratch1.value, offset)
        elif loc.is_imm():
            self.mc.gen_load_int(r.scratch1.value, loc.value)
            self.mc.STW(r.sp.value, r.scratch1.value, offset)
        else:
            raise AssertionError("regalloc_push: unsupported loc %s" % loc)

    def regalloc_pop(self, loc, already_pushed):
        """Pop a value from the stack to loc."""
        offset = -DOUBLE_WORD * (already_pushed + 1)
        if loc.is_core_reg():
            self.mc.LDW(loc.value, r.sp.value, offset)
        elif loc.is_reg_pair():
            self.mc.LDD(loc.value, r.sp.value, offset)
        elif loc.is_stack():
            if loc.type == FLOAT:
                self.mc.LDD(r.d14.value, r.sp.value, offset)
                self.mc.store_pair_to_jitframe(r.d14.value, loc.value)
            else:
                self.mc.LDW(r.scratch1.value, r.sp.value, offset)
                self.mc.store_to_jitframe(r.scratch1.value, loc.value)
        else:
            raise AssertionError("regalloc_pop: unsupported loc %s" % loc)

    def regalloc_prepare_move(self, src, dst, tmp):
        """If src-to-dst needs a temp (stack-to-stack or imm-to-stack),
        move src to tmp and return tmp. Otherwise return src."""
        if dst.is_stack() and (src.is_stack() or src.is_imm()):
            self.regalloc_mov(src, tmp)
            return tmp
        return src

    def simple_call(self, fnloc, arglocs, result_loc=None):
        """Emit a simple C function call.

        fnloc: ImmLocation with function address
        arglocs: list of locations to pass as arguments
        result_loc: location for the result (or None)
        """
        from rpython.jit.backend.hexagon.callbuilder import HexagonCallBuilder
        if result_loc is None:
            result_type = 'v'
            result_size = 0
        elif result_loc.is_float():
            result_type = 'f'
            result_size = DOUBLE_WORD
        else:
            result_type = 'i'
            result_size = WORD
        cb = HexagonCallBuilder(self, fnloc, arglocs, result_loc,
                                result_type, result_size)
        cb.emit()

    def _store_force_index(self, guard_op):
        """Store the guard's faildescr into jf_force_descr."""
        faildescr = guard_op.getdescr()
        faildescrindex = self.get_gcref_from_faildescr(faildescr)
        ofs = self.cpu.get_ofs_of_frame_field('jf_force_descr')
        self.load_from_gc_table(r.scratch1.value, faildescrindex)
        self.mc.store_to_jitframe(r.scratch1.value, ofs)

    # -------------------------------------------------------------------
    # GC support
    # -------------------------------------------------------------------

    def push_gcmap(self, mc, gcmap, store=True):
        """Store a GC map pointer into the jitframe's jf_gcmap field."""
        assert store
        ofs = self.cpu.get_ofs_of_frame_field('jf_gcmap')
        ptr = rffi.cast(lltype.Signed, gcmap)
        mc.gen_load_int(r.scratch1.value, ptr)
        mc.store_to_jitframe(r.scratch1.value, ofs)

    def pop_gcmap(self, mc):
        """Clear the jitframe's jf_gcmap field (write 0)."""
        ofs = self.cpu.get_ofs_of_frame_field('jf_gcmap')
        mc.gen_load_int(r.scratch1.value, 0)
        mc.store_to_jitframe(r.scratch1.value, ofs)

    def _store_and_reset_exception(self, mc, excvalloc, exctploc):
        """Save current exception and clear it."""
        mc.gen_load_int(r.scratch1.value, self.cpu.pos_exc_value())
        mc.LDW(excvalloc.value, r.scratch1.value, 0)
        mc.gen_load_int(r.scratch1.value, self.cpu.pos_exception())
        mc.LDW(exctploc.value, r.scratch1.value, 0)
        # Clear both
        mc.gen_load_int(r.scratch2.value, 0)
        mc.gen_load_int(r.scratch1.value, self.cpu.pos_exc_value())
        mc.STW(r.scratch1.value, r.scratch2.value, 0)
        mc.gen_load_int(r.scratch1.value, self.cpu.pos_exception())
        mc.STW(r.scratch1.value, r.scratch2.value, 0)

    def _restore_exception(self, mc, excvalloc, exctploc):
        """Restore a previously saved exception."""
        mc.gen_load_int(r.scratch1.value, self.cpu.pos_exc_value())
        mc.STW(r.scratch1.value, excvalloc.value, 0)
        mc.gen_load_int(r.scratch1.value, self.cpu.pos_exception())
        mc.STW(r.scratch1.value, exctploc.value, 0)

    def _reload_frame_if_necessary(self, mc, tmplocs=[]):
        """Reload the jitframe pointer if GC may have moved it.

        After a C call that might trigger GC, the frame could have been
        relocated. If using shadow stack, reload fp from there.
        """
        gcrootmap = self.cpu.gc_ll_descr.gcrootmap
        if gcrootmap and gcrootmap.is_shadow_stack:
            rst = gcrootmap.get_root_stack_top_addr()
            mc.gen_load_int(r.scratch1.value, rst)
            mc.LDW(r.scratch1.value, r.scratch1.value, 0)
            # root_stack_top points one past the last entry; go back one WORD
            mc.ADDI(r.scratch1.value, r.scratch1.value, -WORD)
            mc.LDW(r.fp.value, r.scratch1.value, 0)

    # -------------------------------------------------------------------
    # Register allocation moves
    # -------------------------------------------------------------------

    def regalloc_mov(self, prev_loc, loc):
        """Move a value between locations for register allocation."""
        if prev_loc.is_imm():
            self._mov_imm_to_loc(prev_loc, loc)
        elif prev_loc.is_core_reg():
            self._mov_reg_to_loc(prev_loc, loc)
        elif prev_loc.is_stack():
            self._mov_stack_to_loc(prev_loc, loc)
        elif prev_loc.is_reg_pair():
            self._mov_pair_to_loc(prev_loc, loc)
        else:
            raise AssertionError("regalloc_mov: unknown prev_loc type")

    def _mov_imm_to_loc(self, imm_loc, loc):
        if loc.is_core_reg():
            self.mc.gen_load_int(loc.value, imm_loc.value)
        elif loc.is_stack():
            self.mc.gen_load_int(r.scratch1.value, imm_loc.value)
            self.mc.store_to_jitframe(r.scratch1.value, loc.value)
        else:
            raise AssertionError("_mov_imm_to_loc: unsupported dest")

    def _mov_reg_to_loc(self, reg_loc, loc):
        if loc.is_core_reg():
            if reg_loc.value != loc.value:
                self.mc.MV(loc.value, reg_loc.value)
        elif loc.is_stack():
            self.mc.store_to_jitframe(reg_loc.value, loc.value)
        else:
            raise AssertionError("_mov_reg_to_loc: unsupported dest")

    def _mov_stack_to_loc(self, stack_loc, loc):
        if loc.is_core_reg():
            self.mc.load_from_jitframe(loc.value, stack_loc.value)
        elif loc.is_stack():
            self.mc.load_from_jitframe(r.scratch1.value, stack_loc.value)
            self.mc.store_to_jitframe(r.scratch1.value, loc.value)
        else:
            raise AssertionError("_mov_stack_to_loc: unsupported dest")

    def _mov_pair_to_loc(self, pair_loc, loc):
        if loc.is_reg_pair():
            if pair_loc.value != loc.value:
                self.mc.MV(loc.even_reg(), pair_loc.even_reg())
                self.mc.MV(loc.odd_reg(), pair_loc.odd_reg())
        elif loc.is_stack():
            self.mc.store_pair_to_jitframe(pair_loc.value, loc.value)
        else:
            raise AssertionError("_mov_pair_to_loc: unsupported dest")

    def mov_loc_to_raw_stack(self, loc, pos):
        """Save a value from loc to the raw stack at SP+pos."""
        if loc.is_core_reg():
            self.mc.STW(r.sp.value, loc.value, pos)
        elif loc.is_stack():
            self.mc.load_from_jitframe(r.scratch1.value, loc.value)
            self.mc.STW(r.sp.value, r.scratch1.value, pos)
        elif loc.is_imm():
            self.mc.gen_load_int(r.scratch1.value, loc.value)
            self.mc.STW(r.sp.value, r.scratch1.value, pos)
        elif loc.is_reg_pair():
            self.mc.STD(r.sp.value, loc.value, pos)
        else:
            raise AssertionError("mov_loc_to_raw_stack: unsupported loc")


# ---------------------------------------------------------------------------
# VLIW packet optimizer helpers
# ---------------------------------------------------------------------------

# Operations that are control flow boundaries -- packets cannot span these
_CONTROL_FLOW_OPS = dict.fromkeys([
    rop.JUMP,
    rop.FINISH,
    rop.LABEL,
    rop.GUARD_TRUE,
    rop.GUARD_FALSE,
    rop.GUARD_VALUE,
    rop.GUARD_CLASS,
    rop.GUARD_NONNULL,
    rop.GUARD_ISNULL,
    rop.GUARD_NONNULL_CLASS,
    rop.GUARD_GC_TYPE,
    rop.GUARD_IS_OBJECT,
    rop.GUARD_SUBCLASS,
    rop.GUARD_NO_EXCEPTION,
    rop.GUARD_EXCEPTION,
    rop.GUARD_NOT_INVALIDATED,
    rop.GUARD_NOT_FORCED,
    rop.GUARD_NOT_FORCED_2,
    rop.GUARD_NO_OVERFLOW,
    rop.GUARD_OVERFLOW,
])


def _is_control_flow_op(opnum):
    """Check if an operation is a control flow boundary for packet bundling."""
    if opnum in _CONTROL_FLOW_OPS:
        return True
    if rop.is_guard(opnum):
        return True
    if rop.is_call(opnum):
        return True
    return False


# Binary ALU operations with argloc layout: [l0, l1, res]
# These emit a single instruction when l1 is a register OR when the
# immediate fits in the instruction's immediate field.
# INT_XOR is included: with a register operand it emits 1 instruction;
# with an immediate it emits 2+ (caught by the n_emitted check).
_BINARY_ALU_OPS = dict.fromkeys([
    rop.INT_ADD,
    rop.INT_SUB,
    rop.INT_MUL,
    rop.INT_AND,
    rop.INT_OR,
    rop.INT_XOR,
    rop.INT_LSHIFT,
    rop.INT_RSHIFT,
    rop.UINT_RSHIFT,
    rop.NURSERY_PTR_INCREMENT,
])

# Move operations with argloc layout: [l0, res]
_MOVE_OPS = dict.fromkeys([
    rop.SAME_AS_I,
    rop.SAME_AS_R,
    rop.CAST_PTR_TO_INT,
    rop.CAST_INT_TO_PTR,
])

# GC load operations -- after the rewrite phase, getfield_gc/getarrayitem_gc
# are replaced by gc_load variants.
# Argloc layout: [base_loc, ofs_loc, res_loc, nsize_loc]
_GC_LOAD_OPS = dict.fromkeys([
    rop.GC_LOAD_I,
    rop.GC_LOAD_R,
    rop.GC_LOAD_F,
])

# GC store operations -- after rewrite, setfield_gc becomes gc_store.
# Argloc layout: [value_loc, base_loc, ofs_loc, size_loc]
_GC_STORE_OPS = dict.fromkeys([
    rop.GC_STORE,
])

# Legacy field operations that may appear in tests or before full rewrite.
# Argloc layout: [base, ofs, res] for loads, [base, ofs, val] for stores.
_GETFIELD_OPS = dict.fromkeys([
    rop.GETFIELD_GC_I,
    rop.GETFIELD_GC_R,
])

_SETFIELD_OPS = dict.fromkeys([
    rop.SETFIELD_GC,
])


def _extract_op_reg_info(opnum, arglocs):
    """Extract register read/write info for a single-instruction operation.

    Returns (reads, writes, flags) if the operation is recognized,
    or None if unknown (caller should insert a barrier).

    Only called when the operation emitted exactly 1 instruction.
    """
    if opnum in _BINARY_ALU_OPS:
        l0, l1, res = arglocs[:3]
        reads = [l0.value]
        if not l1.is_imm():
            reads.append(l1.value)
        return reads, [res.value], 0

    if opnum in _MOVE_OPS:
        l0, res = arglocs[:2]
        if l0.is_imm():
            return [], [res.value], 0
        return [l0.value], [res.value], 0

    # GC load: [base_loc, ofs_loc, res_loc, nsize_loc]
    if opnum in _GC_LOAD_OPS:
        base_loc, ofs_loc, res_loc = arglocs[0], arglocs[1], arglocs[2]
        reads = [base_loc.value]
        if not ofs_loc.is_imm():
            reads.append(ofs_loc.value)
        return reads, [res_loc.value], FLAG_MEM_LOAD

    # GC store: [value_loc, base_loc, ofs_loc, size_loc]
    if opnum in _GC_STORE_OPS:
        value_loc, base_loc, ofs_loc = arglocs[0], arglocs[1], arglocs[2]
        reads = [value_loc.value, base_loc.value]
        if not ofs_loc.is_imm():
            reads.append(ofs_loc.value)
        return reads, [], FLAG_MEM_STORE

    # Legacy getfield: [base, ofs, res]
    if opnum in _GETFIELD_OPS:
        base, ofs, res = arglocs[:3]
        reads = [base.value]
        if not ofs.is_imm():
            reads.append(ofs.value)
        return reads, [res.value], FLAG_MEM_LOAD

    # Legacy setfield: [base, ofs, val]
    if opnum in _SETFIELD_OPS:
        base, ofs, val = arglocs[:3]
        reads = [base.value, val.value]
        if not ofs.is_imm():
            reads.append(ofs.value)
        return reads, [], FLAG_MEM_STORE

    return [], [], -1  # sentinel: unknown op
