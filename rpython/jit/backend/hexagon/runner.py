"""
CPU class for the Hexagon JIT backend.
"""

from rpython.jit.backend.llsupport.llmodel import AbstractLLCPU
from rpython.jit.backend.hexagon import arch
from rpython.jit.backend.hexagon import registers as r
from rpython.jit.backend.hexagon.vector_ext import HexagonVectorExt
from rpython.rlib import rgc, rmmap
from rpython.rtyper.lltypesystem import llmemory, rffi, lltype
from rpython.translator.tool.cbuild import ExternalCompilationInfo

# Hexagon has no hardware double-precision division.
# Resolve __hexagon_divdf3 from libgcc/compiler-rt.
_divdf3_eci = ExternalCompilationInfo(post_include_bits=[
    'extern double __hexagon_divdf3(double, double);'
])
_divdf3_func = rffi.llexternal(
    '__hexagon_divdf3',
    [rffi.DOUBLE, rffi.DOUBLE], rffi.DOUBLE,
    compilation_info=_divdf3_eci,
    _nowrapper=True, sandboxsafe=True)


class CPU_HEXAGON(AbstractLLCPU):
    backend_name = 'hexagon'
    IS_64_BIT = False

    supports_floats = True
    supports_longlong = True
    supports_singlefloats = True

    # R30 (fp) holds the JITFRAME pointer
    frame_reg = r.fp

    # Map register number (0-31) to JITFRAME slot index.
    # R0 (return value) goes to index 0 because AbstractFailDescr
    # assumes the return value is at index 0.
    #
    # Allocatable: R0-R13 (indices 0-13), R24 (index 14), R25 (index 15)
    # Non-allocatable regs get -1 (not stored in frame).
    all_reg_indexes = [
         0,  1,  2,  3,  4,  5,  6,  7,   # R0-R7
         8,  9, 10, 11, 12, 13, -1, -1,   # R8-R15 (R14,R15 scratch)
        -1, -1, -1, -1, -1, -1, -1, -1,   # R16-R23 (float pairs)
        14, 15, -1, -1, -1, -1, -1, -1,   # R24,R25,R26-R31
    ]

    # The inverse map: JITFRAME index -> register
    gen_regs = [
        r.r0,  r.r1,  r.r2,  r.r3,  r.r4,  r.r5,  r.r6,  r.r7,
        r.r8,  r.r9,  r.r10, r.r11, r.r12, r.r13,
        r.r24, r.r25,
    ]

    # Float registers (register pairs) stored in JITFRAME slots 16-23
    # D16 occupies slots 16-17, D18 occupies 18-19, D20: 20-21, D22: 22-23
    float_regs = r.allocatable_float_pairs

    JITFRAME_FIXED_SIZE = arch.JITFRAME_FIXED_SIZE

    HAS_CODEMAP = True

    # HVX vector extension support
    vector_ext = HexagonVectorExt()

    def __init__(self, rtyper, stats, opts=None, translate_support_code=False,
                 gcdescr=None):
        AbstractLLCPU.__init__(self, rtyper, stats, opts,
                               translate_support_code, gcdescr)

    def setup(self):
        from rpython.jit.backend.hexagon.assembler import AssemblerHexagon
        self.assembler = AssemblerHexagon(self, self.translate_support_code)

    @rgc.no_release_gil
    def setup_once(self):
        self.assembler.setup_once()
        if self.HAS_CODEMAP:
            self.codemap.setup()
        self.vector_ext.setup_once(self.assembler)

    @rgc.no_release_gil
    def finish_once(self):
        AbstractLLCPU.finish_once(self)
        self.assembler.finish_once()

    def compile_bridge(self, faildescr, inputargs, operations,
                       original_loop_token, log=True, logger=None):
        clt = original_loop_token.compiled_loop_token
        clt.compiling_a_bridge()
        return self.assembler.assemble_bridge(logger, faildescr, inputargs,
                                              operations, original_loop_token,
                                              log=log)

    def redirect_call_assembler(self, oldlooptoken, newlooptoken):
        self.assembler.redirect_call_assembler(oldlooptoken, newlooptoken)

    @rgc.no_release_gil
    def invalidate_loop(self, looptoken):
        from rpython.jit.backend.hexagon.codebuilder import InstrBuilder

        rmmap.enter_assembler_writing()
        try:
            for jmp, tgt in looptoken.compiled_loop_token.invalidate_positions:
                mc = InstrBuilder()
                mc.JUMP(tgt)
                mc.copy_to_raw_memory(jmp)
        finally:
            rmmap.leave_assembler_writing()

        looptoken.compiled_loop_token.invalidate_positions = []

    def cast_ptr_to_int(x):
        adr = llmemory.cast_ptr_to_adr(x)
        return CPU_HEXAGON.cast_adr_to_int(adr)
    cast_ptr_to_int._annspecialcase_ = 'specialize:arglltype(0)'
    cast_ptr_to_int = staticmethod(cast_ptr_to_int)

    def build_regalloc(self):
        ''' for tests'''
        from rpython.jit.backend.hexagon.regalloc import Regalloc
        assert self.assembler is not None
        return Regalloc(self.assembler)


for _i, _r in enumerate(r.allocatable_registers):
    assert CPU_HEXAGON.all_reg_indexes[_r.value] == _i
assert arch.NUM_MANAGED_GPRS == len(r.allocatable_registers)
