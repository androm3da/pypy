from rpython.rlib import rmmap
from rpython.jit.backend.test.calling_convention_test import CallingConvTests, parse
from rpython.jit.backend.hexagon.codebuilder import InstrBuilder
from rpython.jit.backend.hexagon import registers as r
from rpython.jit.metainterp.resoperation import InputArgInt, InputArgFloat

boxint = InputArgInt
boxfloat = InputArgFloat.fromfloat


class TestHexagonCallingConvention(CallingConvTests):
    # ../../test/calling_convention_test.py

    def make_function_returning_stack_pointer(self):
        rmmap.enter_assembler_writing()
        try:
            mc = InstrBuilder()
            # R0 = R29 (SP) -- return stack pointer in R0
            mc.TFR(r.r0.value, r.sp.value)
            # jumpr R31 (return)
            mc.JUMPR(r.lr.value)
            return mc.materialize(self.cpu, [])
        finally:
            rmmap.leave_assembler_writing()

    def get_alignment_requirements(self):
        return 8  # Hexagon ABI requires 8-byte stack alignment
