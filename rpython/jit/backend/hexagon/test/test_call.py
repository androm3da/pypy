from rpython.jit.backend.hexagon.test.test_basic import JitHexagonMixin
from rpython.jit.metainterp.test import test_call


class TestCall(JitHexagonMixin, test_call.CallTest):
    # for the individual tests see
    # ====> ../../../metainterp/test/test_call.py
    pass
