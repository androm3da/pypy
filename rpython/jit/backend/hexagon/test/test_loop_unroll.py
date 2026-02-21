from rpython.jit.backend.hexagon.test.test_basic import JitHexagonMixin
from rpython.jit.metainterp.test import test_loop_unroll


class TestLoopSpec(JitHexagonMixin, test_loop_unroll.LoopUnrollTest):
    # for the individual tests see
    # ====> ../../../metainterp/test/test_loop_unroll.py
    pass
