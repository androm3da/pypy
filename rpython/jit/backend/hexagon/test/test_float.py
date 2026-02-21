from rpython.jit.backend.hexagon.test.test_basic import JitHexagonMixin
from rpython.jit.metainterp.test.test_float import FloatTests


class TestFloat(JitHexagonMixin, FloatTests):
    # for the individual tests see
    # ====> ../../../metainterp/test/test_float.py
    pass
