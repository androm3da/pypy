from rpython.jit.backend.hexagon.test.test_basic import JitHexagonMixin
from rpython.jit.metainterp.test.test_del import DelTests


class TestDel(JitHexagonMixin, DelTests):
    # for the individual tests see
    # ====> ../../../metainterp/test/test_del.py
    pass
