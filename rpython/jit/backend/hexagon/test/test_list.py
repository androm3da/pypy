from rpython.jit.metainterp.test.test_list import ListTests
from rpython.jit.backend.hexagon.test.test_basic import JitHexagonMixin


class TestList(JitHexagonMixin, ListTests):
    # for individual tests see
    # ====> ../../../metainterp/test/test_list.py
    pass
