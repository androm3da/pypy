from rpython.jit.metainterp.test import test_string
from rpython.jit.backend.hexagon.test.test_basic import JitHexagonMixin


class TestString(JitHexagonMixin, test_string.TestLLtype):
    # for the individual tests see
    # ====> ../../../metainterp/test/test_string.py
    pass


class TestUnicode(JitHexagonMixin, test_string.TestLLtypeUnicode):
    # for the individual tests see
    # ====> ../../../metainterp/test/test_string.py
    pass
