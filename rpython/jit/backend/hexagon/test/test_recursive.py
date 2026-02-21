import py
from rpython.jit.metainterp.test.test_recursive import RecursiveTests
from rpython.jit.backend.hexagon.test.test_basic import JitHexagonMixin


class TestRecursive(JitHexagonMixin, RecursiveTests):
    # for the individual tests see
    # ====> ../../../metainterp/test/test_recursive.py
    pass
