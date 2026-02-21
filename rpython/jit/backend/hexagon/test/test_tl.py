from rpython.jit.metainterp.test.test_tl import ToyLanguageTests
from rpython.jit.backend.hexagon.test.test_basic import JitHexagonMixin


class TestTL(JitHexagonMixin, ToyLanguageTests):
    # for the individual tests see
    # ====> ../../../metainterp/test/test_tl.py
    pass
