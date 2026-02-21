from rpython.jit.metainterp.test.test_virtualref import VRefTests
from rpython.jit.backend.hexagon.test.test_basic import JitHexagonMixin


class TestVRef(JitHexagonMixin, VRefTests):
    # for the individual tests see
    # ====> ../../../metainterp/test/test_virtualref.py
    pass
