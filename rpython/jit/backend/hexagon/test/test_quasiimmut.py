from rpython.jit.backend.hexagon.test.test_basic import JitHexagonMixin
from rpython.jit.metainterp.test import test_quasiimmut


class TestLoopSpec(JitHexagonMixin, test_quasiimmut.QuasiImmutTests):
    # for the individual tests see
    # ====> ../../../metainterp/test/test_quasiimmut.py
    pass
