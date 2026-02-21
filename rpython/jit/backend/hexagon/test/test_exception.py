import py
from rpython.jit.backend.hexagon.test.test_basic import JitHexagonMixin
from rpython.jit.metainterp.test.test_exception import ExceptionTests


class TestExceptions(JitHexagonMixin, ExceptionTests):
    # for the individual tests see
    # ====> ../../../metainterp/test/test_exception.py
    pass
