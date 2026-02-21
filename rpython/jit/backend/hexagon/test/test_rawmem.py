from rpython.jit.backend.hexagon.test.test_basic import JitHexagonMixin
from rpython.jit.metainterp.test.test_rawmem import RawMemTests


class TestRawMem(JitHexagonMixin, RawMemTests):
    # for the individual tests see
    # ====> ../../../metainterp/test/test_rawmem.py
    pass
