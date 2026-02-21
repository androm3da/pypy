from rpython.jit.backend.hexagon.test.test_basic import JitHexagonMixin
from rpython.jit.metainterp.test.test_dict import DictTests


class TestDict(JitHexagonMixin, DictTests):
    # for the individual tests see
    # ====> ../../../metainterp/test/test_dict.py
    pass
