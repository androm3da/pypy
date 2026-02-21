from rpython.jit.backend.hexagon.test.test_basic import JitHexagonMixin
from rpython.jit.metainterp.test.test_zvector import VectorizeTests


class JitHexagonVectorMixin(JitHexagonMixin):
    # Vector tests need the unroll optimization enabled
    enable_opts = 'intbounds:rewrite:virtualize:string:earlyforce:pure:heap:unroll'


class TestVector(JitHexagonVectorMixin, VectorizeTests):
    # for the individual tests see
    # ====> ../../../metainterp/test/test_zvector.py
    pass
