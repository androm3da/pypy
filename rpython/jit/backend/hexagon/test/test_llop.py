from rpython.jit.backend.hexagon.test.test_basic import JitHexagonMixin
from rpython.jit.metainterp.test.test_llop import TestLLOp as _TestLLOp


class TestLLOp(JitHexagonMixin, _TestLLOp):
    # for the individual tests see
    # ====> ../../../metainterp/test/test_llop.py

    # do NOT test the blackhole implementation of gc_store_indexed. It cannot
    # work inside tests because llmodel.py:bh_gc_store_indexed_* receive a
    # symbolic as the offset.
    TEST_BLACKHOLE = False
