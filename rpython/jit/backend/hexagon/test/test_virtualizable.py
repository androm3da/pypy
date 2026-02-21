import py
from rpython.jit.metainterp.test.test_virtualizable import ImplicitVirtualizableTests
from rpython.jit.backend.hexagon.test.test_basic import JitHexagonMixin


class TestVirtualizable(JitHexagonMixin, ImplicitVirtualizableTests):
    def test_blackhole_should_not_reenter(self):
        py.test.skip("Assertion error & llinterp mess")
