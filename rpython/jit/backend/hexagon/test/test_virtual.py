from rpython.jit.metainterp.test.test_virtual import VirtualTests, VirtualMiscTests
from rpython.jit.backend.hexagon.test.test_basic import JitHexagonMixin


class MyClass:
    pass


class TestsVirtual(JitHexagonMixin, VirtualTests):
    # for the individual tests see
    # ====> ../../../metainterp/test/test_virtual.py
    _new_op = 'new_with_vtable'
    _field_prefix = 'inst_'

    @staticmethod
    def _new():
        return MyClass()


class TestsVirtualMisc(JitHexagonMixin, VirtualMiscTests):
    # for the individual tests see
    # ====> ../../../metainterp/test/test_virtual.py
    pass
