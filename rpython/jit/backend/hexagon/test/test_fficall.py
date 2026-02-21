from rpython.jit.metainterp.test import test_fficall
from rpython.jit.backend.hexagon.test.test_basic import JitHexagonMixin


class TestFfiCall(JitHexagonMixin, test_fficall.FfiCallTests):
    # for the individual tests see
    # ====> ../../../metainterp/test/test_fficall.py

    def _add_libffi_types_to_ll2types_maybe(self):
        # Teach ll2ctypes in advance about the addresses of various types.*
        # structures, needed for blackhole interp to convert integers back to
        # lltype pointers.
        from rpython.rtyper.lltypesystem import rffi, lltype, ll2ctypes
        from rpython.rlib.jit_libffi import types
        for key, value in types.__dict__.iteritems():
            if isinstance(value, lltype._ptr):
                addr = rffi.cast(lltype.Signed, value)
                ll2ctypes._int2obj[addr] = value
