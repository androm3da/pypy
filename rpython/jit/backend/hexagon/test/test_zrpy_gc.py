import py
from rpython.jit.backend.llsupport.test.zrpy_gc_test import CompileFrameworkTests

try:
    if not py.test.config.option.run_slow_tests:
        py.test.skip("use --slow to execute this long-running test")
except AttributeError:
    pass  # not running under pytest


class TestShadowStack(CompileFrameworkTests):
    gcrootfinder = "shadowstack"
    gc = "incminimark"
