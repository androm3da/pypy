import py
from rpython.jit.backend.llsupport.test.zrpy_releasegil_test import ReleaseGILTests

try:
    if not py.test.config.option.run_slow_tests:
        py.test.skip("use --slow to execute this long-running test")
except AttributeError:
    pass  # not running under pytest


class TestShadowStack(ReleaseGILTests):
    gcrootfinder = "shadowstack"
