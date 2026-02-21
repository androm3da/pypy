import py
from rpython.jit.backend.llsupport.test.zrpy_vmprof_test import CompiledVmprofTest

try:
    if not py.test.config.option.run_slow_tests:
        py.test.skip("use --slow to execute this long-running test")
except AttributeError:
    pass  # not running under pytest


class TestZVMprof(CompiledVmprofTest):
    gcrootfinder = "shadowstack"
    gc = "incminimark"
