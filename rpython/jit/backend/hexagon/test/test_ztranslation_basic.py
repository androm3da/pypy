import py
from rpython.jit.backend.llsupport.test.ztranslation_test import TranslationTest

try:
    if not py.test.config.option.run_slow_tests:
        py.test.skip("use --slow to execute this long-running test")
except AttributeError:
    pass  # not running under pytest


class TestTranslationHexagon(TranslationTest):
    pass
