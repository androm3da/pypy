"""
This disables the backend tests on non-Hexagon platforms.
Note that you need "--slow" to run translation tests.

Exception: test_instr_builder.py is allowed on all platforms because its
self-consistency tests only exercise encoding logic (no hardware needed).
The reference-assembler tests are separately skipped when the toolchain
is not available.
"""
import os
import pytest
from rpython.jit.backend import detect_cpu

cpu = detect_cpu.autodetect()
IS_HEXAGON = cpu.startswith('hexagon')
THIS_DIR = os.path.dirname(__file__)

# Test files that can run on any platform (no Hexagon hardware needed)
CROSS_PLATFORM_TESTS = frozenset([
    'test_instr_builder.py',
])


@pytest.hookimpl(tryfirst=True)
def pytest_ignore_collect(path, config):
    path = str(path)
    if not IS_HEXAGON:
        if os.path.commonprefix([path, THIS_DIR]) == THIS_DIR:
            basename = os.path.basename(path)
            if basename in CROSS_PLATFORM_TESTS:
                return False  # allow these tests on any platform
            return True


def pytest_collect_file():
    if not IS_HEXAGON:
        # Only called for test files not ignored above.
        # The cross-platform tests are already allowed through.
        pass
