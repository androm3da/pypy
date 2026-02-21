#!/usr/bin/env python
"""
Backend tests for the Hexagon JIT.

Inherits the full test suite from LLtypeBackendTest which exercises
loop compilation, bridges, guards, GC integration, etc.
"""

from rpython.jit.backend.detect_cpu import getcpuclass
from rpython.jit.backend.test.runner_test import LLtypeBackendTest

CPU = getcpuclass()


class FakeStats(object):
    pass


class TestHexagon(LLtypeBackendTest):
    # for the individual tests see
    # ====> ../../test/runner_test.py

    def get_cpu(self):
        cpu = CPU(rtyper=None, stats=FakeStats())
        cpu.setup_once()
        return cpu

    # Expected instruction patterns for test_compile_asmlen.
    # These are semicolon-delimited instruction mnemonics matching
    # the disassembly of a simple add loop and a bridge.
    add_loop_instructions = (
        'memw; '       # load from jitframe (same_as)
        'add; '        # int_add
        'cmp.eq; '     # guard_true compare
        'if; '         # guard_true conditional jump
        'jump; '       # jump back to loop
        'trap0;'       # guard failure stub
    )
    bridge_loop_instructions = (
        'memw; '       # load frame depth
        'combine; '    # load expected depth
        'cmp.ge; '     # compare
        'if; '         # conditional branch to realloc
        'sub; '        # int_sub
        'memw; '       # store to jitframe
        'combine; '    # load target address
        'jumpr; '      # jump to target
        'trap0;'       # guard failure stub
    )
