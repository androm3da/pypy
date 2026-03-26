"""
A simple standalone target.

The target below specifies None as the argument types list.
This is a case treated specially in driver.py . If the list
of input types is empty, it is meant to be a list of strings,
actually implementing argv of the executable.
"""

from rpython.rlib.jit import JitDriver

jitdriver = JitDriver(greens=[], reds=['n', 'total'])

def debug(msg):
    print "debug:", msg

def jitloop(n):
    total = 0
    while n > 0:
        jitdriver.jit_merge_point(n=n, total=total)
        total = total + n
        n = n - 1
    return total

# __________  Entry point  __________

def entry_point(argv):
    debug("hello world")
    result = jitloop(1100)
    debug("result = " + str(result))
    return 0

# _____ Define and setup target ___

def target(*args):
    return entry_point
