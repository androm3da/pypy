"""Stress target for the Hexagon JIT backend under system emulation.

Exercises: hot int loops with data-dependent branches (bridge compilation),
float arithmetic, GC allocation in loops (lists, strings, dicts), residual
calls, and guard failures.  Usage: targetstress-c <n>
"""

from rpython.rlib import jit
from rpython.rlib.jit import JitDriver, promote

d_int = JitDriver(greens=[], reds='auto', is_recursive=True)
d_flt = JitDriver(greens=[], reds='auto', is_recursive=True)
d_lst = JitDriver(greens=[], reds='auto', is_recursive=True)
d_str = JitDriver(greens=[], reds='auto', is_recursive=True)
d_dct = JitDriver(greens=[], reds='auto', is_recursive=True)
d_ccv = JitDriver(greens=[], reds='auto', is_recursive=True)
d_den = JitDriver(greens=[], reds='auto', is_recursive=True)
d_inr = JitDriver(greens=[], reds='auto', is_recursive=True)
d_out = JitDriver(greens=[], reds='auto', is_recursive=True)


def residual(x):
    if x & 7 == 0:
        return x * 3
    return x + 1
residual._dont_inline_ = True


def int_loop(n):
    total = 0
    i = 0
    while i < n:
        d_int.jit_merge_point()
        if i % 3 == 0:            # guard -> bridge once hot
            total += i * 2
        elif i % 5 == 0:
            total -= residual(i)  # residual call
        else:
            total += i
        i += 1
    return total


def float_loop(n):
    acc = 0.0
    x = 1.5
    i = 0
    while i < n:
        d_flt.jit_merge_point()
        acc += x * 2.5 - float(i) / 4.0
        if acc > 1e9:
            acc *= 0.5
        i += 1
    return int(acc)


def list_loop(n):
    l = []
    i = 0
    while i < n:
        d_lst.jit_merge_point()
        l.append(i ^ 0x55)
        if i & 15 == 15:
            # keep the list small; forces nursery churn
            del l[:8]
        i += 1
    s = 0
    for v in l:
        s += v
    return s


def str_loop(n):
    h = 0
    i = 0
    while i < n:
        d_str.jit_merge_point()
        s = "item" + str(i)
        h = (h * 31 + len(s) + ord(s[0])) & 0x7fffffff
        i += 1
    return h


def dict_loop(n):
    d = {}
    i = 0
    while i < n:
        d_dct.jit_merge_point()
        d[i & 1023] = i
        if i & 1023 in d:
            h = d[i & 1023]
            if h != i:
                return -1
        i += 1
    return len(d)


def _compute_sq(k):
    return k * k + 1


def ccv_loop(n):
    # Exercises COND_CALL_VALUE_I: on-demand computation with caching.
    cache = [0] * 64
    total = 0
    i = 0
    while i < n:
        d_ccv.jit_merge_point()
        k = i & 63
        v = jit.conditional_call_elidable(cache[k], _compute_sq, k)
        cache[k] = v
        total = (total + v) & 0x7fffffff
        i += 1
    return total


def denorm_loop(n):
    # Exercises float_mul with denormal operands (dfmpyfix) in both
    # const*var and var*const forms; the division is a software call
    # that handles denormals correctly and recovers the scale factor.
    tiny = 5e-324               # smallest denormal, 2**-1074
    count = 0
    i = 0
    while i < n:
        d_den.jit_merge_point()
        k = (i & 7) + 1
        p = tiny * float(k)     # exact: k * 2**-1074 is representable
        q = p / tiny
        if int(q) == k:
            count += 1
        r = float(k) * tiny
        if r == p:
            count += 1
        i += 1
    return count


def inner_loop(m):
    s = 0
    j = 0
    while j < m:
        d_inr.jit_merge_point()
        s += j ^ (j >> 3)
        j += 1
    return s


def nested_loop(n):
    # Once inner_loop is compiled, calls to it from outer JIT code
    # become CALL_ASSEMBLER (also exercises the prologue stack check).
    total = 0
    i = 0
    while i < n:
        d_out.jit_merge_point()
        total = (total + inner_loop(i & 31)) & 0x7fffffff
        i += 1
    return total


def entry_point(argv):
    n = 20000
    if len(argv) > 1:
        n = int(argv[1])
    # optional second arg: comma-separated loop names to run (default all);
    # results print as each loop finishes, so partial output survives a
    # crash in a later loop
    sel = "int,float,list,str,dict,ccv,denorm,nested"
    if len(argv) > 2:
        sel = argv[2]
    sel = "," + sel + ","
    if sel.find(",int,") >= 0:
        print "int_loop:", int_loop(n)
    if sel.find(",float,") >= 0:
        print "float_loop:", float_loop(n)
    if sel.find(",list,") >= 0:
        print "list_loop:", list_loop(n)
    if sel.find(",str,") >= 0:
        print "str_loop:", str_loop(n)
    if sel.find(",dict,") >= 0:
        print "dict_loop:", dict_loop(n)
    if sel.find(",ccv,") >= 0:
        print "ccv_loop:", ccv_loop(n)
    if sel.find(",denorm,") >= 0:
        print "denorm_loop:", denorm_loop(n)
    if sel.find(",nested,") >= 0:
        print "nested_loop:", nested_loop(n)
    return 0


def target(*args):
    return entry_point
