from rpython.rlib.rarithmetic import intmask, r_uint, uint_mul_high


from rpython.jit.metainterp.history import ConstInt
from rpython.jit.metainterp.resoperation import ResOperation, rop


# Logic to replace the signed integer division by a constant
# by a few operations involving a UINT_MUL_HIGH.

def _get_LONG_BIT():
    # Import LONG_BIT from intutils which has the correct target value
    # after cross-compilation patching via _update_for_target_long_bit().
    # We cannot use rarithmetic.LONG_BIT because it reflects the host.
    from rpython.jit.metainterp.optimizeopt import intutils
    return intutils.LONG_BIT

def magic_numbers(m):
    LONG_BIT = _get_LONG_BIT()
    assert m == intmask(m)   # as a signed int, we have m < 2**63
    assert m & (m-1) != 0    # not a power of two
    assert m >= 3
    i = 1
    while (r_uint(1) << (i+1)) < r_uint(m):
        i += 1

    # quotient = 2**(LONG_BIT+i) // m
    high_word_dividend = r_uint(1) << i
    quotient = r_uint(0)
    for bit in range(LONG_BIT-1, -1, -1):
        t = quotient + (r_uint(1) << bit)
        # check: is 't * m' small enough to be < 2**(LONG_BIT+i), or not?
        # note that we're really computing (2**(LONG_BIT+i)-1) // m, but the
        # result is the same, because powers of two are not multiples of m.
        if uint_mul_high(t, r_uint(m)) < high_word_dividend:
            quotient = t     # yes, small enough

    # k = 2**(LONG_BIT+i) // m + 1
    k = quotient + r_uint(1)

    assert k != r_uint(0)
    # Proof that k < 2**LONG_BIT holds in all cases, even with the "+1":
    #
    # starting point: 2**i < m < 2**(i+1)  with i < LONG_BIT-1
    # 2**i < m
    # 2**i <= m - (2.0**(i-LONG_BIT+1))  as real number
    # 2**(LONG_BIT+i) <= 2**LONG_BIT * m - 2**(i+1)   as integers again
    # 2**(LONG_BIT+i) < 2**LONG_BIT * m - m
    # 2**(LONG_BIT+i) / float(m) < 2**LONG_BIT-1    real numbers division
    # 2**(LONG_BIT+i) // m < 2**LONG_BIT-1    with the integer division
    #       k        < 2**LONG_BIT

    assert k > (r_uint(1) << (LONG_BIT-1))
    # This is because m < 2**(i+1), so 2**(LONG_BIT+i) // m >= 2**(LONG_BIT-1)

    return (k, i)


def division_operations(n_box, m, known_nonneg=False):
    LONG_BIT = _get_LONG_BIT()
    kk, ii = magic_numbers(m)

    # Turn the division into:
    #     t = n >> (LONG_BIT-1)    # t == 0 or t == -1
    #     return (((n^t) * k) >> (LONG_BIT + i)) ^ t

    # Proof that this gives exactly a = n // m = floor(q), where q
    # is the real number quotient:
    #
    # case t == 0, i.e. 0 <= n < 2**(LONG_BIT-1)
    #
    #     a <= q <= a + (m-1)/m     (we use '/' for the real quotient here)
    #
    #     n * k == n * (2**(LONG_BIT+i) // m + 1)
    #           == n * ceil(2**(LONG_BIT+i) / m)
    #           == n * (2**(LONG_BIT+i) / m + ferr)       for 0 < ferr < 1
    #           == q * 2**(LONG_BIT+i) + err               for 0 < err < n
    #           <  q * 2**(LONG_BIT+i) + n
    #           <= (a + (m-1)/m) * 2**(LONG_BIT+i) + n
    #           == 2**(LONG_BIT+i) * (a + extra)           for 0 <= extra < ?
    #
    #     extra == (m-1)/m + (n / 2**(LONG_BIT+i))
    #
    #     but  n < 2**(LONG_BIT-1) < 2**(LONG_BIT+i)/m  because  m < 2**(i+1)
    #
    #     extra < (m-1)/m + 1/m
    #     extra < 1.
    #
    # case t == -1, i.e. -2**(LONG_BIT-1) <= n <= -1
    #
    #     (note that n^(-1) == ~n)
    #     0 <= ~n < 2**(LONG_BIT-1)
    #     by the previous case we get an answer a == (~n) // m
    #     ~a == n // m    because it's a division truncating towards -inf.

    if not known_nonneg:
        t_box = ResOperation(rop.INT_RSHIFT, [n_box, ConstInt(LONG_BIT - 1)])
        nt_box = ResOperation(rop.INT_XOR, [n_box, t_box])
    else:
        t_box = None
        nt_box = n_box
    mul_box = ResOperation(rop.UINT_MUL_HIGH, [nt_box, ConstInt(intmask(kk))])
    sh_box = ResOperation(rop.UINT_RSHIFT, [mul_box, ConstInt(ii)])
    if not known_nonneg:
        final_box = ResOperation(rop.INT_XOR, [sh_box, t_box])
        return [t_box, nt_box, mul_box, sh_box, final_box]
    else:
        return [mul_box, sh_box]


def modulo_operations(n_box, m, known_nonneg=False):
    operations = division_operations(n_box, m, known_nonneg)

    mul_box = ResOperation(rop.INT_MUL, [operations[-1], ConstInt(m)])
    diff_box = ResOperation(rop.INT_SUB, [n_box, mul_box])
    return operations + [mul_box, diff_box]
