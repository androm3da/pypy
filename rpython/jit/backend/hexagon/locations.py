"""
Location types for the Hexagon JIT backend.
"""

from rpython.jit.backend.hexagon.arch import WORD, DOUBLE_WORD, JITFRAME_FIXED_SIZE
from rpython.jit.metainterp.history import FLOAT, INT


class AssemblerLocation(object):
    _immutable_ = True
    type = INT

    def is_imm(self):
        return False

    def is_stack(self):
        return False

    def is_raw_sp(self):
        return False

    def is_core_reg(self):
        return False

    def is_fp_reg(self):
        return False

    def is_reg_pair(self):
        return False

    def is_pred_reg(self):
        return False

    def is_imm_float(self):
        return False

    def is_float(self):
        return False

    def is_vector_reg(self):
        return False

    def as_key(self):
        raise NotImplementedError

    def get_position(self):
        raise NotImplementedError


class RegisterLocation(AssemblerLocation):
    """A single general-purpose register R0-R31."""
    _immutable_ = True
    width = WORD

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return 'r%d' % self.value

    def __int__(self):
        return self.value

    def is_core_reg(self):
        return True

    def as_key(self):
        return self.value


class RegisterPairLocation(AssemblerLocation):
    """A GPR pair for 64-bit values / doubles.

    Stores the even register number. E.g., D16 represents R17:R16.
    The even register is the low word, the odd register is the high word.
    """
    _immutable_ = True
    type = FLOAT
    width = DOUBLE_WORD

    def __init__(self, even_reg):
        assert even_reg % 2 == 0, "Register pair must use even register"
        self.value = even_reg

    def __repr__(self):
        return 'r%d:r%d' % (self.value + 1, self.value)

    def __int__(self):
        return self.value

    def even_reg(self):
        """Low word register number."""
        return self.value

    def odd_reg(self):
        """High word register number."""
        return self.value + 1

    def is_reg_pair(self):
        return True

    def is_fp_reg(self):
        return True

    def is_float(self):
        return True

    def as_key(self):
        return self.value + 40


class PredicateLocation(AssemblerLocation):
    """A predicate register P0-P3."""
    _immutable_ = True
    width = 1  # 8-bit predicate register

    def __init__(self, value):
        assert 0 <= value <= 3
        self.value = value

    def __repr__(self):
        return 'p%d' % self.value

    def __int__(self):
        return self.value

    def is_pred_reg(self):
        return True

    def as_key(self):
        return self.value + 80


class ImmLocation(AssemblerLocation):
    _immutable_ = True

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return 'imm(%d)' % self.value

    def is_imm(self):
        return True


class StackLocation(AssemblerLocation):
    _immutable_ = True

    def __init__(self, position, fp_offset, type=INT):
        self.position = position
        self.value = fp_offset
        self.type = type
        if type == FLOAT:
            self.width = DOUBLE_WORD
        else:
            self.width = WORD

    def __repr__(self):
        return 'FP(%s)+%d' % (self.type, self.position)

    def get_position(self):
        return self.position

    def is_stack(self):
        return True

    def as_key(self):
        return self.position + 10000

    def is_float(self):
        return self.type == FLOAT


def get_fp_offset(base_ofs, position):
    return base_ofs + WORD * (position + JITFRAME_FIXED_SIZE)


class ConstFloatLoc(AssemblerLocation):
    """A float constant stored in the data section."""
    _immutable_ = True
    type = FLOAT

    def __init__(self, addr):
        # Address of the 8-byte float value in the data section
        self.addr = addr
        self.low_word = 0
        self.high_word = 0

    def __repr__(self):
        return 'const_float(addr=0x%x)' % (self.addr,)

    def is_imm_float(self):
        return True

    def is_float(self):
        return True

    def as_key(self):
        return self.low_word ^ (self.high_word << 16)


class HVXVectorLocation(AssemblerLocation):
    """An HVX vector register V0-V31 (128 bytes each)."""
    _immutable_ = True
    width = 128  # 1024-bit = 128 bytes

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return 'v%d' % self.value

    def __int__(self):
        return self.value

    def is_vector_reg(self):
        return True

    def as_key(self):
        return self.value + 100


class HVXVectorPairLocation(AssemblerLocation):
    """An HVX vector pair register V1:V0, V3:V2, etc. (256 bytes)."""
    _immutable_ = True
    width = 256

    def __init__(self, even_reg):
        assert even_reg % 2 == 0
        self.value = even_reg

    def __repr__(self):
        return 'v%d:v%d' % (self.value + 1, self.value)

    def __int__(self):
        return self.value

    def is_vector_reg(self):
        return True

    def as_key(self):
        return self.value + 140


class HVXPredicateLocation(AssemblerLocation):
    """An HVX vector predicate register Q0-Q3 (128 bits each)."""
    _immutable_ = True

    def __init__(self, value):
        assert 0 <= value <= 3
        self.value = value

    def __repr__(self):
        return 'q%d' % self.value

    def __int__(self):
        return self.value

    def as_key(self):
        return self.value + 180
