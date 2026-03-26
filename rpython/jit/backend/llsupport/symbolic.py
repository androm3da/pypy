import ctypes
import struct as _struct
from rpython.rtyper.lltypesystem import lltype, ll2ctypes, llmemory, rffi
from rpython.rlib.objectmodel import specialize
from rpython.rlib.unroll import unrolling_iterable

@specialize.memo()
def get_field_token(STRUCT, fieldname, translate_support_code):
    if translate_support_code:
        return (llmemory.offsetof(STRUCT, fieldname),
                get_size(getattr(STRUCT, fieldname), True))
    cstruct = ll2ctypes.get_ctypes_type(STRUCT)
    cfield = getattr(cstruct, fieldname)
    return (cfield.offset, cfield.size)

@specialize.memo()
def get_size(TYPE, translate_support_code):
    if translate_support_code:
        if TYPE._is_varsize():
            return llmemory.sizeof(TYPE, 0)
        return llmemory.sizeof(TYPE)
    ctype = ll2ctypes.get_ctypes_type(TYPE)
    return ctypes.sizeof(ctype)

@specialize.memo()
def get_size_of_ptr(translate_support_code):
    return get_size(llmemory.GCREF, translate_support_code)

@specialize.memo()
def get_array_token(T, translate_support_code):
    # T can be an array or a var-sized structure
    if translate_support_code:
        basesize = llmemory.sizeof(T, 0)     # this includes +1 for STR
        if isinstance(T, lltype.Struct):
            SUBARRAY = getattr(T, T._arrayfld)
            itemsize = llmemory.sizeof(SUBARRAY.OF)
            ofs_length = (llmemory.offsetof(T, T._arrayfld) +
                          llmemory.ArrayLengthOffset(SUBARRAY))
        else:
            if T._hints.get('nolength', None):
                ofs_length = -1
            else:
                ofs_length = llmemory.ArrayLengthOffset(T)
            itemsize = llmemory.sizeof(T.OF)
    else:
        if isinstance(T, lltype.Struct):
            assert T._arrayfld is not None, "%r is not variable-sized" % (T,)
            cstruct = ll2ctypes.get_ctypes_type(T)
            cfield = getattr(cstruct, T._arrayfld)
            before_array_part = cfield.offset
            T = getattr(T, T._arrayfld)
        else:
            before_array_part = 0
        carray = ll2ctypes.get_ctypes_type(T)
        if T._hints.get('nolength', None):
            ofs_length = -1
        else:
            assert carray.length.size == _struct.calcsize('l')  # host size
            ofs_length = before_array_part + carray.length.offset
        basesize = before_array_part + carray.items.offset
        basesize += T._hints.get('extra_item_after_alloc', 0)  # +1 for STR
        carrayitem = ll2ctypes.get_ctypes_type(T.OF)
        itemsize = ctypes.sizeof(carrayitem)
    return basesize, itemsize, ofs_length

# ____________________________________________________________

def _get_word_size():
    """Return the target WORD size (bytes per Signed/pointer).

    For cross-compilation, use the target platform's long bit size.
    Falls back to the host's sizeof(Signed) for native compilation.
    """
    try:
        from rpython.translator.platform import platform as current_platform
        target_long_bit = getattr(current_platform, 'target_long_bit', None)
        if target_long_bit is not None:
            return target_long_bit // 8
    except ImportError:
        pass
    return rffi.sizeof(lltype.Signed)

WORD         = _get_word_size()
SIZEOF_CHAR  = get_size(lltype.Char, False)
SIZEOF_SHORT = get_size(rffi.SHORT, False)
SIZEOF_INT   = get_size(rffi.INT, False)
SIZEOF_FLOAT = get_size(lltype.Float, False)

unroll_basic_sizes = unrolling_iterable([
    (lltype.Signed,   lltype.Unsigned, WORD),
    (rffi.SIGNEDCHAR, lltype.Char,     SIZEOF_CHAR),
    (rffi.SHORT,      rffi.USHORT,     SIZEOF_SHORT),
    (rffi.INT,        rffi.UINT,       SIZEOF_INT)])
# does not contain Float ^^^ which must be special-cased
