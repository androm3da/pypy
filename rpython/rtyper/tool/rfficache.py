
# XXX This is completely outdated file, kept here only for bootstrapping
#     reasons. If you touch it, try removing it

import py
import os
from rpython.translator.tool.cbuild import ExternalCompilationInfo
from rpython.tool.udir import udir
from rpython.rlib import rarithmetic
from rpython.rtyper.lltypesystem import lltype
from rpython.tool.gcc_cache import build_executable_cache

def ask_gcc(question, add_source="", ignore_errors=False):
    from rpython.translator.platform import platform
    includes = ['stdlib.h', 'stdio.h', 'sys/types.h']
    if platform.name != 'msvc':
        includes += ['inttypes.h', 'stddef.h']
    include_string = "\n".join(["#include <%s>" % i for i in includes])
    c_source = py.code.Source('''
    // includes
    %s

    %s

    // checking code
    int main(void)
    {
       %s
       return (0);
    }
    ''' % (include_string, add_source, str(question)))
    c_file = udir.join("gcctest.c")
    c_file.write(str(c_source) + '\n')
    eci = ExternalCompilationInfo()
    return build_executable_cache([c_file], eci, ignore_errors=ignore_errors)

def sizeof_c_type(c_typename, **kwds):
    return sizeof_c_types([c_typename], **kwds)[0]

def sizeof_c_types(typenames_c, **kwds):
    lines = ['printf("sizeof %s=%%ld\\n", (long)sizeof(%s));' % (c_typename,
                                                                 c_typename)
             for c_typename in typenames_c]
    question = '\n\t'.join(lines)
    answer = ask_gcc(question, **kwds)
    lines = answer.splitlines()
    assert len(lines) == len(typenames_c)
    result = []
    for line, c_typename in zip(lines, typenames_c):
        answer = line.split('=')
        assert answer[0] == "sizeof " + c_typename
        result.append(int(answer[1]))
    return result

def signof_c_type(c_typename, **kwds):
    question = 'printf("sign %s=%%d\\n", ((%s) -1) <= (%s)0);' % (c_typename,
                                                                  c_typename,
                                                                  c_typename)
    answer = ask_gcc(question, **kwds).strip()
    if answer == 'sign %s=0' % (c_typename,):
        return False
    if answer == 'sign %s=1' % (c_typename,):
        return True
    raise ValueError(answer)


class Platform:
    def __init__(self):
        self.types = {}
        self.numbertype_to_rclass = {}
        self._type_info = {}   # name -> (signed, size_in_bytes)

    def inttype(self, name, c_name, signed, **kwds):
        try:
            return self.types[name]
        except KeyError:
            size = sizeof_c_type(c_name, **kwds)
            return self._make_type(name, signed, size)

    def _make_type(self, name, signed, size):
        self._type_info[name] = (signed, size)
        inttype = rarithmetic.build_int('r_' + name, signed, size*8)
        tp = lltype.build_number(name, inttype)
        # For cross-compilation: if this type aliased Signed/Unsigned but
        # has a different size than the target's 'long', create a distinct
        # Number type.  This handles the case where LONGLONG (8 bytes on
        # a 32-bit target) would alias Signed because r_longlong == r_int
        # on the 64-bit host.
        if tp is lltype.Signed or tp is lltype.Unsigned:
            from rpython.translator.platform import platform as _platform
            target_long_bit = getattr(_platform, 'target_long_bit', None)
            if target_long_bit is not None and size * 8 != target_long_bit:
                tp = lltype.Number(name, inttype)
        self.numbertype_to_rclass[tp] = inttype
        self.types[name] = tp
        return tp

    def update_types_for_cross_compilation(self, target_long_bit):
        """Re-evaluate types after the cross-compilation platform is set.

        During initial setup, types are detected on the host compiler.
        On a 64-bit host, 8-byte signed types (time_t, long long, etc.)
        alias to Signed (= long = 8 bytes).  But on a 32-bit target,
        Signed = long = 4 bytes, so these types must become distinct
        Number types that map to 'long long' in C.
        """
        updates = {}
        for name, (signed, size) in self._type_info.items():
            tp = self.types[name]
            if (tp is lltype.Signed or tp is lltype.Unsigned):
                if size * 8 != target_long_bit:
                    inttype = rarithmetic.build_int(
                        'r_' + name, signed, size * 8,
                        force_creation=True)
                    new_tp = lltype.build_number(name, inttype)
                    self.numbertype_to_rclass[new_tp] = inttype
                    self.types[name] = new_tp
                    updates[name] = new_tp
        return updates

    def populate_inttypes(self, list, **kwds):
        """'list' is a list of (name, c_name, signed)."""
        missing = []
        names_c = []
        for name, c_name, signed in list:
            if name not in self.types:
                missing.append((name, signed))
                names_c.append(c_name)
        if names_c:
            sizes = sizeof_c_types(names_c, **kwds)
            assert len(sizes) == len(missing)
            for (name, signed), size in zip(missing, sizes):
                self._make_type(name, signed, size)

platform = Platform()
