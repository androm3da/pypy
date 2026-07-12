import sys
import os
import platform as _stdlib_platform
from rpython.config.config import OptionDescription, BoolOption, IntOption, ArbitraryOption, FloatOption
from rpython.config.config import ChoiceOption, StrOption, Config, ConflictConfigError
from rpython.config.config import ConfigError
from rpython.config.support import detect_number_of_processors
from rpython.translator.platform import platform as compiler


DEFL_INLINE_THRESHOLD = 32.4    # just enough to inline add__Int_Int()
# and just small enough to prevend inlining of some rlist functions.

DEFL_PROF_BASED_INLINE_THRESHOLD = 32.4
DEFL_LOW_INLINE_THRESHOLD = DEFL_INLINE_THRESHOLD / 2.0

DEFL_GC = "incminimark"   # XXX

DEFL_ROOTFINDER_WITHJIT = "shadowstack"

def _is_target_64_bits():
    try:
        from rpython.translator.platform import platform as compiler
        target_long_bit = getattr(compiler, 'target_long_bit', None)
        if target_long_bit is not None:
            return target_long_bit > 32
    except (ImportError, AttributeError):
        pass
    return sys.maxint > 2147483647

IS_64_BITS = _is_target_64_bits()

SUPPORT__THREAD = (    # whether the particular C compiler supports __thread
    sys.platform.startswith("linux") or     # Linux works
    #sys.platform.startswith("darwin") or   # OS/X >= 10.7 works (*)
    False)
    # Windows doesn't work.  Please
    # add other platforms here if it works on them.
MACHINE = _stdlib_platform.machine()
if MACHINE == 'aarch64':
    SUPPORT__THREAD = False
# (*) NOTE: __thread on OS/X does not work together with
# pthread_key_create(): when the destructor is called, the __thread is
# already freed!

MAINDIR = os.path.dirname(os.path.dirname(__file__))
CACHE_DIR = os.path.realpath(os.path.join(MAINDIR, '_cache', MACHINE))

PLATFORMS = [
    'host',
    'arm',
    'hexagon',
]

translation_optiondescription = OptionDescription(
        "translation", "Translation Options", [
    BoolOption("continuation", "enable single-shot continuations",
               default=False, cmdline="--continuation",
               requires=[("translation.type_system", "lltype")]),
    ChoiceOption("type_system", "Type system to use when RTyping",
                 ["lltype"], cmdline=None, default="lltype"),
    ChoiceOption("backend", "Backend to use for code generation",
                 ["c"], default="c",
                 requires={
                     "c":      [("translation.type_system", "lltype")],
                     },
                 cmdline="-b --backend"),

    BoolOption("shared", "Build as a shared library",
               default=False, cmdline="--shared"),

    BoolOption("log", "Include debug prints in the translation (PYPYLOG=...)",
               default=True, cmdline="--log"),

    # gc
    ChoiceOption("gc", "Garbage Collection Strategy",
                 ["boehm", "ref", "semispace", "statistics",
                  "generation", "hybrid", "minimark",'incminimark', "none"],
                  "ref", requires={
                     "ref": [("translation.rweakref", False), # XXX
                             ("translation.gctransformer", "ref")],
                     "none": [("translation.rweakref", False), # XXX
                             ("translation.gctransformer", "none")],
                     "semispace": [("translation.gctransformer", "framework")],
                     "statistics": [("translation.gctransformer", "framework")],
                     "generation": [("translation.gctransformer", "framework")],
                     "hybrid": [("translation.gctransformer", "framework")],
                     "boehm": [("translation.continuation", False),  # breaks
                               ("translation.gctransformer", "boehm")],
                     "minimark": [("translation.gctransformer", "framework")],
                     "incminimark": [("translation.gctransformer", "framework")],
                     },
                  cmdline="--gc"),
    ChoiceOption("gctransformer", "GC transformer that is used - internal",
                 ["boehm", "ref", "framework", "none"],
                 default="ref", cmdline=None,
                 requires={
                     "boehm": [("translation.gcrootfinder", "n/a"),
                               ("translation.gcremovetypeptr", False)],
                     "ref": [("translation.gcrootfinder", "n/a"),
                             ("translation.gcremovetypeptr", False)],
                     "none": [("translation.gcrootfinder", "n/a"),
                              ("translation.gcremovetypeptr", False)],
                 }),
    BoolOption("gcremovetypeptr", "Remove the typeptr from every object",
               default=IS_64_BITS, cmdline="--gcremovetypeptr"),
    ChoiceOption("gcrootfinder",
                 "Strategy for finding GC Roots (framework GCs only)",
                 ["n/a", "shadowstack"],
                 "shadowstack",
                 cmdline="--gcrootfinder",
                 requires={
                     "shadowstack": [("translation.gctransformer", "framework")],
                    }),

    # other noticeable options
    BoolOption("thread", "enable use of threading primitives",
               default=False, cmdline="--thread"),
    BoolOption("sandbox", "Produce a fully-sandboxed executable",
               default=False, cmdline="--sandbox",
               suggests=[("translation.gc", "generation"),
                         ("translation.gcrootfinder", "shadowstack"),
                         ("translation.thread", False),
                        ]),
    BoolOption("rweakref", "The backend supports RPython-level weakrefs",
               default=True),

    # JIT generation: use -Ojit to enable it
    BoolOption("jit", "generate a JIT",
               default=False,
               suggests=[("translation.gc", DEFL_GC),
                         ("translation.gcrootfinder", DEFL_ROOTFINDER_WITHJIT),
                         ("translation.list_comprehension_operations", True)]),
    ChoiceOption("jit_backend", "choose the backend for the JIT",
                 ["auto", "x86", "x86-without-sse2", 'arm', 'hexagon'],
                 default="auto", cmdline="--jit-backend"),
    ChoiceOption("jit_profiler", "integrate profiler support into the JIT",
                 ["off", "oprofile"],
                 default="off"),
    BoolOption("check_str_without_nul",
               "Forbid NUL chars in strings in some external function calls",
               default=False, cmdline=None),

    # misc
    BoolOption("verbose", "Print extra information", default=False,
               cmdline="--verbose"),
    StrOption("cc", "Specify compiler to use for compiling generated C", cmdline="--cc"),
    BoolOption("profopt", "Enable profile guided optimization. Defaults to enabling this for PyPy. For other training workloads, please specify them in profoptargs",
              cmdline="--profopt", default=False),
    StrOption("profoptargs", "Absolute path to the profile guided optimization training script + the necessary arguments of the script", cmdline="--profoptargs", default=None),
    BoolOption("instrument", "internal: turn instrumentation on",
               default=False, cmdline=None),
    BoolOption("countmallocs", "Count mallocs and frees", default=False,
               cmdline=None),
    BoolOption("countfieldaccess", "Count field access for C structs",
            default=False),
    ChoiceOption("fork_before",
                 "(UNIX) Create restartable checkpoint before step",
                 ["annotate", "rtype", "backendopt", "database", "source",
                  "pyjitpl"],
                 default=None, cmdline="--fork-before"),
    BoolOption("dont_write_c_files",
               "Make the C backend write everyting to /dev/null. " +
               "Useful for benchmarking, so you don't actually involve the disk",
               default=False, cmdline="--dont-write-c-files"),
    ArbitraryOption("instrumentctl", "internal",
               default=None),
    StrOption("output", "Output file name", cmdline="--output"),
    StrOption("secondaryentrypoints",
            "Comma separated list of keys choosing secondary entrypoints",
            cmdline="--entrypoints", default="main"),

    BoolOption("dump_static_data_info", "Dump static data info",
               cmdline="--dump_static_data_info",
               default=False, requires=[("translation.backend", "c")]),

    # portability options
    BoolOption("no__thread",
               "don't use __thread for implementing TLS",
               default=not SUPPORT__THREAD, cmdline="--no__thread",
               negation=False),
    IntOption("make_jobs", "Specify -j argument to make for compilation"
              " (C backend only)",
              cmdline="--make-jobs", default=detect_number_of_processors()),

    # Flags of the TranslationContext:
    BoolOption("list_comprehension_operations",
               "When true, look for and special-case the sequence of "
               "operations that results from a list comprehension and "
               "attempt to pre-allocate the list",
               default=False,
               cmdline='--listcompr'),
    IntOption("withsmallfuncsets",
              "Represent groups of less funtions than this as indices into an array",
               default=0),
    BoolOption("taggedpointers",
               "When true, enable the use of tagged pointers. "
               "If false, use normal boxing",
               default=False),
    BoolOption("keepgoing",
               "Continue annotating when errors are encountered, and report "
               "them all at the end of the annotation phase",
               default=False, cmdline="--keepgoing"),
    BoolOption("lldebug",
               "If true, makes an lldebug build", default=False,
               cmdline="--lldebug"),
    BoolOption("lldebug0",
               "If true, makes an lldebug0 build", default=False,
               cmdline="--lldebug0"),
    BoolOption("lto", "enable link time optimization",
               default=False, cmdline="--lto",
               requires=[("translation.gcrootfinder", "shadowstack")]),
    StrOption("icon", "Path to the (Windows) icon to use for the executable"),
    StrOption("manifest",
              "Path to the (Windows) manifest to embed in the executable"),
    StrOption("libname",
              "Windows: name and possibly location of the lib file to create"),

    OptionDescription("backendopt", "Backend Optimization Options", [
        # control inlining
        BoolOption("inline", "Do basic inlining and malloc removal",
                   default=True),
        FloatOption("inline_threshold", "Threshold when to inline functions",
                  default=DEFL_INLINE_THRESHOLD, cmdline="--inline-threshold"),
        StrOption("inline_heuristic", "Dotted name of an heuristic function "
                  "for inlining",
                default="rpython.translator.backendopt.inline.inlining_heuristic",
                cmdline="--inline-heuristic"),

        BoolOption("print_statistics", "Print statistics while optimizing",
                   default=False),
        BoolOption("merge_if_blocks", "Merge if ... elif chains",
                   cmdline="--if-block-merge", default=True),
        BoolOption("mallocs", "Remove mallocs", default=True),
        BoolOption("constfold", "Constant propagation",
                   default=True),
        # control profile based inlining
        StrOption("profile_based_inline",
                  "Use call count profiling to drive inlining"
                  ", specify arguments",
                  default=None),   # cmdline="--prof-based-inline" fix me
        FloatOption("profile_based_inline_threshold",
                    "Threshold when to inline functions "
                    "for profile based inlining",
                  default=DEFL_PROF_BASED_INLINE_THRESHOLD,
                  ),   # cmdline="--prof-based-inline-threshold" fix me
        StrOption("profile_based_inline_heuristic",
                  "Dotted name of an heuristic function "
                  "for profile based inlining",
                default="rpython.translator.backendopt.inline.inlining_heuristic",
                ),  # cmdline="--prof-based-inline-heuristic" fix me

        BoolOption("remove_asserts",
                   "Remove operations that look like 'raise AssertionError', "
                   "which lets the C optimizer remove the asserts",
                   default=False),
        BoolOption("really_remove_asserts",
                   "Really remove operations that look like 'raise AssertionError', "
                   "without relying on the C compiler",
                   default=False),

        BoolOption("storesink", "Perform store sinking", default=True),
        BoolOption("replace_we_are_jitted",
                   "Replace we_are_jitted() calls by False",
                   default=False, cmdline=None),
        BoolOption("none",
                   "Do not run any backend optimizations",
                   requires=[('translation.backendopt.inline', False),
                             ('translation.backendopt.inline_threshold', 0),
                             ('translation.backendopt.merge_if_blocks', False),
                             ('translation.backendopt.mallocs', False),
                             ('translation.backendopt.constfold', False)])
    ]),

    ChoiceOption("platform",
                 "target platform", ['host'] + PLATFORMS, default='host',
                 cmdline='--platform',
                 suggests={"arm": [("translation.gcrootfinder", "shadowstack"),
                                   ("translation.jit_backend", "arm")],
                           "hexagon": [("translation.gcrootfinder", "shadowstack"),
                                       ("translation.jit_backend", "hexagon")]}),

    BoolOption("split_gc_address_space",
               "Ensure full separation of GC and non-GC pointers", default=False),
    BoolOption("reverse_debugger",
               "Give an executable that writes a log file for reverse debugging",
               default=False, cmdline='--revdb',
               requires=[('translation.split_gc_address_space', True),
                         ('translation.jit', False),
                         ('translation.gc', 'boehm'),
                         ('translation.continuation', False)]),
    BoolOption("rpython_translate",
               "Set to true by rpython/bin/rpython and translate.py",
               default=False),
])

def get_combined_translation_config(other_optdescr=None,
                                    existing_config=None,
                                    overrides=None,
                                    translating=False):
    if overrides is None:
        overrides = {}
    d = BoolOption("translating",
                   "indicates whether we are translating currently",
                   default=False, cmdline=None)
    if other_optdescr is None:
        children = []
        newname = ""
    else:
        children = [other_optdescr]
        newname = other_optdescr._name
    if existing_config is None:
        children += [d, translation_optiondescription]
    else:
        children += [child for child in existing_config._cfgimpl_descr._children
                         if child._name != newname]
    descr = OptionDescription("pypy", "all options", children)
    config = Config(descr, **overrides)
    if translating:
        config.translating = True
    if existing_config is not None:
        for child in existing_config._cfgimpl_descr._children:
            if child._name == newname:
                continue
            value = getattr(existing_config, child._name)
            config._cfgimpl_values[child._name] = value
    return config

# ____________________________________________________________

OPT_LEVELS = ['0', '1', 'size', 'mem', '2', '3', 'jit']
DEFAULT_OPT_LEVEL = '2'

OPT_TABLE_DOC = {
    '0':    'No optimization.  Uses the Boehm GC.',
    '1':    'Enable a default set of optimizations.  Uses the Boehm GC.',
    'size': 'Optimize for the size of the executable.  Uses the Boehm GC.',
    'mem':  'Optimize for run-time memory usage and use a memory-saving GC.',
    '2':    'Enable most optimizations and use a high-performance GC.',
    '3':    'Enable all optimizations and use a high-performance GC.',
    'jit':  'Enable the JIT.',
    }

OPT_TABLE = {
    #level:  gc          backend optimizations...
    '0':    'boehm       nobackendopt',
    '1':    'boehm       lowinline',
    'size': 'boehm       lowinline     remove_asserts',
    'mem':  DEFL_GC + '  lowinline     remove_asserts    removetypeptr',
    '2':    DEFL_GC + '  extraopts',
    '3':    DEFL_GC + '  extraopts     remove_asserts',
    'jit':  DEFL_GC + '  extraopts     jit',
    }

def set_opt_level(config, level):
    """Apply optimization suggestions on the 'config'.
    The optimizations depend on the selected level and possibly on the backend.
    """
    try:
        opts = OPT_TABLE[level]
    except KeyError:
        raise ConfigError("no such optimization level: %r" % (level,))
    words = opts.split()
    gc = words.pop(0)

    # set the GC (only meaningful with lltype)
    # but only set it if it wasn't already suggested to be something else
    if config.translation._cfgimpl_value_owners['gc'] != 'suggested':
        config.translation.suggest(gc=gc)

    # set the backendopts
    for word in words:
        if word == 'nobackendopt':
            config.translation.backendopt.suggest(none=True)
        elif word == 'lowinline':
            config.translation.backendopt.suggest(inline_threshold=
                                                DEFL_LOW_INLINE_THRESHOLD)
        elif word == 'remove_asserts':
            config.translation.backendopt.suggest(remove_asserts=True)
        elif word == 'extraopts':
            config.translation.suggest(withsmallfuncsets=5)
        elif word == 'jit':
            config.translation.suggest(jit=True)
        elif word == 'removetypeptr':
            config.translation.suggest(gcremovetypeptr=True)
        else:
            raise ValueError(word)

    # list_comprehension_operations is needed for translation, because
    # make_sure_not_resized often relies on it, so we always enable them
    config.translation.suggest(list_comprehension_operations=True)

    # finally, make the choice of the gc definitive.  This will fail
    # if we have specified strange inconsistent settings.
    config.translation.gc = config.translation.gc


# ----------------------------------------------------------------

def set_platform(config):
    global IS_64_BITS
    # Ensure rffi types are initialized from the HOST compiler before
    # switching to a cross-compilation target platform.  rffi.setup()
    # runs at import time, querying the current compiler for type sizes.
    # By importing rffi here we guarantee it uses the host compiler,
    # so that ll2ctypes (untranslated mode) works with native types and
    # __int128_t support is detected from the host, not the target.
    import rpython.rtyper.lltypesystem.rffi  # noqa: F401
    from rpython.translator.platform import set_platform
    set_platform(config.translation.platform, config.translation.cc)
    IS_64_BITS = _is_target_64_bits()
    if not IS_64_BITS:
        config.translation.suggest(gcremovetypeptr=False)
    # Update llgroup's HALFSHIFT/HALFWORD for the target word size.
    # llgroup was already imported (via rffi above) with host LONG_BIT,
    # so we must patch it for cross-compilation targets.
    from rpython.translator.platform import platform as _platform
    target_long_bit = getattr(_platform, 'target_long_bit', None)
    if target_long_bit is not None:
        from rpython.rtyper.lltypesystem.llgroup import _update_for_target_long_bit
        _update_for_target_long_bit(target_long_bit)
        # Fix rffi types that aliased Signed/Unsigned on the 64-bit host
        # but must be distinct types on the cross-compilation target.
        # E.g. time_t = 8 bytes on both host and target, but on the host
        # it aliases Signed (= long = 8 bytes) while on a 32-bit target
        # Signed = long = 4 bytes, so time_t needs its own C type.
        from rpython.rtyper.lltypesystem.rffi import _update_types_for_cross_compilation
        _update_types_for_cross_compilation(target_long_bit)
        # Fix JIT optimizer's MAXINT/MININT/IS_64_BIT/LONG_BIT which are
        # derived from the host's sys.maxint at import time.
        from rpython.jit.metainterp.optimizeopt.intutils import (
            _update_for_target_long_bit as _update_intutils)
        _update_intutils(target_long_bit)
        # Also update the auto-generated optimizer rules which have their
        # own copies of MAXINT/MININT/LONG_BIT.
        from rpython.jit.metainterp.optimizeopt.autogenintrules import (
            _update_for_target_long_bit as _update_autogen)
        _update_autogen()
        # Update LONG_BIT in JIT codewriter helpers (int_abs, floordiv, mod)
        from rpython.jit.codewriter.support import (
            _update_long_bit as _update_codewriter)
        _update_codewriter(target_long_bit)
        # Fix uint_mul_high which dispatches on LONG_BIT to choose between
        # r_ulonglong fast-path and manual digit-splitting.  On a 32-bit
        # target we need the r_ulonglong path (32 < 64).
        from rpython.rlib.rarithmetic import _update_uint_mul_high_bits
        _update_uint_mul_high_bits(target_long_bit)
        # Fix the str.find/rfind/count bloom filter width, which must
        # fit in the target's Signed.
        from rpython.rlib.rstring import (
            _update_for_target_long_bit as _update_rstring)
        _update_rstring(target_long_bit)
        # Fix OVF_DIGITS in string_to_int which uses host maxint to skip
        # overflow checking.  On 32-bit target, 10-digit numbers overflow.
        from rpython.rlib.rarithmetic import _update_ovf_digits
        _update_ovf_digits(target_long_bit)
        # Fix rbigint digit sizes: SHIFT=63 (from 64-bit host with __int128)
        # is too large for 32-bit target where Signed is only 32 bits.
        # Reset to SHIFT=31 (native 32-bit configuration).
        from rpython.rlib.rbigint import _update_rbigint_for_target
        _update_rbigint_for_target(target_long_bit)
        # Rebind SignedLongLong/UnsignedLongLong in rint.py to the new
        # cross-compilation types so that llong helper functions use the
        # correct 64-bit types at annotation time.
        from rpython.rtyper.rint import (
            _update_for_cross_compilation as _update_rint)
        _update_rint()
        # Rebind r_longlong/r_ulonglong and LONGLONG/ULONGLONG references
        # in the JIT codewriter support module for cross-compilation.
        from rpython.jit.codewriter.support import (
            _update_for_cross_compilation as _update_support)
        _update_support()

def get_platform(config):
    from rpython.translator.platform import pick_platform
    opt = config.translation.platform
    cc = config.translation.cc
    return pick_platform(opt, cc)



# when running a translation, this is patched
# XXX evil global variable
_GLOBAL_TRANSLATIONCONFIG = None


def get_translation_config():
    """ Return the translation config when translating. When running
    un-translated returns None """
    return _GLOBAL_TRANSLATIONCONFIG

