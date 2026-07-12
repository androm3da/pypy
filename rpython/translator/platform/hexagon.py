"""Support for cross-compilation to Qualcomm Hexagon (hexagon-linux-musl)."""

import os

from rpython.translator.platform.linux import Linux
from rpython.translator.platform.posix import _run_subprocess, GnuMakefile
from rpython.translator.platform import ExecutionResult, log

TOOLCHAIN = os.environ.get('HEXAGON_TOOLCHAIN')
if TOOLCHAIN is None:
    log.error('HEXAGON_TOOLCHAIN: Provide a path to the Hexagon '
              'cross-compilation toolchain in env variable HEXAGON_TOOLCHAIN')
    assert 0, ('Set HEXAGON_TOOLCHAIN to the root of a '
               'clang+llvm-*-cross-hexagon-unknown-linux-musl toolchain')

SYSROOT = os.environ.get(
    'HEXAGON_SYSROOT',
    os.path.join(TOOLCHAIN, 'target', 'hexagon-unknown-linux-musl'))
QEMU = os.environ.get(
    'HEXAGON_QEMU',
    os.path.join(TOOLCHAIN, 'bin', 'qemu-hexagon'))

# Optional prefix (containing include/ and lib/) with extra cross-compiled
# dependencies that the toolchain sysroot lacks, notably libffi.
LIBFFI_PREFIX = os.environ.get('HEXAGON_LIBFFI')


class Hexagon(Linux):
    name = "hexagon"
    target_long_bit = 32
    target_supports_int128 = False
    shared_only = ('-fPIC',)

    available_librarydirs = [
        os.path.join(SYSROOT, 'lib'),
        os.path.join(SYSROOT, 'usr', 'lib'),
    ]
    available_includedirs = [
        os.path.join(SYSROOT, 'usr', 'include'),
    ]
    if LIBFFI_PREFIX:
        available_librarydirs.append(os.path.join(LIBFFI_PREFIX, 'lib'))
        available_includedirs.append(os.path.join(LIBFFI_PREFIX, 'include'))

    cflags = tuple(
        ['-O3', '-pthread', '-mv73',
         '-mlong-calls',
         '-Wall', '-Wno-unused', '-Wno-address',
         '-Wno-ignored-qualifiers',
         '-Wno-duplicate-decl-specifier',
         ]
        + os.environ.get('CFLAGS', '').split())

    link_flags = tuple(
        ['-pthread',
         '--sysroot=' + SYSROOT,
         ]
        + os.environ.get('LDFLAGS', '').split())

    extra_libs = ('-lrt',)

    def __init__(self, cc=None):
        if cc is None:
            cc = os.path.join(TOOLCHAIN, 'bin', 'hexagon-linux-musl-clang')
        self.cc = cc

    def _execute_c_compiler(self, cc, args, outname, cwd=None):
        log.execute(cc + ' ' + ' '.join(args))
        cclist = cc.split()
        cc = cclist[0]
        args = cclist[1:] + args
        returncode, stdout, stderr = _run_subprocess(
            cc, args, self.c_environ, cwd)
        self._handle_error(returncode, stdout, stderr, outname)

    def execute(self, executable, args=None, env=None, compilation_info=None):
        if env is None:
            env = os.environ.copy()
        else:
            env = env.copy()
        if args is None:
            args = []
        qemu_args = ['-L', SYSROOT, str(executable)] + args
        log.execute('qemu-hexagon ' + ' '.join(qemu_args))
        returncode, stdout, stderr = _run_subprocess(
            QEMU, qemu_args, env)
        return ExecutionResult(returncode, stdout, stderr)

    def execute_makefile(self, path_to_makefile, extra_opts=[]):
        if isinstance(path_to_makefile, GnuMakefile):
            path = path_to_makefile.makefile_dir
        else:
            path = path_to_makefile
        log.execute('make %s in %s' % (" ".join(extra_opts), path))
        env = os.environ.copy()
        env['CC'] = self.cc
        returncode, stdout, stderr = _run_subprocess(
            self.make_cmd, ['-C', str(path)] + extra_opts, env)
        self._handle_error(returncode, stdout, stderr, path.join('make'))

    def include_dirs_for_libffi(self):
        return self.available_includedirs

    def library_dirs_for_libffi(self):
        return self.available_librarydirs

    def _preprocess_library_dirs(self, library_dirs):
        return list(library_dirs) + self.available_librarydirs

    def get_multiarch(self):
        return 'hexagon-unknown-linux-musl'
