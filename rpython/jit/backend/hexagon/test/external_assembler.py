"""
Reference assembler interface for Hexagon instruction encoding tests.

Uses llvm-mc to assemble instructions and extract the binary encoding,
which is then compared against our JIT backend's encoding.
"""

import os
import struct
import subprocess
import sys

from rpython.tool.udir import udir

# Tool discovery: set HEXAGON_TOOLS_BIN to the directory containing
# llvm-mc / llvm-objcopy, or ensure they are on PATH.
def _find_tool(name):
    """Find a tool binary, checking HEXAGON_TOOLS_BIN then PATH."""
    env_dir = os.environ.get('HEXAGON_TOOLS_BIN')
    if env_dir:
        path = os.path.join(env_dir, name)
        if os.path.exists(path):
            return path
    return name  # fall back to bare name (PATH lookup)


LLVM_MC = _find_tool('llvm-mc')
LLVM_OBJCOPY = _find_tool('llvm-objcopy')
LLVM_OBJDUMP = _find_tool('llvm-objdump')

# Architecture flags for llvm-mc
HEXAGON_TRIPLE = 'hexagon'
HEXAGON_CPU = 'hexagonv73'
HEXAGON_HVX_ATTR = '+hvx'

# Sentinel instructions used to delimit the instruction under test.
# We use two trap0(#0xFE) before and after as markers.
BEGIN_MARKER_ASM = 'trap0(#0xFE)\n    trap0(#0xFE)'
END_MARKER_ASM = 'trap0(#0xFE)\n    trap0(#0xFE)'

# Binary encoding of trap0(#0xFE) as a single-instruction packet.
# From hexagon-objdump: trap0(#0xFE) = 0x5400DF18
# (little-endian bytes: 18 df 00 54)
BEGIN_MARKER_BIN = b'\x18\xdf\x00\x54' * 2  # two trap0(#0xFE) little-endian
END_MARKER_BIN = BEGIN_MARKER_BIN

_asm_index = [0]


BODY_TEMPLATE = """\
    .text
    .globl _start
_start:
    // begin marker
    {
        trap0(#0xFE)
    }
    {
        trap0(#0xFE)
    }
    // instruction under test
    %s
    // end marker
    {
        trap0(#0xFE)
    }
    {
        trap0(#0xFE)
    }
"""


def _tool_available():
    """Check if llvm-mc is available and supports the Hexagon target."""
    try:
        subprocess.check_output([LLVM_MC, '--version'],
                                stderr=subprocess.STDOUT)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def assemble(instr, hvx=False):
    """Assemble a single Hexagon instruction and return its binary encoding.

    Args:
        instr: Assembly instruction string, e.g. "{ R0 = add(R1, R2) }"
        hvx: If True, enable HVX instructions.

    Returns:
        bytes: The raw binary encoding of the instruction (4 bytes per
               instruction word, little-endian).

    Raises:
        AssertionError if assembly fails or markers not found.
    """
    idx = _asm_index[0]
    _asm_index[0] = idx + 1

    src_file = udir.join('hextest_%d.s' % idx)
    obj_file = udir.join('hextest_%d.o' % idx)
    bin_file = udir.join('hextest_%d.bin' % idx)

    # Write assembly source
    asm_text = BODY_TEMPLATE % instr
    src_file.write(asm_text)

    # Assemble with llvm-mc
    cmd = [LLVM_MC, '--triple=' + HEXAGON_TRIPLE,
           '--mcpu=' + HEXAGON_CPU, '--filetype=obj',
           '-o', str(obj_file), str(src_file)]
    if hvx:
        cmd.append('--mattr=' + HEXAGON_HVX_ATTR)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise AssertionError(
            "llvm-mc assembly failed (rc=%d):\n%s\nSource:\n%s" %
            (proc.returncode, stderr.decode('utf-8', 'replace'), asm_text))

    # Extract .text section as raw binary
    cmd = [LLVM_OBJCOPY, '-O', 'binary', '-j', '.text',
           str(obj_file), str(bin_file)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise AssertionError(
            "llvm-objcopy failed (rc=%d):\n%s" %
            (proc.returncode, stderr.decode('utf-8', 'replace')))

    # Read binary and extract instruction between markers
    data = bin_file.read(mode='rb')

    begin = data.find(BEGIN_MARKER_BIN)
    assert begin >= 0, "begin marker not found in assembled binary"
    begin += len(BEGIN_MARKER_BIN)

    end = data.find(END_MARKER_BIN, begin)
    assert end >= 0, "end marker not found in assembled binary"

    instr_bytes = data[begin:end]
    assert len(instr_bytes) > 0, "no instruction bytes between markers"
    assert len(instr_bytes) % 4 == 0, (
        "instruction bytes not 4-byte aligned: %d bytes" % len(instr_bytes))

    return instr_bytes


def assemble_to_words(instr, hvx=False):
    """Like assemble(), but returns a list of 32-bit integers (little-endian)."""
    data = assemble(instr, hvx=hvx)
    words = []
    for i in range(0, len(data), 4):
        words.append(struct.unpack('<I', data[i:i+4])[0])
    return words


def disassemble(binary_data):
    """Disassemble raw binary Hexagon machine code (for debugging).

    Args:
        binary_data: bytes of machine code

    Returns:
        str: disassembly text
    """
    idx = _asm_index[0]
    _asm_index[0] = idx + 1

    bin_file = udir.join('hexdis_%d.bin' % idx)
    bin_file.write(binary_data, mode='wb')

    cmd = [LLVM_OBJDUMP, '-d', '-b', 'binary',
           '-m', 'hexagon', str(bin_file)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    return stdout.decode('utf-8', 'replace')
