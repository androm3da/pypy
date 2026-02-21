"""
Hexagon CPU and HVX feature detection.

All detection is done at import time (via rffi_platform), since
rffi_platform functions create closures and cannot be called from
RPython at runtime.
"""

from rpython.rtyper.tool.rffi_platform import getdefined, getdefinedinteger


def _detect_hexagon():
    return getdefined('__hexagon__', '')


def _detect_hexagon_version():
    for v in [81, 79, 77, 75, 73, 71, 69, 68, 67, 66, 65, 62, 60]:
        macro = '__HEXAGON_V%d__' % v
        if getdefined(macro, ''):
            return v
    ver = getdefinedinteger('__HEXAGON_ARCH__', '')
    if ver is not None:
        return ver
    return 0


def _detect_hvx():
    return getdefined('__HVX__', '')


def _detect_hvx_length():
    length = getdefinedinteger('__HVX_LENGTH__', '')
    if length is not None:
        return length
    return 128


# Cache results at import time
_is_hexagon = _detect_hexagon()
_hexagon_version = _detect_hexagon_version()
_has_hvx = _detect_hvx()
_hvx_length = _detect_hvx_length()


def detect_hexagon():
    """Detect if we're compiling for Hexagon."""
    return _is_hexagon


def detect_hexagon_version():
    """Detect the Hexagon architecture version."""
    return _hexagon_version


def detect_hvx():
    """Detect if HVX (Hexagon Vector Extensions) is available."""
    return _has_hvx


def detect_hvx_length():
    """Detect HVX vector length in bytes (64 or 128)."""
    return _hvx_length
