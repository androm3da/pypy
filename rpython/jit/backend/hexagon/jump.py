"""
Frame layout remapping for the Hexagon JIT backend.

Handles moving values between source and destination locations, including
cycle detection and resolution through temporary push/pop operations.
"""

from rpython.jit.backend.hexagon import registers as r


def remap_frame_layout(assembler, src_locations, dst_locations, tmpreg):
    pending_dests = len(dst_locations)
    srccount = {}    # maps dst_locations to how many times the same
                     # location appears in src_locations
    for dst in dst_locations:
        key = dst.as_key()
        assert key not in srccount, "duplicate value in dst_locations!"
        srccount[key] = 0
    for i in range(len(dst_locations)):
        src = src_locations[i]
        if src.is_imm():
            continue
        key = src.as_key()
        if key in srccount:
            if key == dst_locations[i].as_key():
                # ignore a move "x = x"
                srccount[key] = -len(dst_locations) - 1
                pending_dests -= 1
            else:
                srccount[key] += 1

    while pending_dests > 0:
        progress = False
        for i in range(len(dst_locations)):
            dst = dst_locations[i]
            key = dst.as_key()
            if srccount[key] == 0:
                srccount[key] = -1       # means "it's done"
                pending_dests -= 1
                src = src_locations[i]
                if not src.is_imm():
                    key = src.as_key()
                    if key in srccount:
                        srccount[key] -= 1
                _move(assembler, src, dst, tmpreg)
                progress = True
        if not progress:
            # we are left with only pure disjoint cycles
            sources = {}     # maps dst_locations to src_locations
            for i in range(len(dst_locations)):
                src = src_locations[i]
                dst = dst_locations[i]
                sources[dst.as_key()] = src
            #
            for i in range(len(dst_locations)):
                dst = dst_locations[i]
                originalkey = dst.as_key()
                if srccount[originalkey] >= 0:
                    assembler.push_locations([dst])
                    while True:
                        key = dst.as_key()
                        assert srccount[key] == 1
                        srccount[key] = -1
                        pending_dests -= 1
                        src = sources[key]
                        if src.as_key() == originalkey:
                            break
                        _move(assembler, src, dst, tmpreg)
                        dst = src
                    assembler.pop_locations([dst])
            assert pending_dests == 0


def _move(assembler, src, dst, tmpreg):
    if dst.is_stack() and src.is_stack():
        assembler.regalloc_mov(src, tmpreg)
        src = tmpreg
    assembler.regalloc_mov(src, dst)


def remap_frame_layout_mixed(assembler,
                             src_locations1, dst_locations1, tmpreg1,
                             src_locations2, dst_locations2, tmpreg2):
    """Handle remapping with two sets of locations (e.g., int + float).

    Finds float/pair stack locations from src_locations2 that would be
    overwritten by dst_locations1, and temporarily pushes them.
    """
    extrapushes = []
    extrapops = []
    dst_keys = {}
    for loc in dst_locations1:
        dst_keys[loc.as_key()] = None
    src_locations2red = []
    dst_locations2red = []
    for i in range(len(src_locations2)):
        loc = src_locations2[i]
        dstloc = dst_locations2[i]
        if loc.is_stack():
            key = loc.as_key()
            if key in dst_keys:
                extrapushes.append(loc)
                extrapops.append(dstloc)
                continue
        src_locations2red.append(loc)
        dst_locations2red.append(dstloc)
    src_locations2 = src_locations2red
    dst_locations2 = dst_locations2red

    assembler.push_locations(extrapushes)

    # remap the integer registers and stack locations
    remap_frame_layout(assembler, src_locations1, dst_locations1, tmpreg1)

    # remap the float/pair registers and stack locations
    remap_frame_layout(assembler, src_locations2, dst_locations2, tmpreg2)

    # pop the extra float/pair stack locations
    assembler.pop_locations(extrapops)
