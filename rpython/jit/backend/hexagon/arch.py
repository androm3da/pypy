"""
Hexagon architecture constants for the JIT backend.
"""

WORD = 4
DOUBLE_WORD = 8
INST_SIZE = 4  # all Hexagon instructions are 32-bit

# Register partitioning (see registers.py for details)
NUM_MANAGED_GPRS = 16      # R0-R13, R24, R25
NUM_MANAGED_FLOAT_PAIRS = 4  # D16, D18, D20, D22

# JITFRAME layout: 16 GPR slots + 1 alignment padding + 4 float pairs x 2 words = 25
# The padding ensures float pairs start at a DOUBLE_WORD-aligned offset
# (required by LDD/STD instructions).
JITFRAME_FIXED_SIZE = NUM_MANAGED_GPRS + 1 + NUM_MANAGED_FLOAT_PAIRS * 2

# Immediate ranges
SINT16_IMM_MIN = -(1 << 15)
SINT16_IMM_MAX = (1 << 15) - 1
SINT11_IMM_MIN = -(1 << 10)
SINT11_IMM_MAX = (1 << 10) - 1

# Parse bits for VLIW packets (bits [15:14] of each instruction)
# 0b00 = duplex (two 16-bit sub-instructions) -- never use for regular insns
# 0b01 = not end of packet (more instructions follow)
# 0b10 = endloop0 (marks end of hardware loop 0, also not-end-of-packet)
# 0b11 = end of packet (last instruction)
PARSE_NOT_END = 0b01       # not end of packet (more instructions follow)
PARSE_END_PACKET = 0b11    # end of packet (last instruction in packet)
PARSE_ENDLOOP0 = 0b10      # end of hardware loop0
PARSE_ENDLOOP1 = 0b01      # end of hardware loop1
PARSE_BITS_SHIFT = 14
PARSE_BITS_MASK = 0b11 << PARSE_BITS_SHIFT

# Hexagon ABI stack alignment
ABI_STACK_ALIGN = 8

# PC-relative branch ranges
# Unconditional jump: 24-bit signed, 4-byte aligned (range: +/- 32MB)
JUMP_IMM_BITS = 24
JUMP_IMM_MIN = -(1 << (JUMP_IMM_BITS - 1))
JUMP_IMM_MAX = (1 << (JUMP_IMM_BITS - 1)) - 4

# Conditional branch: 17-bit signed, 4-byte aligned
COND_JUMP_IMM_BITS = 17
COND_JUMP_IMM_MIN = -(1 << (COND_JUMP_IMM_BITS - 1))
COND_JUMP_IMM_MAX = (1 << (COND_JUMP_IMM_BITS - 1)) - 4
