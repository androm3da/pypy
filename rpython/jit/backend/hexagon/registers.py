"""
Hexagon register definitions and partitioning for the JIT backend.

Register file layout:
  R0-R13:  caller-saved, allocatable for integers
  R14, R15: scratch registers (internal JIT use)
  R16-R23: callee-saved, used as float register pairs
           D16=R17:R16, D18=R19:R18, D20=R21:R20, D22=R23:R22
  R24, R25: callee-saved, allocatable for integers
  R26:     reserved for GIL/thread ID
  R27:     reserved for shadow stack
  R28:     reserved (GP in Hexagon ABI)
  R29:     stack pointer (SP)
  R30:     frame pointer -> holds JITFRAME address
  R31:     link register (LR)

  P0-P3:   predicate registers (8-bit each)
  P0:      scratch predicate for internal use

  V0-V31:  HVX vector registers (128 bytes each)
  Q0-Q3:   HVX vector predicate registers
"""

from rpython.jit.backend.hexagon.locations import (
    RegisterLocation,
    RegisterPairLocation,
    PredicateLocation,
    HVXVectorLocation,
    HVXVectorPairLocation,
    HVXPredicateLocation,
)

# --- General-purpose registers ---

registers = [RegisterLocation(i) for i in range(32)]

(r0,  r1,  r2,  r3,  r4,  r5,  r6,  r7,
 r8,  r9,  r10, r11, r12, r13, r14, r15,
 r16, r17, r18, r19, r20, r21, r22, r23,
 r24, r25, r26, r27, r28, r29, r30, r31) = registers

# Named aliases
sp = r29        # Stack pointer
fp = r30        # Frame pointer (holds JITFRAME address)
lr = r31        # Link register
gp = r28        # Global pointer (reserved by ABI)
scratch1 = r14  # Scratch register 1
scratch2 = r15  # Scratch register 2
thread_id = r26  # Reserved for GIL
shadow_old = r27  # Reserved for shadow stack

# --- Register pairs (for 64-bit / double values) ---

# All 16 pairs
all_register_pairs = [RegisterPairLocation(i) for i in range(0, 32, 2)]

(d0,  d2,  d4,  d6,  d8,  d10, d12, d14,
 d16, d18, d20, d22, d24, d26, d28, d30) = all_register_pairs

# --- Predicate registers ---

predicates = [PredicateLocation(i) for i in range(4)]
(p0, p1, p2, p3) = predicates

scratch_pred = p0  # scratch predicate for internal use

# --- Register partitioning ---

# Integer allocatable: R0-R13, R24, R25 (16 regs)
allocatable_registers = [
    r0, r1, r2, r3, r4, r5, r6, r7,
    r8, r9, r10, r11, r12, r13,
    r24, r25,
]

# Float allocatable: D16, D18, D20, D22 (4 register pairs, using R16-R23)
allocatable_float_pairs = [d16, d18, d20, d22]

# Caller-saved registers (R0-R15 per ABI, but R14-R15 are our scratch)
caller_saved_registers = [
    r0, r1, r2, r3, r4, r5, r6, r7,
    r8, r9, r10, r11, r12, r13, r14, r15,
]

# Callee-saved registers (R16-R27 per ABI)
callee_saved_registers = [
    r16, r17, r18, r19, r20, r21, r22, r23,
    r24, r25, r26, r27,
]

# Callee-saved that we need to save/restore in prologue/epilogue
# (all callee-saved that we use, plus FP and LR)
callee_saved_to_spill = [
    fp,  # R30 - caller's frame pointer (we reuse it for JITFRAME)
    lr,  # R31 - must save/restore across calls
    r16, r17, r18, r19, r20, r21, r22, r23,  # float pairs
    r24, r25,  # integer allocatable
    r26, r27,  # GIL / shadow stack
]

# ABI argument registers
argument_regs = [r0, r1, r2, r3, r4, r5]

# Scratch registers for internal JIT use (not allocatable)
scratch_regs = [r14, r15]

# --- HVX Vector Registers ---

hvx_vector_regs = [HVXVectorLocation(i) for i in range(32)]

(v0,  v1,  v2,  v3,  v4,  v5,  v6,  v7,
 v8,  v9,  v10, v11, v12, v13, v14, v15,
 v16, v17, v18, v19, v20, v21, v22, v23,
 v24, v25, v26, v27, v28, v29, v30, v31) = hvx_vector_regs

# HVX vector pairs
hvx_vector_pairs = [HVXVectorPairLocation(i) for i in range(0, 32, 2)]

# HVX vector predicates
hvx_predicates = [HVXPredicateLocation(i) for i in range(4)]
(q0, q1, q2, q3) = hvx_predicates

# Allocatable HVX vector registers (V0-V15 for user code, V16-V31 reserved)
allocatable_hvx_regs = [
    v0, v1, v2, v3, v4, v5, v6, v7,
    v8, v9, v10, v11, v12, v13, v14, v15,
]
