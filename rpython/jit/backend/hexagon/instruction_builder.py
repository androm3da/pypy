"""
Hexagon instruction encoding generators.

Each generator function creates an assembler method that encodes a specific
instruction type. The gen_all_instr_assemblers(cls) function adds these
methods to the builder class.

All instructions are emitted as single-instruction packets (parse bits = 0b11
at bits [15:14]).
"""

from rpython.jit.backend.hexagon.arch import PARSE_END_PACKET, PARSE_BITS_SHIFT
from rpython.jit.backend.hexagon import instructions as insns

PARSE_BITS = PARSE_END_PACKET << PARSE_BITS_SHIFT  # 0xC000


# ---------------------------------------------------------------------------
# ALU32 three-register: Rd = op(Rs, Rt)
# Bits: opcode@[31:21], Rs@[20:16], pp@[15:14], 0@[13], Rt@[12:8],
#       subop@[7:5], Rd@[4:0]
# ---------------------------------------------------------------------------
def _gen_alu32_rrr(opcode_31_21, subop_7_5):
    bits = (opcode_31_21 << 21) | (subop_7_5 << 5)
    def assemble(self, rd, rs, rt):
        from rpython.rlib.debug import debug_print
        val = bits | PARSE_BITS | (int(rs) << 16) | (int(rt) << 8) | int(rd)
        debug_print("HEXDBG alu32_rrr bits=", bits, " rs=", int(rs),
                    " rt=", int(rt), " rd=", int(rd), " val=", val)
        self.write32(val)
    return assemble


# ---------------------------------------------------------------------------
# ALU32 three-register reversed: Rd = sub(Rt, Rs)
# Same bit layout as alu32_rrr, but the method takes (rd, rt, rs)
# and puts rs in bits 20-16, rt in bits 12-8.
# ---------------------------------------------------------------------------
def _gen_alu32_rrr_rev(opcode_31_21, subop_7_5):
    bits = (opcode_31_21 << 21) | (subop_7_5 << 5)
    def assemble(self, rd, rt, rs):
        self.write32(bits | PARSE_BITS |
                     (int(rs) << 16) | (int(rt) << 8) | int(rd))
    return assemble


# ---------------------------------------------------------------------------
# ALU32 two-register: Rd = op(Rs)
# Bits: opcode@[31:21], Rs@[20:16], pp@[15:14], fixed@[13:5], Rd@[4:0]
# ---------------------------------------------------------------------------
def _gen_alu32_rr(opcode_31_21, fixed_13_5):
    bits = (opcode_31_21 << 21) | (fixed_13_5 << 5)
    def assemble(self, rd, rs):
        self.write32(bits | PARSE_BITS | (int(rs) << 16) | int(rd))
    return assemble


# ---------------------------------------------------------------------------
# ALU32 add immediate: Rd = add(Rs, #s16)
# Bits: 1011@[31:28], Ii{15:9}@[27:21], Rs@[20:16], pp@[15:14],
#       Ii{8:0}@[13:5], Rd@[4:0]
# ---------------------------------------------------------------------------
def _gen_alu32_addi():
    base = 0b1011 << 28
    def assemble(self, rd, rs, imm):
        imm16 = imm & 0xFFFF
        imm_hi = (imm16 >> 9) & 0x7F   # bits 15:9
        imm_lo = imm16 & 0x1FF          # bits 8:0
        self.write32(base | PARSE_BITS |
                     (imm_hi << 21) | (int(rs) << 16) |
                     (imm_lo << 5) | int(rd))
    return assemble


# ---------------------------------------------------------------------------
# ALU32 transfer signed immediate: Rd = #s16
# Bits: 01110000@[31:24], Ii{15:14}@[23:22], 0@[21],
#       Ii{13:9}@[20:16], pp@[15:14], Ii{8:0}@[13:5], Rd@[4:0]
# ---------------------------------------------------------------------------
def _gen_alu32_tfrsi():
    base = 0b01111000 << 24
    def assemble(self, rd, imm):
        imm16 = imm & 0xFFFF
        bits_15_14 = (imm16 >> 14) & 0x3
        bits_13_9 = (imm16 >> 9) & 0x1F
        bits_8_0 = imm16 & 0x1FF
        self.write32(base | PARSE_BITS |
                     (bits_15_14 << 22) |
                     (bits_13_9 << 16) |
                     (bits_8_0 << 5) | int(rd))
    return assemble


# ---------------------------------------------------------------------------
# ALU32 register-immediate (10-bit): Rd = op(Rs, #s10)
# Bits: opcode@[31:22], Ii{9}@[21], Rs@[20:16], pp@[15:14],
#       Ii{8:0}@[13:5], Rd@[4:0]
# ---------------------------------------------------------------------------
def _gen_alu32_ri10(opcode_31_22):
    base = opcode_31_22 << 22
    def assemble(self, rd, rs, imm):
        imm10 = imm & 0x3FF
        imm_hi = (imm10 >> 9) & 0x1    # bit 9
        imm_lo = imm10 & 0x1FF          # bits 8:0
        self.write32(base | PARSE_BITS |
                     (imm_hi << 21) | (int(rs) << 16) |
                     (imm_lo << 5) | int(rd))
    return assemble


# ---------------------------------------------------------------------------
# Shift by register: Rd = op(Rs, Rt)
# Same bit layout as alu32_rrr (Enc_5ab2be / S_3op class).
# Bits: opcode@[31:21], Rs@[20:16], pp@[15:14], 0@[13],
#       Rt@[12:8], subop@[7:5], Rd@[4:0]
# ---------------------------------------------------------------------------
def _gen_shift_rr(opcode_31_21, subop_7_5):
    bits = (opcode_31_21 << 21) | (subop_7_5 << 5)
    def assemble(self, rd, rs, rt):
        self.write32(bits | PARSE_BITS |
                     (int(rs) << 16) | (int(rt) << 8) | int(rd))
    return assemble


# ---------------------------------------------------------------------------
# Shift by immediate: Rd = op(Rs, #u5)
# Bits: opcode@[31:21], Rs@[20:16], pp@[15:14], 0@[13],
#       Ii{4:0}@[12:8], subop@[7:5], Rd@[4:0]
# ---------------------------------------------------------------------------
def _gen_shift_ri5(opcode_31_21, subop_7_5):
    bits = (opcode_31_21 << 21) | (subop_7_5 << 5)
    def assemble(self, rd, rs, shamt):
        self.write32(bits | PARSE_BITS |
                     (int(rs) << 16) | ((shamt & 0x1F) << 8) | int(rd))
    return assemble


# ---------------------------------------------------------------------------
# Compare registers to predicate: Pd = cmp.xx(Rs, Rt)
# Bits: opcode@[31:21], Rs@[20:16], pp@[15:14], 0@[13],
#       Rt@[12:8], 0@[7:2], Pd@[1:0]
# ---------------------------------------------------------------------------
def _gen_cmp_rr(opcode_31_21):
    bits = opcode_31_21 << 21
    def assemble(self, pd, rs, rt):
        self.write32(bits | PARSE_BITS |
                     (int(rs) << 16) | (int(rt) << 8) | int(pd))
    return assemble


# ---------------------------------------------------------------------------
# Compare register-immediate to predicate: Pd = cmp.xx(Rs, #s10)
# Bits: opcode@[31:22], Ii{9}@[21], Rs@[20:16], pp@[15:14],
#       Ii{8:0}@[13:5], 0@[4:2], Pd@[1:0]
# ---------------------------------------------------------------------------
def _gen_cmp_ri(opcode_31_22):
    base = opcode_31_22 << 22
    def assemble(self, pd, rs, imm):
        imm10 = imm & 0x3FF
        imm_hi = (imm10 >> 9) & 0x1
        imm_lo = imm10 & 0x1FF
        self.write32(base | PARSE_BITS |
                     (imm_hi << 21) | (int(rs) << 16) |
                     (imm_lo << 5) | int(pd))
    return assemble


# ---------------------------------------------------------------------------
# Conditional select: Rd = mux(Pu, Rs, Rt)
# Bits: 11110100000@[31:21], Rs@[20:16], pp@[15:14], 0@[13],
#       Rt@[12:8], 0@[7], Pu@[6:5], Rd@[4:0]
# ---------------------------------------------------------------------------
def _gen_mux():
    bits = 0b11110100000 << 21
    def assemble(self, rd, pu, rs, rt):
        self.write32(bits | PARSE_BITS |
                     (int(rs) << 16) | (int(rt) << 8) |
                     (int(pu) << 5) | int(rd))
    return assemble


# ---------------------------------------------------------------------------
# Load with immediate offset: Rd = memX(Rs + #offset)
# Bits: 10010@[31:27], Ii_hi@[26:25], variant@[24:21], Rs@[20:16],
#       pp@[15:14], Ii_lo@[13:5], Rd@[4:0]
#
# The byte offset is passed directly. Lower bits according to offset_scale
# are implicit zeros and not encoded.
# For memb: offset_scale=0, Ii is 11 bits, Ii{10:9}@[26:25], Ii{8:0}@[13:5]
# For memh: offset_scale=1, Ii is 12 bits, Ii{11:10}@[26:25], Ii{9:1}@[13:5]
# For memw: offset_scale=2, Ii is 13 bits, Ii{12:11}@[26:25], Ii{10:2}@[13:5]
# For memd: offset_scale=3, Ii is 14 bits, Ii{13:12}@[26:25], Ii{11:3}@[13:5]
# ---------------------------------------------------------------------------
def _gen_load(variant_24_21, offset_scale):
    base = (0b10010 << 27) | (variant_24_21 << 21)
    def assemble(self, rd, rs, offset):
        # offset is the raw byte offset
        imm = offset & 0xFFFFFFFF  # handle negative offsets
        # Extract the bits above the alignment bits
        hi_shift = 9 + offset_scale
        imm_hi = (imm >> hi_shift) & 0x3
        imm_lo = (imm >> offset_scale) & 0x1FF
        self.write32(base | PARSE_BITS |
                     (imm_hi << 25) | (int(rs) << 16) |
                     (imm_lo << 5) | int(rd))
    return assemble


# ---------------------------------------------------------------------------
# Load pair with immediate offset: Rdd = memd(Rs + #offset)
# Same format as load, but Rdd (pair) in destination field
# ---------------------------------------------------------------------------
def _gen_load_pair(variant_24_21, offset_scale):
    # Same encoding as regular load, just the register is a pair
    return _gen_load(variant_24_21, offset_scale)


# ---------------------------------------------------------------------------
# Store with immediate offset: memX(Rs + #offset) = Rt
# Bits: 1010@[31:28], Ii_hi@[27:25], 1@[24], variant@[23:21], Rs@[20:16],
#       pp@[15:14], Ii_mid@[13], Rt@[12:8], Ii_lo@[7:0]
#
# The scaled immediate is split:
# scaled = offset >> offset_scale
# Ii_hi (3 bits) at [27:25], Ii_mid (1 bit) at [13], Ii_lo (8 bits) at [7:0]
# ---------------------------------------------------------------------------
def _gen_store(variant_23_21, offset_scale):
    base = (0b1010 << 28) | (1 << 24) | (variant_23_21 << 21)
    def assemble(self, rs, rt, offset):
        imm = offset & 0xFFFFFFFF
        scaled = (imm >> offset_scale) & 0xFFF  # 12 bits max
        imm_hi = (scaled >> 9) & 0x7     # 3 bits at [27:25]
        imm_mid = (scaled >> 8) & 0x1    # 1 bit at [13]
        imm_lo = scaled & 0xFF           # 8 bits at [7:0]
        self.write32(base | PARSE_BITS |
                     (imm_hi << 25) | (int(rs) << 16) |
                     (imm_mid << 13) | (int(rt) << 8) | imm_lo)
    return assemble


# ---------------------------------------------------------------------------
# Store pair with immediate offset: memd(Rs + #offset) = Rtt
# Same format as store, but Rtt (pair) in data field
# ---------------------------------------------------------------------------
def _gen_store_pair(variant_24_21, offset_scale):
    return _gen_store(variant_24_21, offset_scale)


# ---------------------------------------------------------------------------
# Unconditional jump/call (PC-relative)
# Bits: opcode@[31:25], Ii{23:15}@[24:16], pp@[15:14],
#       Ii{14:2}@[13:1], 0@[0]
# The immediate is a signed byte offset (must be 4-byte aligned).
# We encode Ii{23:2} (22 bits of the 24-bit offset, dropping bottom 2).
# ---------------------------------------------------------------------------
def _gen_jump_imm(opcode_31_25):
    base = opcode_31_25 << 25
    def assemble(self, offset):
        imm = offset & 0xFFFFFF  # 24-bit
        imm_hi = (imm >> 15) & 0x1FF    # bits 23:15 -> 9 bits
        imm_lo = (imm >> 2) & 0x1FFF    # bits 14:2 -> 13 bits
        self.write32(base | PARSE_BITS |
                     (imm_hi << 16) | (imm_lo << 1))
    return assemble


# ---------------------------------------------------------------------------
# Register indirect jump/call
# Bits: opcode@[31:21], Rs@[20:16], pp@[15:14], 0@[13:0]
# ---------------------------------------------------------------------------
def _gen_jump_reg(opcode_31_21):
    bits = opcode_31_21 << 21
    def assemble(self, rs):
        self.write32(bits | PARSE_BITS | (int(rs) << 16))
    return assemble


# ---------------------------------------------------------------------------
# Conditional jump: if (Pu) jump / if (!Pu) jump
# Bits: opcode@[31:24], Ii{16:15}@[23:22], 0@[21],
#       Ii{14:10}@[20:16], pp@[15:14], Ii{9}@[13], 0@[12:10],
#       Pu@[9:8], Ii{8:2}@[7:1], 0@[0]
# The immediate is a signed 17-bit offset (must be 4-byte aligned).
# We encode Ii{16:2} (15 bits of the 17-bit offset, dropping bottom 2).
# ---------------------------------------------------------------------------
def _gen_cond_jump(sense_bit):
    # Conditional jump: if (Pu) jump:nt / if (!Pu) jump:nt
    # Opcode is always 0x5C at [31:24], sense bit at [21]
    base = 0x5C << 24
    sense = sense_bit << 21
    def assemble(self, pu, offset):
        imm = offset & 0x1FFFF  # 17-bit signed, 4-byte aligned
        bits_16_15 = (imm >> 15) & 0x3
        bits_14_10 = (imm >> 10) & 0x1F
        bit_9 = (imm >> 9) & 0x1
        bits_8_2 = (imm >> 2) & 0x7F
        self.write32(base | sense | PARSE_BITS |
                     (bits_16_15 << 22) |
                     (bits_14_10 << 16) |
                     (bit_9 << 13) |
                     (int(pu) << 8) |
                     (bits_8_2 << 1))
    return assemble


# ---------------------------------------------------------------------------
# XTYPE register-pair operations: Rdd = op(Rss, Rtt)
# Same bit layout as alu32_rrr but uses pair register numbers.
# Bits: opcode@[31:21], Rss@[20:16], pp@[15:14], 0@[13],
#       Rtt@[12:8], subop@[7:5], Rdd@[4:0]
# ---------------------------------------------------------------------------
def _gen_xtype_ppp(opcode_31_21, subop_7_5):
    bits = (opcode_31_21 << 21) | (subop_7_5 << 5)
    def assemble(self, rdd, rss, rtt):
        self.write32(bits | PARSE_BITS |
                     (int(rss) << 16) | (int(rtt) << 8) | int(rdd))
    return assemble


# ---------------------------------------------------------------------------
# XTYPE widening: Rdd = op(Rs)
# Bits: opcode@[31:21], Rs@[20:16], pp@[15:14], fixed@[13:5], Rdd@[4:0]
# ---------------------------------------------------------------------------
def _gen_xtype_pair_from_single(opcode_31_21, fixed_13_5):
    bits = (opcode_31_21 << 21) | (fixed_13_5 << 5)
    def assemble(self, rdd, rs):
        self.write32(bits | PARSE_BITS | (int(rs) << 16) | int(rdd))
    return assemble


# ---------------------------------------------------------------------------
# XTYPE narrowing: Rd = op(Rss)
# Bits: opcode@[31:21], Rss@[20:16], pp@[15:14], fixed@[13:5], Rd@[4:0]
# ---------------------------------------------------------------------------
def _gen_xtype_single_from_pair(opcode_31_21, fixed_13_5):
    bits = (opcode_31_21 << 21) | (fixed_13_5 << 5)
    def assemble(self, rd, rss):
        self.write32(bits | PARSE_BITS | (int(rss) << 16) | int(rd))
    return assemble


# ---------------------------------------------------------------------------
# XTYPE 64-bit compare: Pd = dfcmp.xx(Rss, Rtt)
# Bits: opcode@[31:21], Rss@[20:16], pp@[15:14], 0@[13],
#       Rtt@[12:8], fixed@[7:2], Pd@[1:0]
# ---------------------------------------------------------------------------
def _gen_xtype_cmp_pp(opcode_31_21, fixed_7_2):
    bits = (opcode_31_21 << 21) | (fixed_7_2 << 2)
    def assemble(self, pd, rss, rtt):
        self.write32(bits | PARSE_BITS |
                     (int(rss) << 16) | (int(rtt) << 8) | int(pd))
    return assemble


# ---------------------------------------------------------------------------
# XTYPE accumulator register-pair: Rxx += op(Rss, Rtt)
# Same bit layout as xtype_ppp. Rxx is both input and output.
# Bits: opcode@[31:21], Rss@[20:16], pp@[15:14], 0@[13],
#       Rtt@[12:8], subop@[7:5], Rxx@[4:0]
# ---------------------------------------------------------------------------
def _gen_xtype_acc_ppp(opcode_31_21, subop_7_5):
    bits = (opcode_31_21 << 21) | (subop_7_5 << 5)
    def assemble(self, rxx, rss, rtt):
        self.write32(bits | PARSE_BITS |
                     (int(rss) << 16) | (int(rtt) << 8) | int(rxx))
    return assemble


# ---------------------------------------------------------------------------
# XTYPE multiply single-to-pair: Rdd = mpy(Rs, Rt) (32x32 to 64)
# Same bit layout as alu32_rrr / Enc_be32a5.
# Bits: opcode@[31:21], Rs@[20:16], pp@[15:14], 0@[13],
#       Rt@[12:8], subop@[7:5], Rdd@[4:0]
# ---------------------------------------------------------------------------
def _gen_xtype_mpy_rr_to_pair(opcode_31_21, subop_7_5):
    bits = (opcode_31_21 << 21) | (subop_7_5 << 5)
    def assemble(self, rdd, rs, rt):
        self.write32(bits | PARSE_BITS |
                     (int(rs) << 16) | (int(rt) << 8) | int(rdd))
    return assemble


# ---------------------------------------------------------------------------
# Master function to add all assembler methods to a class
# ---------------------------------------------------------------------------
def gen_all_instr_assemblers(cls):
    # ALU32 three-register
    for mnemonic, opcode, subop in insns.alu32_rrr_instructions:
        setattr(cls, mnemonic, _gen_alu32_rrr(opcode, subop))

    # ALU32 three-register reversed (SUB)
    for mnemonic, opcode, subop in insns.alu32_rrr_rev_instructions:
        setattr(cls, mnemonic, _gen_alu32_rrr_rev(opcode, subop))

    # ALU32 two-register
    for mnemonic, opcode, fixed in insns.alu32_rr_instructions:
        setattr(cls, mnemonic, _gen_alu32_rr(opcode, fixed))

    # ALU32 add immediate
    for entry in insns.alu32_addi_instructions:
        setattr(cls, entry[0], _gen_alu32_addi())

    # ALU32 transfer signed immediate
    for entry in insns.alu32_tfrsi_instructions:
        setattr(cls, entry[0], _gen_alu32_tfrsi())

    # ALU32 register-immediate (10-bit)
    for mnemonic, opcode in insns.alu32_ri10_instructions:
        setattr(cls, mnemonic, _gen_alu32_ri10(opcode))

    # Shift by immediate
    for mnemonic, opcode, subop in insns.shift_ri5_instructions:
        setattr(cls, mnemonic, _gen_shift_ri5(opcode, subop))

    # Shift by register
    for mnemonic, opcode, subop in insns.shift_rr_instructions:
        setattr(cls, mnemonic, _gen_shift_rr(opcode, subop))

    # Compare registers to predicate
    for mnemonic, opcode in insns.cmp_rr_instructions:
        setattr(cls, mnemonic, _gen_cmp_rr(opcode))

    # Compare register-immediate to predicate
    for mnemonic, opcode in insns.cmp_ri_instructions:
        setattr(cls, mnemonic, _gen_cmp_ri(opcode))

    # Conditional select (mux)
    for entry in insns.mux_instructions:
        setattr(cls, entry[0], _gen_mux())

    # Loads
    for mnemonic, variant, scale in insns.load_instructions:
        setattr(cls, mnemonic, _gen_load(variant, scale))

    # Load pair
    for mnemonic, variant, scale in insns.load_pair_instructions:
        setattr(cls, mnemonic, _gen_load_pair(variant, scale))

    # Stores
    for mnemonic, variant, scale in insns.store_instructions:
        setattr(cls, mnemonic, _gen_store(variant, scale))

    # Store pair
    for mnemonic, variant, scale in insns.store_pair_instructions:
        setattr(cls, mnemonic, _gen_store_pair(variant, scale))

    # Unconditional jump/call
    for mnemonic, opcode in insns.jump_imm_instructions:
        setattr(cls, mnemonic, _gen_jump_imm(opcode))

    # Register indirect jump/call
    for mnemonic, opcode in insns.jump_reg_instructions:
        setattr(cls, mnemonic, _gen_jump_reg(opcode))

    # Conditional jump
    for mnemonic, opcode in insns.cond_jump_instructions:
        setattr(cls, mnemonic, _gen_cond_jump(opcode))

    # XTYPE register-pair operations
    for mnemonic, opcode, subop in insns.xtype_ppp_instructions:
        setattr(cls, mnemonic, _gen_xtype_ppp(opcode, subop))

    # XTYPE widening conversions
    for mnemonic, opcode, fixed in insns.xtype_pair_from_single_instructions:
        setattr(cls, mnemonic, _gen_xtype_pair_from_single(opcode, fixed))

    # XTYPE narrowing conversions
    for mnemonic, opcode, fixed in insns.xtype_single_from_pair_instructions:
        setattr(cls, mnemonic, _gen_xtype_single_from_pair(opcode, fixed))

    # XTYPE 64-bit compare
    for mnemonic, opcode, fixed in insns.xtype_cmp_pp_instructions:
        setattr(cls, mnemonic, _gen_xtype_cmp_pp(opcode, fixed))

    # XTYPE accumulator register-pair operations
    for mnemonic, opcode, subop in insns.xtype_acc_ppp_instructions:
        setattr(cls, mnemonic, _gen_xtype_acc_ppp(opcode, subop))

    # XTYPE multiply single->pair
    for mnemonic, opcode, subop in insns.xtype_mpy_rr_to_pair_instructions:
        setattr(cls, mnemonic, _gen_xtype_mpy_rr_to_pair(opcode, subop))
