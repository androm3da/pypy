"""
Hexagon instruction definitions for the JIT backend.

All encodings verified against LLVM tablegen (HexagonDepInstrInfo.td).
Hexagon instructions are 32-bit words. Parse bits at [15:14] indicate
packet position; for single-instruction packets, always 0b11.

Each instruction entry is:
    (mnemonic, encoding_type, encoding_params)

Encoding types define the bit layout and operand format.
The instruction_builder.py generates assembler methods from these tables.
"""

# ---------------------------------------------------------------------------
# ALU32 three-register: Rd = op(Rs, Rt)
# Encoding: opcode@[31:21], Rs@[20:16], 0@[13], Rt@[12:8], subop@[7:5], Rd@[4:0]
# Params: (opcode_31_21, subop_7_5)
# ---------------------------------------------------------------------------
alu32_rrr_instructions = [
    # Rd = add(Rs, Rt)                        opcode[31:21]     sub[7:5]
    ('ADD',       0b11110011000,  0b000),
    # Rd = and(Rs, Rt)
    ('AND',       0b11110001000,  0b000),
    # Rd = or(Rs, Rt)
    ('OR',        0b11110001001,  0b000),
    # Rd = xor(Rs, Rt)
    ('XOR',       0b11110001011,  0b000),
    # Rd = mpyi(Rs, Rt)
    ('MPYI',      0b11101101000,  0b000),
    # Rd = sfadd(Rs, Rt)
    ('SFADD',     0b11101011000,  0b000),
    # Rd = sfsub(Rs, Rt)
    ('SFSUB',     0b11101011000,  0b001),
    # Rd = sfmpy(Rs, Rt)
    ('SFMPY',     0b11101011010,  0b000),
]

# ---------------------------------------------------------------------------
# ALU32 three-register reversed: Rd = sub(Rt, Rs)
# Same encoding as alu32_rrr but assembler takes (rd, rt, rs) and swaps
# Encoding: opcode@[31:21], Rs@[20:16], 0@[13], Rt@[12:8], subop@[7:5], Rd@[4:0]
# Params: (opcode_31_21, subop_7_5)
# The generated method signature is (self, rd, rt, rs) but encodes rs@20-16, rt@12-8
# ---------------------------------------------------------------------------
alu32_rrr_rev_instructions = [
    # Rd = sub(Rt, Rs) -- note operand order
    ('SUB',       0b11110011001,  0b000),
]

# ---------------------------------------------------------------------------
# ALU32 two-register: Rd = op(Rs)
# Encoding: opcode@[31:21], Rs@[20:16], 0@[13:5], Rd@[4:0]
# Params: (opcode_31_21, fixed_13_5)
# ---------------------------------------------------------------------------
alu32_rr_instructions = [
    # Rd = Rs (register transfer)
    ('TFR',       0b01110000011,  0b000000000),
    # Rd = sxtb(Rs) (sign-extend byte)
    ('SXTB',      0b01110000101,  0b000000000),
    # Rd = sxth(Rs) (sign-extend half)
    ('SXTH',      0b01110000111,  0b000000000),
    # Rd = zxth(Rs) (zero-extend half)
    ('ZXTH',      0b01110000110,  0b000000000),
    # Rd = asrh(Rs) (arithmetic shift right by 16)
    ('ASRH',      0b01110000001,  0b000000000),
]

# ---------------------------------------------------------------------------
# ALU32 add immediate: Rd = add(Rs, #s16)
# Encoding: 0b1011@[31:28], Ii{15:9}@[27:21], Rs@[20:16], Ii{8:0}@[13:5], Rd@[4:0]
# Params: none (single encoding)
# ---------------------------------------------------------------------------
alu32_addi_instructions = [
    ('ADDI',),
]

# ---------------------------------------------------------------------------
# ALU32 transfer signed immediate: Rd = #s16
# Encoding: 0b01110000@[31:24], Ii{15:14}@[23:22], 0@[21],
#           Ii{13:9}@[20:16], Ii{8:0}@[13:5], Rd@[4:0]
# Params: none (single encoding)
# ---------------------------------------------------------------------------
alu32_tfrsi_instructions = [
    ('TFRSI',),
]

# ---------------------------------------------------------------------------
# ALU32 register-immediate (10-bit): Rd = op(Rs, #s10)
# Encoding: opcode@[31:22], Ii{9}@[21], Rs@[20:16], Ii{8:0}@[13:5], Rd@[4:0]
# Params: (opcode_31_22,)
# ---------------------------------------------------------------------------
alu32_ri10_instructions = [
    # Rd = and(Rs, #s10)
    ('ANDI',      0b0111011000),
    # Rd = or(Rs, #s10)
    ('ORI',       0b0111011010),
]

# ---------------------------------------------------------------------------
# Shift by immediate: Rd = op(Rs, #u5)
# Encoding: opcode@[31:21], Rs@[20:16], 0@[13], Ii{4:0}@[12:8], subop@[7:5], Rd@[4:0]
# Params: (opcode_31_21, subop_7_5)
# ---------------------------------------------------------------------------
shift_ri5_instructions = [
    # Rd = asr(Rs, #u5)
    ('S2_ASR_I_R',  0b10001100000,  0b000),
    # Rd = lsr(Rs, #u5)
    ('S2_LSR_I_R',  0b10001100000,  0b001),
    # Rd = asl(Rs, #u5)
    ('S2_ASL_I_R',  0b10001100000,  0b010),
]

# ---------------------------------------------------------------------------
# Shift by register: Rd = op(Rs, Rt)
# Encoding class: Enc_5ab2be (S_3op)
# Encoding: opcode@[31:21], Rs@[20:16], 0@[13], Rt@[12:8], subop@[7:5], Rd@[4:0]
# All three shifts share the same opcode, differ only in subop.
# Params: (opcode_31_21, subop_7_5)
# ---------------------------------------------------------------------------
shift_rr_instructions = [
    # Rd = asr(Rs, Rt)
    ('S2_ASR_R_R',  0b11000110010,  0b000),
    # Rd = lsr(Rs, Rt)
    ('S2_LSR_R_R',  0b11000110010,  0b010),
    # Rd = asl(Rs, Rt)
    ('S2_ASL_R_R',  0b11000110010,  0b100),
]

# ---------------------------------------------------------------------------
# Compare registers to predicate: Pd = cmp.xx(Rs, Rt)
# Encoding: opcode@[31:21], Rs@[20:16], 0@[13], Rt@[12:8], 0@[7:2], Pd@[1:0]
# Params: (opcode_31_21,)
# ---------------------------------------------------------------------------
cmp_rr_instructions = [
    # Pd = cmp.eq(Rs, Rt)
    ('CMP_EQ',    0b11110010000),
    # Pd = cmp.gt(Rs, Rt)
    ('CMP_GT',    0b11110010010),
    # Pd = cmp.gtu(Rs, Rt)
    ('CMP_GTU',   0b11110010011),
]

# ---------------------------------------------------------------------------
# Compare register-immediate to predicate: Pd = cmp.xx(Rs, #s10)
# Encoding: opcode@[31:22], Ii{9}@[21], Rs@[20:16], Ii{8:0}@[13:5],
#           0@[4:2], Pd@[1:0]
# Params: (opcode_31_22,)
# ---------------------------------------------------------------------------
cmp_ri_instructions = [
    # Pd = cmp.eq(Rs, #s10)
    ('CMP_EQI',   0b0111010100),
    # Pd = cmp.gt(Rs, #s10)
    ('CMP_GTI',   0b0111010101),
    # Pd = cmp.gtu(Rs, #u9)
    ('CMP_GTUI',  0b0111010110),
]

# ---------------------------------------------------------------------------
# Conditional select: Rd = mux(Pu, Rs, Rt)
# Encoding: 0b11110100000@[31:21], Rs@[20:16], 0@[13], Rt@[12:8],
#           0@[7], Pu@[6:5], Rd@[4:0]
# Params: none (single encoding)
# ---------------------------------------------------------------------------
mux_instructions = [
    ('MUX',),
]

# ---------------------------------------------------------------------------
# Load with immediate offset: Rd = memX(Rs + #offset)
# Encoding: 0b10010@[31:27], Ii_hi@[26:25], variant@[24:21],
#           Rs@[20:16], Ii_lo@[13:5], Rd@[4:0]
# Params: (variant_24_21, offset_scale)
# offset_scale: how many low bits of the byte offset are implicit zeros
# ---------------------------------------------------------------------------
load_instructions = [
    # Rd = memb(Rs + #s11:0) --byte load (signed)
    ('LDB',       0b1000, 0),
    # Rd = memub(Rs + #s11:0) --byte load (unsigned)
    ('LDUB',      0b1001, 0),
    # Rd = memh(Rs + #s11:1) --halfword load (signed)
    ('LDH',       0b1010, 1),
    # Rd = memuh(Rs + #s11:1) --halfword load (unsigned)
    ('LDUH',      0b1011, 1),
    # Rd = memw(Rs + #s11:2) --word load
    ('LDW',       0b1100, 2),
]

# ---------------------------------------------------------------------------
# Load double with immediate offset: Rdd = memd(Rs + #offset)
# Encoding: same as load_instructions but Rdd (pair) at [4:0]
# Params: (variant_24_21, offset_scale)
# ---------------------------------------------------------------------------
load_pair_instructions = [
    # Rdd = memd(Rs + #s11:3) --double load
    ('LDD',       0b1110, 3),
]

# ---------------------------------------------------------------------------
# Store with immediate offset: memX(Rs + #offset) = Rt
# Encoding: 0b1010@[31:28], Ii_hi@[27:25], 1@[24], variant@[23:21],
#           Rs@[20:16], pp@[15:14], Ii_mid@[13], Rt@[12:8], Ii_lo@[7:0]
# Params: (variant_23_21, offset_scale)
# ---------------------------------------------------------------------------
store_instructions = [
    # memb(Rs + #s11:0) = Rt --byte store
    ('STB',       0b000, 0),
    # memh(Rs + #s11:1) = Rt --halfword store
    ('STH',       0b010, 1),
    # memw(Rs + #s11:2) = Rt --word store
    ('STW',       0b100, 2),
]

# ---------------------------------------------------------------------------
# Store double with immediate offset: memd(Rs + #offset) = Rtt
# Encoding: same as store_instructions but Rtt (pair) at [12:8]
# Params: (variant_23_21, offset_scale)
# ---------------------------------------------------------------------------
store_pair_instructions = [
    # memd(Rs + #s11:3) = Rtt --double store
    ('STD',       0b110, 3),
]

# ---------------------------------------------------------------------------
# Unconditional jump/call (PC-relative)
# Encoding: opcode@[31:25], Ii{23:15}@[24:16], Ii{14:2}@[13:1], 0@[0]
# For single-instruction packets, parse bits at [15:14] are set to 0b11.
# Params: (opcode_31_25,)
# ---------------------------------------------------------------------------
jump_imm_instructions = [
    # jump #r22:2
    ('J_IMM',     0b0101100),
    # call #r22:2
    ('CALL_IMM',  0b0101101),
]

# ---------------------------------------------------------------------------
# Register indirect jump/call
# Encoding: opcode@[31:21], Rs@[20:16], 0@[13:0]
# Params: (opcode_31_21,)
# ---------------------------------------------------------------------------
jump_reg_instructions = [
    # jumpr Rs
    ('JUMPR',     0b01010010100),
    # callr Rs
    ('CALLR',     0b01010000101),
]

# ---------------------------------------------------------------------------
# Conditional jump: if (Pu) jump / if (!Pu) jump
# Encoding: 0x5C@[31:24], Ii{16:15}@[23:22], sense@[21],
#           Ii{14:10}@[20:16], Ii{9}@[13], 0@[12:10],
#           Pu@[9:8], Ii{8:2}@[7:1], 0@[0]
# sense=0: if (Pu) jump:nt   sense=1: if (!Pu) jump:nt
# Params: (sense_bit_21,)
# ---------------------------------------------------------------------------
cond_jump_instructions = [
    # if (Pu) jump:nt #r15:2 -- sense=0
    ('J_IF_TRUE',   0),
    # if (!Pu) jump:nt #r15:2 -- sense=1
    ('J_IF_FALSE',  1),
]

# ---------------------------------------------------------------------------
# XTYPE register-pair operations: Rdd = op(Rss, Rtt)
# Encoding: opcode@[31:21], Rss@[20:16], 0@[13], Rtt@[12:8],
#           subop@[7:5], Rdd@[4:0]
# Params: (opcode_31_21, subop_7_5)
# ---------------------------------------------------------------------------
xtype_ppp_instructions = [
    # Rdd = dfadd(Rss, Rtt)  [V66+]
    ('DFADD',     0b11101000000,  0b011),
    # Rdd = dfsub(Rss, Rtt)  [V66+]
    ('DFSUB',     0b11101000100,  0b011),
    # Rdd = dfmpyll(Rss, Rtt)  [V67+]  (low*low partial product)
    ('DFMPYLL',   0b11101000101,  0b011),
    # Rdd = dfmpyfix(Rss, Rtt) [V67+] (fix denormalized inputs)
    ('DFMPYFIX',  0b11101000010,  0b011),
    # Rdd = combine(Rs, Rt)
    ('COMBINEW',  0b11110101000,  0b000),
    # Rdd = add(Rss, Rtt)  (64-bit add)
    ('ADD64',     0b11010011000,  0b111),
    # Rdd = sub(Rss, Rtt)  (64-bit sub) -- actually sub(Rtt, Rss) in HW
    ('SUB64',     0b11010011001,  0b111),
]

# ---------------------------------------------------------------------------
# XTYPE widening conversion: Rdd = op(Rs)
# Encoding: opcode@[31:21], Rs@[20:16], fixed@[13:5], Rdd@[4:0]
# Params: (opcode_31_21, fixed_13_5)
# ---------------------------------------------------------------------------
xtype_pair_from_single_instructions = [
    # Rdd = convert_sf2df(Rs)
    ('CONV_SF2DF', 0b10000100010,  0b000000000),
    # Rdd = convert_w2df(Rs)
    ('CONV_W2DF',  0b10000100011,  0b000000010),
]

# ---------------------------------------------------------------------------
# XTYPE narrowing conversion: Rd = op(Rss)
# Encoding: opcode@[31:21], Rss@[20:16], fixed@[13:5], Rd@[4:0]
# Params: (opcode_31_21, fixed_13_5)
# ---------------------------------------------------------------------------
xtype_single_from_pair_instructions = [
    # Rd = convert_df2sf(Rss)
    ('CONV_DF2SF', 0b10000000111,  0b000000001),
    # Rd = convert_df2w(Rss)
    ('CONV_DF2W',  0b10000100100,  0b000000001),
]

# ---------------------------------------------------------------------------
# XTYPE 64-bit compare to predicate: Pd = dfcmp.xx(Rss, Rtt)
# Encoding: opcode@[31:21], Rss@[20:16], 0@[13], Rtt@[12:8],
#           fixed@[7:2], Pd@[1:0]
# Params: (opcode_31_21, fixed_7_2)
# ---------------------------------------------------------------------------
xtype_cmp_pp_instructions = [
    # Pd = dfcmp.eq(Rss, Rtt)
    ('DFCMP_EQ',   0b11010010111,  0b000000),
    # Pd = dfcmp.gt(Rss, Rtt)
    ('DFCMP_GT',   0b11010010111,  0b001000),
]

# ---------------------------------------------------------------------------
# XTYPE accumulator register-pair operations: Rxx += op(Rss, Rtt)
# Encoding: opcode@[31:21], Rss@[20:16], 0@[13], Rtt@[12:8],
#           subop@[7:5], Rxx@[4:0]
# Same bit layout as xtype_ppp but Rxx is both read and written.
# Params: (opcode_31_21, subop_7_5)
# ---------------------------------------------------------------------------
xtype_acc_ppp_instructions = [
    # Rxx += dfmpylh(Rss, Rtt)  [V67+]
    ('DFMPYLH_ACC', 0b11101010000, 0b011),
    # Rxx += dfmpyhh(Rss, Rtt)  [V67+]
    ('DFMPYHH_ACC', 0b11101010100, 0b011),
]

# ---------------------------------------------------------------------------
# XTYPE multiply single-to-pair: Rdd = mpy(Rs, Rt) (32x32 to 64-bit signed)
# Encoding class: Enc_be32a5
# Encoding: opcode@[31:21], Rs@[20:16], 0@[13], Rt@[12:8], subop@[7:5], Rdd@[4:0]
# Same bit layout as alu32_rrr; the destination is a register pair.
# Params: (opcode_31_21, subop_7_5)
# ---------------------------------------------------------------------------
xtype_mpy_rr_to_pair_instructions = [
    # Rdd = mpy(Rs, Rt)  (M2_dpmpyss_s0 -- signed 32x32 to 64)
    ('M2_DPMPYSS_S0',  0b11100101000,  0b000),
]

# ---------------------------------------------------------------------------
# Hardware loop setup (register trip count): loop0/1($Ii, Rs)
# Encoding class: Enc_864a5a
# Encoding: opcode@[31:21], Rs@[20:16], pp@[15:14], 0@[13],
#           Ii{8:4}@[12:8], 0@[7:5], Ii{3:2}@[4:3], 0@[2:0]
#
# Ii is a PC-relative offset (in instruction words, i.e. byte_offset >> 2).
# Only bits {8:4} and {3:2} of Ii are stored (7 bits), giving +/-256 byte range.
# Params: (opcode_31_21,)
# ---------------------------------------------------------------------------
loop_reg_instructions = [
    # loop0($Ii, Rs) -- set SA0, LC0
    ('LOOP0R',   0b01100000000),
    # loop1($Ii, Rs) -- set SA1, LC1
    ('LOOP1R',   0b01100000001),
]

# ---------------------------------------------------------------------------
# Hardware loop setup (immediate trip count): loop0/1($Ii, #II)
# Encoding class: Enc_4dc228
# Encoding: opcode@[31:21], II{9:5}@[20:16], pp@[15:14], 0@[13],
#           Ii{8:4}@[12:8], II{4:2}@[7:5], Ii{3:2}@[4:3], 0@[2], II{1:0}@[1:0]
#
# Ii: PC-relative offset (same as loop_reg)
# II: 10-bit unsigned trip count
# Params: (opcode_31_21,)
# ---------------------------------------------------------------------------
loop_imm_instructions = [
    # loop0($Ii, #II) -- set SA0, LC0 from immediate
    ('LOOP0I',   0b01101001000),
    # loop1($Ii, #II) -- set SA1, LC1 from immediate
    ('LOOP1I',   0b01101001001),
]

# ---------------------------------------------------------------------------
# NOP instruction
# Encoding: 0x7F00C000 (with parse bits)
# ---------------------------------------------------------------------------
# NOP is defined as a pseudo in the codebuilder

# ---------------------------------------------------------------------------
# System instructions
# ---------------------------------------------------------------------------
# TRAP0, BARRIER, etc. are defined directly in the codebuilder


def _main():
    """Validation of instruction tables."""
    all_mnemonics = set()
    has_error = False

    def check_table(table, name, expected_fields):
        for entry in table:
            mnemonic = entry[0]
            if mnemonic in all_mnemonics:
                print 'error: duplicate mnemonic: %s in %s' % (mnemonic, name)
            all_mnemonics.add(mnemonic)
            if len(entry) != expected_fields:
                print 'error: wrong number of fields for %s in %s' % (
                    mnemonic, name)

    check_table(alu32_rrr_instructions, 'alu32_rrr', 3)
    check_table(alu32_rrr_rev_instructions, 'alu32_rrr_rev', 3)
    check_table(alu32_rr_instructions, 'alu32_rr', 3)
    check_table(alu32_addi_instructions, 'alu32_addi', 1)
    check_table(alu32_tfrsi_instructions, 'alu32_tfrsi', 1)
    check_table(alu32_ri10_instructions, 'alu32_ri10', 2)
    check_table(shift_ri5_instructions, 'shift_ri5', 3)
    check_table(shift_rr_instructions, 'shift_rr', 3)
    check_table(cmp_rr_instructions, 'cmp_rr', 2)
    check_table(cmp_ri_instructions, 'cmp_ri', 2)
    check_table(mux_instructions, 'mux', 1)
    check_table(load_instructions, 'load', 3)
    check_table(load_pair_instructions, 'load_pair', 3)
    check_table(store_instructions, 'store', 3)
    check_table(store_pair_instructions, 'store_pair', 3)
    check_table(jump_imm_instructions, 'jump_imm', 2)
    check_table(jump_reg_instructions, 'jump_reg', 2)
    check_table(cond_jump_instructions, 'cond_jump', 2)
    check_table(xtype_ppp_instructions, 'xtype_ppp', 3)
    check_table(xtype_pair_from_single_instructions, 'xtype_p_r', 3)
    check_table(xtype_single_from_pair_instructions, 'xtype_r_p', 3)
    check_table(xtype_cmp_pp_instructions, 'xtype_cmp_pp', 3)
    check_table(xtype_acc_ppp_instructions, 'xtype_acc_ppp', 3)
    check_table(xtype_mpy_rr_to_pair_instructions, 'xtype_mpy_rr_pair', 3)
    check_table(loop_reg_instructions, 'loop_reg', 2)
    check_table(loop_imm_instructions, 'loop_imm', 2)

    if not has_error:
        print 'defined', len(all_mnemonics), 'instructions successfully'

if __name__ == '__main__':
    _main()
del _main
