#!/usr/bin/env python3
"""Differential test: our interpreter against Unicorn (QEMU's ARM64 core).

Generates random instructions inside each implemented encoding class, runs
each one from the same random state in both emulators, and compares
registers, flags, PC and memory.

Unicorn is a test-time oracle only; it is not part of the emulator.

    pip install unicorn
    cargo build --release --example difftest_runner
    python3 tools/difftest.py [--cases N] [--seed S]
"""
import argparse
import collections
import random
import struct
import subprocess
import sys
from pathlib import Path

from unicorn import (UC_ARCH_ARM64, UC_HOOK_INTR, UC_MODE_ARM, UC_PROT_ALL,
                     Uc, UcError)
from unicorn import arm64_const
from unicorn import unicorn_const as uc_const
from unicorn.arm64_const import (UC_ARM64_REG_FPCR, UC_ARM64_REG_FPSR, UC_ARM64_REG_NZCV,
                                 UC_ARM64_REG_PC, UC_ARM64_REG_Q0,
                                 UC_ARM64_REG_SP, UC_ARM64_REG_X0,
                                 UC_ARM64_REG_X29, UC_ARM64_REG_X30)

CODE_BASE, CODE_SIZE = 0x1000_0000, 0x1000
DATA_BASE, DATA_SIZE = 0x2000_0000, 0x10000
RUNNER = Path(__file__).resolve().parent.parent / "target/release/examples/difftest_runner"

# (name, mask, value): random bits are kept only where mask is 0.
CLASSES = [
    ("adr", 0x1F000000, 0x10000000),
    ("addsub_imm", 0x1F800000, 0x11000000),
    ("logical_imm", 0x1F800000, 0x12000000),
    ("move_wide", 0x1F800000, 0x12800000),
    ("bitfield", 0x1F800000, 0x13000000),
    ("extract", 0x1F800000, 0x13800000),
    ("logical_reg", 0x1F000000, 0x0A000000),
    ("addsub_reg", 0x1F000000, 0x0B000000),
    ("dp_misc", 0x1F000000, 0x1A000000),
    ("dp3", 0x1F000000, 0x1B000000),
    ("b", 0x7C000000, 0x14000000),
    ("bcond", 0xFF000010, 0x54000000),
    ("cbz", 0x7E000000, 0x34000000),
    ("tbz", 0x7E000000, 0x36000000),
    ("branch_reg", 0xFF9FFC1F, 0xD61F0000),
    ("ldst_uimm", 0x3F000000, 0x39000000),
    ("ldst_imm9", 0x3F200000, 0x38000000),
    ("ldst_regoff", 0x3F200C00, 0x38200800),
    ("ld_literal", 0x3F000000, 0x18000000),
    ("ldst_pair", 0x3E000000, 0x28000000),
    ("exclusive", 0x3F000000, 0x08000000),
    ("atomic", 0x3F200C00, 0x38200000),
    # SIMD&FP loads and stores (V = 1).
    ("fp_ldst_uimm", 0x3F000000, 0x3D000000),
    ("fp_ldst_imm9", 0x3F200000, 0x3C000000),
    ("fp_ldst_regoff", 0x3F200C00, 0x3C200800),
    ("fp_ld_literal", 0x3F000000, 0x1C000000),
    ("fp_ldst_pair", 0x3E000000, 0x2C000000),
    ("simd_ldst_multi", 0xBFBF0000, 0x0C000000),
    ("simd_ldst_multi_post", 0xBFA00000, 0x0C800000),
    ("simd_ldst_single", 0xBF9F0000, 0x0D000000),
    ("simd_ldst_single_post", 0xBF800000, 0x0D800000),
    # Scalar floating point.
    ("fp_3src", 0xFF000000, 0x1F000000),
    ("fp_1src", 0xFF207C00, 0x1E204000),
    ("fp_2src", 0xFF200C00, 0x1E200800),
    ("fp_compare", 0xFF20FC07, 0x1E202000),
    ("fp_ccmp", 0xFF200C00, 0x1E200400),
    ("fp_csel", 0xFF200C00, 0x1E200C00),
    ("fp_imm", 0xFF201FE0, 0x1E201000),
    ("fp_int_conv", 0x7F20FC00, 0x1E200000),
    ("fp_fixed_conv", 0x7F200000, 0x1E000000),
    # Advanced SIMD.
    ("simd_modimm", 0x9FF80400, 0x0F000400),
    ("simd_copy", 0x9FE08400, 0x0E000400),
    ("simd_scalar_copy", 0xFFE0FC00, 0x5E000400),
    ("simd_scalar_2misc", 0xDF3E0C00, 0x5E200800),
    ("simd_logical", 0x9F20FC00, 0x0E201C00),
    ("simd_addsub", 0x9F20FC00, 0x0E208400),
    ("simd_3same", 0x9F200400, 0x0E200400),
    ("simd_3same_scalar", 0xDF200400, 0x5E200400),
    ("simd_3same_extra", 0x9F208400, 0x0E008400),
    ("simd_3diff", 0x9F200C00, 0x0E200000),
    ("simd_3diff_scalar", 0xDF200C00, 0x5E200000),
    ("simd_2misc", 0x9F3E0C00, 0x0E200800),
    ("simd_across", 0x9F3E0C00, 0x0E300800),
    ("simd_scalar_pairwise", 0xDF3E0C00, 0x5E300800),
    ("simd_permute", 0xBF208C00, 0x0E000800),
    ("simd_ext", 0xBFE08400, 0x2E000000),
    ("simd_tbl", 0xBFE08C00, 0x0E000000),
    ("simd_shift_imm", 0x9F800400, 0x0F000400),
    ("simd_shift_imm_scalar", 0xDF800400, 0x5F000400),
    ("simd_elem", 0x9F000400, 0x0F000000),
        ("simd_elem_scalar", 0xDF000400, 0x5F000000),
    ("fp16_3same", 0x9F60C400, 0x0E400400),
    ("fp16_3same_scalar", 0xDF60C400, 0x5E400400),
    ("fp16_2misc", 0x9F7E0C00, 0x0E780800),
    ("fp16_2misc_scalar", 0xDF7E0C00, 0x5E780800),
]

# Interesting floating-point bit patterns (half, single, double).
FP16 = [0x0000, 0x8000, 0x3C00, 0xBC00, 0x7C00, 0xFC00, 0x7E00, 0xFE00, 0x7E12,
        0x7C01, 0xFD00, 0x0001, 0x8001, 0x03FF, 0x0400, 0x7BFF, 0x6400, 0x3800,
        0x4248, 0x5BFF, 0x3E00, 0x3555]
FP32 = [0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0x7F800000, 0xFF800000,
        0x7FC00000, 0xFFC00000, 0x7FC01234, 0x7F800001, 0xFFA00042, 0x00000001,
        0x80000001, 0x007FFFFF, 0x00800000, 0x7F7FFFFF, 0x4B000000, 0x3F000000,
        0x40490FDB, 0x4F000000, 0xCF000000, 0x5F000000, 0x4EFFFFFF, 0x3FC00000]
FP64 = [0x0, 0x8000000000000000, 0x3FF0000000000000, 0xBFF0000000000000,
        0x7FF0000000000000, 0xFFF0000000000000, 0x7FF8000000000000, 0xFFF8000000000000,
        0x7FF8000000001234, 0x7FF0000000000001, 0xFFF4000000000042, 0x1,
        0x000FFFFFFFFFFFFF, 0x0010000000000000, 0x7FEFFFFFFFFFFFFF, 0x4330000000000000,
        0x3FE0000000000000, 0x400921FB54442D18, 0x43E0000000000000, 0xC3E0000000000000,
        0x41DFFFFFFFC00000, 0x3FF8000000000000, 0x3FE8000000000000]



def pattern(addr):
    """Initial memory contents. Must match `pattern()` in the runner."""
    return (((addr * 0x9E3779B1) & 0xFFFFFFFFFFFFFFFF) >> 7 ^ (addr >> 3)) & 0xFF


INITIAL = {
    base: bytes(pattern(a) for a in range(base, base + size))
    for base, size in [(CODE_BASE, CODE_SIZE), (DATA_BASE, DATA_SIZE)]
}


def random_reg_value(rng):
    roll = rng.random()
    if roll < 0.45:
        # A pointer into the middle of the data region.
        return DATA_BASE + 0x4000 + rng.randrange(0x8000) & ~(rng.choice([0, 0, 7]))
    if roll < 0.7:
        return rng.randrange(64)
    if roll < 0.8:
        return rng.choice([0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF,
                           0x7FFFFFFFFFFFFFFF, 0x8000000000000000, 0xFFFFFFFFFFFFFFFF])
    return rng.getrandbits(64)


def random_fp_scalar(rng):
    roll = rng.random()
    if roll < 0.15:
        # Lanes of halves: four values, so every lane gets something useful.
        return sum(rng.choice(FP16) << (16 * i) for i in range(4))
    if roll < 0.4:
        return rng.choice(FP32)
    if roll < 0.8:
        return rng.choice(FP64)
    if roll < 0.9:
        # A random "ordinary" double, e.g. 123.456.
        return struct.unpack("<Q", struct.pack("<d", rng.uniform(-1e6, 1e6)))[0]
    return rng.getrandbits(64)


def random_v_value(rng):
    if rng.random() < 0.25:
        return rng.getrandbits(128)
    return random_fp_scalar(rng) | (rng.choice([0, rng.getrandbits(64)]) << 64)


def make_case(rng):
    name, mask, value = rng.choice(CLASSES)
    insn = (rng.getrandbits(32) & ~mask) | value
    pc = CODE_BASE + 0x800 + 4 * rng.randrange(-0x100, 0x100)
    nzcv = rng.getrandbits(4) << 28
    # FPCR: round-to-nearest; DN (default NaN) on in a quarter of cases.
    fpcr = (1 << 25) if rng.random() < 0.25 else 0
    regs = [random_reg_value(rng) for _ in range(31)]
    sp = DATA_BASE + 0x8000 + 16 * rng.randrange(-0x100, 0x100)
    vregs = [random_v_value(rng) for _ in range(32)]
    return name, insn, pc, nzcv, fpcr, regs, sp, vregs


def run_unicorn(insn, pc, nzcv, fpcr, regs, sp, vregs):
    uc = Uc(UC_ARCH_ARM64, UC_MODE_ARM)
    # Newest CPU model, closest to the Switch 2's ARMv8.2 Cortex-A78C.
    uc.ctl_set_cpu_model(arm64_const.UC_CPU_ARM64_MAX)
    uc.mem_map(CODE_BASE, CODE_SIZE, UC_PROT_ALL)
    uc.mem_map(DATA_BASE, DATA_SIZE, uc_const.UC_PROT_READ | uc_const.UC_PROT_WRITE)
    uc.mem_write(CODE_BASE, INITIAL[CODE_BASE])
    uc.mem_write(DATA_BASE, INITIAL[DATA_BASE])
    uc.mem_write(pc, insn.to_bytes(4, "little"))
    for i in range(29):
        uc.reg_write(UC_ARM64_REG_X0 + i, regs[i])
    uc.reg_write(UC_ARM64_REG_X29, regs[29])
    uc.reg_write(UC_ARM64_REG_X30, regs[30])
    uc.reg_write(UC_ARM64_REG_SP, sp)
    uc.reg_write(UC_ARM64_REG_NZCV, nzcv)
    uc.reg_write(UC_ARM64_REG_FPCR, fpcr)
    uc.reg_write(UC_ARM64_REG_FPSR, 0)
    for i in range(32):
        uc.reg_write(UC_ARM64_REG_Q0 + i, vregs[i])
    uc.reg_write(UC_ARM64_REG_PC, pc)

    interrupts = []

    def on_intr(uc_, intno, _):
        interrupts.append(intno)
        uc_.emu_stop()

    uc.hook_add(UC_HOOK_INTR, on_intr)
    status = "ok"
    try:
        uc.emu_start(pc, CODE_BASE + CODE_SIZE + 0x10000, count=1)
    except UcError as e:
        if e.errno == uc_const.UC_ERR_INSN_INVALID:
            status = "unimpl"
        elif e.errno in (uc_const.UC_ERR_READ_UNMAPPED, uc_const.UC_ERR_WRITE_UNMAPPED,
                         uc_const.UC_ERR_FETCH_UNMAPPED, uc_const.UC_ERR_READ_PROT,
                         uc_const.UC_ERR_WRITE_PROT, uc_const.UC_ERR_FETCH_PROT):
            status = "fault"
        else:
            status = f"err{e.errno}"
    # QEMU exception numbers: 1 = undefined instruction, 2 = SVC,
    # 4 = data abort (alignment here), 7 = BRK.
    if interrupts:
        status = {1: "unimpl", 2: "svc", 4: "align", 7: "brk"}.get(
            interrupts[0], f"intr{interrupts[0]}")
    final_pc = uc.reg_read(UC_ARM64_REG_PC)
    if status == "fault" and not (CODE_BASE <= final_pc < CODE_BASE + CODE_SIZE):
        # The instruction completed (a branch out of the code page); Unicorn
        # then faulted fetching the next one, which is outside this test.
        status = "ok"

    regs_out = [uc.reg_read(UC_ARM64_REG_X0 + i) for i in range(29)]
    regs_out += [uc.reg_read(UC_ARM64_REG_X29), uc.reg_read(UC_ARM64_REG_X30)]
    mem_diff = []
    initial_code = bytearray(INITIAL[CODE_BASE])
    off = pc - CODE_BASE
    initial_code[off:off + 4] = insn.to_bytes(4, "little")
    for base, size, init in [(CODE_BASE, CODE_SIZE, bytes(initial_code)),
                             (DATA_BASE, DATA_SIZE, INITIAL[DATA_BASE])]:
        now = bytes(uc.mem_read(base, size))
        if now != init:
            mem_diff += [f"{base + i:x}={now[i]:x}" for i in range(size) if now[i] != init[i]]
    v_out = [uc.reg_read(UC_ARM64_REG_Q0 + i) for i in range(32)]
    qc = (uc.reg_read(UC_ARM64_REG_FPSR) >> 27) & 1
    return status, uc.reg_read(UC_ARM64_REG_PC), uc.reg_read(UC_ARM64_REG_NZCV), regs_out, \
        uc.reg_read(UC_ARM64_REG_SP), v_out, mem_diff, qc


def check_smax_smin(insn, regs, sp, ours):
    """Reference model for LDSMAX/LDSMIN, used because Unicorn's QEMU base
    does not sign-extend the memory operand for sizes below 64 bits.
    Returns True if our result matches the ARM definition."""
    size, rs, rn, rt = insn >> 30, (insn >> 16) & 31, (insn >> 5) & 31, insn & 31
    is_min = (insn >> 12) & 7 == 0b101
    nbytes = 1 << size
    addr = sp if rn == 31 else regs[rn]
    if addr % nbytes:
        return ours[0].startswith("align")
    region = next((b for b in INITIAL if b <= addr and addr + nbytes <= b + len(INITIAL[b])), None)
    if region is None:
        return ours[0] == "fault"
    old_bytes = INITIAL[region][addr - region:addr - region + nbytes]
    old = int.from_bytes(old_bytes, "little")
    mask = (1 << (8 * nbytes)) - 1
    value = (0 if rs == 31 else regs[rs]) & mask

    def signed(x):
        return x - (1 << (8 * nbytes)) if x >> (8 * nbytes - 1) else x

    new = min(old, value, key=signed) if is_min else max(old, value, key=signed)
    new_bytes = new.to_bytes(nbytes, "little")
    expected_mem = [f"{addr + i:x}={new_bytes[i]:x}" for i in range(nbytes) if new_bytes[i] != old_bytes[i]]
    expected_regs = list(regs)
    if rt != 31:
        expected_regs[rt] = old
    return ours[0] == "ok" and ours[3] == expected_regs and ours[6] == expected_mem


def run_unicorn_isolated(*case):
    """Runs Unicorn in a forked child. Some unallocated FP16 encodings hit
    an assertion inside Unicorn's QEMU ("code should not be reached") that
    aborts the whole process."""
    import os
    import pickle
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        try:
            data = pickle.dumps(run_unicorn(*case))
        except BaseException:  # noqa: BLE001 - report any failure as a crash
            data = b""
        os.write(w, data)
        os._exit(0)
    os.close(w)
    chunks = []
    while True:
        chunk = os.read(r, 65536)
        if not chunk:
            break
        chunks.append(chunk)
    os.close(r)
    os.waitpid(pid, 0)
    data = b"".join(chunks)
    return pickle.loads(data) if data else ("qemu-crash",)


def parse_ours(line):
    if line == "skip":
        return None
    head, _, mem = line.partition("|")
    parts = head.split()
    vals = [int(x, 16) for x in parts[1:]]
    return parts[0], vals[0], vals[1], vals[3:34], vals[34], vals[35:67], mem.split(), vals[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--show", type=int, default=15, help="mismatches to print")
    ap.add_argument("--only", default="", help="comma-separated class name prefixes")
    args = ap.parse_args()
    if args.only:
        prefixes = tuple(args.only.split(","))
        CLASSES[:] = [c for c in CLASSES if c[0].startswith(prefixes)]

    rng = random.Random(args.seed)
    cases = [make_case(rng) for _ in range(args.cases)]
    stdin = "".join(
        " ".join(f"{v:x}" for v in [insn, pc, nzcv, fpcr, *regs, sp, *vregs]) + "\n"
        for _, insn, pc, nzcv, fpcr, regs, sp, vregs in cases
    )
    ours_lines = subprocess.run([str(RUNNER)], input=stdin, capture_output=True,
                                text=True, check=True).stdout.splitlines()

    stats = collections.Counter()
    gaps = collections.Counter()
    shown = 0
    for (name, insn, pc, nzcv, fpcr, regs, sp, vregs), line in zip(cases, ours_lines):
        ours = parse_ours(line)
        if ours is None:
            stats["skipped (unpredictable)"] += 1
            continue
        if name.startswith("fp16"):
            theirs = run_unicorn_isolated(insn, pc, nzcv, fpcr, regs, sp, vregs)
            if theirs[0] == "qemu-crash":
                stats["Unicorn crashed" + (" (we reject too)" if ours[0] == "unimpl" else " (WE EXECUTE)")] += 1
                continue
        else:
            theirs = run_unicorn(insn, pc, nzcv, fpcr, regs, sp, vregs)
        o_status, t_status = ours[0], theirs[0]

        if o_status == "unimpl" and t_status == "unimpl":
            stats["both reject"] += 1
            continue
        if o_status == "unimpl":
            gaps[name] += 1
            stats["not implemented yet"] += 1
            continue
        if o_status == "fault" and t_status == "fault":
            stats["both fault"] += 1
            continue
        if o_status.startswith("align"):
            if t_status == "align":
                stats["both alignment-fault"] += 1
                continue
            if o_status == "align-lse2" and t_status == "ok":
                # Unicorn's "max" CPU has FEAT_LSE2 (unaligned atomics within
                # 16 bytes); the Switch 2's Cortex-A78C does not.
                stats["CPU model difference (LSE2)"] += 1
                continue
            if o_status == "align-stx" and t_status == "ok":
                # ARM pseudocode (AArch64.ExclusiveMonitorsPass) checks
                # alignment first; QEMU skips it when the monitor fails.
                stats["known QEMU difference (STXR)"] += 1
                continue
            if t_status == "fault":
                # Alignment is checked before translation in the ARM
                # pseudocode; QEMU reports the unmapped page first.
                stats["known QEMU difference (fault order)"] += 1
                continue

        if name == "atomic" and (insn >> 15) & 1 == 0 and (insn >> 13) & 3 == 0b10:
            # LDSMAX/LDSMIN: checked against our own reference model, see
            # check_smax_smin().
            if check_smax_smin(insn, regs, sp, ours):
                stats["LDSMAX/LDSMIN ok (reference model)"] += 1
                continue

        is_store_exclusive = name == "exclusive" and (insn >> 21) & 0b101 == 0 and not (insn >> 23) & 1
        if is_store_exclusive and o_status == "ok" and t_status == "fault":
            # "It is IMPLEMENTATION DEFINED whether the detection of memory
            # aborts happens before or after the check on the local
            # Exclusives monitor" (AArch64.ExclusiveMonitorsPass). We check
            # the monitor first and report failure (status 1); QEMU faults.
            rs = (insn >> 16) & 31
            if rs == 31 or ours[3][rs] == 1:
                stats["impl. defined (STXR abort order)"] += 1
                continue

        if (name == "simd_shift_imm_scalar" and (insn >> 20) & 0b111 == 0b001
                and (insn >> 11) & 0b11111 == 0b11111 and o_status == t_status == "ok"):
            # FCVTZS/FCVTZU (scalar, fixed-point) on a half: the ARM ARM sets
            # esize = 16 and datasize = esize, so the result is 16 bits.
            # Unicorn's QEMU base writes 32 sign-extended bits.
            rd = insn & 31
            t_v = list(theirs[5])
            t_v[rd] &= 0xFFFF
            if ours[1:5] == theirs[1:5] and ours[5] == t_v and ours[6:] == theirs[6:]:
                stats["known Unicorn bug (FP16 fixed-point width)"] += 1
                continue

        same = (o_status == t_status) and ours[1:] == theirs[1:]
        if same:
            stats["match"] += 1
            continue
        stats["MISMATCH"] += 1
        if shown < args.show:
            shown += 1
            print(f"--- MISMATCH [{name}] insn={insn:08x}")
            asm = subprocess.run(["llvm-mc", "-triple=aarch64", "-mattr=+all", "-disassemble"],
                                 input=" ".join(f"0x{b:02x}" for b in insn.to_bytes(4, "little")),
                                 capture_output=True, text=True).stdout.strip().splitlines()
            print("    ", asm[-1].strip() if asm else "?")
            print(f"     ours:    status={o_status} pc={ours[1]:x} nzcv={ours[2]:x} sp={ours[4]:x}")
            print(f"     unicorn: status={t_status} pc={theirs[1]:x} nzcv={theirs[2]:x} sp={theirs[4]:x}")
            for i in range(31):
                if ours[3][i] != theirs[3][i]:
                    print(f"     x{i}: ours={ours[3][i]:x} unicorn={theirs[3][i]:x} (in={regs[i]:x})")
            for i in range(32):
                if ours[5][i] != theirs[5][i]:
                    print(f"     v{i}: ours={ours[5][i]:032x}\n          unic={theirs[5][i]:032x}"
                          f"\n          in  ={vregs[i]:032x}")
            if ours[6] != theirs[6]:
                print(f"     mem ours={ours[6][:8]} unicorn={theirs[6][:8]}")
            if ours[7] != theirs[7]:
                print(f"     FPSR.QC ours={ours[7]} unicorn={theirs[7]}")
            if fpcr:
                print("     (FPCR.DN set)")

    print()
    for k, v in sorted(stats.items()):
        print(f"{k:28} {v}")
    if gaps:
        print("not implemented, by class:", dict(gaps.most_common()))
    sys.exit(1 if stats["MISMATCH"] else 0)


if __name__ == "__main__":
    main()
