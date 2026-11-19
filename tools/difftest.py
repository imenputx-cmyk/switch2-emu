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
import subprocess
import sys
from pathlib import Path

from unicorn import (UC_ARCH_ARM64, UC_HOOK_INTR, UC_MODE_ARM, UC_PROT_ALL,
                     Uc, UcError)
from unicorn import arm64_const
from unicorn import unicorn_const as uc_const
from unicorn.arm64_const import (UC_ARM64_REG_NZCV, UC_ARM64_REG_PC,
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
]


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


def make_case(rng):
    name, mask, value = rng.choice(CLASSES)
    insn = (rng.getrandbits(32) & ~mask) | value
    pc = CODE_BASE + 0x800 + 4 * rng.randrange(-0x100, 0x100)
    nzcv = rng.getrandbits(4) << 28
    regs = [random_reg_value(rng) for _ in range(31)]
    sp = DATA_BASE + 0x8000 + 16 * rng.randrange(-0x100, 0x100)
    return name, insn, pc, nzcv, regs, sp


def run_unicorn(insn, pc, nzcv, regs, sp):
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
    return status, uc.reg_read(UC_ARM64_REG_PC), uc.reg_read(UC_ARM64_REG_NZCV), regs_out, \
        uc.reg_read(UC_ARM64_REG_SP), mem_diff


def parse_ours(line):
    if line == "skip":
        return None
    head, _, mem = line.partition("|")
    parts = head.split()
    vals = [int(x, 16) for x in parts[1:]]
    return parts[0], vals[0], vals[1], vals[2:33], vals[33], mem.split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--show", type=int, default=15, help="mismatches to print")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cases = [make_case(rng) for _ in range(args.cases)]
    stdin = "".join(
        " ".join(f"{v:x}" for v in [insn, pc, nzcv, *regs, sp]) + "\n"
        for _, insn, pc, nzcv, regs, sp in cases
    )
    ours_lines = subprocess.run([str(RUNNER)], input=stdin, capture_output=True,
                                text=True, check=True).stdout.splitlines()

    stats = collections.Counter()
    gaps = collections.Counter()
    shown = 0
    for (name, insn, pc, nzcv, regs, sp), line in zip(cases, ours_lines):
        ours = parse_ours(line)
        if ours is None:
            stats["skipped (unpredictable)"] += 1
            continue
        theirs = run_unicorn(insn, pc, nzcv, regs, sp)
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
        if o_status == "align" and t_status == "align":
            stats["both alignment-fault"] += 1
            continue
        if o_status == "align" and name == "exclusive" and t_status in ("ok", "fault"):
            # Known QEMU difference: the ARM pseudocode checks alignment
            # first (AArch64.ExclusiveMonitorsPass, CheckAlignment before
            # translation). QEMU skips it for a failing STXR and reports a
            # translation fault first for unaligned unmapped addresses.
            stats["known QEMU difference"] += 1
            continue

        same = (o_status == t_status) and ours[1:] == theirs[1:]
        if same:
            stats["match"] += 1
            continue
        stats["MISMATCH"] += 1
        if shown < args.show:
            shown += 1
            print(f"--- MISMATCH [{name}] insn={insn:08x}")
            asm = subprocess.run(["llvm-mc", "-triple=aarch64", "-disassemble"],
                                 input=" ".join(f"0x{b:02x}" for b in insn.to_bytes(4, "little")),
                                 capture_output=True, text=True).stdout.strip().splitlines()
            print("    ", asm[-1].strip() if asm else "?")
            print(f"     ours:    status={o_status} pc={ours[1]:x} nzcv={ours[2]:x} sp={ours[4]:x}")
            print(f"     unicorn: status={t_status} pc={theirs[1]:x} nzcv={theirs[2]:x} sp={theirs[4]:x}")
            for i in range(31):
                if ours[3][i] != theirs[3][i]:
                    print(f"     x{i}: ours={ours[3][i]:x} unicorn={theirs[3][i]:x} (in={regs[i]:x})")
            if ours[5] != theirs[5]:
                print(f"     mem ours={ours[5][:8]} unicorn={theirs[5][:8]}")

    print()
    for k, v in sorted(stats.items()):
        print(f"{k:28} {v}")
    if gaps:
        print("not implemented, by class:", dict(gaps.most_common()))
    sys.exit(1 if stats["MISMATCH"] else 0)


if __name__ == "__main__":
    main()
