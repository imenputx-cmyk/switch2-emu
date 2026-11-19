#!/usr/bin/env python3
"""Runs a test-program ELF in Unicorn with the same test ABI as emu-run.

Used to check whole-program output against an independent emulator:
    python3 tools/run_unicorn.py tests/programs/float.elf
"""
import struct
import sys

from unicorn import UC_ARCH_ARM64, UC_HOOK_INTR, UC_MODE_ARM, Uc
from unicorn.arm64_const import (UC_ARM64_REG_SP, UC_ARM64_REG_X0, UC_ARM64_REG_X1,
                                 UC_ARM64_REG_X8, UC_CPU_ARM64_MAX)

STACK_TOP, STACK_SIZE = 0x7FFF_0000_0000, 0x10_0000


def run(path):
    elf = open(path, "rb").read()
    uc = Uc(UC_ARCH_ARM64, UC_MODE_ARM)
    uc.ctl_set_cpu_model(UC_CPU_ARM64_MAX)
    entry, phoff = struct.unpack_from("<QQ", elf, 24)
    phnum = struct.unpack_from("<H", elf, 56)[0]
    mapped = set()
    for i in range(phnum):
        p_type, _, off, vaddr, _, filesz, memsz, _ = struct.unpack_from("<IIQQQQQQ", elf, phoff + i * 56)
        if p_type != 1 or memsz == 0:
            continue
        for page in range(vaddr // 4096, (vaddr + memsz - 1) // 4096 + 1):
            if page not in mapped:
                uc.mem_map(page * 4096, 4096)
                mapped.add(page)
        uc.mem_write(vaddr, elf[off:off + filesz])
    uc.mem_map(STACK_TOP - STACK_SIZE, STACK_SIZE)
    uc.reg_write(UC_ARM64_REG_SP, STACK_TOP)

    out = bytearray()
    status = [None]

    def on_svc(u, _intno, _data):
        if u.reg_read(UC_ARM64_REG_X8) == 1:
            out.extend(u.mem_read(u.reg_read(UC_ARM64_REG_X0), u.reg_read(UC_ARM64_REG_X1)))
        else:
            status[0] = u.reg_read(UC_ARM64_REG_X0)
            u.emu_stop()

    uc.hook_add(UC_HOOK_INTR, on_svc)
    uc.emu_start(entry, 0)
    return bytes(out), status[0]


if __name__ == "__main__":
    output, code = run(sys.argv[1])
    sys.stdout.buffer.write(output)
    sys.exit(code)
