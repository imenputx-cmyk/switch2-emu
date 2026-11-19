#!/usr/bin/env python3
"""Assemble AArch64 instructions with llvm-mc and print their encodings.

Usage: tools/asm.py "add x0, x1, #1" "ret"      (or one per line on stdin)
"""
import re
import subprocess
import sys


def assemble(lines):
    """Returns a list of (instruction word, text) pairs."""
    out = subprocess.run(
        # Cortex-A78C is the Switch 2 CPU: ARMv8.2 + LSE, RCpc, FP16, dotprod, CRC.
        ["llvm-mc", "-triple=aarch64", "-mcpu=cortex-a78c", "-show-encoding"],
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    result = []
    for line in out.splitlines():
        m = re.search(r"encoding: \[(.*)\]", line)
        if m:
            word = int.from_bytes(bytes(int(b, 16) for b in m.group(1).split(",")), "little")
            result.append((word, line.split("//")[0].strip()))
    return result


if __name__ == "__main__":
    lines = sys.argv[1:] or [l for l in sys.stdin.read().splitlines() if l.strip()]
    for word, text in assemble(lines):
        print(f"0x{word:08X}  {text}")
