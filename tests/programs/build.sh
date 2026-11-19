#!/bin/sh
# Rebuilds the test program ELF files. The outputs are committed so tests
# do not need a cross compiler. Requires clang and ld.lld.
set -eu
cd "$(dirname "$0")"
common="--target=aarch64-none-elf -O2 -ffreestanding -fno-builtin -nostdlib \
    -fno-math-errno -fno-stack-protector -fno-pic -static \
    -fuse-ld=lld -Wl,-Ttext=0x80000000 -Wl,-e,_start -Wl,-z,max-page-size=4096 -s"
# hello and float stay scalar; simd targets the Switch 2 CPU and lets the
# compiler vectorize.
clang $common -mgeneral-regs-only -o hello.elf hello.c
clang $common -fno-vectorize -fno-slp-vectorize -o float.elf float.c
clang $common -mcpu=cortex-a78c -o simd.elf simd.c
clang $common -mcpu=cortex-a78c -o crypto.elf crypto.c
