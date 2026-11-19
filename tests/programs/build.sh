#!/bin/sh
# Rebuilds the test program ELF files. The outputs are committed so tests
# do not need a cross compiler. Requires clang and ld.lld.
set -eu
cd "$(dirname "$0")"
for src in *.c; do
    clang --target=aarch64-none-elf -O2 -ffreestanding -fno-builtin -nostdlib \
        -mgeneral-regs-only -fno-stack-protector -fno-pic -static \
        -fuse-ld=lld -Wl,-Ttext=0x80000000 -Wl,-e,_start -Wl,-z,max-page-size=4096 -s \
        -o "${src%.c}.elf" "$src"
done
