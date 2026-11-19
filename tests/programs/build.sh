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

# Homebrew: position-independent, converted to NRO with tools/elf2nro.py.
hb="--target=aarch64-none-elf -O2 -ffreestanding -fno-builtin -nostdlib -fPIE \
    -mcpu=cortex-a78c -fno-stack-protector -fuse-ld=lld \
    -Wl,-pie,--no-dynamic-linker,-z,max-page-size=4096,-T,homebrew/link.ld,-s"
clang $hb -o homebrew/hb_hello.elf homebrew/crt0.S homebrew/hb_relocate.c homebrew/hb_hello.c
python3 ../../tools/elf2nro.py homebrew/hb_hello.elf hb_hello.nro

# IPC test: compiled against libnx's header-only IPC code (ISC). Set LIBNX
# to a libnx checkout (default: a fresh shallow clone in /tmp).
LIBNX=${LIBNX:-/tmp/libnx}
[ -d "$LIBNX" ] || git clone -q --depth 1 https://github.com/switchbrew/libnx "$LIBNX"
clang $hb -Ihomebrew/include_stubs -I"$LIBNX/nx/include" -o homebrew/hb_ipc.elf \
    homebrew/crt0.S homebrew/hb_svc.S homebrew/hb_relocate.c homebrew/hb_ipc.c
python3 ../../tools/elf2nro.py homebrew/hb_ipc.elf hb_ipc.nro
rm homebrew/hb_ipc.elf
rm homebrew/hb_hello.elf
