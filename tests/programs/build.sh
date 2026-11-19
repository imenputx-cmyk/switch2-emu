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

# libnx test: libnx's real service clients (sm, applet, hid, time, fs, ...)
# compiled from source with small libc stand-ins (libnx_compat/), so the
# emulator answers the exact IPC traffic of a libnx app's startup.
L="$LIBNX/nx"
LIBNX_SRC="services/sm.c services/applet.c services/hid.c services/time.c \
    services/fs.c services/set.c services/apm.c sf/sessionmgr.c kernel/random.c \
    runtime/alloc.c kernel/mutex.c kernel/condvar.c kernel/event.c kernel/shmem.c \
    kernel/tmem.c kernel/wait.c kernel/virtmem.c kernel/uevent.c kernel/utimer.c \
    runtime/hosversion.c runtime/env.c kernel/svc.s"
(
    cd libnx_compat
    clang $(echo "$hb" | sed 's|homebrew/link.ld|../homebrew/link.ld|') \
        -Os -ffunction-sections -fdata-sections -Wl,--gc-sections \
        -Iinclude -I"$L/include" -I"$L/include/switch" -I"$L/source" \
        -o hb_libnx.elf crt_libnx.S ../homebrew/hb_relocate.c glue.c hb_libnx.c \
        $(for f in $LIBNX_SRC; do echo "$L/source/$f"; done)
    python3 ../../../tools/elf2nro.py hb_libnx.elf ../hb_libnx.nro
    rm hb_libnx.elf
)
