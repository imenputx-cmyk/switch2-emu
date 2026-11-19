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

# Threads with libnx's own mutex, condvar, semaphore and light event code.
(
    cd libnx_compat
    clang $(echo "$hb" | sed 's|homebrew/link.ld|../homebrew/link.ld|') \
        -Os -ffunction-sections -fdata-sections -Wl,--gc-sections \
        -Iinclude -I"$L/include" -I"$L/include/switch" -I"$L/source" \
        -o hb_threads.elf crt_libnx.S ../homebrew/hb_relocate.c glue.c hb_threads.c \
        "$L/source/kernel/mutex.c" "$L/source/kernel/condvar.c" \
        "$L/source/kernel/semaphore.c" "$L/source/kernel/levent.c" "$L/source/kernel/svc.s"
    python3 ../../../tools/elf2nro.py hb_threads.elf ../hb_threads.nro
    rm hb_threads.elf
)

# Graphics: libnx's display stack (vi, binder buffer queue, nvdrv/nvmap)
# and framebuffer API. buffer_producer.c uses a GCC-only variable-length
# array inside a struct; a fixed-size copy is made for clang.
GFX_SRC="services/sm.c services/applet.c services/vi.c services/nv.c \
    services/set.c services/apm.c sf/sessionmgr.c kernel/random.c runtime/alloc.c \
    kernel/mutex.c kernel/condvar.c kernel/event.c kernel/shmem.c kernel/tmem.c \
    kernel/wait.c kernel/virtmem.c kernel/uevent.c kernel/utimer.c kernel/levent.c \
    runtime/hosversion.c runtime/env.c kernel/svc.s arm/cache.s display/binder.c \
    display/framebuffer.c display/native_window.c display/parcel.c nvidia/map.c \
    nvidia/fence.c nvidia/ioctl/nvhost-ctrl.c nvidia/ioctl/nvmap.c"
(
    cd libnx_compat
    sed -e 's/u32 ints\[input->native_handle->num_ints\];/u32 ints[0x80];/' \
        -e 's/memcpy(buf.ints, input->native_handle+1, sizeof(buf.ints));/memcpy(buf.ints, input->native_handle+1, buf.numInts * 4);/' \
        -e 's/parcelWriteFlattenedObject(&parcel, &buf, sizeof(buf));/parcelWriteFlattenedObject(\&parcel, \&buf, sizeof(buf) - sizeof(buf.ints) + buf.numInts * 4);/' \
        "$L/source/display/buffer_producer.c" > buffer_producer_clang.c
    clang $(echo "$hb" | sed 's|homebrew/link.ld|../homebrew/link.ld|') \
        -Os -ffunction-sections -fdata-sections -Wl,--gc-sections \
        -Iinclude -I"$L/include" -I"$L/include/switch" -I"$L/source" -I"$L/source/display" \
        -o hb_gfx.elf crt_libnx.S ../homebrew/hb_relocate.c glue.c hb_gfx.c buffer_producer_clang.c \
        $(for f in $GFX_SRC; do echo "$L/source/$f"; done)
    python3 ../../../tools/elf2nro.py hb_gfx.elf ../hb_gfx.nro
    # Breakout: the same display stack plus hid and libnx's pad API.
    clang $(echo "$hb" | sed 's|homebrew/link.ld|../homebrew/link.ld|') \
        -Os -ffunction-sections -fdata-sections -Wl,--gc-sections \
        -Iinclude -I"$L/include" -I"$L/include/switch" -I"$L/source" -I"$L/source/display" \
        -o hb_breakout.elf crt_libnx.S ../homebrew/hb_relocate.c glue.c hb_breakout.c \
        buffer_producer_clang.c "$L/source/services/hid.c" "$L/source/runtime/pad.c" \
        $(for f in $GFX_SRC; do echo "$L/source/$f"; done)
    python3 ../../../tools/elf2nro.py hb_breakout.elf ../hb_breakout.nro
    # GPU: libnx's nvidia stack (address space, channel, fences). Its
    # nvchannel.c also uses variable-length arrays in structs; the
    # functions doing so (all unused after 4.0.0) are dropped for clang.
    awk '/^Result nvioctlChannel_(SubmitGpfifo|Submit|MapCommandBuffer|UnmapCommandBuffer)\(/{skip=1} !skip{print} skip && /^}/{skip=0}' \
        "$L/source/nvidia/ioctl/nvchannel.c" > nvchannel_clang.c
    cat >> nvchannel_clang.c <<'EOC'
Result nvioctlChannel_SubmitGpfifo(u32 fd, nvioctl_gpfifo_entry *entries, u32 num_entries, u32 flags, nvioctl_fence *fence_inout) {
    (void)fd; (void)entries; (void)num_entries; (void)flags; (void)fence_inout;
    return MAKERESULT(Module_Libnx, LibnxError_IncompatSysVer);
}
EOC
    clang $(echo "$hb" | sed 's|homebrew/link.ld|../homebrew/link.ld|') \
        -Os -ffunction-sections -fdata-sections -Wl,--gc-sections \
        -Iinclude -I"$L/include" -I"$L/include/switch" -I"$L/source" -I"$L/source/display" \
        -I"$L/source/nvidia/ioctl" \
        -o hb_gpu.elf crt_libnx.S ../homebrew/hb_relocate.c glue.c hb_gpu.c \
        buffer_producer_clang.c nvchannel_clang.c \
        $(for f in $GFX_SRC nvidia/gpu.c nvidia/address_space.c nvidia/gpu_channel.c \
            nvidia/channel.c nvidia/ioctl/nvhost-as-gpu.c nvidia/ioctl/nvhost-ctrl-gpu.c; do echo "$L/source/$f"; done)
    python3 ../../../tools/elf2nro.py hb_gpu.elf ../hb_gpu.nro
    rm hb_gfx.elf hb_breakout.elf hb_gpu.elf buffer_producer_clang.c nvchannel_clang.c
)
