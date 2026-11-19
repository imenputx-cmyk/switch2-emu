/* GPU command submission through libnx's own (ISC) nvidia stack: a GPU
 * address space and channel, pushbuffers submitted as GPFIFO entries,
 * fences on syncpoints. Tests the DMA copy engine (constant fill,
 * pitch -> block-linear copy), inline-to-memory uploads and semaphores,
 * then renders frames with 3D-engine clears straight into the display's
 * framebuffers and presents them. Method numbers are from NVIDIA's
 * open-gpu-doc class headers (MIT). */
#include <string.h>
#include "switch/types.h"
#include "switch/result.h"
#include "switch/kernel/svc.h"
#include "switch/runtime/env.h"
#include "switch/services/sm.h"
#include "switch/services/applet.h"
#include "switch/services/vi.h"
#include "switch/services/nv.h"
#include "switch/nvidia/map.h"
#include "switch/nvidia/fence.h"
#include "switch/nvidia/gpu.h"
#include "switch/nvidia/address_space.h"
#include "switch/nvidia/gpu_channel.h"
#include "switch/display/native_window.h"
#include "switch/display/framebuffer.h"

void glue_set_heap(void *base, size_t size);
void glue_setup_tls(void);
void virtmemSetup(void);
void *__libnx_aligned_alloc(size_t alignment, size_t size);

static char out[256];
static u32 out_len;
static void put(const char *s) { while (*s) out[out_len++] = *s++; }
static void put_hex(u64 v) {
    static const char d[] = "0123456789abcdef";
    put("0x");
    int started = 0;
    for (int i = 60; i >= 0; i -= 4) {
        int n = (v >> i) & 15;
        if (n || started || i == 0) { out[out_len++] = d[n]; started = 1; }
    }
}
static void line(const char *label, u64 v) {
    put(label); put_hex(v); put("\n");
    svcOutputDebugString(out, out_len);
    out_len = 0;
}

/* ---- pushbuffer building ---- */

#define SUB_3D 0
#define SUB_COPY 4

static u32 *cmd;          /* command buffer (CPU view) */
static u32 cmd_pos, cmd_start;
static iova_t cmd_iova;   /* its GPU address */

/* Increasing methods: header, then `count` values. */
static void incr(u32 sub, u32 method, u32 count) {
    cmd[cmd_pos++] = (1u << 29) | (count << 16) | (sub << 13) | (method >> 2);
}
/* Non-increasing: all values go to the same method. */
static void nonincr(u32 sub, u32 method, u32 count) {
    cmd[cmd_pos++] = (3u << 29) | (count << 16) | (sub << 13) | (method >> 2);
}
static void w(u32 v) { cmd[cmd_pos++] = v; }
static void set1(u32 sub, u32 method, u32 v) { incr(sub, method, 1); w(v); }
static u32 fbits(float f) { u32 u; memcpy(&u, &f, 4); return u; }

static NvGpuChannel ch;

/* Submits everything written since the last flush and waits for it. */
static Result flush(void) {
    /* Make the GPU signal the channel's syncpoint when done. */
    set1(SUB_3D, 0x02C8, nvGpuChannelGetSyncpointId(&ch)); /* INCREMENT_SYNC_POINT */
    nvGpuChannelIncrFence(&ch);
    Result rc = nvGpuChannelAppendEntry(&ch, cmd_iova + cmd_start * 4, cmd_pos - cmd_start, 0, 0);
    if (R_SUCCEEDED(rc)) rc = nvGpuChannelKickoff(&ch);
    NvFence fence;
    nvGpuChannelGetFence(&ch, &fence);
    if (R_SUCCEEDED(rc)) rc = nvFenceWait(&fence, -1);
    cmd_start = cmd_pos;
    return rc;
}

#define W 1280
#define H 720

static void clear_rect(u32 x0, u32 y0, u32 x1, u32 y1, float r, float g, float b) {
    incr(SUB_3D, 0x0E04, 2);              /* SET_SCISSOR_HORIZONTAL/VERTICAL(0) */
    w(x0 | x1 << 16);
    w(y0 | y1 << 16);
    incr(SUB_3D, 0x0D80, 4);              /* SET_COLOR_CLEAR_VALUE */
    w(fbits(r)); w(fbits(g)); w(fbits(b)); w(fbits(1.0f));
    set1(SUB_3D, 0x19D0, 0x3C);           /* CLEAR_SURFACE: RGBA, target 0 */
}

int hb_main(void *ctx, u64 x1, void *saved_lr) {
    (void)x1;
    glue_setup_tls();
    envSetup(ctx, INVALID_HANDLE, (LoaderReturnFn)saved_lr);
    void *heap = 0;
    svcSetHeapSize(&heap, 0x4000000);
    glue_set_heap(heap, 0x4000000);
    virtmemSetup();
    smInitialize();
    appletInitialize();
    viInitialize(ViServiceType_Default);

    /* The display, as in hb_gfx: but the GPU will draw the frames. */
    ViDisplay disp; ViLayer layer; NWindow win; Framebuffer fb;
    viOpenDefaultDisplay(&disp);
    viCreateLayer(&disp, &layer);
    nwindowCreateFromLayer(&win, &layer);
    line("framebufferCreate = ", framebufferCreate(&fb, &win, W, H, PIXEL_FORMAT_RGBA_8888, 2));

    line("nvGpuInit = ", nvGpuInit());
    line("3D class = ", nvGpuGetCharacteristics()->threed_class);
    NvAddressSpace as;
    line("nvAddressSpaceCreate = ", nvAddressSpaceCreate(&as, 0x10000));
    line("nvGpuChannelCreate = ", nvGpuChannelCreate(&ch, &as, NvChannelPriority_Medium));

    /* Command buffer and a scratch buffer, mapped into the GPU. */
    NvMap cmd_map, scratch_map;
    void *cmd_mem = __libnx_aligned_alloc(0x1000, 0x10000);
    u8 *scratch = __libnx_aligned_alloc(0x1000, 0x10000);
    memset(scratch, 0, 0x10000);
    nvMapCreate(&cmd_map, cmd_mem, 0x10000, 0x1000, NvKind_Pitch, true);
    nvMapCreate(&scratch_map, scratch, 0x10000, 0x1000, NvKind_Pitch, true);
    iova_t scratch_iova, fb_iova;
    line("map cmd = ", nvAddressSpaceMap(&as, nvMapGetHandle(&cmd_map), true, NvKind_Pitch, &cmd_iova));
    line("map scratch = ", nvAddressSpaceMap(&as, nvMapGetHandle(&scratch_map), true, NvKind_Pitch, &scratch_iova));
    line("map framebuffer = ", nvAddressSpaceMap(&as, nvMapGetHandle(&fb.map), true, NvKind_Generic_16BX2, &fb_iova));
    cmd = cmd_mem;

    /* Bind classes to subchannels. */
    set1(SUB_3D, 0, 0xB197);
    set1(SUB_COPY, 0, 0xB0B5);

    /* 1. Copy engine fill: 64 words of 0x12345678 at scratch+0. */
    incr(SUB_COPY, 0x0408, 2); w(scratch_iova >> 32); w((u32)scratch_iova);
    set1(SUB_COPY, 0x0418, 64);                          /* LINE_LENGTH_IN (elements) */
    set1(SUB_COPY, 0x0700, 0x12345678);                  /* SET_REMAP_CONST_A */
    set1(SUB_COPY, 0x0708, 4 | (3 << 16));               /* dst x = const A, 4-byte */
    set1(SUB_COPY, 0x0300, 2 | (1 << 7) | (1 << 8) | (1 << 10)); /* LAUNCH_DMA */

    /* 2. Inline upload of 3 words at scratch+0x400 through the 3D class. */
    incr(SUB_3D, 0x0180, 2); w(12); w(1);                /* LINE_LENGTH_IN, LINE_COUNT */
    incr(SUB_3D, 0x0188, 2); w((scratch_iova + 0x400) >> 32); w((u32)(scratch_iova + 0x400));
    set1(SUB_3D, 0x01B0, 1);                             /* LAUNCH_DMA: pitch */
    nonincr(SUB_3D, 0x01B4, 3); w(0xAAAA0001); w(0xBBBB0002); w(0xCCCC0003); /* LOAD_INLINE_DATA */

    /* 3. Copy 256 bytes x 4 lines from scratch+0 (pitch 256) into a
     *    block-linear surface at scratch+0x1000, then back out to
     *    scratch+0x3000 (pitch 256): must round-trip. */
    incr(SUB_COPY, 0x0400, 8);
    w(scratch_iova >> 32); w((u32)scratch_iova);
    w((scratch_iova + 0x1000) >> 32); w((u32)(scratch_iova + 0x1000));
    w(256); w(0); w(256); w(4);
    incr(SUB_COPY, 0x070C, 2); w(0); w(256);             /* dst: 1 GOB high, 256 bytes wide */
    set1(SUB_COPY, 0x0300, 2 | (1 << 7) | (1 << 9));     /* pitch -> block-linear */
    incr(SUB_COPY, 0x0400, 8);
    w((scratch_iova + 0x1000) >> 32); w((u32)(scratch_iova + 0x1000));
    w((scratch_iova + 0x3000) >> 32); w((u32)(scratch_iova + 0x3000));
    w(0); w(256); w(256); w(4);
    incr(SUB_COPY, 0x0728, 2); w(0); w(256);             /* src: 1 GOB high, 256 bytes wide */
    set1(SUB_COPY, 0x0300, 2 | (1 << 8) | (1 << 9));     /* block-linear -> pitch */

    /* 4. Semaphore release from the copy engine: scratch+0x800 = 0x5E3A. */
    incr(SUB_COPY, 0x0240, 3); w((scratch_iova + 0x800) >> 32); w((u32)(scratch_iova + 0x800)); w(0x5E3A);
    set1(SUB_COPY, 0x0300, 1 << 3);                      /* no transfer, release one word */

    line("submit = ", flush());
    u32 *s32 = (u32 *)scratch;
    line("fill word 0 = ", s32[0]);
    line("fill word 63 = ", s32[63]);
    line("fill word 64 = ", s32[64]);
    line("inline 0 = ", s32[0x400 / 4]);
    line("inline 2 = ", s32[0x400 / 4 + 2]);
    int same = 1;
    for (int k = 0; k < 256; k++) same &= s32[0x3000 / 4 + k] == s32[k];
    int swizzled = 0;
    for (int k = 0; k < 256; k++) swizzled |= s32[0x1000 / 4 + k] != 0;
    line("block-linear round trip = ", same && swizzled);
    line("semaphore = ", s32[0x800 / 4]);

    /* 5. Frames drawn by the GPU: 3D clears into the framebuffer. */
    incr(SUB_3D, 0x0800, 6);                             /* SET_COLOR_TARGET_*(0) */
    w(0); w(0); w(W); w(H); w(0xD5); w(4 << 4);          /* addr set per frame; RGBA8, 16 GOBs */
    set1(SUB_3D, 0x0E00, 1);                             /* SET_SCISSOR_ENABLE(0) */
    for (u32 frame = 0; frame < 3; frame++) {
        u32 stride;
        framebufferBegin(&fb, &stride);
        iova_t target = fb_iova + win.cur_slot * fb.fb_size;
        incr(SUB_3D, 0x0800, 2); w(target >> 32); w((u32)target);
        clear_rect(0, 0, W, H, 0.05f, 0.05f, 0.15f);
        /* Three stripes and a square that moves each frame. */
        clear_rect(0, 160, W, 240, 0.9f, 0.2f, 0.2f);
        clear_rect(0, 320, W, 400, 0.2f, 0.8f, 0.3f);
        clear_rect(0, 480, W, 560, 0.2f, 0.4f, 0.95f);
        clear_rect(100 + frame * 300, 260, 300 + frame * 300, 460, 1.0f, 0.9f, 0.2f);
        Result rc = flush();
        if (rc) { line("frame submit failed: ", rc); return 1; }
        framebufferEnd(&fb);
        line("gpu frame ", frame);
    }
    return 0;
}
