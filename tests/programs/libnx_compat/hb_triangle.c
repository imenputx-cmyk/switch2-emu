/* Triangles drawn by the GPU: Maxwell vertex and pixel shaders
 * (triangle_shaders.h), a vertex buffer with positions and colors, the
 * 3D engine's draw methods, into the display's framebuffers. Uses
 * libnx's own (ISC) nvidia and display code. Method numbers are from
 * NVIDIA's open-gpu-doc class headers (MIT). */
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

#include "triangle_shaders.h"

typedef struct { float x, y, z, w, r, g, b, a; } Vertex;

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

    ViDisplay disp; ViLayer layer; NWindow win; Framebuffer fb;
    viOpenDefaultDisplay(&disp);
    viCreateLayer(&disp, &layer);
    nwindowCreateFromLayer(&win, &layer);
    line("framebufferCreate = ", framebufferCreate(&fb, &win, W, H, PIXEL_FORMAT_RGBA_8888, 2));
    line("nvGpuInit = ", nvGpuInit());
    NvAddressSpace as;
    line("nvAddressSpaceCreate = ", nvAddressSpaceCreate(&as, 0x10000));
    line("nvGpuChannelCreate = ", nvGpuChannelCreate(&ch, &as, NvChannelPriority_Medium));

    /* Command buffer, code (shaders) and vertex buffer, mapped. */
    NvMap cmd_map, code_map, vtx_map;
    void *cmd_mem = __libnx_aligned_alloc(0x1000, 0x10000);
    u8 *code = __libnx_aligned_alloc(0x1000, 0x1000);
    Vertex *vtx = __libnx_aligned_alloc(0x1000, 0x1000);
    memcpy(code, vertex_shader, sizeof(vertex_shader));
    memcpy(code + 0x200, pixel_shader, sizeof(pixel_shader));
    nvMapCreate(&cmd_map, cmd_mem, 0x10000, 0x1000, NvKind_Pitch, true);
    nvMapCreate(&code_map, code, 0x1000, 0x1000, NvKind_Pitch, true);
    nvMapCreate(&vtx_map, vtx, 0x1000, 0x1000, NvKind_Pitch, true);
    iova_t code_iova, vtx_iova, fb_iova;
    nvAddressSpaceMap(&as, nvMapGetHandle(&cmd_map), true, NvKind_Pitch, &cmd_iova);
    nvAddressSpaceMap(&as, nvMapGetHandle(&code_map), true, NvKind_Pitch, &code_iova);
    nvAddressSpaceMap(&as, nvMapGetHandle(&vtx_map), true, NvKind_Pitch, &vtx_iova);
    line("map framebuffer = ", nvAddressSpaceMap(&as, nvMapGetHandle(&fb.map), true, NvKind_Generic_16BX2, &fb_iova));
    cmd = cmd_mem;

    /* Pipeline state. */
    set1(SUB_3D, 0, 0xB197);
    incr(SUB_3D, 0x1608, 2); w(code_iova >> 32); w((u32)code_iova);      /* SET_PROGRAM_REGION_A/B */
    incr(SUB_3D, 0x2000 + 1 * 64, 2); w(1 | 1 << 4); w(0);                /* pipeline 1: vertex at +0 */
    incr(SUB_3D, 0x2000 + 5 * 64, 2); w(1 | 5 << 4); w(0x200);            /* pipeline 5: pixel at +0x200 */
    /* Attribute 0: stream 0 offset 0, attribute 1: offset 16; both
     * R32G32B32A32 floats. */
    incr(SUB_3D, 0x1160, 2);
    w(0 | 0 << 7 | 0x01 << 21 | 7u << 27);
    w(0 | 16 << 7 | 0x01 << 21 | 7u << 27);
    incr(SUB_3D, 0x1C00, 3); w(sizeof(Vertex) | 1 << 12); w(vtx_iova >> 32); w((u32)vtx_iova); /* stream 0 */
    incr(SUB_3D, 0x0A00, 6);                                              /* viewport 0: y up */
    w(fbits(W / 2.0f)); w(fbits(-H / 2.0f)); w(fbits(1.0f));
    w(fbits(W / 2.0f)); w(fbits(H / 2.0f)); w(fbits(0.0f));
    set1(SUB_3D, 0x121C, 1);                                              /* SET_CT_SELECT: 1 target, RT 0 */
    set1(SUB_3D, 0x0E00, 0);                                              /* no scissor */
    incr(SUB_3D, 0x0800, 6); w(0); w(0); w(W); w(H); w(0xD5); w(4 << 4);   /* RT 0: RGBA8, 16 GOBs */

    static const Vertex tri[3] = {
        { 0.0f,  0.8f, 0, 1, 1, 0, 0, 1 },
        { -0.7f, -0.6f, 0, 1, 0, 1, 0, 1 },
        { 0.7f, -0.6f, 0, 1, 0, 0, 1, 1 },
    };
    for (u32 frame = 0; frame < 2; frame++) {
        /* Frame 1 moves the triangle right and draws a strip quad too. */
        float dx = frame * 0.25f;
        for (int i = 0; i < 3; i++) { vtx[i] = tri[i]; vtx[i].x += dx; }
        static const Vertex quad[4] = {
            { -0.95f, 0.95f, 0, 1, 1, 1, 0, 1 }, { -0.95f, 0.65f, 0, 1, 1, 1, 0, 1 },
            { -0.55f, 0.95f, 0, 1, 1, 0, 1, 1 }, { -0.55f, 0.65f, 0, 1, 1, 0, 1, 1 },
        };
        for (int i = 0; i < 4; i++) vtx[3 + i] = quad[i];

        u32 stride;
        framebufferBegin(&fb, &stride);
        iova_t target = fb_iova + win.cur_slot * fb.fb_size;
        incr(SUB_3D, 0x0800, 2); w(target >> 32); w((u32)target);
        incr(SUB_3D, 0x0D80, 4); w(fbits(0.1f)); w(fbits(0.1f)); w(fbits(0.12f)); w(fbits(1.0f));
        set1(SUB_3D, 0x19D0, 0x3C);                                       /* clear */
        set1(SUB_3D, 0x1618, 4);                                          /* BEGIN: triangles */
        set1(SUB_3D, 0x0D74, 0);                                          /* SET_VERTEX_ARRAY_START */
        set1(SUB_3D, 0x0D78, 3);                                          /* DRAW_VERTEX_ARRAY: 3 */
        set1(SUB_3D, 0x1614, 0);                                          /* END */
        if (frame == 1)                                                   /* strip in one method */
            set1(SUB_3D, 0x1214, 3 | 4 << 16 | 5u << 28);
        Result rc = flush();
        if (rc) { line("frame submit failed: ", rc); return 1; }
        framebufferEnd(&fb);
        line("triangle frame ", frame);
    }
    return 0;
}
