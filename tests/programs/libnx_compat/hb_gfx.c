/* Graphics output through libnx's own (ISC) display stack: vi (display,
 * layer), the binder buffer queue (IHOSBinderDriver) and nvdrv/nvmap
 * buffers, via libnx's framebuffer API. Draws a few frames of a test
 * pattern; the emulator presents each queued buffer. */
#include <string.h>
#include "switch/types.h"
#include "switch/result.h"
#include "switch/kernel/svc.h"
#include "switch/runtime/env.h"
#include "switch/runtime/hosversion.h"
#include "switch/services/sm.h"
#include "switch/services/applet.h"
#include "switch/services/vi.h"
#include "switch/display/native_window.h"
#include "switch/display/framebuffer.h"

void glue_set_heap(void *base, size_t size);
void glue_setup_tls(void);
void virtmemSetup(void);

static char out[256];
static u64 out_len;
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

#define W 1280
#define H 720

int hb_main(void *ctx, u64 x1, void *saved_lr) {
    (void)x1;
    glue_setup_tls();
    envSetup(ctx, INVALID_HANDLE, (LoaderReturnFn)saved_lr);
    if (!envHasHeapOverride()) {
        void *heap = 0;
        svcSetHeapSize(&heap, 0x4000000);
        glue_set_heap(heap, 0x4000000);
    }
    virtmemSetup();
    line("smInitialize = ", smInitialize());
    line("appletInitialize = ", appletInitialize());
    line("viInitialize = ", viInitialize(ViServiceType_Default));
    ViDisplay disp;
    line("viOpenDefaultDisplay = ", viOpenDefaultDisplay(&disp));
    ViLayer layer;
    line("viCreateLayer = ", viCreateLayer(&disp, &layer));
    line("viSetLayerScalingMode = ", viSetLayerScalingMode(&layer, ViScalingMode_FitToLayer));
    NWindow win;
    line("nwindowCreateFromLayer = ", nwindowCreateFromLayer(&win, &layer));
    Framebuffer fb;
    line("framebufferCreate = ", framebufferCreate(&fb, &win, W, H, PIXEL_FORMAT_RGBA_8888, 2));
    line("framebufferMakeLinear = ", framebufferMakeLinear(&fb));
    for (u32 frame = 0; frame < 3; frame++) {
        u32 stride;
        u32 *px = framebufferBegin(&fb, &stride);
        if (!px) { line("framebufferBegin failed at frame ", frame); return 1; }
        for (u32 y = 0; y < H; y++)
            for (u32 x = 0; x < W; x++) {
                /* RGBA8888 in memory: R, G, B, A. Moving squares on a gradient. */
                u32 r = x * 255 / W, g = y * 255 / H, b = frame * 100;
                if (((x + frame * 64) / 80 + y / 80) % 2 == 0) b = 255;
                px[y * (stride / 4) + x] = r | g << 8 | b << 16 | 0xFF000000u;
            }
        framebufferEnd(&fb);
        line("presented frame ", frame);
    }
    framebufferClose(&fb);
    nwindowClose(&win);
    viCloseLayer(&layer);
    viCloseDisplay(&disp);
    viExit();
    appletExit();
    return 0;
}
