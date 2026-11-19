/* Breakout: a small game for testing the emulator's display and input
 * paths end to end. It uses libnx's framebuffer API (vi, binder buffer
 * queue, nvdrv) and pad API (hid shared memory).
 *
 * Controls: D-pad / left stick move the paddle, A launches the ball,
 * A restarts after game over, Plus quits. Until a button is pressed the
 * game plays itself (attract mode).
 *
 * The internal resolution is deliberately low (W x H); the front end
 * scales it up, which makes this a test bed for upscaling as well. */
#include <string.h>
#include "switch/types.h"
#include "switch/result.h"
#include "switch/arm/counter.h"
#include "switch/kernel/svc.h"
#include "switch/runtime/env.h"
#include "switch/runtime/pad.h"
#include "switch/services/sm.h"
#include "switch/services/applet.h"
#include "switch/services/hid.h"
#include "switch/services/vi.h"
#include "switch/display/native_window.h"
#include "switch/display/framebuffer.h"

void glue_set_heap(void *base, size_t size);
void glue_setup_tls(void);
void virtmemSetup(void);

#ifndef W
#define W 320
#endif
#ifndef H
#define H 180
#endif

/* ---- output ---- */

static char out[128];
static u32 out_len;
static void put(const char *s) { while (*s) out[out_len++] = *s++; }
static void put_dec(u64 v) {
    char b[24]; int n = 0;
    do { b[n++] = '0' + v % 10; v /= 10; } while (v);
    while (n) out[out_len++] = b[--n];
}
static void flush(void) { svcOutputDebugString(out, out_len); out_len = 0; }

/* ---- drawing ---- */

static u32 *px;
static u32 pitch_px;

static u32 rgb(u32 r, u32 g, u32 b) { return r | g << 8 | b << 16 | 0xFF000000u; }

static void rect(int x, int y, int w, int h, u32 c) {
    if (x < 0) { w += x; x = 0; }
    if (y < 0) { h += y; y = 0; }
    if (x + w > W) w = W - x;
    if (y + h > H) h = H - y;
    for (int j = 0; j < h; j++) {
        u32 *row = px + (y + j) * pitch_px + x;
        for (int i = 0; i < w; i++) row[i] = c;
    }
}

/* 3x5 font: one u16 per glyph, rows top to bottom, 3 bits each. */
static u16 glyph(char c) {
    static const u16 digits[10] = {
        0x7B6F, 0x2C97, 0x73E7, 0x73CF, 0x5BC9, 0x79CF, 0x79EF, 0x7249, 0x7BEF, 0x7BCF,
    };
    if (c >= '0' && c <= '9') return digits[c - '0'];
    switch (c) {
    case 'A': return 0x7BED; case 'B': return 0x6BAE; case 'C': return 0x7927; case 'D': return 0x6B6E;
    case 'E': return 0x79E7; case 'G': return 0x796F; case 'I': return 0x7497;
    case 'K': return 0x5BAD; case 'L': return 0x4927; case 'M': return 0x5FED;
    case 'N': return 0x6B6D; case 'O': return 0x7B6F; case 'P': return 0x7BE4;
    case 'R': return 0x7BAD; case 'S': return 0x79CF; case 'T': return 0x7492;
    case 'U': return 0x5B6F; case 'V': return 0x5B6A; case 'X': return 0x5AAD;
    case '+': return 0x05D0; case '-': return 0x01C0; case ':': return 0x0410;
    default: return 0;
    }
}

static void text(int x, int y, int scale, const char *s, u32 c) {
    for (; *s; s++, x += 4 * scale) {
        u16 g = glyph(*s);
        for (int r = 0; r < 5; r++)
            for (int col = 0; col < 3; col++)
                if (g & (1 << (14 - r * 3 - col)))
                    rect(x + col * scale, y + r * scale, scale, scale, c);
    }
}

static int text_width(const char *s, int scale) { return (int)strlen(s) * 4 * scale - scale; }

static void number(char *buf, u32 v) {
    char t[12]; int n = 0;
    do { t[n++] = '0' + v % 10; v /= 10; } while (v);
    int i = 0;
    while (n) buf[i++] = t[--n];
    buf[i] = 0;
}

/* ---- game ---- */

#define COLS 10
#define ROWS 6
#define BRICK_W (W / COLS)
#define BRICK_H (H / 30)
#define BRICK_TOP (H / 9)
#define PADDLE_W (W / 7)
#define PADDLE_H (H / 45 + 1)
#define PADDLE_Y (H - H / 12)
#define BALL (H / 60 + 2)

static const u32 row_colors[ROWS][3] = {
    {235, 64, 52}, {245, 140, 40}, {240, 210, 50}, {80, 200, 90}, {60, 150, 235}, {150, 90, 220},
};

typedef enum { Serve, Playing, GameOver } Phase;

static struct {
    u8 bricks[ROWS][COLS];
    int left;
    float paddle;   /* paddle centre x */
    float bx, by;   /* ball top-left */
    float vx, vy;   /* pixels per second */
    u32 score, lives, level;
    Phase phase;
    bool attract;   /* the game plays itself until a button is pressed */
    float serve_timer;
} g;

static void reset_bricks(void) {
    for (int r = 0; r < ROWS; r++)
        for (int c = 0; c < COLS; c++)
            g.bricks[r][c] = 1;
    g.left = ROWS * COLS;
}

static void serve(void) {
    g.phase = Serve;
    g.serve_timer = 0.6f;
    g.vx = 0; g.vy = 0;
}

static void new_game(void) {
    reset_bricks();
    g.score = 0; g.lives = 3; g.level = 1;
    g.paddle = W / 2.0f;
    serve();
}

static float speed(void) { return H * (0.9f + 0.15f * g.level); }

static void launch(void) {
    g.phase = Playing;
    float s = speed();
    g.vx = (g.score & 1) ? 0.45f * s : -0.45f * s;
    g.vy = -0.9f * s;
}

static float clampf(float v, float lo, float hi) { return v < lo ? lo : v > hi ? hi : v; }
static float absf(float v) { return v < 0 ? -v : v; }

static void update(float dt, u64 held, u64 pressed, float stick) {
    if (held | pressed)
        g.attract = false;

    if (g.phase == GameOver) {
        if (pressed & HidNpadButton_A || g.attract)
            new_game();
        return;
    }

    /* Paddle. */
    float dir = stick;
    if (held & HidNpadButton_AnyLeft) dir = -1;
    if (held & HidNpadButton_AnyRight) dir = 1;
    if (g.attract) {
        /* Aim off-centre so the ball keeps an angle; vary the side. */
        float aim = ((g.score / 10) % 3 - 1) * PADDLE_W * 0.3f;
        float target = g.phase == Playing ? g.bx + BALL / 2.0f - aim : W / 2.0f;
        dir = clampf((target - g.paddle) / 6.0f, -1, 1);
    }
    g.paddle = clampf(g.paddle + dir * W * 0.9f * dt, PADDLE_W / 2.0f, W - PADDLE_W / 2.0f);

    if (g.phase == Serve) {
        g.bx = g.paddle - BALL / 2.0f;
        g.by = PADDLE_Y - BALL - 1;
        g.serve_timer -= dt;
        if (pressed & HidNpadButton_A || (g.attract && g.serve_timer <= 0))
            launch();
        return;
    }

    /* Ball, in small steps so it cannot tunnel through a brick. */
    int steps = 1 + (int)((absf(g.vx) + absf(g.vy)) * dt / 2);
    float sdt = dt / steps;
    for (int s = 0; s < steps; s++) {
        g.bx += g.vx * sdt;
        g.by += g.vy * sdt;
        if (g.bx < 0) { g.bx = 0; g.vx = absf(g.vx); }
        if (g.bx > W - BALL) { g.bx = W - BALL; g.vx = -absf(g.vx); }
        if (g.by < 0) { g.by = 0; g.vy = absf(g.vy); }

        /* Paddle: the hit position sets the angle. */
        float pl = g.paddle - PADDLE_W / 2.0f;
        if (g.vy > 0 && g.by + BALL >= PADDLE_Y && g.by + BALL <= PADDLE_Y + PADDLE_H + 2 &&
            g.bx + BALL >= pl && g.bx <= pl + PADDLE_W) {
            float hit = (g.bx + BALL / 2.0f - g.paddle) / (PADDLE_W / 2.0f);
            float sp = speed() + 2.0f * (ROWS * COLS - g.left);
            hit = clampf(hit, -1, 1);
            /* Never (almost) vertical: it could bounce in one column forever. */
            if (absf(hit) < 0.2f) hit = hit < 0 ? -0.2f : 0.2f;
            g.vx = hit * 0.8f * sp;
            g.vy = -(sp - absf(g.vx) * 0.5f);
            g.by = PADDLE_Y - BALL;
        }

        /* Bricks. */
        int cx = (int)(g.bx + BALL / 2.0f), cy = (int)(g.by + BALL / 2.0f);
        int c = cx / BRICK_W, r = (cy - BRICK_TOP) / BRICK_H;
        if (cy >= BRICK_TOP && r >= 0 && r < ROWS && c >= 0 && c < COLS && g.bricks[r][c]) {
            g.bricks[r][c] = 0;
            g.left--;
            g.score += 10 * (ROWS - r);
            /* Bounce on the axis with the smaller overlap. */
            float ox = (cx - c * BRICK_W) - BRICK_W / 2.0f, oy = (cy - BRICK_TOP - r * BRICK_H) - BRICK_H / 2.0f;
            if (absf(ox) / BRICK_W > absf(oy) / BRICK_H) g.vx = -g.vx; else g.vy = -g.vy;
            if (g.left == 0) {
                g.level++;
                reset_bricks();
                serve();
                return;
            }
        }

        if (g.by > H) {
            if (--g.lives == 0) g.phase = GameOver;
            else serve();
            return;
        }
    }
}

static void draw(float t) {
    /* A slowly moving gradient: every pixel changes every frame. */
    int shift = (int)(t * 20);
    for (int y = 0; y < H; y++) {
        u32 v = (u32)(((y + shift) % (2 * H)));
        if (v >= (u32)H) v = 2 * H - 1 - v;
        u32 c = rgb(10 + v * 20 / H, 12 + v * 16 / H, 30 + v * 50 / H);
        u32 *row = px + y * pitch_px;
        for (int x = 0; x < W; x++) row[x] = c;
    }
    for (int r = 0; r < ROWS; r++)
        for (int c = 0; c < COLS; c++)
            if (g.bricks[r][c]) {
                const u32 *k = row_colors[r];
                rect(c * BRICK_W + 1, BRICK_TOP + r * BRICK_H + 1, BRICK_W - 2, BRICK_H - 2, rgb(k[0], k[1], k[2]));
                rect(c * BRICK_W + 1, BRICK_TOP + r * BRICK_H + 1, BRICK_W - 2, 1, rgb(k[0] / 2 + 127, k[1] / 2 + 127, k[2] / 2 + 127));
            }
    rect((int)(g.paddle - PADDLE_W / 2.0f), PADDLE_Y, PADDLE_W, PADDLE_H, rgb(230, 230, 240));
    if (g.phase != GameOver)
        rect((int)g.bx, (int)g.by, BALL, BALL, rgb(255, 255, 255));

    int s = H >= 360 ? 2 * (H / 360) : 1;
    char buf[32];
    number(buf, g.score);
    text(3 * s, 3 * s, s, buf, rgb(255, 255, 255));
    for (u32 i = 0; i < g.lives; i++)
        rect(W - (int)(i + 1) * 6 * s - 2 * s, 4 * s, 4 * s, 4 * s, rgb(235, 64, 52));
    char lv[16] = "LEVEL ";
    number(lv + 6, g.level);
    text((W - text_width(lv, s)) / 2, 3 * s, s, lv, rgb(200, 200, 220));
    if (g.phase == GameOver) {
        const char *m = "GAME OVER";
        text((W - text_width(m, 2 * s)) / 2, H / 2 - 6 * s, 2 * s, m, rgb(255, 80, 80));
        const char *m2 = "PRESS A";
        text((W - text_width(m2, s)) / 2, H / 2 + 8 * s, s, m2, rgb(220, 220, 220));
    } else if (g.attract) {
        const char *m = "DEMO - PRESS A";
        text((W - text_width(m, s)) / 2, H / 2 + 10 * s, s, m, rgb(220, 220, 220));
    }
}

int hb_main(void *ctx, u64 x1, void *saved_lr) {
    (void)x1;
    glue_setup_tls();
    envSetup(ctx, INVALID_HANDLE, (LoaderReturnFn)saved_lr);
    void *heap = 0;
    svcSetHeapSize(&heap, 0x4000000);
    glue_set_heap(heap, 0x4000000);
    virtmemSetup();
    if (R_FAILED(smInitialize()) || R_FAILED(appletInitialize()) || R_FAILED(hidInitialize()) ||
        R_FAILED(viInitialize(ViServiceType_Default))) {
        put("breakout: service init failed"); flush();
        return 1;
    }
    padConfigureInput(1, HidNpadStyleSet_NpadStandard);
    PadState pad;
    padInitializeDefault(&pad);

    ViDisplay disp;
    ViLayer layer;
    NWindow win;
    Framebuffer fb;
    if (R_FAILED(viOpenDefaultDisplay(&disp)) || R_FAILED(viCreateLayer(&disp, &layer)) ||
        R_FAILED(nwindowCreateFromLayer(&win, &layer)) ||
        R_FAILED(framebufferCreate(&fb, &win, W, H, PIXEL_FORMAT_RGBA_8888, 2)) ||
        R_FAILED(framebufferMakeLinear(&fb))) {
        put("breakout: display init failed"); flush();
        return 1;
    }

    g.attract = true;
    new_game();
    u64 start = armGetSystemTick(), last = start;
    u32 frames = 0;
    for (;;) {
        padUpdate(&pad);
        u64 held = padGetButtons(&pad), pressed = padGetButtonsDown(&pad);
        if (pressed & HidNpadButton_Plus)
            break;
        HidAnalogStickState st = padGetStickPos(&pad, 0);

        u64 now = armGetSystemTick();
        float dt = (float)armTicksToNs(now - last) / 1e9f;
        last = now;
        if (dt > 0.05f) dt = 0.05f;
        update(dt, held & ~HidNpadButton_Plus, pressed, st.x / 32767.0f);

        u32 stride;
        px = framebufferBegin(&fb, &stride);
        pitch_px = stride / 4;
        draw((float)armTicksToNs(now - start) / 1e9f);
        framebufferEnd(&fb);
        frames++;
    }

    put("breakout: score "); put_dec(g.score);
    put(", level "); put_dec(g.level);
    put(", frames "); put_dec(frames);
    put("\n"); flush();

    framebufferClose(&fb);
    nwindowClose(&win);
    viCloseLayer(&layer);
    viCloseDisplay(&disp);
    viExit();
    hidExit();
    appletExit();
    return 0;
}
