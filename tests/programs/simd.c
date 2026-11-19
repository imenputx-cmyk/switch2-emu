/* Advanced SIMD test program: NEON intrinsics and auto-vectorized loops,
 * compiled for the Switch 2's Cortex-A78C. */
#include "rt.h"
#include <arm_neon.h>

static u64 checksum(const void *p, u64 n) {
    const u8 *b = p;
    u64 h = 0xcbf29ce484222325ul;
    for (u64 i = 0; i < n; i++) h = (h ^ b[i]) * 0x100000001b3ul;
    return h;
}

static void line(const char *label, u64 v) {
    print(label);
    print_hex(v);
    print("\n");
}

/* Auto-vectorized: the compiler turns these into NEON loops. */
__attribute__((noinline)) static void saxpy(float *y, const float *x, float a, int n) {
    for (int i = 0; i < n; i++) y[i] = a * x[i] + y[i];
}

__attribute__((noinline)) static int sum_bytes(const u8 *p, int n) {
    int s = 0;
    for (int i = 0; i < n; i++) s += p[i];
    return s;
}

__attribute__((noinline)) static void to_gray(const u8 *rgb, u8 *gray, int pixels) {
    for (int i = 0; i < pixels; i += 16) {
        uint8x16x3_t px = vld3q_u8(rgb + 3 * i); /* de-interleave R, G, B */
        uint16x8_t lo = vmull_u8(vget_low_u8(px.val[0]), vdup_n_u8(77));
        lo = vmlal_u8(lo, vget_low_u8(px.val[1]), vdup_n_u8(150));
        lo = vmlal_u8(lo, vget_low_u8(px.val[2]), vdup_n_u8(29));
        uint16x8_t hi = vmull_high_u8(px.val[0], vdupq_n_u8(77));
        hi = vmlal_high_u8(hi, px.val[1], vdupq_n_u8(150));
        hi = vmlal_high_u8(hi, px.val[2], vdupq_n_u8(29));
        vst1q_u8(gray + i, vcombine_u8(vrshrn_n_u16(lo, 8), vrshrn_n_u16(hi, 8)));
    }
}

/* Normalizes 4 vectors at once with FRSQRTE + one FRSQRTS Newton step,
 * the way game code typically does it. */
__attribute__((noinline)) static void normalize4(float32x4_t *x, float32x4_t *y, float32x4_t *z) {
    float32x4_t len2 = vfmaq_f32(vfmaq_f32(vmulq_f32(*x, *x), *y, *y), *z, *z);
    float32x4_t r = vrsqrteq_f32(len2);
    r = vmulq_f32(r, vrsqrtsq_f32(vmulq_f32(len2, r), r));
    *x = vmulq_f32(*x, r);
    *y = vmulq_f32(*y, r);
    *z = vmulq_f32(*z, r);
}

int main(void) {
    static float y[64], x[64];
    for (int i = 0; i < 64; i++) {
        x[i] = (float)i * 0.25f;
        y[i] = 100.0f - (float)i;
    }
    saxpy(y, x, 3.0f, 64);
    line("saxpy checksum   = ", checksum(y, sizeof y));

    static u8 bytes[1000];
    for (int i = 0; i < 1000; i++) bytes[i] = (u8)(i * 37 + 11);
    print("sum_bytes        = ");
    print_i64(sum_bytes(bytes, 1000));
    print("\n");

    static u8 rgb[64 * 3], gray[64];
    for (int i = 0; i < 64 * 3; i++) rgb[i] = (u8)(i * 13 + i / 3);
    to_gray(rgb, gray, 64);
    line("grayscale        = ", checksum(gray, sizeof gray));

    int16x8_t a = {32000, -32000, 100, -100, 32767, -32768, 0, 1};
    int16x8_t b = {1000, -1000, 100, -100, 1, -1, 0, 1};
    int16x8_t q = vqaddq_s16(a, b); /* saturating */
    print("vqaddq_s16       =");
    for (int i = 0; i < 8; i++) {
        print(" ");
        print_i64(q[i]);
    }
    print("\n");

    float32x4_t vx = {3, 1, 0, 2}, vy = {4, 2, 0, 3}, vz = {0, 2, 5, 6};
    normalize4(&vx, &vy, &vz);
    line("normalize4       = ", checksum(&vx, 16) ^ checksum(&vy, 16) ^ checksum(&vz, 16));
    print("first normal x*1000 = ");
    print_i64((i64)(vgetq_lane_f32(vx, 0) * 1000.0f));
    print("\n");

    uint8x16_t table = {'z','y','x','w','v','u','t','s','r','q','p','o','n','m','l','k'};
    uint8x16_t idx = {7, 4, 11, 11, 14, 200, 0, 14, 1, 11, 2, 12, 9, 15, 255, 6};
    uint8x16_t out = vqtbl1q_u8(table, idx); /* index >= 16 gives 0 */
    char text[17];
    vst1q_u8((u8 *)text, out);
    for (int i = 0; i < 16; i++)
        if (!text[i]) text[i] = '_';
    text[16] = 0;
    print("tbl              = ");
    print(text);
    print("\n");

    uint8x16_t ua = vdupq_n_u8(3), ub = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    uint32x4_t dot = vdotq_u32(vdupq_n_u32(1000), ua, ub);
    print("vdotq_u32        =");
    for (int i = 0; i < 4; i++) {
        print(" ");
        print_u64(dot[i]);
    }
    print("\n");
    return 0;
}
