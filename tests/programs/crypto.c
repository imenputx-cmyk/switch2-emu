/* Cryptographic Extension test program: AES-128, SHA-1 and SHA-256 with
 * ARM intrinsics, checked against the published test vectors. */
#include "rt.h"
#include <arm_neon.h>

static void print_bytes(const u8 *p, int n) {
    static const char hex[] = "0123456789abcdef";
    for (int i = 0; i < n; i++) {
        char c[2] = {hex[p[i] >> 4], hex[p[i] & 15]};
        sys_write(c, 2);
    }
}

static void print_words_be(const u32 *w, int n) {
    for (int i = 0; i < n; i++) {
        u8 b[4] = {(u8)(w[i] >> 24), (u8)(w[i] >> 16), (u8)(w[i] >> 8), (u8)w[i]};
        print_bytes(b, 4);
    }
}

/* SubWord via AESE: with all four columns equal, ShiftRows is a no-op. */
static u32 sub_word(u32 w) {
    uint8x16_t s = vaeseq_u8(vreinterpretq_u8_u32(vdupq_n_u32(w)), vdupq_n_u8(0));
    return vgetq_lane_u32(vreinterpretq_u32_u8(s), 0);
}

static void aes128_expand(const u8 key[16], uint8x16_t rk[11]) {
    static const u8 rcon[10] = {1, 2, 4, 8, 16, 32, 64, 128, 0x1b, 0x36};
    u32 w[44];
    memcpy(w, key, 16);
    for (int i = 4; i < 44; i++) {
        u32 t = w[i - 1];
        if (i % 4 == 0) t = sub_word((t >> 8) | (t << 24)) ^ rcon[i / 4 - 1];
        w[i] = w[i - 4] ^ t;
    }
    for (int r = 0; r < 11; r++) rk[r] = vreinterpretq_u8_u32(vld1q_u32(&w[4 * r]));
}

static uint8x16_t aes128_encrypt(uint8x16_t s, const uint8x16_t rk[11]) {
    for (int r = 0; r < 9; r++) s = vaesmcq_u8(vaeseq_u8(s, rk[r]));
    return veorq_u8(vaeseq_u8(s, rk[9]), rk[10]);
}

static uint8x16_t aes128_decrypt(uint8x16_t s, const uint8x16_t rk[11]) {
    s = vaesimcq_u8(vaesdq_u8(s, rk[10]));
    for (int r = 9; r > 1; r--) s = vaesimcq_u8(vaesdq_u8(s, vaesimcq_u8(rk[r])));
    return veorq_u8(vaesdq_u8(s, vaesimcq_u8(rk[1])), rk[0]);
}

static const u32 K256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

static void load_block_be(uint32x4_t msg[4], const u8 block[64]) {
    for (int i = 0; i < 4; i++) msg[i] = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block + 16 * i)));
}

static void sha256_block(u32 state[8], const u8 block[64]) {
    uint32x4_t abcd = vld1q_u32(&state[0]), efgh = vld1q_u32(&state[4]);
    uint32x4_t abcd0 = abcd, efgh0 = efgh, msg[4];
    load_block_be(msg, block);
    for (int i = 0; i < 16; i++) {
        uint32x4_t wk = vaddq_u32(msg[i % 4], vld1q_u32(&K256[4 * i]));
        uint32x4_t prev = abcd;
        abcd = vsha256hq_u32(abcd, efgh, wk);
        efgh = vsha256h2q_u32(efgh, prev, wk);
        if (i < 12)
            msg[i % 4] = vsha256su1q_u32(vsha256su0q_u32(msg[i % 4], msg[(i + 1) % 4]),
                                         msg[(i + 2) % 4], msg[(i + 3) % 4]);
    }
    vst1q_u32(&state[0], vaddq_u32(abcd, abcd0));
    vst1q_u32(&state[4], vaddq_u32(efgh, efgh0));
}

static void sha1_block(u32 state[5], const u8 block[64]) {
    static const u32 K[4] = {0x5a827999, 0x6ed9eba1, 0x8f1bbcdc, 0xca62c1d6};
    uint32x4_t abcd = vld1q_u32(state), abcd0 = abcd, msg[4];
    u32 e = state[4], e0 = e;
    load_block_be(msg, block);
    for (int i = 0; i < 20; i++) {
        uint32x4_t wk = vaddq_u32(msg[i % 4], vdupq_n_u32(K[i / 5]));
        u32 next_e = vsha1h_u32(vgetq_lane_u32(abcd, 0));
        if (i < 5)
            abcd = vsha1cq_u32(abcd, e, wk);
        else if (i >= 10 && i < 15)
            abcd = vsha1mq_u32(abcd, e, wk);
        else
            abcd = vsha1pq_u32(abcd, e, wk);
        e = next_e;
        if (i < 16)
            msg[i % 4] = vsha1su1q_u32(vsha1su0q_u32(msg[i % 4], msg[(i + 1) % 4], msg[(i + 2) % 4]),
                                       msg[(i + 3) % 4]);
    }
    vst1q_u32(state, vaddq_u32(abcd, abcd0));
    state[4] = e + e0;
}

/* One padded block for a message shorter than 56 bytes. */
static void pad_block(u8 block[64], const char *msg) {
    u64 n = str_len(msg);
    memset(block, 0, 64);
    memcpy(block, msg, n);
    block[n] = 0x80;
    block[63] = (u8)(n * 8);
    block[62] = (u8)((n * 8) >> 8);
}

int main(void) {
    /* FIPS-197 Appendix C.1 (AES-128). */
    u8 key[16], pt[16], out[16];
    for (int i = 0; i < 16; i++) {
        key[i] = (u8)i;
        pt[i] = (u8)(i * 0x11);
    }
    uint8x16_t rk[11];
    aes128_expand(key, rk);
    uint8x16_t ct = aes128_encrypt(vld1q_u8(pt), rk);
    vst1q_u8(out, ct);
    print("aes128 encrypt = ");
    print_bytes(out, 16);
    vst1q_u8(out, aes128_decrypt(ct, rk));
    print("\naes128 decrypt = ");
    print_bytes(out, 16);

    u8 block[64];
    pad_block(block, "abc");
    u32 s256[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                   0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    sha256_block(s256, block);
    print("\nsha256(abc)    = ");
    print_words_be(s256, 8);

    u32 s1[5] = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476, 0xc3d2e1f0};
    sha1_block(s1, block);
    print("\nsha1(abc)      = ");
    print_words_be(s1, 5);

    poly64_t a = 0x87, b = 0xffffffffffffffffull;
    poly128_t p = vmull_p64(a, b);
    print("\npmull64        = ");
    print_hex((u64)(p >> 64));
    print(" ");
    print_hex((u64)p);
    print("\n");
    return 0;
}
