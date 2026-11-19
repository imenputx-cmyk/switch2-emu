//! The Cryptographic Extension: AES, SHA-1, SHA-256 and 64-bit PMULL.
//!
//! Each function follows the ARM pseudocode for the instruction. Vectors
//! are 128-bit values with element 0 in the low bits, so AES byte `i` of
//! the state is bits [8i+7:8i], in FIPS-197 column-major order.

use std::sync::OnceLock;

/// Multiplication in GF(2^8) with the AES polynomial x^8 + x^4 + x^3 + x + 1.
fn gf_mul(mut a: u8, mut b: u8) -> u8 {
    let mut p = 0u8;
    while b != 0 {
        if b & 1 != 0 {
            p ^= a;
        }
        let carry = a & 0x80 != 0;
        a <<= 1;
        if carry {
            a ^= 0x1B;
        }
        b >>= 1;
    }
    p
}

/// The AES S-box and its inverse, computed from the definition
/// (multiplicative inverse in GF(2^8), then the affine transform).
fn sboxes() -> &'static ([u8; 256], [u8; 256]) {
    static TABLES: OnceLock<([u8; 256], [u8; 256])> = OnceLock::new();
    TABLES.get_or_init(|| {
        let mut sbox = [0u8; 256];
        let mut inv = [0u8; 256];
        for x in 0..=255u8 {
            let mut y = 0u8;
            if x != 0 {
                // x^254 is the inverse of x in GF(2^8).
                let mut acc = 1u8;
                for _ in 0..254 {
                    acc = gf_mul(acc, x);
                }
                y = acc;
            }
            let s = y
                ^ y.rotate_left(1)
                ^ y.rotate_left(2)
                ^ y.rotate_left(3)
                ^ y.rotate_left(4)
                ^ 0x63;
            sbox[x as usize] = s;
            inv[s as usize] = x;
        }
        (sbox, inv)
    })
}

fn bytes(v: u128) -> [u8; 16] {
    v.to_le_bytes()
}

fn from_bytes(b: [u8; 16]) -> u128 {
    u128::from_le_bytes(b)
}

/// AESE: AddRoundKey, ShiftRows, SubBytes.
pub fn aese(state: u128, key: u128) -> u128 {
    let s = bytes(state ^ key);
    let (sbox, _) = sboxes();
    let mut out = [0u8; 16];
    for c in 0..4 {
        for r in 0..4 {
            out[r + 4 * c] = sbox[s[r + 4 * ((c + r) % 4)] as usize];
        }
    }
    from_bytes(out)
}

/// AESD: AddRoundKey, InvShiftRows, InvSubBytes.
pub fn aesd(state: u128, key: u128) -> u128 {
    let s = bytes(state ^ key);
    let (_, inv) = sboxes();
    let mut out = [0u8; 16];
    for c in 0..4 {
        for r in 0..4 {
            out[r + 4 * ((c + r) % 4)] = inv[s[r + 4 * c] as usize];
        }
    }
    from_bytes(out)
}

fn mix(state: u128, m: [u8; 4]) -> u128 {
    let s = bytes(state);
    let mut out = [0u8; 16];
    for c in 0..4 {
        let col = &s[4 * c..4 * c + 4];
        for r in 0..4 {
            out[4 * c + r] = (0..4).fold(0u8, |acc, k| acc ^ gf_mul(m[(k + 4 - r) % 4], col[k]));
        }
    }
    from_bytes(out)
}

/// AESMC: MixColumns.
pub fn aesmc(state: u128) -> u128 {
    mix(state, [2, 3, 1, 1])
}

/// AESIMC: InvMixColumns.
pub fn aesimc(state: u128) -> u128 {
    mix(state, [14, 11, 13, 9])
}

fn word(v: u128, i: u32) -> u32 {
    (v >> (32 * i)) as u32
}

fn with_word(v: u128, i: u32, w: u32) -> u128 {
    (v & !(u128::from(u32::MAX) << (32 * i))) | (u128::from(w) << (32 * i))
}

fn choose(x: u32, y: u32, z: u32) -> u32 {
    ((y ^ z) & x) ^ z
}

fn majority(x: u32, y: u32, z: u32) -> u32 {
    (x & y) | ((x | y) & z)
}

fn parity(x: u32, y: u32, z: u32) -> u32 {
    x ^ y ^ z
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sha1Fn {
    Choose,
    Parity,
    Majority,
}

/// SHA1C, SHA1P, SHA1M: four rounds on `abcd` (Qd) and `e` (Sn) with the
/// schedule words in `w`.
pub fn sha1_rounds(abcd: u128, e: u32, w: u128, f: Sha1Fn) -> u128 {
    let mut x = abcd;
    let mut y = e;
    for i in 0..4 {
        let (b, c, d) = (word(x, 1), word(x, 2), word(x, 3));
        let t = match f {
            Sha1Fn::Choose => choose(b, c, d),
            Sha1Fn::Parity => parity(b, c, d),
            Sha1Fn::Majority => majority(b, c, d),
        };
        y = y
            .wrapping_add(word(x, 0).rotate_left(5))
            .wrapping_add(t)
            .wrapping_add(word(w, i));
        x = with_word(x, 1, word(x, 1).rotate_left(30));
        // <Y, X> = ROL(Y : X, 32) on the 160-bit value.
        let top = word(x, 3);
        x = (x << 32) | u128::from(y);
        y = top;
    }
    x
}

/// SHA1H: fixed rotate.
pub fn sha1h(e: u32) -> u32 {
    e.rotate_left(30)
}

/// SHA1SU0.
pub fn sha1su0(d: u128, n: u128, m: u128) -> u128 {
    let result = ((n & u128::from(u64::MAX)) << 64) | (d >> 64);
    result ^ d ^ m
}

/// SHA1SU1.
pub fn sha1su1(d: u128, n: u128) -> u128 {
    let t = d ^ (n >> 32);
    let mut r = 0u128;
    for i in 0..4 {
        r = with_word(r, i, word(t, i).rotate_left(1));
    }
    with_word(r, 3, word(r, 3) ^ word(t, 0).rotate_left(2))
}

/// `SHA256hash()`: four rounds. Returns the first half (SHA256H) or the
/// second half (SHA256H2) of the updated state.
pub fn sha256_hash(x: u128, y: u128, w: u128, part1: bool) -> u128 {
    let (mut x, mut y) = (x, y);
    for i in 0..4 {
        let chs = choose(word(y, 0), word(y, 1), word(y, 2));
        let maj = majority(word(x, 0), word(x, 1), word(x, 2));
        let y0 = word(y, 0);
        let sigma1 = y0.rotate_right(6) ^ y0.rotate_right(11) ^ y0.rotate_right(25);
        let t1 = word(y, 3)
            .wrapping_add(sigma1)
            .wrapping_add(chs)
            .wrapping_add(word(w, i));
        x = with_word(x, 3, t1.wrapping_add(word(x, 3)));
        let x0 = word(x, 0);
        let sigma0 = x0.rotate_right(2) ^ x0.rotate_right(13) ^ x0.rotate_right(22);
        y = with_word(y, 3, t1.wrapping_add(sigma0).wrapping_add(maj));
        // <Y, X> = ROL(Y : X, 32) on the 256-bit value.
        let (x_top, y_top) = (word(x, 3), word(y, 3));
        x = (x << 32) | u128::from(y_top);
        y = (y << 32) | u128::from(x_top);
    }
    if part1 {
        x
    } else {
        y
    }
}

/// SHA256SU0.
pub fn sha256su0(d: u128, n: u128) -> u128 {
    let t = (u128::from(word(n, 0)) << 96) | (d >> 32);
    (0..4).fold(0u128, |r, i| {
        let e = word(t, i);
        let s = e.rotate_right(7) ^ e.rotate_right(18) ^ (e >> 3);
        with_word(r, i, s.wrapping_add(word(d, i)))
    })
}

/// SHA256SU1.
pub fn sha256su1(d: u128, n: u128, m: u128) -> u128 {
    let t0 = (u128::from(word(m, 0)) << 96) | (n >> 32);
    let sig = |e: u32| e.rotate_right(17) ^ e.rotate_right(19) ^ (e >> 10);
    let mut r = 0u128;
    let t1 = m >> 64;
    for i in 0..2 {
        let v = sig(word(t1, i))
            .wrapping_add(word(d, i))
            .wrapping_add(word(t0, i));
        r = with_word(r, i, v);
    }
    let t1 = r;
    for i in 2..4 {
        let v = sig(word(t1, i - 2))
            .wrapping_add(word(d, i))
            .wrapping_add(word(t0, i));
        r = with_word(r, i, v);
    }
    r
}

/// PMULL/PMULL2 (64-bit): carry-less 64 x 64 -> 128 multiplication.
pub fn clmul64(a: u64, b: u64) -> u128 {
    (0..64).fold(0u128, |acc, i| {
        if (b >> i) & 1 == 1 {
            acc ^ (u128::from(a) << i)
        } else {
            acc
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sbox_matches_fips197() {
        let (sbox, inv) = sboxes();
        assert_eq!(sbox[0x00], 0x63);
        assert_eq!(sbox[0x53], 0xED); // FIPS-197 section 5.1.1 example
        assert_eq!(sbox[0xFF], 0x16);
        assert_eq!(inv[0x63], 0x00);
    }

    #[test]
    fn mix_columns_matches_fips197() {
        // Column db 13 53 45 -> 8e 4d a1 bc (the standard MixColumns example).
        let col = u128::from(u32::from_le_bytes([0xDB, 0x13, 0x53, 0x45]));
        assert_eq!(
            aesmc(col) as u32,
            u32::from_le_bytes([0x8E, 0x4D, 0xA1, 0xBC])
        );
        assert_eq!(aesimc(aesmc(col)) as u32, col as u32);
    }

    #[test]
    fn clmul_known_value() {
        assert_eq!(clmul64(0b11, 0b11), 0b101);
        assert_eq!(clmul64(u64::MAX, 2), u128::from(u64::MAX) << 1);
    }
}
