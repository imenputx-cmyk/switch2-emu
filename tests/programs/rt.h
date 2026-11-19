/* Minimal freestanding runtime for test programs.
 *
 * This is NOT the Switch OS interface. It is a tiny test ABI understood by
 * emu-run: `svc #0` with the call number in x8.
 *   x8 = 1: write(x0 = buffer, x1 = length) to stdout
 *   x8 = 2: exit(x0 = status)
 */
#pragma once
typedef unsigned long u64;
typedef long i64;
typedef unsigned int u32;
typedef unsigned char u8;

static inline void sys_write(const char *buf, u64 len) {
    register u64 x0 __asm__("x0") = (u64)buf;
    register u64 x1 __asm__("x1") = len;
    register u64 x8 __asm__("x8") = 1;
    __asm__ volatile("svc #0" : "+r"(x0) : "r"(x1), "r"(x8) : "memory");
}

static inline __attribute__((noreturn)) void sys_exit(u64 status) {
    register u64 x0 __asm__("x0") = status;
    register u64 x8 __asm__("x8") = 2;
    __asm__ volatile("svc #0" : : "r"(x0), "r"(x8) : "memory");
    __builtin_unreachable();
}

/* The compiler may emit calls to these even in freestanding code. */
void *memset(void *d, int c, u64 n) {
    u8 *p = d;
    while (n--) *p++ = (u8)c;
    return d;
}

void *memcpy(void *d, const void *s, u64 n) {
    u8 *p = d;
    const u8 *q = s;
    while (n--) *p++ = *q++;
    return d;
}

static u64 str_len(const char *s) {
    u64 n = 0;
    while (s[n]) n++;
    return n;
}

static void print(const char *s) { sys_write(s, str_len(s)); }

static void print_u64(u64 v) {
    char buf[24];
    int i = 23;
    buf[i] = 0;
    do {
        buf[--i] = (char)('0' + v % 10);
        v /= 10;
    } while (v);
    print(&buf[i]);
}

static void print_i64(i64 v) {
    if (v < 0) {
        print("-");
        print_u64((u64)0 - (u64)v);
    } else {
        print_u64((u64)v);
    }
}

static void print_hex(u64 v) {
    static const char digits[] = "0123456789abcdef";
    char buf[19];
    buf[0] = '0';
    buf[1] = 'x';
    for (int i = 0; i < 16; i++) buf[2 + i] = digits[(v >> (60 - 4 * i)) & 0xf];
    buf[18] = 0;
    print(buf);
}

int main(void);

__attribute__((section(".text.start"))) void _start(void) { sys_exit((u64)main()); }
