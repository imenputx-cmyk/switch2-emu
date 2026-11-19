/* Minimal C runtime pieces that libnx's service code expects from newlib,
 * written for this project: string functions and a bump allocator on the
 * heap. Enough to run libnx's service clients, not a real libc. */
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

void *memcpy(void *d, const void *s, size_t n) {
    unsigned char *p = d; const unsigned char *q = s;
    while (n--) *p++ = *q++;
    return d;
}
void *memmove(void *d, const void *s, size_t n) {
    unsigned char *p = d; const unsigned char *q = s;
    if (p < q) while (n--) *p++ = *q++;
    else { p += n; q += n; while (n--) *--p = *--q; }
    return d;
}
void *memset(void *d, int c, size_t n) {
    unsigned char *p = d;
    while (n--) *p++ = (unsigned char)c;
    return d;
}
int memcmp(const void *a, const void *b, size_t n) {
    const unsigned char *p = a, *q = b;
    for (; n; n--, p++, q++) if (*p != *q) return *p - *q;
    return 0;
}
size_t strlen(const char *s) { size_t n = 0; while (s[n]) n++; return n; }
size_t strnlen(const char *s, size_t m) { size_t n = 0; while (n < m && s[n]) n++; return n; }
int strcmp(const char *a, const char *b) { while (*a && *a == *b) a++, b++; return (unsigned char)*a - (unsigned char)*b; }
int strncmp(const char *a, const char *b, size_t n) {
    for (; n; n--, a++, b++) { if (*a != *b) return (unsigned char)*a - (unsigned char)*b; if (!*a) return 0; }
    return 0;
}
char *strncpy(char *d, const char *s, size_t n) {
    size_t i = 0;
    for (; i < n && s[i]; i++) d[i] = s[i];
    for (; i < n; i++) d[i] = 0;
    return d;
}
char *strcpy(char *d, const char *s) { char *r = d; while ((*d++ = *s++)); return r; }
char *strchr(const char *s, int c) { for (;; s++) { if (*s == (char)c) return (char *)s; if (!*s) return 0; } }

/* Bump allocator over a heap region set up by hb_libnx.c. */
static uintptr_t heap_cur, heap_end;
void glue_set_heap(void *base, size_t size) { heap_cur = (uintptr_t)base; heap_end = heap_cur + size; }
void *memalign(size_t align, size_t n) {
    uintptr_t p = (heap_cur + align - 1) & ~(uintptr_t)(align - 1);
    if (p + n > heap_end) return 0;
    heap_cur = p + n;
    return (void *)p;
}
void *aligned_alloc(size_t a, size_t n) { return memalign(a, n); }
void *malloc(size_t n) { return memalign(16, n); }
void *calloc(size_t n, size_t s) { void *p = malloc(n * s); if (p) memset(p, 0, n * s); return p; }
void free(void *p) { (void)p; }

/* libnx runtime pieces normally provided by its crt0 and diag code. */
typedef uint32_t Result;
void svcBreak(uint32_t reason, uintptr_t address, uintptr_t size);
__attribute__((noreturn)) void diagAbortWithResult(Result res) {
    /* Report the result through svcBreak, like libnx's diag code does. */
    svcBreak(0, (uintptr_t)&res, sizeof(res));
    for (;;) {}
}

/* Static TLS for __thread variables (AArch64 "variant 1": TPIDR_EL0 points
 * to a 16-byte thread control block, the TLS image follows). libnx sets
 * this up in its thread code; this test does it once for the main thread. */
extern char __tdata_start[], __tdata_end[], __tls_size[];
static _Alignas(64) unsigned char tls_block[16 + 0x400];
void glue_setup_tls(void) {
    size_t data = (size_t)(__tdata_end - __tdata_start);
    memcpy(tls_block + 16, __tdata_start, data);
    __asm__ volatile("msr tpidr_el0, %0" : : "r"(tls_block));
}
