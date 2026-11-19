/* First homebrew test program: boots through the homebrew ABI and
 * exercises the basic Horizon SVCs. Written for this project; SVC numbers
 * and argument registers follow https://switchbrew.org/wiki/SVC. */
typedef unsigned long u64;
typedef unsigned int u32;
typedef unsigned char u8;
typedef long i64;

/* ------------------------------------------------------------------ */
/* Self-relocation: apply R_AARCH64_RELATIVE entries from .dynamic.    */
/* Runs before relocations, so it must not use global pointers.        */

typedef struct { i64 tag; u64 val; } Dyn;
typedef struct { u64 offset; u64 info; i64 addend; } Rela;

void hb_relocate(u64 base, const Dyn *dyn) {
    const Rela *rela = 0;
    u64 relasz = 0;
    for (; dyn->tag != 0; dyn++) {
        if (dyn->tag == 7) rela = (const Rela *)(base + dyn->val);   /* DT_RELA */
        if (dyn->tag == 8) relasz = dyn->val;                       /* DT_RELASZ */
    }
    for (u64 i = 0; i < relasz / sizeof(Rela); i++)
        if ((u32)rela[i].info == 1027) /* R_AARCH64_RELATIVE */
            *(u64 *)(base + rela[i].offset) = base + rela[i].addend;
}

/* ------------------------------------------------------------------ */
/* SVC wrappers                                                        */

#define SVC(n) __asm__ volatile("svc " #n : "+r"(x0), "+r"(x1), "+r"(x2), "+r"(x3) : : "memory")

static u32 svc_output(const char *s, u64 len) {
    register u64 x0 __asm__("x0") = (u64)s, x1 __asm__("x1") = len, x2 __asm__("x2"), x3 __asm__("x3");
    SVC(0x27);
    return (u32)x0;
}

static u32 svc_set_heap_size(u64 *addr, u64 size) {
    register u64 x0 __asm__("x0"), x1 __asm__("x1") = size, x2 __asm__("x2"), x3 __asm__("x3");
    SVC(0x01);
    *addr = x1;
    return (u32)x0;
}

typedef struct {
    u64 base, size;
    u32 type, attr, perm, ipc_refcount, device_refcount, padding;
} MemoryInfo;

static u32 svc_query_memory(MemoryInfo *info, u64 addr) {
    register u64 x0 __asm__("x0") = (u64)info, x1 __asm__("x1"), x2 __asm__("x2") = addr, x3 __asm__("x3");
    SVC(0x06);
    return (u32)x0;
}

static u32 svc_get_info(u64 *out, u32 type, u32 handle, u64 sub) {
    register u64 x0 __asm__("x0"), x1 __asm__("x1") = type, x2 __asm__("x2") = handle, x3 __asm__("x3") = sub;
    SVC(0x29);
    *out = x1;
    return (u32)x0;
}

static u32 svc_map_memory(u64 dst, u64 src, u64 size, u32 id) {
    register u64 x0 __asm__("x0") = dst, x1 __asm__("x1") = src, x2 __asm__("x2") = size, x3 __asm__("x3");
    if (id == 4) SVC(0x04); else SVC(0x05);
    return (u32)x0;
}

static u64 svc_get_system_tick(void) {
    register u64 x0 __asm__("x0"), x1 __asm__("x1"), x2 __asm__("x2"), x3 __asm__("x3");
    SVC(0x1E);
    return x0;
}

#define CUR_PROCESS 0xFFFF8001u

/* ------------------------------------------------------------------ */
/* Tiny formatting                                                     */

static char out[2048];
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

static void flush(void) {
    svc_output(out, out_len);
    out_len = 0;
}

static void line(const char *label, u64 v) { put(label); put_hex(v); put("\n"); flush(); }

/* A pointer table in data: only correct after relocation. */
static const char *const greetings[] = {"relocated ", "pointers ", "work\n"};

typedef struct { u32 key, flags; u64 value[2]; } ConfigEntry;

int hb_main(const ConfigEntry *cfg, u64 x1) {
    put("Hello from homebrew!\n");
    flush();
    for (int i = 0; i < 3; i++) put(greetings[i]);
    flush();

    line("entry x1 = ", x1);
    for (const ConfigEntry *e = cfg; ; e++) {
        if (e->key == 1) line("config MainThreadHandle = ", e->value[0]);
        if (e->key == 7) line("config AppletType = ", e->value[0]);
        if (e->key == 16) line("config HosVersion = ", e->value[0]);
        if (e->key == 0) {
            put("loader: ");
            const char *s = (const char *)e->value[0];
            for (u64 i = 0; i < e->value[1]; i++) out[out_len++] = s[i];
            put("\n");
            flush();
            break;
        }
    }

    u64 heap_region, heap_region_size, aslr, stack_region, v;
    svc_get_info(&heap_region, 4, CUR_PROCESS, 0);
    svc_get_info(&heap_region_size, 5, CUR_PROCESS, 0);
    svc_get_info(&aslr, 12, CUR_PROCESS, 0);
    svc_get_info(&stack_region, 14, CUR_PROCESS, 0);
    line("heap region = ", heap_region);
    line("heap region size = ", heap_region_size);
    line("aslr region = ", aslr);
    line("GetInfo(bad type) = ", svc_get_info(&v, 999, CUR_PROCESS, 0));
    line("GetInfo(bad handle) = ", svc_get_info(&v, 4, 0x1234, 0));

    u64 heap;
    line("SetHeapSize(0x123) = ", svc_set_heap_size(&heap, 0x123));
    line("SetHeapSize(4 MiB) = ", svc_set_heap_size(&heap, 0x400000));
    line("heap = ", heap);
    u64 *words = (u64 *)heap;
    u64 sum = 0;
    for (u64 i = 0; i < 0x400000 / 8; i += 512) words[i] = i * 3;
    for (u64 i = 0; i < 0x400000 / 8; i += 512) sum += words[i];
    line("heap checksum = ", sum);

    MemoryInfo mi;
    svc_query_memory(&mi, (u64)&hb_main);
    put("code:  type "); put_hex(mi.type); put(" perm "); put_hex(mi.perm); put("\n"); flush();
    svc_query_memory(&mi, (u64)&mi);
    put("stack: type "); put_hex(mi.type); put(" perm "); put_hex(mi.perm);
    put(" size "); put_hex(mi.size); put("\n"); flush();
    svc_query_memory(&mi, heap + 0x1000);
    put("heap:  type "); put_hex(mi.type); put(" size "); put_hex(mi.size); put("\n"); flush();

    /* Alias one heap page into the stack region, the way libnx makes
     * thread stacks, and check that writes go through. */
    u64 alias = stack_region + 0x10000000;
    line("MapMemory = ", svc_map_memory(alias, heap, 0x1000, 4));
    svc_query_memory(&mi, heap);
    put("source while mapped: perm "); put_hex(mi.perm); put(" attr "); put_hex(mi.attr); put("\n"); flush();
    *(volatile u64 *)alias = 0x1234567890abcdefull;
    line("UnmapMemory = ", svc_map_memory(alias, heap, 0x1000, 5));
    line("value seen through heap = ", *(volatile u64 *)heap);

    u64 t0 = svc_get_system_tick();
    u64 t1 = svc_get_system_tick();
    line("tick advanced = ", t1 > t0);

    put("Returning 42 to the loader\n");
    flush();
    return 42;
}
