/* Self-relocation for position-independent homebrew: applies the
 * R_AARCH64_RELATIVE entries found through .dynamic. Runs before any
 * relocation is applied, so it must not use global pointers. */
typedef unsigned long u64;
typedef unsigned int u32;
typedef long i64;

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

