// Start code for hb_libnx: like ../homebrew/crt0.S, but also passes the
// loader return address to hb_main (libnx's envSetup wants it).
    .section .crt0, "ax", %progbits
    .global _start
_start:
    b       entry
    .word   mod0 - _start
    .ascii  "HOMEBREW"
    .org    _start + 0x80
entry:
    mov     x19, x0
    mov     x20, x1
    mov     x21, x30
    mov     x22, sp
    adr     x0, _start
    adrp    x1, _DYNAMIC
    add     x1, x1, :lo12:_DYNAMIC
    bl      hb_relocate
    mov     x0, x19
    mov     x1, x20
    mov     x2, x21
    bl      hb_main
    mov     sp, x22
    mov     x30, x21
    ret
    .balign 4
mod0:
    .ascii  "MOD0"
    .word   _DYNAMIC - mod0
    .word   __bss_start__ - mod0
    .word   __bss_end__ - mod0
    .word   0, 0, 0

// __nx_exit(Result rc, LoaderReturnFn ret): return to the loader.
    .text
    .global __nx_exit
__nx_exit:
    br      x1
