// Minimal homebrew start code, written for this project following the
// homebrew ABI on switchbrew (https://switchbrew.org/wiki/Homebrew_ABI):
//   x0 = loader config entries, x1 = -1, x30 = return to the loader.
// The first 0x80 bytes are reserved: elf2nro.py writes the NRO header at
// 0x10. Offset 4 holds the MOD0 offset, as in other homebrew.

    .section .crt0, "ax", %progbits
    .global _start
_start:
    b       entry
    .word   mod0 - _start
    .ascii  "HOMEBREW"
    .org    _start + 0x80

entry:
    mov     x19, x0                 // config entries
    mov     x20, x1                 // -1 for NRO
    mov     x21, x30                // loader return address

    // Apply our own R_AARCH64_RELATIVE relocations (we are PIE).
    adr     x0, _start
    adrp    x1, _DYNAMIC
    add     x1, x1, :lo12:_DYNAMIC
    bl      hb_relocate

    mov     x0, x19
    mov     x1, x20
    bl      hb_main

    // Return to the loader with the result in x0.
    mov     x30, x21
    ret

    .balign 4
mod0:
    .ascii  "MOD0"
    .word   _DYNAMIC - mod0
    .word   __bss_start__ - mod0
    .word   __bss_end__ - mod0
    .word   0, 0                    // eh_frame_hdr start and end (unused)
    .word   0                       // runtime module object (unused)
