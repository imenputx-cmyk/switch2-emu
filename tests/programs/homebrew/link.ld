/* Position-independent homebrew image: text, read-only data and data in
   separate page-aligned segments starting at 0, like NROs expect. */
ENTRY(_start)
PHDRS {
    text PT_LOAD FLAGS(5);
    ro   PT_LOAD FLAGS(4);
    data PT_LOAD FLAGS(6);
    dyn  PT_DYNAMIC;
}
SECTIONS {
    . = 0;
    .text : { KEEP(*(.crt0)) *(.text .text.*) } :text
    . = ALIGN(0x1000);
    .rodata : { *(.rodata .rodata.*) } :ro
    .rela.dyn : { *(.rela.dyn .rela.*) } :ro
    .dynsym : { *(.dynsym) } :ro
    .dynstr : { *(.dynstr) } :ro
    .hash : { *(.hash) } :ro
    .gnu.hash : { *(.gnu.hash) } :ro
    . = ALIGN(0x1000);
    .data : { *(.data .data.*) } :data
    .data.rel.ro : { *(.data.rel.ro .data.rel.ro.*) } :data
    .dynamic : { *(.dynamic) } :data :dyn
    .got : { *(.got .got.plt) } :data
    __bss_start__ = .;
    .bss : { *(.bss .bss.*) *(COMMON) } :data
    __bss_end__ = .;
    /DISCARD/ : { *(.comment) *(.note*) *(.eh_frame*) *(.interp) }
}
