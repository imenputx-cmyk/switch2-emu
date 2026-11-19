#!/usr/bin/env python3
"""Converts a position-independent AArch64 ELF (linked at 0 with
text/ro/data segments) into an NRO, following the format documented at
https://switchbrew.org/wiki/NRO.

    tools/elf2nro.py input.elf output.nro
"""
import struct
import sys

PAGE = 0x1000


def align(x, a=PAGE):
    return (x + a - 1) // a * a


def main(src, dst):
    elf = open(src, "rb").read()
    assert elf[:4] == b"\x7fELF" and elf[4] == 2, "64-bit ELF required"
    phoff = struct.unpack_from("<Q", elf, 32)[0]
    phentsize, phnum = struct.unpack_from("<HH", elf, 54)
    loads = []
    for i in range(phnum):
        p_type, p_flags, off, vaddr, _, filesz, memsz, _ = struct.unpack_from(
            "<IIQQQQQQ", elf, phoff + i * phentsize)
        if p_type == 1:
            loads.append((vaddr, off, filesz, memsz, p_flags))
    loads.sort()
    assert len(loads) == 3, f"expected text, ro and data segments, got {len(loads)}"
    (t, ro, data) = loads
    assert t[0] == 0 and all(seg[0] % PAGE == 0 for seg in loads), "segments must be page aligned"

    data_size = align(data[2])
    image_end = align(data[0] + data[3])
    file_size = data[0] + data_size
    image = bytearray(file_size)
    for vaddr, off, filesz, _, _ in loads:
        image[vaddr:vaddr + filesz] = elf[off:off + filesz]

    text_size = align(t[2])
    ro_size = align(ro[2])
    bss_size = image_end - (data[0] + data_size)
    header = struct.pack(
        "<4sIII" + "II" * 3 + "I" + "I",
        b"NRO0", 0, file_size, 0,
        0, text_size, ro[0], ro_size, data[0], data_size,
        bss_size, 0)
    image[0x10:0x10 + len(header)] = header
    # ModuleId (0x40, 32 bytes) and the remaining fields stay zero.
    open(dst, "wb").write(image)
    print(f"{dst}: text {text_size:#x}, ro {ro_size:#x}, data {data_size:#x}, bss {bss_size:#x}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
