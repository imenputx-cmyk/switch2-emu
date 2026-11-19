# switch2-emu

A learning-first, from-scratch emulator project for the Nintendo Switch 2 hardware architecture, written in Rust.

This project does not include, distribute, or require any copyrighted Nintendo firmware, keys, or games. The goal is the engineering of the core components: CPU, memory, GPU, and the OS service layer.

## Roadmap

1. **Foundation:** Cargo workspace, CI, and test setup. *(in progress)*
2. **Guest CPU state:** AArch64 register file, with tests. *(done: `emu-cpu`)*
3. **Decoder and interpreter:** Decode A64 instructions against the ARM Architecture Reference Manual, one instruction class at a time, with unit tests for each.
4. **Memory and loader:** Guest virtual memory map and loading of homebrew/test ROMs.
5. **Service layer (HLE):** Simple emulation of OS calls.
6. **JIT:** Dynamic translation from AArch64 to x86-64 after the interpreter is correct.
7. **GPU:** Parse the command stream and translate it to Vulkan.
8. **Audio, input, and performance.**

Accuracy comes before speed. The interpreter is the reference that the JIT and GPU work will be tested against.

## Layout

```
crates/
  emu-cpu/   AArch64 guest CPU state (decoder and interpreter to come)
```

## Build and test

```sh
cargo test
```