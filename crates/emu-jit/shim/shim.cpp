// SPDX-License-Identifier: GPL-3.0-or-later
//
// A thin C interface over dynarmic's A64 JIT, so Rust can drive it.
// The guest state lives in `EmuJitState`; Rust supplies memory, SVC and
// exception handling through `EmuJitCallbacks`. The callback design follows
// yuzu's `ARM_Dynarmic_64` (GPL-2.0-or-later), simplified for one core.

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>

#include <dynarmic/interface/A64/a64.h>
#include <dynarmic/interface/A64/config.h>
#include <dynarmic/interface/exclusive_monitor.h>

extern "C" {

struct EmuJitState {
    std::uint64_t x[31];
    std::uint64_t sp;
    std::uint64_t pc;
    std::uint64_t v[64];
    std::uint32_t pstate;
    std::uint32_t fpcr;
    std::uint32_t fpsr;
    std::uint32_t pad;
    std::uint64_t tpidr_el0;
    std::uint64_t tpidrro_el0;
};

// Every callback that returns bool returns true when execution must stop.
struct EmuJitCallbacks {
    void* ctx;
    bool (*read)(void* ctx, std::uint64_t addr, std::uint8_t* out, std::size_t len);
    bool (*write)(void* ctx, std::uint64_t addr, const std::uint8_t* data, std::size_t len);
    bool (*fetch)(void* ctx, std::uint64_t addr, std::uint32_t* out);
    void (*svc)(void* ctx, std::uint32_t imm);
    // Kinds: 0 breakpoint, 1 no-execute fault, 2 undefined instruction.
    void (*exception)(void* ctx, std::uint64_t pc, std::uint32_t kind);
    // Runs `count` instructions starting at `pc` in the interpreter.
    bool (*interpret)(void* ctx, EmuJitState* state, std::uint64_t pc, std::size_t count);
    void (*add_ticks)(void* ctx, std::uint64_t ticks);
    std::uint64_t (*ticks_remaining)(void* ctx);
    std::uint64_t (*cntpct)(void* ctx);
};

}  // extern "C"

namespace {

constexpr auto HaltCallback = Dynarmic::HaltReason::UserDefined1;

struct EmuJit;

struct Callbacks final : Dynarmic::A64::UserCallbacks {
    EmuJit* parent;
    EmuJitCallbacks cb;

    template<typename T>
    T Read(std::uint64_t addr);
    template<typename T>
    void Write(std::uint64_t addr, T value);
    template<typename T>
    bool WriteExclusive(std::uint64_t addr, T value, T expected);

    std::optional<std::uint32_t> MemoryReadCode(std::uint64_t addr) override {
        std::uint32_t insn = 0;
        if (cb.fetch(cb.ctx, addr, &insn)) {
            return std::nullopt;
        }
        return insn;
    }

    std::uint8_t MemoryRead8(std::uint64_t a) override { return Read<std::uint8_t>(a); }
    std::uint16_t MemoryRead16(std::uint64_t a) override { return Read<std::uint16_t>(a); }
    std::uint32_t MemoryRead32(std::uint64_t a) override { return Read<std::uint32_t>(a); }
    std::uint64_t MemoryRead64(std::uint64_t a) override { return Read<std::uint64_t>(a); }
    Dynarmic::A64::Vector MemoryRead128(std::uint64_t a) override {
        return Read<Dynarmic::A64::Vector>(a);
    }
    void MemoryWrite8(std::uint64_t a, std::uint8_t v) override { Write(a, v); }
    void MemoryWrite16(std::uint64_t a, std::uint16_t v) override { Write(a, v); }
    void MemoryWrite32(std::uint64_t a, std::uint32_t v) override { Write(a, v); }
    void MemoryWrite64(std::uint64_t a, std::uint64_t v) override { Write(a, v); }
    void MemoryWrite128(std::uint64_t a, Dynarmic::A64::Vector v) override { Write(a, v); }

    bool MemoryWriteExclusive8(std::uint64_t a, std::uint8_t v, std::uint8_t e) override {
        return WriteExclusive(a, v, e);
    }
    bool MemoryWriteExclusive16(std::uint64_t a, std::uint16_t v, std::uint16_t e) override {
        return WriteExclusive(a, v, e);
    }
    bool MemoryWriteExclusive32(std::uint64_t a, std::uint32_t v, std::uint32_t e) override {
        return WriteExclusive(a, v, e);
    }
    bool MemoryWriteExclusive64(std::uint64_t a, std::uint64_t v, std::uint64_t e) override {
        return WriteExclusive(a, v, e);
    }
    bool MemoryWriteExclusive128(std::uint64_t a, Dynarmic::A64::Vector v,
                                 Dynarmic::A64::Vector e) override {
        return WriteExclusive(a, v, e);
    }

    void InterpreterFallback(std::uint64_t pc, std::size_t count) override;
    void CallSVC(std::uint32_t imm) override;
    void ExceptionRaised(std::uint64_t pc, Dynarmic::A64::Exception exception) override;

    void AddTicks(std::uint64_t ticks) override { cb.add_ticks(cb.ctx, ticks); }
    std::uint64_t GetTicksRemaining() override { return cb.ticks_remaining(cb.ctx); }
    std::uint64_t GetCNTPCT() override { return cb.cntpct(cb.ctx); }
};

struct EmuJit {
    Callbacks callbacks;
    Dynarmic::ExclusiveMonitor monitor{1};
    std::uint64_t tpidr_el0 = 0;
    std::uint64_t tpidrro_el0 = 0;
    std::unique_ptr<Dynarmic::A64::Jit> jit;

    void Halt() { jit->HaltExecution(HaltCallback); }

    void Load(const EmuJitState& s) {
        std::array<std::uint64_t, 31> x;
        std::memcpy(x.data(), s.x, sizeof(s.x));
        jit->SetRegisters(x);
        jit->SetSP(s.sp);
        jit->SetPC(s.pc);
        std::array<Dynarmic::A64::Vector, 32> v;
        std::memcpy(v.data(), s.v, sizeof(s.v));
        jit->SetVectors(v);
        jit->SetPstate(s.pstate);
        jit->SetFpcr(s.fpcr);
        jit->SetFpsr(s.fpsr);
        tpidr_el0 = s.tpidr_el0;
        tpidrro_el0 = s.tpidrro_el0;
    }

    void Save(EmuJitState& s) const {
        const auto x = jit->GetRegisters();
        std::memcpy(s.x, x.data(), sizeof(s.x));
        s.sp = jit->GetSP();
        s.pc = jit->GetPC();
        const auto v = jit->GetVectors();
        std::memcpy(s.v, v.data(), sizeof(s.v));
        s.pstate = jit->GetPstate();
        s.fpcr = jit->GetFpcr();
        s.fpsr = jit->GetFpsr();
        s.tpidr_el0 = tpidr_el0;
        s.tpidrro_el0 = tpidrro_el0;
    }

    // Hands `count` instructions at `pc` to the Rust interpreter.
    void Interpret(std::uint64_t pc, std::size_t count) {
        EmuJitState s{};
        Save(s);
        s.pc = pc;
        const bool stop = callbacks.cb.interpret(callbacks.cb.ctx, &s, pc, count);
        Load(s);
        if (stop) {
            Halt();
        }
    }
};

template<typename T>
T Callbacks::Read(std::uint64_t addr) {
    T value{};
    if (cb.read(cb.ctx, addr, reinterpret_cast<std::uint8_t*>(&value), sizeof(T))) {
        parent->Halt();
    }
    return value;
}

template<typename T>
void Callbacks::Write(std::uint64_t addr, T value) {
    if (cb.write(cb.ctx, addr, reinterpret_cast<const std::uint8_t*>(&value), sizeof(T))) {
        parent->Halt();
    }
}

template<typename T>
bool Callbacks::WriteExclusive(std::uint64_t addr, T value, T expected) {
    T current{};
    if (cb.read(cb.ctx, addr, reinterpret_cast<std::uint8_t*>(&current), sizeof(T))) {
        parent->Halt();
        return false;
    }
    if (std::memcmp(&current, &expected, sizeof(T)) != 0) {
        return false;
    }
    Write(addr, value);
    return true;
}

void Callbacks::InterpreterFallback(std::uint64_t pc, std::size_t count) {
    parent->Interpret(pc, count);
}

void Callbacks::CallSVC(std::uint32_t imm) {
    cb.svc(cb.ctx, imm);
    parent->Halt();
}

void Callbacks::ExceptionRaised(std::uint64_t pc, Dynarmic::A64::Exception exception) {
    using E = Dynarmic::A64::Exception;
    switch (exception) {
    case E::WaitForInterrupt:
    case E::WaitForEvent:
    case E::SendEvent:
    case E::SendEventLocal:
    case E::Yield:
        // Hints: nothing to do on a single host thread.
        parent->jit->SetPC(pc + 4);
        return;
    case E::Breakpoint:
        cb.exception(cb.ctx, pc, 0);
        break;
    case E::NoExecuteFault:
        cb.exception(cb.ctx, pc, 1);
        break;
    default:
        // Let our own interpreter try; it may know the instruction (or
        // reports it as unimplemented).
        parent->Interpret(pc, 1);
        return;
    }
    parent->jit->SetPC(pc);
    parent->Halt();
}

}  // namespace

extern "C" {

EmuJit* emu_jit_new(const EmuJitCallbacks* cb) {
    auto* e = new EmuJit;
    e->callbacks.parent = e;
    e->callbacks.cb = *cb;

    Dynarmic::A64::UserConfig config;
    config.callbacks = &e->callbacks;
    config.processor_id = 0;
    config.global_monitor = &e->monitor;
    config.tpidr_el0 = &e->tpidr_el0;
    config.tpidrro_el0 = &e->tpidrro_el0;
    config.cntfrq_el0 = 19'200'000;
    config.dczid_el0 = 4;
    config.ctr_el0 = 0x8444c004;
    // Our memory is reached through callbacks (no page table yet), so a
    // failed access stops the block right after it.
    config.check_halt_on_memory_access = true;
    config.enable_cycle_counting = true;
    config.code_cache_size = 128 * 1024 * 1024;
    config.define_unpredictable_behaviour = true;
    e->jit = std::make_unique<Dynarmic::A64::Jit>(config);
    return e;
}

void emu_jit_free(EmuJit* e) {
    delete e;
}

// Runs from `state` until a callback stops execution or the tick budget is
// spent; writes the final state back. Returns dynarmic's halt reason bits.
std::uint32_t emu_jit_run(EmuJit* e, EmuJitState* state) {
    e->Load(*state);
    e->jit->ClearExclusiveState();
    const auto reason = e->jit->Run();
    e->Save(*state);
    return static_cast<std::uint32_t>(reason);
}

void emu_jit_invalidate(EmuJit* e, std::uint64_t addr, std::uint64_t size) {
    e->jit->InvalidateCacheRange(addr, static_cast<std::size_t>(size));
}

void emu_jit_clear_cache(EmuJit* e) {
    e->jit->ClearCache();
}

}  // extern "C"
