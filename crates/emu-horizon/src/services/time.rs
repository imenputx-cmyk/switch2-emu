//! `time:u` / `time:a` / `time:s`: clocks. Since 6.0.0 clients read the
//! clocks from a shared memory page instead of calling the clock
//! interfaces; its layout (from libnx's `time.c`):
//!
//! | offset | content                                                    |
//! |--------|------------------------------------------------------------|
//! | 0x00   | standard steady clock: base time (ns) + clock source id    |
//! | 0x38   | local system clock context: offset (s) + steady time point |
//! | 0x80   | network system clock context                               |
//! | 0xC8   | automatic correction enabled (bool)                        |
//!
//! Each object is stored twice after a u32 counter; readers use copy
//! `counter & 1` and retry if the counter changed.
//!
//! Steady time is `(base + ticks_to_ns(tick)) / 1e9` seconds and the
//! system time is that plus the context offset, so the offset is the
//! POSIX time at boot (tick 0).

use super::{obj, CallError, Ctx, Service};
use crate::handles::KObject;

const SHARED_MEMORY_SIZE: u64 = 0x1000;
/// Clock source id shared by the steady clock and the contexts that are
/// based on it.
const SOURCE_ID: [u8; 16] = *b"switch2-emu-time";
const TICK_HZ: u64 = 19_200_000;

pub struct StaticService {
    /// POSIX time when the emulated system booted.
    boot_time: i64,
    shared_memory: Option<usize>,
}

impl Default for StaticService {
    fn default() -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs() as i64);
        StaticService::with_boot_time(now)
    }
}

impl StaticService {
    pub fn with_boot_time(boot_time: i64) -> Self {
        StaticService {
            boot_time,
            shared_memory: None,
        }
    }

    fn shared_memory_contents(&self) -> Vec<u8> {
        let mut m = vec![0u8; SHARED_MEMORY_SIZE as usize];
        // Steady clock: base time 0, both copies.
        for copy in 0..2 {
            let at = 0x08 + copy * 0x18;
            m[at + 8..at + 24].copy_from_slice(&SOURCE_ID);
        }
        // Local and network clock contexts.
        for base in [0x38, 0x80] {
            for copy in 0..2 {
                let at = base + 8 + copy * 0x20;
                m[at..at + 8].copy_from_slice(&self.boot_time.to_le_bytes());
                m[at + 16..at + 32].copy_from_slice(&SOURCE_ID);
            }
        }
        m[0xC8] = 1;
        m
    }
}

fn steady_seconds(tick: u64) -> i64 {
    (tick / TICK_HZ) as i64
}

impl Service for StaticService {
    fn name(&self) -> &str {
        "time:u"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        let boot = self.boot_time;
        match command {
            0 | 1 | 4 => ctx.push_object(obj(SystemClock { boot_time: boot })),
            2 => ctx.push_object(obj(SteadyClock)),
            3 => ctx.push_object(obj(TimeZoneService)),
            20 => {
                // GetSharedMemoryNativeHandle
                let id = match self.shared_memory {
                    Some(id) => id,
                    None => {
                        let init = self.shared_memory_contents();
                        let id = ctx.kernel.new_shared_memory(SHARED_MEMORY_SIZE, init);
                        self.shared_memory = Some(id);
                        id
                    }
                };
                ctx.push_copy(KObject::SharedMemory { id });
            }
            100 => ctx.push_u8(1), // IsStandardUserSystemClockAutomaticCorrectionEnabled
            _ => return Err(CallError::Unimplemented),
        }
        Ok(())
    }
}

pub struct SystemClock {
    boot_time: i64,
}

impl Service for SystemClock {
    fn name(&self) -> &str {
        "time::ISystemClock"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            0 => ctx.push_u64((self.boot_time + steady_seconds(ctx.tick)) as u64),
            2 => {
                // GetSystemClockContext
                ctx.push_u64(self.boot_time as u64);
                ctx.push_u64(steady_seconds(ctx.tick) as u64);
                ctx.push_bytes(&SOURCE_ID);
            }
            _ => return Err(CallError::Unimplemented),
        }
        Ok(())
    }
}

pub struct SteadyClock;

impl Service for SteadyClock {
    fn name(&self) -> &str {
        "time::ISteadyClock"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            0 => {
                // GetCurrentTimePoint
                ctx.push_u64(steady_seconds(ctx.tick) as u64);
                ctx.push_bytes(&SOURCE_ID);
            }
            _ => return Err(CallError::Unimplemented),
        }
        Ok(())
    }
}

pub struct TimeZoneService;

impl Service for TimeZoneService {
    fn name(&self) -> &str {
        "time::ITimeZoneService"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            0 => {
                // GetDeviceLocationName: 0x24 bytes.
                let mut name = [0u8; 0x24];
                name[..3].copy_from_slice(b"UTC");
                ctx.push_bytes(&name);
            }
            _ => return Err(CallError::Unimplemented),
        }
        Ok(())
    }
}
