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
            1 => {}               // SetDeviceLocationName
            2 => ctx.push_u32(1), // GetTotalLocationNameCount
            // LoadLocationNameList: "UTC".
            3 => {
                if let Some(b) = ctx.out_buffer(0) {
                    let mut name = vec![0u8; b.size as usize];
                    let n = name.len().min(3);
                    name[..n].copy_from_slice(&b"UTC"[..n]);
                    ctx.write_out_buffer(0, &name)?;
                }
                ctx.push_u32(1);
            }
            // LoadTimeZoneRule: an empty rule (UTC).
            4 => {
                if let Some(b) = ctx.out_buffer(0) {
                    ctx.write_out_buffer(0, &vec![0; b.size as usize])?;
                }
            }
            // GetTimeZoneRuleVersion: 16 bytes.
            5 => ctx.push_bytes(b"2024a\0\0\0\0\0\0\0\0\0\0\0"),
            // GetDeviceLocationNameAndUpdatedTime: the name, then a
            // steady clock point (0x18 bytes).
            6 => {
                let mut name = [0u8; 0x24];
                name[..3].copy_from_slice(b"UTC");
                ctx.push_bytes(&name);
                ctx.push_bytes(&[0; 0x1C]);
            }
            // ToCalendarTime(WithMyRule)(posix time): UTC.
            100 | 101 => {
                let t = ctx.in_u64(0) as i64;
                ctx.push_bytes(&calendar(t));
            }
            // ToPosixTime(WithMyRule)(calendar time): one time, UTC.
            201 | 202 => {
                let c = ctx.in_bytes(0, 8);
                let year = i64::from(u16::from_le_bytes([c[0], c[1]]) as i16);
                let days = days_from_civil(year, u32::from(c[2]), u32::from(c[3]));
                let t =
                    days * 86_400 + i64::from(c[4]) * 3600 + i64::from(c[5]) * 60 + i64::from(c[6]);
                if ctx.out_buffer(0).is_some() {
                    ctx.write_out_buffer(0, &t.to_le_bytes())?;
                }
                ctx.push_u32(1);
            }
            _ => return Err(CallError::Unimplemented),
        }
        Ok(())
    }
}

/// Days since 1970-01-01 of a civil date (Howard Hinnant's algorithm).
pub fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let m = i64::from(m);
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + i64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// A posix time as a UTC CalendarTime (8 bytes: year, month, day, hour,
/// minute, second) and CalendarAdditionalInfo (day of week, day of year,
/// "UTC", not DST, offset 0).
pub fn calendar(t: i64) -> Vec<u8> {
    let days = t.div_euclid(86_400);
    let secs = t.rem_euclid(86_400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    let mut out = Vec::with_capacity(0x20);
    out.extend_from_slice(&(y as u16).to_le_bytes());
    out.extend_from_slice(&[
        m as u8,
        d as u8,
        (secs / 3600) as u8,
        (secs / 60 % 60) as u8,
        (secs % 60) as u8,
        0,
    ]);
    let weekday = (days + 4).rem_euclid(7) as u32;
    let yday = (days - days_from_civil(y, 1, 1)) as u32;
    out.extend_from_slice(&weekday.to_le_bytes());
    out.extend_from_slice(&yday.to_le_bytes());
    let mut name = [0u8; 8];
    name[..3].copy_from_slice(b"UTC");
    out.extend_from_slice(&name);
    out.extend_from_slice(&0u32.to_le_bytes()); // is DST
    out.extend_from_slice(&0i32.to_le_bytes()); // UTC offset
    out
}

#[cfg(test)]
mod calendar_tests {
    use super::*;

    #[test]
    fn utc_calendar() {
        // 2026-11-21 02:28:05 UTC is a Saturday, day 324 of the year.
        let t = days_from_civil(2026, 11, 21) * 86_400 + 2 * 3600 + 28 * 60 + 5;
        let c = calendar(t);
        assert_eq!(u16::from_le_bytes([c[0], c[1]]), 2026);
        assert_eq!(&c[2..7], &[11, 21, 2, 28, 5]);
        assert_eq!(u32::from_le_bytes(c[8..12].try_into().unwrap()), 6);
        assert_eq!(u32::from_le_bytes(c[12..16].try_into().unwrap()), 324);
        assert_eq!(c.len(), 0x20);
        assert_eq!(days_from_civil(1970, 1, 1), 0);
    }
}
