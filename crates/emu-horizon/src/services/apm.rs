//! `apm`: performance (clock) configuration. Commands as used by libnx's
//! `apm.c`: the manager opens an ISession, which gets and sets the
//! performance configuration of a performance mode.

use super::{obj, CallError, Ctx, Service};

/// "Handheld, CPU 1020 MHz, GPU 384 MHz, MEM 1600 MHz", the usual default.
const DEFAULT_CONFIGURATION: u32 = 0x9222_0007;

pub struct Manager;

impl Service for Manager {
    fn name(&self) -> &str {
        "apm"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            0 => ctx.push_object(obj(Session {
                configuration: [DEFAULT_CONFIGURATION; 2],
            })),
            1 => ctx.push_u32(0), // GetPerformanceMode: handheld.
            _ => return Err(CallError::Unimplemented),
        }
        Ok(())
    }
}

pub struct Session {
    configuration: [u32; 2],
}

impl Service for Session {
    fn name(&self) -> &str {
        "apm::ISession"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        let mode = (ctx.in_u32(0) as usize).min(1);
        match command {
            0 => self.configuration[mode] = ctx.in_u32(4),
            1 => ctx.push_u32(self.configuration[mode]),
            _ => return Err(CallError::Unimplemented),
        }
        Ok(())
    }
}
