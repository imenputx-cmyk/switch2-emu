//! `hid`: input. The client reads controller state from a 0x40000-byte
//! shared memory block (layout in libnx's `hidshmem.h`); commands turn
//! input devices on and configure the controllers. So far the block is
//! left zeroed: no input.

use super::{obj, CallError, Ctx, Service};
use crate::handles::KObject;

pub const SHARED_MEMORY_SIZE: u64 = 0x40000;

#[derive(Default)]
pub struct Hid {
    shared_memory: Option<usize>,
}

impl Service for Hid {
    fn name(&self) -> &str {
        "hid"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            0 => {
                // CreateAppletResource
                let id = *self.shared_memory.get_or_insert_with(|| {
                    ctx.kernel.new_shared_memory(SHARED_MEMORY_SIZE, Vec::new())
                });
                ctx.push_object(obj(AppletResource { shared_memory: id }));
            }
            // Activate{DebugPad,TouchScreen,Mouse,Keyboard,Npad},
            // SetSupportedNpadStyleSet, SetSupportedNpadIdType,
            // DeactivateNpad, SetNpadJoyHoldType, ActivateNpadWithRevision:
            // accepted (the input state is not emulated yet).
            1 | 11 | 21 | 31 | 100 | 102 | 103 | 104 | 120 | 109 => {}
            _ => return Err(CallError::Unimplemented),
        }
        Ok(())
    }
}

pub struct AppletResource {
    shared_memory: usize,
}

impl Service for AppletResource {
    fn name(&self) -> &str {
        "hid::IAppletResource"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            0 => ctx.push_copy(KObject::SharedMemory {
                id: self.shared_memory,
            }),
            _ => return Err(CallError::Unimplemented),
        }
        Ok(())
    }
}
