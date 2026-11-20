//! `appletOE`: the applet manager as seen by an application. Opening the
//! application proxy gives access to the sub-interfaces libnx's
//! `applet.c` uses during `appletInitialize` (command ids from there and
//! the switchbrew "Applet Manager services" page).

use super::{obj, vi, CallError, Ctx, Service};

/// AM's "no message" result (module 128, description 3).
const RESULT_NO_MESSAGE: u32 = 128 | (3 << 9);

/// Applet resource user id of the application.
pub const ARUID: u64 = 0x50;

pub struct AppletOE;

impl Service for AppletOE {
    fn name(&self) -> &str {
        "appletOE"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            // OpenApplicationProxy; 10 is the Switch 2 form (Lichii).
            0 | 10 => ctx.push_object(obj(ApplicationProxy)),
            _ => return Err(CallError::Unimplemented),
        }
        Ok(())
    }
}

pub struct ApplicationProxy;

impl Service for ApplicationProxy {
    fn name(&self) -> &str {
        "am::IApplicationProxy"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        let iface = match command {
            0 => obj(CommonStateGetter::default()),
            1 => obj(SelfController::default()),
            2 => obj(WindowController),
            3 => obj(Empty("am::IAudioController")),
            4 => obj(Empty("am::IDisplayController")),
            11 => obj(Empty("am::ILibraryAppletCreator")),
            20 => obj(ApplicationFunctions),
            1000 => obj(Empty("am::IDebugFunctions")),
            _ => return Err(CallError::Unimplemented),
        };
        ctx.push_object(iface);
        Ok(())
    }
}

/// An interface whose commands are not needed yet.
pub struct Empty(pub &'static str);

impl Service for Empty {
    fn name(&self) -> &str {
        self.0
    }

    fn call(&mut self, _command: u32, _ctx: &mut Ctx) -> Result<(), CallError> {
        Err(CallError::Unimplemented)
    }
}

#[derive(Default)]
pub struct CommonStateGetter {
    message_event: Option<usize>,
}

impl Service for CommonStateGetter {
    fn name(&self) -> &str {
        "am::ICommonStateGetter"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            0 => ctx.push_event(&mut self.message_event, false), // GetEventHandle
            1 => return Err(CallError::Result(RESULT_NO_MESSAGE)), // ReceiveMessage
            5 => ctx.push_u8(0),                                 // GetOperationMode: handheld
            6 => ctx.push_u32(0),                                // GetPerformanceMode: normal
            9 => ctx.push_u8(1),                                 // GetCurrentFocusState: in focus
            66 => {}                                             // SetCpuBoostMode
            _ => return Err(CallError::Unimplemented),
        }
        Ok(())
    }
}

#[derive(Default)]
pub struct SelfController {
    suspended_tick_event: Option<usize>,
}

impl Service for SelfController {
    fn name(&self) -> &str {
        "am::ISelfController"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            // LockExit, UnlockExit, SetScreenShotPermission,
            // Set{OperationMode,PerformanceMode}ChangedNotification,
            // SetFocusHandlingMode, SetRestartMessageEnabled,
            // SetOutOfFocusSuspendingEnabled: accepted.
            1 | 2 | 10 | 11 | 12 | 13 | 14 | 16 => {}
            40 => ctx.push_u64(vi::MANAGED_LAYER_ID), // CreateManagedDisplayLayer
            91 => ctx.push_event(&mut self.suspended_tick_event, false),
            _ => return Err(CallError::Unimplemented),
        }
        Ok(())
    }
}

pub struct WindowController;

impl Service for WindowController {
    fn name(&self) -> &str {
        "am::IWindowController"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            1 => ctx.push_u64(ARUID), // GetAppletResourceUserId
            10 => {}                  // AcquireForegroundRights
            _ => return Err(CallError::Unimplemented),
        }
        Ok(())
    }
}

pub struct ApplicationFunctions;

impl Service for ApplicationFunctions {
    fn name(&self) -> &str {
        "am::IApplicationFunctions"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            // GetDesiredLanguage: a language code as 8 NUL-padded bytes.
            21 => ctx.push_u64(u64::from_le_bytes(*b"en-US\0\0\0")),
            22 => {}              // SetTerminateResult
            40 => ctx.push_u8(0), // NotifyRunning: no exit requested
            _ => return Err(CallError::Unimplemented),
        }
        Ok(())
    }
}
