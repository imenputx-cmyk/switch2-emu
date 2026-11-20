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
            20 => obj(ApplicationFunctions::default()),
            // Switch 2 getters (Lichii): 40 is unknown (every command
            // succeeds), 60 the graphics scale controller.
            40 => obj(super::AutoStub("nx2:IApplicationProxy40".into())),
            60 => obj(GraphicsScaleController),
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
            // GetDefaultDisplayResolution: handheld 1280x720.
            60 => {
                ctx.push_u32(1280);
                ctx.push_u32(720);
            }
            66 => {} // SetCpuBoostMode
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
            // Switch 2: a raw session to the graphics system server (gsv).
            2000 => ctx.push_session(obj(super::nx2::Gsv::default())),
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

#[derive(Default)]
pub struct ApplicationFunctions {
    gpu_error_event: Option<usize>,
}

impl Service for ApplicationFunctions {
    fn name(&self) -> &str {
        "am::IApplicationFunctions"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            // GetDesiredLanguage: a language code as 8 NUL-padded bytes.
            21 => ctx.push_u64(u64::from_le_bytes(*b"en-US\0\0\0")),
            // PopLaunchParameter: none (AM ResultNoDataInChannel).
            1 => return Err(CallError::Result(128 | (2 << 9))),
            20 => ctx.push_u64(0), // EnsureSaveData: nothing to extend
            22 => {}               // SetTerminateResult
            40 => ctx.push_u8(0),  // NotifyRunning: no exit requested
            // GetPseudoDeviceId: a fixed 16-byte id.
            50 => ctx.push_bytes(b"switch2-emu-dev\0"),
            // GetGpuErrorDetectedSystemEvent: never signalled.
            130 => ctx.push_event(&mut self.gpu_error_event, false),
            // Switch 2 (Lichii): empty successes.
            320 | 321 => {}
            _ => return Err(CallError::Unimplemented),
        }
        Ok(())
    }
}

/// Switch 2 `IApplicationProxy` 60 (the name is Lichii's guess):
/// `nn::oe::GetCurrentGraphicsScale` reads command 10 and accepts 10, 20
/// or 30.
pub struct GraphicsScaleController;

impl Service for GraphicsScaleController {
    fn name(&self) -> &str {
        "nx2:IGraphicsScaleController"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            10 => {
                ctx.push_u32(10);
                ctx.push_u32(0);
                ctx.push_u32(0);
            }
            1050 => {} // SetGraphicsScaleChangedNotificationEnabled
            _ => return Err(CallError::Unimplemented),
        }
        Ok(())
    }
}
