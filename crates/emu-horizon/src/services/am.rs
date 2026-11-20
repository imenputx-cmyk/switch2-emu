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

/// Applet messages (`ReceiveMessage`).
pub mod message {
    pub const FOCUS_STATE_CHANGED: u32 = 15;
}

/// `ICommonStateGetter`. An application starts with one message,
/// FocusStateChanged (now in focus), and its message event signalled,
/// as Horizon's lifecycle manager does (Lichii/Eden
/// `am/lifecycle_manager.cpp`); SDK start-up waits for it.
pub struct CommonStateGetter {
    message_event: Option<usize>,
    messages: std::collections::VecDeque<u32>,
    sleep_lock_event: Option<usize>,
    resolution_event: Option<usize>,
    hdcp_event: Option<usize>,
}

impl Default for CommonStateGetter {
    fn default() -> Self {
        CommonStateGetter {
            message_event: None,
            messages: [message::FOCUS_STATE_CHANGED].into(),
            sleep_lock_event: None,
            resolution_event: None,
            hdcp_event: None,
        }
    }
}

impl Service for CommonStateGetter {
    fn name(&self) -> &str {
        "am::ICommonStateGetter"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            0 => {
                // GetEventHandle: signalled while messages are queued.
                let pending = !self.messages.is_empty();
                ctx.push_event(&mut self.message_event, pending);
            }
            1 => {
                // ReceiveMessage
                let Some(m) = self.messages.pop_front() else {
                    return Err(CallError::Result(RESULT_NO_MESSAGE));
                };
                if self.messages.is_empty() {
                    if let Some(e) = self.message_event {
                        ctx.kernel.events[e] = false;
                    }
                }
                ctx.push_u32(m);
            }
            5 => ctx.push_u8(0),  // GetOperationMode: handheld
            6 => ctx.push_u32(0), // GetPerformanceMode: normal
            8 => ctx.push_u8(0),  // GetBootMode: normal
            9 => ctx.push_u8(1),  // GetCurrentFocusState: in focus
            // Sleep locks, VR mode, LCD backlight, home button handling,
            // startup logo and next-program exit requests: accepted.
            10..=12 | 51..=54 | 80 | 100 | 130 | 900 => {}
            13 => ctx.push_event(&mut self.sleep_lock_event, true),
            50 => ctx.push_u8(0), // IsVrModeEnabled
            55 => ctx.push_u8(0), // IsInControllerFirmwareUpdateSection
            // GetDefaultDisplayResolution: handheld 1280x720.
            60 => {
                ctx.push_u32(1280);
                ctx.push_u32(720);
            }
            61 => ctx.push_event(&mut self.resolution_event, false),
            62 => ctx.push_u32(1), // GetHdcpAuthenticationState: authenticated
            63 => ctx.push_event(&mut self.hdcp_event, false),
            66 => {}                // SetCpuBoostMode
            68 => ctx.push_u32(0),  // GetBuiltInDisplayType
            120 => ctx.push_u32(0), // GetAppletLaunchedHistory: none
            200 => ctx.push_u32(0), // GetOperationModeSystemInfo
            300 => ctx.push_u32(1), // GetSettingsPlatformRegion: global
            // GetPanelPhysicalSizeForSpecificTitle (mm) and
            // GetPanelResolutionForSpecificTitle, as Lichii reports them.
            1004 => {
                ctx.push_u32(137.25f32.to_bits());
                ctx.push_u32(77.2f32.to_bits());
            }
            1005 => {
                ctx.push_u32(1280);
                ctx.push_u32(720);
            }
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
            // Exit locks, fatal sections, screenshot and notification
            // settings, display approval, sleep/dimming, idle detection,
            // input detection, album and recording options: accepted.
            1..=4 | 10..=16 | 19 | 50 | 51 | 60..=62 | 65 | 68 | 72 | 100 | 130 => {}
            41 => ctx.push_u8(0),  // IsSystemBufferSharingEnabled: no
            63 => ctx.push_u32(0), // GetIdleTimeDetectionExtension
            67 => ctx.push_u8(0),  // IsIlluminanceAvailable
            69 => ctx.push_u8(0),  // IsAutoSleepDisabled
            90 => ctx.push_u64(0), // GetAccumulatedSuspendedTickValue
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
    friend_invitation_event: Option<usize>,
    notification_event: Option<usize>,
    health_warning_event: Option<usize>,
    operation_mode_refresh_event: Option<usize>,
    /// The preselected user was handed out (once, as on hardware).
    user_popped: bool,
}

impl Service for ApplicationFunctions {
    fn name(&self) -> &str {
        "am::IApplicationFunctions"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            // GetDesiredLanguage: a language code as 8 NUL-padded bytes.
            21 => ctx.push_u64(u64::from_le_bytes(*b"en-US\0\0\0")),
            // PopLaunchParameter(kind): kind 2 is the preselected user
            // (magic 0xC79497CA, "account selected", the uid), once; the
            // rest has none (AM ResultNoDataInChannel). As in Eden.
            1 => {
                if ctx.in_u32(0) != 2 || std::mem::replace(&mut self.user_popped, true) {
                    return Err(CallError::Result(128 | (2 << 9)));
                }
                let mut data = vec![0u8; 0x88];
                data[..4].copy_from_slice(&0xC794_97CAu32.to_le_bytes());
                data[4] = 1;
                data[8..24].copy_from_slice(&super::account::USER_ID.to_le_bytes());
                ctx.push_object(obj(Storage(data)));
            }
            20 => ctx.push_u64(0), // EnsureSaveData: nothing to extend
            22 => {}               // SetTerminateResult
            40 => ctx.push_u8(0),  // NotifyRunning: no exit requested
            // GetPseudoDeviceId: a fixed 16-byte id.
            50 => ctx.push_bytes(b"switch2-emu-dev\0"),
            // GetGpuErrorDetectedSystemEvent: never signalled.
            130 => ctx.push_event(&mut self.gpu_error_event, false),
            // GetDisplayVersion: 16 NUL-padded bytes.
            23 => ctx.push_bytes(b"1.0.0\0\0\0\0\0\0\0\0\0\0\0"),
            25 => ctx.push_u64(0), // ExtendSaveData: nothing to extend
            // GetSaveDataSize: (normal, journal).
            26 => {
                ctx.push_u64(0);
                ctx.push_u64(0);
            }
            // GetSaveDataSizeMax, GetCacheStorageMax (Lichii's values).
            28 | 29 => {
                ctx.push_u64(0xFFF_FFFF);
                ctx.push_u64(0xFFF_FFFF);
            }
            // Home button blocking, media playback state, game play
            // recording, crash reports, copyright frame buffer, user
            // channel: accepted.
            30..=33 | 60 | 66 | 67 | 90 | 100..=102 | 121 | 122 => {}
            65 => ctx.push_u8(0), // IsGamePlayRecordingSupported
            // QueryApplicationPlayStatistics(ByUid): no entries.
            110 | 111 => ctx.push_u32(0),
            123 => ctx.push_u32(u32::MAX), // GetPreviousProgramIndex: none (-1)
            140 => ctx.push_event(&mut self.friend_invitation_event, false),
            150 => ctx.push_event(&mut self.notification_event, false),
            160 => ctx.push_event(&mut self.health_warning_event, false),
            210 => ctx.push_event(&mut self.operation_mode_refresh_event, false),
            // Switch 2 (Lichii): empty successes.
            320 | 321 | 330 => {}
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

/// `IStorage`: a block of data passed between applets.
pub struct Storage(pub Vec<u8>);

impl Service for Storage {
    fn name(&self) -> &str {
        "am::IStorage"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            // Open / OpenTransferStorage: an accessor over a copy.
            0 | 1 => ctx.push_object(obj(StorageAccessor(self.0.clone()))),
            _ => return Err(CallError::Unimplemented),
        }
        Ok(())
    }
}

/// `IStorageAccessor`.
pub struct StorageAccessor(pub Vec<u8>);

impl Service for StorageAccessor {
    fn name(&self) -> &str {
        "am::IStorageAccessor"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            0 => ctx.push_u64(self.0.len() as u64), // GetSize
            // Write(offset, buffer).
            10 => {
                let offset = ctx.in_u64(0) as usize;
                let data = ctx.read_in_buffer(0)?;
                if offset + data.len() > self.0.len() {
                    return Err(CallError::Result(128 | (503 << 9)));
                }
                self.0[offset..offset + data.len()].copy_from_slice(&data);
            }
            // Read(offset) into the buffer.
            11 => {
                let offset = (ctx.in_u64(0) as usize).min(self.0.len());
                let data = self.0[offset..].to_vec();
                ctx.write_out_buffer(0, &data)?;
            }
            _ => return Err(CallError::Unimplemented),
        }
        Ok(())
    }
}
