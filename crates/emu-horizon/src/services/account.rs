//! `acc:u0` / `acc:u1`: user accounts. One user exists and is open, so
//! games find a player and a place for save data. Commands follow
//! Eden's `service/acc` (GPL-3.0-or-later) and switchbrew.

use super::{generic, CallError, Ctx, Service};
use std::cell::RefCell;
use std::rc::Rc;

/// The emulated user's id (128-bit, never zero).
pub const USER_ID: u128 = 0x0000_0000_0000_0001_5EED_0000_2025_0001;

/// Its nickname.
pub const NICKNAME: &[u8] = b"Player";

/// The 0x38-byte ProfileBase: uid, last-edit timestamp, nickname[0x21].
pub fn profile_base() -> Vec<u8> {
    let mut b = vec![0u8; 0x38];
    b[..16].copy_from_slice(&USER_ID.to_le_bytes());
    b[16..24].copy_from_slice(&1_700_000_000u64.to_le_bytes());
    b[24..24 + NICKNAME.len()].copy_from_slice(NICKNAME);
    b
}

fn user_list(ctx: &mut Ctx) -> Result<(), CallError> {
    if let Some(buf) = ctx.out_buffer(0) {
        let mut list = vec![0u8; buf.size as usize];
        if list.len() >= 16 {
            list[..16].copy_from_slice(&USER_ID.to_le_bytes());
        }
        ctx.write_out_buffer(0, &list)?;
    }
    Ok(())
}

/// `IAccountServiceForApplication`.
pub struct AccountService(pub String);

impl Service for AccountService {
    fn name(&self) -> &str {
        &self.0
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            0 => ctx.push_u32(1), // GetUserCount
            1 => ctx.push_u8(u8::from(ctx.in_bytes(0, 16) == USER_ID.to_le_bytes())), // GetUserExistence
            2 | 3 | 131 | 141 => user_list(ctx)?, // ListAllUsers, ListOpenUsers, ...
            4 | 51 => ctx.push_bytes(&USER_ID.to_le_bytes()), // GetLastOpenedUser, TrySelectUserWithoutInteraction
            5 => ctx.push_object(Rc::new(RefCell::new(Profile))), // GetProfile
            50 => ctx.push_u8(0),                             // IsUserRegistrationRequestPermitted
            // InitializeApplicationInfo(V0/V2/...), StoreSaveDataThumbnail,
            // ClearSaveDataThumbnail, SetAccountInfo...
            100 | 110 | 111 | 120 | 140 | 150 | 160 => {}
            101 => ctx.push_object(Rc::new(RefCell::new(ManagerForApplication))),
            // AuthenticateApplicationAsync / CreateGuestLoginRequest: an
            // async context that completes at once (offline).
            102 | 103 => ctx.push_object(generic::object("acc::IAsyncContext")),
            130 => ctx.push_object(Rc::new(RefCell::new(ManagerForApplication))),
            _ => return Err(CallError::Unimplemented),
        }
        Ok(())
    }
}

/// `IProfile`.
pub struct Profile;

impl Service for Profile {
    fn name(&self) -> &str {
        "acc::IProfile"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            // Get: ProfileBase, and UserData (0x80 bytes) into the buffer.
            0 => {
                if ctx.out_buffer(0).is_some() {
                    ctx.write_out_buffer(0, &[0; 0x80])?;
                }
                ctx.push_bytes(&profile_base());
            }
            1 => ctx.push_bytes(&profile_base()), // GetBase
            10 => ctx.push_u32(0),                // GetImageSize: no picture
            11 => ctx.push_u32(0),                // LoadImage
            _ => return Err(CallError::Unimplemented),
        }
        Ok(())
    }
}

/// `IManagerForApplication`: the user's Nintendo Account (offline).
pub struct ManagerForApplication;

impl Service for ManagerForApplication {
    fn name(&self) -> &str {
        "acc::IManagerForApplication"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            0 => {}                                                        // CheckAvailability
            1 => ctx.push_u64(USER_ID as u64),                             // GetAccountId
            2 => ctx.push_object(generic::object("acc::IAsyncContext")), // EnsureIdTokenCacheAsync
            3 => ctx.push_u32(0),                                        // LoadIdTokenCache: empty
            130 => ctx.push_bytes(&[0; 0x68]), // GetNintendoAccountUserResourceCacheForApplication
            150 => ctx.push_object(generic::object("acc::IAsyncContext")), // CreateAuthorizationRequest
            160 | 170 => {}                                                // StoreOpenContext, ...
            _ => return Err(CallError::Unimplemented),
        }
        Ok(())
    }
}
