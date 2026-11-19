//! `sm:`, the service manager: clients register and ask for services by
//! name (commands as used by libnx's `sm.c`: 0 RegisterClient,
//! 1 GetServiceHandle, 4 DetachClient).

use super::{create, CallError, Ctx, Service};

/// sm results use module 21. NotRegistered is description 7 (0xE15).
pub const RESULT_NOT_REGISTERED: u32 = 21 | (7 << 9);
pub const RESULT_INVALID_SERVICE_NAME: u32 = 21 | (6 << 9);

pub struct ServiceManager;

/// Service names are up to 8 bytes, NUL padded, passed as a u64.
pub fn decode_name(raw: u64) -> Option<String> {
    let bytes = raw.to_le_bytes();
    let len = bytes.iter().position(|&b| b == 0).unwrap_or(8);
    if len == 0 || bytes[len..].iter().any(|&b| b != 0) {
        return None;
    }
    String::from_utf8(bytes[..len].to_vec()).ok()
}

impl Service for ServiceManager {
    fn name(&self) -> &str {
        "sm:"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            0 | 4 => Ok(()), // RegisterClient / DetachClient
            1 => {
                let name = decode_name(ctx.in_u64(0))
                    .ok_or(CallError::Result(RESULT_INVALID_SERVICE_NAME))?;
                let service =
                    create(&name, ctx.config).ok_or(CallError::Result(RESULT_NOT_REGISTERED))?;
                ctx.push_object(service);
                Ok(())
            }
            _ => Err(CallError::Unimplemented),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn service_names() {
        assert_eq!(
            decode_name(u64::from_le_bytes(*b"set:sys\0")).as_deref(),
            Some("set:sys")
        );
        assert_eq!(
            decode_name(u64::from_le_bytes(*b"fsp-srv\0")).as_deref(),
            Some("fsp-srv")
        );
        assert_eq!(decode_name(0), None);
        assert_eq!(
            decode_name(u64::from_le_bytes(*b"a\0b\0\0\0\0\0")),
            None,
            "garbage after NUL"
        );
        assert_eq!(RESULT_NOT_REGISTERED, 0xE15);
    }
}
