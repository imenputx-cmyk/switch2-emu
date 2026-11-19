//! Horizon result codes: `module | (description << 9)`, 0 is success.
//! Kernel descriptions as listed in libnx's `result.h` (ISC license).

pub type ResultCode = u32;

pub const SUCCESS: ResultCode = 0;

const fn kernel(description: u32) -> ResultCode {
    1 | (description << 9)
}

pub const NOT_IMPLEMENTED: ResultCode = kernel(33);
pub const INVALID_SIZE: ResultCode = kernel(101);
pub const INVALID_ADDRESS: ResultCode = kernel(102);
pub const OUT_OF_MEMORY: ResultCode = kernel(104);
pub const OUT_OF_HANDLES: ResultCode = kernel(105);
pub const INVALID_MEMORY_STATE: ResultCode = kernel(106);
pub const INVALID_MEMORY_PERMISSIONS: ResultCode = kernel(108);
pub const INVALID_MEMORY_RANGE: ResultCode = kernel(110);
pub const INVALID_HANDLE: ResultCode = kernel(114);
pub const INVALID_COMBINATION: ResultCode = kernel(116);
pub const TIMED_OUT: ResultCode = kernel(117);
pub const INVALID_ENUM_VALUE: ResultCode = kernel(120);
pub const NOT_FOUND: ResultCode = kernel(121);
pub const OUT_OF_RANGE: ResultCode = kernel(119);
pub const INVALID_STATE: ResultCode = kernel(125);

#[cfg(test)]
mod tests {
    #[test]
    fn values_match_known_codes() {
        // 0xCA01 is the well-known InvalidSize code, 0xEA01 TimedOut.
        assert_eq!(super::INVALID_SIZE, 0xCA01);
        assert_eq!(super::TIMED_OUT, 0xEA01);
        assert_eq!(super::INVALID_HANDLE, 0xE401);
    }
}
