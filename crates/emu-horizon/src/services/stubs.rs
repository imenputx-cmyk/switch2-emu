//! Commands Lichii answers with its generic stubs (`StubReplyZeroed`,
//! `StubReplyEvent`; service.cpp): success and 32 zeroed words, plus
//! an event for the event kind. Generated from Lichii's service tables;
//! used when one of our services does not implement a command.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stub {
    Zeroed,
    Event,
}

/// Words of zeroed output a stub reply carries.
pub const ZEROED_WORDS: usize = 32;

pub fn lichii(service: &str, command: u32) -> Option<Stub> {
    let (zeroed, event): (&[u32], &[u32]) = match service {
        "am::IApplicationFunctions" => (
            &[
                0, 10, 11, 13, 14, 15, 24, 35, 36, 68, 70, 71, 72, 80, 112, 113, 124, 131, 170,
                180, 181, 190, 200, 220, 230, 310, 2000,
            ],
            &[37, 500],
        ),
        "am::IAudioController" => (&[5], &[]),
        "am::ICommonStateGetter" => (
            &[
                2, 3, 4, 7, 14, 40, 59, 64, 65, 67, 90, 91, 310, 320, 400, 401, 500, 501, 502, 503,
                600, 650, 651, 910, 2000, 2010,
            ],
            &[15],
        ),
        "am::IDisplayController" => (
            &[0, 1, 2, 3, 4, 6, 9, 11, 13, 15, 19, 21, 28],
            &[10, 12, 14, 16, 17, 18],
        ),
        "am::ILibraryAppletCreator" => (&[1, 2], &[]),
        "am::ISelfController" => (
            &[
                17, 18, 20, 21, 22, 23, 24, 45, 46, 64, 66, 70, 71, 73, 80, 110, 200, 210, 211,
                220, 221, 2100, 2200, 2300, 2400, 2401, 2402,
            ],
            &[],
        ),
        "aoc:u" => (&[9, 13, 200, 300, 301, 302], &[]),
        "audren::IAudioDevice" => (&[9, 21], &[]),
        "bsd:a" => (
            &[
                4, 7, 19, 23, 28, 32, 33, 34, 35, 37, 38, 39, 40, 42, 43, 44, 200, 201,
            ],
            &[41],
        ),
        "bsd:s" => (
            &[
                4, 7, 19, 23, 28, 32, 33, 34, 35, 37, 38, 39, 40, 42, 43, 44, 200, 201,
            ],
            &[41],
        ),
        "bsd:u" => (
            &[
                4, 7, 19, 23, 28, 32, 33, 34, 35, 37, 38, 39, 40, 42, 43, 44, 200, 201,
            ],
            &[41],
        ),
        "bsdcfg" => (
            &[
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 100, 101, 102, 103, 104,
            ],
            &[],
        ),
        "caps:su" => (&[201, 210], &[]),
        "caps:u" => (
            &[
                103, 104, 105, 110, 120, 130, 140, 141, 143, 144, 145, 146, 147, 148,
            ],
            &[],
        ),
        "hid" => (
            &[
                73, 74, 75, 76, 77, 78, 86, 137, 138, 139, 140, 150, 151, 160, 308, 309, 311, 312,
                320, 321, 322, 323, 324, 401, 402, 403, 404, 405, 406, 407, 550, 1005, 3000, 3001,
                3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010, 3011, 3012, 3013, 3014, 3015,
                3110, 3111, 3112, 3113, 3114, 3130, 3131, 3132, 3133, 3134, 3135, 3136, 3137, 3138,
                3139, 3140, 3141, 3142, 3143, 3144, 3145, 3146, 3147, 3148, 3149, 5000,
            ],
            &[],
        ),
        "lm::ILogger" => (&[2, 3], &[]),
        "mii::IDatabaseService" => (&[18, 19, 27, 28], &[]),
        "nifm::IGeneralService" => (&[101], &[]),
        "nx2:gpu" => (&[23, 24, 25, 34, 51, 8001, 8002, 8003, 8004], &[]),
        "nx2:gsv" => (
            &[
                113, 120, 121, 122, 123, 124, 125, 131, 133, 187, 211, 212, 213, 214, 215, 303,
                424, 425, 436, 461, 462, 463, 464, 465, 466, 467, 468, 493, 603, 613, 614, 802,
                803, 813, 814, 1003, 1011, 1012, 1091, 1112, 1113, 1203, 1602, 1603,
            ],
            &[],
        ),
        "nx2:syncpt" => (&[21], &[]),
        "nx2:vi2" => (
            &[
                13, 51, 101, 102, 107, 108, 111, 112, 113, 201, 202, 230, 231, 232, 251, 252, 301,
                302, 351, 352, 353, 354, 355, 356, 358, 359,
            ],
            &[],
        ),
        "set" => (&[10, 13, 14, 15, 16, 17], &[]),
        "ssl::ISslConnection" => (&[6, 24, 25, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37], &[]),
        _ => return None,
    };
    if event.contains(&command) {
        Some(Stub::Event)
    } else if zeroed.contains(&command) {
        Some(Stub::Zeroed)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_stubs() {
        assert_eq!(lichii("hid", 3110), Some(Stub::Zeroed));
        assert_eq!(lichii("am::IApplicationFunctions", 37), Some(Stub::Event));
        assert_eq!(lichii("hid", 0), None);
        assert_eq!(lichii("nope", 1), None);
    }
}
