//! Services and interfaces whose commands only need fixed replies, written
//! as tables. Command numbers and reply shapes follow Eden/Lichii's
//! implementations (GPL-3.0-or-later) and switchbrew. A command missing
//! from its table stops the emulation naming it, so gaps show up.

use super::{CallError, Ctx, Service, ServiceRef};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// The reply of a table command.
#[derive(Debug, Clone, Copy)]
pub enum Reply {
    /// Success, no data.
    Ok,
    /// Success with `n` zero bytes of data.
    Zeros(usize),
    U8(u8),
    U32(u32),
    U64(u64),
    /// Success with an interface (by its table name).
    Object(&'static str),
    /// Success with a copy handle to an event (one per command), signalled
    /// or not.
    Event(bool),
    /// Fails with this result code.
    Error(u32),
    /// Success; the first output buffer is zero-filled and the count 0 is
    /// returned (empty lists).
    EmptyList,
}

/// A service or interface answering from a table.
pub struct TableService {
    name: &'static str,
    commands: &'static [(u32, Reply)],
    events: HashMap<u32, Option<usize>>,
}

impl TableService {
    pub fn new(name: &'static str) -> Option<Self> {
        Some(TableService {
            name,
            commands: table(name)?,
            events: HashMap::new(),
        })
    }
}

/// The interface `name` as a service object.
pub fn object(name: &'static str) -> ServiceRef {
    Rc::new(RefCell::new(
        TableService::new(name).unwrap_or_else(|| panic!("no table for {name}")),
    ))
}

impl Service for TableService {
    fn name(&self) -> &str {
        self.name
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        let Some(&(_, reply)) = self.commands.iter().find(|(c, _)| *c == command) else {
            return Err(CallError::Unimplemented);
        };
        match reply {
            Reply::Ok => {}
            Reply::Zeros(n) => ctx.push_bytes(&vec![0; n]),
            Reply::U8(v) => ctx.push_u8(v),
            Reply::U32(v) => ctx.push_u32(v),
            Reply::U64(v) => ctx.push_u64(v),
            Reply::Object(name) => ctx.push_object(object(name)),
            Reply::Event(signalled) => {
                let mut e = self.events.get(&command).copied().flatten();
                ctx.push_event(&mut e, signalled);
                self.events.insert(command, e);
            }
            Reply::Error(code) => return Err(CallError::Result(code)),
            Reply::EmptyList => {
                if let Some(b) = ctx.out_buffer(0) {
                    ctx.write_out_buffer(0, &vec![0; b.size as usize])?;
                }
                ctx.push_u32(0);
            }
        }
        Ok(())
    }
}

/// nifm results: no network (NIFM ResultNetworkNotAvailable-like).
const NIFM_NO_NETWORK: u32 = 110 | (300 << 9);
/// An IRequest's state: 1 = error (no connection).
const NIFM_REQUEST_ERROR: u32 = 1;

/// The static name of a table service (`name` as registered with sm).
pub fn table_name(name: &str) -> Option<&'static str> {
    NAMES.iter().copied().find(|n| *n == name)
}

const NAMES: &[&str] = &[
    "nfp:user", "nfp:sys", "nfc:user", "nfc:sys", "prepo:u", "prepo:a", "prepo:a2", "prepo:s",
    "nifm:u", "nifm:a", "nifm:s", "friend:u", "friend:v", "friend:m", "friend:s", "friend:a",
    "bcat:u", "bcat:s", "bcat:m", "bcat:a", "olsc:u", "npns:u", "npns:s", "ectx:r", "erpt:c",
    "erpt:r", "psm", "caps:su", "caps:u", "ssl", "bsd:u", "bsd:s", "bsd:a", "ns:am2", "ns:ec",
    "ns:rid", "ns:rt", "ns:web", "ns:ro", "ns:dev", "ns:su", "ns:vm", "mii:e", "mii:u", "lbl",
    "spsm", "idle:sys", "omm", "fgm", "fgm:0", "fgm:9", "ntc", "mm:u", "pm:shell", "pm:bm",
    "pm:info", "pm:dmnt",
];

/// The table for `name`.
pub fn table(name: &str) -> Option<&'static [(u32, Reply)]> {
    use Reply::*;
    Some(match name {
        // Play reports: accepted and dropped.
        "prepo:u" | "prepo:a" | "prepo:a2" | "prepo:s" => &[
            (10100, Ok),
            (10101, Ok),
            (10102, Ok),
            (10103, Ok),
            (10104, Ok),
            (10105, Ok),
            (10106, Ok),
            (10107, Ok),
            (10200, Ok),
            (10300, Ok),
            (20100, Ok),
            (20101, Ok),
            (20102, Ok),
            (20103, Ok),
            (20104, Ok),
            (20200, Ok),
            (30100, Ok),
            (30200, Ok),
            (40100, Ok),
            (40101, Ok),
            (40200, Ok),
            (40300, Ok),
            (50100, Ok),
            (60100, Ok),
            (90100, Ok),
            (90101, Ok),
            (90102, Ok),
        ],
        // Network: present but never connected.
        // Amiibo (nfp) and NFC: initialized, no readers, nothing found
        // (Lichii nfp.cpp / nfc.cpp).
        "nfp:user" | "nfp:sys" => &[(0, Object("nfp::IUser"))],
        "nfp::IUser" => &[
            (0, Ok),        // Initialize
            (1, Ok),        // Finalize
            (2, EmptyList), // ListDevices
            (3, Ok),        // StartDetection
            (4, Ok),        // StopDetection
            (17, Event(false)),
            (18, Event(false)),
            (19, U32(1)), // GetState: initialized
            (20, U32(0)), // GetDeviceState: none
            (21, U32(0)), // GetNpadId
            (23, Event(false)),
            (25, Ok), // StartDetectionWithFilter
        ],
        "nfc:user" | "nfc:sys" => &[(0, Object("nfc::IUser"))],
        "nfc::IUser" => &[
            (0, Ok),             // Initialize (old)
            (1, Ok),             // Finalize (old)
            (2, U32(1)),         // GetState (old)
            (3, U8(0)),          // IsNfcEnabled (old)
            (400, Ok),           // Initialize
            (401, Ok),           // Finalize
            (402, U32(1)),       // GetState
            (403, U8(0)),        // IsNfcEnabled
            (404, EmptyList),    // ListDevices
            (405, U32(0)),       // GetDeviceState
            (406, U32(0)),       // GetNpadId
            (407, Event(false)), // AttachAvailabilityChangeEvent
            (408, Ok),
            (409, Ok),
            (411, Event(false)),
            (412, Event(false)),
        ],
        "nifm:u" | "nifm:a" | "nifm:s" => &[
            (4, Object("nifm::IGeneralService")),
            (5, Object("nifm::IGeneralService")),
        ],
        "nifm::IGeneralService" => &[
            (1, Zeros(0)),                 // GetClientId (buffer)
            (4, Object("nifm::IRequest")), // CreateRequest
            (5, Error(NIFM_NO_NETWORK)),   // GetCurrentNetworkProfile
            (12, Error(NIFM_NO_NETWORK)),  // GetCurrentIpAddress
            (14, Object("nifm::INetworkProfile")),
            (18, Error(NIFM_NO_NETWORK)), // GetInternetConnectionStatus
            (21, U8(0)),                  // IsWirelessCommunicationEnabled
            (22, U8(0)),                  // IsEthernetCommunicationEnabled
            (32, U8(0)),                  // IsAnyInternetRequestAccepted
            (33, U8(0)),                  // IsAnyForegroundRequestAccepted
        ],
        "nifm::INetworkProfile" => &[(0, Ok)],
        "nifm::IRequest" => &[
            (0, U32(NIFM_REQUEST_ERROR)), // GetRequestState
            (1, Error(NIFM_NO_NETWORK)),  // GetResult
            (2, Event(false)),            // GetSystemEventReadableHandles
            (3, Ok),                      // Cancel
            (4, Ok),                      // Submit
            (5, Ok),
            (6, Ok), // SetRequirementPreset
            (8, Ok),
            (11, Ok), // SetConnectionConfirmationOption
            (13, Ok),
            (14, Ok), // SetPersistent
            (15, Ok),
            (17, Ok),
            (19, Ok),
            (21, Zeros(0xC)), // GetAppletInfo
            (22, Zeros(4)),
            (23, Ok),
            (24, Ok),
        ],
        // Friends: none.
        "friend:u" | "friend:v" | "friend:m" | "friend:s" | "friend:a" => &[
            (0, Object("friends::IFriendService")),
            (1, Object("friends::INotificationService")),
            (2, Object("friends::IDaemonSuspendSessionService")),
        ],
        "friends::IFriendService" => &[
            (0, Event(false)),  // GetCompletionEvent
            (1, Ok),            // Cancel
            (10100, EmptyList), // GetFriendListIds
            (10101, EmptyList), // GetFriendList
            (10102, Ok),        // UpdateFriendInfo
            (10110, Ok),
            (10400, EmptyList), // GetBlockedUserListIds
            (10420, U8(0)),     // CheckBlockedUserListAvailability
            (10500, Ok),        // GetProfileList
            (10600, Ok),        // DeclareOpenOnlinePlaySession
            (10601, Ok),        // DeclareCloseOnlinePlaySession
            (10610, Ok),        // UpdateUserPresence
            (10700, Ok),
            (10701, Ok),
            (10702, Ok), // AddPlayHistory
            (11000, Ok),
            (20100, U32(0)), // GetFriendCount
            (20101, U32(0)), // GetNewlyFriendCount
            (20102, Zeros(0)),
            (20103, Ok),
            (20110, U8(0)),
            (20800, Ok),
            (20801, Ok),
            (20900, Ok),
            (30100, Ok),
            (30101, Ok),
        ],
        "friends::INotificationService" => &[
            (0, Event(false)),  // GetEvent
            (1, Ok),            // Clear
            (2, Error(0x2E21)), // Pop: nothing (friends NotificationNotFound-like)
        ],
        "friends::IDaemonSuspendSessionService" => &[],
        // BCAT (news / delivery cache): empty.
        "bcat:u" | "bcat:s" | "bcat:m" | "bcat:a" => &[
            (0, Object("bcat::IBcatService")),
            (1, Object("bcat::IDeliveryCacheStorageService")),
            (2, Object("bcat::IDeliveryCacheStorageService")),
        ],
        "bcat::IBcatService" => &[
            (10100, Object("bcat::IDeliveryCacheProgressService")),
            (20100, Object("bcat::IDeliveryCacheProgressService")),
            (30100, Ok),
            (30101, Ok),
            (30200, Ok),
            (30201, Ok),
            (30202, Ok),
            (30203, Ok),
            (30204, Ok),
            (30205, Ok),
            (30210, Ok),
            (30300, Ok),
            (90100, EmptyList),
            (90200, EmptyList),
        ],
        "bcat::IDeliveryCacheProgressService" => &[(0, Event(true)), (1, Zeros(0x200))],
        "bcat::IDeliveryCacheStorageService" => &[
            (0, Object("bcat::IDeliveryCacheFileService")),
            (1, Object("bcat::IDeliveryCacheDirectoryService")),
            (10, EmptyList), // EnumerateDeliveryCacheDirectory
        ],
        "bcat::IDeliveryCacheFileService" => {
            &[(0, Error(0x780A)), (1, U64(0)), (2, U64(0)), (3, Zeros(32))]
        }
        "bcat::IDeliveryCacheDirectoryService" => {
            &[(0, Error(0x780A)), (1, EmptyList), (2, U32(0))]
        }
        // Online saves, news, notifications, telemetry: present, idle.
        "olsc:u" => &[(0, Object("olsc::IOlscServiceForApplication"))],
        "olsc::IOlscServiceForApplication" => &[
            (0, Ok),
            (13, U8(0)),
            (14, Ok),
            (15, Ok),
            (16, Ok),
            (19, U8(0)),
        ],
        "npns:u" | "npns:s" => &[
            (1, Ok),
            (2, Ok),
            (3, Ok),
            (4, Ok),
            (5, Event(false)),
            (6, Ok),
            (7, Event(false)),
            (101, Ok),
            (102, Ok),
            (103, Ok),
            (104, Ok),
            (111, Error(0x6A5)), // no token
            (112, Error(0x6A5)),
            (113, Ok),
            (114, Ok),
        ],
        "ectx:r" => &[(0, Ok), (1, Ok), (2, Ok)],
        // An account operation that completes at once (offline).
        "acc::IAsyncContext" => &[
            (0, Event(true)), // GetSystemEvent
            (1, Ok),          // Cancel
            (2, U8(1)),       // HasDone
            (3, Ok),          // GetResult
        ],
        "erpt:c" => &[
            (0, Ok),
            (1, Ok),
            (2, Ok),
            (3, Ok),
            (4, Ok),
            (5, Ok),
            (6, Ok),
            (7, Ok),
            (8, Ok),
            (9, Ok),
            (10, Ok),
            (11, Ok),
        ],
        "erpt:r" => &[(0, Object("erpt::ISession"))],
        "erpt::ISession" => &[(0, Object("erpt::IReport")), (1, Object("erpt::IManager"))],
        "erpt::IReport" => &[(0, Ok), (1, Ok), (2, Ok), (3, Ok), (4, Ok), (5, U64(0))],
        "erpt::IManager" => &[(0, Ok), (1, Event(false))],
        // Battery and charging.
        "psm" => &[
            (0, U32(100)), // GetBatteryChargePercentage
            (1, U32(1)),   // GetChargerType: official charger
            (5, U8(1)),    // IsBatteryChargingEnabled
            (7, Object("psm::IPsmSession")),
            (13, U32(2)),           // GetBatteryChargeInfoEvent... level
            (17, U32(0x0050_0000)), // GetBatteryAgePercentage-ish
            (18, U32(0)),
        ],
        "psm::IPsmSession" => &[
            (0, Event(false)),
            (1, Ok),
            (2, Ok),
            (3, Ok),
            (4, Ok),
            (5, Ok),
        ],
        // Screenshots: refused quietly.
        "caps:su" => &[
            (32, Ok),
            (33, Ok),
            (34, Ok),
            (201, Ok),
            (203, Ok),
            (205, Ok),
            (210, Ok),
        ],
        "caps:u" => &[
            (32, Ok),
            (102, EmptyList),
            (103, EmptyList),
            (104, EmptyList),
        ],
        // SSL: contexts exist, connections are never made.
        "ssl" => &[
            (0, Object("ssl::ISslContext")),
            (1, U32(0)), // GetContextCount
            (2, U32(0)), // GetCertificates
            (3, U32(0)), // GetCertificateBufSize
            (5, Ok),     // SetInterfaceVersion
            (100, Object("ssl::ISslContext")),
            (101, Ok),
        ],
        "ssl::ISslContext" => &[
            (0, Ok),
            (1, U32(0)),
            (2, Object("ssl::ISslConnection")),
            (3, U32(0)),
            (4, U64(1)), // ImportServerPki
            (5, U64(1)), // ImportClientPki
            (6, Ok),
            (7, Ok),
            (8, U64(1)), // RegisterInternalPki
            (9, U64(1)),
            (10, U32(0)),
            (11, U64(1)),
            (12, Ok),
            (13, U64(1)),
            (14, U32(0)),
        ],
        "ssl::ISslConnection" => &[
            (0, Ok),
            (1, Ok),
            (2, Ok),
            (3, Ok),
            (4, Ok),
            (5, Ok),
            (6, Ok),
        ],
        // Sockets: initialized, no network.
        "bsd:u" | "bsd:s" | "bsd:a" => &[(0, U32(0)), (1, Ok)],
        // Background tasks, eShop, misc: idle.
        "ns:am2" | "ns:ec" | "ns:rid" | "ns:rt" | "ns:web" | "ns:ro" | "ns:dev" | "ns:su"
        | "ns:vm" => &[
            (7988, Object("ns::IDynamicRightsInterface")),
            (7989, Object("ns::IReadOnlyApplicationControlDataInterface")),
            (7991, Object("ns::IReadOnlyApplicationRecordInterface")),
            (7992, Object("ns::IECommerceInterface")),
            (7993, Object("ns::IApplicationVersionInterface")),
            (7994, Object("ns::IFactoryResetInterface")),
            (7995, Object("ns::IAccountProxyInterface")),
            (7996, Object("ns::IApplicationManagerInterface")),
            (7997, Object("ns::IDownloadTaskInterface")),
            (7998, Object("ns::IContentManagementInterface")),
            (7999, Object("ns::IDocumentInterface")),
        ],
        "ns::IDynamicRightsInterface" => &[
            (0, Ok),
            (1, Ok),
            (2, U32(0)),
            (4, Ok),
            (5, U8(0)),
            (6, Ok),
            (7, U8(0)),
            (8, Ok),
            (9, Ok),
            (10, Ok),
            (11, U8(1)),
            (12, Ok),
            (13, Ok),
            (14, Ok),
            (15, Ok),
            (16, Ok),
            (17, Ok),
            (18, Ok),
            (19, Ok),
            (20, Ok),
            (21, Ok),
        ],
        "ns::IReadOnlyApplicationControlDataInterface" => &[(0, U32(0)), (1, Ok)],
        "ns::IReadOnlyApplicationRecordInterface" => &[(0, U8(1)), (1, Ok), (2, U8(1))],
        "ns::IECommerceInterface" => &[(0, Ok), (1, Ok), (2, Ok), (3, Ok), (4, Ok), (5, Ok)],
        "ns::IApplicationVersionInterface" => &[
            (0, U32(0)),
            (1, Ok),
            (35, Ok),
            (36, Ok),
            (37, Ok),
            (40, Ok),
            (1000, Ok),
        ],
        "ns::IFactoryResetInterface" => &[],
        "ns::IAccountProxyInterface" => &[(0, Ok)],
        "ns::IApplicationManagerInterface" => &[
            (0, EmptyList), // ListApplicationRecord
            (21, Zeros(0)), // GetApplicationContentPath
            (400, U32(0)),  // GetApplicationControlData: no data
            (403, U32(0)),
            (404, Zeros(0x10)), // GetMaxApplicationControlCacheCount
            (1701, U32(0)),
        ],
        "ns::IDownloadTaskInterface" => &[(701, Ok), (702, Ok), (703, Ok), (704, Ok)],
        "ns::IContentManagementInterface" => {
            &[(11, Zeros(0x10)), (43, Ok), (47, U64(0)), (48, U64(0))]
        }
        "ns::IDocumentInterface" => &[(21, Zeros(0)), (23, Ok), (92, Zeros(0x10)), (93, Ok)],
        // Miscellaneous system services games open and rarely use.
        "mii:e" | "mii:u" => &[(0, Object("mii::IDatabaseService"))],
        "mii::IDatabaseService" => &[
            (0, U8(1)),     // IsUpdated
            (1, U8(0)),     // IsFullDatabase
            (2, U32(0)),    // GetCount
            (3, EmptyList), // Get
            (4, EmptyList), // Get1
            (5, Ok),
            (6, Zeros(0x58)), // BuildRandom
            (7, Zeros(0x58)), // BuildDefault
            (22, Ok),
        ],
        "lbl" => &[
            (17, Ok),
            (18, Ok),
            (19, Ok),
            (20, Ok),
            (21, Ok),
            (22, Ok),
            (23, Ok),
        ],
        "spsm" => &[(0, U32(0)), (1, Event(false)), (2, Ok), (3, Ok)],
        "idle:sys" => &[(0, Ok), (1, Ok), (2, Ok), (3, Ok)],
        "omm" => &[(0, U32(0)), (1, Event(false)), (5, U32(0)), (6, Zeros(8))],
        "fgm" | "fgm:0" | "fgm:9" => &[(0, Object("fgm::IRequest"))],
        "fgm::IRequest" => &[(0, Event(true)), (1, Ok), (2, U32(0)), (3, Ok)],
        "ntc" => &[
            (0, Object("ntc::IEnsureNetworkClockAvailabilityService")),
            (100, Ok),
            (101, Ok),
        ],
        "ntc::IEnsureNetworkClockAvailabilityService" => &[
            (0, Ok),
            (1, Event(true)),
            (2, Error(0x2E21)),
            (3, Ok),
            (4, U8(0)),
            (5, U64(0)),
        ],
        "mm:u" => &[
            (0, Ok),
            (1, Ok),
            (2, Ok),
            (3, Ok),
            (4, U32(1)),
            (5, Ok),
            (6, Ok),
            (7, U32(0)),
        ],
        "pm:shell" | "pm:bm" | "pm:info" | "pm:dmnt" => &[(0, U64(0)), (1, Ok), (2, Ok), (3, Ok)],
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every registered name and every interface a table hands out has a
    /// table (a missing one would panic when the game asks for it).
    #[test]
    fn tables_are_complete() {
        let mut todo: Vec<&str> = NAMES.to_vec();
        let mut seen = std::collections::HashSet::new();
        while let Some(name) = todo.pop() {
            if !seen.insert(name) {
                continue;
            }
            let t = table(name).unwrap_or_else(|| panic!("no table for {name}"));
            for (_, r) in t {
                if let Reply::Object(o) = r {
                    todo.push(o);
                }
            }
        }
    }
}
