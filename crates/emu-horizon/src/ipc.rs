//! IPC messages: the HIPC transport layer and the CMIF protocol on top.
//!
//! A client writes a request into the first 0x100 bytes of its thread
//! local storage and calls `svcSendSyncRequest`; the reply is written back
//! to the same place. Layouts follow the switchbrew wiki (IPC Marshalling)
//! and libnx's `sf/hipc.h` and `sf/cmif.h`:
//!
//! ```text
//! u32 header0: type, #send statics (X), #send buffers (A), #recv (B), #exch (W)
//! u32 header1: #data words, recv static mode, has special header
//! [special header: send pid, #copy handles, #move handles] [pid u64]
//! copy handles, move handles, X descriptors, A, B, W descriptors,
//! data words (CMIF payload starts 16-byte aligned), C recv list
//! ```

use emu_memory::{Fault, Memory};

pub const MESSAGE_SIZE: usize = 0x100;
pub const CMIF_IN_MAGIC: u32 = 0x4943_4653; // "SFCI"
pub const CMIF_OUT_MAGIC: u32 = 0x4F43_4653; // "SFCO"

pub const TYPE_CLOSE: u16 = 2;
pub const TYPE_REQUEST: u16 = 4;
pub const TYPE_CONTROL: u16 = 5;
pub const TYPE_REQUEST_WITH_CONTEXT: u16 = 6;
pub const TYPE_CONTROL_WITH_CONTEXT: u16 = 7;

pub const DOMAIN_SEND_MESSAGE: u8 = 1;
pub const DOMAIN_CLOSE: u8 = 2;

/// A memory region passed with a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Buffer {
    pub addr: u64,
    pub size: u64,
}

/// The domain part of a request to a domain session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DomainHeader {
    pub kind: u8,
    pub object_id: u32,
    pub in_objects: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    pub kind: u16,
    pub pid: Option<u64>,
    pub copy_handles: Vec<u32>,
    pub move_handles: Vec<u32>,
    /// X: send statics ("in pointers").
    pub x: Vec<Buffer>,
    /// A: send buffers.
    pub a: Vec<Buffer>,
    /// B: receive buffers.
    pub b: Vec<Buffer>,
    /// W: exchange buffers.
    pub w: Vec<Buffer>,
    /// C: receive list ("out pointers").
    pub c: Vec<Buffer>,
    pub domain: Option<DomainHeader>,
    /// CMIF command id (0 for close requests).
    pub command: u32,
    /// Raw input data after the CMIF header.
    pub data: Vec<u8>,
}

fn word(msg: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(msg[off..off + 4].try_into().unwrap())
}

fn buffer_descriptor(msg: &[u8], off: usize) -> Buffer {
    let (w0, w1, w2) = (word(msg, off), word(msg, off + 4), word(msg, off + 8));
    let addr = u64::from(w1)
        | (u64::from((w2 >> 28) & 0xF) << 32)
        | (u64::from((w2 >> 2) & 0x3F_FFFF) << 36);
    let size = u64::from(w0) | (u64::from((w2 >> 24) & 0xF) << 32);
    Buffer { addr, size }
}

fn static_descriptor(msg: &[u8], off: usize) -> Buffer {
    let (w0, w1) = (word(msg, off), word(msg, off + 4));
    let addr =
        u64::from(w1) | (u64::from((w0 >> 12) & 0xF) << 32) | (u64::from((w0 >> 6) & 0x3F) << 36);
    Buffer {
        addr,
        size: u64::from(w0 >> 16),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseError {
    /// The message does not fit in the command buffer.
    TooLarge,
    /// Wrong CMIF magic or malformed domain header.
    BadPayload,
}

/// Parses a request. `is_domain` tells whether the session is a domain,
/// which adds a domain header to requests.
pub fn parse_request(msg: &[u8; MESSAGE_SIZE], is_domain: bool) -> Result<Request, ParseError> {
    let h0 = word(msg, 0);
    let h1 = word(msg, 4);
    let kind = (h0 & 0xFFFF) as u16;
    let (n_x, n_a, n_b, n_w) = (
        ((h0 >> 16) & 0xF) as usize,
        ((h0 >> 20) & 0xF) as usize,
        ((h0 >> 24) & 0xF) as usize,
        (h0 >> 28) as usize,
    );
    let n_data = (h1 & 0x3FF) as usize;
    let recv_mode = (h1 >> 10) & 0xF;
    let n_c = match recv_mode {
        0 => 0,
        2 => 1, // one buffer for every out pointer
        m if m > 2 => (m - 2) as usize,
        _ => 0,
    };
    let mut off = 8;
    let mut req = Request {
        kind,
        pid: None,
        copy_handles: vec![],
        move_handles: vec![],
        x: vec![],
        a: vec![],
        b: vec![],
        w: vec![],
        c: vec![],
        domain: None,
        command: 0,
        data: vec![],
    };
    let check = |end: usize| {
        if end > MESSAGE_SIZE {
            Err(ParseError::TooLarge)
        } else {
            Ok(())
        }
    };
    if h1 >> 31 != 0 {
        let sp = word(msg, off);
        off += 4;
        if sp & 1 != 0 {
            check(off + 8)?;
            req.pid = Some(u64::from(word(msg, off)) | (u64::from(word(msg, off + 4)) << 32));
            off += 8;
        }
        let (n_copy, n_move) = (((sp >> 1) & 0xF) as usize, ((sp >> 5) & 0xF) as usize);
        check(off + 4 * (n_copy + n_move))?;
        for _ in 0..n_copy {
            req.copy_handles.push(word(msg, off));
            off += 4;
        }
        for _ in 0..n_move {
            req.move_handles.push(word(msg, off));
            off += 4;
        }
    }
    check(off + 8 * n_x + 12 * (n_a + n_b + n_w) + 4 * n_data + 8 * n_c)?;
    for _ in 0..n_x {
        req.x.push(static_descriptor(msg, off));
        off += 8;
    }
    for (count, which) in [(n_a, 0), (n_b, 1), (n_w, 2)] {
        for _ in 0..count {
            let b = buffer_descriptor(msg, off);
            match which {
                0 => req.a.push(b),
                1 => req.b.push(b),
                _ => req.w.push(b),
            }
            off += 12;
        }
    }
    let data_start = off;
    let data_end = off + 4 * n_data;
    for i in 0..n_c {
        let at = data_end + 8 * i;
        let (lo, hi) = (word(msg, at), word(msg, at + 4));
        req.c.push(Buffer {
            addr: u64::from(lo) | (u64::from(hi & 0xFFFF) << 32),
            size: u64::from(hi >> 16),
        });
    }

    if kind >= TYPE_TIPC_FIRST {
        // TIPC: the command is in the type field; the data words follow
        // the descriptors directly, without a CMIF header.
        req.command = u32::from(kind - TYPE_TIPC_FIRST);
        req.data = msg[data_start..data_end].to_vec();
        return Ok(req);
    }
    if kind == TYPE_CLOSE || n_data == 0 {
        return Ok(req);
    }
    // CMIF payload: 16-byte aligned from the start of the message.
    let mut p = (data_start + 15) & !15;
    let mut payload_end = data_end;
    if is_domain && (kind == TYPE_REQUEST || kind == TYPE_REQUEST_WITH_CONTEXT) {
        if p + 16 > data_end {
            return Err(ParseError::BadPayload);
        }
        let dkind = msg[p];
        let n_objects = msg[p + 1] as usize;
        let data_size = u16::from_le_bytes([msg[p + 2], msg[p + 3]]) as usize;
        let object_id = word(msg, p + 4);
        p += 16;
        let objects_at = p + data_size;
        if objects_at + 4 * n_objects > data_end {
            return Err(ParseError::BadPayload);
        }
        let in_objects = (0..n_objects)
            .map(|i| word(msg, objects_at + 4 * i))
            .collect();
        req.domain = Some(DomainHeader {
            kind: dkind,
            object_id,
            in_objects,
        });
        if dkind == DOMAIN_CLOSE {
            return Ok(req);
        }
        payload_end = objects_at;
    }
    if p + 16 > payload_end || word(msg, p) != CMIF_IN_MAGIC {
        return Err(ParseError::BadPayload);
    }
    req.command = word(msg, p + 8);
    req.data = msg[p + 16..payload_end].to_vec();
    Ok(req)
}

/// TIPC (the "tiny IPC" newer services use, Switch 2 graphics services
/// among them) puts command `n` in the message type as `16 + n`; 15
/// closes the session.
pub const TYPE_TIPC_FIRST: u16 = 16;
pub const TYPE_TIPC_CLOSE: u16 = 15;

/// Writes a TIPC reply: the result word and the data words right after
/// the header (and handles), no CMIF header. Returned interfaces must
/// already be in `move_handles`.
pub fn write_tipc_reply(reply: &Reply, kind: u16) -> [u8; MESSAGE_SIZE] {
    let mut msg = [0u8; MESSAGE_SIZE];
    let n_handles = reply.copy_handles.len() + reply.move_handles.len();
    let special = n_handles > 0;
    let mut off = 8;
    if special {
        let sp =
            ((reply.copy_handles.len() as u32) << 1) | ((reply.move_handles.len() as u32) << 5);
        msg[8..12].copy_from_slice(&sp.to_le_bytes());
        off = 12;
        for handle in reply.copy_handles.iter().chain(&reply.move_handles) {
            msg[off..off + 4].copy_from_slice(&handle.to_le_bytes());
            off += 4;
        }
    }
    let n_words = 1 + reply.data.len().div_ceil(4);
    msg[0..4].copy_from_slice(&u32::from(kind).to_le_bytes());
    let h1 = (n_words as u32) | if special { 1 << 31 } else { 0 };
    msg[4..8].copy_from_slice(&h1.to_le_bytes());
    msg[off..off + 4].copy_from_slice(&reply.result.to_le_bytes());
    off += 4;
    let end = (off + reply.data.len()).min(MESSAGE_SIZE);
    msg[off..end].copy_from_slice(&reply.data[..end - off]);
    msg
}

/// What a service answers.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Reply {
    pub result: u32,
    pub data: Vec<u8>,
    pub copy_handles: Vec<u32>,
    pub move_handles: Vec<u32>,
    /// Object ids, for replies on a domain session.
    pub objects: Vec<u32>,
}

/// Writes a CMIF reply. Out data starts 16-byte aligned, like requests;
/// on a domain session it is preceded by the domain out header and
/// followed by the object ids.
pub fn write_reply(reply: &Reply, is_domain: bool) -> [u8; MESSAGE_SIZE] {
    let mut msg = [0u8; MESSAGE_SIZE];
    let payload_len =
        if is_domain { 16 } else { 0 } + 16 + reply.data.len() + 4 * reply.objects.len();
    let n_handles = reply.copy_handles.len() + reply.move_handles.len();
    let special = n_handles > 0;
    let data_words_at = 8 + if special { 4 } else { 0 } + 4 * n_handles;
    let n_words = (16 + payload_len).div_ceil(4);
    let h1 = (n_words as u32) | if special { 1 << 31 } else { 0 };
    msg[4..8].copy_from_slice(&h1.to_le_bytes());
    if special {
        let sp =
            ((reply.copy_handles.len() as u32) << 1) | ((reply.move_handles.len() as u32) << 5);
        msg[8..12].copy_from_slice(&sp.to_le_bytes());
        let mut h = 12;
        for handle in reply.copy_handles.iter().chain(&reply.move_handles) {
            msg[h..h + 4].copy_from_slice(&handle.to_le_bytes());
            h += 4;
        }
    }
    let mut off = (data_words_at + 15) & !15;
    if is_domain {
        msg[off..off + 4].copy_from_slice(&(reply.objects.len() as u32).to_le_bytes());
        off += 16;
    }
    msg[off..off + 4].copy_from_slice(&CMIF_OUT_MAGIC.to_le_bytes());
    msg[off + 8..off + 12].copy_from_slice(&reply.result.to_le_bytes());
    off += 16;
    let end = (off + reply.data.len()).min(MESSAGE_SIZE);
    msg[off..end].copy_from_slice(&reply.data[..end - off]);
    off = end;
    if is_domain {
        for id in &reply.objects {
            msg[off..off + 4].copy_from_slice(&id.to_le_bytes());
            off += 4;
        }
    }
    msg
}

/// Reads the bytes of a buffer passed with a request.
pub fn read_buffer(mem: &Memory, b: Buffer) -> Result<Vec<u8>, Fault> {
    let mut v = vec![0u8; b.size as usize];
    mem.read(b.addr, &mut v)?;
    Ok(v)
}

#[cfg(test)]
mod tests {
    #[test]
    fn tipc_request_and_reply() {
        // TIPC command 204 with two copy handles and two data words.
        let mut msg = [0u8; MESSAGE_SIZE];
        let words: [u32; 7] = [16 + 204, (1 << 31) | 2, 2 << 1, 0x11, 0x22, 1, 0x40];
        for (i, w) in words.iter().enumerate() {
            msg[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
        }
        let req = parse_request(&msg, false).unwrap();
        assert_eq!(
            (req.command, req.copy_handles.clone()),
            (204, vec![0x11, 0x22])
        );
        assert_eq!(req.data, [1, 0, 0, 0, 0x40, 0, 0, 0]);
        // Reply: one copy handle, the result, then data.
        let reply = Reply {
            data: vec![7, 0, 0, 0],
            copy_handles: vec![0x33],
            ..Default::default()
        };
        let out = write_tipc_reply(&reply, 16 + 202);
        let w = |i: usize| u32::from_le_bytes(out[4 * i..4 * i + 4].try_into().unwrap());
        assert_eq!(w(1), (1 << 31) | 2);
        assert_eq!((w(2), w(3), w(4), w(5)), (1 << 1, 0x33, 0, 7));
    }

    use super::*;

    /// A request built by hand the way libnx's cmifMakeRequest lays it
    /// out: command 1 with 8 bytes of data, send pid, one B buffer.
    #[test]
    fn parses_cmif_request_with_pid_and_buffer() {
        let mut m = [0u8; MESSAGE_SIZE];
        let h0: u32 = TYPE_REQUEST as u32 | (1 << 24); // one B buffer
        let n_words = (16 + 16 + 8u32).div_ceil(4);
        let h1: u32 = n_words | (1 << 31);
        m[0..4].copy_from_slice(&h0.to_le_bytes());
        m[4..8].copy_from_slice(&h1.to_le_bytes());
        m[8..12].copy_from_slice(&1u32.to_le_bytes()); // send pid
                                                       // pid placeholder at 12..20, then the B descriptor at 20..32
        let addr: u64 = 0x12_3456_7000;
        let w2: u32 =
            ((((addr >> 36) as u32) & 0x3F_FFFF) << 2) | ((((addr >> 32) as u32) & 0xF) << 28);
        m[20..24].copy_from_slice(&0x40u32.to_le_bytes());
        m[24..28].copy_from_slice(&(addr as u32).to_le_bytes());
        m[28..32].copy_from_slice(&w2.to_le_bytes());
        // data words start at 32, which is already 16-byte aligned
        m[32..36].copy_from_slice(&CMIF_IN_MAGIC.to_le_bytes());
        m[40..44].copy_from_slice(&1u32.to_le_bytes());
        m[48..56].copy_from_slice(&0x003A_6D73_u64.to_le_bytes());
        let r = parse_request(&m, false).unwrap();
        assert_eq!(r.kind, TYPE_REQUEST);
        assert_eq!(r.pid, Some(0));
        assert_eq!(r.b, vec![Buffer { addr, size: 0x40 }]);
        assert_eq!(r.command, 1);
        assert_eq!(&r.data[..8], &0x003A_6D73_u64.to_le_bytes());
    }

    #[test]
    fn reply_layout_with_move_handle() {
        let reply = Reply {
            result: 0,
            data: vec![7, 0, 0, 0],
            move_handles: vec![0xABCD],
            ..Default::default()
        };
        let m = write_reply(&reply, false);
        assert_eq!(word(&m, 4) >> 31, 1, "special header");
        assert_eq!((word(&m, 8) >> 5) & 0xF, 1, "one move handle");
        assert_eq!(word(&m, 12), 0xABCD);
        // Data words start at 16; the CMIF header is 16-byte aligned.
        assert_eq!(word(&m, 16), CMIF_OUT_MAGIC);
        assert_eq!(word(&m, 24), 0, "result");
        assert_eq!(word(&m, 32), 7);
    }
}
