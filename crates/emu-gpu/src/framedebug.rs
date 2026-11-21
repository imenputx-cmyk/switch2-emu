//! Frame debugger: for chosen draws, a text report with every bit the GPU
//! saw (state, shader IR, constant buffers, vertex data) and a PNG of the
//! render target after the draw; PNGs of presented frames.

use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::backend::GuestMemory;
use crate::draw::Draw;

/// `--gpu-frames=DIR[,skip=N][,count=N]`.
#[derive(Debug, Clone)]
pub struct FrameConfig {
    pub dir: PathBuf,
    /// Draws skipped before the first report.
    pub skip: usize,
    /// Draws reported.
    pub count: usize,
}

pub static FRAMES: Mutex<Option<FrameConfig>> = Mutex::new(None);
static DRAWS: AtomicUsize = AtomicUsize::new(0);
static PRESENTS: AtomicUsize = AtomicUsize::new(0);

/// Parses `DIR[,skip=N][,count=N]`.
pub fn parse(spec: &str) -> FrameConfig {
    let mut cfg = FrameConfig {
        dir: PathBuf::new(),
        skip: 0,
        count: 50,
    };
    let mut dir = Vec::new();
    for part in spec.split(',') {
        if let Some(n) = part.strip_prefix("skip=") {
            cfg.skip = n.parse().unwrap_or(0);
        } else if let Some(n) = part.strip_prefix("count=") {
            cfg.count = n.parse().unwrap_or(50);
        } else {
            dir.push(part);
        }
    }
    cfg.dir = PathBuf::from(dir.join(","));
    cfg
}

/// The configuration, if frame dumping is on.
pub fn config() -> Option<FrameConfig> {
    FRAMES.lock().ok()?.clone()
}

/// Hex, binary and float view of `bytes`, one 32-bit word per line.
pub fn bit_dump(base: u64, bytes: &[u8]) -> String {
    let mut s = String::new();
    for (i, w) in bytes.chunks(4).enumerate() {
        let mut b = [0u8; 4];
        b[..w.len()].copy_from_slice(w);
        let v = u32::from_le_bytes(b);
        let _ = writeln!(
            s,
            "  {:#012x}  {v:#010x}  {:08b} {:08b} {:08b} {:08b}  u32 {v:<10}  i32 {:<11}  f32 {}",
            base + 4 * i as u64,
            (v >> 24) as u8,
            (v >> 16) as u8,
            (v >> 8) as u8,
            v as u8,
            v as i32,
            f32::from_bits(v)
        );
    }
    s
}

fn hex_row(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Writes an RGBA8 image as an (uncompressed) PNG.
pub fn write_png(
    path: &std::path::Path,
    width: u32,
    height: u32,
    rgba: &[u8],
) -> std::io::Result<()> {
    fn crc(data: &[u8]) -> u32 {
        let mut c = !0u32;
        for &b in data {
            c ^= u32::from(b);
            for _ in 0..8 {
                c = if c & 1 != 0 {
                    0xEDB8_8320 ^ (c >> 1)
                } else {
                    c >> 1
                };
            }
        }
        !c
    }
    fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
        out.extend((data.len() as u32).to_be_bytes());
        let mut body = kind.to_vec();
        body.extend_from_slice(data);
        out.extend_from_slice(&body);
        out.extend(crc(&body).to_be_bytes());
    }
    let mut raw = Vec::with_capacity((width as usize * 4 + 1) * height as usize);
    for row in rgba.chunks(width as usize * 4).take(height as usize) {
        raw.push(0);
        raw.extend_from_slice(row);
    }
    // zlib, stored blocks.
    let mut z = vec![0x78, 0x01];
    let mut blocks = raw.chunks(65_535).peekable();
    while let Some(b) = blocks.next() {
        z.push(u8::from(blocks.peek().is_none()));
        z.extend((b.len() as u16).to_le_bytes());
        z.extend((!(b.len() as u16)).to_le_bytes());
        z.extend_from_slice(b);
    }
    let (mut a, mut s) = (1u32, 0u32);
    for &b in &raw {
        a = (a + u32::from(b)) % 65_521;
        s = (s + a) % 65_521;
    }
    z.extend(((s << 16) | a).to_be_bytes());
    let mut out = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    let mut ihdr = Vec::new();
    ihdr.extend(width.to_be_bytes());
    ihdr.extend(height.to_be_bytes());
    ihdr.extend([8, 6, 0, 0, 0]);
    chunk(&mut out, b"IHDR", &ihdr);
    chunk(&mut out, b"IDAT", &z);
    chunk(&mut out, b"IEND", &[]);
    std::fs::write(path, out)
}

/// A presented frame (RGBA8): the first three, then every 60th, up to
/// the draw count's worth of files.
pub fn present(width: u32, height: u32, rgba: &[u8]) {
    let Some(cfg) = config() else { return };
    let n = PRESENTS.fetch_add(1, Ordering::Relaxed) + 1;
    if (n <= 3 || n.is_multiple_of(60)) && n <= 3 + 20 * 60 {
        let _ = std::fs::create_dir_all(&cfg.dir);
        let _ = write_png(
            &cfg.dir.join(format!("present-{n:05}.png")),
            width,
            height,
            rgba,
        );
    }
}

/// Reports one draw (after the backend ran it).
pub fn draw(g: &GuestMemory, d: &Draw, shader_words: &[(String, Vec<u8>)]) {
    let Some(cfg) = config() else { return };
    let n = DRAWS.fetch_add(1, Ordering::Relaxed);
    if n < cfg.skip || n >= cfg.skip + cfg.count {
        return;
    }
    let _ = std::fs::create_dir_all(&cfg.dir);
    let mut s = String::new();
    let _ = writeln!(s, "=== draw #{n} ===");
    let _ = writeln!(
        s,
        "topology {}  ranges {:?}  index {:?}",
        d.topology, d.ranges, d.index
    );
    let _ = writeln!(s, "viewport {:?}", d.viewport);
    let _ = writeln!(s, "clip {:?}", d.clip);
    let _ = writeln!(s, "depth {:?}", d.depth);
    let _ = writeln!(s, "blend {:?}  constant {:?}", d.blend, d.blend_constant);
    let _ = writeln!(s, "cull {:?}", d.cull);
    let _ = writeln!(s, "ps header {:?}", d.ps_header);
    for (k, t) in d.color_targets() {
        let _ = writeln!(s, "color target (ps output {k}) {t:?}");
    }
    let _ = writeln!(s, "\n-- vertex attributes (used) --");
    for (i, a) in d.attributes.iter().enumerate() {
        if a.constant || a.widths != 0 {
            let _ = writeln!(s, "  attr {i}: {a:?}");
        }
    }
    let _ = writeln!(s, "-- vertex streams (enabled) --");
    for (i, st) in d.streams.iter().enumerate() {
        if st.enabled {
            let _ = writeln!(s, "  stream {i}: {st:?}");
        }
    }
    for (name, words) in shader_words {
        let _ = writeln!(
            s,
            "\n-- {name}: raw program ({} bytes, 128-bit instructions) --",
            words.len()
        );
        for (i, ins) in words.chunks(16).enumerate() {
            let w: Vec<u32> = ins
                .chunks(4)
                .map(|c| {
                    u32::from_le_bytes([
                        c[0],
                        c[1],
                        c.get(2).copied().unwrap_or(0),
                        c.get(3).copied().unwrap_or(0),
                    ])
                })
                .collect();
            let _ = writeln!(
                s,
                "  {i:04}: {}   opcode {:#05x}",
                w.iter()
                    .map(|x| format!("{x:08x}"))
                    .collect::<Vec<_>>()
                    .join(" "),
                w.first().copied().unwrap_or(0) & 0xFFF
            );
            let _ = writeln!(
                s,
                "        {}",
                w.iter()
                    .map(|x| format!("{x:032b}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
    }
    for (name, prog) in [("vertex", &d.vs), ("pixel", &d.ps)] {
        let _ = writeln!(s, "\n-- {name} shader IR ({} ops) --", prog.len());
        for (i, inst) in prog.iter().enumerate() {
            let _ = writeln!(s, "  {i:04}: {inst:?}");
        }
    }
    for (name, cbufs) in [("vertex", &d.vs_cbufs), ("pixel", &d.ps_cbufs)] {
        for (slot, c) in cbufs.iter().enumerate() {
            let Some(c) = c else { continue };
            let _ = writeln!(s, "\n-- {name} constant buffer {slot}: {c:?} --");
            let len = (c.size as usize).min(256);
            let mut buf = vec![0u8; len];
            match g.read(c.addr, &mut buf) {
                Ok(()) => s += &bit_dump(c.addr, &buf),
                Err(e) => {
                    let _ = writeln!(s, "  unreadable: {e:?}");
                }
            }
        }
    }
    if let Some(t) = &d.root_tables {
        let _ = writeln!(s, "\n-- NVN2 root tables ({} bytes, first 512) --", t.len());
        s += &bit_dump(0, &t[..t.len().min(512)]);
    }
    s += &crate::draw::describe_textures(g, d);
    // First vertices of every enabled stream, and the first indices.
    let first = d.ranges.first().copied().unwrap_or((0, 0));
    if let Some(ix) = &d.index {
        let len = (ix.size * first.1.min(24)) as usize;
        let mut buf = vec![0u8; len];
        if g.read(ix.addr + u64::from(first.0 * ix.size), &mut buf)
            .is_ok()
        {
            let _ = writeln!(
                s,
                "\n-- first {} indices (size {}) --",
                first.1.min(24),
                ix.size
            );
            s += &bit_dump(ix.addr, &buf);
        }
    }
    for (i, st) in d.streams.iter().enumerate().filter(|(_, st)| st.enabled) {
        let len = (st.stride as usize * 4).clamp(4, 512);
        let mut buf = vec![0u8; len];
        if g.read(
            st.addr + u64::from(first.0) * u64::from(st.stride),
            &mut buf,
        )
        .is_ok()
        {
            let _ = writeln!(
                s,
                "\n-- stream {i}: first vertices (stride {}) --",
                st.stride
            );
            for (v, row) in buf.chunks(st.stride.max(1) as usize).enumerate() {
                let _ = writeln!(s, "  vertex {v}: {}", hex_row(row));
            }
            s += &bit_dump(st.addr, &buf[..buf.len().min(64)]);
        }
    }
    // The render target after the draw.
    if let Some(t) = d.target {
        match t.read_linear(g) {
            Ok(px) => {
                let bpp = t.format.bytes_per_pixel() as usize;
                let mut rgba = Vec::with_capacity(px.len() / bpp * 4);
                let (mut lit, mut lo, mut hi) = (0usize, [f32::MAX; 4], [f32::MIN; 4]);
                for p in px.chunks_exact(bpp) {
                    let c = t.format.decode(p);
                    if c[0] != 0.0 || c[1] != 0.0 || c[2] != 0.0 {
                        lit += 1;
                    }
                    for k in 0..4 {
                        lo[k] = lo[k].min(c[k]);
                        hi[k] = hi[k].max(c[k]);
                    }
                    rgba.extend(c.map(|v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8));
                }
                let total = px.len() / bpp;
                let _ = writeln!(
                    s,
                    "\n-- render target 0 after the draw: {}x{}, {lit}/{total} pixels not black, min {lo:?}, max {hi:?} --",
                    t.width, t.height
                );
                let _ = writeln!(s, "first pixels (raw bytes):");
                for (i, p) in px.chunks_exact(bpp).take(8).enumerate() {
                    let _ = writeln!(s, "  px {i}: {}", hex_row(p));
                }
                let _ = write_png(
                    &cfg.dir.join(format!("draw-{n:06}-rt0.png")),
                    t.width,
                    t.height,
                    &rgba,
                );
            }
            Err(e) => {
                let _ = writeln!(s, "\n-- render target 0 unreadable: {e:?} --");
            }
        }
    }
    let _ = std::fs::write(cfg.dir.join(format!("draw-{n:06}.txt")), s);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_dump_shows_binary_and_float() {
        let s = bit_dump(0x1000, &0x3f80_0000u32.to_le_bytes());
        assert!(s.contains("00111111 10000000 00000000 00000000"));
        assert!(s.contains("f32 1"));
    }

    #[test]
    fn png_has_signature_and_end() {
        let p = std::env::temp_dir().join("fd-test.png");
        write_png(&p, 2, 2, &[255; 16]).unwrap();
        let b = std::fs::read(&p).unwrap();
        assert_eq!(&b[..8], &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        assert_eq!(&b[b.len() - 8..b.len() - 4], b"IEND");
    }

    #[test]
    fn parses_options() {
        let c = parse("C:\\x\\frames,skip=10,count=5");
        assert_eq!((c.skip, c.count), (10, 5));
        assert_eq!(c.dir, PathBuf::from("C:\\x\\frames"));
    }
}
