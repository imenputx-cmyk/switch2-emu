//! The audio renderer's voices: parsing RequestUpdate input, decoding
//! wave buffers (PCM16, PCM float, ADPCM), resampling and mixing to
//! stereo.
//!
//! Ported (simplified) from Lichii/Eden audio_core (GPL-3.0-or-later):
//! `renderer/behavior/info_updater.cpp` (input layout),
//! `renderer/voice/voice_info.cpp` (wave buffer and play state
//! bookkeeping) and `renderer/command/data_source/decode.cpp` (decoding
//! and wave buffer consumption). Effects, sub-mixes and splitter volumes
//! are not applied: each voice channel goes to the output through its
//! channel resource's mix volumes (buffers 0..5 as FL, FR, FC, LFE, RL,
//! RR), or straight to the speakers when routed through a splitter.

/// Wave buffers per voice.
const WAVE_BUFFERS: usize = 4;
/// Channels per voice.
const MAX_CHANNELS: usize = 6;
/// Mix buffers per channel resource.
const MIX_BUFFERS: usize = 24;
const UNUSED_SPLITTER: u32 = u32::MAX;
const UNUSED_MIX: u32 = i32::MAX as u32;

mod format {
    pub const PCM16: u8 = 2;
    pub const PCM_FLOAT: u8 = 5;
    pub const ADPCM: u8 = 6;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum Play {
    Started,
    #[default]
    Stopped,
    RequestStop,
    Paused,
}

#[derive(Debug, Clone, Copy)]
struct WaveBuffer {
    address: u64,
    size: u64,
    start: i32,
    end: i32,
    looping: bool,
    stream_ended: bool,
    sent_to_dsp: bool,
    loop_count: i32,
    context: u64,
    context_size: u64,
    loop_start: u32,
    loop_end: u32,
}

impl Default for WaveBuffer {
    /// Empty, and counted as already handed over (Eden).
    fn default() -> Self {
        WaveBuffer {
            address: 0,
            size: 0,
            start: 0,
            end: 0,
            looping: false,
            stream_ended: false,
            sent_to_dsp: true,
            loop_count: 0,
            context: 0,
            context_size: 0,
            loop_start: 0,
            loop_end: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct Adpcm {
    header: u16,
    yn0: i16,
    yn1: i16,
}

/// Per channel resource: decoding state (Eden VoiceState).
#[derive(Debug, Clone, Default)]
struct ChannelState {
    valid: [bool; WAVE_BUFFERS],
    index: usize,
    consumed: u32,
    played: u64,
    offset: u32,
    fraction: f64,
    loop_count: i32,
    adpcm: Adpcm,
    /// Decoded samples from the current resampling position on (kept
    /// across frames for interpolation).
    carry: Vec<f32>,
}

#[derive(Debug, Clone, Copy, Default)]
struct Voice {
    in_use: bool,
    is_new: bool,
    play: Play,
    format: u8,
    sample_rate: u32,
    channels: usize,
    pitch: f32,
    volume: f32,
    wave_buffer_count: u32,
    data_address: u64,
    data_size: u64,
    mix_id: u32,
    splitter_id: u32,
    buffers: [WaveBuffer; WAVE_BUFFERS],
    resources: [u32; MAX_CHANNELS],
    flush: u32,
}

/// A mix (Eden MixInfo): its buffers go to a destination mix through
/// a volume matrix, or through a splitter.
#[derive(Debug, Clone)]
struct Mix {
    in_use: bool,
    volume: f32,
    buffers: usize,
    dest_mix: u32,
    dest_splitter: u32,
    /// `matrix[i * MIX_BUFFERS + j]`: from buffer i to destination j.
    matrix: Vec<f32>,
}

/// A splitter destination (Eden SplitterDestinationData).
#[derive(Debug, Clone, Copy)]
struct Destination {
    in_use: bool,
    mix_id: u32,
    volumes: [f32; MIX_BUFFERS],
}

/// Mix id of the final mix.
const FINAL_MIX: usize = 0;

/// Reads guest memory; false if unmapped.
pub type ReadFn<'a> = dyn Fn(u64, &mut [u8]) -> bool + 'a;

/// Voices, channel states and mix volumes of one renderer.
#[derive(Debug, Default)]
pub struct Voices {
    voices: Vec<Voice>,
    states: Vec<ChannelState>,
    volumes: Vec<[f32; MIX_BUFFERS]>,
    mixes: Vec<Mix>,
    /// Destination ids of each splitter.
    splitters: Vec<Vec<u32>>,
    destinations: Vec<Destination>,
    effects: Vec<super::audfx::Effect>,
    /// Output sample rate (for effects).
    rate: u32,
}

fn u32_at(b: &[u8], o: usize) -> u32 {
    b.get(o..o + 4)
        .map_or(0, |s| u32::from_le_bytes(s.try_into().unwrap()))
}

fn u64_at(b: &[u8], o: usize) -> u64 {
    b.get(o..o + 8)
        .map_or(0, |s| u64::from_le_bytes(s.try_into().unwrap()))
}

fn f32_at(b: &[u8], o: usize) -> f32 {
    f32::from_bits(u32_at(b, o))
}

fn byte(b: &[u8], o: usize) -> u8 {
    b.get(o).copied().unwrap_or(0)
}

impl Voices {
    pub fn new(count: u32) -> Voices {
        let n = count as usize;
        Voices {
            voices: vec![Voice::default(); n],
            states: vec![ChannelState::default(); n],
            volumes: vec![[0.0; MIX_BUFFERS]; n],
            mixes: Vec::new(),
            splitters: Vec::new(),
            destinations: Vec::new(),
            effects: Vec::new(),
            rate: 48_000,
        }
    }

    /// Applies the effect section (0xC0 bytes per effect).
    pub fn update_effects(&mut self, data: &[u8], count: u32) {
        let count = count as usize;
        if self.effects.len() < count {
            self.effects.resize_with(count, super::audfx::Effect::new);
        }
        for (e, p) in self.effects.iter_mut().zip(data.chunks_exact(0xC0)) {
            e.update(p);
        }
    }

    /// Runs mix `m`'s effects on its buffers, in process order.
    fn apply_effects(&mut self, m: usize, bufs: &mut [Vec<f32>]) {
        let mut order: Vec<usize> = (0..self.effects.len())
            .filter(|&i| self.effects[i].mix_id as usize == m)
            .collect();
        order.sort_by_key(|&i| self.effects[i].order);
        let rate = self.rate;
        for i in order {
            self.effects[i].apply(bufs, rate);
        }
    }

    /// Applies the mix section (MixInfo::InParameter, 0x930 bytes each,
    /// optionally after a 0x20-byte dirty-only header) for `count` mixes.
    pub fn update_mixes(&mut self, data: &[u8], count: u32) {
        let count = count as usize;
        if self.mixes.len() < count {
            self.mixes.resize(
                count,
                Mix {
                    in_use: false,
                    volume: 0.0,
                    buffers: 0,
                    dest_mix: UNUSED_MIX,
                    dest_splitter: UNUSED_SPLITTER,
                    matrix: vec![0.0; MIX_BUFFERS * MIX_BUFFERS],
                },
            );
        }
        let (start, entries, dirty) = if data.len() == count * 0x930 {
            (0, count, false)
        } else if data.len() >= 0x20 {
            let n = (u32_at(data, 4) as usize).min((data.len() - 0x20) / 0x930);
            (0x20, n, true)
        } else {
            return;
        };
        for i in 0..entries {
            let p = &data[start + i * 0x930..start + (i + 1) * 0x930];
            let id = if dirty { u32_at(p, 0x10) as usize } else { i };
            let Some(m) = self.mixes.get_mut(id) else {
                continue;
            };
            m.in_use = byte(p, 0xC) != 0;
            if !m.in_use {
                continue;
            }
            m.volume = f32_at(p, 0);
            m.buffers = (u32_at(p, 8) as usize).min(MIX_BUFFERS);
            for (k, v) in m.matrix.iter_mut().enumerate() {
                *v = f32_at(p, 0x24 + 4 * k);
            }
            m.dest_mix = u32_at(p, 0x924);
            m.dest_splitter = u32_at(p, 0x928);
        }
    }

    /// Applies the splitter section ('SNDH' header, 'SNDI' infos with
    /// their destination ids, 'SNDD' destinations of 0x70, 0x90 or 0xA8
    /// bytes by revision).
    pub fn update_splitters(&mut self, data: &[u8]) {
        if data.len() < 0x20 || &data[..4] != b"SNDH" {
            return;
        }
        let infos = u32_at(data, 4) as usize;
        let dests = u32_at(data, 8) as usize;
        let mut at = 0x20;
        for _ in 0..infos {
            if data.get(at..at + 4) != Some(b"SNDI") {
                break;
            }
            let id = u32_at(data, at + 4) as usize;
            let n = (u32_at(data, at + 0xC) as usize).min(256);
            let ids: Vec<u32> = (0..n).map(|k| u32_at(data, at + 0x10 + 4 * k)).collect();
            if id < 4096 {
                if self.splitters.len() <= id {
                    self.splitters.resize(id + 1, Vec::new());
                }
                self.splitters[id] = ids;
            }
            at += 0x1C + 4 * n;
        }
        if dests == 0 {
            return;
        }
        let stride = [0xA8, 0x90, 0x70]
            .into_iter()
            .find(|&s| {
                at + s * dests <= data.len() && {
                    // The next destination starts right after this one.
                    dests == 1 || data.get(at + s..at + s + 4) == Some(b"SNDD")
                }
            })
            .unwrap_or(0x70);
        let in_use_at = match stride {
            0xA8 => 0x9C,
            0x90 => 0x84,
            _ => 0x6C,
        };
        for k in 0..dests {
            let p = at + k * stride;
            if data.get(p..p + 4) != Some(b"SNDD") {
                continue;
            }
            let id = u32_at(data, p + 4) as usize;
            if id >= 1 << 16 {
                continue;
            }
            if self.destinations.len() <= id {
                self.destinations.resize(
                    id + 1,
                    Destination {
                        in_use: false,
                        mix_id: UNUSED_MIX,
                        volumes: [0.0; MIX_BUFFERS],
                    },
                );
            }
            let d = &mut self.destinations[id];
            d.in_use = byte(data, p + in_use_at) != 0;
            d.mix_id = u32_at(data, p + 0x68);
            for (j, v) in d.volumes.iter_mut().enumerate() {
                *v = f32_at(data, p + 8 + 4 * j);
            }
        }
    }

    /// Destination `index` of splitter `splitter`, if configured.
    fn destination(&self, splitter: u32, index: usize) -> Option<Option<&Destination>> {
        let ids = self.splitters.get(splitter as usize)?;
        let id = *ids.get(index)?;
        Some(
            self.destinations
                .get(id as usize)
                .filter(|d| d.in_use && (d.mix_id as usize) < self.mixes.len()),
        )
    }

    /// Steps from mix `m` to the final mix (for processing order).
    fn distance(&self, m: usize) -> usize {
        let mut d = 0;
        let mut cur = m;
        while cur != FINAL_MIX && d < self.mixes.len() {
            let mix = &self.mixes[cur];
            cur = if (mix.dest_mix as usize) < self.mixes.len() {
                mix.dest_mix as usize
            } else if mix.dest_splitter != UNUSED_SPLITTER {
                match self.destination(mix.dest_splitter, 0) {
                    Some(Some(dst)) => dst.mix_id as usize,
                    _ => return d + 1,
                }
            } else {
                return d + 1;
            };
            d += 1;
        }
        d
    }

    /// Applies the channel resource and voice sections of an update
    /// (`resources` and `voices` are their input bytes) and returns the
    /// voices' output statuses (0x10 bytes each, in input order).
    pub fn update(&mut self, resources: &[u8], voices: &[u8]) -> Vec<u8> {
        let n = self.voices.len();
        for r in resources.chunks_exact(0x70).take(n) {
            if byte(r, 0x64) != 0 {
                let id = (u32_at(r, 0) as usize).min(n - 1);
                for (k, v) in self.volumes[id].iter_mut().enumerate() {
                    *v = f32_at(r, 4 + 4 * k);
                }
            }
        }
        for v in &mut self.voices {
            v.in_use = false;
        }
        let mut out = vec![0u8; n * 0x10];
        if n == 0 {
            return out;
        }
        // Two layouts: 0x170 bytes, or 0x188 (float biquads, REV15+).
        let stride = voices.len() / n;
        let v2 = stride >= 0x188;
        if stride < 0x170 {
            return out;
        }
        for i in 0..n {
            let p = &voices[i * stride..(i + 1) * stride];
            if byte(p, 9) == 0 {
                continue;
            }
            let id = u32_at(p, 0) as usize;
            if id >= n {
                continue;
            }
            let (wb_count, data, mix, wbs, res, flush) = if v2 {
                (0x54, 0x60, 0x70, 0x78, 0x158, 0x171)
            } else {
                (0x3C, 0x48, 0x58, 0x60, 0x140, 0x159)
            };
            let channels = (u32_at(p, 0x18) as usize).clamp(1, MAX_CHANNELS);
            let is_new = byte(p, 8) != 0;
            let mut resources = [0u32; MAX_CHANNELS];
            for (c, r) in resources.iter_mut().enumerate() {
                *r = u32_at(p, res + 4 * c).min(n as u32 - 1);
            }
            let voice = &mut self.voices[id];
            if is_new {
                *voice = Voice {
                    is_new: true,
                    ..Voice::default()
                };
                for &r in &resources[..channels] {
                    self.states[r as usize] = ChannelState::default();
                }
            }
            voice.in_use = true;
            voice.format = byte(p, 0xB);
            voice.sample_rate = u32_at(p, 0xC);
            voice.channels = channels;
            voice.pitch = f32_at(p, 0x1C);
            voice.volume = f32_at(p, 0x20);
            voice.wave_buffer_count = u32_at(p, wb_count);
            voice.data_address = u64_at(p, data);
            voice.data_size = u64_at(p, data + 8);
            voice.mix_id = u32_at(p, mix);
            voice.splitter_id = u32_at(p, mix + 4);
            voice.resources = resources;
            voice.flush += u32::from(byte(p, flush));
            voice.play = match (byte(p, 0xA), voice.play) {
                (0, _) => Play::Started,
                (1, Play::Stopped) => Play::Stopped,
                (1, _) => Play::RequestStop,
                (2, _) => Play::Paused,
                (_, s) => s,
            };
            for k in 0..WAVE_BUFFERS {
                let w = &p[wbs + 0x38 * k..wbs + 0x38 * (k + 1)];
                // Only buffers the game has not handed over yet change.
                if byte(w, 0x1A) != 0 {
                    continue;
                }
                voice.buffers[k] = WaveBuffer {
                    address: u64_at(w, 0),
                    size: u64_at(w, 8),
                    start: u32_at(w, 0x10) as i32,
                    end: u32_at(w, 0x14) as i32,
                    looping: byte(w, 0x18) != 0,
                    stream_ended: byte(w, 0x19) != 0,
                    sent_to_dsp: false,
                    loop_count: u32_at(w, 0x1C) as i32,
                    context: u64_at(w, 0x20),
                    context_size: u64_at(w, 0x28),
                    loop_start: u32_at(w, 0x30),
                    loop_end: u32_at(w, 0x34),
                };
            }
            // Output status (Eden WriteOutStatus).
            let o = &mut out[i * 0x10..(i + 1) * 0x10];
            if !voice.is_new {
                let s = &self.states[resources[0] as usize];
                o[..8].copy_from_slice(&s.played.to_le_bytes());
                o[8..12].copy_from_slice(&s.consumed.to_le_bytes());
            }
        }
        out
    }

    /// Renders `frames` stereo frames at `rate` Hz (interleaved L, R).
    pub fn render(&mut self, frames: usize, rate: u32, read: &ReadFn) -> Vec<f32> {
        self.rate = rate;
        let mixing = self.mixes.get(FINAL_MIX).is_some_and(|m| m.in_use);
        // Each mix's buffers (Eden mix buffers, kept per mix).
        let mut bufs: Vec<Vec<Vec<f32>>> = self
            .mixes
            .iter()
            .map(|m| vec![vec![0f32; frames]; if m.in_use { m.buffers } else { 0 }])
            .collect();
        let mut direct = vec![0f32; frames * 2];
        let mut buf = vec![0f32; frames];
        for vi in 0..self.voices.len() {
            if !self.voices[vi].in_use {
                continue;
            }
            if !self.prepare(vi) {
                continue;
            }
            let v = self.voices[vi];
            if v.wave_buffer_count == 0 || v.play != Play::Started {
                continue;
            }
            for c in 0..v.channels {
                let r = v.resources[c] as usize;
                buf.iter_mut().for_each(|s| *s = 0.0);
                decode(&v, c, &mut self.states[r], rate, &mut buf, read);
                let g = v.volume;
                // Eden GenerateVoiceMixCommand: into the voice's mix with
                // its channel volumes, or into each splitter
                // destination (c, c + channels, ...) with its volumes.
                let mut sends: Vec<(usize, [f32; MIX_BUFFERS])> = Vec::new();
                if mixing && (v.mix_id as usize) < self.mixes.len() {
                    sends.push((v.mix_id as usize, self.volumes[r]));
                } else if mixing && v.splitter_id != UNUSED_SPLITTER {
                    let mut i = c;
                    while let Some(d) = self.destination(v.splitter_id, i) {
                        if let Some(d) = d {
                            sends.push((d.mix_id as usize, d.volumes));
                        }
                        i += v.channels;
                    }
                } else if v.mix_id != UNUSED_MIX || v.splitter_id != UNUSED_SPLITTER {
                    // No mixes known: straight to the speakers.
                    let vol = &self.volumes[r];
                    let (l, rr) = (
                        vol[0] + 0.707 * vol[2] + vol[4],
                        vol[1] + 0.707 * vol[2] + vol[5],
                    );
                    for (f, &s) in buf.iter().enumerate() {
                        direct[2 * f] += s * g * l;
                        direct[2 * f + 1] += s * g * rr;
                    }
                }
                for (m, vol) in sends {
                    for (j, dst) in bufs[m].iter_mut().enumerate() {
                        let k = g * vol[j];
                        if k != 0.0 {
                            for (o, &s) in dst.iter_mut().zip(&buf) {
                                *o += s * k;
                            }
                        }
                    }
                }
            }
            self.voices[vi].is_new = false;
        }
        if !mixing {
            return direct;
        }
        // Sub-mixes, farthest from the final mix first (Eden SortInfo,
        // GenerateMixCommands).
        let mut order: Vec<usize> = (0..self.mixes.len())
            .filter(|&m| m != FINAL_MIX && self.mixes[m].in_use)
            .collect();
        order.sort_by_key(|&m| std::cmp::Reverse(self.distance(m)));
        for m in order {
            let mix = self.mixes[m].clone();
            let mut src = std::mem::take(&mut bufs[m]);
            self.apply_effects(m, &mut src);
            if (mix.dest_mix as usize) < self.mixes.len() {
                let dm = mix.dest_mix as usize;
                for (i, s) in src.iter().enumerate() {
                    for (j, dst) in bufs[dm].iter_mut().enumerate() {
                        let k = mix.volume * mix.matrix[i * MIX_BUFFERS + j];
                        if k != 0.0 {
                            for (o, &x) in dst.iter_mut().zip(s) {
                                *o += x * k;
                            }
                        }
                    }
                }
            } else if mix.dest_splitter != UNUSED_SPLITTER && mix.buffers > 0 {
                let mut dest_id = 0;
                while let Some(d) = self.destination(mix.dest_splitter, dest_id) {
                    if let Some(d) = d {
                        let input = &src[dest_id % mix.buffers];
                        let (dm, vols) = (d.mix_id as usize, d.volumes);
                        for (j, dst) in bufs[dm].iter_mut().enumerate() {
                            let k = mix.volume * vols[j];
                            if k != 0.0 {
                                for (o, &x) in dst.iter_mut().zip(input) {
                                    *o += x * k;
                                }
                            }
                        }
                    }
                    dest_id += 1;
                }
            }
        }
        // The final mix to stereo: FL, FR, FC, LFE, RL, RR.
        let mut fb = std::mem::take(&mut bufs[FINAL_MIX]);
        self.apply_effects(FINAL_MIX, &mut fb);
        bufs[FINAL_MIX] = fb;
        let fin = &self.mixes[FINAL_MIX];
        let b = &bufs[FINAL_MIX];
        let ch = |k: usize, f: usize| b.get(k).map_or(0.0, |x| x[f]);
        let mut out = direct;
        for f in 0..frames {
            let (c, l, r) = (0.707 * ch(2, f), ch(0, f) + ch(4, f), ch(1, f) + ch(5, f));
            let mono = b.len() == 1;
            out[2 * f] += fin.volume * if mono { ch(0, f) } else { l + c };
            out[2 * f + 1] += fin.volume * if mono { ch(0, f) } else { r + c };
        }
        out
    }

    /// Eden UpdateParametersForCommandGeneration: hands wave buffers to
    /// the "DSP" and handles flushes and stop requests. Whether the
    /// voice plays this frame.
    fn prepare(&mut self, vi: usize) -> bool {
        let v = &mut self.voices[vi];
        let chans: Vec<usize> = v.resources[..v.channels]
            .iter()
            .map(|&r| r as usize)
            .collect();
        if v.is_new {
            for &r in &chans {
                self.states[r] = ChannelState::default();
            }
        }
        while v.flush > 0 {
            v.flush -= 1;
            let k = self.states[chans[0]].index;
            v.buffers[k].sent_to_dsp = true;
            for &r in &chans {
                let s = &mut self.states[r];
                if s.index == k {
                    s.index = (s.index + 1) % WAVE_BUFFERS;
                    s.consumed += 1;
                }
                s.valid[k] = false;
            }
        }
        match v.play {
            Play::Started => {
                for k in 0..WAVE_BUFFERS {
                    if !v.buffers[k].sent_to_dsp {
                        for &r in &chans {
                            self.states[r].valid[k] = true;
                        }
                        v.buffers[k].sent_to_dsp = true;
                    }
                }
                self.states[chans[0]].valid.iter().any(|&b| b)
            }
            Play::RequestStop => {
                for k in 0..WAVE_BUFFERS {
                    v.buffers[k].sent_to_dsp = true;
                    for &r in &chans {
                        let s = &mut self.states[r];
                        if s.valid[k] {
                            s.index = (s.index + 1) % WAVE_BUFFERS;
                            s.consumed += 1;
                        }
                        s.valid[k] = false;
                    }
                }
                for &r in &chans {
                    let s = &mut self.states[r];
                    s.offset = 0;
                    s.played = 0;
                    s.adpcm = Adpcm::default();
                    s.fraction = 0.0;
                    s.carry.clear();
                }
                v.play = Play::Stopped;
                false
            }
            _ => false,
        }
    }
}

/// Decodes up to `want` source samples of channel `channel` from wave
/// buffer `w` at sample `offset` past its start; returns them.
fn decode_buffer(
    v: &Voice,
    channel: usize,
    w: &WaveBuffer,
    (start, end): (i32, i32),
    (offset, want): (u32, u32),
    adpcm: &mut Adpcm,
    read: &ReadFn,
) -> Vec<f32> {
    if w.address == 0 || w.size == 0 || start >= end {
        return Vec::new();
    }
    let pos = start as u64 + u64::from(offset);
    let n = want.min((end as u64).saturating_sub(pos) as u32) as usize;
    if n == 0 {
        return Vec::new();
    }
    let ch = v.channels as u64;
    match v.format {
        format::PCM16 | format::PCM_FLOAT => {
            let size = if v.format == format::PCM16 { 2 } else { 4 };
            let mut raw = vec![0u8; n * v.channels * size];
            if !read(w.address + pos * ch * size as u64, &mut raw) {
                return Vec::new();
            }
            raw.chunks_exact(size * v.channels)
                .map(|frame| {
                    let s = &frame[channel * size..(channel + 1) * size];
                    if size == 2 {
                        f32::from(i16::from_le_bytes([s[0], s[1]])) / 32768.0
                    } else {
                        f32::from_le_bytes(s.try_into().unwrap()).clamp(-1.0, 1.0)
                    }
                })
                .collect()
        }
        format::ADPCM => {
            let mut coeffs = [0i16; 16];
            let mut c = [0u8; 32];
            if read(v.data_address, &mut c[..(v.data_size as usize).min(32)]) {
                for (k, x) in coeffs.iter_mut().enumerate() {
                    *x = i16::from_le_bytes([c[2 * k], c[2 * k + 1]]);
                }
            }
            decode_adpcm(w.address, pos as u32, n as u32, &coeffs, adpcm, read)
        }
        _ => Vec::new(),
    }
}

/// Eden DecodeAdpcm: `count` samples from sample `start`.
fn decode_adpcm(
    address: u64,
    start: u32,
    count: u32,
    coeffs: &[i16; 16],
    ctx: &mut Adpcm,
    read: &ReadFn,
) -> Vec<f32> {
    const SAMPLES: u32 = 14;
    const NIBBLES: u32 = 16;
    let mut position = (start / SAMPLES) * NIBBLES + start % SAMPLES;
    if !start.is_multiple_of(SAMPLES) {
        position += 2;
    }
    let bytes = (count / SAMPLES + 2) * 8 + 8;
    let mut data = vec![0u8; bytes as usize];
    if !read(address + u64::from(position / 2), &mut data) {
        return Vec::new();
    }
    let mut header = ctx.header;
    let (mut yn0, mut yn1) = (i32::from(ctx.yn0), i32::from(ctx.yn1));
    let mut scale = u32::from(header & 0xF);
    let mut ci = usize::from((header >> 4) & 7);
    let mut out = Vec::with_capacity(count as usize);
    let mut at = 0usize;
    while out.len() < count as usize {
        if position.is_multiple_of(NIBBLES) {
            header = u16::from(data.get(at).copied().unwrap_or(0));
            at += 1;
            scale = u32::from(header & 0xF);
            ci = usize::from((header >> 4) & 7);
            position += 2;
        }
        let b = data.get(at).copied().unwrap_or(0);
        let code = if position & 1 != 0 {
            at += 1;
            b & 0xF
        } else {
            b >> 4
        };
        let code = i32::from(((code << 4) as i8) >> 4);
        let xn = code << scale;
        let prediction = i32::from(coeffs[ci * 2]) * yn0 + i32::from(coeffs[ci * 2 + 1]) * yn1;
        let sample = (((xn << 11) + 0x400 + prediction) >> 11).clamp(-0x8000, 0x7FFF);
        yn1 = yn0;
        yn0 = sample;
        out.push(sample as f32 / 32768.0);
        position += 1;
    }
    *ctx = Adpcm {
        header,
        yn0: yn0 as i16,
        yn1: yn1 as i16,
    };
    out
}

/// Eden DecodeFromWaveBuffers with linear resampling: fills `out`
/// (target-rate samples) for one channel, consuming wave buffers.
fn decode(
    v: &Voice,
    channel: usize,
    s: &mut ChannelState,
    rate: u32,
    out: &mut [f32],
    read: &ReadFn,
) {
    let ratio = f64::from(v.sample_rate.max(1)) / f64::from(rate.max(1)) * f64::from(v.pitch);
    if !(ratio > 0.0 && ratio < 64.0) {
        return;
    }
    // Source samples needed: positions fraction + i * ratio (i < len)
    // and the next one, for interpolation.
    let last_pos = s.fraction + (out.len().max(1) - 1) as f64 * ratio;
    let needed = last_pos.floor() as u32 + 2;
    let mut src = std::mem::take(&mut s.carry);
    while (src.len() as u32) < needed {
        if s.index >= WAVE_BUFFERS {
            s.index = 0;
        }
        if !s.valid[s.index] {
            break;
        }
        let w = v.buffers[s.index];
        if s.offset == 0 && v.format == format::ADPCM && w.context != 0 {
            let mut c = [0u8; 6];
            if read(w.context, &mut c[..(w.context_size as usize).min(6)]) {
                s.adpcm = Adpcm {
                    header: u16::from_le_bytes([c[0], c[1]]),
                    yn0: i16::from_le_bytes([c[2], c[3]]),
                    yn1: i16::from_le_bytes([c[4], c[5]]),
                };
            }
        }
        let (mut start, mut end) = (w.start, w.end);
        if w.looping && s.loop_count > 0 && w.loop_start <= w.loop_end {
            start = w.loop_start as i32;
            end = w.loop_end as i32;
        }
        let want = needed - src.len() as u32;
        let got = decode_buffer(
            v,
            channel,
            &w,
            (start, end),
            (s.offset, want),
            &mut s.adpcm,
            read,
        );
        let decoded = got.len() as u32;
        src.extend(got);
        s.played += u64::from(decoded);
        s.offset += decoded;
        if decoded > 0 && (s.offset as i64) < i64::from(end) - i64::from(start) {
            continue;
        }
        s.offset = 0;
        let end_buffer = |s: &mut ChannelState| {
            s.valid[s.index] = false;
            s.loop_count = 0;
            if w.stream_ended {
                s.played = 0;
            }
            s.index = (s.index + 1) % WAVE_BUFFERS;
            s.consumed += 1;
        };
        if w.looping {
            s.loop_count += 1;
            if w.loop_count >= 0 && (s.loop_count > w.loop_count || decoded == 0) {
                end_buffer(s);
            }
            if decoded == 0 {
                break;
            }
        } else {
            end_buffer(s);
        }
    }
    // Starved: the rest is silence.
    let available = src.len();
    let mut pos = s.fraction;
    for o in out.iter_mut() {
        let i = pos.floor() as usize;
        let f = (pos - pos.floor()) as f32;
        *o = if i + 1 < available {
            src[i] + (src[i + 1] - src[i]) * f
        } else {
            0.0
        };
        pos += ratio;
    }
    let consumed = pos.floor() as usize;
    if consumed < available {
        s.fraction = pos - pos.floor();
        s.carry = src.split_off(consumed);
    } else {
        s.fraction = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One voice: PCM16 mono at 48 kHz, two wave buffers of 100 samples.
    fn voice_input(play: u8, sent: bool) -> (Vec<u8>, Vec<u8>) {
        let mut res = vec![0u8; 0x70];
        res[4..8].copy_from_slice(&1.0f32.to_bits().to_le_bytes()); // FL
        res[8..12].copy_from_slice(&1.0f32.to_bits().to_le_bytes()); // FR
        res[0x64] = 1;
        let mut p = vec![0u8; 0x170];
        p[8] = 1; // is_new
        p[9] = 1;
        p[0xA] = play;
        p[0xB] = format::PCM16;
        p[0xC..0x10].copy_from_slice(&48_000u32.to_le_bytes());
        p[0x18..0x1C].copy_from_slice(&1u32.to_le_bytes());
        p[0x1C..0x20].copy_from_slice(&1.0f32.to_bits().to_le_bytes());
        p[0x20..0x24].copy_from_slice(&0.5f32.to_bits().to_le_bytes());
        p[0x3C..0x40].copy_from_slice(&2u32.to_le_bytes());
        p[0x58..0x5C].copy_from_slice(&0u32.to_le_bytes()); // final mix
        p[0x5C..0x60].copy_from_slice(&UNUSED_SPLITTER.to_le_bytes());
        for k in 0..2 {
            let w = 0x60 + 0x38 * k;
            p[w..w + 8].copy_from_slice(&(0x1000u64 + 200 * k as u64).to_le_bytes());
            p[w + 8..w + 16].copy_from_slice(&200u64.to_le_bytes());
            p[w + 0x14..w + 0x18].copy_from_slice(&100u32.to_le_bytes());
            p[w + 0x1A] = u8::from(sent);
        }
        // Slots the game has not filled are marked as already sent.
        for k in 2..4 {
            p[0x60 + 0x38 * k + 0x1A] = 1;
        }
        (res, p)
    }

    #[test]
    fn pcm_voice_plays_and_consumes_buffers() {
        // Samples: 0x4000 (0.5) everywhere.
        let read = |_: u64, b: &mut [u8]| {
            for c in b.chunks_exact_mut(2) {
                c.copy_from_slice(&0x4000i16.to_le_bytes());
            }
            true
        };
        let mut v = Voices::new(1);
        let (res, p) = voice_input(0, false);
        v.update(&res, &p);
        let out = v.render(150, 48_000, &read);
        // 0.5 sample * 0.5 voice volume * 1.0 mix volume.
        assert_eq!(out[10], 0.25);
        assert_eq!(out[11], 0.25);
        // 150 + 1 samples read: the first buffer is consumed.
        let mut p2 = p.clone();
        p2[8] = 0;
        p2[0x60 + 0x1A] = 1;
        p2[0x60 + 0x38 + 0x1A] = 1;
        let status = v.update(&res, &p2);
        assert_eq!(u32::from_le_bytes(status[8..12].try_into().unwrap()), 1);
        // The second runs out mid-frame: silence after it.
        let out = v.render(100, 48_000, &read);
        assert_eq!(out[2 * 99], 0.0);
        let status = v.update(&res, &p2);
        assert_eq!(u32::from_le_bytes(status[8..12].try_into().unwrap()), 2);
    }

    #[test]
    fn sub_mix_routes_to_final_mix() {
        let read = |_: u64, b: &mut [u8]| {
            for c in b.chunks_exact_mut(2) {
                c.copy_from_slice(&0x4000i16.to_le_bytes());
            }
            true
        };
        let mix = |volume: f32, dest: u32, gain: f32| {
            let mut p = vec![0u8; 0x930];
            p[..4].copy_from_slice(&volume.to_bits().to_le_bytes());
            p[8..12].copy_from_slice(&2u32.to_le_bytes());
            p[0xC] = 1;
            for i in 0..2 {
                let at = 0x24 + 4 * (i * MIX_BUFFERS + i);
                p[at..at + 4].copy_from_slice(&gain.to_bits().to_le_bytes());
            }
            p[0x924..0x928].copy_from_slice(&dest.to_le_bytes());
            p[0x928..0x92C].copy_from_slice(&UNUSED_SPLITTER.to_le_bytes());
            p
        };
        let mut mixes = mix(1.0, UNUSED_MIX, 0.0);
        mixes.extend(mix(1.0, 0, 0.5)); // sub-mix 1 -> final, half volume
        let mut v = Voices::new(1);
        let (res, mut p) = voice_input(0, false);
        p[0x58..0x5C].copy_from_slice(&1u32.to_le_bytes()); // to the sub-mix
        v.update(&res, &p);
        v.update_mixes(&mixes, 2);
        let out = v.render(16, 48_000, &read);
        // 0.5 sample * 0.5 voice * 1.0 channel volume * 0.5 matrix.
        assert_eq!((out[4], out[5]), (0.125, 0.125));
    }

    #[test]
    fn adpcm_frames_decode() {
        // Header: scale 0, coefficient set 0 (0, 0): samples are the
        // nibbles themselves (<< 11 >> 11).
        let mut frame = [0u8; 8];
        frame[0] = 0;
        frame[1] = 0x17; // 1, 7
        frame[2] = 0xF8; // -1, -8
        let read = |a: u64, b: &mut [u8]| {
            for (i, x) in b.iter_mut().enumerate() {
                *x = frame.get((a as usize + i) % 8).copied().unwrap_or(0);
            }
            true
        };
        let mut ctx = Adpcm::default();
        let s = decode_adpcm(0, 0, 4, &[0; 16], &mut ctx, &read);
        let want = [1.0, 7.0, -1.0, -8.0].map(|x: f32| x / 32768.0);
        assert_eq!(s, want);
    }
}
