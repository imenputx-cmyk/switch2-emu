//! Audio renderer effects on mix buffers: buffer mixer, biquad filter,
//! delay and reverb, ported to f32 from Lichii/Eden audio_core
//! (GPL-3.0-or-later): `renderer/command/effect/{biquad_filter,delay,
//! reverb}.cpp`, `renderer/command/mix/mix.cpp` and the effect parameter
//! layouts in `renderer/effect/*.h`. Aux, capture, I3DL2 reverb, the
//! limiter and the compressor pass their input through.

// The tables are Lichii's values as written.
#![allow(clippy::excessive_precision)]

/// Channels an effect processes at most.
const MAX_CHANNELS: usize = 6;

mod kind {
    pub const MIX: u8 = 1;
    pub const DELAY: u8 = 3;
    pub const REVERB: u8 = 4;
    pub const BIQUAD: u8 = 6;
}

/// One effect slot (EffectInfoBase::InParameter, 0xC0 bytes).
#[derive(Debug, Clone)]
pub struct Effect {
    kind: u8,
    enabled: bool,
    pub mix_id: u32,
    pub order: u32,
    params: Vec<u8>,
    state: State,
}

#[derive(Debug, Clone, Default)]
enum State {
    #[default]
    None,
    Biquad([[f64; 4]; MAX_CHANNELS]),
    Delay(Box<Delay>),
    Reverb(Box<Reverb>),
}

fn u16_at(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}

fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

/// A Q14 fixed-point parameter.
fn q14(b: &[u8], o: usize) -> f32 {
    u32_at(b, o) as i32 as f32 / 16384.0
}

impl Effect {
    /// Applies one effect's input parameters (0xC0 bytes).
    pub fn update(&mut self, p: &[u8]) {
        let kind = p[0];
        if kind != self.kind || p[1] != 0 {
            self.state = State::None;
        }
        self.kind = kind;
        self.enabled = p[2] != 0;
        self.mix_id = u32_at(p, 4);
        self.order = u32_at(p, 0x18);
        self.params = p[0x20..0xC0].to_vec();
    }

    pub fn new() -> Effect {
        Effect {
            kind: 0,
            enabled: false,
            mix_id: u32::MAX,
            order: 0,
            params: vec![0; 0xA0],
            state: State::None,
        }
    }

    /// Runs the effect on `bufs` (one mix's buffers) at `rate` Hz.
    pub fn apply(&mut self, bufs: &mut [Vec<f32>], rate: u32) {
        let p = self.params.clone();
        let len = bufs.len();
        let index = move |v: u8| -> Option<usize> {
            let i = v as i8;
            (i >= 0 && (i as usize) < len).then_some(i as usize)
        };
        match self.kind {
            kind::MIX => {
                if !self.enabled {
                    return;
                }
                let count = (u32_at(&p, 0x90) as usize).min(24);
                for k in 0..count {
                    let (Some(i), Some(o)) = (index(p[k]), index(p[0x18 + k])) else {
                        continue;
                    };
                    let v = f32::from_bits(u32_at(&p, 0x30 + 4 * k));
                    let src = bufs[i].clone();
                    for (d, s) in bufs[o].iter_mut().zip(src) {
                        *d += s * v;
                    }
                }
            }
            kind::BIQUAD | kind::DELAY | kind::REVERB => {
                let channels = match self.kind {
                    kind::BIQUAD => p[0x16].min(MAX_CHANNELS as u8) as usize,
                    _ => (u16_at(&p, 0x0E) as usize).min(MAX_CHANNELS),
                };
                let ins: Vec<Option<usize>> = (0..channels).map(|c| index(p[c])).collect();
                let outs: Vec<Option<usize>> = (0..channels).map(|c| index(p[6 + c])).collect();
                if ins.iter().chain(&outs).any(Option::is_none) {
                    return;
                }
                let ins: Vec<usize> = ins.into_iter().flatten().collect();
                let outs: Vec<usize> = outs.into_iter().flatten().collect();
                let input: Vec<Vec<f32>> = ins.iter().map(|&i| bufs[i].clone()).collect();
                let mut output = input.clone();
                if self.enabled {
                    match self.kind {
                        kind::BIQUAD => self.biquad(&p, &input, &mut output),
                        kind::DELAY => self.delay(&p, &input, &mut output),
                        _ => self.reverb(&p, &input, &mut output, rate),
                    }
                }
                for (o, data) in outs.iter().zip(output) {
                    bufs[*o] = data;
                }
            }
            _ => {}
        }
    }

    /// ApplyBiquadFilterFloat: Q14 coefficients (version 1, the type
    /// byte at 0x16 follows them) or floats (version 2, a padding word
    /// at 0x0C).
    fn biquad(&mut self, p: &[u8], input: &[Vec<f32>], output: &mut [Vec<f32>]) {
        let v2 = u32_at(p, 0x0C) == 0 && p[0x24] != 0;
        let (b, a): ([f64; 3], [f64; 2]) = if v2 {
            (
                [0, 1, 2].map(|k| f64::from(f32::from_bits(u32_at(p, 0x10 + 4 * k)))),
                [0, 1].map(|k| f64::from(f32::from_bits(u32_at(p, 0x1C + 4 * k)))),
            )
        } else {
            (
                [0, 1, 2].map(|k| f64::from(u16_at(p, 0x0C + 2 * k) as i16) / 16384.0),
                [0, 1].map(|k| f64::from(u16_at(p, 0x12 + 2 * k) as i16) / 16384.0),
            )
        };
        if !matches!(self.state, State::Biquad(_)) {
            self.state = State::Biquad([[0.0; 4]; MAX_CHANNELS]);
        }
        let State::Biquad(states) = &mut self.state else {
            return;
        };
        for (c, (inp, out)) in input.iter().zip(output.iter_mut()).enumerate() {
            let s = &mut states[c];
            for (x, y) in inp.iter().zip(out.iter_mut()) {
                let x = f64::from(*x);
                let v = x * b[0] + s[0] * b[1] + s[1] * b[2] + s[2] * a[0] + s[3] * a[1];
                *y = v.clamp(-1.0, 1.0) as f32;
                s[1] = s[0];
                s[0] = x;
                s[3] = s[2];
                s[2] = v;
            }
        }
    }

    /// ApplyDelay.
    fn delay(&mut self, p: &[u8], input: &[Vec<f32>], output: &mut [Vec<f32>]) {
        let n = input.len();
        if !matches!(n, 1 | 2 | 4 | 6) {
            return;
        }
        let sample_rate = q14(p, 0x18);
        if !matches!(&self.state, State::Delay(d) if d.lines.len() == n) {
            // InitializeDelayEffect.
            let max = (0.064 * sample_rate.floor() * u32_at(p, 0x10) as f32) as usize;
            let count = ((u32_at(p, 0x14) as f32 * sample_rate / 1000.0) as usize).min(max);
            self.state = State::Delay(Box::new(Delay {
                lines: (0..n).map(|_| DelayLine::new(count.max(1))).collect(),
                lowpass_z: [0.0; MAX_CHANNELS],
            }));
        }
        let State::Delay(state) = &mut self.state else {
            return;
        };
        // SetDelayEffectParameter.
        let mut spread = q14(p, 0x2C);
        let feedback = q14(p, 0x20) * 0.979_980_47;
        let delay_feedback = feedback * (1.0 - spread);
        if n == 4 || n == 6 {
            spread /= 2.0;
        }
        let cross = spread * feedback;
        let lowpass_feedback = q14(p, 0x30) * 0.949_951_17;
        let lowpass = 1.0 - lowpass_feedback;
        let (in_gain, wet, dry) = (q14(p, 0x1C), q14(p, 0x24), q14(p, 0x28));
        let (d, x) = (delay_feedback, cross);
        let matrix: Vec<Vec<f32>> = match n {
            1 => vec![vec![feedback]],
            2 => vec![vec![d, x], vec![x, d]],
            4 => vec![
                vec![d, x, x, 0.0],
                vec![x, d, 0.0, x],
                vec![x, 0.0, d, x],
                vec![0.0, x, x, d],
            ],
            _ => vec![
                vec![d, 0.0, x, 0.0, x, 0.0],
                vec![0.0, d, x, 0.0, 0.0, x],
                vec![x, x, d, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, q14(p, 0x20), 0.0, 0.0],
                vec![x, 0.0, 0.0, 0.0, d, x],
                vec![0.0, x, 0.0, 0.0, x, d],
            ],
        };
        let frames = input[0].len();
        for i in 0..frames {
            let delayed: Vec<f32> = state.lines.iter().map(DelayLine::read).collect();
            for c in 0..n {
                let fb: f32 = (0..n).map(|j| delayed[j] * matrix[j][c]).sum();
                let gained = input[c][i] * in_gain + fb;
                state.lowpass_z[c] = gained * lowpass + state.lowpass_z[c] * lowpass_feedback;
                let z = state.lowpass_z[c];
                state.lines[c].write(z);
            }
            for c in 0..n {
                output[c][i] = input[c][i] * dry + delayed[c] * wet;
            }
        }
    }

    /// ApplyReverbEffect (with UpdateReverbEffectParameter applied in
    /// full each frame instead of smoothed).
    fn reverb(&mut self, p: &[u8], input: &[Vec<f32>], output: &mut [Vec<f32>], rate: u32) {
        let n = input.len();
        if !matches!(n, 1 | 2 | 4 | 6) {
            return;
        }
        // Samples per millisecond (Q14).
        let mut sr = q14(p, 0x10);
        if sr < 1.0 {
            sr = rate as f32 / 1000.0;
        }
        if !matches!(&self.state, State::Reverb(_)) {
            self.state = State::Reverb(Box::new(Reverb::new(sr)));
        }
        let State::Reverb(r) = &mut self.state else {
            return;
        };
        let early_mode = (u32_at(p, 0x14) as usize).min(4);
        let late_mode = (u32_at(p, 0x20) as usize).min(4);
        let pre_delay = q14(p, 0x1C).max(0.0);
        for t in 0..10 {
            r.early_times[t] =
                (((pre_delay + EARLY_TIMES[early_mode][t]) * sr) as usize).min(r.pre.len() - 1);
            r.early_gains[t] = (q14(p, 0x18) * EARLY_GAINS[early_mode][t]).clamp(0.0, 1.0);
        }
        if n == 2 {
            r.early_gains[4] *= 0.5;
            r.early_gains[5] *= 0.5;
        }
        r.pre_time =
            (((pre_delay + EARLY_TIMES[early_mode][10]) * sr) as usize).min(r.pre.len() - 1);
        let coloration = q14(p, 0x30).clamp(0.0, 1.0);
        let decay_time = {
            let d = q14(p, 0x28);
            if d <= 0.0 {
                0.0001
            } else {
                d
            }
        };
        let hf_ratio = q14(p, 0x2C).clamp(0.001, 0.999);
        let cos = (1280.0 / sr * std::f32::consts::PI / 180.0).cos();
        let pow10 = |v: f32| -> f32 {
            if v >= 0.0 {
                1.0
            } else if v <= -5.3 {
                0.0
            } else {
                10f32.powf(v)
            }
        };
        for i in 0..4 {
            r.fdn[i].set_delay(((FDN_TIMES[late_mode][i] * sr) as usize).max(1));
            r.decay[i].set_delay(((DECAY_TIMES[late_mode][i] * sr) as usize).max(1));
            r.decay_gain[i] = 0.599_975_6 * (1.0 - coloration);
            let a = (r.fdn[i].max as f32 + r.decay[i].max as f32) * -3.0;
            let b = a / (decay_time * sr);
            let (c, d) = if hf_ratio > 0.9949 {
                (0.0, 1.0)
            } else {
                let e =
                    pow10(((1.0 / hf_ratio) - 1.0) * 2.0 / 100.0 * (b / 10.0)).clamp(0.0001, 1.0);
                let f = (1.0 - e).max(0.0001);
                let g = 2.0 - cos * e * 2.0;
                let h = (g * g - f * f * 4.0).max(0.0).sqrt();
                let c = (g - h) / (f * 2.0);
                (c, 1.0 - c)
            };
            r.hf_prev_gain[i] = c;
            r.hf_gain[i] = pow10(b / 1000.0) * d * 0.707_092_3;
        }
        let taps: [usize; 10] = match n {
            1 => [0; 10],
            2 => [0, 0, 1, 1, 0, 1, 0, 0, 1, 1],
            4 => [0, 0, 1, 1, 0, 1, 2, 2, 3, 3],
            _ => [0, 0, 1, 1, 2, 2, 4, 4, 5, 5],
        };
        let (base_gain, late_gain) = (q14(p, 0x34), q14(p, 0x24));
        let (wet, dry) = (q14(p, 0x38), q14(p, 0x3C));
        for s in 0..input[0].len() {
            let mut out = [0f32; MAX_CHANNELS];
            for t in 0..10 {
                let v = r.pre.tap(r.early_times[t]) * r.early_gains[t];
                out[taps[t]] += v;
                if n == 6 {
                    out[3] += v;
                }
            }
            if n == 6 {
                out[3] *= 0.2;
            }
            let x: f32 = (0..n).map(|c| input[c][s]).sum::<f32>() * base_gain;
            r.pre.write(x);
            for i in 0..4 {
                r.feedback[i] = r.feedback[i] * r.hf_prev_gain[i] + r.fdn[i].read() * r.hf_gain[i];
            }
            let pre = r.pre.tap(r.pre_time) * late_gain;
            let f = r.feedback;
            let mix = [
                f[2] + f[1] + pre,
                -f[0] - f[3] + pre,
                f[0] - f[3] + pre,
                f[1] - f[2] + pre,
            ];
            let mut ap = [0f32; 4];
            for i in 0..4 {
                // Axfx2AllPassTick.
                let g = r.decay_gain[i];
                let v = r.decay[i].read();
                let mixed = mix[i] - v * g;
                let o = r.decay[i].tick(mixed) + mixed * g;
                r.fdn[i].tick(o);
                ap[i] = o;
            }
            for c in 0..n {
                let late = if n == 6 {
                    let six = [ap[0], ap[1], ap[2] - ap[3], ap[3], ap[2], ap[3]];
                    if c == 2 {
                        r.center.tick(six[2] * 0.5)
                    } else {
                        six[c]
                    }
                } else {
                    ap[c]
                };
                output[c][s] = input[c][s] * dry + (out[c] + late) * wet;
            }
        }
    }
}

impl Default for Effect {
    fn default() -> Self {
        Effect::new()
    }
}

#[derive(Debug, Clone)]
struct DelayLine {
    buf: Vec<f32>,
    pos: usize,
}

impl DelayLine {
    fn new(n: usize) -> DelayLine {
        DelayLine {
            buf: vec![0.0; n],
            pos: 0,
        }
    }

    fn read(&self) -> f32 {
        self.buf[self.pos]
    }

    fn write(&mut self, v: f32) {
        self.buf[self.pos] = v;
        self.pos = (self.pos + 1) % self.buf.len();
    }
}

#[derive(Debug, Clone)]
struct Delay {
    lines: Vec<DelayLine>,
    lowpass_z: [f32; MAX_CHANNELS],
}

/// Eden ReverbDelayLine: a ring of `max + 1` samples used as a delay of
/// `count` samples.
#[derive(Debug, Clone)]
struct ReverbLine {
    buf: Vec<f32>,
    max: usize,
    count: usize,
    input: usize,
    output: usize,
}

impl ReverbLine {
    fn new(max: usize) -> ReverbLine {
        let max = max.max(1);
        ReverbLine {
            buf: vec![0.0; max + 1],
            max,
            count: max,
            input: 0,
            output: 0,
        }
    }

    fn len(&self) -> usize {
        self.count.max(1)
    }

    fn set_delay(&mut self, count: usize) {
        let count = count.min(self.max);
        if count != self.count {
            self.count = count.max(1);
            self.input %= self.count;
            self.output = (self.input + 1) % self.count;
        }
    }

    fn read(&self) -> f32 {
        self.buf[self.output]
    }

    fn write(&mut self, v: f32) {
        self.buf[self.input] = v;
        self.input = (self.input + 1) % self.len();
    }

    fn tick(&mut self, v: f32) -> f32 {
        let out = self.read();
        self.output = (self.output + 1) % self.len();
        self.write(v);
        out
    }

    fn tap(&self, index: usize) -> f32 {
        let n = self.len();
        self.buf[(self.input + n - (index % n) - 1) % n]
    }
}

#[derive(Debug, Clone)]
struct Reverb {
    fdn: [ReverbLine; 4],
    decay: [ReverbLine; 4],
    pre: ReverbLine,
    center: ReverbLine,
    early_times: [usize; 10],
    early_gains: [f32; 10],
    pre_time: usize,
    decay_gain: [f32; 4],
    hf_prev_gain: [f32; 4],
    hf_gain: [f32; 4],
    feedback: [f32; 4],
}

impl Reverb {
    /// InitializeReverbEffect (the long pre-delay: 350 ms).
    fn new(sr: f32) -> Reverb {
        Reverb {
            fdn: FDN_MAX.map(|t| ReverbLine::new((t * sr) as usize)),
            decay: DECAY_MAX.map(|t| ReverbLine::new((t * sr) as usize)),
            pre: ReverbLine::new((350.0 * sr) as usize),
            center: ReverbLine::new((5.0 * sr) as usize),
            early_times: [0; 10],
            early_gains: [0.0; 10],
            pre_time: 0,
            decay_gain: [0.0; 4],
            hf_prev_gain: [0.0; 4],
            hf_gain: [0.0; 4],
            feedback: [0.0; 4],
        }
    }
}

const FDN_MAX: [f32; 4] = [53.953_247, 79.192_566, 116.238_77, 170.615_3];
const DECAY_MAX: [f32; 4] = [7.0, 9.0, 13.0, 17.0];

const EARLY_TIMES: [[f32; 11]; 5] = [
    [
        0.0, 3.5, 2.799988, 3.899963, 2.699951, 13.399963, 7.899963, 8.399963, 9.899963, 12.0, 12.5,
    ],
    [
        0.0, 11.799988, 5.5, 11.199951, 10.399963, 38.099976, 22.199951, 29.599976, 21.199951,
        24.799988, 40.0,
    ],
    [
        0.0, 41.5, 20.5, 41.299988, 0.0, 29.5, 33.799988, 45.199951, 46.799988, 0.0, 50.0,
    ],
    [
        33.099976, 43.299988, 22.799988, 37.899963, 14.899963, 35.299988, 17.899963, 34.199951,
        0.0, 43.299988, 50.0,
    ],
    [0.0; 11],
];

const EARLY_GAINS: [[f32; 10]; 5] = [
    [
        0.699951, 0.679993, 0.699951, 0.679993, 0.699951, 0.679993, 0.699951, 0.679993, 0.679993,
        0.679993,
    ],
    [
        0.699951, 0.679993, 0.699951, 0.679993, 0.699951, 0.679993, 0.679993, 0.679993, 0.679993,
        0.679993,
    ],
    [
        0.5, 0.699951, 0.699951, 0.679993, 0.5, 0.679993, 0.679993, 0.699951, 0.679993, 0.0,
    ],
    [
        0.929993, 0.919983, 0.869995, 0.859985, 0.939941, 0.809998, 0.799988, 0.769958, 0.759949,
        0.649963,
    ],
    [0.0; 10],
];

const FDN_TIMES: [[f32; 4]; 5] = [
    [53.953247, 79.192566, 116.23877, 130.615295],
    [53.953247, 79.192566, 116.23877, 170.615295],
    [5.0, 10.0, 5.0, 10.0],
    [47.029968, 71.0, 103.0, 170.0],
    [53.953247, 79.192566, 116.23877, 170.615295],
];

const DECAY_TIMES: [[f32; 4]; 5] = [
    [7.0, 9.0, 13.0, 17.0],
    [7.0, 9.0, 13.0, 17.0],
    [1.0, 1.0, 1.0, 1.0],
    [7.0, 7.0, 13.0, 9.0],
    [7.0, 9.0, 13.0, 17.0],
];

#[cfg(test)]
mod tests {
    use super::*;

    fn effect(kind: u8, specific: &[(usize, &[u8])]) -> Effect {
        let mut p = vec![0u8; 0xC0];
        p[0] = kind;
        p[2] = 1;
        for (at, bytes) in specific {
            p[0x20 + at..0x20 + at + bytes.len()].copy_from_slice(bytes);
        }
        let mut e = Effect::new();
        e.update(&p);
        e
    }

    #[test]
    fn buffer_mixer_adds_scaled_input() {
        let mut e = effect(
            kind::MIX,
            &[
                (0, &[0]),
                (0x18, &[1]),
                (0x30, &0.5f32.to_le_bytes()),
                (0x90, &1u32.to_le_bytes()),
            ],
        );
        let mut bufs = vec![vec![1.0; 4], vec![0.25; 4]];
        e.apply(&mut bufs, 48_000);
        assert_eq!(bufs[1], [0.75; 4]);
    }

    #[test]
    fn delay_echoes_after_its_time() {
        // Mono, 48 kHz (Q14), 1 ms delay (48 samples), wet 1, dry 0,
        // in gain 1, no feedback.
        let q = |v: f32| ((v * 16384.0) as i32).to_le_bytes();
        let mut e = effect(
            kind::DELAY,
            &[
                (0, &[0]),
                (6, &[0]),
                (0x0E, &1u16.to_le_bytes()),
                (0x10, &10u32.to_le_bytes()),
                (0x14, &1u32.to_le_bytes()),
                (0x18, &q(48_000.0)),
                (0x1C, &q(1.0)),
                (0x24, &q(1.0)),
            ],
        );
        let mut bufs = vec![vec![0.0; 100]];
        bufs[0][0] = 1.0;
        e.apply(&mut bufs, 48_000);
        assert_eq!(bufs[0][0], 0.0);
        assert_eq!(bufs[0][48], 1.0);
    }

    #[test]
    fn reverb_produces_a_tail() {
        let q = |v: f32| ((v * 16384.0) as i32).to_le_bytes();
        let mut e = effect(
            kind::REVERB,
            &[
                (0, &[0, 1]),
                (6, &[0, 1]),
                (0x0E, &2u16.to_le_bytes()),
                (0x10, &q(48.0)),
                (0x18, &q(1.0)),
                (0x24, &q(1.0)),
                (0x28, &q(1.5)),
                (0x2C, &q(0.5)),
                (0x34, &q(0.5)),
                (0x38, &q(1.0)),
            ],
        );
        let mut tail = 0.0f32;
        for frame in 0..40 {
            let mut bufs = vec![vec![0.0; 240], vec![0.0; 240]];
            if frame == 0 {
                bufs[0][0] = 1.0;
            }
            e.apply(&mut bufs, 48_000);
            if frame > 4 {
                tail += bufs[0].iter().map(|v| v.abs()).sum::<f32>();
            }
            assert!(bufs[0].iter().all(|v| v.is_finite()));
        }
        assert!(tail > 0.0, "no reverb tail");
    }
}
