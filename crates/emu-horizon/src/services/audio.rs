//! Audio: `aud:u` (Switch 2's entry point), `audren:u` (the audio
//! renderer) and `audout:u`. Renderer voices are decoded and mixed to
//! stereo on each update (see [`super::audren`]); audio-out buffers are
//! played as they are queued. Both go to `System::audio_out`.
//!
//! Layouts follow Eden's audio_core (`renderer/behavior/info_updater`,
//! `renderer/system.cpp`; GPL-3.0-or-later) and Lichii's
//! `nx2_audio_user_client.h` (aud:u 0/1/2 hand out the in/out/renderer
//! managers, NX2_NOTES rows 38 and 39).

use super::{obj, CallError, Ctx, Service};

/// The renderer's parameters (AudioRendererParameterInternal).
#[derive(Debug, Clone, Copy, Default)]
pub struct RendererParams {
    pub sample_rate: u32,
    pub sample_count: u32,
    pub mixes: u32,
    pub voices: u32,
    pub sinks: u32,
    pub effects: u32,
    pub revision: u32,
}

impl RendererParams {
    fn parse(ctx: &Ctx) -> Self {
        RendererParams {
            sample_rate: ctx.in_u32(0),
            sample_count: ctx.in_u32(4),
            mixes: ctx.in_u32(8),
            voices: ctx.in_u32(0x10),
            sinks: ctx.in_u32(0x14),
            effects: ctx.in_u32(0x18),
            revision: ctx.in_u32(0x30),
        }
    }

    /// Work memory the game must provide. Ours is not used; the size only
    /// has to be plausible and page aligned.
    fn work_buffer_size(&self) -> u64 {
        let size = 0x10_0000
            + u64::from(self.voices) * 0x1000
            + u64::from(self.effects) * 0x1_0000
            + u64::from(self.mixes) * 0x4000
            + u64::from(self.sinks) * 0x1000;
        size.next_multiple_of(0x1000)
    }

    /// Memory pools: one per effect plus four per voice (Eden).
    fn memory_pools(&self) -> u32 {
        self.effects + self.voices * 4
    }
}

/// The "REV" magic revisions count from (`MakeMagic('R','E','V','0')`).
const REV0: u32 = u32::from_le_bytes(*b"REV0");

/// Revision number of a "REVn" tag; Switch 2 SDKs tag updates "REvn"
/// (Lichii row 39), so the case of the letters is ignored.
fn revision_number(tag: u32) -> u32 {
    (tag & !0x2020_2020).wrapping_sub(REV0 & !0x2020_2020)
}

/// Builds RequestUpdate's output for `input` (UpdateDataHeader, then
/// memory pools, voices, effects, sinks, performance, behavior errors and
/// render info, as Eden's System::Update writes them).
pub fn update_output(
    p: &RendererParams,
    input: &[u8],
    frames: u64,
    voice_status: Option<&[u8]>,
) -> Vec<u8> {
    let word = |o: usize| {
        input
            .get(o..o + 4)
            .map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()))
    };
    let tag = word(0);
    let in_behaviour = word(4) as usize;
    let in_pools = word(8) as usize;
    let mut out = vec![0u8; 0x40];
    let mut sizes = [0u32; 11]; // header words 1..=10
                                // Memory pools: requested attaches become attached, detaches detached
                                // (states: 4 RequestAttach -> 5 Attached, 2 RequestDetach -> 3).
    let pools_at = 0x40 + in_behaviour;
    let pool_count = p.memory_pools() as usize;
    for i in 0..pool_count {
        let state = if i * 0x20 + 0x20 <= in_pools {
            word(pools_at + i * 0x20 + 0x10)
        } else {
            0
        };
        let out_state = match state {
            4 => 5,
            2 => 3,
            s => s,
        };
        let mut status = [0u8; 0x10];
        status[..4].copy_from_slice(&out_state.to_le_bytes());
        out.extend_from_slice(&status);
    }
    sizes[2] = (pool_count * 0x10) as u32;
    // Voices: their statuses.
    let mut voices = vec![0u8; p.voices as usize * 0x10];
    if let Some(v) = voice_status {
        let n = v.len().min(voices.len());
        voices[..n].copy_from_slice(&v[..n]);
    }
    out.extend(voices);
    sizes[3] = p.voices * 0x10;
    // Effects: version 2 statuses (0x90) from revision 9 on.
    let effect_size = if revision_number(tag) >= 9 {
        0x90
    } else {
        0x10
    };
    out.extend(std::iter::repeat_n(0, p.effects as usize * effect_size));
    sizes[5] = p.effects * effect_size as u32;
    // Sinks.
    out.extend(std::iter::repeat_n(0, p.sinks as usize * 0x20));
    sizes[7] = p.sinks * 0x20;
    // Performance: no history.
    out.extend([0u8; 0x10]);
    sizes[8] = 0x10;
    // Behavior: no errors.
    out.extend([0u8; 0xB0]);
    sizes[1] = 0xB0;
    // Render info: frames elapsed.
    let mut info = [0u8; 0x10];
    info[..8].copy_from_slice(&frames.to_le_bytes());
    out.extend(info);
    sizes[10] = 0x10;
    // The header: the caller's revision tag, section sizes, total size.
    out[..4].copy_from_slice(&tag.to_le_bytes());
    for (i, s) in sizes.iter().enumerate().skip(1) {
        out[4 * i..4 * i + 4].copy_from_slice(&s.to_le_bytes());
    }
    let total = out.len() as u32;
    out[0x3C..0x40].copy_from_slice(&total.to_le_bytes());
    out
}

/// `aud:u` (Switch 2): hands out the audio managers.
pub struct AudioUserClient;

impl Service for AudioUserClient {
    fn name(&self) -> &str {
        "aud:u"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            0 => return Err(CallError::Result(AUDIO_IN_UNAVAILABLE)), // GetAudioInManager
            1 => ctx.push_object(obj(AudioOutManager)),
            2 => ctx.push_object(obj(AudioRendererManager)),
            _ => return Err(CallError::Unimplemented),
        }
        Ok(())
    }
}

/// Audio results (module 153).
const fn audio_result(d: u32) -> u32 {
    153 | (d << 9)
}
const AUDIO_IN_UNAVAILABLE: u32 = audio_result(1);

/// `audren:u`: IAudioRendererManager.
pub struct AudioRendererManager;

impl Service for AudioRendererManager {
    fn name(&self) -> &str {
        "audren:u"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            // OpenAudioRenderer(ForManualExecution)(params, aruid; work
            // memory, process).
            0 | 3 => ctx.push_object(obj(AudioRenderer::new(RendererParams::parse(ctx)))),
            1 => ctx.push_u64(RendererParams::parse(ctx).work_buffer_size()),
            // GetAudioDeviceService(WithRevisionInfo).
            2 | 4 => ctx.push_object(obj(AudioDevice::default())),
            _ => return Err(CallError::Unimplemented),
        }
        Ok(())
    }
}

/// `IAudioRenderer`.
pub struct AudioRenderer {
    params: RendererParams,
    running: bool,
    frames: u64,
    event: Option<usize>,
    voices: super::audren::Voices,
}

impl AudioRenderer {
    fn new(params: RendererParams) -> Self {
        AudioRenderer {
            params,
            running: false,
            frames: 0,
            event: None,
            voices: super::audren::Voices::new(params.voices),
        }
    }

    /// Applies an update's voice sections and renders one frame; the
    /// voices' output statuses.
    fn render(&mut self, input: &[u8], ctx: &mut Ctx) -> Vec<u8> {
        let word = |o: usize| {
            input
                .get(o..o + 4)
                .map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap())) as usize
        };
        // Input order: behaviour, memory pools, channel resources, voices.
        let resources_at = 0x40 + word(4) + word(8);
        let voices_at = resources_at + word(0x10);
        let section = |at: usize, len: usize| input.get(at..at + len).unwrap_or(&[]);
        let status = self.voices.update(
            section(resources_at, word(0x10)),
            section(voices_at, word(0x0C)),
        );
        // Then effects, splitters, mixes.
        let effects_at = voices_at + word(0x0C);
        self.voices
            .update_effects(section(effects_at, word(0x14)), self.params.effects);
        let splitters_at = voices_at + word(0x0C) + word(0x14);
        let mixes_at = splitters_at + word(0x24);
        self.voices
            .update_splitters(section(splitters_at, word(0x24)));
        self.voices
            .update_mixes(section(mixes_at, word(0x18)), self.params.mixes);
        if self.running {
            let mem = &*ctx.mem;
            let read = |a: u64, b: &mut [u8]| mem.read(a, b).is_ok();
            let mixed = self.voices.render(
                self.params.sample_count as usize,
                self.params.sample_rate,
                &read,
            );
            ctx.system.audio_out.push(&mixed, self.params.sample_rate);
        }
        status
    }
}

impl Service for AudioRenderer {
    fn name(&self) -> &str {
        "audren::IAudioRenderer"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            0 => ctx.push_u32(self.params.sample_rate),
            1 => ctx.push_u32(self.params.sample_count),
            2 => ctx.push_u32(self.params.mixes),
            3 => ctx.push_u32(u32::from(!self.running)), // GetState: 0 started, 1 stopped
            // RequestUpdate(Auto): input buffer -> output and performance
            // buffers. Each update counts as a rendered frame.
            4 | 10 => {
                let input = ctx.read_in_buffer(0)?;
                if self.running {
                    self.frames += 1;
                }
                let status = self.render(&input, ctx);
                let out = update_output(&self.params, &input, self.frames, Some(&status));
                ctx.write_out_buffer(0, &out)?;
                if let Some(b) = ctx.out_buffer(1) {
                    ctx.write_out_buffer(1, &vec![0; b.size as usize])?;
                }
            }
            5 => self.running = true,
            6 => self.running = false,
            // QuerySystemEvent: the DSP signals it once per audio frame
            // (5 ms of guest time); starts signalled.
            7 => {
                let new = self.event.is_none();
                ctx.push_event(&mut self.event, true);
                if let (true, Some(e)) = (new, self.event) {
                    ctx.system.audio_events.push(e);
                }
            }
            8 | 11 | 12 => {} // SetRenderingTimeLimit, ExecuteAudioRendererRendering, SetVoiceDropParameter
            9 => ctx.push_u32(100),
            13 => ctx.push_u32(0), // GetVoiceDropParameter
            14 => {}               // SetAsyncMemoryPoolControlEnabled
            _ => return Err(CallError::Unimplemented),
        }
        Ok(())
    }
}

/// `IAudioDevice`: one output device, at full volume.
#[derive(Default)]
pub struct AudioDevice {
    event: Option<usize>,
    input_event: Option<usize>,
    output_event: Option<usize>,
}

const DEVICE_NAME: &[u8] = b"AudioTvOutput";

fn device_name_buffer(ctx: &mut Ctx) -> Result<(), CallError> {
    let mut name = [0u8; 0x100];
    name[..DEVICE_NAME.len()].copy_from_slice(DEVICE_NAME);
    ctx.write_out_buffer(0, &name)
}

impl Service for AudioDevice {
    fn name(&self) -> &str {
        "audren::IAudioDevice"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            // ListAudioDeviceName(Auto): one name.
            0 | 6 => {
                device_name_buffer(ctx)?;
                ctx.push_u32(1);
            }
            1 | 7 => {}                                 // SetAudioDeviceOutputVolume
            2 | 8 => ctx.push_u32(1.0f32.to_bits()),    // GetAudioDeviceOutputVolume
            3 | 10 | 13 => device_name_buffer(ctx)?,    // GetActiveAudioDeviceName
            4 => ctx.push_event(&mut self.event, true), // QueryAudioDeviceSystemEvent
            5 => ctx.push_u32(2),                       // GetActiveChannelCount
            11 => ctx.push_event(&mut self.input_event, false),
            12 => ctx.push_event(&mut self.output_event, false),
            // ListAudioOutputDeviceName: one output device.
            14 => {
                device_name_buffer(ctx)?;
                ctx.push_u32(1);
            }
            // Acquire(Input|Output)DeviceNotification (17.0+): an event.
            15 | 17 => ctx.push_event(&mut self.output_event, false),
            // Release...Notification, SetAudioDeviceOutputVolumeAutoTuneEnabled.
            16 | 18 | 19 => {}
            // IsAudioDeviceOutputVolumeAutoTuneEnabled, IsActiveOutputDeviceEstimatedLowLatency.
            20 | 21 => ctx.push_u32(0),
            _ => return Err(CallError::Unimplemented),
        }
        Ok(())
    }
}

/// `audout:u`: IAudioOutManager.
pub struct AudioOutManager;

impl Service for AudioOutManager {
    fn name(&self) -> &str {
        "audout:u"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            // ListAudioOuts(Auto): "DeviceOut".
            0 | 2 => {
                let mut name = [0u8; 0x100];
                name[..9].copy_from_slice(b"DeviceOut");
                ctx.write_out_buffer(0, &name)?;
                ctx.push_u32(1);
            }
            // OpenAudioOut(Auto)(sample rate, channels...; name):
            // 48 kHz, 2 channels, PCM16, stopped; the name echoed.
            1 | 3 => {
                let mut name = [0u8; 0x100];
                name[..9].copy_from_slice(b"DeviceOut");
                ctx.write_out_buffer(0, &name)?;
                ctx.push_u32(48_000);
                ctx.push_u32(2);
                ctx.push_u32(2); // PcmInt16
                ctx.push_u32(1); // stopped
                ctx.push_object(obj(AudioOut::default()));
            }
            _ => return Err(CallError::Unimplemented),
        }
        Ok(())
    }
}

/// `IAudioOut`: buffers are released as soon as they are appended.
#[derive(Default)]
pub struct AudioOut {
    running: bool,
    released: Vec<u64>,
    event: Option<usize>,
}

impl Service for AudioOut {
    fn name(&self) -> &str {
        "audout::IAudioOut"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            0 => ctx.push_u32(u32::from(!self.running)), // GetAudioOutState
            1 => self.running = true,
            2 => self.running = false,
            // AppendAudioOutBuffer(Auto)(tag; buffer): AudioOutBuffer
            // {next, samples, capacity, size, offset}, PCM16 stereo.
            3 | 7 => {
                let b = ctx.read_in_buffer(0)?;
                let field = |o: usize| {
                    b.get(o..o + 8)
                        .map_or(0, |s| u64::from_le_bytes(s.try_into().unwrap()))
                };
                let (addr, size, offset) = (field(8), field(0x18), field(0x20));
                if self.running && addr != 0 && size > 0 && size < 1 << 22 {
                    let mut pcm = vec![0u8; size as usize];
                    if ctx.mem.read(addr + offset, &mut pcm).is_ok() {
                        let samples: Vec<f32> = pcm
                            .chunks_exact(2)
                            .map(|s| f32::from(i16::from_le_bytes([s[0], s[1]])) / 32768.0)
                            .collect();
                        ctx.system.audio_out.push(&samples, 48_000);
                    }
                }
                self.released.push(ctx.in_u64(0));
            }
            4 => ctx.push_event(&mut self.event, false), // RegisterBufferEvent
            // GetReleasedAudioOutBuffers(Auto): tags into the buffer.
            5 | 8 => {
                let cap = ctx.out_buffer(0).map_or(0, |b| b.size as usize / 8);
                let n = self.released.len().min(cap);
                let tags: Vec<u8> = self
                    .released
                    .drain(..n)
                    .flat_map(u64::to_le_bytes)
                    .collect();
                if n > 0 {
                    ctx.write_out_buffer(0, &tags)?;
                }
                ctx.push_u32(n as u32);
            }
            6 => ctx.push_u8(u8::from(self.released.contains(&ctx.in_u64(0)))), // ContainsAudioOutBuffer
            9 => ctx.push_u32(self.released.len() as u32), // GetAudioOutBufferCount
            10 => ctx.push_u64(0),                         // GetAudioOutPlayedSampleCount
            11 => ctx.push_u8(1),                          // FlushAudioOutBuffers
            12 => {}                                       // SetAudioOutVolume
            13 => ctx.push_u32(1.0f32.to_bits()),          // GetAudioOutVolume
            _ => return Err(CallError::Unimplemented),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_output_layout() {
        let p = RendererParams {
            sample_rate: 48_000,
            sample_count: 240,
            mixes: 2,
            voices: 3,
            sinks: 1,
            effects: 1,
            revision: 0,
        };
        // Input: "REv13" header with a behaviour section of 0x10 and the
        // memory pools (1 + 12) of 0x20, the first requesting attach.
        let pools = p.memory_pools() as usize;
        let mut input = vec![0u8; 0x40 + 0x10 + pools * 0x20];
        input[..4].copy_from_slice(&(u32::from_le_bytes(*b"REv0") + 13).to_le_bytes());
        input[4..8].copy_from_slice(&0x10u32.to_le_bytes());
        input[8..12].copy_from_slice(&((pools * 0x20) as u32).to_le_bytes());
        input[0x50 + 0x10..0x50 + 0x14].copy_from_slice(&4u32.to_le_bytes());
        let out = update_output(&p, &input, 7, None);
        let w = |o: usize| u32::from_le_bytes(out[o..o + 4].try_into().unwrap());
        assert_eq!(w(0), u32::from_le_bytes(input[..4].try_into().unwrap()));
        assert_eq!(w(8), (pools * 0x10) as u32);
        assert_eq!(w(0x0C), 3 * 0x10);
        assert_eq!(w(0x14), 0x90); // effect status version 2
        assert_eq!(w(0x1C), 0x20);
        assert_eq!(w(0x3C) as usize, out.len());
        assert_eq!(w(0x40), 5); // first pool attached
        assert_eq!(
            u64::from_le_bytes(out[out.len() - 0x10..out.len() - 8].try_into().unwrap()),
            7
        );
    }
}
