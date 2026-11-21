//! Host audio output: plays `System::audio_out` (48 kHz stereo) on the
//! default output device with cpal.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};
use emu_horizon::services::AudioOutput;

/// Starts playback; keep the returned stream alive while the guest runs.
pub fn start(queue: AudioOutput) -> Option<cpal::Stream> {
    let device = cpal::default_host().default_output_device()?;
    let config = device.default_output_config().ok()?;
    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => build::<f32>(&device, config.into(), queue),
        cpal::SampleFormat::I16 => build::<i16>(&device, config.into(), queue),
        cpal::SampleFormat::I32 => build::<i32>(&device, config.into(), queue),
        cpal::SampleFormat::U16 => build::<u16>(&device, config.into(), queue),
        f => {
            eprintln!("[emu-run] audio: unsupported output format {f}");
            return None;
        }
    };
    match stream {
        Ok(s) => {
            if let Err(e) = s.play() {
                eprintln!("[emu-run] audio: {e}");
                return None;
            }
            Some(s)
        }
        Err(e) => {
            eprintln!("[emu-run] audio: {e}");
            None
        }
    }
}

fn build<T>(
    device: &cpal::Device,
    config: cpal::StreamConfig,
    queue: AudioOutput,
) -> Result<cpal::Stream, cpal::Error>
where
    T: SizedSample + FromSample<f32>,
{
    let channels = usize::from(config.channels).max(1);
    // Guest frames per device frame.
    let step = f64::from(AudioOutput::RATE) / f64::from(config.sample_rate.max(1));
    let mut pos = 0f64;
    let mut frame = [0f32; 2];
    let mut buf = Vec::new();
    device.build_output_stream(
        config,
        move |out: &mut [T], _: &cpal::OutputCallbackInfo| {
            let frames = out.len() / channels;
            // Pull the guest frames this callback covers.
            let need = (pos + frames as f64 * step).floor() as usize;
            buf.resize(2 * need, 0.0);
            let got = queue.pop(&mut buf) / 2;
            for (i, o) in out.chunks_mut(channels).enumerate() {
                let k = (pos + i as f64 * step).floor() as usize;
                if k < got {
                    frame = [buf[2 * k], buf[2 * k + 1]];
                } else {
                    // Starved: fade the last sample out.
                    frame = frame.map(|s| s * 0.9);
                }
                for (c, s) in o.iter_mut().enumerate() {
                    let v = match (channels, c) {
                        (1, _) => 0.5 * (frame[0] + frame[1]),
                        (_, 0) => frame[0],
                        (_, 1) => frame[1],
                        _ => 0.0,
                    };
                    *s = T::from_sample(v.clamp(-1.0, 1.0));
                }
            }
            pos = (pos + frames as f64 * step).fract();
        },
        |e| eprintln!("[emu-run] audio: {e}"),
        None,
    )
}
