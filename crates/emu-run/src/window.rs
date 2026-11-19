//! A window showing the frames the guest presents (minifb).

use emu_horizon::services::{Frame, Presenter};
use minifb::{Key, ScaleMode, Window, WindowOptions};

/// Returns a frame callback that draws into a window; it returns false
/// once the window is closed or Escape is pressed.
pub fn presenter(title: &str) -> Presenter {
    let title = title.to_string();
    let mut window: Option<Window> = None;
    let mut pixels: Vec<u32> = Vec::new();
    Box::new(move |frame: &Frame| {
        let (w, h) = (frame.width as usize, frame.height as usize);
        let win = match window.as_mut() {
            Some(win) => win,
            None => {
                let opts = WindowOptions {
                    resize: true,
                    scale_mode: ScaleMode::AspectRatioStretch,
                    ..WindowOptions::default()
                };
                match Window::new(&title, w, h, opts) {
                    Ok(win) => window.insert(win),
                    Err(e) => {
                        eprintln!("[emu-run] cannot open a window: {e}");
                        return false;
                    }
                }
            }
        };
        if !win.is_open() || win.is_key_down(Key::Escape) {
            return false;
        }
        pixels.clear();
        pixels.extend(
            frame
                .rgba
                .chunks_exact(4)
                .map(|p| u32::from(p[0]) << 16 | u32::from(p[1]) << 8 | u32::from(p[2])),
        );
        win.update_with_buffer(&pixels, w, h).is_ok()
    })
}
