//! A window showing the frames the guest presents (minifb), and the
//! keyboard as a controller:
//!
//! | key            | button        |
//! |----------------|---------------|
//! | arrows         | D-pad         |
//! | W A S D        | left stick    |
//! | X / Z          | A / B         |
//! | C / V          | X / Y         |
//! | Q / E          | L / R         |
//! | Enter / Space  | Plus          |
//! | Backspace      | Minus         |
//! | Escape         | close         |

use emu_horizon::services::hid::{button, InputState};
use emu_horizon::services::{Frame, Presenter};
use minifb::{Key, ScaleMode, Window, WindowOptions};

/// Returns a frame callback that draws into a window; it returns false
/// once the window is closed or Escape is pressed.
pub fn presenter(title: &str) -> Presenter {
    let title = title.to_string();
    let mut window: Option<Window> = None;
    let mut pixels: Vec<u32> = Vec::new();
    Box::new(move |frame: &Frame, input: &mut InputState| {
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
        let ok = win.update_with_buffer(&pixels, w, h).is_ok();
        *input = read_keys(win);
        ok
    })
}

fn read_keys(win: &Window) -> InputState {
    let down = |k| win.is_key_down(k);
    let mut buttons = 0;
    for (key, bit) in [
        (Key::Left, button::LEFT),
        (Key::Right, button::RIGHT),
        (Key::Up, button::UP),
        (Key::Down, button::DOWN),
        (Key::X, button::A),
        (Key::Z, button::B),
        (Key::C, button::X),
        (Key::V, button::Y),
        (Key::Q, button::L),
        (Key::E, button::R),
        (Key::Enter, button::PLUS),
        (Key::Space, button::PLUS),
        (Key::Backspace, button::MINUS),
    ] {
        if down(key) {
            buttons |= bit;
        }
    }
    let axis = |neg, pos| match (down(neg), down(pos)) {
        (true, false) => -32767,
        (false, true) => 32767,
        _ => 0,
    };
    InputState {
        buttons,
        stick_l: (axis(Key::A, Key::D), axis(Key::S, Key::W)),
        stick_r: (0, 0),
    }
}
