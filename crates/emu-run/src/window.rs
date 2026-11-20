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
use std::cell::RefCell;
use std::rc::Rc;

/// The window, once the first frame opened it.
#[derive(Clone, Default)]
pub struct Shown(Rc<RefCell<Option<Window>>>);

impl Shown {
    /// Keeps the window (with the last frame) open until it is closed or
    /// Escape is pressed. Does nothing if no frame was shown.
    pub fn wait_until_closed(&self) {
        let mut w = self.0.borrow_mut();
        let Some(win) = w.as_mut() else { return };
        if !win.is_open() || win.is_key_down(Key::Escape) {
            return;
        }
        eprintln!("[emu-run] the program has stopped; close the window or press Escape");
        while win.is_open() && !win.is_key_down(Key::Escape) {
            win.update();
            std::thread::sleep(std::time::Duration::from_millis(16));
        }
    }
}

/// Returns a frame callback that draws into a window; it returns false
/// once the window is closed or Escape is pressed. `Shown` gives access
/// to the window after the program stops.
pub fn presenter(title: &str) -> (Presenter, Shown) {
    let title = title.to_string();
    let shown = Shown::default();
    let window = shown.clone();
    let mut pixels: Vec<u32> = Vec::new();
    let present = Box::new(move |frame: &Frame, input: &mut InputState| {
        let (w, h) = (frame.width as usize, frame.height as usize);
        let mut window = window.0.borrow_mut();
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
    });
    (present, shown)
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
        (Key::R, button::ZL),
        (Key::F, button::ZR),
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
    #[cfg_attr(not(windows), allow(unused_mut))]
    let mut state = InputState {
        buttons,
        stick_l: (axis(Key::A, Key::D), axis(Key::S, Key::W)),
        stick_r: (axis(Key::J, Key::L), axis(Key::K, Key::I)),
    };
    #[cfg(windows)]
    xinput::merge(&mut state);
    state
}

/// The first connected XInput (Xbox-style) controller, mapped by
/// position to the Switch layout: the bottom face button is B, the right
/// one A, and so on.
#[cfg(windows)]
mod xinput {
    use super::button;
    use emu_horizon::services::hid::InputState;
    use windows_sys::Win32::UI::Input::XboxController::*;

    pub fn merge(state: &mut InputState) {
        for user in 0..4 {
            // SAFETY: XInputGetState writes a XINPUT_STATE.
            let mut s: XINPUT_STATE = unsafe { std::mem::zeroed() };
            if unsafe { XInputGetState(user, &mut s) } != 0 {
                continue;
            }
            let pad = s.Gamepad;
            let b = u32::from(pad.wButtons);
            for (mask, bit) in [
                (XINPUT_GAMEPAD_A, button::B),
                (XINPUT_GAMEPAD_B, button::A),
                (XINPUT_GAMEPAD_X, button::Y),
                (XINPUT_GAMEPAD_Y, button::X),
                (XINPUT_GAMEPAD_LEFT_SHOULDER, button::L),
                (XINPUT_GAMEPAD_RIGHT_SHOULDER, button::R),
                (XINPUT_GAMEPAD_START, button::PLUS),
                (XINPUT_GAMEPAD_BACK, button::MINUS),
                (XINPUT_GAMEPAD_DPAD_LEFT, button::LEFT),
                (XINPUT_GAMEPAD_DPAD_RIGHT, button::RIGHT),
                (XINPUT_GAMEPAD_DPAD_UP, button::UP),
                (XINPUT_GAMEPAD_DPAD_DOWN, button::DOWN),
                (XINPUT_GAMEPAD_LEFT_THUMB, button::STICK_L),
                (XINPUT_GAMEPAD_RIGHT_THUMB, button::STICK_R),
            ] {
                if b & u32::from(mask) != 0 {
                    state.buttons |= bit;
                }
            }
            if pad.bLeftTrigger > 64 {
                state.buttons |= button::ZL;
            }
            if pad.bRightTrigger > 64 {
                state.buttons |= button::ZR;
            }
            // Sticks, with a dead zone; keyboard input wins when held.
            let axis = |v: i16| {
                let v = i32::from(v).clamp(-32767, 32767);
                if v.abs() < 7849 {
                    0
                } else {
                    v
                }
            };
            if state.stick_l == (0, 0) {
                state.stick_l = (axis(pad.sThumbLX), axis(pad.sThumbLY));
            }
            if state.stick_r == (0, 0) {
                state.stick_r = (axis(pad.sThumbRX), axis(pad.sThumbRY));
            }
            return;
        }
    }
}
