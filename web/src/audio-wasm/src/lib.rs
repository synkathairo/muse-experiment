//! WASM bindings for the audio-dsp crate.
//!
//! Web Audio handles I/O only (mic capture, file decode, tone generation,
//! playback). All DSP — filtering, spectra, pitch — runs in the Rust core so
//! the same code compiles to native targets unchanged.
//!
//! Boundary shape:
//! - [`Lab`]: stateful filter chain. Feed it frames; get filtered frames.
//! - [`spectrum_db`]: dB magnitude spectrum as `Float32Array` (per-frame, so
//!   no JSON encoding overhead).
//! - [`pitch_json`]: YIN estimate + note mapping as JSON (called ~10x/sec,
//!   so JSON is fine here).

use audio_dsp::filter::{Biquad, FilterKind};
use js_sys::Float32Array;
use wasm_bindgen::prelude::*;

/// Stateful filter chain: samples in, filtered samples out.
#[wasm_bindgen]
pub struct Lab {
    sample_rate: f32,
    filters: Vec<Biquad>,
}

#[wasm_bindgen]
impl Lab {
    /// Create a chain for `sample_rate` Hz (use the source's real rate:
    /// mic/tone usually 48000, decoded files usually 44100).
    #[wasm_bindgen(constructor)]
    pub fn new(sample_rate: f32) -> Lab {
        Lab {
            sample_rate,
            filters: Vec::new(),
        }
    }

    /// Replace the chain with a single filter. `kind` is one of:
    /// lowpass, highpass, bandpass, notch, peaking, lowshelf, highshelf.
    /// An unknown kind clears the chain. Returns false on unknown kind.
    pub fn set_filter(&mut self, kind: &str, freq: f32, q: f32, gain_db: f32) -> bool {
        match FilterKind::from_name(kind) {
            Some(k) => {
                self.filters = vec![Biquad::new(k, self.sample_rate, freq, q, gain_db)];
                true
            }
            None => {
                self.filters.clear();
                false
            }
        }
    }

    /// Remove all filters (bypass: output == input).
    pub fn clear_filters(&mut self) {
        self.filters.clear();
    }

    /// Drop filter state (call when the input source changes to avoid a thump).
    pub fn reset(&mut self) {
        for f in &mut self.filters {
            f.reset();
        }
    }

    /// Run the filter chain over `input`, returning the filtered samples.
    pub fn process(&mut self, input: &[f32]) -> Float32Array {
        let mut out = input.to_vec();
        for f in &mut self.filters {
            f.process_block(&mut out);
        }
        Float32Array::from(out.as_slice())
    }
}

/// dB magnitude spectrum of `input` (length must be a power of two).
/// Returns `n/2` bins; bin `k` is `k * sample_rate / n` Hz.
#[wasm_bindgen]
pub fn spectrum_db(input: &[f32]) -> Float32Array {
    Float32Array::from(audio_dsp::fft::spectrum_db(input).as_slice())
}

/// YIN pitch estimate of `input` at `sample_rate` Hz, as JSON:
/// `{"freq":440.2,"confidence":0.97,"note":"A4","cents":1.8}`
/// or `{"pitch":null}` when the input is silent/unpitched.
#[wasm_bindgen]
pub fn pitch_json(input: &[f32], sample_rate: f32) -> String {
    match audio_dsp::pitch::yin_pitch(input, sample_rate) {
        Some(p) => {
            let (note, cents) = audio_dsp::note::note_of(p.freq).unwrap_or(("?".into(), 0.0));
            serde_json::json!({
                "freq": p.freq,
                "confidence": p.confidence,
                "note": note,
                "cents": cents,
            })
            .to_string()
        }
        None => r#"{"pitch":null}"#.to_string(),
    }
}
