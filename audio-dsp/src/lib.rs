//! audio-dsp: from-scratch educational audio DSP building blocks.
//!
//! A portable, dependency-free core: the same code compiles to native and to
//! WebAssembly (see `web/src/audio-wasm`). Modules:
//!
//! - [`fft`]: radix-2 FFT and dB magnitude spectra.
//! - [`filter`]: RBJ-cookbook biquad filter bank (low/high/band-pass, notch,
//!   peaking, low/high-shelf), stateful per instance.
//! - [`pitch`]: YIN fundamental-frequency estimation.
//! - [`note`]: frequency to note-name + cents mapping.
//!
//! All DSP takes the sample rate as a parameter — nothing is hardcoded to
//! 44.1 kHz or 48 kHz, so mic input, decoded files, and synthesized tones
//! share one code path.

pub mod fft;
pub mod filter;
pub mod note;
pub mod pitch;
