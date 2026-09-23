# audio-dsp

From-scratch educational audio DSP in Rust. No dependencies — every algorithm
is implemented by hand so the demo page can show its work.

## Modules

- **`fft`** — radix-2 Cooley–Tukey FFT, Hann-windowed dB magnitude spectra.
- **`filter`** — seven biquad filter types (low/high/band-pass, notch, peaking,
  low/high-shelf) from the RBJ Audio EQ Cookbook, as stateful
  transposed-direct-form-II sections.
- **`pitch`** — YIN fundamental-frequency estimation (de Cheveigné & Kawahara,
  2002): periodicity in the time domain, so harmonics don't cause octave errors.
- **`note`** — frequency → note name + cents deviation (12-TET, A4 = 440 Hz).

## Design rules

- The sample rate is always a parameter, never hardcoded — mic input
  (usually 48 kHz), decoded files (usually 44.1 kHz), and synthesized tones
  share one code path.
- No browser code in this crate. The web demo lives in `web/src/audio-wasm`
  (thin wasm-bindgen wrapper) and `web/site/demos/audio/` (the page).

## Tests

```sh
cargo test
```

22 tests: spectral peak placement, filter DC/gain spot-checks against the
cookbook's expected responses, YIN on pure and harmonic-rich tones plus
silence/noise rejection, and note-mapping conventions.
