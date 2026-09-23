//! Biquad filter bank: seven filter types from the RBJ cookbook.
//!
//! Each [`Biquad`] is a stateful second-order IIR section (transposed direct
//! form II). Coefficients follow Robert Bristow-Johnson's "Audio EQ Cookbook"
//! with `a0` normalized to 1. `gain_db` only affects the peaking and shelving
//! types; the others ignore it.

use std::f32::consts::PI;

/// The seven filter types in the bank.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterKind {
    LowPass,
    HighPass,
    BandPass,
    Notch,
    Peaking,
    LowShelf,
    HighShelf,
}

impl FilterKind {
    /// Parse the kebab-case names used by the demo page / WASM boundary.
    pub fn from_name(name: &str) -> Option<FilterKind> {
        match name {
            "lowpass" => Some(FilterKind::LowPass),
            "highpass" => Some(FilterKind::HighPass),
            "bandpass" => Some(FilterKind::BandPass),
            "notch" => Some(FilterKind::Notch),
            "peaking" => Some(FilterKind::Peaking),
            "lowshelf" => Some(FilterKind::LowShelf),
            "highshelf" => Some(FilterKind::HighShelf),
            _ => None,
        }
    }

    /// Whether this filter type uses the gain parameter.
    pub fn uses_gain(self) -> bool {
        matches!(
            self,
            FilterKind::Peaking | FilterKind::LowShelf | FilterKind::HighShelf
        )
    }
}

/// A stateful biquad section. Create with [`Biquad::new`], then feed samples
/// through [`Biquad::process`] / [`Biquad::process_block`].
#[derive(Clone, Debug)]
pub struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    s1: f32,
    s2: f32,
}

impl Biquad {
    /// Build a filter.
    ///
    /// - `sample_rate`: Hz, e.g. 48000.0
    /// - `freq`: center/cutoff frequency in Hz, must be in `(0, sample_rate/2)`
    /// - `q`: resonance / bandwidth control, must be > 0
    /// - `gain_db`: peak/shelf gain in dB (peaking & shelves only)
    ///
    /// Out-of-range parameters are clamped into a safe range rather than
    /// panicking, since they arrive from UI sliders.
    pub fn new(kind: FilterKind, sample_rate: f32, freq: f32, q: f32, gain_db: f32) -> Biquad {
        let fs = sample_rate.max(1.0);
        let f0 = freq.clamp(fs * 1e-4, fs * 0.499);
        let q = q.clamp(0.05, 20.0);
        let a = 10.0f32.powf(gain_db.clamp(-24.0, 24.0) / 40.0);

        let w0 = 2.0 * PI * f0 / fs;
        let (sin_w0, cos_w0) = (w0.sin(), w0.cos());
        let alpha = sin_w0 / (2.0 * q);

        // Raw RBJ coefficients; normalized by a0 below.
        let (b0, b1, b2, a0, a1, a2) = match kind {
            FilterKind::LowPass => {
                let (b0, b1, b2) = ((1.0 - cos_w0) / 2.0, 1.0 - cos_w0, (1.0 - cos_w0) / 2.0);
                (b0, b1, b2, 1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha)
            }
            FilterKind::HighPass => {
                let (b0, b1, b2) = ((1.0 + cos_w0) / 2.0, -(1.0 + cos_w0), (1.0 + cos_w0) / 2.0);
                (b0, b1, b2, 1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha)
            }
            FilterKind::BandPass => {
                // Constant 0 dB peak-gain variant.
                (alpha, 0.0, -alpha, 1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha)
            }
            FilterKind::Notch => (
                1.0,
                -2.0 * cos_w0,
                1.0,
                1.0 + alpha,
                -2.0 * cos_w0,
                1.0 - alpha,
            ),
            FilterKind::Peaking => {
                let (b0, b1, b2) = (1.0 + alpha * a, -2.0 * cos_w0, 1.0 - alpha * a);
                (b0, b1, b2, 1.0 + alpha / a, -2.0 * cos_w0, 1.0 - alpha / a)
            }
            FilterKind::LowShelf => {
                let sq = 2.0 * alpha * a.sqrt();
                let b0 = a * ((a + 1.0) - (a - 1.0) * cos_w0 + sq);
                let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0);
                let b2 = a * ((a + 1.0) - (a - 1.0) * cos_w0 - sq);
                let a0 = (a + 1.0) + (a - 1.0) * cos_w0 + sq;
                let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0);
                let a2 = (a + 1.0) + (a - 1.0) * cos_w0 - sq;
                (b0, b1, b2, a0, a1, a2)
            }
            FilterKind::HighShelf => {
                let sq = 2.0 * alpha * a.sqrt();
                let b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + sq);
                let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0);
                let b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - sq);
                let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + sq;
                let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0);
                let a2 = (a + 1.0) - (a - 1.0) * cos_w0 - sq;
                (b0, b1, b2, a0, a1, a2)
            }
        };

        Biquad {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            s1: 0.0,
            s2: 0.0,
        }
    }

    /// Process one sample (transposed direct form II).
    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.s1;
        self.s1 = self.b1 * x - self.a1 * y + self.s2;
        self.s2 = self.b2 * x - self.a2 * y;
        y
    }

    /// Process a block in place.
    pub fn process_block(&mut self, buf: &mut [f32]) {
        for s in buf.iter_mut() {
            *s = self.process(*s);
        }
    }

    /// Clear internal state (e.g. when the input source changes).
    pub fn reset(&mut self) {
        self.s1 = 0.0;
        self.s2 = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    const FS: f32 = 48000.0;

    /// Steady-state amplitude of `filter` to a sine at `freq`.
    fn sine_gain(kind: FilterKind, freq: f32, q: f32, gain_db: f32, test_freq: f32) -> f32 {
        let mut f = Biquad::new(kind, FS, freq, q, gain_db);
        // Settle the transient first.
        for i in 0..(FS as usize) {
            let x = (2.0 * PI * test_freq * i as f32 / FS).sin();
            f.process(x);
        }
        let n = 4096;
        let mut peak = 0.0f32;
        for i in 0..n {
            let x = (2.0 * PI * test_freq * i as f32 / FS).sin();
            peak = peak.max(f.process(x).abs());
        }
        peak
    }

    #[test]
    fn lowpass_passes_dc_blocks_nyquist() {
        assert!((sine_gain(FilterKind::LowPass, 1000.0, 0.707, 0.0, 20.0) - 1.0).abs() < 0.01);
        assert!(sine_gain(FilterKind::LowPass, 1000.0, 0.707, 0.0, 20000.0) < 0.01);
    }

    #[test]
    fn highpass_blocks_dc_passes_high() {
        // DC via a constant block.
        let mut f = Biquad::new(FilterKind::HighPass, FS, 1000.0, 0.707, 0.0);
        let mut buf = vec![1.0f32; 48000];
        f.process_block(&mut buf);
        assert!(
            buf[47999].abs() < 1e-3,
            "highpass leaked DC: {}",
            buf[47999]
        );
        assert!((sine_gain(FilterKind::HighPass, 1000.0, 0.707, 0.0, 10000.0) - 1.0).abs() < 0.05);
    }

    #[test]
    fn notch_kills_its_center() {
        let killed = sine_gain(FilterKind::Notch, 1000.0, 4.0, 0.0, 1000.0);
        let kept = sine_gain(FilterKind::Notch, 1000.0, 4.0, 0.0, 100.0);
        assert!(killed < 0.05, "notch didn't kill center: {killed}");
        assert!((kept - 1.0).abs() < 0.05, "notch hurt far band: {kept}");
    }

    #[test]
    fn peaking_zero_db_is_identity() {
        let mut f = Biquad::new(FilterKind::Peaking, FS, 2000.0, 1.0, 0.0);
        let mut max_err = 0.0f32;
        for i in 0..8192 {
            let x = (2.0 * PI * 440.0 * i as f32 / FS).sin();
            max_err = max_err.max((f.process(x) - x).abs());
        }
        assert!(max_err < 1e-4, "0 dB peaking not identity: {max_err}");
    }

    #[test]
    fn peaking_boost_matches_gain() {
        // +6 dB at center => amplitude ratio ~2.
        let g = sine_gain(FilterKind::Peaking, 1000.0, 1.0, 6.0, 1000.0);
        assert!((g - 2.0).abs() < 0.1, "peaking +6dB gave {g}");
    }

    #[test]
    fn shelves_tilt_the_ends() {
        // +6 dB lowshelf at 200 Hz should roughly double deep bass, ~unity at 10 kHz.
        let bass = sine_gain(FilterKind::LowShelf, 200.0, 0.707, 6.0, 30.0);
        let treble = sine_gain(FilterKind::LowShelf, 200.0, 0.707, 6.0, 10000.0);
        assert!((bass - 2.0).abs() < 0.15, "lowshelf bass gain {bass}");
        assert!(
            (treble - 1.0).abs() < 0.05,
            "lowshelf leaked into treble {treble}"
        );
        let bass2 = sine_gain(FilterKind::HighShelf, 8000.0, 0.707, 6.0, 30.0);
        let treble2 = sine_gain(FilterKind::HighShelf, 8000.0, 0.707, 6.0, 18000.0);
        assert!(
            (bass2 - 1.0).abs() < 0.05,
            "highshelf leaked into bass {bass2}"
        );
        assert!(
            (treble2 - 2.0).abs() < 0.15,
            "highshelf treble gain {treble2}"
        );
    }

    #[test]
    fn bandpass_peaks_near_center() {
        let at = sine_gain(FilterKind::BandPass, 1000.0, 2.0, 0.0, 1000.0);
        let off = sine_gain(FilterKind::BandPass, 1000.0, 2.0, 0.0, 4000.0);
        assert!(at > 0.9, "bandpass center gain {at}");
        assert!(off < 0.3, "bandpass off-center leaked {off}");
    }

    #[test]
    fn reset_clears_ring() {
        let mut f = Biquad::new(FilterKind::LowPass, FS, 500.0, 2.0, 0.0);
        for _ in 0..1000 {
            f.process(1.0);
        }
        f.reset();
        assert_eq!(f.process(0.0), 0.0);
    }

    #[test]
    fn from_name_roundtrip() {
        for (name, kind) in [
            ("lowpass", FilterKind::LowPass),
            ("highpass", FilterKind::HighPass),
            ("bandpass", FilterKind::BandPass),
            ("notch", FilterKind::Notch),
            ("peaking", FilterKind::Peaking),
            ("lowshelf", FilterKind::LowShelf),
            ("highshelf", FilterKind::HighShelf),
        ] {
            assert_eq!(FilterKind::from_name(name), Some(kind));
        }
        assert_eq!(FilterKind::from_name("wah"), None);
    }
}
