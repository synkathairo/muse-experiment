//! YIN fundamental-frequency estimation (de Cheveigné & Kawahara, 2002).
//!
//! Pitch is periodicity in the time domain, not the tallest FFT bin — a
//! guitar string's loudest spectral peak is often a harmonic. YIN finds the
//! lag that best predicts the signal from itself via the difference function,
//! normalized by its own running mean, then refines with parabolic
//! interpolation. Returns `None` for silence or unvoiced input.

/// A pitch estimate: frequency in Hz and a 0..1 confidence.
#[derive(Clone, Copy, Debug)]
pub struct Pitch {
    pub freq: f32,
    pub confidence: f32,
}

/// Estimate the fundamental frequency of `samples`.
///
/// `sample_rate` in Hz. Needs at least a few periods in the window; 1024+
/// samples works well for 50 Hz–2 kHz. Returns `None` when the input is
/// silent or no lag passes the periodicity test.
pub fn yin_pitch(samples: &[f32], sample_rate: f32) -> Option<Pitch> {
    const THRESHOLD: f32 = 0.10; // cumulative-mean-normalized-difference gate
    const MIN_CONFIDENCE: f32 = 0.30;
    const MIN_HZ: f32 = 40.0;
    const MAX_HZ: f32 = 2000.0;

    let n = samples.len();
    if n < 64 || sample_rate <= 0.0 {
        return None;
    }

    // Energy gate: don't hallucinate pitch in silence.
    let energy = samples.iter().map(|x| x * x).sum::<f32>() / n as f32;
    if energy < 1e-8 {
        return None;
    }

    let max_tau = ((sample_rate / MIN_HZ) as usize).min(n / 2);
    let min_tau = ((sample_rate / MAX_HZ) as usize).max(2);
    if max_tau <= min_tau {
        return None;
    }

    // 1. Difference function: d[tau] = sum (x[j] - x[j+tau])^2.
    let mut diff = vec![0.0f32; max_tau + 1];
    for tau in 1..=max_tau {
        let mut s = 0.0f32;
        // Unrolled pair loop keeps this cheap: ~2M MACs for 2048 samples.
        let mut j = 0;
        while j + 1 < n - tau {
            let d0 = samples[j] - samples[j + tau];
            let d1 = samples[j + 1] - samples[j + 1 + tau];
            s += d0 * d0 + d1 * d1;
            j += 2;
        }
        if j < n - tau {
            let d0 = samples[j] - samples[j + tau];
            s += d0 * d0;
        }
        diff[tau] = s;
    }

    // 2. Cumulative mean normalized difference.
    let mut cmnd = vec![0.0f32; max_tau + 1];
    cmnd[0] = 1.0;
    let mut running = 0.0f32;
    for tau in 1..=max_tau {
        running += diff[tau];
        cmnd[tau] = diff[tau] * tau as f32 / running.max(1e-12);
    }

    // 3. First dip below threshold; walk to the local minimum (avoids
    // octave-up picks when a multiple of the period also dips).
    let mut best = None;
    for t in min_tau..=max_tau {
        if cmnd[t] < THRESHOLD {
            let mut m = t;
            while m < max_tau && cmnd[m + 1] < cmnd[m] {
                m += 1;
            }
            best = Some(m);
            break;
        }
    }
    let t = best?;

    // 4. Parabolic interpolation for sub-sample accuracy.
    let t_interp = if t > 0 && t < max_tau {
        let (a, b, c) = (cmnd[t - 1], cmnd[t], cmnd[t + 1]);
        let denom = a - 2.0 * b + c;
        if denom.abs() > 1e-9 {
            t as f32 + 0.5 * (a - c) / denom
        } else {
            t as f32
        }
    } else {
        t as f32
    };

    let confidence = (1.0 - cmnd[t]).clamp(0.0, 1.0);
    if confidence < MIN_CONFIDENCE {
        return None;
    }
    Some(Pitch {
        freq: sample_rate / t_interp,
        confidence,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn tone(freq: f32, fs: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / fs).sin())
            .collect()
    }

    #[test]
    fn finds_a440() {
        let p = yin_pitch(&tone(440.0, 48000.0, 2048), 48000.0).expect("no pitch");
        assert!((p.freq - 440.0).abs() < 1.5, "got {}", p.freq);
        assert!(p.confidence > 0.9, "confidence {}", p.confidence);
    }

    #[test]
    fn finds_low_e_string() {
        // Guitar low E: 82.41 Hz. Needs a long enough window for ~2 periods.
        let p = yin_pitch(&tone(82.41, 48000.0, 4096), 48000.0).expect("no pitch");
        assert!((p.freq - 82.41).abs() < 1.0, "got {}", p.freq);
    }

    #[test]
    fn rejects_octave_error_on_harmonic_rich_tone() {
        // Fundamental + strong 2nd/3rd harmonics; YIN should still pick 110.
        let n = 2048;
        let x: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / 48000.0;
                (2.0 * PI * 110.0 * t).sin()
                    + 0.8 * (2.0 * PI * 220.0 * t).sin()
                    + 0.6 * (2.0 * PI * 330.0 * t).sin()
            })
            .collect();
        let p = yin_pitch(&x, 48000.0).expect("no pitch");
        assert!((p.freq - 110.0).abs() < 2.0, "octave error: got {}", p.freq);
    }

    #[test]
    fn silence_gives_none() {
        assert!(yin_pitch(&vec![0.0f32; 2048], 48000.0).is_none());
        assert!(yin_pitch(&vec![1e-6f32; 2048], 48000.0).is_none());
    }

    #[test]
    fn noise_gives_none_or_low_confidence() {
        // Deterministic pseudo-noise; YIN should not confidently call it pitched.
        let mut s = 0x12345678u32;
        let x: Vec<f32> = (0..2048)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect();
        match yin_pitch(&x, 48000.0) {
            None => {}
            Some(p) => assert!(p.confidence < 0.9, "noise called pitched: {p:?}"),
        }
    }
}
