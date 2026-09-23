//! Hand-rolled radix-2 Cooley–Tukey FFT and magnitude spectra.
//!
//! Everything here is `f32` and dependency-free. For the demo page we only
//! need forward spectra; the inverse is omitted on purpose (smaller API,
//! smaller WASM).

use std::f32::consts::PI;

/// In-place forward FFT on `re`/`im`. Length must be a power of two, >= 2.
pub fn fft_in_place(re: &mut [f32], im: &mut [f32]) {
    let n = re.len();
    assert!(n == im.len(), "re/im length mismatch");
    assert!(
        n.is_power_of_two() && n >= 2,
        "FFT length must be a power of two >= 2"
    );

    // Bit-reversal permutation.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    // Butterflies, smallest to largest.
    let mut len = 2;
    while len <= n {
        let half = len >> 1;
        let theta = -2.0 * PI / len as f32;
        let (w_len_re, w_len_im) = (theta.cos(), theta.sin());
        let mut start = 0;
        while start < n {
            let (mut w_re, mut w_im) = (1.0f32, 0.0f32);
            for k in 0..half {
                let a = start + k;
                let b = a + half;
                let vr = re[b] * w_re - im[b] * w_im;
                let vi = re[b] * w_im + im[b] * w_re;
                let (ur, ui) = (re[a], im[a]);
                re[a] = ur + vr;
                im[a] = ui + vi;
                re[b] = ur - vr;
                im[b] = ui - vi;
                let t = w_re;
                w_re = w_re * w_len_re - w_im * w_len_im;
                w_im = t * w_len_im + w_im * w_len_re;
            }
            start += len;
        }
        len <<= 1;
    }
}

/// Magnitude spectrum in dB, length `n/2`.
///
/// Applies a Hann window first (tames spectral leakage from the rectangular
/// window), then returns `20*log10(|X[k]| / n)` per bin. Silence floors at
/// about -240 dB via a small epsilon clamp rather than `-inf`.
pub fn spectrum_db(input: &[f32]) -> Vec<f32> {
    let n = input.len();
    assert!(
        n.is_power_of_two() && n >= 2,
        "spectrum length must be a power of two >= 2"
    );
    let mut re = vec![0.0f32; n];
    let mut im = vec![0.0f32; n];
    for (i, &s) in input.iter().enumerate() {
        // Hann window.
        let w = 0.5 - 0.5 * (2.0 * PI * i as f32 / n as f32).cos();
        re[i] = s * w;
    }
    fft_in_place(&mut re, &mut im);
    let norm = n as f32;
    (0..n / 2)
        .map(|k| {
            let mag = (re[k] * re[k] + im[k] * im[k]).sqrt() / norm;
            20.0 * mag.max(1e-12).log10()
        })
        .collect()
}

/// Bin index of the strongest spectral peak (argmax of [`spectrum_db`]).
pub fn peak_bin(input: &[f32]) -> usize {
    let spec = spectrum_db(input);
    spec.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn sine(n: usize, bin: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI * bin as f32 * i as f32 / n as f32).sin())
            .collect()
    }

    #[test]
    fn impulse_gives_flat_spectrum() {
        let n = 256;
        let mut x = vec![0.0f32; n];
        // NB: at n/2, not 0 — the Hann window is exactly 0 at the endpoints,
        // so an impulse at index 0 would test nothing but the -240 dB floor.
        x[n / 2] = 1.0;
        let spec = spectrum_db(&x);
        assert_eq!(spec.len(), n / 2);
        let (min, max) = spec
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| {
                (a.min(v), b.max(v))
            });
        // Hann-windowed impulse: flat within a few dB (window spreads energy evenly).
        assert!(max - min < 6.0, "impulse spectrum not flat: {min}..{max}");
    }

    #[test]
    fn sine_peaks_at_its_bin() {
        let n = 512;
        let x = sine(n, 40);
        assert_eq!(peak_bin(&x), 40);
    }

    #[test]
    fn fft_roundtrip_energy() {
        // Parseval-ish sanity: a unit sine's peak bin dominates the spectrum.
        // (Excludes the ±2 neighbor bins: the Hann main lobe legitimately
        // spreads ~6 dB of energy into them.)
        let n = 256;
        let x = sine(n, 32);
        let spec = spectrum_db(&x);
        let peak = spec[32];
        let second = spec
            .iter()
            .enumerate()
            .filter(|(i, _)| i.abs_diff(32) > 2)
            .map(|(_, v)| *v)
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(
            peak - second > 20.0,
            "sine peak not dominant: {peak} vs {second}"
        );
    }

    #[test]
    #[should_panic]
    fn rejects_non_power_of_two() {
        spectrum_db(&vec![0.0f32; 100]);
    }
}
