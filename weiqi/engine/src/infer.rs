//! Hand-rolled f32 inference for the LOCKED net arch (PLAN.md §3).
//!
//! Architecture: 4 × conv(64ch, 3×3, pad 1) + ReLU trunk; policy head
//! conv(64→2, 1×1) → flatten → linear(162→82), softmax at inference
//! (index 81 = pass); value head conv(64→1, 1×1) → flatten → linear(81→32) →
//! ReLU → linear(32→1) → tanh.
//!
//! Weights load from the flat fp16 little-endian binary produced by
//! `train/gotrain/export.py` — no header, tensor order IS the format:
//! ```text
//! conv1 w[64,6,3,3] b[64]; conv2 w[64,64,3,3] b[64]; conv3 same; conv4 same;
//! pol_conv w[2,64,1,1] b[2]; pol_fc w[82,162] b[82];
//! val_conv w[1,64,1,1] b[1]; val_fc1 w[32,81] b[32]; val_fc2 w[1,32] b[1]
//! ```
//! Total: 130,522 params → 261,044 bytes. Pure Rust, no ML crates — this must
//! also compile to `wasm32-unknown-unknown`.

use crate::features::FEATURE_LEN;

/// Total trainable parameters in the LOCKED arch.
pub const N_PARAMS: usize = 130_522;
/// Exact byte length of a weight blob: one fp16 LE per param, no header.
pub const WEIGHT_BYTES: usize = N_PARAMS * 2; // 261,044

const TRUNK_C: usize = 64;
const BOARD_H: usize = 9;
const BOARD_W: usize = 9;
const POLICY_DIM: usize = 82; // 81 points row-major + pass at index 81

/// Weight loading failure.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WeightError {
    /// Blob length mismatch. Expected is always [`WEIGHT_BYTES`].
    BadLength { expected: usize, got: usize },
}

/// The LOCKED network with f32 weights (converted from fp16 on load).
#[derive(Debug)]
pub struct Net {
    c1_w: Vec<f32>,
    c1_b: Vec<f32>,
    c2_w: Vec<f32>,
    c2_b: Vec<f32>,
    c3_w: Vec<f32>,
    c3_b: Vec<f32>,
    c4_w: Vec<f32>,
    c4_b: Vec<f32>,
    pol_w: Vec<f32>,
    pol_b: Vec<f32>,
    pf_w: Vec<f32>,
    pf_b: Vec<f32>,
    val_w: Vec<f32>,
    val_b: Vec<f32>,
    vf1_w: Vec<f32>,
    vf1_b: Vec<f32>,
    vf2_w: Vec<f32>,
    vf2_b: Vec<f32>,
}

/// Network output: move probabilities and win-probability estimate
/// (from the side-to-move's perspective, tanh → (−1, 1)).
pub struct Output {
    /// Softmax over 82 logits; index 81 = pass.
    pub policy: [f32; POLICY_DIM],
    /// Value head output after tanh.
    pub value: f32,
}

impl Net {
    /// Load weights from a flat fp16 little-endian blob in the LOCKED tensor
    /// order (see module docs). Returns [`WeightError::BadLength`] unless the
    /// blob is exactly [`WEIGHT_BYTES`] long.
    pub fn from_fp16_le(bytes: &[u8]) -> Result<Net, WeightError> {
        if bytes.len() != WEIGHT_BYTES {
            return Err(WeightError::BadLength {
                expected: WEIGHT_BYTES,
                got: bytes.len(),
            });
        }
        let vals: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        debug_assert_eq!(vals.len(), N_PARAMS);
        let mut cur = 0usize;
        let mut take = |n: usize| -> Vec<f32> {
            let s = cur;
            cur += n;
            vals[s..s + n].to_vec()
        };
        Ok(Net {
            c1_w: take(64 * 6 * 3 * 3),
            c1_b: take(64),
            c2_w: take(64 * 64 * 3 * 3),
            c2_b: take(64),
            c3_w: take(64 * 64 * 3 * 3),
            c3_b: take(64),
            c4_w: take(64 * 64 * 3 * 3),
            c4_b: take(64),
            pol_w: take(2 * 64),
            pol_b: take(2),
            pf_w: take(82 * 162),
            pf_b: take(82),
            val_w: take(64),
            val_b: take(1),
            vf1_w: take(32 * 81),
            vf1_b: take(32),
            vf2_w: take(32),
            vf2_b: take(1),
        })
    }

    /// Forward pass. Input: 486 features from [`crate::features::encode`].
    /// Deterministic: same input → bit-identical output.
    pub fn forward(&self, input: &[f32; FEATURE_LEN]) -> Output {
        // Trunk: 4 × conv3×3 pad1 + ReLU (specialized branch-free kernel).
        let mut t = conv3x3_pad1(input, 6, &self.c1_w, &self.c1_b, TRUNK_C);
        relu(&mut t);
        for (w, b) in [
            (&self.c2_w, &self.c2_b),
            (&self.c3_w, &self.c3_b),
            (&self.c4_w, &self.c4_b),
        ] {
            t = conv3x3_pad1(&t, TRUNK_C, w, b, TRUNK_C);
            relu(&mut t);
        }

        // Policy head: conv 64→2 (1×1), flatten, linear 162→82, softmax.
        let p = conv2d(&t, TRUNK_C, &self.pol_w, &self.pol_b, 2, 1, 0);
        let logits = linear(&p, &self.pf_w, &self.pf_b, POLICY_DIM);
        let mut policy = [0.0f32; POLICY_DIM];
        policy.copy_from_slice(&logits);
        softmax(&mut policy);

        // Value head: conv 64→1 (1×1), flatten, linear 81→32, ReLU,
        // linear 32→1, tanh.
        let v = conv2d(&t, TRUNK_C, &self.val_w, &self.val_b, 1, 1, 0);
        let mut h = linear(&v, &self.vf1_w, &self.vf1_b, 32);
        relu(&mut h);
        let s = linear(&h, &self.vf2_w, &self.vf2_b, 1);

        Output {
            policy,
            value: s[0].tanh(),
        }
    }
}

/// 2-D convolution over a 9×9 board, no batching.
/// `input`: [in_c][9][9] row-major; `weight`: [out_c][in_c][k][k] row-major;
/// `bias`: [out_c]. Returns [out_c][9][9] row-major.
fn conv2d(
    input: &[f32],
    in_c: usize,
    weight: &[f32],
    bias: &[f32],
    out_c: usize,
    k: usize,
    pad: usize,
) -> Vec<f32> {
    let h = BOARD_H;
    let w = BOARD_W;
    let mut out = vec![0.0f32; out_c * h * w];
    for oc in 0..out_c {
        for r in 0..h {
            for c in 0..w {
                let mut acc = bias[oc];
                for ic in 0..in_c {
                    for kr in 0..k {
                        let ir = r as isize + kr as isize - pad as isize;
                        if ir < 0 || ir >= h as isize {
                            continue;
                        }
                        for kc in 0..k {
                            let icc = c as isize + kc as isize - pad as isize;
                            if icc < 0 || icc >= w as isize {
                                continue;
                            }
                            acc += weight[((oc * in_c + ic) * k + kr) * k + kc]
                                * input[(ic * h + ir as usize) * w + icc as usize];
                        }
                    }
                }
                out[(oc * h + r) * w + c] = acc;
            }
        }
    }
    out
}

/// 3×3 pad-1 convolution specialized for the trunk. The interior 7×7 cells
/// are computed branch-free with the 9 taps unrolled, so the inner channel
/// loop is a clean reduction that LLVM auto-vectorizes (notably to f32x4
/// with `+simd128` on wasm32); border cells use the scalar checked fallback.
/// Same math as `conv2d(input, in_c, weight, bias, out_c, 3, 1)` up to fp
/// reassociation noise (differentially tested).
fn conv3x3_pad1(
    input: &[f32],
    in_c: usize,
    weight: &[f32],
    bias: &[f32],
    out_c: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; out_c * 81];
    for oc in 0..out_c {
        let w_oc = &weight[oc * in_c * 9..(oc + 1) * in_c * 9];
        let out_oc = &mut out[oc * 81..(oc + 1) * 81];
        let b = bias[oc];
        // Interior: no bounds checks possible (r,c in 1..8, taps ±1 stay in
        // 0..9). Offsets are relative to the center tap x0.
        for r in 1..8usize {
            for c in 1..8usize {
                let mut acc = b;
                for ic in 0..in_c {
                    let w = &w_oc[ic * 9..ic * 9 + 9];
                    let x0 = ic * 81 + r * 9 + c;
                    acc += w[0] * input[x0 - 10];
                    acc += w[1] * input[x0 - 9];
                    acc += w[2] * input[x0 - 8];
                    acc += w[3] * input[x0 - 1];
                    acc += w[4] * input[x0];
                    acc += w[5] * input[x0 + 1];
                    acc += w[6] * input[x0 + 8];
                    acc += w[7] * input[x0 + 9];
                    acc += w[8] * input[x0 + 10];
                }
                out_oc[r * 9 + c] = acc;
            }
        }
        // Border ring: scalar with bounds checks.
        for r in 0..9usize {
            for c in 0..9usize {
                if (1..8).contains(&r) && (1..8).contains(&c) {
                    continue;
                }
                let mut acc = b;
                for ic in 0..in_c {
                    for kr in 0..3usize {
                        let ir = r as isize + kr as isize - 1;
                        if !(0..9).contains(&ir) {
                            continue;
                        }
                        for kc in 0..3usize {
                            let icc = c as isize + kc as isize - 1;
                            if !(0..9).contains(&icc) {
                                continue;
                            }
                            acc += w_oc[(ic * 3 + kr) * 3 + kc]
                                * input[(ic * 9 + ir as usize) * 9 + icc as usize];
                        }
                    }
                }
                out_oc[r * 9 + c] = acc;
            }
        }
    }
    out
}

/// Fully-connected layer: `y = Wx + b`, `weight` [out_dim][in_dim] row-major.
fn linear(x: &[f32], weight: &[f32], bias: &[f32], out_dim: usize) -> Vec<f32> {
    let in_dim = x.len();
    let mut y = vec![0.0f32; out_dim];
    for o in 0..out_dim {
        let mut acc = bias[o];
        for (i, &xi) in x.iter().enumerate() {
            acc += weight[o * in_dim + i] * xi;
        }
        y[o] = acc;
    }
    y
}

fn relu(v: &mut [f32]) {
    for x in v.iter_mut() {
        if *x < 0.0 {
            *x = 0.0;
        }
    }
}

/// In-place softmax, max-subtracted for numerical stability.
fn softmax(logits: &mut [f32]) {
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in logits.iter_mut() {
        *x = (*x - m).exp();
        sum += *x;
    }
    for x in logits.iter_mut() {
        *x /= sum;
    }
}

/// IEEE-754 binary16 → f32, bit manipulation. Handles zeros, subnormals,
/// normals, infinities, and NaNs.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    let f32bits = if exp == 0 {
        if mant == 0 {
            sign << 31 // signed zero
        } else {
            // Subnormal: value = mantissa × 2^-24. Renormalize to 1.xxx × 2^e.
            let mut m = mant;
            let mut e: i32 = -14;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            (sign << 31) | (((e + 127) as u32) << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (mant << 13) // inf / NaN
    } else {
        (sign << 31) | ((exp + 112) << 23) | (mant << 13) // 2^(exp-15) → 2^(exp+112-127)
    };
    f32::from_bits(f32bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic all-zero weight blob (correct length).
    fn zero_blob() -> Vec<u8> {
        vec![0u8; WEIGHT_BYTES]
    }

    #[test]
    fn tensor_order_sums_to_n_params() {
        // The LOCKED export order from the module docs, summed independently.
        let sizes = [
            64 * 6 * 3 * 3,
            64,
            64 * 64 * 3 * 3,
            64,
            64 * 64 * 3 * 3,
            64,
            64 * 64 * 3 * 3,
            64,
            2 * 64 * 1 * 1,
            2,
            82 * 162,
            82,
            1 * 64 * 1 * 1,
            1,
            32 * 81,
            32,
            1 * 32,
            1,
        ];
        assert_eq!(sizes.iter().sum::<usize>(), N_PARAMS);
        assert_eq!(N_PARAMS, 130_522);
        assert_eq!(WEIGHT_BYTES, 261_044);
    }

    #[test]
    fn rejects_bad_length() {
        assert_eq!(
            Net::from_fp16_le(&vec![0u8; 100]).unwrap_err(),
            WeightError::BadLength {
                expected: WEIGHT_BYTES,
                got: 100
            }
        );
        assert_eq!(
            Net::from_fp16_le(&vec![0u8; WEIGHT_BYTES + 2]).unwrap_err(),
            WeightError::BadLength {
                expected: WEIGHT_BYTES,
                got: WEIGHT_BYTES + 2
            }
        );
        assert!(Net::from_fp16_le(&zero_blob()).is_ok());
    }

    #[test]
    fn f16_conversion_spot_checks() {
        assert_eq!(f16_to_f32(0x3C00), 1.0); // 1.0
        assert_eq!(f16_to_f32(0xBC00), -1.0); // -1.0
        assert_eq!(f16_to_f32(0x3800), 0.5); // 0.5
        assert_eq!(f16_to_f32(0x0000), 0.0); // +0
        assert_eq!(f16_to_f32(0x8000), -0.0); // -0
        assert_eq!(f16_to_f32(0x7C00), f32::INFINITY); // inf
        assert!(f16_to_f32(0x7E00).is_nan()); // NaN
        // Smallest subnormal: 2^-24.
        assert!((f16_to_f32(0x0001) - 5.9604645e-8).abs() < 1e-14);
        // Largest normal below inf: 65504.
        assert_eq!(f16_to_f32(0x7BFF), 65504.0);
    }

    #[test]
    fn zero_weights_give_uniform_policy_and_zero_value() {
        // Every weight and bias is 0: trunk is all zeros, policy logits are
        // all zero -> uniform softmax; value path is 0 -> tanh(0) = 0.
        let net = Net::from_fp16_le(&zero_blob()).unwrap();
        let out = net.forward(&[0.0f32; FEATURE_LEN]);
        let expected = 1.0f32 / POLICY_DIM as f32;
        for &p in out.policy.iter() {
            assert!((p - expected).abs() < 1e-6, "p={p} expected={expected}");
        }
        let sum: f32 = out.policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert_eq!(out.value, 0.0);
        // Nonzero input still yields the same (all weights are zero).
        let mut feat = [0.0f32; FEATURE_LEN];
        feat[0] = 1.0;
        feat[485] = 1.0;
        let out2 = net.forward(&feat);
        assert_eq!(out.policy, out2.policy);
        assert_eq!(out.value, out2.value);
    }

    #[test]
    fn forward_is_deterministic() {
        let net = Net::from_fp16_le(&zero_blob()).unwrap();
        let mut feat = [0.0f32; FEATURE_LEN];
        for (i, x) in feat.iter_mut().enumerate() {
            *x = ((i * 7) % 3) as f32;
        }
        let a = net.forward(&feat);
        let b = net.forward(&feat);
        assert_eq!(a.policy, b.policy);
        assert_eq!(a.value, b.value);
    }

    #[test]
    fn conv3x3_pad1_matches_generic_conv2d() {
        // Deterministic pseudo-random weights/input; the two kernels must
        // agree up to fp reassociation noise.
        let mut s: u64 = 0x12345678;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s = s.wrapping_mul(0x2545F4914F6CDD1D);
            ((s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32
        };
        for &(in_c, out_c) in &[(6usize, 64usize), (64, 64), (64, 3)] {
            let input: Vec<f32> = (0..in_c * 81).map(|_| rnd()).collect();
            let weight: Vec<f32> = (0..out_c * in_c * 9).map(|_| rnd()).collect();
            let bias: Vec<f32> = (0..out_c).map(|_| rnd()).collect();
            let a = conv2d(&input, in_c, &weight, &bias, out_c, 3, 1);
            let b = conv3x3_pad1(&input, in_c, &weight, &bias, out_c);
            assert_eq!(a.len(), b.len());
            let max_diff = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff < 1e-4,
                "in_c={in_c} out_c={out_c}: max_diff={max_diff}"
            );
        }
    }

    #[test]
    fn conv3x3_matches_hand_computation() {
        // Single input spike at (4,4) in channel 0; kernel taps are 1..=9.
        // out[r][c] samples the spike through tap (kr,kc) = (5-r, 5-c), so
        // out[3..=5][3..=5] shows the kernel rotated 180°: 9 8 7 / 6 5 4 / 3 2 1.
        // (This is cross-correlation, matching PyTorch conv2d — which is what
        // the exported weights assume.)
        let mut input = vec![0.0f32; 1 * 81];
        input[4 * 9 + 4] = 1.0;
        let weight: Vec<f32> = (1..=9).map(|v| v as f32).collect();
        let out = conv2d(&input, 1, &weight, &[0.0], 1, 3, 1);
        for r in 0..9 {
            for c in 0..9 {
                let got = out[r * 9 + c];
                let expected = if (3..=5).contains(&r) && (3..=5).contains(&c) {
                    ((5 - r) * 3 + (5 - c) + 1) as f32
                } else {
                    0.0
                };
                assert_eq!(got, expected, "mismatch at ({r},{c})");
            }
        }
    }

    #[test]
    fn conv1x1_and_bias() {
        // 2 input channels (all 1.0 / all 2.0), weights [2,3], bias 1:
        // out = 2*1 + 3*2 + 1 = 9 everywhere.
        let mut input = vec![0.0f32; 2 * 81];
        for i in 0..81 {
            input[i] = 1.0;
            input[81 + i] = 2.0;
        }
        let out = conv2d(&input, 2, &[2.0, 3.0], &[1.0], 1, 1, 0);
        assert!(out.iter().all(|&x| x == 9.0));
    }

    #[test]
    fn softmax_is_monotone_and_normalized() {
        let mut v = [0.0f32, 1.0, 2.0, -1.0];
        softmax(&mut v);
        assert!((v.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(v[2] > v[1] && v[1] > v[0] && v[0] > v[3]);
        assert!(v.iter().all(|&x| x > 0.0));
    }

    #[test]
    fn full_net_output_shapes_sane() {
        // Encode a real mid-game position and run the zero net: shapes must
        // hold and policy must be a valid distribution.
        let mut g = crate::rules::Game::new();
        g.play(crate::rules::Move::Play(40)).unwrap();
        g.play(crate::rules::Move::Play(41)).unwrap();
        let feat = crate::features::encode(&g);
        let net = Net::from_fp16_le(&zero_blob()).unwrap();
        let out = net.forward(&feat);
        assert_eq!(out.policy.len(), 82);
        assert!((-1.0..=1.0).contains(&out.value));
    }
}
