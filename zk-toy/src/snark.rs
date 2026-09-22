//! Homomorphic hiding, blind evaluation, and the prove/verify protocol.
//!
//! This follows the pairing-free toy construction (cf. Vitalik Buterin's
//! QAP blog series):
//!
//! * Homomorphic hiding: E(x) = h^x mod 1543, where h generates the unique
//!   order-257 subgroup of F_1543^*. Because h^257 = 1, exponents live in
//!   the additive group Z_257, which we identify with our circuit field
//!   F_257. The map is homomorphic: E(a)*E(b) = E(a+b), E(a)^k = E(k*a).
//!   (1543 is prime and 257 divides 1542 = 6*257, so such a subgroup exists.
//!   It is *toy-sized*: discrete logs are brute-forceable, which the
//!   verifier below actually does. Real SNARKs use pairings instead.)
//! * Trusted setup: sample secret s and alpha, publish the CRS
//!   { E(s^i) }, { E(alpha*s^i) } (i = 0..=3) and { E(s^i) } (i = 0..=2)
//!   for H. Discard s, alpha (the "toxic waste"). The prover never learns
//!   s or alpha; the verifier (who ran setup) knows them.
//! * Blind evaluation: the prover computes E(P(s)) = prod_i E(s^i)^{p_i}
//!   for their polynomials without ever learning s.
//! * KEA check: the prover also commits to the alpha-shifted evaluations
//!   E(alpha*P(s)); the verifier checks E_alpha == E^alpha. Without knowing
//!   alpha, the prover can only produce a consistent pair by applying the
//!   same coefficients to the shifted CRS (knowledge-of-exponent assumption).
//! * Divisibility check: the verifier recovers a=A(s), b=B(s), c=C(s),
//!   hh=H(s) by brute-force discrete log (toy-only!) and checks
//!   a*b - c == hh * t(s) in F_257.
//! * Public inputs: variables 0 ("one") and 5 ("out") are public. The prover
//!   commits only to the *private* part of A, B, C; the verifier adds the
//!   public contribution itself via the homomorphism, so a proof only
//!   verifies against the public inputs the verifier supplies.

use crate::field::Fp;
use crate::poly::Poly;
use crate::qap::Qap;
use crate::r1cs::{self, N_VARS};

/// Modulus for the hiding group. 1543 is prime and 257 | (1543 - 1).
pub const HIDE_MOD: u32 = 1543;

/// Indices of the public variables: "one" and "out".
pub const PUBLIC_VARS: [usize; 2] = [r1cs::VAR_ONE, r1cs::VAR_OUT];

/// a^e mod HIDE_MOD.
fn hpow(mut base: u32, mut exp: u32) -> u32 {
    let mut acc = 1u32;
    base %= HIDE_MOD;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = acc * base % HIDE_MOD;
        }
        base = base * base % HIDE_MOD;
        exp >>= 1;
    }
    acc
}

/// Find h of multiplicative order exactly 257 mod 1543.
/// Since 1543 is prime, for any g with g^6 != 1 the element h = g^6
/// satisfies h^257 = g^1542 = 1 and h != 1, hence has order 257 (prime).
pub fn subgroup_generator() -> u32 {
    for g in 2..HIDE_MOD {
        let h = hpow(g, 6);
        if h != 1 {
            debug_assert_eq!(hpow(h, 257), 1);
            return h;
        }
    }
    unreachable!("no generator found")
}

/// The hiding function E(x) = h^x mod 1543, for x in F_257.
pub fn hide(h: u32, x: Fp) -> u32 {
    hpow(h, x.value() as u32)
}

/// Brute-force discrete log: find x in 0..257 with h^x == target.
/// Only feasible because the group is toy-sized. Returns None if the
/// target is not in the subgroup (i.e. a malformed proof element).
pub fn dlog(h: u32, target: u32) -> Option<u32> {
    let mut cur = 1u32;
    for x in 0..257u32 {
        if cur == target {
            return Some(x);
        }
        cur = cur * h % HIDE_MOD;
    }
    None
}

/// Tiny deterministic PRNG (xorshift64*) so demos and tests are reproducible.
pub struct XorShift64(u64);

impl XorShift64 {
    pub fn new(seed: u64) -> Self {
        XorShift64(seed | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    pub fn next_fp(&mut self) -> Fp {
        Fp::new((self.next_u64() % 257) as u32)
    }
}

/// The common reference string plus the setup secrets (known to the
/// verifier / trusted setup, never to the prover).
pub struct Setup {
    pub s: Fp,
    pub alpha: Fp,
    pub h: u32,
    /// E(s^i) for i = 0..=3 (for A, B, C, which have degree <= 3).
    pub enc_powers: Vec<u32>,
    /// E(alpha * s^i) for i = 0..=3.
    pub enc_alpha_powers: Vec<u32>,
    /// E(s^i) for i = 0..=2 (for H, which has degree <= 2).
    pub enc_h_powers: Vec<u32>,
}

/// Run the trusted setup: sample s, alpha and build the CRS.
/// Resamples s if it lands on a root of t (where t(s) = 0 would make the
/// divisibility check vacuous).
pub fn setup(rng: &mut XorShift64, qap: &Qap) -> Setup {
    let h = subgroup_generator();
    loop {
        let s = rng.next_fp();
        if qap.t.eval(s).is_zero() {
            continue;
        }
        let alpha = rng.next_fp();
        // s^i in F_257, then hidden as h^(s^i).
        let mut s_pow = Fp::ONE;
        let mut enc_powers = Vec::with_capacity(4);
        let mut enc_alpha_powers = Vec::with_capacity(4);
        for _ in 0..4 {
            enc_powers.push(hide(h, s_pow));
            enc_alpha_powers.push(hide(h, alpha * s_pow));
            s_pow = s_pow * s;
        }
        let mut s_pow = Fp::ONE;
        let mut enc_h_powers = Vec::with_capacity(3);
        for _ in 0..3 {
            enc_h_powers.push(hide(h, s_pow));
            s_pow = s_pow * s;
        }
        return Setup { s, alpha, h, enc_powers, enc_alpha_powers, enc_h_powers };
    }
}

/// Evaluate E(P(s)) blindly: prod_i E(s^i)^{p_i}, using only the CRS.
fn blind_eval(enc_powers: &[u32], coeffs: &[Fp]) -> u32 {
    let mut acc = 1u32;
    for (i, &c) in coeffs.iter().enumerate() {
        acc = acc * hpow(enc_powers[i], c.value() as u32) % HIDE_MOD;
    }
    acc
}

/// A proof: blind evaluations of A, B, C (private parts), H, plus the
/// alpha-shifted evaluations of A, B, C for the KEA check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Proof {
    pub ea: u32,
    pub eb: u32,
    pub ec: u32,
    pub eh: u32,
    pub ea_alpha: u32,
    pub eb_alpha: u32,
    pub ec_alpha: u32,
}

/// Prove knowledge of a witness. The prover knows the witness and the CRS
/// (the enc_* fields of the setup) but NOT s or alpha.
pub fn prove(setup: &Setup, qap: &Qap, witness: &[Fp; N_VARS]) -> Result<Proof, &'static str> {
    let (a_mat, b_mat, c_mat) = r1cs::matrices();
    if !r1cs::satisfied(&a_mat, &b_mat, &c_mat, witness) {
        return Err("witness does not satisfy the R1CS");
    }
    let (a_full, b_full, c_full) = qap.combine(witness);
    let hpoly = qap
        .quotient(&a_full, &b_full, &c_full)
        .ok_or("QAP division not exact")?;
    debug_assert!(hpoly.degree() <= 2);

    // Split off the public part; the prover commits only to the private part.
    let priv_part = |full: &Poly, cols: &[Poly]| -> Poly {
        let mut p = full.clone();
        for &j in &PUBLIC_VARS {
            let term = &cols[j] * &Poly::constant(witness[j]);
            p = &p - &term;
        }
        p
    };
    let a_priv = priv_part(&a_full, &qap.a_polys);
    let b_priv = priv_part(&b_full, &qap.b_polys);
    let c_priv = priv_part(&c_full, &qap.c_polys);

    let ce = |poly: &Poly, powers: &[u32]| blind_eval(powers, poly.coeffs());
    Ok(Proof {
        ea: ce(&a_priv, &setup.enc_powers),
        eb: ce(&b_priv, &setup.enc_powers),
        ec: ce(&c_priv, &setup.enc_powers),
        eh: ce(&hpoly, &setup.enc_h_powers),
        ea_alpha: ce(&a_priv, &setup.enc_alpha_powers),
        eb_alpha: ce(&b_priv, &setup.enc_alpha_powers),
        ec_alpha: ce(&c_priv, &setup.enc_alpha_powers),
    })
}

/// Verify a proof against the given public inputs (must cover exactly the
/// PUBLIC_VARS indices). Returns false on any check failure.
pub fn verify(setup: &Setup, qap: &Qap, proof: &Proof, public_inputs: &[(usize, Fp)]) -> bool {
    // The public inputs must name exactly the public variables.
    if public_inputs.len() != PUBLIC_VARS.len() {
        return false;
    }
    for &j in &PUBLIC_VARS {
        if !public_inputs.iter().any(|&(i, _)| i == j) {
            return false;
        }
    }
    let pub_val = |j: usize| -> Fp {
        public_inputs.iter().find(|&&(i, _)| i == j).unwrap().1
    };

    let cols = [&qap.a_polys, &qap.b_polys, &qap.c_polys];
    let e_priv = [proof.ea, proof.eb, proof.ec];
    let e_priv_alpha = [proof.ea_alpha, proof.eb_alpha, proof.ec_alpha];
    let alpha_exp = setup.alpha.value() as u32;

    let mut e_full = [0u32; 3];
    for k in 0..3 {
        // The verifier adds the public contribution itself: E_full = E_priv * h^{P_pub(s)}.
        let mut pub_eval = Fp::ZERO;
        for &j in &PUBLIC_VARS {
            pub_eval = pub_eval + pub_val(j) * cols[k][j].eval(setup.s);
        }
        let full = e_priv[k] * hide(setup.h, pub_eval) % HIDE_MOD;
        let full_alpha =
            e_priv_alpha[k] * hide(setup.h, setup.alpha * pub_eval) % HIDE_MOD;
        // KEA check: the alpha-shift must be consistent.
        if full_alpha != hpow(full, alpha_exp) {
            return false;
        }
        e_full[k] = full;
    }

    // Divisibility check at s, recovering the hidden evaluations by
    // brute-force discrete log (possible only because the group is tiny).
    let (a, b, c, hh) = match (
        dlog(setup.h, e_full[0]),
        dlog(setup.h, e_full[1]),
        dlog(setup.h, e_full[2]),
        dlog(setup.h, proof.eh),
    ) {
        (Some(a), Some(b), Some(c), Some(hh)) => (a, b, c, hh),
        _ => return false,
    };
    let ts = qap.t.eval(setup.s);
    Fp::new(a) * Fp::new(b) - Fp::new(c) == Fp::new(hh) * ts
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_setup() -> (Setup, Qap) {
        let qap = crate::qap::build();
        let mut rng = XorShift64::new(0x1234_5678);
        let setup = setup(&mut rng, &qap);
        (setup, qap)
    }

    #[test]
    fn generator_has_order_257() {
        let h = subgroup_generator();
        assert_ne!(h, 1);
        assert_eq!(hpow(h, 257), 1);
        // order divides 257 (prime) and h != 1, so the order is exactly 257.
    }

    #[test]
    fn hiding_is_homomorphic() {
        let h = subgroup_generator();
        let (a, b) = (Fp::new(30), Fp::new(250));
        assert_eq!(hide(h, a) * hide(h, b) % HIDE_MOD, hide(h, a + b));
        assert_eq!(hpow(hide(h, a), 5), hide(h, a * Fp::new(5)));
    }

    #[test]
    fn dlog_roundtrip() {
        let h = subgroup_generator();
        for x in [0, 1, 2, 100, 256] {
            assert_eq!(dlog(h, hide(h, Fp::new(x))), Some(x));
        }
        // 0 is not in the subgroup generated by h (h^x is never 0 mod 1543).
        assert_eq!(dlog(h, 0), None);
    }

    #[test]
    fn blind_eval_matches_direct() {
        let (setup, _qap) = test_setup();
        // P(x) = 3 + 2x + x^2; blind eval must equal h^{P(s)}.
        let p = Poly::from_coeffs(vec![Fp::new(3), Fp::new(2), Fp::new(1)]);
        let blind = blind_eval(&setup.enc_powers, p.coeffs());
        assert_eq!(blind, hide(setup.h, p.eval(setup.s)));
    }

    #[test]
    fn end_to_end_honest_prover() {
        let (setup, qap) = test_setup();
        let w = r1cs::witness(Fp::new(3));
        let proof = prove(&setup, &qap, &w).expect("prove failed");
        let public = [(r1cs::VAR_ONE, Fp::ONE), (r1cs::VAR_OUT, Fp::new(35))];
        assert!(verify(&setup, &qap, &proof, &public));
    }

    #[test]
    fn wrong_public_input_rejected() {
        let (setup, qap) = test_setup();
        let w = r1cs::witness(Fp::new(3));
        let proof = prove(&setup, &qap, &w).expect("prove failed");
        // Claim the output is 36 instead of 35.
        let public = [(r1cs::VAR_ONE, Fp::ONE), (r1cs::VAR_OUT, Fp::new(36))];
        assert!(!verify(&setup, &qap, &proof, &public));
    }

    #[test]
    fn bad_witness_cannot_prove() {
        let (setup, qap) = test_setup();
        let mut w = r1cs::witness(Fp::new(3));
        w[r1cs::VAR_OUT] = Fp::new(36);
        assert!(prove(&setup, &qap, &w).is_err());
    }

    #[test]
    fn tampered_alpha_shift_rejected() {
        let (setup, qap) = test_setup();
        let w = r1cs::witness(Fp::new(3));
        let mut proof = prove(&setup, &qap, &w).expect("prove failed");
        proof.ea_alpha = (proof.ea_alpha + 1) % HIDE_MOD; // tamper
        let public = [(r1cs::VAR_ONE, Fp::ONE), (r1cs::VAR_OUT, Fp::new(35))];
        assert!(!verify(&setup, &qap, &proof, &public));
    }

    #[test]
    fn tampered_commitment_rejected() {
        let (setup, qap) = test_setup();
        let w = r1cs::witness(Fp::new(3));
        let mut proof = prove(&setup, &qap, &w).expect("prove failed");
        proof.eh = (proof.eh + 1) % HIDE_MOD; // tamper with H commitment
        let public = [(r1cs::VAR_ONE, Fp::ONE), (r1cs::VAR_OUT, Fp::new(35))];
        assert!(!verify(&setup, &qap, &proof, &public));
    }

    #[test]
    fn setup_avoids_roots_of_t() {
        // Run setup many times; t(s) must never vanish.
        let qap = crate::qap::build();
        let mut rng = XorShift64::new(999);
        for _ in 0..50 {
            let setup = setup(&mut rng, &qap);
            assert!(!qap.t.eval(setup.s).is_zero());
        }
    }
}
