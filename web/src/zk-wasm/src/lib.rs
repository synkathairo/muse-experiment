//! WASM bindings for the zk-toy SNARK demo.
//!
//! Stateless boundary: both `prove` and `verify` rebuild the trusted setup
//! deterministically from `seed` (XorShift64), so no setup secrets ever
//! cross into JS. The page drives the story:
//!   1. prove(secret_x, seed) -> JSON proof (or {"error": ...})
//!   2. verify(proof_json, public_out, seed) -> bool
//! A proof verifies only against the correct public output, and the
//! verifier never sees the secret.

use wasm_bindgen::prelude::*;
use zk_toy::{field::Fp, qap, r1cs, snark};

#[derive(serde::Deserialize)]
struct ProofJson {
    ea: u32,
    eb: u32,
    ec: u32,
    eh: u32,
    ea_alpha: u32,
    eb_alpha: u32,
    ec_alpha: u32,
}

fn build_setup(seed: u32) -> (snark::Setup, qap::Qap) {
    let qap = qap::build();
    let mut rng = snark::XorShift64::new(seed as u64);
    let setup = snark::setup(&mut rng, &qap);
    (setup, qap)
}

/// Generate a proof of knowledge of x with x^3 + x + 5 == out (mod 257).
/// Returns the proof as JSON, or {"error": "..."} if x is out of range.
#[wasm_bindgen]
pub fn prove(secret_x: u32, seed: u32) -> String {
    if secret_x > 256 {
        return r#"{"error":"secret x must be in 0..=256"}"#.to_string();
    }
    let (setup, qap) = build_setup(seed);
    let witness = r1cs::witness(Fp::new(secret_x));
    match snark::prove(&setup, &qap, &witness) {
        Ok(p) => serde_json::json!({
            "ea": p.ea, "eb": p.eb, "ec": p.ec, "eh": p.eh,
            "ea_alpha": p.ea_alpha, "eb_alpha": p.eb_alpha, "ec_alpha": p.ec_alpha,
        })
        .to_string(),
        Err(e) => serde_json::json!({ "error": e }).to_string(),
    }
}

/// Verify a proof (as produced by `prove`) against a claimed public output.
/// The verifier learns nothing about the secret beyond the statement's truth.
#[wasm_bindgen]
pub fn verify(proof_json: &str, public_out: u32, seed: u32) -> bool {
    let (setup, qap) = build_setup(seed);
    let pj: ProofJson = match serde_json::from_str(proof_json) {
        Ok(p) => p,
        Err(_) => return false,
    };
    let proof = snark::Proof {
        ea: pj.ea,
        eb: pj.eb,
        ec: pj.ec,
        eh: pj.eh,
        ea_alpha: pj.ea_alpha,
        eb_alpha: pj.eb_alpha,
        ec_alpha: pj.ec_alpha,
    };
    let public = [
        (r1cs::VAR_ONE, Fp::ONE),
        (r1cs::VAR_OUT, Fp::new(public_out % 257)),
    ];
    snark::verify(&setup, &qap, &proof, &public)
}

/// The public output for a secret x: x^3 + x + 5 mod 257.
/// (Trivial in JS too; exposed here so field semantics stay in one place.)
#[wasm_bindgen]
pub fn public_output(secret_x: u32) -> u32 {
    let x = Fp::new(secret_x % 257);
    (x * x * x + x + Fp::new(5)).value() as u32
}
