//! Demo: prove knowledge of x = 3 with x^3 + x + 5 = 35, without revealing x.
//!
//! The prover only ever handles the witness and the CRS (the encrypted
//! powers); the secret evaluation point `s` and `alpha` never leave setup.

use zk_toy::field::Fp;
use zk_toy::qap;
use zk_toy::r1cs;
use zk_toy::snark;

fn main() {
    // 1. The circuit: R1CS -> QAP (public, known to everyone).
    let qap = qap::build();
    println!("QAP built over F_257 with {} constraints.", r1cs::N_CONS);

    // 2. Trusted setup (verifier side). s and alpha are toxic waste.
    let mut rng = snark::XorShift64::new(0xC0FFEE);
    let setup = snark::setup(&mut rng, &qap);
    println!("Setup done. CRS has {} encrypted powers.", setup.enc_powers.len());

    // 3. The prover knows x = 3, builds the witness, and proves.
    let secret_x = Fp::new(3);
    let witness = r1cs::witness(secret_x);
    let proof = snark::prove(&setup, &qap, &witness).expect("proving failed");
    println!("Proof generated: {proof:?}");

    // 4. The verifier checks the proof against the public inputs
    //    (one = 1, out = 35). It never sees x.
    let public = [(r1cs::VAR_ONE, Fp::ONE), (r1cs::VAR_OUT, Fp::new(35))];
    let ok = snark::verify(&setup, &qap, &proof, &public);
    println!("Honest proof verifies: {ok}");
    assert!(ok);

    // 5. A proof does not verify against the wrong public output...
    let wrong_public = [(r1cs::VAR_ONE, Fp::ONE), (r1cs::VAR_OUT, Fp::new(36))];
    let ok_wrong = snark::verify(&setup, &qap, &proof, &wrong_public);
    println!("Proof with wrong public output (36) verifies: {ok_wrong}");
    assert!(!ok_wrong);

    // 6. ...and you cannot even generate a proof from a bad witness.
    let mut bad_witness = r1cs::witness(Fp::new(4)); // x = 4 -> out = 73, not 35
    bad_witness[r1cs::VAR_OUT] = Fp::new(35); // try to fake the output
    match snark::prove(&setup, &qap, &bad_witness) {
        Ok(_) => println!("ERROR: bad witness produced a proof!"),
        Err(e) => println!("Bad witness correctly rejected at prove time: {e}"),
    }

    println!("\nDone. The verifier learned nothing about x beyond the fact");
    println!("that the prover knows an x with x^3 + x + 5 = 35.");
}
