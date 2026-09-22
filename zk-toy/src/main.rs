//! zk-toy demo binary.
//!
//! Modes (the default never touches stdin, so `cargo run` is CI-safe):
//!   (no args)   scripted demo: prove x = 3, verify, show rejections
//!   tour        guided walk of the pipeline with your own secret x
//!   attack      try to forge or tamper; see which check catches you
//!   -h, --help  show usage
//!
//! The prover only ever handles the witness and the CRS (the encrypted
//! powers); the secret evaluation point `s` and `alpha` never leave setup.
//! The `tour` mode lifts that curtain on purpose — it narrates from an
//! omniscient viewpoint so you can see every number.

use std::io::{self, Write};
use std::process::ExitCode;

use zk_toy::field::Fp;
use zk_toy::poly::Poly;
use zk_toy::qap::{self, Qap};
use zk_toy::r1cs;
use zk_toy::snark::{self, Proof, Setup};

const USAGE: &str = "\
usage: zk-toy [mode]

modes:
  (no args)   scripted demo: prove knowledge of x = 3, verify, show rejections
  tour        guided walk of the whole pipeline with a secret x you choose
  attack      play the adversary: forge and tamper, see which check catches you
  -h, --help  show this message

examples:
  cargo run --release            # scripted demo (default)
  cargo run --release -- tour    # interactive tour
  cargo run --release -- attack  # interactive attack menu";

fn print_help() {
    println!("{USAGE}");
}

fn print_help_stderr() {
    eprintln!("{USAGE}");
}

/// Read one line from stdin; returns None on EOF.
fn read_line() -> Option<String> {
    let mut buf = String::new();
    print!("> ");
    io::stdout().flush().ok()?;
    match io::stdin().read_line(&mut buf) {
        Ok(0) => None,
        Ok(_) => Some(buf.trim().to_string()),
        Err(_) => None,
    }
}

fn pause() {
    print!("  [press enter] ");
    io::stdout().flush().ok();
    let mut buf = String::new();
    let _ = io::stdin().read_line(&mut buf);
}

/// Pretty-print a polynomial, using negative reps where shorter:
/// x^4 - 1 instead of 256 + x^4.
fn fmt_poly(p: &Poly) -> String {
    let mut out = String::new();
    let mut first = true;
    for (i, &coef) in p.coeffs().iter().enumerate().rev() {
        let v = coef.value() as i32;
        if v == 0 {
            continue;
        }
        let neg = v > 128;
        let mag = if neg { 257 - v } else { v };
        let body = match i {
            0 => format!("{mag}"),
            1 if mag == 1 => "x".to_string(),
            1 => format!("{mag}*x"),
            _ if mag == 1 => format!("x^{i}"),
            _ => format!("{mag}*x^{i}"),
        };
        if first {
            if neg {
                out.push('-');
            }
            out.push_str(&body);
            first = false;
        } else {
            out.push_str(if neg { " - " } else { " + " });
            out.push_str(&body);
        }
    }
    if first {
        "0".to_string()
    } else {
        out
    }
}

fn prompt_secret_x() -> Option<Fp> {
    loop {
        println!("Pick your secret x (an integer 0..=256):");
        match read_line() {
            None => return None,
            Some(s) => match s.parse::<u32>() {
                Ok(n) if n <= 256 => return Some(Fp::new(n)),
                _ => println!("  '{s}' is not in range — try again."),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Mode 1: scripted demo (default). Unchanged behavior, no stdin.
// ---------------------------------------------------------------------------

fn demo() {
    // 1. The circuit: R1CS -> QAP (public, known to everyone).
    let qap = qap::build();
    println!("QAP built over F_257 with {} constraints.", r1cs::N_CONS);

    // 2. Trusted setup (verifier side). s and alpha are toxic waste.
    let mut rng = snark::XorShift64::new(0xC0FFEE);
    let setup = snark::setup(&mut rng, &qap);
    println!(
        "Setup done. CRS has {} encrypted powers.",
        setup.enc_powers.len()
    );

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

// ---------------------------------------------------------------------------
// Mode 2: guided tour. Omniscient narrator; every number on screen.
// ---------------------------------------------------------------------------

fn tour() {
    println!("=== zk-toy guided tour ===\n");
    println!("We will prove knowledge of an x with x^3 + x + 5 = out,");
    println!("walking every stage: witness -> R1CS -> QAP -> setup ->");
    println!("blind evaluation -> verification. Nothing is hidden from YOU;");
    println!("the prover in this story just plays their part.\n");

    let Some(x) = prompt_secret_x() else {
        println!("bye!");
        return;
    };
    println!("\nYour secret is x = {x}. The verifier must never learn it.\n");

    // --- witness ---
    println!("--- 1. witness ---");
    let w = r1cs::witness(x);
    let names = ["one", "x", "sym1", "y", "sym2", "out"];
    for (name, val) in names.iter().zip(w.iter()) {
        println!("  {name:>5} = {val}");
    }
    println!("  (sym1 = x*x, y = sym1*x, sym2 = y+x, out = sym2+5, all mod 257)");
    pause();

    // --- R1CS ---
    println!("\n--- 2. R1CS check, row by row ---");
    let (a_mat, b_mat, c_mat) = r1cs::matrices();
    let dot = |row: &[Fp; r1cs::N_VARS]| -> Fp {
        row.iter()
            .zip(w.iter())
            .map(|(m, v)| *m * *v)
            .fold(Fp::ZERO, |acc, t| acc + t)
    };
    for i in 0..r1cs::N_CONS {
        let (da, db, dc) = (dot(&a_mat[i]), dot(&b_mat[i]), dot(&c_mat[i]));
        let mark = if da * db == dc { "ok" } else { "FAIL" };
        println!("  row {}: ({da}) x ({db}) = {dc}   [{mark}]", i + 1);
    }
    pause();

    // --- QAP ---
    println!("\n--- 3. QAP ---");
    let qap = qap::build();
    println!("  target t(x) = {}", fmt_poly(&qap.t));
    let (a_poly, b_poly, c_poly) = qap.combine(&w);
    let h_poly = qap
        .quotient(&a_poly, &b_poly, &c_poly)
        .expect("division must be exact for an honest witness");
    println!("  A(x) = {}", fmt_poly(&a_poly));
    println!("  B(x) = {}", fmt_poly(&b_poly));
    println!("  C(x) = {}", fmt_poly(&c_poly));
    println!("  h(x) = {}", fmt_poly(&h_poly));
    println!("  and indeed A*B - C = h*t as polynomials.");
    pause();

    // --- setup (narrator view) ---
    println!("\n--- 4. trusted setup (narrator view!) ---");
    let mut rng = snark::XorShift64::new(0xC0FFEE);
    let setup = snark::setup(&mut rng, &qap);
    println!("  secret s     = {}   <-- toxic waste; the prover never sees this", setup.s);
    println!("  secret alpha = {}   <-- toxic waste; the prover never sees this", setup.alpha);
    println!(
        "  CRS: {} encrypted powers E(s^i), {} alpha-shifted, {} for H",
        setup.enc_powers.len(),
        setup.enc_alpha_powers.len(),
        setup.enc_h_powers.len()
    );
    println!("  (E(v) = h^v mod 1543; exponents live in F_257, matching the circuit field)");
    pause();

    // --- blind evaluation ---
    println!("\n--- 5. blind evaluation ---");
    let proof = snark::prove(&setup, &qap, &w).expect("proving failed");
    println!("  The prover, knowing only the witness and the CRS, computes:");
    println!("    E(A(s)) = {}      E(B(s)) = {}", proof.ea, proof.eb);
    println!("    E(C(s)) = {}      E(H(s)) = {}", proof.ec, proof.eh);
    println!("  plus the alpha-shifted E(a*A(s)), E(a*B(s)), E(a*C(s)).");
    println!("  These are just numbers mod 1543 — s itself never appears.");
    pause();

    // --- verification ---
    println!("\n--- 6. verification ---");
    let public = [(r1cs::VAR_ONE, Fp::ONE), (r1cs::VAR_OUT, w[r1cs::VAR_OUT])];
    println!("  public inputs claimed: one = 1, out = {}", w[r1cs::VAR_OUT]);
    let ok = snark::verify(&setup, &qap, &proof, &public);
    // Show the arithmetic the verifier performed. This mirrors
    // snark::verify: mix in the public part, then brute-force the
    // discrete logs (toy-only!) and check a*b - c == hh*t(s).
    let cols = [&qap.a_polys, &qap.b_polys, &qap.c_polys];
    let e_priv = [proof.ea, proof.eb, proof.ec];
    let pub_of = |j: usize| public.iter().find(|&&(i, _)| i == j).unwrap().1;
    let mut recovered = [0u32; 4];
    for k in 0..3 {
        let mut pub_eval = Fp::ZERO;
        for &j in &snark::PUBLIC_VARS {
            pub_eval = pub_eval + pub_of(j) * cols[k][j].eval(setup.s);
        }
        let full = e_priv[k] * snark::hide(setup.h, pub_eval) % snark::HIDE_MOD;
        recovered[k] = snark::dlog(setup.h, full).expect("dlog failed");
    }
    recovered[3] = snark::dlog(setup.h, proof.eh).expect("dlog failed");
    let (a, b, c, hh) = (recovered[0], recovered[1], recovered[2], recovered[3]);
    let ts = qap.t.eval(setup.s);
    println!("  verifier recovers (brute-force dlog, toy-only!):");
    println!("    A(s) = {a}, B(s) = {b}, C(s) = {c}, H(s) = {hh}, t(s) = {ts}");
    println!("  checks {a}*{b} - {c} == {hh}*{ts}  (mod 257)");
    let lhs = Fp::new(a) * Fp::new(b) - Fp::new(c);
    let rhs = Fp::new(hh) * ts;
    println!("  i.e. {lhs} == {rhs}  ->  verifies: {ok}");
    assert!(ok);

    println!("\nDone. The verifier is convinced an x exists with x^3 + x + 5 = {},", w[r1cs::VAR_OUT]);
    println!("and it never saw x = {x}. Try `attack` mode to break this.");
}

// ---------------------------------------------------------------------------
// Mode 3: attack. You are the adversary; the verifier plays fair.
// ---------------------------------------------------------------------------

fn attack_menu() -> Option<u32> {
    println!("\nChoose your attack (or 'q' to quit):");
    println!("  1. bad witness       lie in the witness, try to prove");
    println!("  2. wrong statement   honest proof, verify against out = 36... er, out+1");
    println!("  3. tamper alpha      modify the alpha-shifted commitment");
    println!("  4. tamper H          modify the quotient commitment");
    println!("  5. tamper proof      modify a blind evaluation");
    match read_line()?.as_str() {
        "1" => Some(1),
        "2" => Some(2),
        "3" => Some(3),
        "4" => Some(4),
        "5" => Some(5),
        _ => None,
    }
}

fn attack() {
    println!("=== zk-toy attack mode ===\n");
    println!("You are the adversary. The verifier plays fair.");
    println!("Each attack shows you exactly which check catches it, and why.\n");

    let qap: Qap = qap::build();
    let mut rng = snark::XorShift64::new(0xC0FFEE);
    let setup: Setup = snark::setup(&mut rng, &qap);
    let honest_proof = snark::prove(&setup, &qap, &r1cs::witness(Fp::new(3)))
        .expect("honest prove failed");
    let x = Fp::new(3);
    let out = r1cs::witness(x)[r1cs::VAR_OUT];
    let public = [(r1cs::VAR_ONE, Fp::ONE), (r1cs::VAR_OUT, out)];

    loop {
        let Some(choice) = attack_menu() else {
            println!("bye! the proof system remains unbroken.");
            return;
        };
        match choice {
            1 => {
                println!("\n-- attack 1: bad witness --");
                println!("  You claim x = 4 but fake out = {out} in the witness.");
                let mut w = r1cs::witness(Fp::new(4));
                w[r1cs::VAR_OUT] = out;
                match snark::prove(&setup, &qap, &w) {
                    Ok(_) => println!("  UNBROKEN SYSTEM?! a proof was produced (this should never happen)"),
                    Err(e) => {
                        println!("  caught at PROVE time: {e}");
                        println!("  why: the R1CS is unsatisfied, so the QAP division");
                        println!("       A*B - C = h*t has a remainder — no quotient exists.");
                    }
                }
            }
            2 => {
                println!("\n-- attack 2: wrong statement --");
                let fake_out = out + Fp::ONE;
                println!("  Honest proof, but you verify it against out = {fake_out}.");
                let fake_public = [(r1cs::VAR_ONE, Fp::ONE), (r1cs::VAR_OUT, fake_out)];
                let ok = snark::verify(&setup, &qap, &honest_proof, &fake_public);
                println!("  verifies: {ok}");
                println!("  why: the verifier mixes in its OWN public inputs homomorphically,");
                println!("       so the proof only verifies against the claimed statement.");
            }
            3 => {
                println!("\n-- attack 3: tamper with the alpha-shift --");
                let mut p: Proof = honest_proof.clone();
                p.ea_alpha = (p.ea_alpha + 1) % snark::HIDE_MOD;
                println!("  You nudge E(alpha*A(s)) by 1 and hope nobody notices.");
                let ok = snark::verify(&setup, &qap, &p, &public);
                println!("  verifies: {ok}");
                println!("  why: the KEA check E(a*A(s)) == E(A(s))^alpha fails.");
                println!("       Without knowing alpha you cannot forge a consistent pair —");
                println!("       this binds you to actually knowing the polynomial.");
            }
            4 => {
                println!("\n-- attack 4: tamper with the quotient commitment --");
                let mut p: Proof = honest_proof.clone();
                p.eh = (p.eh + 1) % snark::HIDE_MOD;
                println!("  You nudge E(H(s)) by 1.");
                let ok = snark::verify(&setup, &qap, &p, &public);
                println!("  verifies: {ok}");
                println!("  why: the divisibility check a*b - c == hh*t(s) fails,");
                println!("       because hh no longer equals the true H(s).");
            }
            5 => {
                println!("\n-- attack 5: tamper with a blind evaluation --");
                let mut p: Proof = honest_proof.clone();
                p.ea = (p.ea + 1) % snark::HIDE_MOD;
                println!("  You nudge E(A(s)) by 1, leaving its alpha-shift untouched.");
                let ok = snark::verify(&setup, &qap, &p, &public);
                println!("  verifies: {ok}");
                println!("  why: the KEA check fails — E(a*A(s)) != E(A(s))^alpha anymore.");
                println!("       Every committed evaluation is chained to its shift.");
            }
            _ => unreachable!(),
        }
    }
}

// ---------------------------------------------------------------------------

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        demo();
        return ExitCode::SUCCESS;
    }
    match args[0].as_str() {
        "-h" | "--help" => {
            print_help();
            ExitCode::SUCCESS
        }
        "tour" => {
            tour();
            ExitCode::SUCCESS
        }
        "attack" => {
            attack();
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("error: unknown mode '{other}'\n");
            print_help_stderr();
            ExitCode::from(2)
        }
    }
}
