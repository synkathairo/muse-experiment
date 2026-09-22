//! R1CS for the classic example: "I know x such that x^3 + x + 5 = 35".
//!
//! Flattening into multiplication gates (each gate becomes one constraint
//! of the form (A . s) * (B . s) = (C . s)):
//!
//!   sym1 = x * x          (1)  x       * x       = sym1
//!   y    = sym1 * x       (2)  sym1    * x       = y
//!   sym2 = y + x          (3)  (y + x) * 1       = sym2
//!   out  = sym2 + 5       (4)  (sym2 + 5) * 1    = out
//!
//! Solution vector s = [one, x, sym1, y, sym2, out].
//! For the honest witness x = 3: s = [1, 3, 9, 27, 30, 35].

use crate::field::Fp;

pub const N_VARS: usize = 6;
pub const N_CONS: usize = 4;

/// Variable indices into the solution vector.
pub const VAR_ONE: usize = 0;
pub const VAR_X: usize = 1;
pub const VAR_SYM1: usize = 2;
pub const VAR_Y: usize = 3;
pub const VAR_SYM2: usize = 4;
pub const VAR_OUT: usize = 5;

/// The R1CS constraint matrices (A, B, C), each N_CONS x N_VARS.
pub fn matrices() -> (
    [[Fp; N_VARS]; N_CONS],
    [[Fp; N_VARS]; N_CONS],
    [[Fp; N_VARS]; N_CONS],
) {
    let f = Fp::new;
    // A rows: left factors of each constraint.
    let a = [
        [f(0), f(1), f(0), f(0), f(0), f(0)], // x
        [f(0), f(0), f(1), f(0), f(0), f(0)], // sym1
        [f(0), f(1), f(0), f(1), f(0), f(0)], // y + x
        [f(5), f(0), f(0), f(0), f(1), f(0)], // sym2 + 5
    ];
    // B rows: right factors.
    let b = [
        [f(0), f(1), f(0), f(0), f(0), f(0)], // x
        [f(0), f(1), f(0), f(0), f(0), f(0)], // x
        [f(1), f(0), f(0), f(0), f(0), f(0)], // 1
        [f(1), f(0), f(0), f(0), f(0), f(0)], // 1
    ];
    // C rows: outputs.
    let c = [
        [f(0), f(0), f(1), f(0), f(0), f(0)], // sym1
        [f(0), f(0), f(0), f(1), f(0), f(0)], // y
        [f(0), f(0), f(0), f(0), f(1), f(0)], // sym2
        [f(0), f(0), f(0), f(0), f(0), f(1)], // out
    ];
    (a, b, c)
}

/// Build the full witness (solution vector) from a candidate x.
pub fn witness(x: Fp) -> [Fp; N_VARS] {
    let sym1 = x * x;
    let y = sym1 * x;
    let sym2 = y + x;
    let out = sym2 + Fp::new(5);
    [Fp::ONE, x, sym1, y, sym2, out]
}

/// Check that (A . s) * (B . s) = (C . s) for every constraint.
pub fn satisfied(
    a: &[[Fp; N_VARS]; N_CONS],
    b: &[[Fp; N_VARS]; N_CONS],
    c: &[[Fp; N_VARS]; N_CONS],
    s: &[Fp; N_VARS],
) -> bool {
    for i in 0..N_CONS {
        let dot = |row: &[Fp; N_VARS]| -> Fp {
            row.iter()
                .zip(s.iter())
                .map(|(m, v)| *m * *v)
                .fold(Fp::ZERO, |x, y| x + y)
        };
        if dot(&a[i]) * dot(&b[i]) != dot(&c[i]) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn honest_witness_satisfies() {
        let (a, b, c) = matrices();
        let w = witness(Fp::new(3));
        assert_eq!(w, [Fp::ONE, Fp::new(3), Fp::new(9), Fp::new(27), Fp::new(30), Fp::new(35)]);
        assert!(satisfied(&a, &b, &c, &w));
    }

    #[test]
    fn other_x_gives_other_out_but_still_satisfies() {
        // The R1CS only checks internal consistency; x = 4 yields out = 73.
        let (a, b, c) = matrices();
        let w = witness(Fp::new(4));
        assert_eq!(w[VAR_OUT], Fp::new(73)); // 64 + 4 + 5
        assert!(satisfied(&a, &b, &c, &w));
    }

    #[test]
    fn tampered_witness_fails() {
        let (a, b, c) = matrices();
        let mut w = witness(Fp::new(3));
        w[VAR_OUT] = Fp::new(36); // lie about the output
        assert!(!satisfied(&a, &b, &c, &w));
        let mut w2 = witness(Fp::new(3));
        w2[VAR_X] = Fp::new(4); // lie about x while keeping the rest
        assert!(!satisfied(&a, &b, &c, &w2));
    }
}
