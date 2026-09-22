//! QAP transformation: R1CS -> polynomials.
//!
//! For each of the N_CONS constraints we pick an evaluation point r_i (the
//! 4th roots of unity in F_257). Each *column* j of a constraint matrix is
//! interpolated into a polynomial (A_j, B_j, C_j) with A_j(r_i) = A[i][j].
//!
//! Then, with the witness s as coefficients:
//!   A(x) = sum_j s_j * A_j(x),   B(x) = sum_j s_j * B_j(x),
//!   C(x) = sum_j s_j * C_j(x)
//! and the R1CS is satisfied iff A(r_i)*B(r_i) - C(r_i) = 0 for all i, i.e.
//! iff the *target polynomial* t(x) = prod_i (x - r_i) divides
//! A(x)*B(x) - C(x). The quotient is h(x):
//!   A(x) * B(x) - C(x) = h(x) * t(x).

use crate::field::Fp;
use crate::poly::Poly;
use crate::r1cs::{self, N_CONS, N_VARS};

/// The QAP: per-variable polynomials plus the target polynomial.
pub struct Qap {
    /// Evaluation points: the 4th roots of unity in F_257.
    pub roots: Vec<Fp>,
    pub a_polys: Vec<Poly>,
    pub b_polys: Vec<Poly>,
    pub c_polys: Vec<Poly>,
    /// t(x) = prod_i (x - r_i). Here this equals x^4 - 1.
    pub t: Poly,
}

/// Smallest primitive root of F_257^* (found by search; 3 works).
fn primitive_root() -> Fp {
    let mut g = Fp::new(2);
    // 257 - 1 = 256 = 2^8, so g is primitive iff g^128 != 1.
    while g.pow(128) == Fp::ONE {
        g = g + Fp::ONE;
    }
    g
}

/// Build the QAP from the R1CS matrices.
pub fn build() -> Qap {
    let g = primitive_root();
    let w = g.pow(64); // primitive 4th root of unity: w^4 = 1, w^2 != 1
    let w2 = w * w;
    let w3 = w2 * w;
    let roots = vec![Fp::ONE, w, w2, w3];
    debug_assert!(roots.iter().enumerate().all(|(i, r)| roots[i + 1..].iter().all(|s| r != s)));

    let (a_mat, b_mat, c_mat) = r1cs::matrices();
    let mut a_polys = Vec::with_capacity(N_VARS);
    let mut b_polys = Vec::with_capacity(N_VARS);
    let mut c_polys = Vec::with_capacity(N_VARS);
    for j in 0..N_VARS {
        let col = |m: &[[Fp; N_VARS]; N_CONS]| -> Vec<Fp> { (0..N_CONS).map(|i| m[i][j]).collect() };
        a_polys.push(Poly::lagrange(&roots, &col(&a_mat)));
        b_polys.push(Poly::lagrange(&roots, &col(&b_mat)));
        c_polys.push(Poly::lagrange(&roots, &col(&c_mat)));
    }

    let t = Poly::vanishing(&roots);
    Qap { roots, a_polys, b_polys, c_polys, t }
}

impl Qap {
    /// A(x) = sum_j coeffs[j] * A_j(x) (and likewise B, C).
    pub fn combine(&self, coeffs: &[Fp; N_VARS]) -> (Poly, Poly, Poly) {
        let lin = |polys: &[Poly]| -> Poly {
            let mut acc = Poly::zero();
            for (p, &c) in polys.iter().zip(coeffs.iter()) {
                if !c.is_zero() {
                    acc = &acc + &(p * &Poly::constant(c));
                }
            }
            acc
        };
        (lin(&self.a_polys), lin(&self.b_polys), lin(&self.c_polys))
    }

    /// Compute h(x) = (A*B - C) / t, or None if the division is not exact
    /// (which happens exactly when the witness does NOT satisfy the R1CS).
    pub fn quotient(&self, a: &Poly, b: &Poly, c: &Poly) -> Option<Poly> {
        let num = &(a * b) - c;
        let (q, r) = num.divmod(&self.t);
        if r.is_zero() { Some(q) } else { None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roots_are_fourth_roots_of_unity() {
        let qap = build();
        assert_eq!(qap.roots.len(), 4);
        for r in &qap.roots {
            assert_eq!(r.pow(4), Fp::ONE);
        }
        // distinct
        for i in 0..4 {
            for j in (i + 1)..4 {
                assert_ne!(qap.roots[i], qap.roots[j]);
            }
        }
    }

    #[test]
    fn target_is_x4_minus_1() {
        let qap = build();
        // t(x) = prod (x - r_i) over all 4th roots of unity = x^4 - 1.
        assert_eq!(qap.t, Poly::from_coeffs(vec![-Fp::ONE, Fp::ZERO, Fp::ZERO, Fp::ZERO, Fp::ONE]));
    }

    #[test]
    fn column_polys_match_matrix() {
        // A_j(r_i) must equal the matrix entry A[i][j].
        let qap = build();
        let (a_mat, b_mat, c_mat) = r1cs::matrices();
        for (i, r) in qap.roots.iter().enumerate() {
            for j in 0..N_VARS {
                assert_eq!(qap.a_polys[j].eval(*r), a_mat[i][j], "A[{i}][{j}]");
                assert_eq!(qap.b_polys[j].eval(*r), b_mat[i][j], "B[{i}][{j}]");
                assert_eq!(qap.c_polys[j].eval(*r), c_mat[i][j], "C[{i}][{j}]");
            }
        }
    }

    #[test]
    fn divisibility_holds_for_honest_witness() {
        let qap = build();
        let w = r1cs::witness(Fp::new(3));
        let (a, b, c) = qap.combine(&w);
        let h = qap.quotient(&a, &b, &c).expect("division must be exact");
        assert!(h.degree() <= 2, "deg h should be <= 2, got {}", h.degree());
        // A*B - C == h*t as polynomials.
        let lhs = &(&a * &b) - &c;
        let rhs = &h * &qap.t;
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn divisibility_fails_for_bad_witness() {
        let qap = build();
        let mut w = r1cs::witness(Fp::new(3));
        w[r1cs::VAR_OUT] = Fp::new(36);
        let (a, b, c) = qap.combine(&w);
        assert!(qap.quotient(&a, &b, &c).is_none());
    }
}
