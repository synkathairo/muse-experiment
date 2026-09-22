//! Univariate polynomials over F_257.
//!
//! Coefficients are stored lowest-degree first and always trimmed, so the
//! zero polynomial is the empty coefficient vector.

use crate::field::Fp;
use std::ops::{Add, Mul, Sub};

/// A univariate polynomial over F_257.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Poly {
    coeffs: Vec<Fp>, // coeffs[i] = coefficient of x^i, no trailing zeros
}

impl Poly {
    /// Build from coefficients (lowest-degree first); trims trailing zeros.
    pub fn from_coeffs(mut coeffs: Vec<Fp>) -> Self {
        while coeffs.last() == Some(&Fp::ZERO) {
            coeffs.pop();
        }
        Poly { coeffs }
    }

    pub fn zero() -> Self {
        Poly { coeffs: Vec::new() }
    }

    pub fn one() -> Self {
        Poly::from_coeffs(vec![Fp::ONE])
    }

    pub fn constant(c: Fp) -> Self {
        Poly::from_coeffs(vec![c])
    }

    /// The polynomial x.
    pub fn var() -> Self {
        Poly::from_coeffs(vec![Fp::ZERO, Fp::ONE])
    }

    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty()
    }

    /// Degree; defined as 0 for the zero polynomial (guard with is_zero).
    pub fn degree(&self) -> usize {
        if self.coeffs.is_empty() {
            0
        } else {
            self.coeffs.len() - 1
        }
    }

    pub fn leading(&self) -> Fp {
        *self.coeffs.last().unwrap_or(&Fp::ZERO)
    }

    pub fn coeffs(&self) -> &[Fp] {
        &self.coeffs
    }

    /// Evaluate at `x` using Horner's method.
    pub fn eval(&self, x: Fp) -> Fp {
        let mut acc = Fp::ZERO;
        for &c in self.coeffs.iter().rev() {
            acc = acc * x + c;
        }
        acc
    }

    /// Polynomial long division. Returns (quotient, remainder) with
    /// self = quotient * divisor + remainder, deg(remainder) < deg(divisor).
    pub fn divmod(&self, divisor: &Poly) -> (Poly, Poly) {
        assert!(!divisor.is_zero(), "division by zero polynomial");
        let mut rem = self.clone();
        let mut quot = vec![Fp::ZERO; 0];
        let dn = divisor.degree();
        let lc_inv = divisor.leading().inv();
        while !rem.is_zero() && rem.degree() >= dn {
            let shift = rem.degree() - dn;
            let coeff = rem.leading() * lc_inv;
            if quot.len() <= shift {
                quot.resize(shift + 1, Fp::ZERO);
            }
            quot[shift] = quot[shift] + coeff;
            // rem -= coeff * x^shift * divisor
            for (i, &d) in divisor.coeffs.iter().enumerate() {
                let idx = i + shift;
                rem.coeffs[idx] = rem.coeffs[idx] - coeff * d;
            }
            // re-trim
            while rem.coeffs.last() == Some(&Fp::ZERO) {
                rem.coeffs.pop();
            }
        }
        (Poly::from_coeffs(quot), rem)
    }

    /// Lagrange interpolation: the unique degree < n polynomial passing
    /// through (xs[i], ys[i]).
    pub fn lagrange(xs: &[Fp], ys: &[Fp]) -> Poly {
        assert_eq!(xs.len(), ys.len(), "xs and ys length mismatch");
        assert!(!xs.is_empty(), "need at least one point");
        let n = xs.len();
        let mut acc = Poly::zero();
        for i in 0..n {
            // basis_i(x) = prod_{j != i} (x - x_j) / (x_i - x_j)
            let mut num = Poly::one();
            let mut den = Fp::ONE;
            for j in 0..n {
                if i == j {
                    continue;
                }
                num = num * &Poly::from_coeffs(vec![-xs[j], Fp::ONE]);
                den = den * (xs[i] - xs[j]);
            }
            let term = num * &Poly::constant(ys[i] * den.inv());
            acc = acc + &term;
        }
        acc
    }

    /// The monic vanishing polynomial prod_i (x - roots[i]).
    pub fn vanishing(roots: &[Fp]) -> Poly {
        let mut acc = Poly::one();
        for &r in roots {
            acc = acc * &Poly::from_coeffs(vec![-r, Fp::ONE]);
        }
        acc
    }
}

impl Add<&Poly> for &Poly {
    type Output = Poly;
    fn add(self, rhs: &Poly) -> Poly {
        let n = self.coeffs.len().max(rhs.coeffs.len());
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let a = self.coeffs.get(i).copied().unwrap_or(Fp::ZERO);
            let b = rhs.coeffs.get(i).copied().unwrap_or(Fp::ZERO);
            out.push(a + b);
        }
        Poly::from_coeffs(out)
    }
}

impl Sub<&Poly> for &Poly {
    type Output = Poly;
    fn sub(self, rhs: &Poly) -> Poly {
        let n = self.coeffs.len().max(rhs.coeffs.len());
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let a = self.coeffs.get(i).copied().unwrap_or(Fp::ZERO);
            let b = rhs.coeffs.get(i).copied().unwrap_or(Fp::ZERO);
            out.push(a - b);
        }
        Poly::from_coeffs(out)
    }
}

impl Mul<&Poly> for &Poly {
    type Output = Poly;
    fn mul(self, rhs: &Poly) -> Poly {
        if self.is_zero() || rhs.is_zero() {
            return Poly::zero();
        }
        let mut out = vec![Fp::ZERO; self.coeffs.len() + rhs.coeffs.len() - 1];
        for (i, &a) in self.coeffs.iter().enumerate() {
            for (j, &b) in rhs.coeffs.iter().enumerate() {
                out[i + j] = out[i + j] + a * b;
            }
        }
        Poly::from_coeffs(out)
    }
}

// Forwarding impls so owned values compose ergonomically with references.
impl Mul<&Poly> for Poly {
    type Output = Poly;
    fn mul(self, rhs: &Poly) -> Poly {
        &self * rhs
    }
}

impl Add<&Poly> for Poly {
    type Output = Poly;
    fn add(self, rhs: &Poly) -> Poly {
        &self + rhs
    }
}

impl Sub<&Poly> for Poly {
    type Output = Poly;
    fn sub(self, rhs: &Poly) -> Poly {
        &self - rhs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fp(v: u32) -> Fp {
        Fp::new(v)
    }

    #[test]
    fn eval_horner() {
        // 1 + 2x + 3x^2 at x = 2 -> 1 + 4 + 12 = 17
        let p = Poly::from_coeffs(vec![fp(1), fp(2), fp(3)]);
        assert_eq!(p.eval(fp(2)), fp(17));
        assert_eq!(p.eval(fp(0)), fp(1));
    }

    #[test]
    fn mul_simple() {
        // (x + 1)(x - 1) = x^2 - 1
        let a = Poly::from_coeffs(vec![fp(1), fp(1)]);
        let b = Poly::from_coeffs(vec![fp(256), fp(1)]); // x - 1
        let c = &a * &b;
        assert_eq!(c, Poly::from_coeffs(vec![fp(256), fp(0), fp(1)]));
    }

    #[test]
    fn divmod_exact() {
        // (x^2 + 1)(x + 2) = x^3 + 2x^2 + x + 2; divide by (x + 2).
        let num = Poly::from_coeffs(vec![fp(2), fp(1), fp(2), fp(1)]);
        let den = Poly::from_coeffs(vec![fp(2), fp(1)]);
        let (q, r) = num.divmod(&den);
        assert_eq!(q, Poly::from_coeffs(vec![fp(1), fp(0), fp(1)]));
        assert!(r.is_zero());
    }

    #[test]
    fn divmod_with_remainder() {
        // (x^2 + 1) / (x + 1): quotient x - 1, remainder 2.
        let num = Poly::from_coeffs(vec![fp(1), fp(0), fp(1)]);
        let den = Poly::from_coeffs(vec![fp(1), fp(1)]);
        let (q, r) = num.divmod(&den);
        assert_eq!(q, Poly::from_coeffs(vec![fp(256), fp(1)])); // x - 1
        assert_eq!(r, Poly::from_coeffs(vec![fp(2)]));
        // check: q*den + r == num
        let back = &(&q * &den) + &r;
        assert_eq!(back, num);
    }

    #[test]
    fn lagrange_roundtrip() {
        let xs = vec![fp(1), fp(2), fp(3), fp(4)];
        let ys = vec![fp(10), fp(20), fp(30), fp(40)];
        let p = Poly::lagrange(&xs, &ys);
        for (x, y) in xs.iter().zip(ys.iter()) {
            assert_eq!(p.eval(*x), *y);
        }
        assert!(p.degree() <= 3);
    }

    #[test]
    fn vanishing_poly() {
        let roots = vec![fp(1), fp(2), fp(3)];
        let t = Poly::vanishing(&roots);
        for r in &roots {
            assert_eq!(t.eval(*r), Fp::ZERO);
        }
        assert_eq!(t.eval(fp(4)), fp(6)); // (4-1)(4-2)(4-3) = 6
        assert_eq!(t.leading(), Fp::ONE); // monic
    }
}
