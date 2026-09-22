//! Prime field F_257 arithmetic.
//!
//! We work over the prime field F_p with p = 257 because:
//!   * 257 is prime, so every nonzero element has an inverse;
//!   * p - 1 = 256 = 2^8 is smooth (a power of two), so the multiplicative
//!     group F_p^* contains primitive n-th roots of unity for every n | 256,
//!     in particular the 4th roots of unity we need as the QAP evaluation
//!     domain for our 4-constraint R1CS;
//!   * it is small enough that every intermediate value is human-readable,
//!     which is the whole point of a toy.
//!
//! Values are stored reduced mod 257 in a u16; products of two reduced
//! values fit in u32, so no overflow is possible.

use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

/// The field modulus.
pub const MODULUS: u32 = 257;

/// An element of F_257, always stored reduced (0 <= value < 257).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Fp(u16);

impl Fp {
    pub const ZERO: Fp = Fp(0);
    pub const ONE: Fp = Fp(1);

    /// Reduce `v` mod 257.
    pub fn new(v: u32) -> Self {
        Fp((v % MODULUS) as u16)
    }

    pub fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// The reduced representative, as an integer in 0..257.
    pub fn value(self) -> u16 {
        self.0
    }

    /// Exponentiation by squaring.
    pub fn pow(self, mut e: u32) -> Self {
        let mut base = self;
        let mut acc = Fp::ONE;
        while e > 0 {
            if e & 1 == 1 {
                acc = acc * base;
            }
            base = base * base;
            e >>= 1;
        }
        acc
    }

    /// Multiplicative inverse via Fermat's little theorem: a^(p-2).
    /// Panics in debug builds on zero (which has no inverse).
    pub fn inv(self) -> Self {
        debug_assert!(!self.is_zero(), "inverse of zero");
        self.pow(MODULUS - 2)
    }
}

impl Add for Fp {
    type Output = Fp;
    fn add(self, rhs: Fp) -> Fp {
        Fp::new(self.0 as u32 + rhs.0 as u32)
    }
}

impl Sub for Fp {
    type Output = Fp;
    fn sub(self, rhs: Fp) -> Fp {
        // add MODULUS to avoid underflow
        Fp::new(self.0 as u32 + MODULUS - rhs.0 as u32)
    }
}

impl Mul for Fp {
    type Output = Fp;
    fn mul(self, rhs: Fp) -> Fp {
        Fp::new(self.0 as u32 * rhs.0 as u32)
    }
}

impl Neg for Fp {
    type Output = Fp;
    fn neg(self) -> Fp {
        Fp::new(MODULUS - self.0 as u32)
    }
}

impl fmt::Display for Fp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduction() {
        assert_eq!(Fp::new(257), Fp::ZERO);
        assert_eq!(Fp::new(300), Fp::new(43));
        assert_eq!(Fp::new(0), Fp::ZERO);
    }

    #[test]
    fn add_mul() {
        assert_eq!(Fp::new(200) + Fp::new(100), Fp::new(43)); // 300 mod 257
        assert_eq!(Fp::new(16) * Fp::new(16), Fp::new(256));
        assert_eq!(Fp::new(16) * Fp::new(17), Fp::new(15)); // 272 mod 257
    }

    #[test]
    fn sub_neg_wrap() {
        assert_eq!(Fp::ZERO - Fp::ONE, Fp::new(256));
        assert_eq!(-Fp::new(5), Fp::new(252));
        assert_eq!(Fp::new(3) - Fp::new(3), Fp::ZERO);
    }

    #[test]
    fn inverse_all_nonzero() {
        // Every nonzero element has an inverse: a * a^-1 == 1.
        for a in 1..MODULUS {
            let x = Fp::new(a);
            assert_eq!(x * x.inv(), Fp::ONE, "inverse failed for {a}");
        }
    }

    #[test]
    fn fermat_little() {
        // a^256 == 1 for all nonzero a.
        for a in [2, 3, 5, 100, 256] {
            assert_eq!(Fp::new(a).pow(256), Fp::ONE);
        }
        assert_eq!(Fp::ZERO.pow(5), Fp::ZERO);
        assert_eq!(Fp::new(7).pow(0), Fp::ONE);
    }

    #[test]
    fn fourth_roots_of_unity_exist() {
        // Because 4 | 256, F_257^* has primitive 4th roots of unity.
        let mut gen = Fp::new(2);
        while gen.pow(128) == Fp::ONE {
            gen = gen + Fp::ONE;
        }
        let w = gen.pow(64);
        assert_ne!(w, Fp::ONE);
        assert_ne!(w * w, Fp::ONE);
        assert_eq!(w * w * w * w, Fp::ONE);
    }
}
