//! zk-toy: a from-scratch educational zk-SNARK.
//!
//! Pipeline: prime field -> polynomials -> R1CS -> QAP -> homomorphic hiding
//! -> blind evaluation + KEA check -> prove / verify.
//!
//! Standard library only. See README.md for the math walkthrough.

pub mod field;
pub mod poly;
pub mod qap;
pub mod r1cs;
