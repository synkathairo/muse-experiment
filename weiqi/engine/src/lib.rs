//! `weiqi-engine`: pure-Rust core for the 9x9 weiqi/go demo.
//!
//! - [`rules`]: board, Tromp–Taylor-style legality, simple ko, Chinese area scoring.
//! - [`features`]: the 6 LOCKED input planes from PLAN.md §3 (bit-exact vs Python).
//! - [`infer`]: hand-rolled f32 forward pass of the LOCKED net arch; loads the flat
//!   fp16 little-endian weight blobs produced by `train/gotrain/export.py`.
//!
//! No dependencies, no platform-specific code: this crate must also compile to
//! `wasm32-unknown-unknown` for the browser demo (the thin wasm-bindgen shim lives
//! in `web/src/go-wasm/` and path-depends on this crate).

pub mod features;
pub mod infer;
pub mod rules;

pub use rules::{Color, Game, IllegalMove, Move, Score, KOMI, N_MOVES, N_POINTS};
