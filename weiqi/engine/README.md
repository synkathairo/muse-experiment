# weiqi-engine

Pure-Rust core for the 9×9 weiqi/go demo: rules, feature encoder, and
hand-rolled neural-net inference. No dependencies, no platform-specific code —
this crate also compiles to `wasm32-unknown-unknown` (the thin wasm-bindgen
shim in `web/src/go-wasm/` path-depends on it).

## Modules

- **`rules`** — 9×9 board, Tromp–Taylor-style legality, simple ko, Chinese area
  scoring, 7.5 komi (provisional; see `KOMI`). `Game::play` returns the capture
  count or an `IllegalMove` reason; the game is unchanged on illegal moves.
- **`features`** — the 6 LOCKED input planes from PLAN.md §3, row-major,
  row 0 = top of board. Must stay bit-exact vs `train/gotrain/features.py`
  (values are exactly 0.0/1.0; golden-position cross-check pending).
- **`infer`** — f32 forward pass of the LOCKED arch (4×conv64 trunk, policy +
  value heads). Loads the flat fp16 little-endian weight blob from
  `train/gotrain/export.py`; tensor order is documented in `infer.rs` and
  asserted by test (`tensor_order_sums_to_n_params`).

## Test

```sh
cargo test
```

Covers: captures (incl. a genuine snapback, which is *not* ko), simple ko
(recapture illegal, legal again after a ko threat, ko flips sides), suicide,
occupied/pass/game-over handling, scoring of known positions, two-pass game
end, the encoder on hand-computed positions, fp16 conversion spot checks, and
the forward pass (shapes, uniform-policy/zero-value on zero weights,
determinism, hand-computed conv).
