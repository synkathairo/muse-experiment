# Golden test vectors

Cross-check between the Python training code and the Rust engine/inference.

For each index `i`:
- `golden_{i}.json` — description + source SGF + torch seed
- `golden_{i}_planes.npy` — (6,9,9) float32 encoder output. The Rust plane encoder must match this **bit-exactly**.
- `golden_{i}_policy.npy` — (82,) float32 policy logits from a fixed-seed GoNet (see json). Rust forward pass must match within fp tolerance (both sides: fp16 weights, f32 compute).
- `golden_{i}_value.npy` — scalar float32 value head output, same tolerance.

Regenerate with: `python -m gotrain.make_golden --out ../golden` from `weiqi/train/`.
