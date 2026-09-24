# muse-experiment

Playground repo experiment for testing [Muse agent bot](https://muse.ai/) (`wally-musebot-synkathairo[bot]`). The agent accesses the repo via a (repository-)scoped Github App token.

## Projects

- `zk-toy/` — a toy SNARK playground (finite fields, polynomials, R1CS, QAP, prover/verifier demo). [Live demo](https://synkathairo.github.io/muse-experiment/demos/zk/)
- `lwe-toy/` — toy Regev LWE cryptosystem + working BKW attack demo (Python, stdlib only).
- `game-toy/` — Axelrod-style iterated prisoner's dilemma tournament (Python, stdlib only).
- `fluid-sim/` — 2D stable-fluids (Stam 1999) simulation in Rust, WASM demo. [Live demo](https://synkathairo.github.io/muse-experiment/demos/fluid/)
- `audio-dsp/` — portable Rust DSP crate (hand-rolled FFT, biquads) with a web demo. [Live demo](https://synkathairo.github.io/muse-experiment/demos/audio/)
- `weiqi/` — 9×9 Go with tiny neural-net opponents: supervised cloning from human games + from-scratch self-play PPO, with a "watch it learn" checkpoint time-machine (in progress).
- `research-skills/` — five AI-assistant research workflow skills (paper search/Q&A, literature reviews, paper-vs-code audits, topic watches, gated replications), adapted from design patterns in [Feynman](https://github.com/companion-inc/feynman) (MIT).
- `web/` — the GitHub Pages site shell; demos ship under `web/site/demos/`.
