# weiqi — 9×9 Go demo project

Two tiny neural nets that play 9×9 Go in the browser, plus a "watch it learn"
time-machine over their training checkpoints. See `~/workspace/go-demo/PLAN.md`
for the full plan (currently being executed).

- `train/` — standalone Python training package (supervised cloning +
  hand-rolled self-play PPO). Knows nothing about the website.
- `theory.md` — the ML/game-theory framework behind both training legs,
  written for the exhibit page and for reproducers.
- `golden/` — cross-check vectors pinning Python/Rust agreement on the encoder
  planes (bit-exact) and the forward pass (fp tolerance).
- `engine/` — Rust rules + hand-rolled inference (separate workstream).

The only artifact that crosses from here to the website is the fp16 weight
file (`gotrain.export`), copied by `deploy_weights.sh` (added with the web
phase). Raw SGF datasets and optimizer checkpoints are never committed.
