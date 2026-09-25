# weiqi/train — training package

Standalone Python package for training the 9×9 Go nets used by the demo.
Knows nothing about the website; its outward artifacts are the fp16 weight
files (via `gotrain.export`) committed under `web/site/demos/go/weights/`.

Two training legs, one architecture (`GoNet`, locked spec — PLAN.md §3):

- **Autodidact** (self-play PPO vs periodically refreshed frozen snapshots)
  — the active leg. This is what runs on the laptop.
- **Imitator** (supervised behavioral cloning from OGS 9×9 games) — parked;
  the corpus pull failed and is not being retried (see § Supervised, below).

Theory for both legs lives in `weiqi/theory.md`.

## Layout

- `gotrain/net.py` — the shared `GoNet` architecture (**locked spec**, PLAN.md §3)
- `gotrain/features.py` — the 6 input planes (locked spec)
- `gotrain/rules.py` — minimal 9×9 rules (captures, simple ko, no suicide)
- `gotrain/selfplay.py` — self-play env: N boards stepped sequentially in one
  process, learner vs frozen-snapshot opponent, Tromp-Taylor scoring (7.5 komi),
  two-pass termination, `legal_mask()` / `score()` helpers
- `gotrain/ppo.py` — PPO with GAE; max-ply truncations are episodic terminals
  for GAE (bootstrap 0), not value-bootstrapped
- `gotrain/train_selfplay.py` — the trainer (entry point for the Autodidact)
- `gotrain/gtp.py` — GTP v2 server wrapping a checkpoint (greedy policy), so
  the net can play matches against GNU Go
- `eval_vs_gnugo.py` — match runner: spawns `gnugo --mode gtp` and the net,
  relays moves, arbitrates with `gotrain.rules` + Tromp-Taylor 7.5 komi
- `gotrain/sgf.py`, `gotrain/dataset.py`, `gotrain/train_cloning.py`,
  `gotrain/ogs_pull.py` — supervised leg (parked)
- `gotrain/export.py` — checkpoint → fp16 little-endian weight file
  (locked format, 261,044 bytes)
- `gotrain/make_golden.py` — golden vectors for the Rust side (`weiqi/golden/`)
- `tests/` — smoke tests + the three-way rules differential fuzz
  (`test_rules_fuzz.py`: Python rules vs independent positional-ko reference
  vs the Rust engine; runs `cargo` when available, else skips)

## Self-play quickstart (the laptop run)

```bash
cd weiqi/train
uv sync   # creates .venv and installs the locked deps (pyproject.toml + uv.lock)
uv run python -c "import torch; print(torch.backends.mps.is_available())"  # expect True on Apple Silicon
```

`uv sync` is the whole setup — no manual `uv venv`, no `pip install`, no
activation. `uv run` executes in the project environment (re-syncing if the
lockfile changed). Dependencies are declared in `pyproject.toml` (`torch>=2.14,<2.15`, `numpy==2.5.2`,
Python ≥3.12) and the exact per-platform artifacts are frozen in `uv.lock`
(Linux resolves torch `2.14.0+cpu` from PyTorch's CPU index — no 2.5GB CUDA
bundle on CPU-only boxes; macOS gets the MPS wheel from PyPI), so every
machine installs the identical environment.

`--device` defaults to `auto` (cuda > mps > cpu); the resolved device is
logged at startup. Pass `--device cpu` explicitly only to debug a backend.

Smoke test first (~15 min, confirms the backend works):

```bash
uv run python -m gotrain.train_selfplay --out runs/smoke --total-steps 200000
```

The real run — start **clean** (no `--resume` from the pilot; it trained
under the old GAE/max-ply semantics, fixed since — see § Pilot caveats):

```bash
caffeinate -i uv run python -m gotrain.train_selfplay \
    --out runs/laptop_clean --total-steps 100000000
```

Run it under `tmux` (or `nohup`) so closing the terminal doesn't kill it;
keep the machine plugged in and set it to never sleep. Speed is dominated
by the single-process Python game simulation, not the net. Measured
2026-09-24 on an M1 Mac (MPS): ~450–480 steps/sec, so 100M steps ≈ 2.5 days
and 200M ≈ 5 days. (The old CPU pilot managed ~200 steps/sec: 100M ≈ 6 days.)

Resume after any interruption (optimizer, step, and RNG state all restore):

```bash
uv run python -m gotrain.train_selfplay --out runs/laptop_clean \
    --resume runs/laptop_clean/latest.pt
```

Key flags: `--num-envs` (default 32), `--rollout-steps` (128),
`--total-steps`, `--lr` (2.5e-4), `--opp-refresh-every` (10 iters),
`--eval-every` (20 iters, `--eval-games` 20 vs random / greedy / snapshot),
`--ckpt-every` (10 iters), `--seed`.

## Checkpoints and logs

- `runs/<name>/train.log` — one line per PPO iteration:
  `iter N: step X sps=...` plus `ep_rew`, `ep_len`, PPO diagnostics
  (`pg`, `v`, `ent`, `kl`, `clipfrac`, `ev`) and periodic
  `eval_vs_random=… eval_vs_greedy=… eval_vs_snapshot=…` lines.
- `latest.pt` — rolling checkpoint (model + optimizer + snapshot + step +
  RNG); written every `--ckpt-every` iterations.
- `snap_{step}.pt` — frozen opponent snapshots, log-spaced in env steps
  (1k, 3k, 10k, …, 3M, 10M, …); these are what the demo's time machine
  can load.
- At the end of a run the final model is fp16-exported next to the
  checkpoints (cf. `runs/auto_pilot/autodidact-final.bin`).

## Exporting weights for the demo

```bash
uv run python -m gotrain.export --checkpoint runs/laptop_clean/snap_050000000.pt \
    --out-dir ./exports/ --name selfplay-50m.bin
```

Copy the `.bin` to `web/site/demos/go/weights/`, add one entry to
`web/site/demos/go/manifest.json` (see PLAN.md §4.6 for the time-machine
contract), and push — no code changes needed.

## Benchmarking vs GNU Go

```bash
brew install gnugo   # macOS; on Debian: apt-get install gnugo (/usr/games/gnugo)
python eval_vs_gnugo.py --checkpoint runs/laptop_clean/snap_050000000.pt \
    --levels 1 3 5 10 --games 8 --out eval/gnugo_bench.json
```

Pilot baseline (2026-09-24, `snap_003000000`, greedy, 32 games, gnugo 3.8
`--chinese-rules`, 7.5 komi): **net 5–27** (1–7 / 1–7 / 1–7 / 2–6 across
levels 1/3/5/10), avg margin ~18 pts — and **0–16 as Black, 5–16 as White**.
The net never won as Black: opening/first-move play is the hole; komi bails
it out as White. Check color-conditioned win rates in future training.
Full record: `eval/gnugo_bench_2026-09-24.json`.

### Benchmarking search strength (the demo's MCTS toggle)

Two GTP engines wrap the same PUCT search as the demo toggle (Rust
`engine/src/mcts.rs`), so the benchmark measures the toggle itself:

- **Rust** (`weiqi/engine/src/bin/mcts_gtp.rs`): native binary, single-threaded
  like the demo. Build once: `cargo build --release --bin mcts_gtp`.
- **Python** (`python -m gotrain.mcts_gtp`): faithful port of the Rust search
  (3/3 test positions agree move-for-move), with `--batch N` evaluating N
  leaves per forward pass on GPU (default 16; `--batch 1` is exactly the
  sequential algorithm). Device auto-selects cuda > mps > cpu.

Plug either into the harness with `--engine-cmd` (shlex-split; `--checkpoint`
is then not needed):

```bash
python eval_vs_gnugo.py \
  --engine-cmd "python -m gotrain.mcts_gtp --checkpoint runs/smoke/latest.pt --sims 100 --batch 16" \
  --levels 1 3 5 --games 4 --out eval/gnugo_mcts100.json
# or: --engine-cmd "../engine/target/release/mcts_gtp --blob <weights.bin> --sims 100"
```

`--sims` mirrors the demo toggle (50/100/200); `--dirichlet-eps` defaults to
0.15 like the toggle (the Rust binary also accepts `--dirichlet-eps`).
Run the sim settings in parallel terminals — games are independent. Compare
each against the greedy baseline above: the question is whether search buys
Elo vs GNU Go, and whether more sims buys more.

`--temperature` (Python greedy engine only) samples `softmax(logits/T)` over
legal moves instead of argmax; it does not apply to the search engines, which
take `--temperature` over root visit counts instead (0 = argmax, default).

## Supervised leg (Imitator) — parked

`train_cloning.py` trains on (planes → human move) pairs from OGS 9×9 ranked
games. The overnight corpus pull (2026-09-24) was throttled to 2 of 60,000
games and died; per standing constraint it is not being retried and OGS is
not being contacted without explicit authorization. A 150-game /
6,342-position validation-only run completed (2,000 steps, val acc 0.2313),
proving resume + fp16 export work. The supervised timeline in the demo is
intentionally empty until an authorized corpus source exists. Raw SGF data
lives in `data/` (gitignored, never committed).

Notes from the corpus investigation (2026-09-24): the KGS bulk archives
(u-go.net monthlies, 4d+ collection) are **19×19 only** — zero 9×9 games —
so KGS was dropped in favor of OGS. OGS games are played under Japanese
rules while the demo scores Chinese/area; for cloning this is harmless
(`gotrain.rules` just truncates a game at the first move illegal under our
ruleset, and the net never sees a ruleset label).

Reproduce the pipeline (needs a corpus in `data/` first):

```bash
uv run python -m gotrain.dataset --sgf data/ogs --out data/ds_full
uv run python -m gotrain.train_cloning --data data/ds_full --out runs/imit_v1 \
    --max-steps 60000 --val-every 1000
uv run python -m gotrain.export --checkpoint runs/imit_v1/snap_0003000.pt \
    --out-dir ./exports/ --name imit-003k.bin
```

## Pilot caveats

`runs/auto_pilot/` (3M steps, 2026-09-24) is a **pipeline-validation** run,
not a strength result: it trained under incorrect GAE episode boundaries
and old max-ply bootstrapping semantics, both fixed since (`ppo.py`,
`selfplay.py`). Do not present its weights as a strong or shippable model;
the demo page carries the same caveat.
