# weiqi/train — supervised training package (Imitator)

Standalone Python package for training the 9×9 Go nets used by the demo.
Knows nothing about the website; its only outward artifact is the fp16
weight file produced by `gotrain.export`.

## Layout

- `gotrain/net.py` — the shared `GoNet` architecture (**locked spec**, PLAN.md §3)
- `gotrain/features.py` — the 6 input planes (locked spec)
- `gotrain/rules.py` — minimal 9×9 rules for SGF replay (captures, simple ko)
- `gotrain/sgf.py` — SGF parsing, 9×9 filter
- `gotrain/dataset.py` — (planes → move) pair building, train/val split
- `gotrain/train_cloning.py` — behavioral cloning; resumable, log-spaced snapshots
- `gotrain/export.py` — checkpoint → fp16 little-endian weight file (locked format)
- `gotrain/make_golden.py` — golden vectors for the Rust side (`weiqi/golden/`)
- `tests/test_smoke.py` — SGF parse, net shapes, param count, export round-trip

## Data sources

- **OGS** (corpus): polite rate-limited pull of 9×9 ranked human games via the
  public API (`gotrain.ogs_pull`). ToS reviewed 2026-09-24: no scraping ban
  found; the "no computer help" clause covers live games only. A courtesy note
  to the OGS devs before any bulk pull is still recommended (PLAN.md §9).
- **KGS archives** (NOT used): investigated 2026-09-24 — the u-go.net monthly
  archives (and the 4d+ collection) are **19×19 only** (zero 9×9 games found),
  and gokgs.com offers per-user archives, not a bulk dump. No bulk 9×9 KGS
  source exists, so KGS was dropped in favor of OGS.

Raw SGF data lives in `data/` (gitignored, never committed).

## Reproduce

```bash
# venv with CPU torch, e.g.:
uv venv .venv && uv pip install torch numpy --index-url https://download.pytorch.org/whl/cpu

# 1. fetch KGS zips into data/kgs/
# 2. build dataset (game-level train/val split, chunked to bound RAM)
python -m gotrain.dataset --sgf data/kgs --out data/ds_full

# 3. train (resumable; snapshots at 1k/3k/10k/30k/... gradient steps)
python -m gotrain.train_cloning --data data/ds_full --out runs/imit_v1 \
    --max-steps 60000 --val-every 1000

# resume after interruption:
python -m gotrain.train_cloning --data data/ds_full --out runs/imit_v1 \
    --resume runs/imit_v1/latest.pt

# 4. export a checkpoint to the locked fp16 format
python -m gotrain.export --checkpoint runs/imit_v1/snap_0003000.pt \
    --out-dir ./exports/ --name imit-003k.bin

# 5. smoke tests
python -m unittest tests.test_smoke -v
```

## Rules note

KGS games were played under **Japanese rules**; the demo itself assumes
**Chinese area scoring with 7.5 komi** (provisional — open decision, PLAN.md §9).
For behavioral cloning this mismatch is harmless: the ruleset only gates which
( position → move ) pairs we keep during replay, and `gotrain.rules` filters
leniently (truncates a game at the first move illegal under simple
Tromp-Taylor-style rules rather than arguing about it). The net never sees a
ruleset label.
