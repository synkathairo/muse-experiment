"""Phase 1 of the search-distillation probe: generate MCTS targets.

Each worker plays temp-0.2 self-play games with the checkpoint, samples
positions, and runs MCTS (search_with_visits) on each. Output NPZ:
  obs    (N,6,9,9) float32 - input planes
  policy (N,82)   float32 - normalized root visit distribution
  value  (N,)     float32 - mean backed-up root value (side-to-move)
  mask   (N,82)   bool    - legal move mask

Usage:
  python -m gotrain.gen_search_targets --checkpoint .../latest.pt \\
      --games 400 --sims 100 --jobs 8 --out /tmp/search_targets.npz
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

JOB_CHECKPOINT = ""
JOB_SIMS = 100
JOB_TEMP = 0.2
JOB_PLIES = (20, 150)


def _init_worker(checkpoint: str, sims: int, temp: float,
                 plies: tuple[int, int]) -> None:
    # Re-executed in each spawn worker: globals do not inherit from the parent.
    global JOB_CHECKPOINT, JOB_SIMS, JOB_TEMP, JOB_PLIES
    JOB_CHECKPOINT = checkpoint
    JOB_SIMS = sims
    JOB_TEMP = temp
    JOB_PLIES = plies
    import torch
    torch.set_num_threads(1)  # one thread per worker; scale via --jobs


def _worker(game_idx: int) -> dict[str, np.ndarray]:
    # Local imports: worker processes start clean.
    from . import rules, selfplay
    from .features import index_to_move
    from .mcts import PASS, SearchConfig, Searcher, _clone
    from .net_aux import GoNetAux

    from .diagnose_search import load_net, sample_move

    device = torch.device("cpu")
    net = load_net(JOB_CHECKPOINT, device)
    rng = np.random.default_rng(1000 + game_idx)
    searcher = Searcher(net, device, SearchConfig(simulations=JOB_SIMS,
                                                 batch=16, seed=2000 + game_idx))

    b = rules.Board()
    hist: list[rules.Board] = []
    passes = 0
    while passes < 2 and len(hist) < 320:
        obs = selfplay.observe(b, b.to_play)
        mask = selfplay.bot_mask(b, b.to_play)
        with torch.no_grad():
            logits, _ = net(torch.from_numpy(obs).unsqueeze(0).to(device))
        m = sample_move(logits[0].float().cpu().numpy(), mask, JOB_TEMP, rng)
        hist.append(_clone(b))
        if m == PASS:
            passes += 1
            b.play(None, b.to_play)
        else:
            passes = 0
            b.play(index_to_move(m), b.to_play)

    lo, hi = JOB_PLIES
    cand = [pos for pos in hist[lo:hi]
            if int(selfplay.bot_mask(pos, pos.to_play).sum()) > 1]
    obs_l, vis_l, val_l, mask_l = [], [], [], []
    for pos in cand:
        mask = selfplay.bot_mask(pos, pos.to_play)
        mv, visits, root_value = searcher.search_with_visits(pos, 0)
        obs_l.append(selfplay.observe(pos, pos.to_play))
        vis_l.append(visits.astype(np.float32))
        val_l.append(root_value)
        mask_l.append(mask)
    return {
        "obs": np.stack(obs_l).astype(np.float32) if obs_l
        else np.zeros((0, 6, 9, 9), np.float32),
        "policy": np.stack(vis_l) if vis_l else np.zeros((0, 82), np.float32),
        "value": np.array(val_l, np.float32),
        "mask": np.stack(mask_l) if mask_l else np.zeros((0, 82), bool),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--plies", default="20,150")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lo, hi = (int(x) for x in args.plies.split(","))
    # (Worker config travels via the pool initializer; module globals here
    # are parent-side only under the spawn start method.)

    import multiprocessing as mp
    import time
    ctx = mp.get_context("spawn")
    t0 = time.time()
    parts: list[dict[str, np.ndarray]] = []
    with ctx.Pool(args.jobs, initializer=_init_worker,
                  initargs=(args.checkpoint, args.sims, args.temp,
                            (lo, hi))) as pool:
        for i, part in enumerate(pool.imap_unordered(
                _worker, range(args.seed, args.seed + args.games))):
            parts.append(part)
            if (i + 1) % 20 == 0 or (i + 1) == args.games:
                dt = time.time() - t0
                npos = sum(len(p["obs"]) for p in parts)
                print(f"  {i + 1}/{args.games} games, {npos} positions "
                      f"({dt:.0f}s)", flush=True)

    obs = np.concatenate([p["obs"] for p in parts])
    policy = np.concatenate([p["policy"] for p in parts])
    value = np.concatenate([p["value"] for p in parts])
    mask = np.concatenate([p["mask"] for p in parts])
    print(f"generated {len(obs)} positions from {args.games} games")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, obs=obs, policy=policy, value=value, mask=mask)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
