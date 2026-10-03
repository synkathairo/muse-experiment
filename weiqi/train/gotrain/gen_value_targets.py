"""Generate value-probe targets: self-play games with terminal outcomes.

For every position (before each move): the 6-plane encoding, the color to move,
and the eventual game outcome from the player-to-move's perspective
(1 = win, 0 = loss, 0.5 = draw), from Tromp-Taylor scoring with komi.

Usage:
    PYTHONPATH=. python -m gotrain.gen_value_targets \
        --checkpoint ~/workspace/runs/ppo_bc/latest.pt \
        --games 1600 --jobs 8 --temperature 0.2 --seed 0 \
        --out ~/workspace/runs/value_probe/targets.npz

Split by game downstream (no position-level shuffling across the split).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys

import numpy as np
import torch

from gotrain import features, rules, selfplay
from gotrain.net import GoNet
from gotrain.net_aux import GoNetAux, TRUNK_PREFIX


def load_model(path: str) -> torch.nn.Module:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    net = GoNetAux() if any(k.startswith(TRUNK_PREFIX) for k in sd) else GoNet()
    net.load_state_dict(sd)
    net.eval()
    return net


def genmove(model: torch.nn.Module, board: rules.Board, color: int,
            temperature: float, rng: np.random.Generator) -> tuple[int, int] | None:
    mask = selfplay.bot_mask(board, color)
    planes = features.encode(
        board.stones(color),
        board.stones(rules.opponent(color)),
        last_move=board.last_move,
        black_to_move=(color == rules.BLACK),
        ko_point=board.ko,
    )
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(planes).unsqueeze(0))
    logits = logits.squeeze(0).numpy().astype(np.float64)
    logits[~mask] = -np.inf
    if temperature <= 0:
        idx = int(np.argmax(logits))
    else:
        z = logits / temperature
        finite = np.isfinite(z)
        z = z - z[finite].max()
        probs = np.zeros_like(z)
        probs[finite] = np.exp(z[finite])
        s = probs.sum()
        idx = int(rng.choice(probs.size, p=probs / s)) if s > 0 else int(np.argmax(logits))
    move = None if idx == 81 else divmod(idx, 9)
    assert board.play(move, color), f"engine emitted illegal move {move}"
    return move


def play_game(model: torch.nn.Module, temperature: float,
              seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (planes[N,6,9,9], color_to_move[N], outcome[N])."""
    rng = np.random.default_rng(seed)
    board = rules.Board(9)
    planes_list: list[np.ndarray] = []
    colors: list[int] = []
    passes = 0
    for _ in range(400):
        for color in (rules.BLACK, rules.WHITE):
            planes_list.append(features.encode(
                board.stones(color),
                board.stones(rules.opponent(color)),
                last_move=board.last_move,
                black_to_move=(color == rules.BLACK),
                ko_point=board.ko,
            ))
            colors.append(color)
            move = genmove(model, board, color, temperature, rng)
            passes = passes + 1 if move is None else 0
            if passes >= 2:
                break
        if passes >= 2:
            break
    b, w = selfplay.score(board)
    colors_np = np.array(colors, dtype=np.int64)
    outcome = np.where(
        colors_np == rules.BLACK,
        np.sign(b - w),   # +1 black wins, -1 white wins, 0 draw
        np.sign(w - b),
    ).astype(np.float32)
    y = (outcome + 1.0) / 2.0  # -> {0, 0.5, 1} from player-to-move's view
    return (np.stack(planes_list).astype(np.float32), colors_np, y)


_state: dict = {}


def _init(checkpoint: str, temperature: float) -> None:
    _state["model"] = load_model(checkpoint)
    _state["temperature"] = temperature


def _play(task: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    game_idx, seed = task
    planes, colors, y = play_game(_state["model"], _state["temperature"], seed)
    return planes, colors, y, game_idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--games", type=int, default=1600)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tasks = [(i, args.seed * 100003 + i) for i in range(args.games)]
    all_planes, all_colors, all_y, game_ids = [], [], [], []
    with mp.Pool(args.jobs, initializer=_init,
                 initargs=(args.checkpoint, args.temperature)) as pool:
        for planes, colors, y, gi in pool.imap_unordered(_play, tasks):
            all_planes.append(planes)
            all_colors.append(colors)
            all_y.append(y)
            game_ids.append(np.full(len(y), gi, dtype=np.int32))
            if len(game_ids) % 200 == 0:
                print(f"  {len(game_ids)}/{args.games} games", flush=True)

    np.savez_compressed(
        args.out,
        planes=np.concatenate(all_planes),
        color=np.concatenate(all_colors),
        y=np.concatenate(all_y),
        game=np.concatenate(game_ids),
    )
    n = sum(len(p) for p in all_planes)
    print(f"saved {args.out}: {n} positions from {args.games} games", flush=True)


if __name__ == "__main__":
    sys.exit(main())
