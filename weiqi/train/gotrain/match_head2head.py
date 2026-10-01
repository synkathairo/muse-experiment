"""Head-to-head match between two checkpoints (mirrors gotrain/gtp.py move logic).

Usage:
    PYTHONPATH=. python -m gotrain.match_head2head \
        --black-ckpt A.pt --white-ckpt B.pt --games 256 --jobs 8 \
        --temperature 0.2 --out /tmp/match.json

Colors alternate each game; game 0 has --black-ckpt as Black.
"""
from __future__ import annotations

import argparse
import json
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


def play_game(black_model: torch.nn.Module, white_model: torch.nn.Module,
              temperature: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    board = rules.Board(9)
    passes = 0
    moves = 0
    for _ in range(400):
        for color, model in ((rules.BLACK, black_model), (rules.WHITE, white_model)):
            move = genmove(model, board, color, temperature, rng)
            moves += 1
            passes = passes + 1 if move is None else 0
            if passes >= 2:
                b, w = selfplay.score(board)
                return {"winner": "B" if b > w else "W", "moves": moves,
                        "score_b": b, "score_w": w}
    b, w = selfplay.score(board)
    return {"winner": "B" if b > w else "W", "moves": moves,
            "score_b": b, "score_w": w, "capped": True}


_state: dict = {}


def _init(black_ckpt: str, white_ckpt: str, temperature: float) -> None:
    _state["black"] = load_model(black_ckpt)
    _state["white"] = load_model(white_ckpt)
    _state["temperature"] = temperature


def _play(task: tuple[int, int]) -> dict:
    game_idx, seed = task
    # alternate colors: even games keep the ckpt assignment, odd games swap
    if game_idx % 2 == 0:
        r = play_game(_state["black"], _state["white"], _state["temperature"], seed)
        r["black_ckpt"] = "A"
    else:
        r = play_game(_state["white"], _state["black"], _state["temperature"], seed)
        r["black_ckpt"] = "B"
    r["game"] = game_idx
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--black-ckpt", required=True)
    ap.add_argument("--white-ckpt", required=True)
    ap.add_argument("--games", type=int, default=256)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp/match.json")
    args = ap.parse_args()

    tasks = [(i, args.seed * 100000 + i) for i in range(args.games)]
    with mp.Pool(args.jobs, initializer=_init,
                 initargs=(args.black_ckpt, args.white_ckpt, args.temperature)) as pool:
        results = pool.map(_play, tasks)

    results.sort(key=lambda r: r["game"])
    with open(args.out, "w") as f:
        json.dump(results, f)
    a_wins = sum(1 for r in results
                 if (r["winner"] == "B") == (r["black_ckpt"] == "A"))
    print(f"A (--black-ckpt) {a_wins}-{args.games - a_wins} "
          f"({100.0 * a_wins / args.games:.1f}%)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
