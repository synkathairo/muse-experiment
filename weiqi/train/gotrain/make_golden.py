"""Generate golden test vectors for the Rust side (weiqi/engine).

For a few small handcrafted SGF positions:
  - replay with gotrain.rules, encode with gotrain.features -> planes .npy
  - forward through a FIXED-SEED GoNet -> fp32 policy logits + value .npy

The Rust encoder must match planes bit-exactly; the Rust forward pass must
match logits/value within fp tolerance (fp16 weights -> f32 compute on both
sides, so small epsilon expected).

Usage:
    python -m gotrain.make_golden --out ../golden
Writes golden_{i}_{planes,policy,value}.npy + golden_{i}.json (sgf + description).
"""

import argparse
import json
import os

import numpy as np
import torch

from . import features, rules, sgf
from .net import GoNet

GOLDEN_SGFS = [
    ("empty board, black to play",
     "(;GM[1]FF[4]SZ[9])"),
    ("opening: two stones each, white to play, last move marked",
     "(;GM[1]FF[4]SZ[9];B[cc];W[gg];B[gc];W[cg])"),
    ("capture: black to play captures the white stone at fe",
     "(;GM[1]FF[4]SZ[9];B[de];W[ee];B[ed];W[aa];B[ef];W[ii])"),
    ("pass as last move, black to play",
     "(;GM[1]FF[4]SZ[9];B[cc];W[];B[gc])"),
]


def build_position(sgf_text):
    game = sgf.parse_sgf(sgf_text)
    assert game is not None and game.size == 9
    board = rules.Board(9)
    for color_s, move in game.moves:
        color = rules.BLACK if color_s == "B" else rules.WHITE
        assert board.play(move, color), f"illegal move in golden SGF: {move}"
    to_move = board.to_play
    planes = features.encode(
        board.stones(to_move),
        board.stones(rules.opponent(to_move)),
        last_move=board.last_move,
        black_to_move=(to_move == rules.BLACK),
        ko_point=board.ko,
    )
    return planes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    torch.manual_seed(args.seed)
    model = GoNet()
    model.eval()

    for i, (desc, s) in enumerate(GOLDEN_SGFS):
        planes = build_position(s)
        with torch.no_grad():
            logits, value = model(torch.from_numpy(planes).unsqueeze(0))
        logits = logits.squeeze(0).numpy().astype(np.float32)
        value = float(value.squeeze(0).numpy())
        np.save(os.path.join(args.out, f"golden_{i}_planes.npy"), planes)
        np.save(os.path.join(args.out, f"golden_{i}_policy.npy"), logits)
        np.save(os.path.join(args.out, f"golden_{i}_value.npy"),
                np.array(value, dtype=np.float32))
        with open(os.path.join(args.out, f"golden_{i}.json"), "w") as f:
            json.dump({"index": i, "description": desc, "sgf": s,
                       "torch_seed": args.seed}, f, indent=2)
        print(f"golden_{i}: {desc}")

    with open(os.path.join(args.out, "README.md"), "w") as f:
        f.write(
            "# Golden test vectors\n\n"
            "Cross-check between the Python training code and the Rust engine/inference.\n\n"
            "For each index `i`:\n"
            "- `golden_{i}.json` — description + source SGF + torch seed\n"
            "- `golden_{i}_planes.npy` — (6,9,9) float32 encoder output. "
            "The Rust plane encoder must match this **bit-exactly**.\n"
            "- `golden_{i}_policy.npy` — (82,) float32 policy logits from a fixed-seed "
            "GoNet (see json). Rust forward pass must match within fp tolerance "
            "(both sides: fp16 weights, f32 compute).\n"
            "- `golden_{i}_value.npy` — scalar float32 value head output, same tolerance.\n\n"
            "Regenerate with: `python -m gotrain.make_golden --out ../golden` "
            "from `weiqi/train/`.\n"
        )


if __name__ == "__main__":
    main()
