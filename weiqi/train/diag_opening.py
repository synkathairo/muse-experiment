"""Diagnose the net's Black/opening weakness.

1. Empty-board policy top moves per checkpoint: did the opening get worse,
   or never good?
2. Opening diversity: distinct first moves over 40 temp-0.5 samples.
3. Value trace of greedy self-play for 12 plies, from Black's perspective:
   where does the value collapse?

Modes (run from weiqi/train):
  python diag_opening.py                              # legacy: 100k/300k/1M/3M demo blobs
  python diag_opening.py --checkpoint runs/smoke/latest.pt   # any local .pt
  python diag_opening.py --blob-tag 6M                # single demo blob
"""
import argparse
import numpy as np
import torch

from gotrain.net import GoNet
from gotrain.export import read_export
from gotrain import features as F

WEIGHTS = "../../web/site/demos/go/weights/selfplay-clean-%s.bin"


def load_blob(tag):
    sd = read_export(WEIGHTS % tag)
    m = GoNet()
    m.load_state_dict({k: torch.from_numpy(v) for k, v in sd.items()})
    m.eval()
    return m


def load_ckpt(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    m = GoNet()
    m.load_state_dict(sd)
    m.eval()
    step = ck.get("step") if isinstance(ck, dict) else None
    return m, step


def infer(m, planes):
    with torch.no_grad():
        logits, value = m(torch.from_numpy(planes[None]).float())
    p = torch.softmax(logits[0], -1).numpy()
    return p, float(value[0])


def show_board_top(p, k=10):
    idx = np.argsort(-p[:81])[:k]
    out = []
    for i in idx:
        r, c = i // 9, i % 9
        out.append(f"({r},{c}) {p[i]:.3f}")
    return out


def empty_planes(black_to_move=True):
    return F.encode(np.zeros((9, 9), bool), np.zeros((9, 9), bool),
                    black_to_move=black_to_move)


ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", help=".pt file, e.g. runs/smoke/latest.pt")
ap.add_argument("--blob-tag", help="single demo blob tag, e.g. 6M")
args = ap.parse_args()

models = {}
if args.checkpoint:
    m, step = load_ckpt(args.checkpoint)
    models["ckpt"] = m
    target = "ckpt"
    print(f"loaded {args.checkpoint} (step={step})")
elif args.blob_tag:
    models[args.blob_tag] = load_blob(args.blob_tag)
    target = args.blob_tag
else:
    for tag in ["100k", "300k", "1M", "3M"]:
        models[tag] = load_blob(tag)
    target = "3M"

print("=== 1. Empty-board policy top-10 as Black, per checkpoint ===")
for tag, m in models.items():
    p, v = infer(m, empty_planes(True))
    print(f"{tag}: value={v:+.3f} pass_p={p[81]:.3f}")
    print("   ", show_board_top(p))

print(f"\n=== 2. {target} opening diversity: Black's first move over 40 temp-0.5 samples ===")
m = models[target]
rng = np.random.default_rng(0)
firsts = []
for _ in range(40):
    p, _ = infer(m, empty_planes(True))
    legal = np.arange(81)
    w = np.maximum(p[legal], 1e-9) ** (1 / 0.5)
    w /= w.sum()
    firsts.append(int(rng.choice(legal, p=w)))
from collections import Counter
c = Counter((i // 9, i % 9) for i in firsts)
print("distinct first moves:", len(c), "->", c.most_common(8))

print(f"\n=== 3. Value trace: {target} greedy self-play, 12 plies (Black's perspective) ===")
from gotrain.rules import Board, BLACK, WHITE
b = Board()
moves = []
vals = []
for ply in range(12):
    color = BLACK if ply % 2 == 0 else WHITE
    own = np.array([[b.grid[r][c] == color for c in range(9)] for r in range(9)])
    opp = np.array([[b.grid[r][c] == (3 - color) for c in range(9)] for r in range(9)])
    lm = b.last_move
    planes = F.encode(own, opp, last_move=lm, black_to_move=(color == BLACK),
                      ko_point=b.ko)
    p, v = infer(m, planes)
    vals.append(v if color == BLACK else -v)  # from Black's perspective
    # greedy legal move
    legal = [(r, c) for r in range(9) for c in range(9) if b.is_legal(r, c, color)]
    if not legal:
        b.play(None, color); moves.append("pass"); continue
    scored = sorted(legal, key=lambda rc: -p[rc[0] * 9 + rc[1]])
    mv = scored[0]
    b.play(mv, color)
    moves.append(f"{'B' if color == BLACK else 'W'}{mv}")
print("moves:", moves)
print("black-perspective value by ply:", [f"{v:+.2f}" for v in vals])
