"""Generate KataGo distillation dataset for the 2x2 architecture diagnostic.

Pipeline:
  1. KataGo self-play games -> SGF files
  2. Replay each position through our Board -> 6-plane obs (matches training)
  3. KataGo analysis (subprocess, JSON protocol) -> policy + value targets
  4. Save compressed npz: obs, policy (82), value, mask (82)

CLOSED (2026-09-26): the 130K student can't absorb the 10.5M-transformer
teacher — probe top-1 29.1% vs >40% gate. Kept for the record.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from typing import NoReturn

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gotrain.selfplay import observe, BLACK, WHITE
from gotrain.rules import Board

PASS: int = 81


def parse_sgf_moves(sgf_text: str) -> list[tuple[int, tuple[int, int] | None]]:
    """Extract move list as (color, (r, c) or None for pass)."""
    import re
    moves: list[tuple[int, tuple[int, int] | None]] = []
    # Match ;B[aa] or ;W[aa] or ;B[] (pass)
    for m in re.finditer(r';([BW])\[([a-s]{0,2})\]', sgf_text):
        color = BLACK if m.group(1) == 'B' else WHITE
        sq = m.group(2)
        if not sq:
            moves.append((color, None))
        else:
            c = ord(sq[0]) - ord('a')
            r = ord(sq[1]) - ord('a')
            moves.append((color, (r, c)))
    return moves


def board_to_katago_query(board: Board, color: int, qid: str,
                          komi: float = 7.5) -> NoReturn:
    """Build a katago analysis query for the current position."""
    # KataGo wants moves in GTP coordinates
    moves = []
    # We need the full move history; reconstruct from board is hard,
    # so we track it during replay instead.
    raise NotImplementedError("use replay loop below")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sgf-dir", required=True)
    ap.add_argument("--katago", default="katago")
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-visits", type=int, default=100)
    ap.add_argument("--max-positions", type=int, default=100000)
    ap.add_argument("--komi", type=float, default=7.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sgf_files = sorted(glob.glob(os.path.join(args.sgf_dir, "*.sgf")))
    print(f"Found {len(sgf_files)} SGF files")
    if not sgf_files:
        sys.exit(1)

    # Step 1: replay games, collect (obs, gtp_moves_so_far, color_to_move)
    positions: list = []  # (obs, moves_gtp, color, mask)
    for sf in sgf_files:
        with open(sf) as f:
            sgf = f.read()
        moves = parse_sgf_moves(sgf)
        board = Board()
        history_gtp: list[tuple[str, int]] = []
        for color, rc in moves:
            if len(positions) >= args.max_positions:
                break
            # Encode position BEFORE this move
            obs = observe(board, color)
            # Legal move mask
            mask = np.zeros(82, dtype=bool)
            for r in range(9):
                for c in range(9):
                    if board.is_legal(r, c, color):
                        mask[r * 9 + c] = True
            mask[PASS] = True  # pass always legal
            positions.append((obs, list(history_gtp), color, mask))
            # Play the move (None = pass)
            board.play(rc, color)
            if rc is None:
                history_gtp.append(("pass", color))
            else:
                r, c = rc
                letter = chr(ord('A') + c + (1 if c >= 8 else 0))
                gtp = f"{letter}{9 - r}"
                history_gtp.append((gtp, color))
        if len(positions) >= args.max_positions:
            break

    print(f"Collected {len(positions)} positions")

    # Step 2: build analysis queries
    queries: list = []
    for i, (obs, hist, color, mask) in enumerate(positions):
        qmoves = [[gtp, "B" if c == BLACK else "W"] for gtp, c in hist]
        queries.append({
            "id": str(i),
            "initialStones": [],
            "moves": qmoves,
            "rules": "chinese",
            "komi": args.komi,
            "boardXSize": 9,
            "boardYSize": 9,
            "analyzeTurns": [len(qmoves)],
        })

    # Step 3: run katago analysis
    cmd = [args.katago, "analysis", "-model", args.model,
           "-config", args.config]
    inp = "\n".join(json.dumps(q) for q in queries) + "\n"
    print(f"Running katago analysis on {len(queries)} positions "
          f"(maxVisits via config)...")
    proc = subprocess.run(cmd, input=inp, capture_output=True, text=True,
                          timeout=7200)
    if proc.returncode != 0:
        print(proc.stderr[-3000:], file=sys.stderr)
        raise RuntimeError("katago analysis failed")

    results = {}
    for line in proc.stdout.strip().split("\n"):
        if line.strip():
            r = json.loads(line)
            results[r["id"]] = r
    print(f"Got {len(results)} analysis results")

    # Step 4: build arrays
    N = len(positions)
    obs_arr = np.stack([p[0] for p in positions]).astype(np.float32)
    policy_arr = np.zeros((N, 82), dtype=np.float32)
    value_arr = np.zeros(N, dtype=np.float32)
    mask_arr = np.stack([p[3] for p in positions])

    for i, (obs, hist, color, mask) in enumerate(positions):
        r = results[str(i)]
        # Policy: distribute policy mass over legal moves
        info = r["moveInfos"]
        total_visits = sum(mi["visits"] for mi in info) or 1
        for mi in info:
            gtp = mi["move"]
            idx = gtp_to_idx(gtp)
            policy_arr[i, idx] = mi["visits"] / total_visits
        # Value: scoreMean from side-to-move perspective -> [-1,1] via tanh
        # KataGo reports scoreLead; map winrate if available
        if "rootInfo" in r:
            ri = r["rootInfo"]
            # Use winrate if present, else sigmoid of scoreMean / 10
            if "winrate" in ri:
                value_arr[i] = ri["winrate"] * 2 - 1
            else:
                value_arr[i] = np.tanh(ri.get("scoreLead", 0) / 15.0)
        # Renormalize policy over legal moves only
        policy_arr[i] *= mask
        s = policy_arr[i].sum()
        if s > 0:
            policy_arr[i] /= s
        else:
            # Fallback: uniform over legal
            policy_arr[i] = mask.astype(np.float32)
            policy_arr[i] /= policy_arr[i].sum()

    np.savez_compressed(args.out, obs=obs_arr, policy=policy_arr,
                        value=value_arr, mask=mask_arr)
    print(f"Saved {args.out}: obs{obs_arr.shape} policy{policy_arr.shape}")


def gtp_to_idx(gtp: str) -> int:
    if gtp.lower() == "pass":
        return PASS
    col = ord(gtp[0].upper()) - ord('A')
    if gtp[0].upper() > 'I':
        col -= 1
    row = 9 - int(gtp[1:])
    return row * 9 + col


if __name__ == "__main__":
    main()
