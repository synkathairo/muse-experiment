"""Build (planes -> move) datasets from SGF games.

For each game we replay with gotrain.rules and emit one training pair per move:
input = 6-plane encoding from the mover's perspective, target = move index
(0..80 row-major, 81 = pass), value target = game result from the mover's
perspective (+1 win / -1 loss), masked to 0 when the result is unknown.

Lenient filtering (PLAN.md §4): on the first move that is illegal under our
simple rules we truncate the game (keep pairs so far, drop the rest) — KGS
games are Japanese-rules; legality edge cases are rare and not worth a fight.
Games excluded outright: timeouts, <10 moves, bot players.

Game-level train/val split via a stable hash so re-runs agree.

Usage:
    python -m gotrain.dataset --sgf data/kgs --out data/ds_full [--val-frac 0.02]

Writes data/ds_full/{train,val}_{x,y,z,zm}.npy (memmap-friendly) + meta.json.
"""

import argparse
import hashlib
import json
import os

import numpy as np

from . import features, rules, sgf

MIN_MOVES = 10
CHUNK_POSITIONS = 100_000


def game_split(source_label, game_idx, val_frac, seed):
    h = hashlib.md5(f"{seed}|{source_label}|{game_idx}".encode()).hexdigest()
    return (int(h[:8], 16) / 0xFFFFFFFF) < val_frac


def game_to_pairs(game):
    """Replay a game -> list of (planes, move_idx, z, zm). Truncates on illegal move."""
    board = rules.Board(9)
    for r, c in game.setup_black:
        board.grid[r][c] = rules.BLACK
    for r, c in game.setup_white:
        board.grid[r][c] = rules.WHITE
    z_black = sgf.result_to_z(game.result)
    pairs = []
    for color_s, move in game.moves:
        color = rules.BLACK if color_s == "B" else rules.WHITE
        if color != board.to_play:
            break  # color order diverged (e.g. odd setup): keep the good prefix
        own = board.stones(color)
        opp = board.stones(rules.opponent(color))
        planes = features.encode(
            own, opp,
            last_move=board.last_move,
            black_to_move=(color == rules.BLACK),
            ko_point=board.ko,
        )
        if z_black is None:
            z, zm = 0.0, 0.0
        else:
            z = z_black if color == rules.BLACK else -z_black
            zm = 1.0
        pairs.append((planes, features.move_to_index(move), z, zm))
        if not board.play(move, color):
            break  # lenient: truncate on first illegal move
    return pairs


class ChunkWriter:
    """Accumulate positions; flush to .npz chunks to bound RAM."""

    def __init__(self, chunk_dir, prefix):
        self.chunk_dir = chunk_dir
        self.prefix = prefix
        self.bufs = []
        self.n_buf = 0
        self.chunk_idx = 0
        self.total = 0
        os.makedirs(chunk_dir, exist_ok=True)

    def add(self, pairs):
        for planes, mi, z, zm in pairs:
            self.bufs.append((planes, mi, z, zm))
            self.n_buf += 1
        if self.n_buf >= CHUNK_POSITIONS:
            self.flush()

    def flush(self):
        if not self.n_buf:
            return
        n = self.n_buf
        X = np.empty((n, 6, 9, 9), dtype=np.float32)
        y = np.empty((n,), dtype=np.int64)
        z = np.empty((n,), dtype=np.float32)
        zm = np.empty((n,), dtype=np.float32)
        for i, (planes, mi, zv, zvm) in enumerate(self.bufs):
            X[i] = planes
            y[i] = mi
            z[i] = zv
            zm[i] = zvm
        path = os.path.join(self.chunk_dir, f"{self.prefix}_{self.chunk_idx:04d}.npz")
        np.savez_compressed(path, X=X, y=y, z=z, zm=zm)
        self.chunk_idx += 1
        self.total += n
        self.bufs = []
        self.n_buf = 0

    def chunk_paths(self):
        self.flush()
        return [
            os.path.join(self.chunk_dir, f"{self.prefix}_{i:04d}.npz")
            for i in range(self.chunk_idx)
        ]


def concat_chunks(chunk_paths, out_dir, split):
    """Concatenate chunk npzs into final .npy files (proper format, mmap-able)."""
    from numpy.lib.format import open_memmap
    total = 0
    for p in chunk_paths:
        with np.load(p) as d:
            total += d["y"].shape[0]
    X = open_memmap(os.path.join(out_dir, f"{split}_x.npy"), dtype=np.float32,
                    mode="w+", shape=(total, 6, 9, 9))
    y = open_memmap(os.path.join(out_dir, f"{split}_y.npy"), dtype=np.int64,
                    mode="w+", shape=(total,))
    z = open_memmap(os.path.join(out_dir, f"{split}_z.npy"), dtype=np.float32,
                    mode="w+", shape=(total,))
    zm = open_memmap(os.path.join(out_dir, f"{split}_zm.npy"), dtype=np.float32,
                     mode="w+", shape=(total,))
    off = 0
    for p in chunk_paths:
        with np.load(p) as d:
            n = d["y"].shape[0]
            X[off:off + n] = d["X"]
            y[off:off + n] = d["y"]
            z[off:off + n] = d["z"]
            zm[off:off + n] = d["zm"]
            off += n
        os.remove(p)
    for m in (X, y, z, zm):
        m.flush()
    return total


def build_dataset(sgf_paths, out_dir, val_frac=0.02, seed=1234):
    os.makedirs(out_dir, exist_ok=True)
    chunk_dir = os.path.join(out_dir, "_chunks")
    train_w = ChunkWriter(chunk_dir, "train")
    val_w = ChunkWriter(chunk_dir, "val")
    n_games = n_kept = n_pairs = 0
    game_idx = 0
    for label, text in sgf.iter_sgf_texts(sgf_paths):
        game = sgf.parse_sgf(text)
        game_idx += 1
        if game is None:
            continue
        n_games += 1
        if sgf.is_timeout(game.result):
            continue
        if len(game.moves) < MIN_MOVES:
            continue
        if sgf.looks_like_bot(game.black_name) or sgf.looks_like_bot(game.white_name):
            continue
        try:
            pairs = game_to_pairs(game)
        except Exception:
            continue  # one weird game must not kill the build
        if not pairs:
            continue
        n_kept += 1
        n_pairs += len(pairs)
        if game_split(label, game_idx, val_frac, seed):
            val_w.add(pairs)
        else:
            train_w.add(pairs)
        if n_games % 20000 == 0:
            print(f"  ...{n_games} games scanned, {n_kept} kept, {n_pairs} pairs", flush=True)
    print(f"scanned {n_games} games, kept {n_kept}, {n_pairs} pairs; concatenating...", flush=True)
    n_train = concat_chunks(train_w.chunk_paths(), out_dir, "train")
    n_val = concat_chunks(val_w.chunk_paths(), out_dir, "val")
    os.rmdir(chunk_dir)
    meta = {
        "val_frac": val_frac, "seed": seed,
        "games_scanned": n_games, "games_kept": n_kept,
        "n_train": n_train, "n_val": n_val,
        "sources": list(sgf_paths),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {out_dir}: train={n_train} val={n_val}", flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sgf", nargs="+", required=True, help=".sgf files / dirs / .zips")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    build_dataset(args.sgf, args.out, args.val_frac, args.seed)


if __name__ == "__main__":
    main()
