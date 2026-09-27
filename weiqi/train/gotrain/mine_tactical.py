"""Tactical-blunder positions: mining + blunder-rate evaluation.

A position is *tactical* when the side to move has a 1-ply forced tactical move:

- CAPTURE: an opponent group with exactly 1 liberty L, where playing L is
  legal (it captures by construction; legality only fails on ko).
- ESCAPE: an own group with exactly 1 liberty L, where playing L is legal and
  leaves the resulting own group with >= 2 liberties (a clean escape).

The correct set is the union of all such moves; a position enters the set only
when the correct set is non-empty. Blunder = the net's greedy masked-argmax
move is not in the correct set.

Sources: (a) frozen-baseline self-play games, (b) frozen-baseline vs GNU Go
games the net lost. Positions are stored as (grid bytes, last_move, ko,
to_move, correct moves) — re-encodable without game history. Caveat: eval-time
legality uses an empty history, so positional-superko edge cases are ignored;
simple ko IS handled via board.ko.
"""

import json
import os
import sys

import numpy as np
import torch

from . import rules
from .rules import Board, EMPTY, BLACK, WHITE, opponent
from .selfplay import legal_mask, observe
from . import tactical as tacmod

N = 9


def _libs_after(board, r, c, color):
    """Liberty count of the new own group if `color` plays (r, c).

    Tentative apply + revert, mirroring Board.is_legal's pattern. Assumes the
    move is legal (call is_legal first).
    """
    p = r * N + c
    grid = board.grid
    grid[r][c] = color
    captured = board._would_capture(p, color)
    saved = [s for s in captured]
    for s in saved:
        grid[s // N][s % N] = EMPTY
    _, libs = board._group_int(p)
    n = len(libs)
    grid[r][c] = EMPTY
    for s in saved:
        grid[s // N][s % N] = opponent(color)
    return n


def tactical_answers(board, color):
    """Sorted [(move_idx, kind)] of 1-ply tactical moves; kind in
    {"capture", "escape"}. Empty list => not a tactical position."""
    ans = {}
    seen = bytearray(N * N)
    opp = opponent(color)
    for r in range(N):
        for c in range(N):
            p = r * N + c
            stone = board.grid[r][c]
            if stone == EMPTY or seen[p]:
                continue
            pts, libs = board._group_int(p)
            for s in pts:
                seen[s] = 1
            if len(libs) != 1:
                continue
            lr, lc = board._rc[next(iter(libs))]
            if not board.is_legal(lr, lc, color):
                continue
            mi = lr * N + lc
            if stone == opp:
                ans[mi] = "capture"
            elif _libs_after(board, lr, lc, color) >= 2:
                ans[mi] = "escape"
    return sorted(ans.items())


def encode_pos(board, color, use_tactical):
    if use_tactical:
        return tacmod.encode_tactical(board, color)
    return observe(board, color)


def greedy_move(model, board, color, use_tactical):
    planes = encode_pos(board, color, use_tactical)
    mask = legal_mask(board, color)
    with torch.no_grad():
        logits = model(torch.from_numpy(planes).unsqueeze(0))[0].squeeze(0)
    logits = logits.numpy().astype(np.float64)
    logits[~mask] = -np.inf
    return int(np.argmax(logits))


def snapshot(board, color):
    """Serialisable position record (no history)."""
    raw = bytes(b for row in board.grid for b in row)
    return {
        "grid": raw.hex(),
        "last_move": list(board.last_move) if board.last_move else None,
        "ko": list(board.ko) if board.ko else None,
        "to_move": "b" if color == BLACK else "w",
    }


def restore(rec):
    """Rebuild a Board from a snapshot record (empty history — see caveat)."""
    raw = bytes.fromhex(rec["grid"])
    b = Board(N)
    it = iter(raw)
    b.grid = [[next(it) for _ in range(N)] for _ in range(N)]
    b.last_move = tuple(rec["last_move"]) if rec["last_move"] else None
    b.ko = tuple(rec["ko"]) if rec["ko"] else None
    color = BLACK if rec["to_move"] == "b" else WHITE
    return b, color


def mine_selfplay(model_ckpt, n_games=40, seed=0, temperature=0.7,
                  max_positions=500):
    """Play the frozen 6-plane baseline vs itself (sampled), mining tactical
    positions at every turn. Returns a list of position records."""
    from .net import GoNet
    from .net_aux import to_gonet_state_dict

    ck = torch.load(model_ckpt, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model = GoNet()
    model.load_state_dict(to_gonet_state_dict(sd))
    model.eval()
    rng = np.random.default_rng(seed)
    positions = []
    seen = set()
    for _ in range(n_games):
        b = Board(N)
        color = BLACK
        passes = 0
        plies = 0
        while passes < 2 and plies < 243:
            answers = tactical_answers(b, color)
            if answers:
                rec = snapshot(b, color)
                key = (rec["grid"], rec["to_move"])
                if key not in seen:
                    seen.add(key)
                    gm = greedy_move(model, b, color, use_tactical=False)
                    correct = [mi for mi, _ in answers]
                    rec["correct"] = [{"move": mi, "kind": k}
                                      for mi, k in answers]
                    rec["baseline_move"] = gm
                    rec["baseline_blunder"] = gm not in correct
                    rec["source"] = "selfplay"
                    positions.append(rec)
                    if len(positions) >= max_positions:
                        return positions
            # sampled move for game diversity
            mask = legal_mask(b, color)
            planes = observe(b, color)
            with torch.no_grad():
                logits = model(torch.from_numpy(planes).unsqueeze(0))[0]
                logits = logits.squeeze(0).numpy().astype(np.float64)
            logits[~mask] = -np.inf
            z = (logits - logits[mask].max()) / temperature
            probs = np.zeros_like(z)
            probs[mask] = np.exp(z[mask])
            probs /= probs.sum()
            mi = int(rng.choice(82, p=probs))
            mv = None if mi == 81 else divmod(mi, 9)
            assert b.play(mv, color), "miner generated illegal move"
            passes = passes + 1 if mv is None else 0
            color = opponent(color)
            plies += 1
    return positions


def mine_vs_gnugo(net_argv, gnugo_argv, n_games=8, max_positions=500):
    """Play the frozen baseline (as a GTP subprocess) vs GNU Go, mining
    tactical positions from the net's turns in games the net LOST."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
    from eval_vs_gnugo import GTPClient
    from gotrain.gtp import from_gtp_vertex

    positions = []
    seen = set()
    for gi in range(n_games):
        gnugo = GTPClient(gnugo_argv)
        net = GTPClient(net_argv)
        try:
            for eng in (gnugo, net):
                eng.command("boardsize 9")
                eng.command("clear_board")
                eng.command("komi 7.5")
            net_is_black = (gi % 2 == 0)
            arbiter = Board(N)
            pending = []  # (record, answers) at net's turns
            moves, passes = 0, 0
            winner = None
            while moves < 250:
                black_to_move = (moves % 2 == 0)
                me = net if (black_to_move == net_is_black) else gnugo
                them = gnugo if me is net else net
                color = "black" if black_to_move else "white"
                gcolor = BLACK if black_to_move else WHITE
                if me is net:
                    answers = tactical_answers(arbiter, gcolor)
                    rec = snapshot(arbiter, gcolor) if answers else None
                ok, resp = me.command(f"genmove {color}", timeout=120)
                if not ok:
                    winner = "them"
                    break
                v = resp.strip().lower()
                if v == "resign":
                    winner = "them"
                    break
                mv = from_gtp_vertex(v)
                if not arbiter.play(mv, gcolor):
                    winner = "them"
                    break
                them.command(f"play {color} {v}")
                if me is net and rec is not None:
                    mi = 81 if mv is None else mv[0] * N + mv[1]
                    pending.append((rec, answers, mi))
                moves += 1
                passes = passes + 1 if mv is None else 0
                if passes >= 2:
                    break
            if winner is None:
                sb, sw = _score(arbiter)
                net_won = (sb > sw) == net_is_black
                winner = "net" if net_won else "gnugo"
            if winner == "gnugo":
                for rec, answers, mi in pending:
                    key = (rec["grid"], rec["to_move"])
                    if key in seen:
                        continue
                    seen.add(key)
                    correct = [m for m, _ in answers]
                    rec["correct"] = [{"move": m, "kind": k}
                                      for m, k in answers]
                    rec["baseline_move"] = mi
                    rec["baseline_blunder"] = mi not in correct
                    rec["source"] = "gnugo-loss"
                    positions.append(rec)
                    if len(positions) >= max_positions:
                        return positions
        finally:
            gnugo.close()
            net.close()
    return positions


def _score(board):
    from .selfplay import score
    return score(board)


def load_model_for_eval(checkpoint):
    """(model, use_tactical): auto-detect 13-plane vs 6-plane checkpoints."""
    from .net import GoNet, GoNetTactical
    from .net_aux import to_gonet_state_dict

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    sd = to_gonet_state_dict(sd)
    if sd["conv1.weight"].shape[1] == 13:
        model = GoNetTactical()
        model.load_state_dict(sd)
        return model.eval(), True
    model = GoNet()
    model.load_state_dict(sd)
    return model.eval(), False


def eval_blunders(checkpoint, positions):
    """Blunder rate of `checkpoint` on mined positions.

    Returns dict with n, blunders, rate, and capture/escape splits.
    """
    model, use_tactical = load_model_for_eval(checkpoint)
    n = len(positions)
    bl = cap_n = cap_bl = esc_n = esc_bl = 0
    for rec in positions:
        board, color = restore(rec)
        gm = greedy_move(model, board, color, use_tactical)
        correct = {c["move"] for c in rec["correct"]}
        kinds = {c["move"]: c["kind"] for c in rec["correct"]}
        if gm not in correct:
            bl += 1
        # split credit: a position counts for a kind if any correct move is
        # of that kind and the played move is not correct at all
        if any(k == "capture" for k in kinds.values()):
            cap_n += 1
            cap_bl += gm not in correct
        if any(k == "escape" for k in kinds.values()):
            esc_n += 1
            esc_bl += gm not in correct
    return {
        "n": n,
        "blunders": bl,
        "rate": bl / n if n else 0.0,
        "capture": {"n": cap_n, "blunders": cap_bl,
                    "rate": cap_bl / cap_n if cap_n else 0.0},
        "escape": {"n": esc_n, "blunders": esc_bl,
                   "rate": esc_bl / esc_n if esc_n else 0.0},
    }


def main():
    ap = __import__("argparse").ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mine", help="mine positions from frozen baseline")
    m.add_argument("--checkpoint", default=tacmod.BASE_CHECKPOINT)
    m.add_argument("--selfplay-games", type=int, default=40)
    m.add_argument("--gnugo-games", type=int, default=8)
    m.add_argument("--gnugo", default="gnugo")
    m.add_argument("--out", required=True)
    e = sub.add_parser("eval", help="blunder rate of a checkpoint on a set")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--positions", required=True)
    args = ap.parse_args()

    if args.cmd == "mine":
        net_argv = [sys.executable, "-m", "gotrain.gtp",
                    "--checkpoint", args.checkpoint]
        # same GNU Go invocation as the benchmark ladder (eval_vs_gnugo.py)
        gnugo_argv = [args.gnugo, "--mode", "gtp", "--quiet",
                      "--boardsize", "9", "--chinese-rules",
                      "--komi", "7.5", "--level", "5"]
        pos = mine_selfplay(args.checkpoint,
                            n_games=args.selfplay_games)
        print(f"self-play positions: {len(pos)}", flush=True)
        if args.gnugo_games > 0:
            g = mine_vs_gnugo(net_argv, gnugo_argv,
                              n_games=args.gnugo_games)
            print(f"gnugo-loss positions: {len(g)}", flush=True)
            pos.extend(g)
        # dedup across sources (same key scheme)
        seen, uniq = set(), []
        for rec in pos:
            key = (rec["grid"], rec["to_move"])
            if key not in seen:
                seen.add(key)
                uniq.append(rec)
        bl = sum(r["baseline_blunder"] for r in uniq)
        print(f"total {len(uniq)} positions; "
              f"frozen-baseline blunder rate {bl}/{len(uniq)}={bl/len(uniq):.3f}",
              flush=True)
        with open(args.out, "w") as f:
            json.dump(uniq, f)
        print(f"wrote {args.out}", flush=True)
    else:
        with open(args.positions) as f:
            positions = json.load(f)
        res = eval_blunders(args.checkpoint, positions)
        print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
