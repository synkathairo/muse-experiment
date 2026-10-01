"""Phase 0 of the search-distillation probe: diagnose where the MCTS gain lives.

Compares greedy, exhaustive one-ply successor-value selection, and MCTS
(at two sim counts) on representative on-policy positions from the given
checkpoint. Logs agreement, policy rank of each search move, and prior
entropy at disagreements.

Reads (per the Astra consult):
- one-ply ~= MCTS  -> gain is policy/value inconsistency (distill-friendly)
- MCTS >> one-ply  -> genuine lookahead (viable but harder)
- MCTS-50 ~= MCTS-100 -> generate distillation targets at 50 sims
- MCTS ~= greedy   -> the ladder gain was a matchup artifact; kill the probe

Usage:
  python -m gotrain.diagnose_search --checkpoint ~/workspace/runs/ppo_bc/latest.pt
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from . import rules, selfplay
from .features import index_to_move
from .mcts import PASS, SearchConfig, Searcher
from .net_aux import GoNetAux


def load_net(ckpt_path: str, device: torch.device) -> GoNetAux:
    net = GoNetAux()
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    net.load_state_dict(sd)
    net.eval()
    return net


def sample_move(logits: np.ndarray, mask: np.ndarray, temp: float,
                rng: np.random.Generator) -> int:
    if temp <= 0:
        lp = logits - logits[mask > 0].max()
    else:
        lp = logits / temp
        lp = lp - lp[mask > 0].max()
    e = np.exp(lp) * mask
    p = e / e.sum()
    return int(rng.choice(82, p=p))


def play_game(net: GoNetAux, device: torch.device, temp: float,
              rng: np.random.Generator, max_plies: int = 300) -> list[rules.Board]:
    """Play a self-play game; return cloned boards at each ply."""
    from .mcts import _clone
    b = rules.Board()
    hist: list[rules.Board] = []
    passes = 0
    while passes < 2 and len(hist) < max_plies:
        obs = selfplay.observe(b, b.to_play)
        mask = selfplay.bot_mask(b, b.to_play)
        with torch.no_grad():
            logits, _ = net(torch.from_numpy(obs).unsqueeze(0).to(device))
        m = sample_move(logits[0].float().cpu().numpy(), mask, temp, rng)
        hist.append(_clone(b))
        if m == PASS:
            passes += 1
            b.play(None, b.to_play)
        else:
            passes = 0
            b.play(index_to_move(m), b.to_play)
    return hist


def one_ply_move(net: GoNetAux, device: torch.device,
                 board: rules.Board) -> tuple[int, np.ndarray]:
    """Exhaustive one-ply: argmax over successors of -V(successor).

    Returns (move, successor_values) with values from the mover's perspective.
    """
    from .mcts import _clone
    mask = selfplay.bot_mask(board, board.to_play)
    legal = np.nonzero(mask)[0]
    succ_boards: list[rules.Board] = []
    for m in legal:
        nb = _clone(board)
        nb.play(index_to_move(int(m)), board.to_play)
        succ_boards.append(nb)
    planes = np.stack([selfplay.observe(sb, sb.to_play) for sb in succ_boards])
    with torch.no_grad():
        _, vals = net(torch.from_numpy(planes).to(device))
    v_opp = vals.float().cpu().numpy().ravel()  # opponent's perspective
    scores = -v_opp  # mover's perspective
    best = legal[int(np.argmax(scores))]
    return int(best), scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--positions-per-game", type=int, default=10)
    ap.add_argument("--sims", default="50,100",
                    help="comma-separated MCTS sim counts to compare")
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="JSON report path")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    net = load_net(args.checkpoint, device)
    rng = np.random.default_rng(args.seed)
    sim_counts = [int(x) for x in args.sims.split(",")]

    positions: list[rules.Board] = []
    for g in range(args.games):
        hist = play_game(net, device, args.temp, rng)
        usable = [b for b in hist[20:140]
                  if int(selfplay.bot_mask(b, b.to_play).sum()) > 1]
        if len(usable) < args.positions_per_game:
            continue
        idx = rng.choice(len(usable), args.positions_per_game, replace=False)
        positions.extend(usable[i] for i in idx)
    print(f"collected {len(positions)} positions from {args.games} games")

    methods = ["greedy", "oneply"] + [f"mcts{s}" for s in sim_counts]
    agree = {a: {b: 0 for b in methods} for a in methods}
    rank_of: dict[str, list[int]] = {m: [] for m in methods[1:]}
    entropy_at_disagree: dict[str, list[float]] = {m: [] for m in methods[1:]}
    n = 0

    searchers = {s: Searcher(net, device, SearchConfig(simulations=s, batch=16,
                                                      seed=args.seed))
                 for s in sim_counts}
    for pos in positions:
        mask = selfplay.bot_mask(pos, pos.to_play)
        obs = selfplay.observe(pos, pos.to_play)
        with torch.no_grad():
            logits_t, _ = net(torch.from_numpy(obs).unsqueeze(0).to(device))
        logits = logits_t[0].float().cpu().numpy()
        lp = logits - logits[mask > 0].max()
        e = np.exp(lp) * mask
        prior = e / e.sum()
        order = np.argsort(-prior)  # rank 0 = top prior

        moves: dict[str, int] = {}
        moves["greedy"] = int(np.argmax(prior))
        moves["oneply"], _ = one_ply_move(net, device, pos)
        for s in sim_counts:
            mv, visits, _ = searchers[s].search_with_visits(pos, 0)
            moves[f"mcts{s}"] = mv
            rank_of[f"mcts{s}"].append(int(np.nonzero(order == mv)[0][0]))
        rank_of["oneply"].append(int(np.nonzero(order == moves["oneply"])[0][0]))

        ent = float(-(prior[prior > 0] * np.log(prior[prior > 0])).sum())
        for m in methods[1:]:
            if moves[m] != moves["greedy"]:
                entropy_at_disagree[m].append(ent)
        for a in methods:
            for b_ in methods:
                if moves[a] == moves[b_]:
                    agree[a][b_] += 1
        n += 1

    print(f"\nagreement over {n} positions (fraction):")
    header = " " * 10 + "".join(f"{m:>10}" for m in methods)
    print(header)
    for a in methods:
        row = "".join(f"{agree[a][b_] / n:>10.2f}" for b_ in methods)
        print(f"{a:>10}{row}")
    print("\nmean policy rank of method's move (0 = top prior):")
    for m in methods[1:]:
        print(f"  {m:>10}: {np.mean(rank_of[m]):.2f}")
    print("mean prior entropy where method disagrees with greedy:")
    ent_summary: dict[str, float | None] = {}
    for m in methods[1:]:
        vals = entropy_at_disagree[m]
        ent_summary[m] = float(np.mean(vals)) if vals else None
        disp = f"{ent_summary[m]:.2f}" if ent_summary[m] is not None else "n/a"
        print(f"  {m:>10}: {disp}  (n={len(vals)})")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "n": n,
                "agreement": {a: {b_: agree[a][b_] / n for b_ in methods}
                              for a in methods},
                "mean_rank": {m: float(np.mean(rank_of[m])) for m in methods[1:]},
                "mean_entropy_at_disagree": ent_summary,
            }, f, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
