"""GTP server: MCTS over a GoNet checkpoint, for benchmarking search strength.

Speaks GTP v2 on stdin/stdout so eval_vs_gnugo.py can use it via
--engine-cmd. Rules/scoring authority is gotrain.rules + gotrain.selfplay
(Tromp-Taylor, komi from the GTP `komi` command); this process only
generates moves via gotrain.mcts.

Usage:
    python -m gotrain.mcts_gtp --checkpoint runs/smoke/latest.pt \
        --sims 100 --batch 16
"""

import argparse
import sys

import torch

from . import rules, selfplay
from .gtp import from_gtp_vertex, gtp_color, to_gtp_vertex
from .mcts import SearchConfig, Searcher
from .net import GoNet


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MCTSEngine:
    def __init__(self, model, device, cfg):
        self.searcher = Searcher(model, device, cfg)
        self.board = rules.Board(9)
        self.passes = 0  # consecutive passes so far (search terminal state)

    def genmove(self, color):
        assert color == self.board.to_play, "engine/arbiter desync"
        m = self.searcher.search(self.board, self.passes)
        move = None if m == 81 else divmod(m, 9)
        assert self.board.play(move, color), f"search emitted illegal move {move}"
        self.passes = self.passes + 1 if move is None else 0
        return to_gtp_vertex(move)


COMMANDS = [
    "protocol_version", "name", "version", "known_command", "list_commands",
    "boardsize", "clear_board", "komi", "play", "genmove", "quit",
]


def serve(engine):
    for line in sys.stdin:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        cid, cmd = "", parts[0]
        if cmd[0].isdigit():
            cid, cmd = cmd, parts[1]
            args = parts[2:]
        else:
            args = parts[1:]

        def reply(data=""):
            prefix = f"={cid}" if cid else "="
            sys.stdout.write(f"{prefix} {data}\n\n".rstrip() + "\n\n")
            sys.stdout.flush()

        def err(msg):
            prefix = f"?{cid}" if cid else "?"
            sys.stdout.write(f"{prefix} {msg}\n\n")
            sys.stdout.flush()

        try:
            if cmd == "protocol_version":
                reply("2")
            elif cmd == "name":
                reply("weiqi-mcts-py")
            elif cmd == "version":
                reply("0.1.0")
            elif cmd == "known_command":
                reply("true" if args[0] in COMMANDS else "false")
            elif cmd == "list_commands":
                reply("\n" + "\n".join(COMMANDS))
            elif cmd == "boardsize":
                if args[0] != "9":
                    err("only 9x9 supported")
                else:
                    reply("")
            elif cmd == "clear_board":
                engine.board = rules.Board(9)
                engine.passes = 0
                reply("")
            elif cmd == "komi":
                selfplay.set_komi(float(args[0]))
                reply("")
            elif cmd == "play":
                color = gtp_color(args[0])
                move = from_gtp_vertex(args[1])
                if engine.board.play(move, color):
                    engine.passes = engine.passes + 1 if move is None else 0
                    reply("")
                else:
                    err(f"illegal move {args[1]}")
            elif cmd == "genmove":
                reply(engine.genmove(gtp_color(args[0])))
            elif cmd == "quit":
                reply("")
                return
            else:
                err(f"unknown command: {cmd}")
        except Exception as e:  # noqa: BLE001 - GTP must never die mid-match
            err(f"internal error: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16,
                    help="leaves evaluated per forward pass (1 = sequential, "
                         "exactly the Rust/demo algorithm)")
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--dirichlet-eps", type=float, default=0.15)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="final move selection temperature over visits; "
                         "0 = argmax (default)")
    args = ap.parse_args()
    device = pick_device()
    print(f"mcts_gtp: device={device} sims={args.sims} batch={args.batch}",
          file=sys.stderr)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model = GoNet()
    model.load_state_dict(sd)
    model.to(device)
    cfg = SearchConfig(simulations=args.sims, batch=args.batch,
                       c_puct=args.c_puct, dirichlet_eps=args.dirichlet_eps,
                       temperature=args.temperature)
    serve(MCTSEngine(model, device, cfg))


if __name__ == "__main__":
    main()
