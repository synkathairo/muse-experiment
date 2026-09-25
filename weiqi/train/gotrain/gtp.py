"""GTP server wrapping a trained GoNet checkpoint.

Speaks GTP v2 on stdin/stdout so the net can play matches against GNU Go
(`gnugo --mode gtp`) via a match runner. Rules/scoring authority is
gotrain.rules + gotrain.selfplay (Tromp-Taylor area scoring, 7.5 komi);
this process only generates moves.

Move selection is argmax at --temperature 0 (default, legacy greedy) or
samples from softmax(logits / temperature) over legal moves otherwise.

Usage:
    python -m gotrain.gtp --checkpoint runs/auto_pilot/snap_003000000.pt
    python -m gotrain.gtp --checkpoint runs/smoke/latest.pt --temperature 0.5
"""

import argparse
import sys

import numpy as np
import torch

from . import features, rules, selfplay
from .net import GoNet
from .net_aux import to_gonet_state_dict

COLS = "ABCDEFGHJKLMNOPQRST"  # GTP skips 'I'


def to_gtp_vertex(move):
    """(r, c) with r=0 top -> GTP 'D4'; None -> 'pass'."""
    if move is None:
        return "pass"
    r, c = move
    return f"{COLS[c]}{9 - r}"


def from_gtp_vertex(s):
    """GTP 'D4'/'pass' -> (r, c) or None."""
    s = s.strip().lower()
    if s == "pass":
        return None
    col_c = s[0].upper()
    row_n = int(s[1:])
    c = COLS.index(col_c)
    r = 9 - row_n
    if not (0 <= r < 9 and 0 <= c < 9):
        raise ValueError(f"vertex out of range: {s}")
    return (r, c)


def gtp_color(s):
    s = s.strip().lower()
    if s in ("b", "black"):
        return rules.BLACK
    if s in ("w", "white"):
        return rules.WHITE
    raise ValueError(f"bad color: {s}")


class GTPEngine:
    def __init__(self, model, temperature=0.0):
        self.model = model
        self.model.eval()
        self.temperature = temperature
        self.board = rules.Board(9)

    def genmove(self, color):
        mask = selfplay.legal_mask(self.board, color)
        planes = features.encode(
            self.board.stones(color),
            self.board.stones(rules.opponent(color)),
            last_move=self.board.last_move,
            black_to_move=(color == rules.BLACK),
            ko_point=self.board.ko,
        )
        with torch.no_grad():
            logits, _ = self.model(torch.from_numpy(planes).unsqueeze(0))
        logits = logits.squeeze(0).numpy().astype(np.float64)
        logits[~mask] = -np.inf
        if self.temperature <= 0:
            idx = int(np.argmax(logits))
        else:
            z = logits / self.temperature
            finite = np.isfinite(z)
            z = z - z[finite].max()
            probs = np.zeros_like(z)
            probs[finite] = np.exp(z[finite])
            s = probs.sum()
            idx = (int(np.random.choice(probs.size, p=probs / s))
                   if s > 0 else int(np.argmax(logits)))
        move = None if idx == 81 else divmod(idx, 9)
        assert self.board.play(move, color), f"engine emitted illegal move {move}"
        return to_gtp_vertex(move)

    def final_score(self):
        b, w = selfplay.score(self.board)
        if b > w:
            return f"B+{b - w:.1f}"
        return f"W+{w - b:.1f}"


COMMANDS = [
    "protocol_version", "name", "version", "known_command", "list_commands",
    "boardsize", "clear_board", "komi", "play", "genmove", "final_score", "quit",
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

        # canonical GTP response form: "= [id] data" / "? [id] message"
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
                reply("weiqi-net")
            elif cmd == "version":
                reply("pilot-3m")
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
                reply("")
            elif cmd == "komi":
                reply("")  # fixed 7.5 in scoring
            elif cmd == "play":
                color = gtp_color(args[0])
                move = from_gtp_vertex(args[1])
                if engine.board.play(move, color):
                    reply("")
                else:
                    err(f"illegal move {args[1]}")
            elif cmd == "genmove":
                reply(engine.genmove(gtp_color(args[0])))
            elif cmd == "final_score":
                reply(engine.final_score())
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
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = greedy argmax (default); >0 samples softmax(logits/T)")
    args = ap.parse_args()
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model = GoNet()
    model.load_state_dict(to_gonet_state_dict(sd))
    serve(GTPEngine(model, temperature=args.temperature))


if __name__ == "__main__":
    main()
