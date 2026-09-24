"""Play our net (greedy, via gotrain.gtp) against GNU Go over GTP.

Spawns `gnugo --mode gtp` and our GTP engine as subprocesses, relays moves,
and arbitrates with gotrain.rules + Tromp-Taylor scoring (7.5 komi). GNU Go
runs with --chinese-rules so both sides agree on area scoring.

Usage:
    python eval_vs_gnugo.py --checkpoint runs/auto_pilot/snap_003000000.pt \\
        --levels 1 3 5 --games 4 --out /tmp/gnugo_bench.json
"""

import argparse
import json
import subprocess
import sys
import time

from gotrain import rules, selfplay
from gotrain.gtp import from_gtp_vertex, gtp_color  # noqa: F401 (vertex parse)

NET_COLORS = ("black", "white")


class GTPClient:
    def __init__(self, argv):
        self.proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def command(self, cmd, timeout=180):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()
        lines, data = [], []
        start = time.time()
        while True:
            if time.time() - start > timeout:
                raise TimeoutError(f"GTP timeout on: {cmd}")
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"GTP engine died on: {cmd}")
            s = line.rstrip("\n")
            if s == "":
                break
            lines.append(s)
        if not lines or not lines[0][:1] in ("=", "?"):
            raise RuntimeError(f"bad GTP response to {cmd!r}: {lines}")
        ok = lines[0][0] == "="
        # Strip "= "/"? " prefix. (We never send numeric command ids,
        # so no id-echo stripping is needed.)
        first = lines[0][1:].strip()
        rest = [l for l in lines[1:]]
        payload = "\n".join([first] + rest).strip()
        return ok, payload

    def close(self):
        try:
            self.command("quit", timeout=10)
        except Exception:
            pass
        self.proc.terminate()


def play_game(gnugo, net, net_is_black, move_timeout):
    arbiter = rules.Board(9)
    for eng in (gnugo, net):
        eng.command("boardsize 9")
        eng.command("clear_board")
        eng.command("komi 7.5")
    passes, moves = 0, 0
    winner, reason = None, ""
    while moves < 250:
        black_to_move = (moves % 2 == 0)
        me = net if (black_to_move == net_is_black) else gnugo
        them = gnugo if me is net else net
        color = "black" if black_to_move else "white"
        ok, resp = me.command(f"genmove {color}", timeout=move_timeout)
        if not ok:
            winner = "them"
            reason = f"{'net' if me is net else 'gnugo'} genmove error: {resp}"
            break
        v = resp.strip().lower()
        if v == "resign":
            winner = "them"
            reason = f"{'net' if me is net else 'gnugo'} resigned"
            break
        try:
            mv = from_gtp_vertex(v)
        except Exception as e:
            winner = "them"
            reason = f"unparseable genmove {resp!r}: {e}"
            break
        gcolor = rules.BLACK if black_to_move else rules.WHITE
        if not arbiter.play(mv, gcolor):
            winner = "them"
            reason = f"{'net' if me is net else 'gnugo'} illegal move {resp}"
            break
        ok2, r2 = them.command(f"play {color} {v}")
        if not ok2:
            # opponent rejects a move the arbiter accepted: trust arbiter, note it
            reason = f"relay rejected by {'net' if them is net else 'gnugo'}: {r2}"
        moves += 1
        passes = passes + 1 if mv is None else 0
        if passes >= 2:
            break
    else:
        reason = "move cap 250"
    if winner is None:
        b, w = selfplay.score(arbiter)
        winner = "net" if (b > w) == net_is_black else "gnugo"
        reason = f"score B:{b:.1f} W:{w:.1f}"
    return winner, reason, moves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--games", type=int, default=4,
                    help="games per level (alternating colors)")
    ap.add_argument("--out", default="/tmp/gnugo_bench.json")
    ap.add_argument("--gnugo", default="gnugo")
    args = ap.parse_args()

    py = sys.executable
    results = []
    for level in args.levels:
        gnugo = GTPClient([args.gnugo, "--mode", "gtp", "--quiet",
                           "--boardsize", "9", "--chinese-rules",
                           "--komi", "7.5", "--level", str(level)])
        net = GTPClient([py, "-m", "gotrain.gtp", "--checkpoint", args.checkpoint])
        # sanity: both speak GTP
        for eng, nm in ((gnugo, "gnugo"), (net, "net")):
            ok, resp = eng.command("protocol_version", timeout=30)
            assert ok and resp.strip() == "2", f"{nm} GTP broken: {resp}"
        for gi in range(args.games):
            net_is_black = (gi % 2 == 0)
            t0 = time.time()
            winner, reason, moves = play_game(
                gnugo, net, net_is_black,
                move_timeout=300 if level >= 10 else 120)
            dt = time.time() - t0
            results.append({"level": level, "game": gi,
                            "net_black": net_is_black, "winner": winner,
                            "reason": reason, "moves": moves,
                            "seconds": round(dt, 1)})
            print(f"level {level} game {gi} "
                  f"({'net' if net_is_black else 'gnugo'} black): "
                  f"{winner} wins [{reason}] ({moves} moves, {dt:.0f}s)",
                  flush=True)
        gnugo.close()
        net.close()

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")
    for level in args.levels:
        rs = [r for r in results if r["level"] == level]
        nw = sum(1 for r in rs if r["winner"] == "net")
        print(f"level {level}: net {nw}-{len(rs) - nw}")


if __name__ == "__main__":
    main()
