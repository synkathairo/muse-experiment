"""Play our net (via gotrain.gtp) against GNU Go over GTP.

Spawns `gnugo --mode gtp` and our GTP engine as subprocesses, relays moves,
and arbitrates with gotrain.rules + Tromp-Taylor scoring (7.5 komi). GNU Go
runs with --chinese-rules so both sides agree on area scoring.

Usage:
    python eval_vs_gnugo.py --checkpoint runs/auto_pilot/snap_003000000.pt \\
        --levels 1 3 5 --games 4 --out /tmp/gnugo_bench.json
"""

import argparse
import json
import shlex
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


def play_one(task):
    """Play one game with fresh GTP subprocesses. `task` is a plain tuple
    so it pickles across process boundaries for --jobs > 1."""
    level, gi, gnugo_argv, net_argv, move_timeout, temperature = task
    gnugo = GTPClient(gnugo_argv)
    net = GTPClient(net_argv)
    try:
        # sanity: both speak GTP
        for eng, nm in ((gnugo, "gnugo"), (net, "net")):
            ok, resp = eng.command("protocol_version", timeout=30)
            assert ok and resp.strip() == "2", f"{nm} GTP broken: {resp}"
        net_is_black = (gi % 2 == 0)
        t0 = time.time()
        winner, reason, moves = play_game(
            gnugo, net, net_is_black, move_timeout=move_timeout)
        dt = time.time() - t0
        return {"level": level, "game": gi,
                "net_black": net_is_black, "winner": winner,
                "reason": reason, "moves": moves,
                "temperature": temperature,
                "seconds": round(dt, 1)}
    finally:
        gnugo.close()
        net.close()


def _report(r, prefix=""):
    print(f"{prefix}level {r['level']} game {r['game']} "
          f"({'net' if r['net_black'] else 'gnugo'} black): "
          f"{r['winner']} wins [{r['reason']}] "
          f"({r['moves']} moves, {r['seconds']}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="torch checkpoint for the Python GTP engine "
                         "(not needed with --engine-cmd)")
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--games", type=int, default=4,
                    help="games per level (alternating colors)")
    ap.add_argument("--out", default="/tmp/gnugo_bench.json")
    ap.add_argument("--gnugo", default="gnugo")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="net move temperature: 0 = greedy (default), >0 samples")
    ap.add_argument("--engine-cmd", default=None,
                    help="override the net GTP engine command (shlex-split). "
                         "E.g. \"./target/release/mcts_gtp --blob <w.bin> --sims 100\". "
                         "When set, --checkpoint is not needed.")
    ap.add_argument("--jobs", type=int, default=1,
                    help="games to run in parallel (default 1 = sequential). "
                         "Each game gets its own GNU Go + engine subprocesses.")
    args = ap.parse_args()

    py = sys.executable
    if args.engine_cmd:
        net_argv = shlex.split(args.engine_cmd)
    else:
        if not args.checkpoint:
            ap.error("--checkpoint is required unless --engine-cmd is given")
        net_argv = [py, "-m", "gotrain.gtp", "--checkpoint", args.checkpoint,
                    "--temperature", str(args.temperature)]

    tasks = []
    for level in args.levels:
        gnugo_argv = [args.gnugo, "--mode", "gtp", "--quiet",
                      "--boardsize", "9", "--chinese-rules",
                      "--komi", "7.5", "--level", str(level)]
        move_timeout = 300 if level >= 10 else 120
        for gi in range(args.games):
            tasks.append((level, gi, gnugo_argv, net_argv,
                          move_timeout, args.temperature))

    results = []
    if args.jobs <= 1:
        for t in tasks:
            r = play_one(t)
            results.append(r)
            _report(r)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(play_one, t) for t in tasks]
            try:
                done = 0
                for fut in as_completed(futs):
                    r = fut.result()
                    results.append(r)
                    done += 1
                    _report(r, prefix=f"[{done}/{len(tasks)}] ")
            except KeyboardInterrupt:
                ex.shutdown(cancel_futures=True)
                raise

    results.sort(key=lambda r: (r["level"], r["game"]))
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")
    for level in args.levels:
        rs = [r for r in results if r["level"] == level]
        nw = sum(1 for r in rs if r["winner"] == "net")
        print(f"level {level}: net {nw}-{len(rs) - nw}")


if __name__ == "__main__":
    main()
