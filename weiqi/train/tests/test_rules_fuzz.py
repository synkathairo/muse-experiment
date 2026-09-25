"""Differential rules fuzz: gotrain.rules vs naive reference (and Rust engine).

The naive reference in tests/ref_rules.py is deliberately independent of
gotrain/rules.py: flat board, copy-and-simulate legality, and POSITIONAL ko
(a move is ko-banned iff it recreates any previous board (positional superko)) instead of the
lone-stone heuristic. Agreement between the two on random games plus
hand-built ko/snapback/seki/suicide positions is strong evidence both are
right. The Rust engine (weiqi/engine) is diffed too when cargo is available.

Run from weiqi/train/:  python -m unittest tests.test_rules_fuzz -v
"""

import os
import random
import shutil
import subprocess
import unittest

from gotrain.rules import BLACK, WHITE, Board
from gotrain import selfplay
from tests import ref_rules
from tests.ref_rules import RefBoard

ENGINE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "engine")
DRIVER_SRC = os.path.join(ENGINE_DIR, "examples", "fuzz_driver.rs")


def idx(r, c):
    return r * 9 + c


KO_FIGHT = [
    idx(0, 1), idx(1, 1), idx(1, 0), idx(3, 1), idx(1, 2), idx(2, 0),
    idx(8, 8), idx(2, 2), idx(2, 1),          # B captures -> ko at (1,1)
    ("try", idx(1, 1)),                        # W immediate recapture: ILLEGAL
    idx(8, 7), idx(7, 8), idx(1, 1),          # W recaptures later -> ko at (2,1)
    ("try", idx(2, 1)),                        # B immediate recapture: ILLEGAL
    "pass", "pass",
]
SNAPBACK = [
    idx(8, 8), idx(1, 1), idx(0, 1), idx(1, 3), idx(0, 3), idx(3, 2),
    idx(2, 2), idx(2, 1), idx(8, 7), idx(2, 3), idx(8, 6), idx(0, 2),
    idx(1, 2),   # B captures W(0,2): 1 stone, 2-stone group -> NO ko
    idx(0, 2),   # W recaptures: takes B(1,2)+B(2,2), LEGAL (not ko)
    "pass", "pass",
]
SEKI_NEUTRAL = [
    idx(0, 0), idx(0, 3), idx(0, 1), idx(0, 2), idx(1, 0), idx(1, 3),
    idx(2, 0), idx(2, 3), idx(2, 1), idx(2, 2),
    "pass", "pass",
]
SUICIDE = [
    idx(8, 8), idx(1, 1), idx(8, 7), idx(1, 2), idx(8, 6), idx(1, 3),
    idx(8, 5), idx(2, 1), idx(8, 4), idx(2, 3), idx(8, 3), idx(3, 1),
    idx(8, 2), idx(3, 2), idx(8, 1), idx(3, 3),
    ("try", idx(2, 2)),   # B suicide into white ring: ILLEGAL
    idx(0, 0), idx(0, 8),
    "pass", "pass",
]
FIXTURES = {
    "ko_fight": KO_FIGHT,
    "snapback": SNAPBACK,
    "seki_neutral": SEKI_NEUTRAL,
    "suicide": SUICIDE,
}


def py_stones(board):
    return "".join(str(board.grid[r][c]) for r in range(9) for c in range(9))


def py_ko(board):
    return "-" if board.ko is None else str(board.ko[0] * 9 + board.ko[1])


def check_two_way(moves):
    """Play `moves` on gotrain Board and RefBoard; assert full agreement."""
    pyb, refb = Board(9), RefBoard()
    color = BLACK
    ref_caps = [0, 0]
    for mv in moves:
        if isinstance(mv, tuple):  # ('try', i): must be illegal on both
            _, i = mv
            assert not pyb.is_legal(i // 9, i % 9, color), f"py allows {i}"
            assert not refb.legal_moves(color)[i], f"ref allows {i}"
            continue  # illegal attempt: turn does not pass
        is_pass = mv == "pass"
        foe = WHITE if color == BLACK else BLACK
        foe_before = sum(
            1 for r in range(9) for c in range(9) if pyb.grid[r][c] == foe)
        if is_pass:
            assert pyb.play(None, color)
            assert refb.play(None, color) == 0
            got = 0
        else:
            assert pyb.play((mv // 9, mv % 9), color), f"py rejects {mv}"
            got = refb.play(mv, color)
        foe_after = sum(
            1 for r in range(9) for c in range(9) if pyb.grid[r][c] == foe)
        assert foe_before - foe_after == got, "capture count diverges"
        mi = 0 if color == BLACK else 1
        assert refb.caps[color] - ref_caps[mi] == got
        ref_caps[mi] = refb.caps[color]
        assert py_stones(pyb) == "".join(str(v) for v in refb.b), "stones"
        rk = "-" if refb.ko is None else str(refb.ko)
        assert py_ko(pyb) == rk, f"ko: py {py_ko(pyb)} ref {rk}"
        nxt = foe
        for i in range(81):
            pl = pyb.is_legal(i // 9, i % 9, nxt)
            rl = refb.legal_moves(nxt)[i]
            assert pl == rl, f"legal mask diverges at {i}"
        # pass (index 81) is always legal on both
        assert refb.legal_moves(nxt)[81]
        color = nxt
    pb, pw = selfplay.score(pyb)
    fb, fw = refb.score()
    assert abs(pb - fb) < 1e-9 and abs(pw - fw) < 1e-9, \
        f"score: py ({pb},{pw}) ref ({fb},{fw})"


class TestRulesFuzz(unittest.TestCase):
    def test_tricky_positions(self):
        for name, moves in FIXTURES.items():
            with self.subTest(name=name):
                check_two_way(moves)

    def test_random_games(self):
        rng = random.Random(20260924)
        for g in range(4):
            b, color = Board(9), BLACK
            moves, consec, plies = [], 0, 0
            while consec < 2 and plies < 500:
                legal = [i for i in range(81) if b.is_legal(i // 9, i % 9, color)]
                if not legal or plies > 120 or rng.random() < 0.05:
                    moves.append("pass")
                    consec += 1
                    b.play(None, color)
                else:
                    i = rng.choice(legal)
                    moves.append(i)
                    consec = 0
                    assert b.play((i // 9, i % 9), color)
                color = WHITE if color == BLACK else BLACK
                plies += 1
            with self.subTest(game=g):
                check_two_way(moves)

    def test_rust_engine_agrees(self):
        """Three-way check against weiqi/engine via the fuzz_driver example.
        Skipped when cargo is unavailable."""
        cargo = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
        if cargo is None or not os.path.exists(cargo):
            self.skipTest("cargo not available")
        build = subprocess.run(
            [cargo, "build", "--example", "fuzz_driver"],
            cwd=ENGINE_DIR, capture_output=True, text=True, timeout=300)
        self.assertEqual(build.returncode, 0, build.stderr[-2000:])
        driver = os.path.join(
            ENGINE_DIR, "target", "debug", "examples", "fuzz_driver")
        lines = []
        for moves in FIXTURES.values():
            lines.append("new")
            for mv in moves:
                if isinstance(mv, tuple):
                    lines.append(f"play {mv[1]}")
                elif mv == "pass":
                    lines.append("play pass")
                else:
                    lines.append(f"play {mv}")
            lines.append("score")
        proc = subprocess.run(
            [driver], input="\n".join(lines) + "\n",
            capture_output=True, text=True, timeout=300)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        out = [l for l in proc.stdout.splitlines() if l.strip()]
        p = 0
        for name, moves in FIXTURES.items():
            self.assertEqual(out[p], "game", name)
            p += 1
            pyb = Board(9)
            color = BLACK
            for mv in moves:
                line = out[p]
                p += 1
                if isinstance(mv, tuple):
                    self.assertTrue(line.startswith("err"),
                                    f"{name}: rust accepted illegal {mv[1]}")
                    continue
                is_pass = mv == "pass"
                if is_pass:
                    self.assertTrue(pyb.play(None, color))
                else:
                    self.assertTrue(pyb.play((mv // 9, mv % 9), color))
                self.assertTrue(line.startswith("ok"),
                                f"{name} move {mv}: {line}")
                d = dict(kv.split("=", 1) for kv in line[3:].split())
                self.assertEqual(d["stones"], py_stones(pyb), f"{name} {mv}")
                self.assertEqual(d["ko"], py_ko(pyb), f"{name} {mv}")
                color = WHITE if color == BLACK else BLACK
            self.assertTrue(out[p].startswith("score"), name)
            _, rb, rw = out[p].split()
            p += 1
            pb, pw = selfplay.score(pyb)
            self.assertAlmostEqual(float(rb), pb, msg=name)
            self.assertAlmostEqual(float(rw), pw, msg=name)


if __name__ == "__main__":
    unittest.main()
