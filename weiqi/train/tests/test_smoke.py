"""Smoke tests for the train/ package.

Run from weiqi/train/:  python -m unittest tests.test_smoke -v
"""

import os
import tempfile
import unittest

import numpy as np
import torch

from gotrain import dataset, features, net, rules, sgf
from gotrain.export import export_weights, read_export

# A real 9x9 game, downloaded from OGS (game 73233171, Sadaharu 6d vs
# Xaloc 3d, Tianyuan Nines Title Tournament 2023, W+R, Chinese rules).
# Genuine downloaded record, kept verbatim (OGS nested-paren format).
REAL_GAME_SGF = '(;FF[4]\nCA[UTF-8]\nGM[1]\nDT[2025-03-12]\nPC[OGS: https://online-go.com/game/73233171]\nGN[Tournament Game: Tianyuan Nines Title Tournament 2023 (112661) R:3 (Xaloc vs Sadaharu)]\nPB[Sadaharu]\nPW[Xaloc]\nBR[6d]\nWR[3d]\nTM[604800]OT[86400 fischer]\nRE[W+R]\nSZ[9]\nKM[7.5]\nRU[Chinese]\nC[Xaloc: Hi, have a good game\n]\n;B[ee]\nC[Xaloc: Hi, have a good game\n]\n(;W[gd]\n(;B[fc]\n(;W[gc]\n(;B[gf]\n(;W[fb]\n(;B[eb]\n(;W[ec]\n(;B[dc]\n(;W[fd]\n(;B[ed]\n(;W[cg]\n(;B[eg]\n(;W[cd]\n(;B[fc]\n(;W[ff]\n(;B[fe]\n(;W[hf]\n(;B[ge]\n(;W[he]\n(;B[hg]\n(;W[ec]\n(;B[bd]\n(;W[be]\n(;B[bc]\n(;W[ce]\n(;B[fc]\n(;W[gg]\n(;B[gh]\n(;W[ec]\n(;B[ch]\n(;W[bh]\n(;B[fc]\n(;W[ig]\n(;B[hh]\n(;W[ec]\n(;B[dg]\n(;W[dh]\n(;B[fc]\n(;W[gb]\n(;B[cf]\n(;W[bg]\n(;B[eh]\n(;W[ci]\n(;B[ec]\n(;W[cc]\n(;B[cb]\n(;W[bf]\n(;B[ih]\n(;W[id]\n(;B[if]\n(;W[ib]\nC[Xaloc: Thank you for the game\n]\n))))))))))))))))))))))))))))))))))))))))))))))))))))'


class TestNet(unittest.TestCase):
    def test_param_count_locked(self):
        self.assertEqual(net.GoNet().param_count(), net.EXPECTED_PARAMS)
        self.assertEqual(net.EXPECTED_PARAMS, 130522)

    def test_forward_shapes(self):
        m = net.GoNet()
        m.eval()
        with torch.no_grad():
            logits, value = m(torch.randn(3, 6, 9, 9))
        self.assertEqual(tuple(logits.shape), (3, 82))
        self.assertEqual(tuple(value.shape), (3,))
        self.assertTrue(torch.all(value >= -1.0) and torch.all(value <= 1.0))

    def test_export_order_covers_all_params(self):
        m = net.GoNet()
        total = sum(int(np.prod(shape)) for _, shape in net.EXPORT_ORDER)
        self.assertEqual(total, m.param_count())


class TestFeatures(unittest.TestCase):
    def test_planes(self):
        own = np.zeros((9, 9), dtype=bool)
        opp = np.zeros((9, 9), dtype=bool)
        own[0, 0] = True   # top-left black
        opp[8, 8] = True   # bottom-right white
        p = features.encode(own, opp, last_move=(0, 0),
                            black_to_move=False, ko_point=(4, 4))
        self.assertEqual(p.shape, (6, 9, 9))
        self.assertEqual(p.dtype, np.float32)
        self.assertEqual(p[0].sum(), 1)          # own stones (white to move)
        self.assertEqual(p[1].sum(), 1)          # opp stones
        self.assertEqual(p[2].sum(), 81 - 2)    # empty
        self.assertEqual(p[3, 0, 0], 1.0)
        self.assertEqual(p[3].sum(), 1)
        self.assertEqual(p[4].sum(), 0)          # white to move
        self.assertEqual(p[5, 4, 4], 1.0)
        self.assertEqual(p[5].sum(), 1)

    def test_move_index_roundtrip(self):
        self.assertEqual(features.move_to_index((0, 0)), 0)
        self.assertEqual(features.move_to_index((8, 8)), 80)
        self.assertEqual(features.move_to_index(None), 81)
        self.assertEqual(features.index_to_move(81), None)
        self.assertEqual(features.index_to_move(40), (4, 4))


class TestRules(unittest.TestCase):
    def test_capture(self):
        b = rules.Board(9)
        # white stone at (1,1) surrounded on 3 sides, black plays the last liberty
        b.play((0, 1), rules.BLACK)
        b.play((1, 1), rules.WHITE)
        b.play((1, 0), rules.BLACK)
        b.play((8, 8), rules.WHITE)  # elsewhere
        b.play((1, 2), rules.BLACK)
        b.play((8, 7), rules.WHITE)
        self.assertTrue(b.play((2, 1), rules.BLACK))
        self.assertEqual(b.grid[1][1], rules.EMPTY)

    def test_suicide_rejected(self):
        b = rules.Board(9)
        b.play((0, 1), rules.WHITE)
        b.play((1, 0), rules.WHITE)
        self.assertFalse(b.play((0, 0), rules.BLACK))

    def test_simple_ko(self):
        # . B W .
        # B W . W
        # . B W .
        # Black captures at (1,2); white's instant recapture at (1,1) is ko-banned.
        b = rules.Board(9)
        B, W = rules.BLACK, rules.WHITE
        for (r, c), col in {(0, 1): B, (1, 0): B, (2, 1): B,
                            (1, 1): W, (0, 2): W, (2, 2): W, (1, 3): W}.items():
            b.grid[r][c] = col
        b.to_play = B
        self.assertTrue(b.play((1, 2), B))
        self.assertEqual(b.grid[1][1], rules.EMPTY)  # white stone captured
        self.assertEqual(b.ko, (1, 1))
        self.assertFalse(b.is_legal(1, 1, W))        # ko recapture banned
        # after white plays elsewhere, ko lifts and white may recapture
        self.assertTrue(b.play((8, 8), W))
        self.assertIsNone(b.ko)
        self.assertTrue(b.play((1, 1), W))
        self.assertEqual(b.grid[1][2], rules.EMPTY)  # black stone recaptured


class TestSGF(unittest.TestCase):
    @unittest.skipIf(REAL_GAME_SGF is None, "no downloaded game embedded yet")
    def test_real_game(self):
        g = sgf.parse_sgf(REAL_GAME_SGF)
        self.assertIsNotNone(g)
        self.assertEqual(g.size, 9)
        self.assertGreater(len(g.moves), 10)
        # first coord 'aa' == (0,0) top-left; colors alternate
        colors = [c for c, _ in g.moves]
        self.assertTrue(all(c in ("B", "W") for c in colors))

    def test_variation_ignored(self):
        g = sgf.parse_sgf("(;GM[1]FF[4]SZ[9];B[aa](;W[bb])(;W[cc]))")
        self.assertEqual(g.moves, [("B", (0, 0))])

    def test_pass(self):
        g = sgf.parse_sgf("(;GM[1]FF[4]SZ[9];B[aa];W[])")
        self.assertEqual(g.moves, [("B", (0, 0)), ("W", None)])

    def test_ogs_nested_main_line(self):
        # OGS threads the main line through nested single-node parens.
        g = sgf.parse_sgf("(;GM[1]FF[4]SZ[9];B[ee](;W[gd](;B[fc](;W[gc]))))")
        self.assertEqual(g.moves, [("B", (4, 4)), ("W", (3, 6)),
                                   ("B", (2, 5)), ("W", (2, 6))])

    def test_ogs_quoted_values(self):
        g = sgf.parse_sgf("(;GM[1]FF[4]SZ[9]RE['B+R']PB['foo'];B[aa])")
        self.assertEqual(g.result, "B+R")
        self.assertEqual(g.black_name, "foo")

    def test_timeout_ogs_notation(self):
        self.assertTrue(sgf.is_timeout("W+T"))
        self.assertTrue(sgf.is_timeout("B+Time"))
        self.assertFalse(sgf.is_timeout("B+2.5"))

    def test_handicap_white_to_move(self):
        g = sgf.parse_sgf("(;GM[1]FF[4]SZ[9]HA[2]AB[cc][gg]RE[W+R];W[ee];B[aa])")
        self.assertEqual(g.to_play, "W")
        pairs = dataset.game_to_pairs(g)
        # white's move replays; the pair is from white's perspective, and
        # white won, so z=+1
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0][2], 1.0)


class TestExport(unittest.TestCase):
    def test_roundtrip_byte_exact(self):
        torch.manual_seed(0)
        m = net.GoNet()
        m.eval()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "w.bin")
            export_weights(m, p)
            with open(p, "rb") as f:
                blob = f.read()
            expected = b"".join(
                t.to(torch.float16).numpy().astype("<f2").tobytes()
                for t in net.ordered_tensors(m)
            )
            self.assertEqual(blob, expected)
            # size check: 130522 params * 2 bytes
            self.assertEqual(len(blob), 130522 * 2)
            back = read_export(p)
            for key, shape in net.EXPORT_ORDER:
                self.assertEqual(back[key].shape, shape)
            orig = m.pol_fc.weight.detach().numpy()
            np.testing.assert_allclose(back["pol_fc.weight"], orig, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
