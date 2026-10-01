"""Tests for search-distillation support in gotrain.mcts.

Run from weiqi/train/:  python -m pytest tests/test_search_distill.py -q
"""
from __future__ import annotations

import unittest

import numpy as np
import torch

from gotrain import rules, selfplay
from gotrain.features import index_to_move
from gotrain.mcts import PASS, SearchConfig, Searcher
from gotrain.net import GoNet


def _midgame_board() -> rules.Board:
    # Deterministic 30-ply opening: not tactical, just off the empty board.
    moves = [
        (2, 2), (6, 6), (2, 6), (6, 2), (4, 4), (3, 3),
        (5, 5), (2, 4), (6, 4), (4, 2), (4, 6), (3, 5),
        (5, 3), (1, 2), (7, 6), (2, 7), (6, 1), (4, 3),
        (3, 4), (5, 6), (6, 5), (1, 5), (7, 2), (5, 1),
        (1, 7), (7, 4), (3, 1), (5, 7), (0, 4), (8, 4),
    ]
    b = rules.Board()
    for i, (r, c) in enumerate(moves):
        ok = b.play((r, c), b.to_play)
        assert ok, f"opening move {(r, c)} illegal at ply {i}"
    return b


class TestSearchWithVisits(unittest.TestCase):
    def test_visits_sum_to_simulations(self) -> None:
        net = GoNet()
        net.eval()
        b = _midgame_board()
        sims = 50
        s = Searcher(net, torch.device("cpu"),
                     SearchConfig(simulations=sims, batch=8, seed=42))
        move, visits, root_value = s.search_with_visits(b, 0)
        self.assertEqual(visits.shape, (82,))
        # Normalized distribution sums to 1 == raw counts sum to sims.
        self.assertAlmostEqual(float(visits.sum()), 1.0, places=6)
        self.assertGreaterEqual(move, 0)
        self.assertLessEqual(move, 81)
        mask = selfplay.bot_mask(b, b.to_play)
        self.assertEqual(mask[move], 1)
        # No visits leak onto illegal moves.
        self.assertTrue(np.all(visits[mask == 0] == 0.0))
        self.assertGreaterEqual(root_value, -1.0)
        self.assertLessEqual(root_value, 1.0)

    def test_agrees_with_search_on_move(self) -> None:
        net = GoNet()
        net.eval()
        b = _midgame_board()
        cfg = SearchConfig(simulations=40, batch=8, seed=123, temperature=0.0)
        s1 = Searcher(net, torch.device("cpu"), cfg)
        s2 = Searcher(net, torch.device("cpu"), cfg)
        # Same seed + temperature 0 => deterministic; both must pick argmax N.
        self.assertEqual(s1.search(b, 0), s2.search_with_visits(b, 0)[0])

    def test_pass_position_returns_pass_visits(self) -> None:
        net = GoNet()
        net.eval()
        b = rules.Board()
        s = Searcher(net, torch.device("cpu"), SearchConfig(simulations=10))
        move, visits, _ = s.search_with_visits(b, 2)  # two passes: game over
        self.assertEqual(move, PASS)
        self.assertEqual(visits[PASS], 1.0)

    def test_index_roundtrip(self) -> None:
        self.assertEqual(index_to_move(81), None)
        self.assertEqual(index_to_move(0), (0, 0))


if __name__ == "__main__":
    unittest.main()
