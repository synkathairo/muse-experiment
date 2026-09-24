"""Tests for the self-play RL stack: env legality/scoring, PPO smoke.

Run from weiqi/train/:  python -m unittest tests.test_selfplay -v
"""

import os
import tempfile
import unittest

import numpy as np
import torch

from gotrain import features, selfplay
from gotrain.net import GoNet, EXPECTED_PARAMS
from gotrain.ppo import PPOConfig, compute_gae, ppo_update
from gotrain.rules import BLACK, WHITE, EMPTY, Board, opponent
from gotrain.selfplay import SelfPlayGo, legal_mask, observe, score, winner
from gotrain.train_selfplay import save_ckpt, load_ckpt, random_opponent
from gotrain.export import export_weights


def reference_score(grid):
    """Independent Tromp-Taylor reference scorer (union-find over empty points).

    Deliberately a different algorithm from gotrain.selfplay.score (which
    flood-fills with a seen-grid): empty points are merged with a disjoint-set
    union, then each component's bordering colors decide whose territory it is.
    Fuzz-compared against score() below; any divergence is a scoring bug.
    """
    n = len(grid)
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    empties = [(r, c) for r in range(n) for c in range(n) if grid[r][c] == EMPTY]
    for p in empties:
        parent[p] = p
    for r, c in empties:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            q = (r + dr, c + dc)
            if q in parent:
                rq, rp = find(q), find((r, c))
                if rq != rp:
                    parent[rq] = rp

    black_stones = white_stones = 0
    for r in range(n):
        for c in range(n):
            if grid[r][c] == BLACK:
                black_stones += 1
            elif grid[r][c] == WHITE:
                white_stones += 1

    comp_borders = {}
    comp_size = {}
    for r, c in empties:
        root = find((r, c))
        comp_size[root] = comp_size.get(root, 0) + 1
        borders = comp_borders.setdefault(root, set())
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n:
                v = grid[nr][nc]
                if v == BLACK or v == WHITE:
                    borders.add(v)

    black_terr = sum(s for root, s in comp_size.items()
                     if comp_borders[root] == {BLACK})
    white_terr = sum(s for root, s in comp_size.items()
                     if comp_borders[root] == {WHITE})
    return (black_stones + black_terr,
            white_stones + white_terr + selfplay.KOMI)


def random_masked_player(obs, masks):
    return random_opponent(obs, masks)


class TestScoring(unittest.TestCase):
    def test_empty_board_white_wins_on_komi(self):
        b = Board(9)
        bs, ws = score(b)
        self.assertEqual(bs, 0)
        self.assertEqual(ws, 7.5)
        self.assertEqual(winner(b), WHITE)

    def test_surrounded_point_is_territory(self):
        b = Board(9)
        for r in range(9):
            for c in range(9):
                if (r, c) != (0, 0):
                    b.grid[r][c] = BLACK
        bs, ws = score(b)
        self.assertEqual(bs, 80 + 1)  # 80 stones + 1 surrounded point
        self.assertEqual(ws, 7.5)
        self.assertEqual(winner(b), BLACK)

    def test_neutral_point_touches_both(self):
        b = Board(9)
        b.grid[0][0] = BLACK
        b.grid[0][2] = WHITE
        # (0,1) touches both colors -> dame, counts for nobody
        bs, ws = score(b)
        self.assertEqual(bs, 1)
        self.assertEqual(ws, 1 + 7.5)
        self.assertEqual(winner(b), WHITE)

    def test_two_pass_game_empty_board(self):
        # learner Black passes, opponent (always-pass) passes -> White wins
        env = SelfPlayGo(num_envs=1, seed=0,
                         opponent_fn=lambda o, m: np.array([81]))
        env.reset()
        obs, rewards, dones, terms, lens = env.step(np.array([81]))
        self.assertTrue(dones[0])
        self.assertTrue(terms[0])          # true terminal, not truncation
        self.assertEqual(rewards[0], -1.0)  # Black learner loses to komi
        self.assertEqual(lens[0], 2)

    def test_score_matches_independent_reference(self):
        # fuzz selfplay.score against the independent union-find
        # reimplementation above; any divergence is a scoring bug.
        rng = np.random.default_rng(20260924)
        for trial in range(80):
            b = Board(9)
            if trial % 2 == 0:
                # realistic: random legal play
                color = BLACK
                for _ in range(rng.integers(0, 50)):
                    moves = [(r, c) for r in range(9) for c in range(9)
                             if b.grid[r][c] == EMPTY and b.is_legal(r, c, color)]
                    if not moves:
                        break
                    r, c = moves[rng.integers(len(moves))]
                    self.assertTrue(b.play((r, c), color))
                    color = opponent(color)
            else:
                # adversarial: arbitrary stone soup (scoring needs no legality)
                for r in range(9):
                    for c in range(9):
                        v = rng.random()
                        b.grid[r][c] = (BLACK if v < 0.35
                                        else (WHITE if v < 0.7 else EMPTY))
            bs, ws = score(b)
            rbs, rws = reference_score(b.grid)
            self.assertEqual(bs, rbs, f"black score mismatch, trial {trial}")
            self.assertEqual(ws, rws, f"white score mismatch, trial {trial}")


class TestEnvLegality(unittest.TestCase):
    def test_random_games_all_legal_and_terminate(self):
        # Any illegal move raises AssertionError inside env.step; mask
        # computation bugs would surface here across many positions.
        env = SelfPlayGo(num_envs=8, seed=1, opponent_fn=random_masked_player)
        obs = env.reset()
        rng = np.random.default_rng(0)
        total_steps = 0
        finished = 0
        while finished < 8 and total_steps < 4000:
            masks = env.legal_masks_learner()
            # pass is always legal -> mask never empty
            self.assertTrue(masks[:, 81].all())
            actions = np.array([rng.choice(np.flatnonzero(m)) for m in masks])
            obs, rewards, dones, terms, lens = env.step(actions)
            self.assertTrue(set(np.unique(rewards)).issubset({-1.0, 0.0, 1.0}))
            for i in np.where(dones)[0]:
                finished += 1
                self.assertGreater(lens[i], 0)
                self.assertLessEqual(lens[i], 243)
            if np.any(dones):
                obs[np.where(dones)[0]] = env.reset(np.where(dones)[0])
            total_steps += 1
        self.assertEqual(finished, 8)

    def test_learner_alternates_color(self):
        env = SelfPlayGo(num_envs=2, seed=0, opponent_fn=random_masked_player)
        env.reset()
        first = env.learner_color.copy()
        env.reset()
        second = env.learner_color.copy()
        self.assertTrue((first == BLACK).all())
        self.assertTrue((second == WHITE).all())

    def test_planes_match_features_encode(self):
        # env observations must equal features.encode called explicitly
        env = SelfPlayGo(num_envs=2, seed=3, opponent_fn=random_masked_player)
        obs = env.reset()
        rng = np.random.default_rng(2)
        for _ in range(5):
            masks = env.legal_masks_learner()
            actions = np.array([rng.choice(np.flatnonzero(m)) for m in masks])
            obs, _, dones, _, _ = env.step(actions)
            if np.any(dones):
                obs[np.where(dones)[0]] = env.reset(np.where(dones)[0])
        for i in range(2):
            if env.done[i]:
                continue
            board = env.boards[i]
            color = int(env.learner_color[i])
            expected = features.encode(
                board.stones(color), board.stones(opponent(color)),
                last_move=board.last_move,
                black_to_move=(color == BLACK), ko_point=board.ko)
            np.testing.assert_array_equal(observe(board, color), expected)
            np.testing.assert_array_equal(
                env._observe_learner(np.array([i]))[0], expected)


class TestPPO(unittest.TestCase):
    def test_gae_shapes_and_terminal_bootstrap(self):
        T, N = 8, 4
        rewards = torch.zeros(T, N)
        rewards[-1] = 1.0
        values = torch.zeros(T, N)
        terms = torch.zeros(T, N, dtype=torch.bool)
        terms[-1] = True
        truncs = torch.zeros(T, N, dtype=torch.bool)
        adv, ret = compute_gae(rewards, values, terms, truncs,
                               torch.zeros(N), torch.zeros(N, dtype=torch.bool))
        self.assertEqual(adv.shape, (T, N))
        self.assertEqual(ret.shape, (T, N))
        # true terminal at the end: no bootstrapping past it
        self.assertTrue(torch.isfinite(adv).all())

    def test_ppo_update_finite(self):
        torch.manual_seed(0)
        model = GoNet()
        opt = torch.optim.Adam(model.parameters(), lr=2.5e-4)
        cfg = PPOConfig(update_epochs=2, minibatch_size=16)
        B = 32
        obs = torch.randn(B, 6, 9, 9)
        with torch.no_grad():
            logits, values = model(obs)
        # realistic masks: random ~half the moves illegal (pass always legal).
        # (An all-legal mask once hid a 0*-inf=nan entropy bug — never again.)
        masks = torch.rand(B, 82) > 0.5
        masks[:, 81] = True
        actions = logits.masked_fill(~masks, float("-inf")).argmax(-1)
        logps = torch.log_softmax(
            logits.masked_fill(~masks, float("-inf")), -1
        ).gather(1, actions.unsqueeze(1)).squeeze(1)
        adv = torch.randn(B)
        ret = torch.randn(B)
        stats = ppo_update(model, opt, cfg, obs, actions, logps, adv, ret,
                           values.detach().view(-1), masks)
        for k, v in stats.items():
            if k == "early_stop":
                continue
            self.assertTrue(np.isfinite(v), f"{k} not finite: {v}")

    def test_checkpoint_roundtrip_and_export_size(self):
        model = GoNet()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        snap = GoNet()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "latest.pt")
            save_ckpt(p, model, opt, snap, step=12345, ppo_iter=7,
                      snap_ptr=2, hparams={"lr": 1e-3})
            model2, opt2, snap2 = GoNet(), torch.optim.Adam(GoNet().parameters()), GoNet()
            ck = load_ckpt(p, model2, opt2, snap2, torch.device("cpu"))
            self.assertEqual(ck["step"], 12345)
            self.assertEqual(ck["ppo_iter"], 7)
            for a, b in zip(model.state_dict().values(), model2.state_dict().values()):
                self.assertTrue(torch.equal(a, b))
            out = os.path.join(d, "w.bin")
            export_weights(model2, out)
            # locked export size: 130522 params x fp16
            self.assertEqual(os.path.getsize(out), EXPECTED_PARAMS * 2)

    def test_evaluate_finishes_and_scores(self):
        # regression: evaluate() once crashed resetting finished envs
        # (env.reset clears env.done before the boolean-mask assignment)
        from gotrain.train_selfplay import evaluate
        torch.manual_seed(0)
        wr = evaluate(GoNet(), random_opponent, n_games=6,
                      device=torch.device("cpu"), seed=0)
        self.assertGreaterEqual(wr, 0.0)
        self.assertLessEqual(wr, 1.0)

    def test_masked_sampling_never_illegal(self):
        # policy heavily favoring an illegal move must still sample legally
        from gotrain.train_selfplay import sample_actions
        torch.manual_seed(0)
        model = GoNet()
        with torch.no_grad():
            model.pol_fc.bias[:] = 0.0
            model.pol_fc.bias[0] = 100.0  # desperately wants move 0
        masks = torch.ones(4, 82, dtype=torch.bool)
        masks[:, 0] = False  # ...which is illegal here
        obs = torch.zeros(4, 6, 9, 9)
        for _ in range(20):
            actions, _, _ = sample_actions(model, obs, masks)
            self.assertFalse((actions == 0).any())


    def test_gae_does_not_bootstrap_across_terminal(self):
        # regression: compute_gae masked with terms[t+1] instead of terms[t],
        # leaking the next episode's value into a finished episode's advantage.
        T, N = 3, 1
        rewards = torch.tensor([[0.0], [1.0], [0.0]])
        values = torch.tensor([[0.5], [0.5], [0.5]])
        terms = torch.tensor([[False], [True], [False]])
        truncs = torch.zeros(T, N, dtype=torch.bool)
        adv, _ = compute_gae(rewards, values, terms, truncs,
                             next_value=torch.zeros(N),
                             next_term=torch.zeros(N, dtype=torch.bool),
                             gamma=1.0, gae_lambda=1.0)
        # t=1 ended the episode: adv[1] = 1 - 0.5 = 0.5 (no bootstrap);
        # t=0 bootstraps V[1] normally: adv[0] = (0 + 0.5 - 0.5) + 0.5 = 0.5.
        # The buggy version gave adv[0] = -0.5 (masked by terms[1]).
        self.assertAlmostEqual(adv[0, 0].item(), 0.5, places=5)
        self.assertAlmostEqual(adv[1, 0].item(), 0.5, places=5)
        self.assertAlmostEqual(adv[2, 0].item(), -0.5, places=5)

    def test_gae_truncation_is_episodic_terminal(self):
        # scored max-ply endings are episodic terminals for GAE: the reward is
        # kept, but no value bootstraps past them (only the final rollout obs
        # is preserved, so a within-rollout truncation would otherwise leak
        # the NEXT episode's value into this episode's advantage).
        T, N = 3, 1
        rewards = torch.tensor([[0.0], [1.0], [0.0]])
        values = torch.tensor([[0.5], [0.5], [0.5]])
        terms = torch.zeros(T, N, dtype=torch.bool)
        truncs = torch.tensor([[False], [True], [False]])
        adv, _ = compute_gae(rewards, values, terms, truncs,
                             next_value=torch.zeros(N),
                             next_term=torch.zeros(N, dtype=torch.bool),
                             gamma=1.0, gae_lambda=1.0)
        # t=1 truncated: adv[1] = 1 - 0.5 = 0.5, no bootstrap;
        # t=0 bootstraps V[1] normally: adv[0] = (0 + 0.5 - 0.5) + 0.5 = 0.5.
        # The old code gave adv[1] = 1 + 0.5 - 0.5 = 1.0 (bootstrapped V[2]).
        self.assertAlmostEqual(adv[0, 0].item(), 0.5, places=5)
        self.assertAlmostEqual(adv[1, 0].item(), 0.5, places=5)
        self.assertAlmostEqual(adv[2, 0].item(), -0.5, places=5)

    def test_gae_final_truncation_does_not_bootstrap(self):
        # truncation on the last rollout step: next_value must not leak in.
        T, N = 2, 1
        rewards = torch.tensor([[0.0], [1.0]])
        values = torch.tensor([[0.5], [0.5]])
        terms = torch.zeros(T, N, dtype=torch.bool)
        truncs = torch.zeros(T, N, dtype=torch.bool)
        adv, _ = compute_gae(rewards, values, terms, truncs,
                             next_value=torch.tensor([999.0]),
                             next_term=torch.zeros(N, dtype=torch.bool),
                             next_trunc=torch.ones(N, dtype=torch.bool),
                             gamma=1.0, gae_lambda=1.0)
        self.assertAlmostEqual(adv[1, 0].item(), 0.5, places=5)

    def test_resume_restores_hparams_not_cli_defaults(self):
        # regression (2026-09-24): resuming without --total-steps kept the 2M
        # default while the run was past it, making the annealed LR negative
        # (gradient ascent) and destroying the policy in a single iteration.
        import argparse
        from gotrain.train_selfplay import apply_resumed_hparams
        args = argparse.Namespace(out="new_dir", device="cpu", total_steps=2000000,
                                  lr=2.5e-4, seed=7, num_envs=32,
                                  rollout_steps=128)
        ck = {"hparams": {"out": "old_dir", "device": "cuda",
                          "total_steps": 3000000, "lr": 1e-4, "seed": 7,
                          "num_envs": 32, "rollout_steps": 128}}
        apply_resumed_hparams(args, ck)
        self.assertEqual(args.total_steps, 3000000)  # checkpoint wins
        self.assertEqual(args.lr, 1e-4)
        self.assertEqual(args.out, "new_dir")    # run dir stays the resumer's
        self.assertEqual(args.device, "cpu")     # launch env stays the resumer's
        # and the annealed LR at the resumed iteration is positive
        total_iters = max(1, (args.total_steps + 4095) // 4096)
        lr_now = args.lr * max(0.0, 1.0 - 330 / total_iters)
        self.assertGreater(lr_now, 0.0)

    def test_resume_explicit_cli_overrides_checkpoint(self):
        # 2026-09-24: user resumed with --total-steps 3000000; it was silently
        # reverted to the checkpoint's 200000 and the run exited immediately
        # ("done at step 200704"). Explicit flags must win over the checkpoint.
        import argparse
        from gotrain.train_selfplay import apply_resumed_hparams
        args = argparse.Namespace(out="runs/smoke", device="auto", total_steps=3000000,
                                  lr=2.5e-4, seed=7, num_envs=32, rollout_steps=128)
        ck = {"hparams": {"out": "runs/smoke", "device": "auto",
                          "total_steps": 200000, "lr": 2.5e-4, "seed": 7,
                          "num_envs": 32, "rollout_steps": 128}}
        apply_resumed_hparams(args, ck, explicit={"total_steps"})
        self.assertEqual(args.total_steps, 3000000)  # explicit CLI wins
        self.assertEqual(args.lr, 2.5e-4)             # rest still from checkpoint
        self.assertEqual(args.seed, 7)


class TestEvaluateTally(unittest.TestCase):
    def test_tally_counts_all_when_room(self):
        from gotrain.train_selfplay import _tally_finished
        wins, played, counted = _tally_finished(
            np.array([0, 2]), np.array([-1.0, 1.0, -1.0]), 1, 0, 5)
        self.assertEqual((wins, played), (1, 2))
        self.assertEqual(list(counted), [0, 2])

    def test_tally_caps_simultaneous_finishes(self):
        # played=1 of n_games=2, then two envs finish on the same step: only
        # the first is counted, so the win rate covers exactly n_games games.
        from gotrain.train_selfplay import _tally_finished
        wins, played, counted = _tally_finished(
            np.array([0, 1]), np.array([1.0, -1.0]), 0, 1, 2)
        self.assertEqual(played, 2)
        self.assertEqual(wins, 1)
        self.assertEqual(list(counted), [0])


if __name__ == "__main__":
    unittest.main()
