"""Tests for the KataGo-style auxiliary heads (ownership + score margin).

Run from weiqi/train/:  python -m unittest tests.test_aux -v
"""

import unittest

import numpy as np
import torch

from gotrain.net import GoNet, N_POINTS, EXPECTED_PARAMS
from gotrain.net_aux import (GoNetAux, AUX_KEYS, check_trunk_resume,
                             load_trunk_from_gonet, to_gonet_state_dict)
from gotrain.ppo import PPOConfig
from gotrain.rules import BLACK, WHITE, EMPTY, Board
from gotrain.selfplay import (SelfPlayGo, margin_label, ownership_labels,
                              score, to_learner_perspective)
from gotrain.train_selfplay import (attach_finished_game_labels, aux_update,
                                    random_opponent)


def play_moves(moves):
    """Board with the given [(color, r, c), ...] stones played in order."""
    b = Board(9)
    for color, r, c in moves:
        assert b.play((r, c), color), f"illegal test move {(r, c)}"
    return b


class TestOwnershipLabels(unittest.TestCase):
    def test_empty_board(self):
        own = ownership_labels(Board(9))
        self.assertEqual(own.shape, (81,))
        self.assertTrue((own == 0).all())
        # margin = 0 - (0 + 7.5)
        self.assertAlmostEqual(margin_label(Board(9)), -7.5)

    def test_single_stone_owns_everything(self):
        b = play_moves([(BLACK, 0, 0)])
        own = ownership_labels(b)
        # the one empty region borders only black -> all black territory
        self.assertTrue((own == 1).all())
        self.assertAlmostEqual(margin_label(b), 81 - 7.5)

    def test_two_stones_neutral_region(self):
        b = play_moves([(BLACK, 0, 0), (WHITE, 8, 8)])
        own = ownership_labels(b)
        self.assertEqual(own[0], 1)      # black stone
        self.assertEqual(own[80], -1)    # white stone
        # the shared empty region touches both colors -> neutral
        self.assertEqual(own[40], 0)
        self.assertTrue((np.abs(own[1:80]) <= 1).all())
        self.assertAlmostEqual(margin_label(b), 1 - (1 + 7.5))

    def test_enclosed_corner_territory(self):
        # black wall encloses the 2x2 corner (0..1, 0..1)
        wall = [(BLACK, 2, 0), (BLACK, 2, 1), (BLACK, 2, 2),
                (BLACK, 0, 2), (BLACK, 1, 2)]
        b = play_moves(wall + [(WHITE, 5, 5)])
        own = ownership_labels(b).reshape(9, 9)
        for r, c in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            self.assertEqual(own[r, c], 1, f"corner {(r, c)} not black terr")
        self.assertEqual(own[5, 5], -1)
        # the big region touches both colors -> neutral
        self.assertEqual(own[4, 4], 0)
        # margin: black 5 stones + 4 terr; white 1 stone; komi 7.5
        self.assertAlmostEqual(margin_label(b), 9 - (1 + 7.5))

    def test_margin_consistent_with_score(self):
        rng = np.random.default_rng(0)
        for _ in range(5):
            moves = []
            cells = [(r, c) for r in range(9) for c in range(9)]
            rng.shuffle(cells)
            for k, (r, c) in enumerate(cells[:20]):
                moves.append((BLACK if k % 2 == 0 else WHITE, r, c))
            b = Board(9)
            ok_moves = [m for m in moves if b.play((m[1], m[2]), m[0])]
            self.assertTrue(ok_moves)
            sc_b, sc_w = score(b)
            self.assertAlmostEqual(margin_label(b), sc_b - sc_w)


class TestPerspective(unittest.TestCase):
    def test_black_passthrough_white_flip(self):
        own = np.array([1.0, -1.0, 0.0])
        o, m = to_learner_perspective(own, 3.5, BLACK)
        self.assertTrue(np.array_equal(o, own) and m == 3.5)
        o, m = to_learner_perspective(own, 3.5, WHITE)
        self.assertTrue(np.array_equal(o, -own) and m == -3.5)


class TestGoNetAux(unittest.TestCase):
    def test_forward_shapes(self):
        torch.manual_seed(0)
        aux = GoNetAux()
        x = torch.randn(2, 6, 9, 9)
        logits, value = aux(x)                      # drop-in for GoNet
        self.assertEqual(logits.shape, (2, 82))
        self.assertEqual(value.shape, (2,))
        _, _, own, margin = aux.forward_aux(x)
        self.assertEqual(own.shape, (2, N_POINTS))
        self.assertEqual(margin.shape, (2,))
        self.assertTrue(bool(((own >= -1) & (own <= 1)).all()))  # tanh

    def test_policy_value_match_locked_trunk(self):
        """GoNetAux(x)[:2] == GoNet-with-same-trunk-weights(x): the wrapper
        must not change what the locked net computes."""
        torch.manual_seed(1)
        aux = GoNetAux()
        g = GoNet()
        g.load_state_dict(aux.trunk.state_dict())
        x = torch.randn(3, 6, 9, 9)
        la, va = aux(x)
        lg, vg = g(x)
        self.assertTrue(torch.allclose(la, lg, atol=1e-6))
        self.assertTrue(torch.allclose(va, vg, atol=1e-6))

    def test_trunk_param_count_unchanged(self):
        self.assertEqual(GoNetAux().trunk.param_count(), EXPECTED_PARAMS)

    def test_trunk_resume_from_gonet(self):
        torch.manual_seed(2)
        g = GoNet()
        aux = GoNetAux()
        res = load_trunk_from_gonet(aux, g.state_dict())
        check_trunk_resume(res.missing_keys)   # raises unless only aux keys
        self.assertFalse(res.unexpected_keys)
        self.assertEqual(set(res.missing_keys), AUX_KEYS)
        for k, v in g.state_dict().items():
            self.assertTrue(torch.equal(aux.state_dict()["trunk." + k], v))

    def test_gonet_loader_accepts_aux_checkpoint(self):
        torch.manual_seed(3)
        aux = GoNetAux()
        g = GoNet()
        g.load_state_dict(to_gonet_state_dict(aux.state_dict()))  # strict
        for k, v in g.state_dict().items():
            self.assertTrue(torch.equal(aux.trunk.state_dict()[k], v))
        # plain GoNet dicts pass through untouched
        sd = g.state_dict()
        self.assertIs(to_gonet_state_dict(sd), sd)


class TestAuxUpdate(unittest.TestCase):
    def test_loss_decreases(self):
        torch.manual_seed(4)
        aux = GoNetAux()
        opt = torch.optim.Adam(aux.parameters(), lr=3e-3)
        cfg = PPOConfig()
        B = 16
        obs = torch.randn(B, 6, 9, 9)
        own_tgt = torch.randint(-1, 2, (B, N_POINTS)).float()
        mgn_tgt = torch.randn(B) * 0.2
        with torch.no_grad():
            _, _, own0, mgn0 = aux.forward_aux(obs)
            l0 = (0.5 * ((own0 - own_tgt) ** 2).mean()
                  + 0.5 * ((mgn0 - mgn_tgt) ** 2).mean()).item()
        stats = aux_update(aux, opt, cfg, obs, own_tgt, mgn_tgt,
                           0.5, 0.5, epochs=5)
        with torch.no_grad():
            _, _, own1, mgn1 = aux.forward_aux(obs)
            l1 = (0.5 * ((own1 - own_tgt) ** 2).mean()
                  + 0.5 * ((mgn1 - mgn_tgt) ** 2).mean()).item()
        self.assertLess(l1, l0)
        self.assertIn("aux_own", stats)
        self.assertIn("aux_margin", stats)


class TestLabelPlumbing(unittest.TestCase):
    def test_attach_finished_game_labels(self):
        # tiny max_plies so games finish fast; labels must attach to exactly
        # the finished game's steps, with one margin value per game
        env = SelfPlayGo(num_envs=2, seed=0, max_plies=12,
                         opponent_fn=random_opponent)
        env.reset()
        T = 12
        b_own = torch.zeros(T, 2, N_POINTS)
        b_margin = torch.zeros(T, 2)
        b_valid = torch.zeros(T, 2, dtype=torch.bool)
        ep_steps = [[], []]
        for t in range(T):
            for i in range(2):
                ep_steps[i].append(t)
            masks = env.legal_masks_learner()
            acts = np.array([np.random.choice(np.flatnonzero(masks[i]))
                             for i in range(2)])
            _, _, dones, _, _ = env.step(acts)
            for i in np.where(dones)[0]:
                attach_finished_game_labels(env, i, ep_steps[i],
                                            b_own, b_margin, b_valid)
                # every labeled step of this game shares its margin label
                mvals = b_margin[ep_steps[i], i].numpy()
                self.assertTrue((mvals == mvals[0]).all())
                # ownership entries are in {-1, 0, 1}
                self.assertTrue(
                    bool(((b_own[ep_steps[i], i].abs() <= 1)).all()))
                ep_steps[i] = []
            if np.any(dones):
                env.reset(np.where(dones)[0])
        self.assertTrue(bool(b_valid.any()))


class TestTrunkMappedOptimizerResume(unittest.TestCase):
    """Resuming a real plain-GoNet checkpoint (e.g. the 15.5M run) into the
    always-GoNetAux trainer: the checkpoint's optimizer was built over the
    plain net's params, so load_ckpt must migrate its Adam state onto the
    trunk instead of failing with a param-group size mismatch."""

    def _plain_checkpoint(self, path):
        from gotrain.train_selfplay import save_ckpt
        gonet = GoNet()
        opt = torch.optim.Adam(gonet.parameters(), lr=2.5e-4)
        # populate Adam state (step / exp_avg / exp_avg_sq), as a real run would
        for _ in range(3):
            logits, value = gonet(torch.randn(4, 6, 9, 9))
            (logits.sum() + value.sum()).backward()
            opt.step()
            opt.zero_grad()
        old_opt_sd = opt.state_dict()
        save_ckpt(path, gonet, opt, gonet, step=15503360, ppo_iter=100,
                  snap_ptr=9, hparams={"lr": 2.5e-4, "train_komi": 6.5})
        return old_opt_sd

    def test_optimizer_state_migrated_onto_trunk(self):
        import os
        import tempfile
        from gotrain.train_selfplay import load_ckpt
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "latest.pt")
            old_opt_sd = self._plain_checkpoint(path)

            policy = GoNetAux()
            snapshot = GoNetAux()
            optimizer = torch.optim.Adam(policy.parameters(), lr=2.5e-4)
            # this exact call raised ValueError (param group size mismatch)
            # before the optimizer-state migration existed
            load_ckpt(path, policy, optimizer, snapshot, torch.device("cpu"))

            # trunk weights mapped
            for (n_new, p_new), (n_old, p_old) in zip(
                    sorted(policy.trunk.named_parameters()),
                    sorted(GoNet().named_parameters())):
                self.assertEqual(n_new, n_old)

            # Adam state migrated by name: trunk keeps exp_avg/exp_avg_sq/step
            new_named = dict(policy.named_parameters())
            new_index = {n: i for i, n in enumerate(new_named)}
            new_opt_sd = optimizer.state_dict()
            old_state_by_pos = old_opt_sd["state"]
            old_names = [n for n, _ in GoNet().named_parameters()]
            self.assertEqual(len(new_opt_sd["param_groups"]), 1)
            self.assertAlmostEqual(
                new_opt_sd["param_groups"][0]["lr"], 2.5e-4)
            migrated = 0
            for pos, old_name in enumerate(old_names):
                old_id = old_opt_sd["param_groups"][0]["params"][pos]
                new_id = new_index["trunk." + old_name]
                self.assertIn(new_id, new_opt_sd["state"])
                for k in ("exp_avg", "exp_avg_sq", "step"):
                    old_v = old_state_by_pos[old_id][k]
                    new_v = new_opt_sd["state"][new_id][k]
                    if torch.is_tensor(old_v):
                        self.assertTrue(torch.equal(old_v, new_v), k)
                    else:
                        self.assertEqual(old_v, new_v)
                migrated += 1
            self.assertEqual(migrated, len(old_names))

            # aux-head params get no carried-over state (Adam inits fresh)
            aux_ids = {new_index[n] for n in new_named if n in AUX_KEYS}
            self.assertTrue(aux_ids)
            self.assertFalse(aux_ids & set(new_opt_sd["state"].keys()))

            # snapshot trunk mapped too, and training can proceed: one mixed
            # PPO+aux step must not raise inside Adam
            logits, value = policy(torch.randn(4, 6, 9, 9))
            _, _, own, margin = policy.forward_aux(torch.randn(4, 6, 9, 9))
            loss = (logits.sum() + value.sum() + own.sum() + margin.sum())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # raises if the migrated state is malformed


if __name__ == "__main__":
    unittest.main()
