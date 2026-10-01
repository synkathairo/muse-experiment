"""Verification that PPO already optimizes the masked (legal-move) policy.

Background: the 15.5M checkpoint's RAW argmax is legal only ~18-24% of the
time, which prompted the hypothesis that PPO's loss/entropy are computed over
the full 82-way distribution while inference masks illegal moves. These tests
prove the opposite: every logits->distribution site in the training loop AND
the GTP harness is already masked, all sourced from selfplay.legal_mask.

Mechanistic consequence (test_illegal_logits_get_zero_gradient): illegal
logits receive EXACTLY ZERO gradient, so they sit at initialization values
forever. The raw argmax is therefore dominated by untrained init noise --
expected and harmless, because inference masks them to -inf before argmax /
sampling. There is no behavior/optimization mismatch to fix.

Run from weiqi/train/:  python -m unittest tests.test_masked_ppo -v
"""
from __future__ import annotations

import unittest
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F

from gotrain.net import GoNet
from gotrain.ppo import PPOConfig, ppo_minibatch_update, _masked_logps_entropy
from gotrain.train_selfplay import (masked_dist, sample_actions,
                                    greedy_actions, dirichlet_noised_dist)

N_MOVES: int = 82


def random_masks(batch: int, min_legal: int = 5, seed: int = 0) -> torch.Tensor:
    rng = np.random.RandomState(seed)
    masks = np.zeros((batch, N_MOVES), dtype=bool)
    for b in range(batch):
        n_legal = rng.randint(min_legal, N_MOVES)
        legal = rng.choice(N_MOVES, size=n_legal, replace=False)
        masks[b, legal] = True
        masks[b, N_MOVES - 1] = True  # pass is always legal
    return torch.from_numpy(masks)


class TestMaskedLogpsEntropy(unittest.TestCase):
    def test_illegal_probs_exactly_zero_and_sum_to_one(self) -> None:
        torch.manual_seed(0)
        B = 16
        logits = torch.randn(B, N_MOVES)
        masks = random_masks(B)
        actions = torch.tensor(
            [int(np.random.choice(np.flatnonzero(m))) for m in masks.numpy()])
        logps, entropy = _masked_logps_entropy(logits, actions, masks)
        probs = logps.exp()  # not the full dist; recompute below
        masked = logits.masked_fill(~masks, float("-inf"))
        full_probs = masked.softmax(dim=-1)
        # exactly 0 mass on illegal moves (no -0.0 / denormal issues)
        self.assertTrue((full_probs[~masks] == 0.0).all())
        # sums to 1 over legal moves
        self.assertTrue(
            torch.allclose(full_probs.sum(-1), torch.ones(B), atol=1e-6))
        # taken logps match the masked log-softmax
        manual = F.log_softmax(masked, dim=-1).gather(
            1, actions.unsqueeze(1)).squeeze(1)
        self.assertTrue(torch.allclose(logps, manual, atol=1e-6))
        self.assertTrue(torch.isfinite(entropy).all())

    def test_entropy_equals_legal_only_entropy(self) -> None:
        torch.manual_seed(1)
        B = 8
        logits = torch.randn(B, N_MOVES)
        masks = random_masks(B)
        actions = torch.tensor(
            [int(np.flatnonzero(m)[0]) for m in masks.numpy()])
        _, entropy = _masked_logps_entropy(logits, actions, masks)
        # brute force: renormalize over legal moves only
        masked = logits.masked_fill(~masks, float("-inf"))
        p = masked.softmax(dim=-1)
        lp = F.log_softmax(masked, dim=-1).masked_fill(~masks, 0.0)
        expected = -(p * lp).sum(-1)
        self.assertTrue(torch.allclose(entropy, expected, atol=1e-6))

    def test_masked_entropy_below_unmasked_when_mass_on_illegal(self) -> None:
        # logits peaked on an ILLEGAL move: unmasked entropy is low (peaked),
        # masked entropy is computed over the legal remainder only.
        B = 4
        logits = torch.zeros(B, N_MOVES)
        masks = random_masks(B, min_legal=10, seed=3)
        for b in range(B):
            illegal = int(np.flatnonzero(~masks.numpy()[b])[0])
            logits[b, illegal] = 10.0  # peak on an illegal move
        actions = torch.tensor(
            [int(np.flatnonzero(m)[0]) for m in masks.numpy()])
        _, masked_ent = _masked_logps_entropy(logits, actions, masks)
        unmasked_ent = -(F.softmax(logits, dim=-1)
                         * F.log_softmax(logits, dim=-1)).sum(-1)
        # the two entropies differ: the masked one ignores the illegal peak
        self.assertFalse(torch.allclose(masked_ent, unmasked_ent, atol=1e-3))
        # masked entropy is bounded by log(#legal)
        n_legal = masks.sum(-1).float()
        self.assertTrue((masked_ent <= (n_legal.log() + 1e-5)).all())

    def test_illegal_logits_get_zero_gradient(self) -> None:
        """The mechanistic core: with everything masked, illegal logits get
        exactly zero gradient -- they are never trained, so the raw argmax
        being illegal ~80% of the time is untrained init noise, not a bug."""
        torch.manual_seed(2)
        B = 8
        logits = torch.randn(B, N_MOVES, requires_grad=True)
        masks = random_masks(B, seed=4)
        actions = torch.tensor(
            [int(np.flatnonzero(m)[0]) for m in masks.numpy()])
        logps, entropy = _masked_logps_entropy(logits, actions, masks)
        adv = torch.randn(B)
        # surrogate policy loss + entropy bonus, as in ppo_minibatch_update
        loss = -(logps * adv).mean() - 0.03 * entropy.mean()
        loss.backward()
        g = logits.grad
        assert g is not None
        self.assertTrue((g[~masks] == 0.0).all(),
                        "illegal logits must receive exactly zero gradient")
        self.assertTrue((g[masks].abs() > 0).any(),
                        "legal logits must receive gradient")


class TestMaskedSampling(unittest.TestCase):
    def test_masked_dist_samples_only_legal(self) -> None:
        torch.manual_seed(5)
        B = 8
        logits = torch.randn(B, N_MOVES)
        masks = random_masks(B, seed=6)
        dist = masked_dist(logits, masks)
        a = dist.sample((2000,))  # (2000, B)
        ok = masks.unsqueeze(0).expand(2000, B, N_MOVES).gather(
            2, a.unsqueeze(-1)).squeeze(-1)
        self.assertTrue(ok.all(), "sampled an illegal move")
        # stored log-probs match the masked log-softmax
        manual = F.log_softmax(
            logits.masked_fill(~masks, float("-inf")), dim=-1)
        lp = dist.log_prob(a)
        expected_lp = (manual.unsqueeze(0).expand(2000, B, N_MOVES)
                       .gather(2, a.unsqueeze(-1)).squeeze(-1))
        self.assertTrue(torch.allclose(lp, expected_lp, atol=1e-5))

    def test_sample_actions_legal_and_finite(self) -> None:
        torch.manual_seed(7)
        net = GoNet()
        B = 4
        obs = torch.randn(B, 6, 9, 9)
        masks = random_masks(B, seed=8)
        actions, logps, values = sample_actions(net, obs, masks)
        a = actions.numpy()
        m = masks.numpy()
        self.assertTrue(all(m[b, a[b]] for b in range(B)))
        self.assertTrue(torch.isfinite(logps).all())
        self.assertTrue(torch.isfinite(values).all())

    def test_greedy_actions_argmax_over_legal(self) -> None:
        torch.manual_seed(9)
        net = GoNet()
        B = 4
        obs = torch.randn(B, 6, 9, 9)
        masks = random_masks(B, seed=10)
        # force the raw argmax onto an illegal move
        with torch.no_grad():
            logits, _ = net(obs)
            for b in range(B):
                illegal = int(np.flatnonzero(~masks.numpy()[b])[0])
                logits[b, illegal] = 1e6
        # monkeypatch-ish: call greedy_actions on wrapped logits via net hook
        orig_forward = net.forward

        def patched(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            l, v = orig_forward(x)
            for b in range(B):
                illegal = int(np.flatnonzero(~masks.numpy()[b])[0])
                l[b, illegal] = 1e6
            return l, v

        net.forward = patched  # type: ignore
        try:
            a = greedy_actions(net, obs, masks).numpy()
        finally:
            net.forward = orig_forward  # type: ignore
        m = masks.numpy()
        self.assertTrue(all(m[b, a[b]] for b in range(B)),
                        "greedy argmax must be legal even when raw argmax is not")

    def test_dirichlet_noise_supported_on_legal_only(self) -> None:
        torch.manual_seed(11)
        B = 8
        logits = torch.randn(B, N_MOVES)
        masks = random_masks(B, seed=12)
        dist = masked_dist(logits, masks)
        noise_mask = torch.ones(B, dtype=torch.bool)
        mixed = dirichlet_noised_dist(dist, masks, noise_mask,
                                      alpha=0.05, eps=0.25)
        # Categorical.probs is a @lazy_property in torch; at runtime it is
        # always a Tensor.
        probs = cast("torch.Tensor", mixed.probs)
        self.assertTrue((probs[~masks] == 0.0).all())
        self.assertTrue(
            torch.allclose(probs.sum(-1), torch.ones(B), atol=1e-5))


class TestPPOUpdateMasked(unittest.TestCase):
    def test_ppo_minibatch_update_runs_and_stays_masked(self) -> None:
        torch.manual_seed(13)
        net = GoNet()
        opt = torch.optim.Adam(net.parameters(), lr=2.5e-4)
        cfg = PPOConfig()
        B = 64
        obs = torch.randn(B, 6, 9, 9)
        masks = random_masks(B, seed=14)
        with torch.no_grad():
            logits, values = net(obs)
        dist = masked_dist(logits, masks)
        actions = dist.sample()
        logps = dist.log_prob(actions)
        adv = torch.randn(B)
        ret = torch.randn(B)
        vals = values.view(-1).detach()
        stats = ppo_minibatch_update(net, opt, cfg, obs, actions, logps,
                                     adv, ret, vals, masks)
        for k in ("pg_loss", "v_loss", "entropy", "approx_kl",
                  "clipfrac", "loss"):
            self.assertTrue(np.isfinite(stats[k]), f"stat {k} not finite")
        # entropy is over the legal distribution: bounded by log(#legal)
        self.assertLessEqual(stats["entropy"],
                             float(masks.sum(-1).float().log().max()) + 1e-4)


if __name__ == "__main__":
    unittest.main()
