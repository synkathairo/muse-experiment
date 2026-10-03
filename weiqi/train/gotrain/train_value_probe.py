"""Value-head probe: is the value head the bottleneck?

Trains a fresh value head (identical architecture to the net's own head) on the
frozen champion trunk, using terminal game outcomes as targets. Compares held-out
Brier score of the fresh head vs the existing value head.

Split is by game (every 10th game held out). If the fresh head is clearly better
calibrated, the existing value head is undertrained / target-limited and fixing it
is the next lever. If not, the value head is fine and we look elsewhere.

Usage:
    PYTHONPATH=. python -m gotrain.train_value_probe \
        --data ~/workspace/runs/value_probe/targets.npz \
        --checkpoint ~/workspace/runs/ppo_bc/latest.pt \
        --epochs 30 --batch-size 2048 --lr 1e-3 \
        --out ~/workspace/runs/value_probe/probe.json
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gotrain.net import GoNet
from gotrain.net_aux import GoNetAux, TRUNK_PREFIX


class FreshValueHead(nn.Module):
    """Identical architecture to GoNet's value head, randomly initialized."""

    def __init__(self) -> None:
        super().__init__()
        self.val_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.val_fc1 = nn.Linear(81, 32)
        self.val_fc2 = nn.Linear(32, 1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.val_fc1(self.val_conv(feat).flatten(1)))
        return torch.tanh(self.val_fc2(h)).squeeze(1)


def load_trunk(checkpoint: str) -> nn.Module:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    net = GoNetAux() if any(k.startswith(TRUNK_PREFIX) for k in sd) else GoNet()
    net.load_state_dict(sd)
    net.eval()
    for p in net.parameters():
        p.requires_grad = False
    # Unwrap the aux wrapper: the probe is about the shared trunk + value head.
    trunk: nn.Module = net.trunk if isinstance(net, GoNetAux) else net
    return trunk


def brier(probs: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((probs - y) ** 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    d = np.load(args.data)
    planes, y, game = d["planes"], d["y"].astype(np.float64), d["game"]
    order = np.argsort(game, kind="stable")
    planes, y, game = planes[order], y[order], game[order]
    ugames = np.unique(game)
    test_games = set(ugames[::10])
    test_mask = np.array([g in test_games for g in game])
    print(f"{len(planes)} positions, {len(ugames)} games; "
          f"train {int((~test_mask).sum())}, test {int(test_mask.sum())}", flush=True)
    print(f"label mean: train {y[~test_mask].mean():.3f}, test {y[test_mask].mean():.3f}",
          flush=True)

    net = load_trunk(args.checkpoint)
    head = FreshValueHead()
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)

    Xtr = torch.from_numpy(planes[~test_mask])
    ytr = torch.from_numpy((2 * y[~test_mask] - 1).astype(np.float32))
    Xte = torch.from_numpy(planes[test_mask])
    yte = y[test_mask]

    # Baseline: existing value head on the held-out set.
    with torch.no_grad():
        _, v_old = net(Xte)
        p_old = ((v_old.numpy() + 1) / 2).clip(0, 1)
    brier_old = brier(p_old, yte)
    brier_naive = brier(np.full_like(yte, 0.5), yte)
    print(f"baseline Brier: existing head {brier_old:.4f}, naive 0.5 {brier_naive:.4f}",
          flush=True)

    n = len(Xtr)
    for ep in range(args.epochs):
        perm = torch.randperm(n)
        tot, cnt = 0.0, 0
        head.train()
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            with torch.no_grad():
                feat = net.trunk.features(Xtr[idx])
            pred = head(feat)
            loss = F.mse_loss(pred, ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
            cnt += len(idx)
        if (ep + 1) % 5 == 0 or ep == 0:
            head.eval()
            with torch.no_grad():
                pv = head(net.trunk.features(Xte)).numpy()
                b = brier(((pv + 1) / 2).clip(0, 1), yte)
            print(f"epoch {ep + 1}/{args.epochs} train_mse={tot / cnt:.4f} "
                  f"heldout_brier={b:.4f}", flush=True)

    # Final held-out comparison, overall and by game stage.
    head.eval()
    with torch.no_grad():
        p_new = ((head(net.trunk.features(Xte)).numpy() + 1) / 2).clip(0, 1)
    # move index within game as a stage proxy (game[] is sorted)
    first_idx = np.sort(np.unique(game[test_mask], return_index=True)[1])
    gidx = np.searchsorted(first_idx, np.arange(len(yte)), side="right") - 1
    move_idx = np.arange(len(yte)) - first_idx[gidx]
    stages = np.clip(move_idx // 40, 0, 2)
    result = {
        "brier_existing": brier_old,
        "brier_fresh": brier(p_new, yte),
        "brier_naive": brier_naive,
        "n_test": int(test_mask.sum()),
        "by_stage": {},
    }
    for s, name in [(0, "early"), (1, "mid"), (2, "late")]:
        m = stages == s
        result["by_stage"][name] = {
            "brier_existing": brier(p_old[m], yte[m]),
            "brier_fresh": brier(p_new[m], yte[m]),
            "n": int(m.sum()),
        }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1), flush=True)


if __name__ == "__main__":
    sys.exit(main())
