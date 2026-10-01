"""Supervised distillation diagnostic: 2x2 architecture comparison.

Trains 4 architectures on a fixed KataGo-generated dataset:
  1. GoNet (130K baseline)
  2. GoNetWide (466K)
  3. GoNetPool (global pooling)
  4. GoNetWidePool (both)

Dataset format (NPZ from gen_katago_data.py):
  obs: (N, 6, 9, 9) float32 - input planes
  policy: (N, 82) float32 - KataGo policy targets (visit-count distribution)
  value: (N,) float32 - KataGo value estimates
  mask: (N, 82) bool - legal move masks

Metrics: cross-entropy, top-1 accuracy, KL divergence, value MSE.
Per Astra: measure all, not just top-1.

Usage:
  python -m gotrain.train_distill --data /path/to/katago_positions.npz \
      --out /tmp/distill --epochs 20 --batch-size 256
"""
from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .net import GoNet
from .net_wide import GoNetWide, GoNetPool, GoNetWidePool

NetT = GoNet | GoNetWide | GoNetPool | GoNetWidePool

ARCHITECTURES: dict[str, type[NetT]] = {
    "baseline": GoNet,
    "wide": GoNetWide,
    "pool": GoNetPool,
    "widepool": GoNetWidePool,
}


def load_dataset(path: str) -> dict[str, torch.Tensor | None]:
    d = np.load(path)
    # Support both formats: new (obs/policy/value/mask) and legacy
    # distill dataset (planes/teacher/masks, no value)
    if "planes" in d.files:
        # Legacy: teacher = raw KataGo logits, need softmax
        teacher_logits = torch.from_numpy(d["teacher"])
        teacher_probs = torch.softmax(teacher_logits, dim=1)
        out: dict[str, torch.Tensor | None] = {
            "obs": torch.from_numpy(d["planes"]),
            "policy": teacher_probs,
            "mask": torch.from_numpy(d["masks"]),
            "value": None,  # legacy dataset has no value targets
        }
    else:
        out = {
            "obs": torch.from_numpy(d["obs"]),
            "policy": torch.from_numpy(d["policy"]),
            "value": torch.from_numpy(d["value"]),
            "mask": torch.from_numpy(d["mask"]),
        }
    return out


def evaluate(net: NetT, obs: torch.Tensor, policy: torch.Tensor,
             value: torch.Tensor | None, mask: torch.Tensor,
             device: torch.device | str, batch_size: int = 1024
             ) -> dict[str, Any]:
    """Compute CE, top-1, KL, value MSE on the full dataset."""
    net.eval()
    total_ce, total_kl, total_mse = 0.0, 0.0, 0.0
    top1_correct, total = 0, 0
    n = len(obs)
    has_value = value is not None
    with torch.no_grad():
        for i in range(0, n, batch_size):
            b = slice(i, min(i + batch_size, n))
            x = obs[b].to(device)
            p_targ = policy[b].to(device)
            m = mask[b].to(device)

            logits, v_pred = net(x)
            # Mask illegal moves for top-1
            masked_logits = logits.masked_fill(~m, -1e9)

            # Cross-entropy vs soft targets
            logp = F.log_softmax(logits, dim=1)
            ce = -(p_targ * logp).sum(dim=1).mean()

            # KL divergence
            kl = (p_targ * (p_targ.clamp(min=1e-9).log() - logp)).sum(dim=1).mean()

            # Top-1 accuracy (masked)
            pred = masked_logits.argmax(dim=1)
            target = p_targ.argmax(dim=1)
            top1_correct += (pred == target).sum().item()
            total += x.shape[0]

            # Value MSE (if targets available)
            if has_value and value is not None:
                v_targ = value[b].to(device)
                mse = F.mse_loss(v_pred, v_targ)
                total_mse += mse.item() * x.shape[0]

            total_ce += ce.item() * x.shape[0]
            total_kl += kl.item() * x.shape[0]

    out: dict[str, Any] = {
        "ce": total_ce / n,
        "kl": total_kl / n,
        "top1": top1_correct / total,
    }
    out["value_mse"] = total_mse / n if has_value else None
    return out


def train_one(net: NetT, train_data: dict[str, torch.Tensor | None],
              val_data: dict[str, torch.Tensor | None],
              device: torch.device | str, epochs: int, batch_size: int,
              lr: float = 1e-3) -> NetT:
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    obs = train_data["obs"]
    assert obs is not None
    n = len(obs)
    has_value = train_data["value"] is not None
    net.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            x = obs[idx].to(device)
            p_targ = train_data["policy"]
            assert p_targ is not None
            p_targ = p_targ[idx].to(device)

            logits, v_pred = net(x)
            logp = F.log_softmax(logits, dim=1)
            # Policy loss: CE vs KataGo soft targets
            p_loss = -(p_targ * logp).sum(dim=1).mean()
            loss = p_loss
            # Value loss: MSE vs KataGo value (if available)
            if has_value:
                v_targ = train_data["value"]
                assert v_targ is not None
                v_targ = v_targ[idx].to(device)
                v_loss = F.mse_loss(v_pred, v_targ)
                loss = loss + v_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
    return net


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="NPZ with obs/policy/value/mask")
    ap.add_argument("--out", default="/tmp/distill")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--archs", nargs="+", default=list(ARCHITECTURES),
                    choices=list(ARCHITECTURES))
    args = ap.parse_args()

    if args.device == "auto":
        device: torch.device | str = (
            "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device
    print(f"device={device}")

    data = load_dataset(args.data)
    obs_all = data["obs"]
    assert obs_all is not None
    n = len(obs_all)
    n_val = int(n * args.val_frac)
    # Fixed split for comparability across architectures
    rng = torch.Generator().manual_seed(42)
    perm = torch.randperm(n, generator=rng)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    def _split(v: torch.Tensor | None) -> torch.Tensor | None:
        return v[train_idx] if v is not None else None

    def _split_v(v: torch.Tensor | None) -> torch.Tensor | None:
        return v[val_idx] if v is not None else None

    train_data = {k: _split(v) for k, v in data.items()}
    val_data = {k: _split_v(v) for k, v in data.items()}
    print(f"train={len(train_idx)} val={len(val_idx)}")

    results: dict[str, Any] = {}
    for name in args.archs:
        print(f"\n=== {name} ===")
        net = ARCHITECTURES[name]().to(device)
        print(f"params: {net.param_count():,}")
        t0 = time.time()
        train_one(net, train_data, val_data, device, args.epochs,
                  args.batch_size, args.lr)
        dt = time.time() - t0
        val_obs, val_pol, val_mask = val_data["obs"], val_data["policy"], val_data["mask"]
        assert val_obs is not None and val_pol is not None and val_mask is not None
        metrics = evaluate(net, val_obs, val_pol,
                           val_data["value"], val_mask, device)
        metrics["train_time_s"] = dt
        metrics["params"] = net.param_count()
        results[name] = metrics
        vmse = f" value_mse={metrics['value_mse']:.4f}" if metrics['value_mse'] is not None else ""
        print(f"  ce={metrics['ce']:.4f} kl={metrics['kl']:.4f} "
              f"top1={metrics['top1']:.3f}{vmse} "
              f"({dt:.0f}s)")

    print("\n=== Summary ===")
    print(f"{'arch':<12} {'params':>8} {'ce':>8} {'kl':>8} {'top1':>7}")
    for name in args.archs:
        m = results[name]
        print(f"{name:<12} {m['params']:>8,} {m['ce']:>8.4f} {m['kl']:>8.4f} "
              f"{m['top1']:>7.3f}")

    # Save results
    import json
    import os
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out}/results.json")


if __name__ == "__main__":
    main()
