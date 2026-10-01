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

import numpy as np
import torch
import torch.nn.functional as F

from .net import GoNet
from .net_aux import GoNetAux, TRUNK_PREFIX
from .net_wide import GoNetWide, GoNetPool, GoNetWidePool

NetT = GoNet | GoNetWide | GoNetPool | GoNetWidePool | GoNetAux

ARCHITECTURES: dict[str, type[NetT]] = {
    "baseline": GoNet,
    "wide": GoNetWide,
    "pool": GoNetPool,
    "widepool": GoNetWidePool,
}

# Architectures a --init-checkpoint can warm-start from (auto-detected from
# the state-dict key prefix). Kept separate from ARCHITECTURES so the 2x2
# diagnostic's default --archs list is unchanged.
CKPT_ARCHITECTURES: dict[str, type[NetT]] = {
    "baseline": GoNet,
    "aux": GoNetAux,
}


def detect_ckpt_arch(state_dict: dict) -> str:
    return "aux" if any(k.startswith(TRUNK_PREFIX) for k in state_dict) \
        else "baseline"


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
             ) -> dict[str, float | None]:
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

    out: dict[str, float | None] = {
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


def load_bc_data(bc_dir: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load Go Quest supervised data (train_x.npy / train_y.npy)."""
    import os
    x = torch.from_numpy(np.load(os.path.join(bc_dir, "train_x.npy")))
    y = torch.from_numpy(np.load(os.path.join(bc_dir, "train_y.npy"))).long()
    return x, y


def finetune(args: argparse.Namespace, device: torch.device | str) -> None:
    """Search-distillation probe, Phase 2.

    Warm-starts from --init-checkpoint (model + optimizer state, arch
    auto-detected) and fine-tunes on MCTS visit-distribution targets
    (--data), optionally retaining the human BC stream (--bc-data,
    --bc-coef) and distilling the search root value (--value-coef).

    The control arm is distill_coef=0: same steps, same BC exposure.
    """
    import os
    ck = torch.load(args.init_checkpoint, map_location=device,
                    weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    arch_name = detect_ckpt_arch(sd)
    net = CKPT_ARCHITECTURES[arch_name]().to(device)
    net.load_state_dict(sd)
    print(f"warm-start: arch={arch_name} "
          f"params={net.param_count():,} from {args.init_checkpoint}")

    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    if isinstance(ck, dict) and "optimizer" in ck:
        opt.load_state_dict(ck["optimizer"])
        print("loaded optimizer state (warm Adam)")
    if args.ft_lr is not None:
        for g in opt.param_groups:
            g["lr"] = args.ft_lr
        print(f"lr overridden to {args.ft_lr}")
    else:
        print(f"lr={opt.param_groups[0]['lr']} (from checkpoint)")

    data = load_dataset(args.data)
    d_obs = data["obs"]
    d_pol = data["policy"]
    d_val = data["value"]
    assert d_obs is not None and d_pol is not None
    n_d = len(d_obs)
    has_value = d_val is not None and args.value_coef > 0
    print(f"distill positions: {n_d} (value targets: {has_value})")

    bc_x = bc_y = None
    n_b = 0
    if args.bc_coef > 0:
        if not args.bc_data:
            raise SystemExit("--bc-data is required when --bc-coef > 0")
        bc_x, bc_y = load_bc_data(args.bc_data)
        n_b = len(bc_x)
        print(f"BC positions: {n_b} coef={args.bc_coef}")

    net.train()
    rng = torch.Generator().manual_seed(args.seed)
    bs = args.batch_size
    run = {"dce": 0.0, "bce": 0.0, "vmse": 0.0, "n": 0}
    t0 = time.time()
    for step in range(1, args.steps + 1):
        d_idx = torch.randint(0, n_d, (bs,), generator=rng)
        x = d_obs[d_idx].to(device)
        p_targ = d_pol[d_idx].to(device)
        logits, v_pred = net(x)
        logp = F.log_softmax(logits, dim=1)
        dce = -(p_targ * logp).sum(dim=1).mean()
        loss = args.distill_coef * dce
        bce = torch.zeros((), device=device)
        if args.bc_coef > 0:
            assert bc_x is not None and bc_y is not None
            b_idx = torch.randint(0, n_b, (bs,), generator=rng)
            bx = bc_x[b_idx].to(device)
            by = bc_y[b_idx].to(device)
            blogits, _ = net(bx)
            bce = F.cross_entropy(blogits, by)
            loss = loss + args.bc_coef * bce
        vmse = torch.zeros((), device=device)
        if has_value:
            assert d_val is not None
            v_targ = d_val[d_idx].to(device)
            vmse = F.mse_loss(v_pred, v_targ)
            loss = loss + args.value_coef * vmse

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        opt.step()

        run["dce"] += dce.item()
        run["bce"] += bce.item()
        run["vmse"] += vmse.item()
        run["n"] += 1
        if step % args.log_every == 0:
            dt = time.time() - t0
            print(f"  step {step}/{args.steps} "
                  f"distill_ce={run['dce'] / run['n']:.4f} "
                  f"bc_ce={run['bce'] / run['n']:.4f} "
                  f"value_mse={run['vmse'] / run['n']:.4f} "
                  f"({dt / run['n'] * 1000:.0f}ms/step)", flush=True)
            run = {"dce": 0.0, "bce": 0.0, "vmse": 0.0, "n": 0}
            t0 = time.time()

    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, "finetune.pt")
    torch.save({
        "model": net.state_dict(),
        "optimizer": opt.state_dict(),
        "steps": args.steps,
        "args": vars(args),
    }, ckpt_path)
    print(f"saved {ckpt_path}")
    with open(os.path.join(args.out, "finetune.json"), "w") as f:
        import json
        json.dump({
            "arch": arch_name,
            "steps": args.steps,
            "distill_coef": args.distill_coef,
            "bc_coef": args.bc_coef,
            "value_coef": args.value_coef,
            "distill_positions": n_d,
        }, f, indent=1)


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
    # -- search-distillation fine-tune mode (Phase 2 of the probe) --------
    ap.add_argument("--init-checkpoint", default=None,
                    help="warm-start from a training checkpoint (model + "
                    "optimizer); arch auto-detected")
    ap.add_argument("--bc-data", default=None,
                    help="dir with Go Quest train_x.npy/train_y.npy")
    ap.add_argument("--bc-coef", type=float, default=0.0,
                    help="human-BC hard-CE coefficient (0.1 mirrors PPO+BC)")
    ap.add_argument("--distill-coef", type=float, default=1.0,
                    help="MCTS soft-target CE coefficient (0.0 = control arm)")
    ap.add_argument("--value-coef", type=float, default=0.0,
                    help="search root-value MSE coefficient")
    ap.add_argument("--steps", type=int, default=3000,
                    help="minibatch steps in fine-tune mode")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--ft-lr", type=float, default=None,
                    help="override the checkpoint's optimizer lr")
    args = ap.parse_args()

    if args.device == "auto":
        device: torch.device | str = (
            "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device
    print(f"device={device}")

    if args.init_checkpoint:
        finetune(args, device)
        return

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

    results: dict[str, dict[str, float | None]] = {}
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
