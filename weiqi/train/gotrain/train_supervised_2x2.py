"""Supervised 2x2 architecture diagnostic on human game data.

Trains all four architectures (baseline, wide, pool, widepool) on
(position -> human move) pairs and compares top-1 accuracy.

Usage:
    # 1. Build dataset from SGFs:
    python -m gotrain.dataset --sgf /path/to/go9 --out data/goquest
    # 2. Run diagnostic:
    python -m gotrain.train_supervised_2x2 --data data/goquest --out /tmp/sup_2x2

Citation: Go Quest 9x9 game records (Tanasa / Go Quest app), shared by
Hiroshi Yamashita to the computer-go mailing list, Dec 28, 2015.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .net import GoNet
from .net_wide import GoNetWide, GoNetPool, GoNetWidePool

ARCHS = {
    "baseline": GoNet,
    "wide": GoNetWide,
    "pool": GoNetPool,
    "widepool": GoNetWidePool,
}


def load_split(data_dir, split):
    X = np.load(os.path.join(data_dir, f"{split}_x.npy"))
    y = np.load(os.path.join(data_dir, f"{split}_y.npy"))
    return torch.from_numpy(X), torch.from_numpy(y).long()


@torch.no_grad()
def evaluate(net, X, y, device, batch_size=2048):
    net.eval()
    correct, total = 0, 0
    n = len(X)
    for i in range(0, n, batch_size):
        xb = X[i:i + batch_size].to(device)
        yb = y[i:i + batch_size].to(device)
        logits, _ = net(xb)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.shape[0]
    return correct / total


def train_one(net, X_train, y_train, device, epochs, batch_size, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(X_train)
    net.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = X_train[idx].to(device)
            yb = y_train[idx].to(device)
            logits, _ = net(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset dir from gotrain.dataset")
    ap.add_argument("--out", required=True, help="output dir for results.json")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--archs", nargs="+", default=["baseline", "wide", "pool", "widepool"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-train", type=int, default=0,
                    help="cap training positions (0 = all)")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    X_train, y_train = load_split(args.data, "train")
    X_val, y_val = load_split(args.data, "val")
    if args.max_train and len(X_train) > args.max_train:
        # Deterministic subsample for quick experiments
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X_train), args.max_train, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
    print(f"train={len(X_train)} val={len(X_val)}")

    os.makedirs(args.out, exist_ok=True)
    results = {}
    for name in args.archs:
        print(f"\n=== {name} ===")
        net = ARCHS[name]()
        print(f"params: {net.param_count():,}")
        t0 = time.perf_counter()
        train_one(net, X_train, y_train, device, args.epochs,
                  args.batch_size, args.lr)
        acc = evaluate(net, X_val, y_val, device)
        dt = time.perf_counter() - t0
        results[name] = {
            "params": net.param_count(),
            "top1": acc,
            "train_time_s": dt,
        }
        print(f"  top1={acc:.4f} ({dt:.0f}s)")

    print("\n=== Summary ===")
    print(f"{'arch':<12} {'params':>8} {'top1':>7}")
    for name in args.archs:
        m = results[name]
        print(f"{name:<12} {m['params']:>8,} {m['top1']:>7.4f}")

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out}/results.json")


if __name__ == "__main__":
    main()
