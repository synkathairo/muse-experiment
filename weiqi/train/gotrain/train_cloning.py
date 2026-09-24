"""Supervised behavioral cloning (Imitator) — PLAN.md §4.

Trains gotrain.net.GoNet on (planes -> move) pairs built by gotrain.dataset.

Resumability (survives VM restarts):
  - rolling `latest.pt` (model + optimizer + step + RNG states), saved every
    --ckpt-every steps (default 500);
  - frozen log-spaced snapshots `snap_{step}.pt` at 1k, 3k, 10k, 30k, ... —
    never overwritten, feed the time-machine.

Resume with:  python -m gotrain.train_cloning --data <dir> --out <dir> --resume <dir>/latest.pt

Usage:
    python -m gotrain.train_cloning --data data/ds_full --out runs/imit_v1 \\
        --max-steps 60000 --val-every 1000
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .net import GoNet

SNAP_STEPS = [1000, 3000, 10000, 30000, 100000, 300000, 1000000]
VAL_SUBSET = 20000


def load_split(data_dir, split):
    X = np.load(os.path.join(data_dir, f"{split}_x.npy"), mmap_mode="r")
    y = np.load(os.path.join(data_dir, f"{split}_y.npy"), mmap_mode="r")
    z = np.load(os.path.join(data_dir, f"{split}_z.npy"), mmap_mode="r")
    zm = np.load(os.path.join(data_dir, f"{split}_zm.npy"), mmap_mode="r")
    return X, y, z, zm


@torch.no_grad()
def evaluate(model, X, y, z, zm, idx, batch=2048):
    model.eval()
    tot_loss = tot_acc = tot_vmse = tot_vmask = 0.0
    n = len(idx)
    for s in range(0, n, batch):
        b = idx[s:s + batch]
        xb = torch.from_numpy(np.asarray(X[b]))
        yb = torch.from_numpy(np.asarray(y[b]))
        logits, value = model(xb)
        tot_loss += F.cross_entropy(logits, yb, reduction="sum").item()
        tot_acc += (logits.argmax(1) == yb).sum().item()
        zb = torch.from_numpy(np.asarray(z[b]))
        zmb = torch.from_numpy(np.asarray(zm[b]))
        if zmb.sum().item() > 0:
            tot_vmse += (F.mse_loss(value, zb, reduction="none") * zmb).sum().item()
            tot_vmask += zmb.sum().item()
    model.train()
    return {
        "loss": tot_loss / n,
        "acc": tot_acc / n,
        "value_mse": (tot_vmse / tot_vmask) if tot_vmask else float("nan"),
    }


def save_ckpt(path, model, opt, step, snap_ptr, best_acc, bad_evals, hparams):
    torch.save({
        "step": step,
        "snap_ptr": snap_ptr,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "best_acc": best_acc,
        "bad_evals": bad_evals,
        "hparams": hparams,
    }, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset dir from gotrain.dataset")
    ap.add_argument("--out", required=True, help="run dir (checkpoints + train.log)")
    ap.add_argument("--max-steps", type=int, default=60000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--patience", type=int, default=8,
                    help="val evals without acc improvement before early stop (0=off)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--resume", default=None, help="path to latest.pt")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    Xtr, ytr, ztr, zmtr = load_split(args.data, "train")
    Xva, yva, zva, zmva = load_split(args.data, "val")
    n_train, n_val = len(ytr), len(yva)
    print(f"train={n_train} val={n_val} threads={torch.get_num_threads()}", flush=True)
    va_idx = np.random.RandomState(args.seed).choice(
        n_val, size=min(VAL_SUBSET, n_val), replace=False)

    model = GoNet()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    hparams = {k: v for k, v in vars(args).items() if k != "resume"}

    step = 0
    snap_ptr = 0
    best_acc = 0.0
    bad_evals = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        step = ck["step"]
        snap_ptr = ck["snap_ptr"]
        best_acc = ck.get("best_acc", 0.0)
        bad_evals = ck.get("bad_evals", 0)
        torch.set_rng_state(ck["torch_rng"])
        np.random.set_state(ck["numpy_rng"])
        print(f"resumed from {args.resume} at step {step}", flush=True)

    logf = open(os.path.join(args.out, "train.log"), "a")
    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    log(f"start: {json.dumps(hparams)} resuming_at={step}")
    t0 = time.time()
    model.train()
    stopped = False
    while step < args.max_steps and not stopped:
        perm = torch.randperm(n_train)
        for s in range(0, n_train, args.batch_size):
            if step >= args.max_steps or stopped:
                break
            b = perm[s:s + args.batch_size]
            xb = torch.from_numpy(np.asarray(Xtr[b]))
            yb = torch.from_numpy(np.asarray(ytr[b]))
            zb = torch.from_numpy(np.asarray(ztr[b]))
            zmb = torch.from_numpy(np.asarray(zmtr[b]))
            opt.zero_grad()
            logits, value = model(xb)
            loss = F.cross_entropy(logits, yb)
            if zmb.sum().item() > 0:
                vloss = (F.mse_loss(value, zb, reduction="none") * zmb).sum() / zmb.sum()
                loss = loss + args.value_weight * vloss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1

            while snap_ptr < len(SNAP_STEPS) and step >= SNAP_STEPS[snap_ptr]:
                sp = os.path.join(args.out, f"snap_{SNAP_STEPS[snap_ptr]:07d}.pt")
                save_ckpt(sp, model, opt, step, snap_ptr + 1, best_acc, bad_evals, hparams)
                log(f"frozen snapshot -> {sp}")
                snap_ptr += 1

            if step % args.ckpt_every == 0:
                save_ckpt(os.path.join(args.out, "latest.pt"), model, opt,
                          step, snap_ptr, best_acc, bad_evals, hparams)

            if step % args.val_every == 0:
                m = evaluate(model, Xva, yva, zva, zmva, va_idx)
                el = time.time() - t0
                log(f"step {step}: val_loss={m['loss']:.4f} val_acc={m['acc']:.4f} "
                    f"val_vmse={m['value_mse']:.4f} steps_per_s={step / el:.2f}")
                if m["acc"] > best_acc + 1e-4:
                    best_acc = m["acc"]
                    bad_evals = 0
                    save_ckpt(os.path.join(args.out, "best.pt"), model, opt,
                              step, snap_ptr, best_acc, bad_evals, hparams)
                else:
                    bad_evals += 1
                    if args.patience and bad_evals >= args.patience:
                        log(f"early stop: no val_acc improvement for {bad_evals} evals")
                        stopped = True
                        break

    save_ckpt(os.path.join(args.out, "latest.pt"), model, opt,
              step, snap_ptr, best_acc, bad_evals, hparams)
    log(f"done at step {step}, best_val_acc={best_acc:.4f}")
    logf.close()


if __name__ == "__main__":
    main()
