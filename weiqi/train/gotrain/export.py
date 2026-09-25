"""Export a checkpoint to the locked fp16 weight format (PLAN.md §3).

Flat float16 little-endian binary, no header. Tensor order IS the format —
see gotrain.net.EXPORT_ORDER (single source of truth).

Usage:
    python -m gotrain.export --checkpoint runs/imit_v1/snap_0003000.pt \\
        --out-dir ./exports/ --name imit-003k.bin

Batch mode — nearest snapshot per log-spaced target, manifest JSON printed
for pasting into web/site/demos/go/manifest.json:
    python -m gotrain.export --run-dir runs/laptop_clean \\
        --targets 100k,300k,1M,3M,10M,30M,100M \\
        --out-dir ../../web/site/demos/go/weights \\
        --name-prefix selfplay-clean-
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import torch

from .net import GoNet, EXPORT_ORDER, ordered_tensors
from .net_aux import to_gonet_state_dict


def export_weights(model, out_path):
    tensors = ordered_tensors(model)
    with open(out_path, "wb") as f:
        for t in tensors:
            f.write(t.to(torch.float16).numpy().astype("<f2").tobytes())
    return out_path


def read_export(path):
    """Read a weight file back into {key: float32 ndarray} (for tests / checks)."""
    raw = np.fromfile(path, dtype="<f2").astype(np.float32)
    out, off = {}, 0
    for key, shape in EXPORT_ORDER:
        n = int(np.prod(shape))
        out[key] = raw[off:off + n].reshape(shape)
        off += n
    assert off == raw.size, f"trailing bytes: {raw.size - off}"
    return out


def export_checkpoint(checkpoint, out_path):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model = GoNet()
    model.load_state_dict(to_gonet_state_dict(sd))
    model.eval()
    export_weights(model, out_path)
    return os.path.getsize(out_path)


def parse_target(s):
    s = s.strip().lower()
    mult = 1
    if s.endswith("k"):
        mult, s = 1_000, s[:-1]
    elif s.endswith("m"):
        mult, s = 1_000_000, s[:-1]
    return int(float(s) * mult)


def label_for(steps):
    if steps % 1_000_000 == 0:
        return f"{steps // 1_000_000}M"
    if steps % 1_000 == 0:
        return f"{steps // 1_000}k"
    return str(steps)


def nearest_snapshot(run_dir, target):
    cands = []
    for p in glob.glob(os.path.join(run_dir, "snap_*.pt")):
        m = re.search(r"snap_(\d+)\.pt$", p)
        if m:
            cands.append((int(m.group(1)), p))
    if not cands:
        raise SystemExit(f"no snap_*.pt found in {run_dir}")
    return min(cands, key=lambda sp: abs(sp[0] - target))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint", help=".pt file (single-export mode)")
    g.add_argument("--run-dir", help="dir of snap_*.pt (batch mode with --targets)")
    ap.add_argument("--targets",
                    help="comma-separated step counts, e.g. 100k,300k,1M,3M (batch mode)")
    ap.add_argument("--out-dir", default="./exports/")
    ap.add_argument("--name", default=None,
                    help="output filename, single mode (default: ckpt stem + .bin)")
    ap.add_argument("--name-prefix", default="selfplay-",
                    help="output filename prefix, batch mode (default: selfplay-)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.checkpoint:
        name = args.name or (os.path.splitext(os.path.basename(args.checkpoint))[0] + ".bin")
        out_path = os.path.join(args.out_dir, name)
        size = export_checkpoint(args.checkpoint, out_path)
        print(f"wrote {out_path} ({size} bytes)")
        return

    if not args.targets:
        raise SystemExit("--run-dir requires --targets")
    seen = set()
    entries = []
    for t in args.targets.split(","):
        target = parse_target(t)
        steps, ckpt = nearest_snapshot(args.run_dir, target)
        if ckpt in seen:
            print(f"skip {t}: nearest {os.path.basename(ckpt)} already exported")
            continue
        seen.add(ckpt)
        label = label_for(steps)
        name = f"{args.name_prefix}{label}.bin"
        out_path = os.path.join(args.out_dir, name)
        size = export_checkpoint(ckpt, out_path)
        print(f"wrote {out_path} ({size} bytes)  <- {os.path.basename(ckpt)}")
        entries.append({
            "file": f"weights/{name}",
            "label": label,
            "steps": steps,
        })
    entries.sort(key=lambda e: e["steps"])
    print("\nmanifest entries (paste under bots.<timeline>.stops):")
    for e in entries:
        print("    " + json.dumps(e) + ",")


if __name__ == "__main__":
    main()
