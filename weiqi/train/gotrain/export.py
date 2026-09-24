"""Export a checkpoint to the locked fp16 weight format (PLAN.md §3).

Flat float16 little-endian binary, no header. Tensor order IS the format —
see gotrain.net.EXPORT_ORDER (single source of truth).

Usage:
    python -m gotrain.export --checkpoint runs/imit_v1/snap_0003000.pt \\
        --out-dir ./exports/ --name imit-003k.bin
"""

import argparse
import os

import numpy as np
import torch

from .net import GoNet, EXPORT_ORDER, ordered_tensors


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help=".pt file")
    ap.add_argument("--out-dir", default="./exports/")
    ap.add_argument("--name", default=None, help="output filename (default: ckpt stem + .bin)")
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model = GoNet()
    model.load_state_dict(sd)
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    name = args.name or (os.path.splitext(os.path.basename(args.checkpoint))[0] + ".bin")
    out_path = os.path.join(args.out_dir, name)
    export_weights(model, out_path)
    size = os.path.getsize(out_path)
    print(f"wrote {out_path} ({size} bytes)")


if __name__ == "__main__":
    main()
