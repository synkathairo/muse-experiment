"""Tactical input planes — EXPERIMENTAL (phase 1: Python training only).

Hypothesis: the 130K net wastes capacity computing group liberties through
convolutions; handing it tactical state directly tests whether the strength
wall is representation rather than capacity.

13 planes x 9x9, float32, row-major, side-to-move perspective. Planes 0-5 are
IDENTICAL to gotrain.features.encode (the locked 6-plane spec); planes 6-12
are new:

Plane  0: own stones (side to move)            [locked]
Plane  1: opponent stones                     [locked]
Plane  2: empty points                        [locked]
Plane  3: last move (1.0 at last played point) [locked]
Plane  4: color to play (all 1.0 if Black)    [locked]
Plane  5: ko-banned point                     [locked]
Plane  6: own stones whose group has exactly 1 liberty (own stones in atari)
Plane  7: own stones whose group has exactly 2 liberties
Plane  8: own stones whose group has >= 3 liberties
Plane  9: opponent stones whose group has exactly 1 liberty (capturable now)
Plane 10: opponent stones whose group has exactly 2 liberties
Plane 11: opponent stones whose group has >= 3 liberties
Plane 12: ko-prohibition point (same semantics as plane 5; duplicated so the
          tactical block is self-contained)

Liberty counts come from Board.liberty_map() — exact, deterministic.

Checkpoint branching: a 6-plane GoNet checkpoint maps into GoNetTactical by
copying the 6 existing input-channel weights and ZERO-initializing the 7 new
ones, so the branched policy is bit-identical to the source until training
touches the new channels (verified by tests/test_tactical.py).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from . import rules
from .net import GoNetTactical

N_TAC_PLANES: int = 13
N_NEW_PLANES: int = 7  # planes 6..12

# 15.5M baseline checkpoint (full training checkpoint, plain GoNet).
BASE_CHECKPOINT: str = "/home/hatch/workspace/user/files/latest_smoke_15m.pt"
BASE_STEP: int = 15503360


def encode_tactical(board: rules.Board, color: int) -> np.ndarray:
    """(13,9,9) float32 planes from `color`-to-move's perspective."""
    planes = np.zeros((N_TAC_PLANES, 9, 9), dtype=np.float32)
    own = board.stones(color)
    opp = board.stones(rules.opponent(color))
    planes[0] = own
    planes[1] = opp
    planes[2] = ~(own | opp)
    if board.last_move is not None:
        planes[3, board.last_move[0], board.last_move[1]] = 1.0
    if color == rules.BLACK:
        planes[4, :, :] = 1.0
    if board.ko is not None:
        planes[5, board.ko[0], board.ko[1]] = 1.0
        planes[12, board.ko[0], board.ko[1]] = 1.0
    libs = board.liberty_map()
    planes[6] = own & (libs == 1)
    planes[7] = own & (libs == 2)
    planes[8] = own & (libs >= 3)
    planes[9] = opp & (libs == 1)
    planes[10] = opp & (libs == 2)
    planes[11] = opp & (libs >= 3)
    return planes


def _migrate_adam_state(src_opt_sd: dict | None, src_sd: dict,
                        model: torch.nn.Module,
                        opt: torch.optim.Optimizer) -> None:
    """Copy Adam moments from a source optimizer state dict into `opt`.

    `src_sd` is the source model's state dict (ordered); `model` is the
    target module. Parameters are matched by name in state-dict order.
    For `conv1.weight` (6->13 in-channels) the 6 old channels' moments are
    copied and the 7 new channels' moments are zero-initialized. Parameters
    with no source moments (or shape mismatches beyond conv1) start fresh.
    """
    if not src_opt_sd or "state" not in src_opt_sd:
        return
    src_state = src_opt_sd["state"]
    src_names = list(src_sd.keys())
    tgt_names = list(model.state_dict().keys())
    # Map source param index -> target param index via name order. The
    # optimizer state keys are indices into the source model's parameters()
    # order, which matches state_dict() order for these modules.
    tgt_index = {n: i for i, n in enumerate(tgt_names)}
    new_state: dict[int, dict[str, Any]] = {}
    for src_idx, name in enumerate(src_names):
        if src_idx not in src_state:
            continue
        tgt_idx = tgt_index.get(name)
        if tgt_idx is None:
            continue
        st = src_state[src_idx]
        if name == "conv1.weight":
            # Expand moments from 6 to 13 input channels.
            migrated: dict[str, Any] = {}
            for k, v in st.items():
                if isinstance(v, torch.Tensor) and v.shape == src_sd[name].shape:
                    nv = torch.zeros_like(model.state_dict()[name])
                    nv[:, :6].copy_(v)
                    migrated[k] = nv
                else:
                    migrated[k] = v.clone() if isinstance(v, torch.Tensor) else v
            new_state[tgt_idx] = migrated
        else:
            # Copy only if shapes match; otherwise start fresh.
            ok = True
            for k, v in st.items():
                if isinstance(v, torch.Tensor):
                    tgt_shape = model.state_dict()[name].shape
                    if v.shape != tgt_shape and k in ("exp_avg", "exp_avg_sq"):
                        ok = False
                        break
            if ok:
                new_state[tgt_idx] = {
                    k: (v.clone() if isinstance(v, torch.Tensor) else v)
                    for k, v in st.items()
                }
    opt.load_state_dict({"state": new_state,
                         "param_groups": opt.state_dict()["param_groups"]})


def branch_checkpoint(src_path: str = BASE_CHECKPOINT,
                      dst_path: str | None = None, seed: int = 7,
                      total_steps: int | None = None,
                      lr: float = 1e-4) -> tuple[str, GoNetTactical]:
    """Branch a 6-plane GoNet training checkpoint into a 13-plane training
    checkpoint for GoNetTactical, zero-initializing the 7 new input channels.

    Produces a complete training checkpoint (model + opponent snapshot +
    migrated Adam state + RNG + hparams with tactical=True) that
    train_selfplay.py --resume can consume directly. The branched policy is
    identical to the source policy until training updates the new channels.
    Adam moments are migrated for all trunk parameters (conv1's 6 old
    input channels keep their moments; the 7 new channels start at zero),
    so the branch continues optimization rather than restarting it.
    """
    import os
    from .net_aux import to_gonet_state_dict

    ck = torch.load(src_path, map_location="cpu", weights_only=False)
    src_sd = to_gonet_state_dict(ck["model"])
    if src_sd["conv1.weight"].shape[1] != 6:
        raise ValueError("branch source must be a 6-plane GoNet checkpoint, got "
                         f"{tuple(src_sd['conv1.weight'].shape)}")

    model = GoNetTactical()
    with torch.no_grad():
        msd = model.state_dict()
        for k, v in src_sd.items():
            if k not in msd:
                raise RuntimeError(f"source key {k!r} has no target in GoNetTactical")
            if k == "conv1.weight":
                msd[k][:, :6].copy_(v)
                msd[k][:, 6:].zero_()
            else:
                if msd[k].shape != v.shape:
                    raise RuntimeError(
                        f"shape mismatch for {k}: {tuple(v.shape)} vs {tuple(msd[k].shape)}")
                msd[k].copy_(v)

    step = int(ck.get("step", BASE_STEP))
    ppo_iter = int(ck.get("ppo_iter", 0))
    if total_steps is None:
        total_steps = step + 500_000

    hparams = dict(
        num_envs=32, total_steps=total_steps, rollout_steps=128,
        minibatch_size=256, update_epochs=4, lr=lr,
        gamma=0.99, gae_lambda=0.95, clip_coef=0.2, vf_coef=0.5,
        ent_coef=0.03, max_grad_norm=0.5,
        dirichlet_plies=12, dirichlet_alpha=0.05, dirichlet_eps=0.25,
        ownership=False, aux_own_w=0.5, aux_margin_w=0.5, aux_epochs=2,
        opp_refresh_every=10, eval_every=20, eval_games=20, ckpt_every=10,
        max_plies=243, train_komi=6.5, seed=seed,
        reward_mode="winloss", reward_scale=15.0, device="auto",
        tactical=True, no_anneal_lr=True,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # Migrate Adam moments from the source checkpoint so the branch continues
    # optimization instead of restarting it. Parameter order matches between
    # GoNet and GoNetTactical (identical trunk); only conv1.weight changes
    # shape (6->13 in-channels). Its 6 old channels keep their moments, the
    # 7 new channels start at zero.
    _migrate_adam_state(ck.get("optimizer"), src_sd, model, opt)
    snapshot = GoNetTactical()
    snapshot.load_state_dict(model.state_dict())
    out = {
        "step": step,
        "ppo_iter": ppo_iter,
        "snap_ptr": int(ck.get("snap_ptr", 0)),
        "model": model.state_dict(),
        "opponent": snapshot.state_dict(),
        "optimizer": opt.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "hparams": hparams,
    }
    if dst_path is None:
        dst_path = os.path.join(os.path.dirname(src_path) or ".",
                                "tactical_branch.pt")
    torch.save(out, dst_path)
    return dst_path, model


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=BASE_CHECKPOINT)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-4)
    a = ap.parse_args()
    dst, _ = branch_checkpoint(a.src, a.dst, seed=a.seed,
                               total_steps=a.total_steps, lr=a.lr)
    print(f"branched checkpoint -> {dst}")
