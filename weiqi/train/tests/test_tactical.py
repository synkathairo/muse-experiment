"""Tests for the experimental tactical input planes (gotrain.tactical).

Covers: liberty-plane correctness on hand-constructed positions (own/opponent
perspective, 1/2/>=3 buckets, ko encoding), planes 0-5 matching the locked
spec, parameter count, zero-init checkpoint branching equivalence, tactical
env observations, and the tactical export order.
"""

import os

import numpy as np
import pytest
import torch

from gotrain import rules, tactical
from gotrain.net import (GoNet, GoNetTactical, EXPECTED_PARAMS_TACTICAL,
                         ordered_tensors_tactical)
from gotrain.rules import Board, BLACK, WHITE
from gotrain.selfplay import SelfPlayGo, legal_mask
from gotrain import features


def play_seq(moves):
    """Play an alternating B/W sequence on a fresh 9x9 board."""
    b = Board(9)
    for i, (r, c) in enumerate(moves):
        assert b.play((r, c), BLACK if i % 2 == 0 else WHITE), f"illegal at {i}"
    return b


def test_empty_board_tactical_planes_zero():
    b = Board(9)
    p = tactical.encode_tactical(b, BLACK)
    assert p.shape == (13, 9, 9)
    assert p.dtype == np.float32
    assert p[0].sum() == 0 and p[1].sum() == 0
    assert p[2].sum() == 81  # all empty
    assert p[3].sum() == 0 and p[5].sum() == 0
    assert p[6:13].sum() == 0  # no stones => no liberty planes, no ko
    assert p[4].sum() == 81  # black to move


def test_liberty_buckets_and_perspective():
    # B(0,0) corner stone, W(0,1): black stone has exactly 1 liberty (1,0);
    # white stone has exactly 2 liberties (0,2),(1,1).
    b = play_seq([(0, 0), (0, 1)])
    p = tactical.encode_tactical(b, BLACK)  # black to move
    assert p[6, 0, 0] == 1.0   # own, 1 liberty (in atari)
    assert p[10, 0, 1] == 1.0  # opp, 2 liberties
    assert p[6].sum() == 1 and p[10].sum() == 1
    assert p[7].sum() == 0 and p[8].sum() == 0
    assert p[9].sum() == 0 and p[11].sum() == 0

    # White's perspective: buckets swap sides.
    q = tactical.encode_tactical(b, WHITE)
    assert q[7, 0, 1] == 1.0   # white's own stone, 2 liberties
    assert q[9, 0, 0] == 1.0   # black stone now opponent, 1 liberty
    assert q[4].sum() == 0     # white to move => plane 4 empty
    assert q[6].sum() == 0 and q[10].sum() == 0


def test_liberty_bucket_ge3():
    # Lone center stone: 4 liberties.
    b = play_seq([(4, 4)])
    p = tactical.encode_tactical(b, BLACK)
    assert p[8, 4, 4] == 1.0
    assert p[6].sum() == 0 and p[7].sum() == 0
    # Group of two shares the bucket: B(4,4)+B(4,5) have 6 liberties.
    b2 = play_seq([(4, 4), (0, 0), (4, 5)])
    p2 = tactical.encode_tactical(b2, BLACK)
    assert p2[8, 4, 4] == 1.0 and p2[8, 4, 5] == 1.0
    assert p2[8].sum() == 2


def test_liberty_map_matches_planes():
    # liberty_map: exact count per stone, 0 on empty.
    b = play_seq([(0, 0), (0, 1), (4, 4)])
    lm = b.liberty_map()
    assert lm.shape == (9, 9)
    assert lm[0, 0] == 1
    assert lm[0, 1] == 2
    assert lm[4, 4] == 4
    assert lm[1, 1] == 0


def test_ko_planes():
    # Build a simple ko: B captures the lone W(4,4) at (5,4); the capturing
    # stone is a lone single stone with one liberty, so ko=(4,4).
    b = play_seq([(3, 4), (5, 3), (4, 3), (5, 5), (4, 5), (6, 4),
                  (0, 0), (4, 4), (5, 4)])
    assert b.ko == (4, 4), f"ko setup failed, ko={b.ko}"
    p = tactical.encode_tactical(b, WHITE)  # white to move
    assert p[5, 4, 4] == 1.0 and p[5].sum() == 1.0   # locked ko plane
    assert p[12, 4, 4] == 1.0 and p[12].sum() == 1.0  # tactical ko plane
    # The ko-banned point is not a legal move for white.
    assert not legal_mask(b, WHITE)[4 * 9 + 4]


def test_planes_0_to_5_match_locked_spec():
    b = play_seq([(3, 3), (4, 4), (3, 4), (5, 5)])
    for color in (BLACK, WHITE):
        pt = tactical.encode_tactical(b, color)
        pl = features.encode(
            b.stones(color), b.stones(rules.opponent(color)),
            last_move=b.last_move,
            black_to_move=(color == BLACK), ko_point=b.ko)
        np.testing.assert_array_equal(pt[:6], pl)


def test_tactical_param_count():
    m = GoNetTactical()
    assert m.param_count() == EXPECTED_PARAMS_TACTICAL == 134554
    assert tuple(m.conv1.weight.shape) == (64, 13, 3, 3)
    # everything past conv1 is untouched vs the locked net
    g = GoNet()
    for k in g.state_dict():
        if k.startswith("conv1"):
            continue
        assert tuple(m.state_dict()[k].shape) == tuple(g.state_dict()[k].shape)


def test_branch_zero_init_equivalence(tmp_path):
    src = tactical.BASE_CHECKPOINT
    if not os.path.exists(src):
        pytest.skip("15.5M baseline checkpoint not present")
    dst = str(tmp_path / "branch.pt")
    _, branched = tactical.branch_checkpoint(src, dst, seed=7,
                                             total_steps=16003360)
    sd = torch.load(src, map_location="cpu", weights_only=False)
    from gotrain.net_aux import to_gonet_state_dict
    src_sd = to_gonet_state_dict(sd["model"])
    bw = branched.state_dict()["conv1.weight"]
    # existing channels copied, new channels exactly zero
    torch.testing.assert_close(bw[:, :6], src_sd["conv1.weight"])
    assert (bw[:, 6:] == 0).all()
    # every other parameter copied exactly
    for k, v in src_sd.items():
        if k == "conv1.weight":
            continue
        torch.testing.assert_close(branched.state_dict()[k], v)

    # policy equivalence on real positions (new channels contribute 0)
    base = GoNet()
    base.load_state_dict(src_sd)
    base.eval()
    branched.eval()
    rng = np.random.default_rng(0)
    boards = []
    for _ in range(6):
        b = Board(9)
        col = BLACK
        for _ in range(int(rng.integers(5, 60))):
            mv = legal_mask(b, col)
            idx = int(rng.choice(np.flatnonzero(mv)))
            assert b.play(None if idx == 81 else divmod(idx, 9), col)
            col = rules.opponent(col)
        boards.append((b, col))
    with torch.no_grad():
        for b, col in boards:
            l6 = base(torch.from_numpy(
                features.encode(b.stones(col), b.stones(rules.opponent(col)),
                                last_move=b.last_move,
                                black_to_move=(col == BLACK),
                                ko_point=b.ko)).unsqueeze(0).float())
            l13 = branched(torch.from_numpy(
                tactical.encode_tactical(b, col)).unsqueeze(0))
            torch.testing.assert_close(l13[0], l6[0], atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(l13[1], l6[1], atol=1e-5, rtol=1e-4)

    # branched checkpoint is a complete training checkpoint
    ck = torch.load(dst, map_location="cpu", weights_only=False)
    for k in ("step", "ppo_iter", "snap_ptr", "model", "opponent",
              "optimizer", "torch_rng", "numpy_rng", "hparams"):
        assert k in ck, k
    assert ck["hparams"]["tactical"] is True
    assert ck["step"] == 15503360
    assert tuple(ck["model"]["conv1.weight"].shape) == (64, 13, 3, 3)


def test_tactical_env_obs_shape():
    env = SelfPlayGo(num_envs=2, seed=1, tactical=True)
    assert env.n_planes == 13
    obs = env.reset()
    assert obs.shape == (2, 13, 9, 9)
    masks = env.legal_masks_learner()
    acts = np.array([int(np.flatnonzero(m)[0]) for m in masks])
    obs2, _, _, _, _ = env.step(acts)
    assert obs2.shape == (2, 13, 9, 9)
    env6 = SelfPlayGo(num_envs=2, seed=1)
    assert env6.n_planes == 6
    assert env6.reset().shape == (2, 6, 9, 9)


def test_tactical_export_order(tmp_path):
    from gotrain.export import export_weights_tactical
    from gotrain.net import EXPORT_ORDER_TACTICAL
    m = GoNetTactical()
    assert sum(int(np.prod(s)) for _, s in EXPORT_ORDER_TACTICAL) == 134554
    assert len(ordered_tensors_tactical(m)) == len(EXPORT_ORDER_TACTICAL)
    out = str(tmp_path / "tac.bin")
    export_weights_tactical(m, out)
    assert os.path.getsize(out) == 134554 * 2
