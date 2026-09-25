"""Training-only auxiliary heads for the Autodidact leg (PLAN.md §5).

KataGo's recipe — D. Wu, "Accelerating Self-Play Learning in Go"
(https://arxiv.org/abs/1912.02414): win/loss is a sparse, high-variance
signal. Predicting per-point ownership and final score margin gives the
trunk dense spatial feedback every game and a much better-informed
value head.

Design discipline:
  - gotrain.net.GoNet stays LOCKED (params, fp16 export order, Rust
    inference all untouched). GoNetAux wraps it as `self.trunk` and adds
    the two heads; the demo/export path never sees them.
  - forward(x) returns (logits, value) exactly like GoNet, so every
    existing call site (rollouts, eval, GTP engines) works unchanged.
    forward_aux(x) returns (logits, value, own, margin) for the
    auxiliary update.
  - Ownership target: +1 point owned by the side to move, -1 by the
    opponent, 0 neutral (gotrain.selfplay.ownership_labels, converted
    with to_learner_perspective). The head output is tanh-squashed and
    trained with MSE — the same bounded style as the value head.
  - Margin target: (my_score - opp_score) / 81, raw scalar head, MSE.
"""

import torch
import torch.nn as nn

from .net import GoNet, N_POINTS

TRUNK_PREFIX = "trunk."

# Auxiliary head parameter names (used to verify resume-mapping strictness).
AUX_KEYS = {
    "own_conv.weight", "own_conv.bias",
    "margin_conv.weight", "margin_conv.bias",
    "margin_fc1.weight", "margin_fc1.bias",
    "margin_fc2.weight", "margin_fc2.bias",
}


class GoNetAux(nn.Module):
    """GoNet trunk + ownership (81) + score-margin (1) heads. Training only."""

    def __init__(self):
        super().__init__()
        self.trunk = GoNet()
        # ownership head: 1x1 conv -> 81 tanh-squashed ownership scores
        self.own_conv = nn.Conv2d(64, 1, kernel_size=1)
        # margin head mirrors the value head, but no tanh (raw score)
        self.margin_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.margin_fc1 = nn.Linear(N_POINTS, 32)
        self.margin_fc2 = nn.Linear(32, 1)

    def _forward_all(self, x):
        f = self.trunk.features(x)
        logits = self.trunk.pol_fc(self.trunk.pol_conv(f).flatten(1))
        v = torch.relu(self.trunk.val_fc1(self.trunk.val_conv(f).flatten(1)))
        value = torch.tanh(self.trunk.val_fc2(v)).squeeze(1)
        own = torch.tanh(self.own_conv(f).flatten(1))
        m = torch.relu(self.margin_fc1(self.margin_conv(f).flatten(1)))
        margin = self.margin_fc2(m).squeeze(1)
        return logits, value, own, margin

    def forward(self, x):
        logits, value, _, _ = self._forward_all(x)
        return logits, value

    def forward_aux(self, x):
        """(logits, value, own, margin) for the auxiliary loss update."""
        return self._forward_all(x)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


def to_gonet_state_dict(state_dict):
    """GoNetAux-shaped state dict -> plain GoNet state dict (trunk weights).

    Passes plain GoNet dicts through unchanged, so engine entry points
    (gtp.py, export.py, mcts_gtp.py) can load either checkpoint flavor.
    """
    if any(k.startswith(TRUNK_PREFIX) for k in state_dict):
        return {k[len(TRUNK_PREFIX):]: v for k, v in state_dict.items()
                if k.startswith(TRUNK_PREFIX)}
    return state_dict


def load_trunk_from_gonet(aux_model, gonet_state_dict):
    """Initialize an aux model's trunk from a plain GoNet checkpoint
    (e.g. the 15.5M run); aux heads keep their random init.

    Returns the (missing_keys, unexpected_keys) from the non-strict load;
    the caller should verify the missing keys are exactly the aux heads'.
    """
    prefixed = {TRUNK_PREFIX + k: v for k, v in gonet_state_dict.items()}
    return aux_model.load_state_dict(prefixed, strict=False)


def check_trunk_resume(missing_keys):
    """Validate that a trunk-mapped resume left ONLY the aux heads unloaded."""
    missing = set(missing_keys)
    unknown = missing - AUX_KEYS
    if unknown:
        raise RuntimeError(
            f"trunk resume left non-aux keys unloaded: {sorted(unknown)}")
