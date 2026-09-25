"""Shared 9x9 Go network — LOCKED spec, see PLAN.md §3.

Any change here must be mirrored in weiqi/engine (Rust inference) and the fp16
export order below. Architecture is the controlled variable of the whole
nurture-vs-nature comparison (PLAN.md §3, "Architecture discipline").
"""

import torch
import torch.nn as nn

# Exact fp16 export order (PLAN.md §3). (state_dict key, shape). This list IS the
# file format: flat float16 little-endian, no header, tensors written back to back
# in this order, each in C-contiguous (row-major) layout.
EXPORT_ORDER = [
    ("conv1.weight", (64, 6, 3, 3)),
    ("conv1.bias", (64,)),
    ("conv2.weight", (64, 64, 3, 3)),
    ("conv2.bias", (64,)),
    ("conv3.weight", (64, 64, 3, 3)),
    ("conv3.bias", (64,)),
    ("conv4.weight", (64, 64, 3, 3)),
    ("conv4.bias", (64,)),
    ("pol_conv.weight", (2, 64, 1, 1)),
    ("pol_conv.bias", (2,)),
    ("pol_fc.weight", (82, 162)),
    ("pol_fc.bias", (82,)),
    ("val_conv.weight", (1, 64, 1, 1)),
    ("val_conv.bias", (1,)),
    ("val_fc1.weight", (32, 81)),
    ("val_fc1.bias", (32,)),
    ("val_fc2.weight", (1, 32)),
    ("val_fc2.bias", (1,)),
]

# Locked total parameter count (PLAN.md §3). Asserted by tests/test_smoke.py.
EXPECTED_PARAMS = 130522

# Move index 81 = pass; 0..80 = row-major board points (row 0 = top, col 0 = left).
PASS_INDEX = 81
N_POINTS = 81
N_MOVES = 82


class GoNet(nn.Module):
    """6-plane input -> policy logits (82) + value scalar in [-1, 1].

    Trunk: 4x conv 64ch 3x3 pad1 ReLU, no batchnorm (keeps WASM inference simple).
    Policy head: conv 64->2 (1x1) -> flatten -> linear 162->82.
    Value head:  conv 64->1 (1x1) -> flatten -> linear 81->32 -> ReLU -> linear 32->1 -> tanh.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pol_conv = nn.Conv2d(64, 2, kernel_size=1)
        self.pol_fc = nn.Linear(162, 82)
        self.val_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.val_fc1 = nn.Linear(81, 32)
        self.val_fc2 = nn.Linear(32, 1)

    def features(self, x):
        """Trunk feature map (B,64,9,9) after conv4, before the heads.

        Param-neutral and export-neutral: adds no parameters and leaves
        EXPORT_ORDER untouched, so the locked spec (PLAN.md §3) still holds.
        Exists so training-only auxiliary heads (gotrain.net_aux) can read
        the trunk without duplicating it; Rust inference is unaffected.
        """
        t = torch.relu(self.conv1(x))
        t = torch.relu(self.conv2(t))
        t = torch.relu(self.conv3(t))
        return torch.relu(self.conv4(t))

    def forward(self, x):
        t = self.features(x)
        logits = self.pol_fc(self.pol_conv(t).flatten(1))
        v = torch.relu(self.val_fc1(self.val_conv(t).flatten(1)))
        value = torch.tanh(self.val_fc2(v)).squeeze(1)
        return logits, value

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


def ordered_tensors(model):
    """Return model tensors as a list in EXPORT_ORDER, verifying shapes."""
    sd = model.state_dict()
    out = []
    for key, shape in EXPORT_ORDER:
        t = sd[key]
        if tuple(t.shape) != shape:
            raise ValueError(f"export order mismatch for {key}: {tuple(t.shape)} != {shape}")
        out.append(t.detach().contiguous())
    return out
