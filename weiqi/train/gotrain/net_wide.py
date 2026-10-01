"""Experimental wider / global-pooling Go net variants for the capacity diagnostic.

These are NOT part of the locked spec (PLAN.md §3). They exist only for the
Astra-recommended 2x2 diagnostic: does widening help (capacity hypothesis),
does global pooling help (architecture hypothesis), or both/neither?

- GoNetWide: same 4-layer CNN as GoNet, 128 channels instead of 64 (~466K params)
- GoNetPool: GoNet + global pooling path (GAP -> FC -> broadcast -> concat -> 1x1 mix + ReLU)
- GoNetWidePool: both widened and pooled

Per Astra: keep depth at 4 layers (don't confound with depth changes).
The global pooling MUST include a nonlinearity after mixing (mix_1x1 + ReLU),
otherwise the global vector is just a linear offset.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GoNetWide(nn.Module):
    """128-channel variant of GoNet (~466K params). Tests capacity hypothesis."""

    def __init__(self, in_planes: int = 6, channels: int = 128) -> None:
        super().__init__()
        self.channels: int = channels
        self.conv1: nn.Conv2d = nn.Conv2d(in_planes, channels, kernel_size=3, padding=1)
        self.conv2: nn.Conv2d = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv3: nn.Conv2d = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv4: nn.Conv2d = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.pol_conv: nn.Conv2d = nn.Conv2d(channels, 2, kernel_size=1)
        self.pol_fc: nn.Linear = nn.Linear(162, 82)
        self.val_conv: nn.Conv2d = nn.Conv2d(channels, 1, kernel_size=1)
        self.val_fc1: nn.Linear = nn.Linear(81, 32)
        self.val_fc2: nn.Linear = nn.Linear(32, 1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        t = torch.relu(self.conv1(x))
        t = torch.relu(self.conv2(t))
        t = torch.relu(self.conv3(t))
        return torch.relu(self.conv4(t))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.features(x)
        logits = self.pol_fc(self.pol_conv(t).flatten(1))
        v = torch.relu(self.val_fc1(self.val_conv(t).flatten(1)))
        value = torch.tanh(self.val_fc2(v)).squeeze(1)
        return logits, value

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class GoNetPool(nn.Module):
    """GoNet + global pooling path. Tests global-context hypothesis.

    After the conv trunk: global average pool -> FC -> ReLU -> broadcast to
    9x9 -> concat with trunk features -> 1x1 conv + ReLU to mix. The mixing
    layer is critical: without it, the global vector is just a linear offset
    (the heads after the trunk are linear).
    """

    def __init__(self, in_planes: int = 6, channels: int = 64,
                 global_dim: int = 32) -> None:
        super().__init__()
        self.channels: int = channels
        self.global_dim: int = global_dim
        self.conv1: nn.Conv2d = nn.Conv2d(in_planes, channels, kernel_size=3, padding=1)
        self.conv2: nn.Conv2d = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv3: nn.Conv2d = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv4: nn.Conv2d = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        # Global path: GAP -> FC -> ReLU
        self.global_fc: nn.Linear = nn.Linear(channels, global_dim)
        # Mix local + global: 1x1 conv on concatenated features + ReLU
        self.mix_conv: nn.Conv2d = nn.Conv2d(channels + global_dim, channels, kernel_size=1)
        self.pol_conv: nn.Conv2d = nn.Conv2d(channels, 2, kernel_size=1)
        self.pol_fc: nn.Linear = nn.Linear(162, 82)
        self.val_conv: nn.Conv2d = nn.Conv2d(channels, 1, kernel_size=1)
        self.val_fc1: nn.Linear = nn.Linear(81, 32)
        self.val_fc2: nn.Linear = nn.Linear(32, 1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        t = torch.relu(self.conv1(x))
        t = torch.relu(self.conv2(t))
        t = torch.relu(self.conv3(t))
        t = torch.relu(self.conv4(t))
        # Global path
        g = torch.relu(self.global_fc(t.mean(dim=(2, 3))))  # (B, global_dim)
        g = g[:, :, None, None].expand(-1, -1, 9, 9)  # (B, global_dim, 9, 9)
        t = torch.relu(self.mix_conv(torch.cat([t, g], dim=1)))
        return t

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.features(x)
        logits = self.pol_fc(self.pol_conv(t).flatten(1))
        v = torch.relu(self.val_fc1(self.val_conv(t).flatten(1)))
        value = torch.tanh(self.val_fc2(v)).squeeze(1)
        return logits, value

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class GoNetWidePool(GoNetPool):
    """128-channel + global pooling. Tests both hypotheses together."""

    def __init__(self, in_planes: int = 6, channels: int = 128,
                 global_dim: int = 32) -> None:
        super().__init__(in_planes=in_planes, channels=channels,
                         global_dim=global_dim)


# Expected param counts (for tests)
EXPECTED_WIDE: int = 466000  # approximate; exact computed in test
EXPECTED_POOL: int = 135000  # approximate; exact computed in test
