from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    """Layer normalization that operates on NCHW tensors by normalizing channels."""

    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.num_channels = num_channels
        self.layer_norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Permute to NHWC so that LayerNorm can act on the channel dimension.
        x = x.permute(0, 2, 3, 1)
        x = self.layer_norm(x)
        return x.permute(0, 3, 1, 2)

    @property
    def weight(self) -> torch.nn.Parameter:
        return self.layer_norm.weight

    @property
    def bias(self) -> torch.nn.Parameter:
        return self.layer_norm.bias
