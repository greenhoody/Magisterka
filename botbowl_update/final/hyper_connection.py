from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class HyperConnection(nn.Module):
    """
    Static hyper-connections (HC) based on arXiv:2409.19606.

    This module operates on a hyper hidden tensor H shaped as:
      - [B, N, C, H, W] for CNN feature maps
      - [B, N, D] for vectors

    It produces:
      - mix_h: [B, N+1, ...] (first slot is h0, remaining are width-connected H')
      - beta:  [N] (depth connection weights for the layer output)
    """

    def __init__(self, rate: int, layer_id: int, dim: int, dynamic: bool = False):
        super().__init__()
        self.rate = rate
        self.layer_id = layer_id
        self.dynamic = dynamic

        # Static beta (B in the paper): initialized to ones.
        self.static_beta = nn.Parameter(torch.ones(rate))

        # Static alpha (Am | Ar in the paper): [rate, rate+1]
        init_alpha0 = torch.zeros(rate, 1)
        init_alpha0[layer_id % rate, 0] = 1.0
        init_alpha = torch.cat([init_alpha0, torch.eye(rate)], dim=1)
        self.static_alpha = nn.Parameter(init_alpha)

        if self.dynamic:
            self.dynamic_alpha_fn = nn.Parameter(torch.zeros(dim, rate + 1))
            self.dynamic_alpha_scale = nn.Parameter(torch.ones(1) * 0.01)
            self.dynamic_beta_fn = nn.Parameter(torch.zeros(dim))
            self.dynamic_beta_scale = nn.Parameter(torch.ones(1) * 0.01)

    def _pool_for_dynamic(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, N, C, H, W] or [B, N, D]
        if h.dim() == 5:
            return h.mean(dim=(-2, -1))
        if h.dim() == 3:
            return h
        raise ValueError(f"Unsupported hyper hidden shape: {h.shape}")

    def width_connection(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # h: [B, N, ...] -> mix_h: [B, N+1, ...]
        if self.dynamic:
            pooled = self._pool_for_dynamic(h)
            wc_weight = torch.tanh(pooled @ self.dynamic_alpha_fn)
            dynamic_alpha = wc_weight * self.dynamic_alpha_scale
            alpha = dynamic_alpha + self.static_alpha[None, ...]
            dc_weight = torch.tanh(pooled @ self.dynamic_beta_fn)
            dynamic_beta = dc_weight * self.dynamic_beta_scale
            beta = dynamic_beta + self.static_beta[None, ...]
            mix_h = torch.einsum("bnm,bm...->bn...", alpha.transpose(-1, -2), h)
            return mix_h, beta

        alpha = self.static_alpha  # [N, N+1]
        mix_h = torch.einsum("nm,bm...->bn...", alpha.t(), h)
        return mix_h, self.static_beta

    def depth_connection(
        self, mix_h: torch.Tensor, h_out: torch.Tensor, beta: torch.Tensor
    ) -> torch.Tensor:
        # mix_h: [B, N+1, ...]; h_out: [B, ...]; beta: [N]
        h_prime = mix_h[:, 1:, ...]
        dims_excluding_batch = h_out.dim() - 1
        if beta.dim() == 1:
            beta_shape = (1, self.rate) + (1,) * dims_excluding_batch
        else:
            beta_shape = (beta.shape[0], self.rate) + (1,) * dims_excluding_batch
        scaled = h_out.unsqueeze(1) * beta.view(*beta_shape)
        return scaled + h_prime


class HyperConnectionStack(nn.Module):
    """
    Wrap a stack of blocks with hyper-connections.
    """

    def __init__(self, blocks: nn.ModuleList, rate: int, dim: int, dynamic: bool = False):
        super().__init__()
        self.blocks = blocks
        self.rate = rate
        self.dim = dim
        self.dynamic = dynamic
        self.hc_layers = nn.ModuleList(
            [
                HyperConnection(rate=rate, layer_id=i, dim=dim, dynamic=dynamic)
                for i in range(len(blocks))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expand to hyper hidden matrix: [B, N, C, H, W]
        h = x.unsqueeze(1).expand(-1, self.rate, -1, -1, -1).contiguous()
        for block, hc in zip(self.blocks, self.hc_layers):
            mix_h, beta = hc.width_connection(h)
            h0 = mix_h[:, 0, ...]
            h_out = block(h0)
            h = hc.depth_connection(mix_h, h_out, beta)
        return h.sum(dim=1)
