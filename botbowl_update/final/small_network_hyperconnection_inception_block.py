"""
Template for BotBowl A2C/PPO policy networks.

This file documents the minimum interface and constraints required by the
training scripts in this repo. Use it as a starting point for new models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from hyper_connection import HyperConnectionStack


@dataclass
class NetworkRequirements:
    """
    Hard requirements for compatibility with training loops in this repo.

    1) forward(spatial, non_spatial) -> (value, logits)
       - value shape: [B, 1]
       - logits shape: [B, action_space]

    2) act(spatial, non_spatial, action_mask) -> (value, actions)
       - applies action_mask to logits (invalid actions -> -inf)
       - samples actions from masked softmax
       - returns actions shape: [B, 1] or [B]

    3) evaluate_actions(spatial, non_spatial, actions, action_mask)
       -> (action_log_probs, values, dist_entropy)
       - action_log_probs shape: [B, 1]
       - values shape: [B, 1]
       - dist_entropy: scalar

    4) All tensors must be on the same device.
    5) Must be torchscript/serialization friendly (no lambdas in modules).
    """


class SpatialInceptionBlock(nn.Module):
    """
    Parallel convolution branches with different kernel sizes.
    kernels: list of (out_channels, kernel_size)
    """

    def __init__(self, in_ch: int, kernels: Tuple[Tuple[int, int], ...] | list):
        super().__init__()
        branches = []
        for out_ch, ks in kernels:
            pad = ks // 2
            branches.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=ks, padding=pad),
                    nn.ReLU(),
                )
            )
        self.branches = nn.ModuleList(branches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [branch(x) for branch in self.branches]
        return torch.cat(feats, dim=1)


class InceptionBlock(nn.Module):
    """
    Inception transform used inside HyperConnectionStack.
    Projects channels back to in_ch so hyper width stays constant.
    """

    def __init__(self, in_ch: int, kernels: Tuple[Tuple[int, int], ...] | list):
        super().__init__()
        self.inception = SpatialInceptionBlock(in_ch, kernels)
        out_ch = sum(ch for ch, _ in kernels)
        self.proj = None
        if out_ch != in_ch:
            self.proj = nn.Conv2d(out_ch, in_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.inception(x)
        if self.proj is not None:
            y = self.proj(y)
        return F.relu(y)


class CustomPolicy(nn.Module):
    def __init__(
        self,
        spatial_shape: Tuple[int, int, int],
        non_spatial_size: int,
        action_space: int,
        hidden_nodes: int = 512,
    ) -> None:
        super().__init__()
        c, h, w = spatial_shape
        kernels = [(32, 3), (16, 5), (16, 7), (8, 9)]
        spatial_ch = 32 + 16 + 16 + 8

        self.spatial_stem = nn.Sequential(
            nn.Conv2d(c, spatial_ch, kernel_size=1),
            nn.ReLU(),
        )
        self.spatial = HyperConnectionStack(
            nn.ModuleList(
                [
                    InceptionBlock(spatial_ch, kernels),
                    InceptionBlock(spatial_ch, kernels),
                ]
            ),
            rate=2,
            dim=spatial_ch,
            dynamic=False,
        )
        self.non_spatial = nn.Sequential(
            nn.Linear(non_spatial_size, non_spatial_size),
            nn.ReLU(),
            nn.Linear(non_spatial_size, non_spatial_size),
            nn.ReLU(),
        )

        trunk_in = spatial_ch * h * w + non_spatial_size
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, hidden_nodes),
            nn.ReLU(),
        )

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(hidden_nodes, hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes, hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes, action_space),
        )

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_nodes, hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, spatial: torch.Tensor, non_spatial: torch.Tensor):
        # spatial: [B, C, H, W], non_spatial: [B, 1, N] or [B, N]
        if non_spatial.dim() == 3:
            non_spatial = non_spatial.squeeze(1)

        x = self.spatial_stem(spatial)
        x = self.spatial(x)
        spatial_feat = x.flatten(1)
        non_spatial_feat = self.non_spatial(non_spatial)
        z = torch.cat([spatial_feat, non_spatial_feat], dim=1)
        z = self.trunk(z)

        logits = self.actor(z)
        value = self.critic(z)
        return value, logits

    @staticmethod
    def _masked_probs(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        mask = action_mask.bool()
        if mask.dim() > 2:
            mask = mask.view(mask.shape[0], -1)

        masked_logits = logits.masked_fill(~mask, -1e9)
        probs = F.softmax(masked_logits, dim=1)
        probs = probs * mask.float()

        sums = probs.sum(dim=1, keepdim=True)
        probs = probs / sums.clamp_min(1e-12)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

        # Fallback for degenerate rows (should not happen in BotBowl, but keeps CUDA sampling safe).
        bad_rows = probs.sum(dim=1, keepdim=True) <= 0
        if bad_rows.any():
            fallback = mask.float()
            fallback_sums = fallback.sum(dim=1, keepdim=True)
            no_valid = fallback_sums <= 0
            if no_valid.any():
                fallback[no_valid.expand_as(fallback)] = 1.0
                fallback_sums = fallback.sum(dim=1, keepdim=True)
            fallback = fallback / fallback_sums
            probs = torch.where(bad_rows.expand_as(probs), fallback, probs)
        return probs

    @torch.no_grad()
    def act(
        self,
        spatial: torch.Tensor,
        non_spatial: torch.Tensor,
        action_mask: torch.Tensor,
    ):
        value, logits = self.forward(spatial, non_spatial)
        probs = self._masked_probs(logits, action_mask)
        actions = probs.multinomial(num_samples=1)
        return value, actions

    def evaluate_actions(
        self,
        spatial: torch.Tensor,
        non_spatial: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor,
    ):
        value, logits = self.forward(spatial, non_spatial)
        probs = self._masked_probs(logits, action_mask)
        log_probs = torch.log(probs.clamp_min(1e-12))
        action_log_probs = log_probs.gather(1, actions.long())
        dist_entropy = -(log_probs * probs).sum(1).mean()
        return action_log_probs, value, dist_entropy
