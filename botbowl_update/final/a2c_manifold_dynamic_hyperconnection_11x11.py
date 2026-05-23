"""
A2C/PPO policy for full-team BotBowl with dynamic hyperconnections.

The module keeps the same CustomPolicy interface as the other networks in this
directory, so it can be selected with policy_module in the training scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

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
       - returns actions shape: [B, 1]

    3) evaluate_actions(spatial, non_spatial, actions, action_mask)
       -> (action_log_probs, values, dist_entropy)
       - action_log_probs shape: [B, 1]
       - values shape: [B, 1]
       - dist_entropy: scalar
    """


KernelSpec = Tuple[int, int]
FULL_PITCH_LENGTH = 26
FULL_PITCH_WIDTH = 15
FULL_PITCH_SHAPE = (FULL_PITCH_WIDTH, FULL_PITCH_LENGTH)
RECOMMENDED_HIDDEN_NODES = 318


class ChannelManifoldNorm(nn.Module):
    """
    Project each board cell feature vector onto a learnable-radius sphere.

    This bounds the spatial representation before hyperconnection mixing while
    preserving channel-wise direction information useful for policy decisions.
    """

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.radius = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        centered = x - x.mean(dim=1, keepdim=True)
        norm = centered.pow(2).sum(dim=1, keepdim=True).add(self.eps).sqrt()
        return centered / norm * self.radius + self.bias


class SpatialInceptionBlock(nn.Module):
    """Parallel spatial branches with kernels matched to the BotBowl board."""

    def __init__(self, in_ch: int, kernels: Sequence[KernelSpec]) -> None:
        super().__init__()
        branches = []
        for out_ch, kernel_size in kernels:
            padding = kernel_size // 2
            branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_ch,
                        out_ch,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=padding,
                        bias=True,
                    ),
                    nn.PReLU(num_parameters=out_ch),
                )
            )
        self.branches = nn.ModuleList(branches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [branch(x) for branch in self.branches]
        return torch.cat(feats, dim=1)


class ManifoldInceptionBlock(nn.Module):
    """Inception transform used inside the dynamic HyperConnectionStack."""

    def __init__(
        self,
        channels: int,
        kernels: Sequence[KernelSpec],
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.inception = SpatialInceptionBlock(channels, kernels)
        out_ch = sum(out_ch for out_ch, _ in kernels)
        self.project = nn.Sequential(
            nn.Conv2d(out_ch, channels, kernel_size=1, bias=True),
        )
        self.manifold = ChannelManifoldNorm(channels)
        self.activation = nn.PReLU(num_parameters=channels)
        self.dropout = nn.Dropout2d(dropout)
        self.residual_gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.inception(x)
        y = self.project(y)
        y = self.manifold(y)
        y = self.dropout(y)
        gate = torch.sigmoid(self.residual_gate)
        return self.activation(x + gate * y)


class CustomPolicy(nn.Module):
    recommended_hidden_nodes = RECOMMENDED_HIDDEN_NODES

    def __init__(
        self,
        spatial_shape: Tuple[int, int, int],
        non_spatial_size: int,
        action_space: int,
        hidden_nodes: int = RECOMMENDED_HIDDEN_NODES,
        feature_channels: int = 64,
        non_spatial_embedding: int = 128,
        hyper_rate: int = 3,
        block_kernels: Sequence[Sequence[KernelSpec]] | None = None,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        channels, height, width = spatial_shape
        if block_kernels is None:
            block_kernels = (
                ((24, 3), (20, 5), (20, 7)),
                ((24, 3), (20, 5), (20, 7)),
                ((24, 3), (20, 5), (20, 7)),
                ((24, 3), (20, 5), (20, 7)),
            )

        if feature_channels <= 0:
            raise ValueError("feature_channels must be positive")
        if hyper_rate <= 0:
            raise ValueError("hyper_rate must be positive")
        if not block_kernels or not block_kernels[0]:
            raise ValueError("block_kernels must define at least one inception block")

        self.spatial_shape = spatial_shape
        self.full_pitch_shape = FULL_PITCH_SHAPE
        self.board_positional_bias = nn.Parameter(
            torch.zeros(1, feature_channels, height, width)
        )
        self.spatial_stem = nn.Sequential(
            nn.Conv2d(channels, feature_channels, kernel_size=1, bias=True),
            nn.PReLU(num_parameters=feature_channels),
            ChannelManifoldNorm(feature_channels),
        )

        spatial_blocks = []
        for block_spec in block_kernels:
            if not block_spec:
                raise ValueError("Each inception block must define at least one kernel")
            spatial_blocks.append(
                ManifoldInceptionBlock(
                    channels=feature_channels,
                    kernels=block_spec,
                    dropout=dropout,
                )
            )

        self.spatial = HyperConnectionStack(
            nn.ModuleList(spatial_blocks),
            rate=hyper_rate,
            dim=feature_channels,
            dynamic=True,
        )
        spatial_features = feature_channels * height * width
        self.spatial_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(spatial_features, hidden_nodes),
            nn.PReLU(num_parameters=hidden_nodes),
        )

        self.non_spatial = nn.Sequential(
            nn.Linear(non_spatial_size, non_spatial_embedding),
            nn.PReLU(num_parameters=non_spatial_embedding),
            nn.Linear(non_spatial_embedding, non_spatial_embedding),
            nn.PReLU(num_parameters=non_spatial_embedding),
        )

        fused_features = hidden_nodes + non_spatial_embedding
        self.trunk = nn.Sequential(
            nn.Linear(fused_features, hidden_nodes),
            nn.PReLU(num_parameters=hidden_nodes),
            nn.Dropout(dropout),
        )

        self.actor = nn.Sequential(
            nn.Linear(hidden_nodes, hidden_nodes),
            nn.PReLU(num_parameters=hidden_nodes),
            nn.Linear(hidden_nodes, hidden_nodes),
            nn.PReLU(num_parameters=hidden_nodes),
            nn.Linear(hidden_nodes, action_space),
        )

        self.critic = nn.Sequential(
            nn.Linear(hidden_nodes, hidden_nodes),
            nn.PReLU(num_parameters=hidden_nodes),
            nn.Linear(hidden_nodes, hidden_nodes // 2),
            nn.PReLU(num_parameters=hidden_nodes // 2),
            nn.Linear(hidden_nodes // 2, 1),
        )

    def forward(
        self, spatial: torch.Tensor, non_spatial: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if non_spatial.dim() == 3:
            non_spatial = non_spatial.squeeze(1)

        x = self.spatial_stem(spatial)
        x = x + self.board_positional_bias
        x = self.spatial(x)
        spatial_feat = self.spatial_head(x)
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
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)

        value, logits = self.forward(spatial, non_spatial)
        probs = self._masked_probs(logits, action_mask)
        log_probs = torch.log(probs.clamp_min(1e-12))
        action_log_probs = log_probs.gather(1, actions.long())
        dist_entropy = -(log_probs * probs).sum(1).mean()
        return action_log_probs, value, dist_entropy
