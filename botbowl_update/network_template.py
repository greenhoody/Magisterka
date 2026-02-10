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


class CustomPolicy(nn.Module):
    def __init__(
        self,
        spatial_shape: Tuple[int, int, int],
        non_spatial_size: int,
        action_space: int,
        hidden_nodes: int = 256,
    ) -> None:
        super().__init__()
        c, h, w = spatial_shape

        # Example trunk: replace with your own.
        self.spatial = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.non_spatial = nn.Sequential(
            nn.Linear(non_spatial_size, hidden_nodes),
            nn.ReLU(),
        )

        trunk_in = 32 + hidden_nodes
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, hidden_nodes),
            nn.ReLU(),
        )

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(hidden_nodes, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, action_space),
        )

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_nodes, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, spatial: torch.Tensor, non_spatial: torch.Tensor):
        # spatial: [B, C, H, W], non_spatial: [B, 1, N] or [B, N]
        if non_spatial.dim() == 3:
            non_spatial = non_spatial.squeeze(1)

        spatial_feat = self.spatial(spatial).flatten(1)
        non_spatial_feat = self.non_spatial(non_spatial)
        z = torch.cat([spatial_feat, non_spatial_feat], dim=1)
        z = self.trunk(z)

        logits = self.actor(z)
        value = self.critic(z)
        return value, logits

    @torch.no_grad()
    def act(
        self,
        spatial: torch.Tensor,
        non_spatial: torch.Tensor,
        action_mask: torch.Tensor,
    ):
        value, logits = self.forward(spatial, non_spatial)
        masked_logits = logits.clone()
        masked_logits[~action_mask] = float("-inf")
        probs = F.softmax(masked_logits, dim=1)
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
        masked_logits = logits.clone()
        masked_logits[~action_mask] = float("-inf")
        log_probs = F.log_softmax(masked_logits, dim=1)
        probs = F.softmax(masked_logits, dim=1)
        action_log_probs = log_probs.gather(1, actions.long())
        dist_entropy = -(log_probs * probs).sum(1).mean()
        return action_log_probs, value, dist_entropy
