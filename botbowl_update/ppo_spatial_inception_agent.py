from __future__ import annotations

import copy
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import botbowl
from botbowl.ai.env import BotBowlEnv, EnvConf

from ppo_residual_spatial_inception_agent import (
    compute_gae,
    masked_logits,
    ppo_update,
)


class SpatialInceptionBlock(nn.Module):
    """
    Pojedynczy blok wielogałęziowy bez skipów: conv(k_i) → BN → PReLU → concat.
    """

    def __init__(self, in_ch: int, kernels: List[Tuple[int, int]]):
        super().__init__()
        branches = []
        for out_ch, ks in kernels:
            pad = ks // 2
            branches.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=ks, padding=pad, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.PReLU(num_parameters=out_ch),
                )
            )
        self.branches = nn.ModuleList(branches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([branch(x) for branch in self.branches], dim=1)


class InceptionBlock(nn.Module):
    """
    Spatial Inception + BN + ReLU (bez sumowania z wejściem).
    """

    def __init__(self, in_ch: int, kernels: List[Tuple[int, int]]):
        super().__init__()
        self.inception = SpatialInceptionBlock(in_ch, kernels)
        self.out_ch = sum(o for o, _ in kernels)
        self.bn = nn.BatchNorm2d(self.out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.inception(x)
        y = self.bn(y)
        return F.relu(y, inplace=True)


class CNNPolicy(nn.Module):
    """
    Wersja PPO bez skipów: stem 1×1 → bloki Inception → GAP → trunk → głowy.
    API w pełni zgodne z ppo_residual_spatial_inception_agent.
    """

    def __init__(
        self,
        spatial_obs_space: Tuple[int, int, int],
        non_spatial_obs_space: int,
        hidden_nodes: int,
        kernels: List[Tuple[int, int]],
        residual_blocks: int,
        actions: int,
    ):
        super().__init__()
        c_in, _, _ = spatial_obs_space
        base_ch = max(32, kernels[0][0] if kernels else 32)
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, base_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        blocks = []
        ch = base_ch
        for _ in range(residual_blocks):
            block = InceptionBlock(ch, kernels)
            blocks.append(block)
            ch = block.out_ch
        self.blocks = nn.Sequential(*blocks)
        self.out_ch = ch
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.nonspatial = nn.Sequential(
            nn.Linear(non_spatial_obs_space, hidden_nodes),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_nodes, self.out_ch),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(
            nn.Linear(self.out_ch * 2, hidden_nodes),
            nn.ReLU(inplace=True),
        )
        self.policy = nn.Linear(hidden_nodes, actions)
        self.value = nn.Linear(hidden_nodes, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def _prepare_non_spatial(non_spatial: torch.Tensor) -> torch.Tensor:
        if non_spatial.dim() > 2:
            return non_spatial.reshape(non_spatial.shape[0], -1)
        return non_spatial

    def _trunk_features(self, spatial: torch.Tensor, non_spatial: torch.Tensor) -> torch.Tensor:
        x = self.stem(spatial)
        x = self.blocks(x)
        x = self.gap(x).flatten(1)
        y = self.nonspatial(self._prepare_non_spatial(non_spatial))
        return self.trunk(torch.cat([x, y], dim=1))

    def forward(self, spatial: torch.Tensor, non_spatial: torch.Tensor):
        z = self._trunk_features(spatial, non_spatial)
        logits = self.policy(z)
        value = self.value(z)
        return value, logits

    @torch.no_grad()
    def act(self, spatial: torch.Tensor, non_spatial: torch.Tensor, action_mask: torch.Tensor):
        value, logits = self.forward(spatial, non_spatial)
        logits = masked_logits(logits, action_mask.bool())
        dist = Categorical(logits=logits)
        return value, dist.sample().unsqueeze(-1).long()

    def evaluate_actions(
        self,
        spatial: torch.Tensor,
        non_spatial: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor,
    ):
        values, logits = self.forward(spatial, non_spatial)
        logits = masked_logits(logits, action_mask.bool())
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions.squeeze(-1))
        entropy = dist.entropy().mean()
        return log_probs, values, entropy


class PPOAgent(botbowl.Agent):
    """Wrapper identyczny z wersją residual, ładuje model PPO z dysku."""

    env: BotBowlEnv

    def __init__(
        self,
        name: str,
        env_conf: EnvConf,
        scripted_func: Optional[Callable[[botbowl.Game], Optional[botbowl.Action]]] = None,
        filename: Optional[str] = None,
        exclude_pathfinding_moves: bool = True,
    ):
        super().__init__(name)
        if filename is None:
            raise ValueError("filename with a serialized PPO policy must be provided")

        self.env = BotBowlEnv(env_conf)
        self.exclude_pathfinding_moves = exclude_pathfinding_moves
        self.scripted_func = scripted_func
        self.action_queue = []

        self.policy: CNNPolicy = torch.load(filename, map_location=torch.device("cpu"))
        self.policy.eval()
        self.end_setup = False

    def new_game(self, game, team):
        pass

    @staticmethod
    def _update_obs(array: np.ndarray) -> torch.Tensor:
        return torch.unsqueeze(torch.from_numpy(array.copy()), dim=0)

    def act(self, game):
        if self.action_queue:
            return self.action_queue.pop(0)

        if self.scripted_func is not None:
            scripted_action = self.scripted_func(game)
            if scripted_action is not None:
                return scripted_action

        self.env.game = game
        spatial_obs, non_spatial_obs, action_mask = map(
            PPOAgent._update_obs, self.env.get_state()
        )
        non_spatial_obs = torch.unsqueeze(non_spatial_obs, dim=0)

        _, actions = self.policy.act(
            spatial_obs.float(),
            non_spatial_obs.float(),
            action_mask,
        )
        action_idx = actions[0]
        action_objects = self.env._compute_action(action_idx)

        self.action_queue = action_objects
        return self.action_queue.pop(0)

    def end_game(self, game):
        pass
