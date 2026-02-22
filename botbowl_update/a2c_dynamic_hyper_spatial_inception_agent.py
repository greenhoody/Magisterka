from typing import Callable, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import botbowl
from botbowl.ai.env import EnvConf, BotBowlEnv
from examples.a2c.a2c_env import a2c_scripted_actions
from botbowl.ai.layers import *
from layer_norm import LayerNorm2d
from hyper_connection import HyperConnection, HyperConnectionStack
try:
    from botbowl.utils.serialization import load_policy_checkpoint
except ImportError:
    from serialization_utils import load_policy_checkpoint

# Architecture
model_name = "test-3"
env_name = f"botbowl-3"
model_filename = f"models/{env_name}/{model_name}.nn"
log_filename = f"logs/{env_name}/{env_name}.dat"


class SpatialInceptionBlock(nn.Module):
    def __init__(self, in_ch: int, kernels):
        super().__init__()
        branches = []
        for out_ch, ks in kernels:
            p = ks // 2
            branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_ch,
                        out_ch,
                        kernel_size=ks,
                        stride=1,
                        padding=p,
                        bias=False,
                    ),
                    LayerNorm2d(out_ch),
                    nn.PReLU(num_parameters=out_ch),
                )
            )
        self.branches = nn.ModuleList(branches)

    def forward(self, x):
        return torch.cat([branch(x) for branch in self.branches], dim=1)


class InceptionBlock(nn.Module):
    """
    Spatial Inception + LN + ReLU (no residual). Used as T(h) inside hyper-connection.
    Output channels are projected back to in_ch to keep hyper matrix width fixed.
    """

    def __init__(self, in_ch: int, kernels):
        super().__init__()
        self.inception = SpatialInceptionBlock(in_ch, kernels)
        inner_ch = sum(out for out, _ in kernels)
        self.proj = None
        if inner_ch != in_ch:
            self.proj = nn.Sequential(
                nn.Conv2d(inner_ch, in_ch, kernel_size=1, bias=False),
                LayerNorm2d(in_ch),
            )
        self.out_ch = in_ch
        self.bn = LayerNorm2d(in_ch)

    def forward(self, x):
        y = self.inception(x)
        if self.proj is not None:
            y = self.proj(y)
        y = self.bn(y)
        return F.relu(y, inplace=True)


class CNNPolicy(nn.Module):
    def __init__(
        self,
        spatial_shape: Tuple[int, int, int],
        non_spatial_inputs: int,
        hidden_nodes: int,
        kernels: List[Tuple[int, int]],
        residual_blocks: int,
        actions: int,
        hyper_rate: int = 2,
    ):
        super().__init__()
        c_in, _, _ = spatial_shape

        base_ch = max(32, kernels[0][0] if kernels else c_in)
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, base_ch, kernel_size=1, bias=False),
            LayerNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )

        blocks = []
        ch = base_ch
        for _ in range(residual_blocks):
            block = InceptionBlock(ch, kernels)
            blocks.append(block)
            ch = block.out_ch
        self.hyper_blocks = HyperConnectionStack(
            nn.ModuleList(blocks), rate=hyper_rate, dim=base_ch, dynamic=True
        )
        self.out_ch = ch

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.non_spatial = nn.Sequential(
            nn.Linear(non_spatial_inputs, hidden_nodes),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_nodes, self.out_ch),
            nn.ReLU(inplace=True),
        )

        self.trunk = nn.Sequential(
            nn.Linear(self.out_ch * 2, hidden_nodes),
            nn.ReLU(inplace=True),
        )

        self.actor = nn.Linear(hidden_nodes, actions)
        self.critic = nn.Linear(hidden_nodes, 1)
        self.train()
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (LayerNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def _prepare_non_spatial(non_spatial_input: torch.Tensor) -> torch.Tensor:
        if non_spatial_input.dim() > 2:
            return non_spatial_input.view(non_spatial_input.shape[0], -1)
        return non_spatial_input

    def _trunk_features(self, spatial_input: torch.Tensor, non_spatial_input: torch.Tensor):
        x = self.stem(spatial_input)
        x = self.hyper_blocks(x)
        x = self.gap(x).flatten(1)
        y = self.non_spatial(self._prepare_non_spatial(non_spatial_input))
        return self.trunk(torch.cat([x, y], dim=1))

    def forward(self, spatial_input, non_spatial_input):
        z = self._trunk_features(spatial_input, non_spatial_input)
        policy_logits = self.actor(z)
        value = self.critic(z)
        return value, policy_logits

    def act(self, spatial_inputs, non_spatial_input, action_mask):
        values, action_probs = self.get_action_probs(
            spatial_inputs, non_spatial_input, action_mask=action_mask
        )
        actions = action_probs.multinomial(1)
        for i, action in enumerate(actions):
            correct_action = action
            while not action_mask[i][correct_action]:
                correct_action = action_probs[i].multinomial(1)
            actions[i] = correct_action
        return values, actions

    def evaluate_actions(
        self, spatial_inputs, non_spatial_input, actions, actions_mask
    ):
        value, policy = self(spatial_inputs, non_spatial_input)
        actions_mask = actions_mask.view(-1, 1, actions_mask.shape[2]).squeeze().bool()
        policy[~actions_mask] = float("-inf")
        log_probs = F.log_softmax(policy, dim=1)
        probs = F.softmax(policy, dim=1)
        action_log_probs = log_probs.gather(1, actions)
        log_probs = torch.where(
            log_probs[None, :] == float("-inf"),
            torch.tensor(0.0, device=log_probs.device),
            log_probs,
        )
        dist_entropy = -(log_probs * probs).sum(-1).mean()
        return action_log_probs, value, dist_entropy

    def get_action_probs(self, spatial_input, non_spatial_input, action_mask):
        values, actions = self(spatial_input, non_spatial_input)
        if action_mask is not None:
            actions[~action_mask] = float("-inf")
        action_probs = F.softmax(actions, dim=1)
        return values, action_probs


SAFE_POLICY_COMPONENTS = [
    CNNPolicy,
    SpatialInceptionBlock,
    InceptionBlock,
    LayerNorm2d,
    HyperConnection,
    HyperConnectionStack,
]


class A2CAgent(Agent):
    env: BotBowlEnv

    def __init__(
        self,
        name,
        env_conf: EnvConf,
        scripted_func: Callable[[Game], Optional[Action]] = None,
        filename=model_filename,
        exclude_pathfinding_moves=True,
    ):
        super().__init__(name)
        self.env = BotBowlEnv(env_conf)
        self.exclude_pathfinding_moves = exclude_pathfinding_moves

        self.scripted_func = scripted_func
        self.action_queue = []

        self.policy = load_policy_checkpoint(
            filename, SAFE_POLICY_COMPONENTS, map_location="cpu"
        )
        self.policy.eval()
        self.end_setup = False

    def new_game(self, game, team):
        pass

    @staticmethod
    def _update_obs(array: np.ndarray):
        return torch.unsqueeze(torch.from_numpy(array.copy()), dim=0)

    def act(self, game):
        if len(self.action_queue) > 0:
            return self.action_queue.pop(0)

        if self.scripted_func is not None:
            scripted_action = self.scripted_func(game)
            if scripted_action is not None:
                return scripted_action

        self.env.game = game

        spatial_obs, non_spatial_obs, action_mask = map(
            A2CAgent._update_obs, self.env.get_state()
        )
        non_spatial_obs = torch.unsqueeze(non_spatial_obs, dim=0)

        _, actions = self.policy.act(
            Variable(spatial_obs.float()),
            Variable(non_spatial_obs.float()),
            Variable(action_mask),
        )

        action_idx = actions[0]
        action_objects = self.env._compute_action(action_idx)

        self.action_queue = action_objects
        return self.action_queue.pop(0)
