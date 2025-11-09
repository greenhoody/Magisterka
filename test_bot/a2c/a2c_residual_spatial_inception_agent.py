from typing import Callable

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

# Architecture
model_name = "test-3"
env_name = f"botbowl-3"
model_filename = f"models/{env_name}/{model_name}.nn"
log_filename = f"logs/{env_name}/{env_name}.dat"


class SpatialInceptionBlock(nn.Module):
    def __init__(self, spatial_shape, kernels):
        super(SpatialInceptionBlock, self).__init__()
        branches = []
        # kernels pairs (number of kernels, size of kernels)
        for out_channels, kernel_s in kernels:
            p = kernel_s // 2
            branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        spatial_shape[0],
                        out_channels,
                        kernel_size=kernel_s,
                        stride=1,
                        padding=p,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.PReLU(num_parameters=out_channels),
                )
            )
        self.branches = nn.ModuleList(branches)

    def forward(self, x):
        outputs = [branch(x) for branch in self.branches]
        # konkatenacja po kanale
        return torch.cat(outputs, dim=1)


class CNNPolicy(nn.Module):
    def __init__(
        self,
        spatial_shape,
        non_spatial_inputs,
        hidden_nodes,
        kernels,
        residual_blocks,
        actions,
    ):
        super(CNNPolicy, self).__init__()
        # My Spatial input
        # Spatial inception-res
        inception_blocks = []

        for i in range(residual_blocks):
            inception_blocks.append(SpatialInceptionBlock(spatial_shape, kernels))
        self.inception_blocks = nn.ModuleList(inception_blocks)
        # My Non-spatial
        self.linear0 = nn.Linear(non_spatial_inputs, hidden_nodes)

        # combining spatial and non spatial analizys.

        # po zbudowaniu self.inception_blocks
        # kernels to teraz [(out1,k1), (out2,k2), ..., (outN,kN)]
        total_out_ch = sum(out_ch for out_ch, _ in kernels)
        # teraz policz rozmiar strumienia przestrzennego
        stream_size = total_out_ch * spatial_shape[1] * spatial_shape[2]
        # dodaj wymiar nie-przestrzenny
        stream_size += hidden_nodes
        # old
        # stream_size = kernels[1] * spatial_shape[1] * spatial_shape[2]
        # stream_size += hidden_nodes
        self.linear1 = nn.Linear(stream_size, stream_size)
        self.critic_linear = nn.Linear(stream_size, hidden_nodes)
        self.actor_linear = nn.Linear(stream_size, stream_size)
        # The outputs
        self.critic = nn.Linear(hidden_nodes, 1)
        self.actor = nn.Linear(stream_size, actions)
        self.train()
        self.reset_parameters()

    def reset_parameters(self):
        """
        He‐inicjalizacja dla Conv2d i Linear,
        BatchNorm weight→1, bias→0.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He‐inicjalizacja wariant normalny
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                # gamma=1, beta=0
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, spatial_input, non_spatial_input):
        """
        The forward functions defines how the data flows through the graph (layers)
        """
        # My Forward
        #
        # 1) Spatial → Inception-Res blocks
        x = spatial_input  # shape [B, C, H, W]
        for block in self.inception_blocks:
            residual = x
            out = block(x)
            x = out + residual  # dalej [B, C_out, H, W]

        x = x.flatten(start_dim=1)  # → [B, C_out*H*W]

        # 2) Non-spatial → dense
        y = self.linear0(non_spatial_input)  # [B, hidden_nodes]
        y = F.relu(y, inplace=True)
        y = y.flatten(start_dim=1)
        # 3) Concatenate i kolejna warstwa
        xy = torch.cat([x, y], dim=1)  # [B, stream_size]
        z = self.linear1(xy)  # [B, hidden_nodes]
        z = F.relu(z, inplace=True)
        lc = self.critic_linear(z)
        la = self.actor_linear(z)
        # 4) Dwa wyjścia
        value = self.critic(lc)  # [B, 1]
        actor = self.actor(la)  # [B, actions]

        return value, actor

    def act(self, spatial_inputs, non_spatial_input, action_mask):
        values, action_probs = self.get_action_probs(
            spatial_inputs, non_spatial_input, action_mask=action_mask
        )
        actions = action_probs.multinomial(1)
        # In rare cases, multinomial can  sample an action with p=0, so let's avoid that
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
            log_probs[None, :] == float("-inf"), torch.tensor(0.0), log_probs
        )
        dist_entropy = -(log_probs * probs).sum(-1).mean()
        return action_log_probs, value, dist_entropy

    def get_action_probs(self, spatial_input, non_spatial_input, action_mask):
        values, actions = self(spatial_input, non_spatial_input)
        # Masking step: Inspired by: http://juditacs.github.io/2018/12/27/masked-attention.html
        if action_mask is not None:
            actions[~action_mask] = float("-inf")
        action_probs = F.softmax(actions, dim=1)
        return values, action_probs


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

        # MODEL
        self.policy = torch.load(filename)
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

    def end_game(self, game):
        pass


def main():
    # Register the bot to the framework
    def _make_my_a2c_bot(name, env_size=11):
        return A2CAgent(
            name=name,
            env_conf=EnvConf(size=env_size),
            scripted_func=a2c_scripted_actions,
            filename=model_filename,
        )

    botbowl.register_bot("my-a2c-bot", _make_my_a2c_bot)

    # Load configurations, rules, arena and teams
    config = botbowl.load_config("bot-bowl")
    config.competition_mode = False
    config.pathfinding_enabled = False
    ruleset = botbowl.load_rule_set(config.ruleset)
    arena = botbowl.load_arena(config.arena)
    home = botbowl.load_team_by_filename("human", ruleset)
    away = botbowl.load_team_by_filename("human", ruleset)
    config.competition_mode = False
    config.debug_mode = False

    # Play 10 games
    wins = 0
    draws = 0
    n = 10
    is_home = True
    tds_away = 0
    tds_home = 0
    for i in range(n):
        if is_home:
            away_agent = botbowl.make_bot("random")
            home_agent = botbowl.make_bot("my-a2c-bot")
        else:
            away_agent = botbowl.make_bot("my-a2c-bot")
            home_agent = botbowl.make_bot("random")
        game = botbowl.Game(
            i, home, away, home_agent, away_agent, config, arena=arena, ruleset=ruleset
        )
        game.config.fast_mode = True

        print("Starting game", (i + 1))
        game.init()
        print("Game is over")

        winner = game.get_winner()
        if winner is None:
            draws += 1
        elif winner == home_agent and is_home:
            wins += 1
        elif winner == away_agent and not is_home:
            wins += 1

        tds_home += game.get_agent_team(home_agent).state.score
        tds_away += game.get_agent_team(away_agent).state.score

    print(f"Home/Draws/Away: {wins}/{draws}/{n - wins - draws}")
    print(f"Home TDs per game: {tds_home / n}")
    print(f"Away TDs per game: {tds_away / n}")


if __name__ == "__main__":
    main()
