import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import botbowl


class SpatialInceptionBlock(nn.Module):
    def __init__(self, in_channels: int, kernels):
        super(SpatialInceptionBlock, self).__init__()
        branches = []
        # kernels pairs (number of kernels, size of kernels)
        for out_channels, kernel_s in kernels:
            p = kernel_s // 2
            branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
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
        self.out_channels = sum(out_ch for out_ch, _ in kernels)
        self.residual_proj = None
        if self.out_channels != in_channels:
            self.residual_proj = nn.Sequential(
                nn.Conv2d(in_channels, self.out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.out_channels),
            )

    def forward(self, x):
        outputs = [branch(x) for branch in self.branches]
        # konkatenacja po kanale
        out = torch.cat(outputs, dim=1)
        residual = x if self.residual_proj is None else self.residual_proj(x)
        return out + residual


class SpatialInceptionCNN(nn.Module):
    def __init__(
        self,
        spatial_shape,
        non_spatial_inputs,
        hidden_nodes,
        kernels,
        residual_blocks,
        actions,
    ):
        super(SpatialInceptionCNN, self).__init__()
        # My Spatial input Spatial inception-res
        inception_blocks = []
        current_channels = spatial_shape[0]
        for _ in range(residual_blocks):
            block = SpatialInceptionBlock(current_channels, kernels)
            inception_blocks.append(block)
            current_channels = block.out_channels
        self.inception_blocks = nn.ModuleList(inception_blocks)
        self.spatial_channels = current_channels
        # Non-spatial
        self.linear0 = nn.Linear(non_spatial_inputs, hidden_nodes)

        # Combining spatial and non spatial analizys

        # po zbudowaniu self.inception_blocks
        # kernels to teraz [(out1,k1), (out2,k2), ..., (outN,kN)]
        total_out_ch = self.spatial_channels
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
        He-inicjalizacja dla Conv2d i Linear,
        BatchNorm weight -> 1, bias -> 0
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
            x = block(x)  # dalej [B, C_out, H, W]

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
        actions_mask = actions_mask.view(actions_mask.shape[0], -1).bool()
        flat_mask = actions_mask
        invalid_rows = ~flat_mask.any(dim=1)
        if invalid_rows.any():
            actions_mask[invalid_rows] = True
        policy = policy.masked_fill(~actions_mask, -1e9)
        log_probs = F.log_softmax(policy, dim=1)
        probs = F.softmax(policy, dim=1)
        action_log_probs = log_probs.gather(1, actions)
        dist_entropy = -(log_probs * probs).sum(-1).mean()
        return action_log_probs, value, dist_entropy

    def get_action_probs(self, spatial_input, non_spatial_input, action_mask):
        values, actions = self(spatial_input, non_spatial_input)
        # Masking step: Inspired by: http://juditacs.github.io/2018/12/27/masked-attention.html
        if action_mask is not None:
            action_mask = action_mask.view(action_mask.shape[0], -1).bool()
            flat_mask = action_mask
            invalid_rows = ~flat_mask.any(dim=1)
            if invalid_rows.any():
                action_mask[invalid_rows] = True
            actions = actions.masked_fill(~action_mask, -1e9)
        action_probs = F.softmax(actions, dim=1)
        return values, action_probs
