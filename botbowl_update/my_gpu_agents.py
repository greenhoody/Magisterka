import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import botbowl


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
        for i in range(residual_blocks):
            inception_blocks.append(SpatialInceptionBlock(spatial_shape, kernels))
        self.inception_blocks = nn.ModuleList(inception_blocks)
        # Non-spatial
        self.linear0 = nn.Linear(non_spatial_inputs, hidden_nodes)

        # Combining spatial and non spatial analizys

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
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_in", nonlinearity="relu"
                    )
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
