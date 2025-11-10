# ppo_spatial_inception_agent.py
from __future__ import annotations
import copy
from typing import Callable, List, Optional, Tuple

import botbowl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from botbowl.ai.env import BotBowlEnv, EnvConf


def masked_logits(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """
    logits:      [B, A]
    action_mask: [B, A] (bool; True = akcja dozwolona)
    """
    neg_inf = torch.finfo(logits.dtype).min / 4
    return torch.where(action_mask, logits, torch.full_like(logits, neg_inf))


class SpatialInceptionBlock(nn.Module):
    """
    Pojedynczy blok wielogałęziowy: conv(k_i) dla każdej gałęzi, BN, PReLU -> concat po kanale.
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
        return torch.cat([b(x) for b in self.branches], dim=1)


class InceptionResidualBlock(nn.Module):
    """
    Spatial Inception + skip connection.
    Jeżeli liczba kanałów się nie zgadza, używa projekcji 1×1, by dopasować wymiar.
    """
    def __init__(self, in_ch: int, kernels: List[Tuple[int, int]]):
        super().__init__()
        self.inception = SpatialInceptionBlock(in_ch, kernels)
        out_ch = sum(o for o, _ in kernels)
        self.proj = None
        if out_ch != in_ch:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.inception(x)
        skip = x if self.proj is None else self.proj(x)
        y = y + skip
        y = self.bn(y)
        return F.relu(y, inplace=True)


class CNNPolicy(nn.Module):
    """
    PPO policy z blokami Spatial Inception (rezidual).
    Interfejs identyczny jak w ppo_resNet_agent - Copy.py:
      - forward(spatial, non_spatial) -> (values, logits)
      - act(spatial, non_spatial, action_mask) -> (values, actions)
      - evaluate_actions(spatial, non_spatial, actions, action_mask)
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
        c_in, h, w = spatial_obs_space

        # Stem: wąskie dopasowanie kanałów, żeby zacząć od sensownej szerokości
        base_ch = max(32, kernels[0][0] if kernels else 32)
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, base_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )

        # Stos bloków Inception-Residual; kanały mogą się zmieniać między blokami
        blocks = []
        ch = base_ch
        for _ in range(residual_blocks):
            block = InceptionResidualBlock(ch, kernels)
            blocks.append(block)
            ch = sum(o for o, _ in kernels) if sum(o for o, _ in kernels) != ch else ch
        self.blocks = nn.Sequential(*blocks)
        self.out_ch = ch

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Tor niefiszowy (non-spatial) -> do tej samej szerokości co spatial po GAP (out_ch)
        self.nonspatial = nn.Sequential(
            nn.Linear(non_spatial_obs_space, hidden_nodes),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_nodes, self.out_ch),
            nn.ReLU(inplace=True),
        )

        # Trunk po złączeniu: [B, out_ch] (GAP spatial) + [B, out_ch] (nonspatial)
        self.trunk = nn.Sequential(
            nn.Linear(self.out_ch * 2, hidden_nodes),
            nn.ReLU(inplace=True),
        )

        # Głowy
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
        # spatial: [B, C, H, W]; non_spatial: [B, D]
        x = self.stem(spatial)
        x = self.blocks(x)
        x = self.gap(x).flatten(1)     # [B, out_ch]
        y = self.nonspatial(self._prepare_non_spatial(non_spatial))  # [B, out_ch]
        z = torch.cat([x, y], dim=1)   # [B, 2*out_ch]
        return self.trunk(z)

    def forward(self, spatial: torch.Tensor, non_spatial: torch.Tensor):
        z = self._trunk_features(spatial, non_spatial)
        logits = self.policy(z)
        value = self.value(z)
        return value, logits

    @torch.no_grad()
    def act(self, spatial: torch.Tensor, non_spatial: torch.Tensor, action_mask: torch.Tensor):
        """
        Zwraca (values, actions) zgodnie z dotychczasowym oczekiwaniem:
        actions ma kształt [B, 1].
        """
        value, logits = self.forward(spatial, non_spatial)
        logits = masked_logits(logits, action_mask.bool())
        dist = Categorical(logits=logits)
        a = dist.sample().unsqueeze(-1).long()
        return value, a

    def evaluate_actions(
        self,
        spatial: torch.Tensor,
        non_spatial: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor,
    ):
        """
        Zwraca (action_log_probs, values, dist_entropy),
        interfejs zgodny z Twoim ppo_resNet_agent.
        """
        values, logits = self.forward(spatial, non_spatial)
        logits = masked_logits(logits, action_mask.bool())
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions.squeeze(-1))
        entropy = dist.entropy().mean()
        return log_probs, values, entropy


def compute_gae(
    rewards: torch.Tensor, masks: torch.Tensor, values: torch.Tensor, gamma: float, lam: float
):
    """
    rewards: [T, N, 1]
    masks:   [T+1, N, 1]  (1 = kontynuacja, 0 = done)
    values:  [T+1, N, 1]  (bootstrap)
    -> returns, advantages: [T, N, 1]
    """
    T, N, _ = rewards.shape
    advantages = torch.zeros(T, N, 1, device=rewards.device)
    gae = torch.zeros(N, 1, device=rewards.device)
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] * masks[t] - values[t]
        gae = delta + gamma * lam * masks[t] * gae
        advantages[t] = gae
    returns = advantages + values[:-1]
    return returns, advantages


def ppo_update(
    policy: CNNPolicy,
    optimizer: torch.optim.Optimizer,
    memory,  # zgodny z obecnym Memory (A2C/PPO bufor kroków)
    *,
    clip_param: float = 0.2,
    ppo_epochs: int = 4,
    num_mini_batch: int = 4,
    value_loss_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
):
    """
    Minimal-invasive PPO: używamy istniejących pól z Memory (bez zmian struktury).
    Kopiujemy starą politykę do obliczenia old log-probs i wartości.
    """
    T = memory.rewards.shape[0]
    N = memory.rewards.shape[1]

    spatial = memory.spatial_obs      # [T+1, N, C, H, W]
    nonsp = memory.non_spatial_obs    # [T+1, N, 1, D]
    amask = memory.action_masks       # [T+1, N, A]
    actions = memory.actions          # [T, N, 1]
    rewards = memory.rewards          # [T, N, 1]
    masks = memory.masks              # [T+1, N, 1]

    def flat_obs(x_tn):
        return x_tn.view(x_tn.shape[0] * x_tn.shape[1], *x_tn.shape[2:])

    old_policy = copy.deepcopy(policy).eval()

    with torch.no_grad():
        v_all, _ = old_policy.forward(
            flat_obs(spatial), flat_obs(nonsp.squeeze(2))
        )
        v_all = v_all.view(spatial.shape[0], N, 1)  # [T+1, N, 1]

        returns, adv = compute_gae(rewards, masks, v_all, gamma, gae_lambda)
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        _, old_logits = old_policy.forward(
            flat_obs(spatial[:-1]),
            flat_obs(nonsp[:-1].squeeze(2)),
        )
        old_logits = old_logits.view(T, N, -1)
        old_logits = masked_logits(old_logits, amask[:-1].bool())
        dist_old = Categorical(logits=old_logits)
        old_log_probs = dist_old.log_prob(actions.squeeze(-1))

    B = T * N
    inds = torch.randperm(B, device=spatial.device)
    mb_size = max(1, B // num_mini_batch)

    policy.train()
    value_losses, policy_losses, entropies = [], [], []

    for _ in range(ppo_epochs):
        for start in range(0, B, mb_size):
            mb_idx = inds[start:start + mb_size]

            s_mb = flat_obs(spatial[:-1])[mb_idx]
            ns_mb = flat_obs(nonsp[:-1].squeeze(2))[mb_idx]
            a_mb = actions.view(-1, 1)[mb_idx]
            m_mb = flat_obs(amask[:-1])[mb_idx].bool()
            ret_mb = returns.view(-1, 1)[mb_idx]
            adv_mb = adv.view(-1, 1)[mb_idx]
            oldlp_mb = old_log_probs.view(-1)[mb_idx]

            new_logp_mb, v_mb, ent_mb = policy.evaluate_actions(s_mb, ns_mb, a_mb, m_mb)
            ratio = (new_logp_mb - oldlp_mb).exp()

            s1 = ratio * adv_mb.squeeze(-1)
            s2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv_mb.squeeze(-1)
            policy_loss = -(torch.min(s1, s2)).mean()

            value_loss = F.mse_loss(v_mb, ret_mb)
            loss = value_loss_coef * value_loss + policy_loss - entropy_coef * ent_mb

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            value_losses.append(value_loss.detach())
            policy_losses.append(policy_loss.detach())
            entropies.append(ent_mb.detach())

    return (
        torch.stack(value_losses).mean().item(),
        torch.stack(policy_losses).mean().item(),
        torch.stack(entropies).mean().item(),
    )


class PPOAgent(botbowl.Agent):
    """Wrapper that loads a saved PPO policy and exposes the BotBowl Agent API."""

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

        with torch.no_grad():
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
