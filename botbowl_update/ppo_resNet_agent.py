# ppo_resnet_agent.py
from __future__ import annotations
import copy
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def masked_logits(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """
    logits:   [B, A]
    action_mask: [B, A] bool (True = dozwolone)
    """
    # duża ujemna stała dla zabronionych akcji
    neg_inf = torch.finfo(logits.dtype).min / 4
    return torch.where(action_mask, logits, torch.full_like(logits, neg_inf))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, ks: int = 3):
        super().__init__()
        pad = ks // 2
        self.conv1 = nn.Conv2d(channels, channels, ks, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, ks, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        return F.relu(out, inplace=True)


class CNNPolicy(nn.Module):
    """
    ResNet-owa polityka pod PPO. Zostawiamy kompatybilny interfejs:
      - forward(spatial, non_spatial) -> (values, logits)
      - act(spatial, non_spatial, action_mask) -> (values, actions)
      - evaluate_actions(spatial, non_spatial, actions, action_mask)
    Argumenty zgodne z dotychczasowym wywołaniem w a2c_example.py.
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
        # Wejściowy "stem": 1x1 żeby dopasować kanały do stałej szerokości
        base_ch = max(32, kernels[0][0] if kernels else 32)
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, base_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        # Rdzeń ResNet
        self.blocks = nn.Sequential(*[ResidualBlock(base_ch, ks=3) for _ in range(residual_blocks)])

        # Globalne uśrednianie + spłaszczenie
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Tor niefiszkowy (non-spatial)
        self.nonspatial = nn.Sequential(
            nn.Linear(non_spatial_obs_space, hidden_nodes),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_nodes, base_ch),
            nn.ReLU(inplace=True),
        )

        # Trunk po złączeniu
        trunk_in = base_ch + base_ch  # GAP(spatial) + nonspatial
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, hidden_nodes),
            nn.ReLU(inplace=True),
        )

        # Głowy
        self.policy = nn.Linear(hidden_nodes, actions)
        self.value = nn.Linear(hidden_nodes, 1)

    def _trunk_features(self, spatial: torch.Tensor, non_spatial: torch.Tensor) -> torch.Tensor:
        # spatial: [B, C, H, W], non_spatial: [B, D]
        x = self.stem(spatial)
        x = self.blocks(x)
        x = self.gap(x).flatten(1)  # [B, base_ch]
        y = self.nonspatial(non_spatial)  # [B, base_ch]
        t = torch.cat([x, y], dim=1)
        return self.trunk(t)

    def forward(self, spatial: torch.Tensor, non_spatial: torch.Tensor):
        z = self._trunk_features(spatial, non_spatial)
        logits = self.policy(z)
        value = self.value(z)
        return value, logits

    @torch.no_grad()
    def act(self, spatial: torch.Tensor, non_spatial: torch.Tensor, action_mask: torch.Tensor):
        """
        Zwraca (values, actions) żeby zachować kompatybilność z istniejącym kodem.
        actions: kształt [B, 1] (tak, jak oczekuje a2c_example.py)
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
        Zwraca (action_log_probs, values, dist_entropy)
        """
        values, logits = self.forward(spatial, non_spatial)
        logits = masked_logits(logits, action_mask.bool())
        dist = Categorical(logits=logits)
        # actions: [B, 1] -> [B]
        log_probs = dist.log_prob(actions.squeeze(-1))
        entropy = dist.entropy().mean()
        return log_probs, values, entropy


def compute_gae(
    rewards: torch.Tensor, masks: torch.Tensor, values: torch.Tensor, gamma: float, lam: float
):
    """
    rewards: [T, N, 1]
    masks:   [T+1, N, 1]  (1 = kontynuacja, 0 = done)
    values:  [T+1, N, 1]  (bootstrapowane wartości)
    Zwraca:
      returns:    [T, N, 1]
      advantages: [T, N, 1]
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
    memory,  # obiekt Memory z a2c_example.py
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
    Minimal-invasive PPO: korzystamy z istniejącego Memory (bez dopisywania pól).
    Obliczamy old log-probsy i wartości z kopii polityki przed aktualizacją.
    """
    T = memory.rewards.shape[0]
    N = memory.rewards.shape[1]

    # Przygotuj tensory
    spatial = memory.spatial_obs      # [T+1, N, C, H, W]
    nonsp = memory.non_spatial_obs    # [T+1, N, 1, D]  -> zrzucimy wymiar 1
    amask = memory.action_masks       # [T+1, N, A]
    actions = memory.actions          # [T, N, 1]
    rewards = memory.rewards          # [T, N, 1]
    masks = memory.masks              # [T+1, N, 1]

    # Sklej w batch: krok po kroku nie jest konieczny — policzymy wszystko hurtowo.
    def flat_obs(x_tn):
        # x_tn ma [T(+1), N, ...] -> płaskie [T(+1)*N, ...]
        return x_tn.view(x_tn.shape[0] * x_tn.shape[1], *x_tn.shape[2:])

    # Kopia "old policy"
    old_policy = copy.deepcopy(policy).eval()

    with torch.no_grad():
        # Wartości dla wszystkich kroków (do GAE)
        v_all, _ = old_policy.forward(
            flat_obs(spatial), flat_obs(nonsp.squeeze(2))
        )
        v_all = v_all.view(spatial.shape[0], N, 1)  # [T+1, N, 1]

        returns, adv = compute_gae(rewards, masks, v_all, gamma, gae_lambda)
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        # Old log_probs (na starych parametrach)
        # Bierzemy tylko kroki 0..T-1
        _, old_logits = old_policy.forward(
            flat_obs(spatial[:-1]),
            flat_obs(nonsp[:-1].squeeze(2)),
        )
        old_logits = old_logits.view(T, N, -1)
        old_logits = masked_logits(old_logits, amask[:-1].bool())
        dist_old = Categorical(logits=old_logits)
        old_log_probs = dist_old.log_prob(actions.squeeze(-1))

    # Zbuduj mini-batche
    B = T * N
    inds = torch.randperm(B, device=spatial.device)
    mb_size = max(1, B // num_mini_batch)

    policy.train()
    value_losses = []
    policy_losses = []
    entropies = []

    for _ in range(ppo_epochs):
        for start in range(0, B, mb_size):
            mb_idx = inds[start:start + mb_size]

            # Wyciągnij porcję danych
            s_mb = flat_obs(spatial[:-1])[mb_idx]
            ns_mb = flat_obs(nonsp[:-1].squeeze(2))[mb_idx]
            a_mb = actions.view(-1, 1)[mb_idx]
            m_mb = flat_obs(amask[:-1])[mb_idx].bool()
            ret_mb = returns.view(-1, 1)[mb_idx]
            adv_mb = adv.view(-1, 1)[mb_idx]
            oldlp_mb = old_log_probs.view(-1)[mb_idx]

            # Nowe oceny
            new_logp_mb, v_mb, ent_mb = policy.evaluate_actions(s_mb, ns_mb, a_mb, m_mb)
            ratio = (new_logp_mb - oldlp_mb).exp()

            surr1 = ratio * adv_mb.squeeze(-1)
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv_mb.squeeze(-1)
            policy_loss = -(torch.min(surr1, surr2)).mean()

            value_loss = F.mse_loss(v_mb, ret_mb)
            loss = value_loss_coef * value_loss + policy_loss - entropy_coef * ent_mb

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            value_losses.append(value_loss.detach())
            policy_losses.append(policy_loss.detach())
            entropies.append(ent_mb.detach())

    # Zwrot średnich, jeśli chcesz logować
    return (
        torch.stack(value_losses).mean().item(),
        torch.stack(policy_losses).mean().item(),
        torch.stack(entropies).mean().item(),
    )
