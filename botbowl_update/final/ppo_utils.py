from __future__ import annotations

import torch

def probability_ratio(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the standard PPO ratio exp(new_logp - old_logp).
    """
    return torch.exp(new_logp - old_logp)
