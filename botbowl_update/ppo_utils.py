from __future__ import annotations

import torch

# Maximum absolute difference between new and old log-probabilities before exp().
# Prevents torch.exp from overflowing to inf when policies change too much.
LOG_RATIO_CLAMP = 20.0


def probability_ratio(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    max_log_diff: float = LOG_RATIO_CLAMP,
) -> torch.Tensor:
    """
    Compute exp(new_logp - old_logp) with a clamp on the log difference.

    PPO ratios can blow up in large action spaces because log probabilities are
    unbounded below. Clamping the log difference before exponentiation avoids
    inf/NaN ratios that would otherwise destroy the gradients.
    """
    log_ratio = new_logp - old_logp
    clamped = torch.clamp(log_ratio, -max_log_diff, max_log_diff)
    return torch.exp(clamped)
