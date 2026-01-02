"""
Utility helpers for swapping scripted opponents to saved checkpoints once
training metrics reach desired thresholds.
"""

from __future__ import annotations

import copy
import os
from datetime import datetime
from typing import Callable, Optional

import torch
import torch.nn as nn


class CheckpointOpponentScheduler:
    """
    Saves the latest learner policy, loads it as an opponent via the provided
    factory, and swaps it into every vectorized environment once the desired
    performance thresholds are met.
    """

    def __init__(
        self,
        envs,
        exp_identifier: str,
        make_agent_fn: Callable[..., object],
        model_dir: str,
        min_updates: int = 0,
        difficulty_threshold: float = 1.0,
        win_rate_threshold: float = 0.55,
    ):
        self.envs = envs
        self.exp_identifier = exp_identifier
        self.make_agent_fn = make_agent_fn
        self.model_dir = model_dir
        self.min_updates = max(min_updates, 0)
        self.difficulty_threshold = difficulty_threshold
        self.win_rate_threshold = win_rate_threshold
        self.last_swap_update: Optional[int] = None
        self.swap_counter = 0

    def maybe_swap(
        self,
        update_idx: int,
        policy: nn.Module,
        difficulty: float,
        win_rate: Optional[float],
    ) -> Optional[str]:
        if win_rate is None:
            return None
        if update_idx < self.min_updates:
            return None
        if difficulty + 1e-8 < self.difficulty_threshold:
            return None
        if win_rate < self.win_rate_threshold:
            return None
        if self.last_swap_update == update_idx:
            return None

        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        model_name = (
            f"{self.exp_identifier}_opp_{self.swap_counter}_{timestamp}.nn"
        )
        model_path = os.path.join(self.model_dir, model_name)

        device = next(policy.parameters()).device
        cpu_copy = copy.deepcopy(policy).to(torch.device("cpu"))
        torch.save(cpu_copy, model_path)
        policy.to(device)

        opponent = self.make_agent_fn(name=model_name, filename=model_path)
        self.envs.swap(opponent)

        self.swap_counter += 1
        self.last_swap_update = update_idx
        return model_path
