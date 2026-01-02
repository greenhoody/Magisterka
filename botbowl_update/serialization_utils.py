from __future__ import annotations

from contextlib import nullcontext
from typing import Iterable, Sequence

import torch


def _normalize_map_location(value):
    if value is None:
        return None
    if isinstance(value, str):
        return torch.device(value)
    return value


def load_policy_checkpoint(
    filename: str,
    safe_classes: Sequence[type] | Iterable[type] | None = None,
    *,
    map_location: str | torch.device | None = "cpu",
):
    """
    Load a serialized torch.nn.Module checkpoint while allowing custom classes.

    PyTorch 2.6 switched torch.load(weights_only=True) by default, and also
    tightened pickle globals via serialization.safe_globals. This helper keeps
    older checkpoints compatible while still letting callers express which
    classes should be trusted.
    """

    safe_classes = tuple(safe_classes or ())
    serialization = getattr(torch, "serialization", None)
    safe_ctx = nullcontext()

    if safe_classes and serialization is not None:
        if hasattr(serialization, "safe_globals"):
            safe_ctx = serialization.safe_globals(safe_classes)
        elif hasattr(serialization, "add_safe_globals"):
            serialization.add_safe_globals(safe_classes)

    load_kwargs = {}
    normalized_map = _normalize_map_location(map_location)
    if normalized_map is not None:
        load_kwargs["map_location"] = normalized_map

    with safe_ctx:
        try:
            return torch.load(filename, weights_only=False, **load_kwargs)
        except TypeError:
            return torch.load(filename, **load_kwargs)
