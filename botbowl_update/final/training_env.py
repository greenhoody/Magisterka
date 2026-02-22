from __future__ import annotations

import os

ENV_SIZE_ENV = "BOTBOWL_ENV_SIZE"


def resolve_env_size(default_size: int) -> int:
    """
    Return the environment size requested via BOTBOWL_ENV_SIZE, or fall back to
    the provided default (must be positive).
    """
    if default_size <= 0:
        raise ValueError("default_size must be positive")

    override = os.environ.get(ENV_SIZE_ENV)
    if not override:
        return default_size

    try:
        size = int(override)
    except ValueError as exc:
        raise ValueError(
            f"{ENV_SIZE_ENV} must be an integer, got '{override}'"
        ) from exc

    if size <= 0:
        raise ValueError(f"{ENV_SIZE_ENV} must be positive, got {size}")
    return size
