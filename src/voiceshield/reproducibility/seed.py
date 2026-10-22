"""Deterministic random seeding utilities for Python, NumPy, and PyTorch."""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set seeds across Python standard library, NumPy, and PyTorch.

    Args:
        seed: Integer seed value.
        deterministic: If True, configures PyTorch to use deterministic algorithms
                       (with warn_only=True on MPS/CPU when specific operators lack deterministic kernels).
    """
    logger.info("Setting global random seed to %d (deterministic=%s)", seed, deterministic)

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Seed Apple Silicon MPS if available
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        try:
            torch.mps.manual_seed(seed)
        except Exception as exc:
            logger.debug("MPS manual_seed failed: %s", exc)

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as exc:
            logger.warning("Deterministic algorithms could not be enabled: %s", exc)
