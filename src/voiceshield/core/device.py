"""Hardware device selection and fallback utilities for Apple Silicon (MPS) and CPU."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceStatus:
    """Diagnostic status for the selected computation device."""

    device_name: str
    is_mps: bool
    mps_built: bool
    mps_available: bool
    explanation: str


def resolve_device_status(
    preferred: str | None = None,
    fallback_to_cpu: bool = True,
) -> DeviceStatus:
    """Resolve and explain the optimal torch device.

    Args:
        preferred: Explicitly requested device ('mps', 'cpu', or None to read environment).
        fallback_to_cpu: Whether to safely fall back to CPU if MPS is unavailable.

    Returns:
        DeviceStatus with full diagnostics.
    """
    mps_built = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_built())
    mps_available = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())

    target = preferred or os.getenv("VOICESHIELD_DEVICE", "mps").lower()

    if target == "mps":
        if mps_built and mps_available:
            return DeviceStatus(
                device_name="mps",
                is_mps=True,
                mps_built=mps_built,
                mps_available=mps_available,
                explanation="Apple Silicon Metal Performance Shaders (MPS) is built, available, and active.",
            )
        if fallback_to_cpu:
            reason = (
                "MPS was requested, but "
                + ("MPS is not built in this PyTorch binary. " if not mps_built else "")
                + ("MPS is not available on this hardware/OS. " if not mps_available else "")
                + "Safely falling back to CPU."
            )
            logger.warning(reason)
            return DeviceStatus(
                device_name="cpu",
                is_mps=False,
                mps_built=mps_built,
                mps_available=mps_available,
                explanation=reason,
            )
        raise RuntimeError("MPS device was requested, but MPS is not available on this system.")

    return DeviceStatus(
        device_name="cpu",
        is_mps=False,
        mps_built=mps_built,
        mps_available=mps_available,
        explanation="CPU device explicitly selected.",
    )


def get_device(
    preferred: str | None = None,
    fallback_to_cpu: bool = True,
) -> torch.device:
    """Return the selected torch.device instance.

    Args:
        preferred: Explicitly requested device ('mps' or 'cpu').
        fallback_to_cpu: Whether to safely fall back to CPU if MPS is unavailable.

    Returns:
        torch.device initialized to the resolved target.
    """
    status = resolve_device_status(preferred=preferred, fallback_to_cpu=fallback_to_cpu)
    return torch.device(status.device_name)
