"""Audio quality metrics and clipping diagnostics calculated before and after normalization."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class AudioQualityMetrics(BaseModel):
    """Untouched and transformed audio quality diagnostics."""

    peak_level: float = Field(description="Maximum absolute sample amplitude.")
    rms: float = Field(description="Root-mean-square energy.")
    clipping_fraction: float = Field(
        description="Fraction of samples with absolute amplitude >= 0.999.",
    )
    is_clipped: bool = Field(description="Whether the audio exhibits digital clipping distortion.")
    is_silent: bool = Field(description="Whether audio energy is effectively silence (RMS < 1e-4).")
    snr_db_estimate: float = Field(description="Estimated signal-to-noise ratio in decibels.")
    quality_status: str = Field(description="Summary quality classification string.")


def compute_quality_metrics(
    samples: np.ndarray[Any, np.dtype[np.float32]],
    clipping_threshold: float = 0.999,
) -> AudioQualityMetrics:
    """Compute untouched physical quality metrics for 1D or 2D audio samples.

    Args:
        samples: Audio sample array (float32).
        clipping_threshold: Absolute value threshold considered a clipped sample.

    Returns:
        Populated AudioQualityMetrics instance.
    """
    if samples.size == 0:
        return AudioQualityMetrics(
            peak_level=0.0,
            rms=0.0,
            clipping_fraction=0.0,
            is_clipped=False,
            is_silent=True,
            snr_db_estimate=0.0,
            quality_status="empty",
        )

    flat = np.abs(samples.ravel())
    peak = float(np.max(flat))
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))

    # Detect clipping: count samples at or above clipping threshold
    clipping_mask = flat >= clipping_threshold
    clipping_fraction = float(np.mean(clipping_mask))
    is_clipped = bool(clipping_fraction > 0.0005)  # > 0.05% clipped

    is_silent = bool(rms < 1e-4)

    # Robust SNR estimation: 95th percentile (signal) vs 10th percentile (noise floor)
    # in logarithmic power domain
    p95 = float(np.percentile(flat, 95))
    p10 = float(np.percentile(flat, 10))
    if p10 > 1e-6:
        snr_est = float(20.0 * np.log10(max(p95 / p10, 1.0)))
    else:
        snr_est = float(20.0 * np.log10(max(p95 / 1e-6, 1.0))) if p95 > 1e-6 else 0.0

    # Determine qualitative status
    if is_silent:
        status = "silent"
    elif is_clipped:
        status = "clipped"
    elif rms < 0.01:
        status = "low_energy"
    elif snr_est < 12.0:
        status = "marginal_snr"
    else:
        status = "clean"

    return AudioQualityMetrics(
        peak_level=peak,
        rms=rms,
        clipping_fraction=clipping_fraction,
        is_clipped=is_clipped,
        is_silent=is_silent,
        snr_db_estimate=snr_est,
        quality_status=status,
    )
