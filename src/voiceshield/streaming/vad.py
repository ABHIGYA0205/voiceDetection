"""Replaceable Voice Activity Detection (VAD) interface and default energy-based implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseVAD(ABC):
    """Abstract interface for Voice Activity Detection."""

    @abstractmethod
    def compute_speech_duration(
        self,
        samples: np.ndarray[Any, np.dtype[np.float32]],
        sample_rate: int = 16000,
    ) -> float:
        """Compute the total duration of active speech in the given audio window in seconds.

        Args:
            samples: 1D float32 audio waveform.
            sample_rate: Sampling rate in Hz (default: 16,000 Hz).

        Returns:
            Total active speech duration in seconds.
        """
        ...

    def is_speech(
        self,
        samples: np.ndarray[Any, np.dtype[np.float32]],
        min_duration_seconds: float = 0.8,
        sample_rate: int = 16000,
    ) -> bool:
        """Check if audio contains at least the required minimum duration of active speech."""
        return self.compute_speech_duration(samples, sample_rate) >= min_duration_seconds

    def reset(self) -> None:  # noqa: B027
        """Reset internal state for stateful VAD implementations.

        Default implementation is an intentional no-op for stateless VADs.
        """
        pass


class EnergyVAD(BaseVAD):
    """Frame-level Root-Mean-Square (RMS) energy-based Voice Activity Detector."""

    def reset(self) -> None:
        """Reset internal state. EnergyVAD is stateless, so this is a no-op."""
        pass

    def __init__(
        self,
        energy_threshold: float = 0.015,
        subframe_duration_seconds: float = 0.02,
    ) -> None:
        """Initialize EnergyVAD.

        Args:
            energy_threshold: RMS amplitude threshold above which a subframe is classified as speech.
            subframe_duration_seconds: Duration of analysis subframes in seconds (default: 20 ms).
        """
        if energy_threshold < 0:
            raise ValueError(f"energy_threshold must be non-negative, got {energy_threshold}.")
        if subframe_duration_seconds <= 0:
            raise ValueError(
                f"subframe_duration_seconds must be positive, got {subframe_duration_seconds}."
            )

        self.energy_threshold = float(energy_threshold)
        self.subframe_duration_seconds = float(subframe_duration_seconds)

    def compute_speech_duration(
        self,
        samples: np.ndarray[Any, np.dtype[np.float32]],
        sample_rate: int = 16000,
    ) -> float:
        """Compute speech duration by measuring RMS energy across contiguous subframes."""
        if samples.size == 0 or sample_rate <= 0:
            return 0.0

        subframe_samples = max(1, int(round(self.subframe_duration_seconds * sample_rate)))
        num_subframes = samples.size // subframe_samples
        if num_subframes == 0:
            # Single partial frame
            rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
            return (samples.size / sample_rate) if rms >= self.energy_threshold else 0.0

        # Reshape to 2D for vectorised subframe RMS calculation
        truncated_len = num_subframes * subframe_samples
        frames = samples[:truncated_len].reshape(num_subframes, subframe_samples).astype(np.float64)
        rms_per_frame = np.sqrt(np.mean(frames**2, axis=1))

        speech_subframes_count = int(np.sum(rms_per_frame >= self.energy_threshold))
        speech_duration = speech_subframes_count * (subframe_samples / sample_rate)

        # Handle any leftover trailing samples
        remainder = samples[truncated_len:]
        if remainder.size > 0:
            rms_rem = float(np.sqrt(np.mean(remainder.astype(np.float64) ** 2)))
            if rms_rem >= self.energy_threshold:
                speech_duration += remainder.size / sample_rate

        return float(speech_duration)
