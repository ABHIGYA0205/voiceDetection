"""Deterministic, high-quality audio resampling via polyphase Kaiser-windowed FIR filtering."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import scipy.signal

from voiceshield.dsp.exceptions import InvalidAudioValueError, UnsupportedAudioError

KAISER_BETA_DEFAULT: float = 5.0


def resample_audio(
    audio: np.ndarray[Any, np.dtype[np.float32]],
    orig_sr: int,
    target_sr: int = 16000,
    kaiser_beta: float = KAISER_BETA_DEFAULT,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Resample 1D mono audio to target_sr using polyphase filtering with a Kaiser window.

    Args:
        audio: 1D float32 mono audio array.
        orig_sr: Source sampling rate in Hz.
        target_sr: Target sampling rate in Hz (default: 16,000 Hz).
        kaiser_beta: Shape parameter for the Kaiser window (default: 5.0).

    Returns:
        1D float32 audio array resampled to target_sr.

    Raises:
        UnsupportedAudioError: If sample rate is non-positive.
        InvalidAudioValueError: If input audio contains non-finite values.
    """
    if orig_sr <= 0 or target_sr <= 0:
        raise UnsupportedAudioError(
            f"Sampling rates must be positive, got orig_sr={orig_sr}, target_sr={target_sr}."
        )

    if audio.ndim != 1:
        raise UnsupportedAudioError(
            f"resample_audio requires 1D mono audio, got shape {audio.shape}."
        )

    if not np.all(np.isfinite(audio)):
        raise InvalidAudioValueError(
            "Audio array contains NaN or Infinite values before resampling."
        )

    if orig_sr == target_sr:
        return np.ascontiguousarray(audio, dtype=np.float32)

    # Calculate exact rational resampling factors
    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd

    # Apply polyphase filtering with Kaiser window
    resampled: np.ndarray[Any, np.dtype[np.float32]] = scipy.signal.resample_poly(
        audio,
        up=up,
        down=down,
        window=("kaiser", kaiser_beta),
    ).astype(np.float32)

    return np.ascontiguousarray(resampled, dtype=np.float32)
