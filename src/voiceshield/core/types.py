"""Audio DSP constants and contract data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np

# Strict Audio Contract Constants
EXPECTED_SAMPLE_RATE: Final[int] = 16000
EXPECTED_CHANNELS: Final[int] = 1
WINDOW_SECONDS: Final[float] = 4.0
HOP_SECONDS: Final[float] = 1.0
WINDOW_SAMPLES: Final[int] = int(WINDOW_SECONDS * EXPECTED_SAMPLE_RATE)  # 64,000
HOP_SAMPLES: Final[int] = int(HOP_SECONDS * EXPECTED_SAMPLE_RATE)  # 16,000
DEFAULT_MIN_SPEECH_SECONDS: Final[float] = 0.8


@dataclass(frozen=True)
class AudioContract:
    """Specification of the input audio format accepted by VoiceShield."""

    sample_rate: int = EXPECTED_SAMPLE_RATE
    channels: int = EXPECTED_CHANNELS
    dtype: str = "float32"
    min_val: float = -1.0
    max_val: float = 1.0
    window_samples: int = WINDOW_SAMPLES
    hop_samples: int = HOP_SAMPLES


@dataclass(frozen=True)
class AudioWindow:
    """A verified 4.0-second window ready for feature extraction and inference."""

    samples: np.ndarray[Any, np.dtype[np.float32]]  # 1D float32 array of shape (64000,)
    start_sample: int
    end_sample: int
    start_time_seconds: float
    end_time_seconds: float
    active_speech_seconds: float
    speech_ratio: float
