"""Pure DSP operations: normalization, resampling, quality metrics, and audio contracts."""

from voiceshield.dsp.exceptions import (
    AudioDurationLimitError,
    AudioPreprocessingError,
    CorruptAudioError,
    EmptyAudioError,
    InvalidAudioValueError,
    UnsupportedAudioError,
)
from voiceshield.dsp.preprocessor import (
    AudioMetadata,
    PreprocessedAudio,
    load_audio,
    pad_or_crop_waveform,
)
from voiceshield.dsp.quality import AudioQualityMetrics, compute_quality_metrics
from voiceshield.dsp.reliability import (
    InputReliabilityAssessment,
    ReliabilityReason,
    ReliabilityState,
    assess_input_reliability,
)
from voiceshield.dsp.resampler import resample_audio

__all__ = [
    "AudioMetadata",
    "AudioPreprocessingError",
    "AudioDurationLimitError",
    "AudioQualityMetrics",
    "CorruptAudioError",
    "EmptyAudioError",
    "InputReliabilityAssessment",
    "InvalidAudioValueError",
    "PreprocessedAudio",
    "ReliabilityReason",
    "ReliabilityState",
    "UnsupportedAudioError",
    "assess_input_reliability",
    "compute_quality_metrics",
    "load_audio",
    "pad_or_crop_waveform",
    "resample_audio",
]
