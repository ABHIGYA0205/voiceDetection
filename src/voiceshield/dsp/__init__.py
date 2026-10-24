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
from voiceshield.dsp.resampler import resample_audio

__all__ = [
    "AudioMetadata",
    "AudioPreprocessingError",
    "AudioDurationLimitError",
    "AudioQualityMetrics",
    "CorruptAudioError",
    "EmptyAudioError",
    "InvalidAudioValueError",
    "PreprocessedAudio",
    "UnsupportedAudioError",
    "compute_quality_metrics",
    "load_audio",
    "pad_or_crop_waveform",
    "resample_audio",
]
