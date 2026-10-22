"""Custom exception hierarchy for audio loading and DSP preprocessing."""

from __future__ import annotations


class AudioPreprocessingError(Exception):
    """Base exception for all audio loading and DSP preprocessing errors."""


class EmptyAudioError(AudioPreprocessingError):
    """Raised when the audio payload or file contains zero samples or empty byte stream."""


class CorruptAudioError(AudioPreprocessingError):
    """Raised when audio file headers or containers cannot be decoded."""


class InvalidAudioValueError(AudioPreprocessingError):
    """Raised when audio contains NaN, Infinite, or out-of-domain numeric values."""


class UnsupportedAudioError(AudioPreprocessingError):
    """Raised when an audio format, channel configuration, or sample rate is not supported."""
