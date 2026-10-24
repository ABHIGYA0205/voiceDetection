"""Deterministic audio preprocessor for VoiceShield.

Enforces 16,000 Hz float32 mono [-1.0, 1.0] audio contract with untouched pre-normalization
quality diagnostics, deterministic channel averaging, polyphase resampling, and strict
error rejection.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from pydantic import BaseModel, Field

from voiceshield.dsp.exceptions import (
    AudioDurationLimitError,
    CorruptAudioError,
    EmptyAudioError,
    InvalidAudioValueError,
    UnsupportedAudioError,
)
from voiceshield.dsp.quality import AudioQualityMetrics, compute_quality_metrics
from voiceshield.dsp.resampler import resample_audio

SUPPORTED_FORMATS = ("WAV", "FLAC")


class AudioMetadata(BaseModel):
    """Provenance and signal diagnostics for an audio signal."""

    duration_seconds: float = Field(description="Signal duration in seconds.")
    original_sample_rate: int = Field(description="Original sample rate in Hz.")
    original_channels: int = Field(description="Original number of channels.")
    original_num_samples: int = Field(description="Original sample count before resampling.")
    resampled_num_samples: int = Field(description="Sample count after resampling to 16 kHz.")
    format: str = Field(description="Container format (e.g. WAV, FLAC).")
    subtype: str = Field(description="Encoding subtype (e.g. PCM_16, FLOAT).")
    pre_norm_quality: AudioQualityMetrics = Field(
        description="Untouched quality metrics measured directly from raw input.",
    )
    post_norm_quality: AudioQualityMetrics = Field(
        description="Quality metrics measured after channel mixing, resampling, and normalization.",
    )


class PreprocessedAudio(BaseModel):
    """Output bundle containing the 16 kHz float32 mono waveform and metadata."""

    waveform: np.ndarray[Any, np.dtype[np.float32]]
    metadata: AudioMetadata

    model_config = {"arbitrary_types_allowed": True}


def load_audio(
    source: str | Path | bytes | io.BytesIO,
    target_sr: int = 16000,
    normalize: bool = False,
    max_duration_seconds: float | None = None,
) -> PreprocessedAudio:
    """Load WAV or FLAC from path or bytes and normalize to 16 kHz mono float32 [-1.0, 1.0].

    Transformations applied in strict deterministic order:
    1. Input container validation (WAV/FLAC only).
    2. Header inspection & resource limit pre-checks (sampling rate, duration ceiling).
    3. Raw sample decoding to float32 (bounded to prevent decompression bombs).
    4. Rejection of empty or non-finite (NaN, Inf) samples.
    5. Amplitude validation: strictly enforce [-1.0, 1.0] domain (unless legacy normalize=True).
    6. Untouched pre-normalization quality measurement.
    7. Deterministic multi-channel to mono conversion via arithmetic mean in float32.
    8. Canonical stateful sinc resampling to target_sr (16 kHz) via process + flush.
    9. Optional legacy peak normalization to [-1.0, 1.0] (non-parity opt-in only).
    10. Post-normalization quality measurement.

    Args:
        source: File path (str/Path), raw bytes, or BytesIO buffer.
        target_sr: Target sample rate in Hz (default: 16,000 Hz).
        normalize: Legacy opt-in non-parity flag. If True, out-of-range floating-point audio
            is peak-normalized instead of rejected. Canonical parity path requires False (default).
        max_duration_seconds: Optional ceiling on acceptable audio duration in seconds.

    Returns:
        PreprocessedAudio bundle.

    Raises:
        EmptyAudioError: If input is empty (0 bytes or 0 samples).
        CorruptAudioError: If file header or audio payload is corrupt.
        UnsupportedAudioError: If format is not WAV/FLAC, rate is out of bounds, or duration limit is exceeded.
        InvalidAudioValueError: If samples contain NaN, Infinite, or out-of-range values.
    """
    # 1. Format inspection and source preparation
    bio: io.BytesIO | Path
    if isinstance(source, bytes):
        if len(source) == 0:
            raise EmptyAudioError("Provided audio bytes payload is empty (0 bytes).")
        bio = io.BytesIO(source)
    elif isinstance(source, io.BytesIO):
        if source.getbuffer().nbytes == 0:
            raise EmptyAudioError("Provided BytesIO buffer is empty (0 bytes).")
        source.seek(0)
        bio = source
    elif isinstance(source, (str, Path)):
        p = Path(source)
        if not p.is_file():
            raise CorruptAudioError(f"Audio file does not exist: {p}")
        if p.stat().st_size == 0:
            raise EmptyAudioError(f"Audio file is empty (0 bytes): {p}")
        bio = p
    else:
        raise UnsupportedAudioError(f"Unsupported audio source type: {type(source).__name__}")

    # 2. Read with soundfile and enforce resource limits
    try:
        with sf.SoundFile(bio) as sfile:
            fmt = sfile.format
            subtype = sfile.subtype
            orig_sr = sfile.samplerate
            orig_channels = sfile.channels

            if fmt not in SUPPORTED_FORMATS:
                raise UnsupportedAudioError(
                    f"Unsupported audio format: {fmt}. Only {SUPPORTED_FORMATS} are permitted."
                )

            if orig_sr < 8000 or orig_sr > 192000:
                raise UnsupportedAudioError(
                    f"Unsupported sampling rate: {orig_sr} Hz. Permitted range is 8,000 Hz to 192,000 Hz."
                )

            if orig_channels < 1 or orig_channels > 8:
                raise UnsupportedAudioError(
                    f"Unsupported channel count: {orig_channels}. Only 1 to 8 channels are supported."
                )

            # Pre-check header duration to prevent decompression memory bombs
            if sfile.frames > 0:
                header_duration = float(sfile.frames / orig_sr)
                if max_duration_seconds is not None and header_duration > max_duration_seconds:
                    raise AudioDurationLimitError(
                        f"Audio duration ({header_duration:.2f}s) exceeds maximum allowed limit of {max_duration_seconds:.2f}s."
                    )

            # Bound read length if max_duration_seconds is set
            frames_to_read = -1
            if max_duration_seconds is not None:
                frames_to_read = int(orig_sr * (max_duration_seconds + 1.0))

            raw_samples: np.ndarray[Any, np.dtype[np.float32]] = sfile.read(
                frames=frames_to_read,
                dtype="float32",
                always_2d=True,
            )
    except sf.LibsndfileError as exc:
        raise CorruptAudioError(f"Failed to decode audio stream: {exc}") from exc
    except Exception as exc:
        if isinstance(exc, (EmptyAudioError, UnsupportedAudioError, CorruptAudioError)):
            raise
        raise CorruptAudioError(f"Unexpected error decoding audio: {exc}") from exc

    # 3. Validation: zero samples and actual duration bounds
    num_samples = raw_samples.shape[0]
    if num_samples == 0:
        raise EmptyAudioError("Audio stream contains zero audio frames.")

    actual_duration = float(num_samples / orig_sr)
    if max_duration_seconds is not None and actual_duration > max_duration_seconds:
        raise AudioDurationLimitError(
            f"Audio duration ({actual_duration:.2f}s) exceeds maximum allowed limit of {max_duration_seconds:.2f}s."
        )

    # 4. Validation: finite numbers (reject NaN / Inf)
    if not np.all(np.isfinite(raw_samples)):
        raise InvalidAudioValueError("Audio stream contains NaN or Infinite sample values.")

    # 5. Amplitude validation: strictly enforce [-1.0, 1.0] domain
    if not normalize and (np.any(raw_samples < -1.0) or np.any(raw_samples > 1.0)):
        raise InvalidAudioValueError("Decoded float audio samples out of valid range [-1.0, 1.0].")

    # 6. Untouched pre-normalization quality measurement
    pre_norm_quality = compute_quality_metrics(raw_samples)

    # 7. Deterministic multi-channel to mono conversion via arithmetic mean
    if orig_channels == 1:
        mono_samples = raw_samples[:, 0]
    else:
        mono_samples = np.mean(raw_samples, axis=1, dtype=np.float32)

    # 8. Canonical stateful sinc resampling
    resampled = resample_audio(mono_samples, orig_sr=orig_sr, target_sr=target_sr)

    # 9. Legacy opt-in peak normalization to [-1.0, 1.0]
    if normalize:
        peak = float(np.max(np.abs(resampled)))
        if peak > 1.0:
            resampled = resampled / peak
        np.clip(resampled, -1.0, 1.0, out=resampled)

    resampled = np.ascontiguousarray(resampled, dtype=np.float32)

    # 10. Post-normalization quality measurement
    post_norm_quality = compute_quality_metrics(resampled)

    duration = float(num_samples / orig_sr)
    metadata = AudioMetadata(
        duration_seconds=duration,
        original_sample_rate=orig_sr,
        original_channels=orig_channels,
        original_num_samples=num_samples,
        resampled_num_samples=len(resampled),
        format=fmt,
        subtype=subtype,
        pre_norm_quality=pre_norm_quality,
        post_norm_quality=post_norm_quality,
    )

    return PreprocessedAudio(waveform=resampled, metadata=metadata)


def pad_or_crop_waveform(
    waveform: np.ndarray[Any, np.dtype[np.float32]],
    target_length: int = 64000,
    pad_mode: str = "right_zero",
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Deterministically crop or pad a 1D waveform to exact target_length.

    Args:
        waveform: 1D float32 audio waveform.
        target_length: Expected length in samples (e.g. 64,000 for 4.0s @ 16 kHz).
        pad_mode: Padding/cropping mode ('right_zero' or 'center').

    Returns:
        1D float32 array with exactly target_length samples.

    Raises:
        InvalidAudioValueError: If target_length <= 0 or waveform is not 1D.
    """
    if target_length <= 0:
        raise InvalidAudioValueError(f"target_length must be positive, got {target_length}.")

    if waveform.ndim != 1:
        raise InvalidAudioValueError(f"waveform must be 1D, got shape {waveform.shape}.")

    current_length = waveform.shape[0]

    if current_length == target_length:
        return np.ascontiguousarray(waveform.copy(), dtype=np.float32)

    if current_length > target_length:
        if pad_mode == "center":
            start = (current_length - target_length) // 2
            cropped = waveform[start : start + target_length]
        else:
            # Deterministic front truncate / head crop
            cropped = waveform[:target_length]
        return np.ascontiguousarray(cropped, dtype=np.float32)

    # current_length < target_length: pad
    pad_amount = target_length - current_length
    if pad_mode == "center":
        pad_left = pad_amount // 2
        pad_right = pad_amount - pad_left
        padded = np.pad(waveform, (pad_left, pad_right), mode="constant", constant_values=0.0)
    else:
        # Default right-padding with zeros
        padded = np.pad(waveform, (0, pad_amount), mode="constant", constant_values=0.0)

    return np.ascontiguousarray(padded, dtype=np.float32)
