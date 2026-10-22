"""Unit tests for deterministic audio loading, polyphase resampling, quality metrics, and crop/pad."""

import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from voiceshield.dsp import (
    CorruptAudioError,
    EmptyAudioError,
    InvalidAudioValueError,
    compute_quality_metrics,
    load_audio,
    pad_or_crop_waveform,
    resample_audio,
)


def _create_synthetic_wav(
    path_or_buf: Path | io.BytesIO,
    samples: np.ndarray[Any, Any],
    sample_rate: int,
    subtype: str = "FLOAT",
) -> None:
    """Helper to write synthetic audio fixtures in memory or on disk."""
    sf.write(path_or_buf, samples, sample_rate, format="WAV", subtype=subtype)


def _create_synthetic_flac(
    path_or_buf: Path | io.BytesIO,
    samples: np.ndarray[Any, Any],
    sample_rate: int,
) -> None:
    """Helper to write synthetic FLAC fixtures."""
    # FLAC requires integer PCM (PCM_16 or PCM_24)
    sf.write(path_or_buf, samples, sample_rate, format="FLAC", subtype="PCM_16")


def test_load_mono_wav_sine_wave(tmp_path: Path) -> None:
    """Verify loading a clean 1.0s 440 Hz mono sine wave at 16 kHz."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    sine = 0.7 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

    wav_file = tmp_path / "sine_440hz.wav"
    _create_synthetic_wav(wav_file, sine, sr)

    processed = load_audio(wav_file)

    assert processed.waveform.shape == (16000,)
    assert processed.waveform.dtype == np.float32
    assert -1.0 <= np.min(processed.waveform) <= np.max(processed.waveform) <= 1.0
    assert processed.metadata.duration_seconds == pytest.approx(1.0, abs=1e-4)
    assert processed.metadata.original_sample_rate == 16000
    assert processed.metadata.original_channels == 1
    assert processed.metadata.original_num_samples == 16000
    assert processed.metadata.resampled_num_samples == 16000
    assert processed.metadata.format == "WAV"
    assert not processed.metadata.pre_norm_quality.is_clipped
    assert not processed.metadata.pre_norm_quality.is_silent
    assert processed.metadata.pre_norm_quality.quality_status == "clean"


def test_load_stereo_flac_alternate_rate(tmp_path: Path) -> None:
    """Verify loading a 48 kHz stereo FLAC file with deterministic downmixing and resampling."""
    sr = 48000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    left = 0.5 * np.sin(2 * np.pi * 300.0 * t).astype(np.float32)
    right = 0.5 * np.cos(2 * np.pi * 600.0 * t).astype(np.float32)
    stereo = np.column_stack([left, right])

    flac_file = tmp_path / "stereo_48k.flac"
    _create_synthetic_flac(flac_file, stereo, sr)

    processed1 = load_audio(flac_file)
    processed2 = load_audio(flac_file)

    # Output must be exactly 16 kHz mono float32 for 2.0s = 32,000 samples
    assert processed1.waveform.shape == (32000,)
    assert processed1.waveform.dtype == np.float32
    assert processed1.metadata.original_sample_rate == 48000
    assert processed1.metadata.original_channels == 2
    assert processed1.metadata.format == "FLAC"

    # Verify bitwise determinism across repeated loads
    np.testing.assert_array_equal(processed1.waveform, processed2.waveform)


def test_load_from_bytes() -> None:
    """Verify loading WAV audio directly from in-memory byte buffer."""
    buf = io.BytesIO()
    samples = 0.4 * np.ones(8000, dtype=np.float32)
    _create_synthetic_wav(buf, samples, 16000)
    raw_bytes = buf.getvalue()

    processed = load_audio(raw_bytes)
    assert processed.waveform.shape == (8000,)
    assert processed.waveform.dtype == np.float32
    assert processed.metadata.original_num_samples == 8000


def test_channel_conversion_deterministic(tmp_path: Path) -> None:
    """Verify that multi-channel audio is downmixed via arithmetic mean in float32."""
    sr = 16000
    left = np.full(4000, 0.2, dtype=np.float32)
    right = np.full(4000, 0.6, dtype=np.float32)
    stereo = np.column_stack([left, right])

    wav_file = tmp_path / "constant_stereo.wav"
    _create_synthetic_wav(wav_file, stereo, sr)

    processed = load_audio(wav_file, normalize=False)
    # Expected mono value is (0.2 + 0.6) / 2 = 0.4
    np.testing.assert_allclose(processed.waveform, 0.4, atol=1e-5)


def test_silence_quality_diagnostics(tmp_path: Path) -> None:
    """Verify that silence fixture is identified as is_silent with zero clipping."""
    sr = 16000
    silence = np.zeros(16000, dtype=np.float32)
    wav_file = tmp_path / "silence.wav"
    _create_synthetic_wav(wav_file, silence, sr)

    processed = load_audio(wav_file)
    assert processed.metadata.pre_norm_quality.is_silent
    assert not processed.metadata.pre_norm_quality.is_clipped
    assert processed.metadata.pre_norm_quality.rms == 0.0
    assert processed.metadata.pre_norm_quality.quality_status == "silent"


def test_clipped_waveform_detection(tmp_path: Path) -> None:
    """Verify that a clipped waveform triggers is_clipped flag and status."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    # Severe clipping: amplitude 2.5 clamped to 1.0
    clipped = np.clip(2.5 * np.sin(2 * np.pi * 200.0 * t), -1.0, 1.0).astype(np.float32)

    wav_file = tmp_path / "clipped.wav"
    _create_synthetic_wav(wav_file, clipped, sr)

    processed = load_audio(wav_file)
    assert processed.metadata.pre_norm_quality.is_clipped
    assert processed.metadata.pre_norm_quality.clipping_fraction > 0.1
    assert processed.metadata.pre_norm_quality.quality_status == "clipped"


def test_resample_determinism_and_fidelity() -> None:
    """Verify polyphase resampler determinism across various standard sampling rates."""
    sr_in = 44100
    t = np.linspace(0, 0.5, int(sr_in * 0.5), endpoint=False, dtype=np.float32)
    x = 0.8 * np.sin(2 * np.pi * 1000.0 * t).astype(np.float32)

    y1 = resample_audio(x, orig_sr=44100, target_sr=16000)
    y2 = resample_audio(x, orig_sr=44100, target_sr=16000)

    np.testing.assert_array_equal(y1, y2)
    assert y1.shape == (8000,)
    assert y1.dtype == np.float32


def test_pad_or_crop_waveform_behavior() -> None:
    """Verify pad_or_crop_waveform under short, exact, and long input conditions."""
    target_len = 64000  # 4.0s @ 16 kHz

    # Case 1: Short audio (16,000 samples) -> pads 48,000 zeros at the end
    short = np.ones(16000, dtype=np.float32)
    padded = pad_or_crop_waveform(short, target_length=target_len)
    assert padded.shape == (target_len,)
    assert padded.dtype == np.float32
    np.testing.assert_array_equal(padded[:16000], 1.0)
    np.testing.assert_array_equal(padded[16000:], 0.0)

    # Case 2: Exact audio length -> unchanged
    exact = np.full(target_len, 0.5, dtype=np.float32)
    result_exact = pad_or_crop_waveform(exact, target_length=target_len)
    assert result_exact.shape == (target_len,)
    np.testing.assert_array_equal(result_exact, exact)

    # Case 3: Long audio (80,000 samples) -> head truncated to 64,000
    long_audio = np.arange(80000, dtype=np.float32)
    cropped = pad_or_crop_waveform(long_audio, target_length=target_len)
    assert cropped.shape == (target_len,)
    np.testing.assert_array_equal(cropped, long_audio[:target_len])

    # Case 4: Center crop
    cropped_center = pad_or_crop_waveform(long_audio, target_length=target_len, pad_mode="center")
    assert cropped_center.shape == (target_len,)
    np.testing.assert_array_equal(cropped_center, long_audio[8000:72000])

    # Case 5: Invalid target length
    with pytest.raises(InvalidAudioValueError, match="target_length must be positive"):
        pad_or_crop_waveform(short, target_length=0)


def test_error_handling_empty_inputs(tmp_path: Path) -> None:
    """Verify that empty files or byte payloads raise EmptyAudioError."""
    with pytest.raises(EmptyAudioError, match="payload is empty"):
        load_audio(b"")

    empty_file = tmp_path / "empty.wav"
    empty_file.touch()
    with pytest.raises(EmptyAudioError, match="file is empty"):
        load_audio(empty_file)


def test_error_handling_corrupt_bytes() -> None:
    """Verify that corrupt byte payload raises CorruptAudioError."""
    corrupt_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00garbage"
    with pytest.raises(CorruptAudioError, match="Failed to decode audio stream"):
        load_audio(corrupt_bytes)


def test_error_handling_nan_and_inf(tmp_path: Path) -> None:
    """Verify that audio containing NaN or Inf sample values raises InvalidAudioValueError."""
    sr = 16000
    samples_with_nan = np.array([0.1, 0.2, np.nan, 0.4], dtype=np.float32)
    nan_file = tmp_path / "nan.wav"
    _create_synthetic_wav(nan_file, samples_with_nan, sr)

    with pytest.raises(InvalidAudioValueError, match="NaN or Infinite"):
        load_audio(nan_file)

    samples_with_inf = np.array([0.1, 0.2, np.inf, 0.4], dtype=np.float32)
    inf_file = tmp_path / "inf.wav"
    _create_synthetic_wav(inf_file, samples_with_inf, sr)

    with pytest.raises(InvalidAudioValueError, match="NaN or Infinite"):
        load_audio(inf_file)


def test_quality_metrics_empty_array() -> None:
    """Verify quality calculation handles empty input gracefully."""
    empty = np.array([], dtype=np.float32)
    metrics = compute_quality_metrics(empty)
    assert metrics.quality_status == "empty"
    assert metrics.is_silent
