"""Deterministic offline file inference engine around the BaseDetector interface."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from voiceshield.config.schema import VoiceShieldConfig
from voiceshield.core.schema import Disclaimers
from voiceshield.dsp.exceptions import AudioPreprocessingError
from voiceshield.dsp.preprocessor import load_audio
from voiceshield.dsp.quality import AudioQualityMetrics, compute_quality_metrics
from voiceshield.models.base import BaseDetector, DetectorOutput


class WindowResult(BaseModel):
    """Evaluation result for an individual 4.0-second temporal window."""

    window_index: int = Field(description="Zero-indexed window sequence number.")
    start_seconds: float = Field(description="Window start offset in seconds.")
    end_seconds: float = Field(description="Window end offset in seconds.")
    raw_spoof_logit: float = Field(description="Raw model logit for spoof (index 0).")
    raw_bonafide_logit: float = Field(description="Raw model logit for bonafide (index 1).")
    raw_bonafide_score: float = Field(description="Raw bona fide logit score (ASVspoof standard).")
    calibrated_synthetic_likelihood: float = Field(
        ge=0.0,
        le=1.0,
        description="Probability of synthetic speech in [0.0, 1.0].",
    )
    is_synthetic_suspected: bool = Field(description="Binary advisory decision for this window.")
    active_speech_seconds: float = Field(description="Estimated active speech duration in window.")
    quality_status: AudioQualityMetrics = Field(
        description="Audio quality diagnostics for this window."
    )


class FileInferenceResult(BaseModel):
    """Aggregated offline inference result for an entire audio file."""

    file_path: str = Field(description="Source file path or identifier.")
    status: str = Field(
        default="success", description="Status ('success', 'error', 'unsupported')."
    )
    error_message: str | None = Field(
        default=None, description="Error details if status != 'success'."
    )
    duration_seconds: float = Field(default=0.0, description="Total duration of audio in seconds.")
    original_sample_rate: int = Field(default=0, description="Native sampling rate in Hz.")
    original_channels: int = Field(default=0, description="Native channel count.")
    total_windows: int = Field(default=0, description="Number of 4.0-second windows evaluated.")
    mean_synthetic_likelihood: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Average synthetic likelihood across analyzed windows.",
    )
    max_synthetic_likelihood: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Peak synthetic likelihood observed across all windows.",
    )
    is_synthetic_suspected: bool = Field(
        default=False,
        description="Overall file-level advisory classification.",
    )
    temporal_rule_explanation: str = Field(
        default="",
        description="Human-interpretable temporal aggregation explanation.",
    )
    windows: list[WindowResult] = Field(
        default_factory=list, description="Per-window evaluation records."
    )
    is_mock_detector: bool = Field(
        default=False,
        description="Whether evaluation was performed with mock model weights for test/UI.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Provenance, model tags, config hash, and checkpoint hashes.",
    )
    disclaimers: Disclaimers = Field(
        default_factory=Disclaimers,
        description="Mandatory compliance and product scope disclaimers.",
    )


def _compute_active_speech_duration(
    samples: np.ndarray[Any, np.dtype[np.float32]],
    sample_rate: int = 16000,
    frame_ms: float = 30.0,
    energy_threshold: float = 0.015,
) -> float:
    """Estimate active speech duration within a chunk using frame-level RMS energy."""
    frame_len = int(round(sample_rate * (frame_ms / 1000.0)))
    if len(samples) < frame_len:
        return 0.0

    num_frames = len(samples) // frame_len
    active_frames = 0
    for i in range(num_frames):
        frame = samples[i * frame_len : (i + 1) * frame_len]
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        if rms >= energy_threshold:
            active_frames += 1

    return round(float(active_frames * (frame_ms / 1000.0)), 2)


def evaluate_audio(
    source: str | Path | bytes | io.BytesIO,
    detector: BaseDetector,
    config: VoiceShieldConfig | None = None,
    hop_seconds: float = 1.0,
    window_seconds: float = 4.0,
    threshold: float = 0.65,
    file_name: str | None = None,
    max_duration_seconds: float | None = None,
) -> FileInferenceResult:
    """Evaluate an audio file using 4.0s sliding windows, reporting per-window scores and aggregation.

    Args:
        source: Audio file path, bytes, or buffer.
        detector: Instantiated BaseDetector implementation.
        config: Optional VoiceShieldConfig for provenance hashing.
        hop_seconds: Step size between consecutive windows in seconds (default: 1.0s).
        window_seconds: Duration of each analysis window in seconds (default: 4.0s).
        threshold: Decision threshold for synthetic suspicion (default: 0.65).
        file_name: Optional explicit file name override for reporting.
        max_duration_seconds: Optional ceiling on acceptable audio duration in seconds.

    Returns:
        Populated FileInferenceResult model.
    """
    path_str = file_name or (str(source) if isinstance(source, (str, Path)) else "in_memory_audio")

    # 1. Load and normalize audio via canonical preprocessor
    try:
        preprocessed = load_audio(source, target_sr=16000, max_duration_seconds=max_duration_seconds)
    except AudioPreprocessingError as exc:
        return FileInferenceResult(
            file_path=path_str,
            status="error",
            error_message=str(exc),
            disclaimers=Disclaimers(),
        )
    except Exception as exc:
        return FileInferenceResult(
            file_path=path_str,
            status="error",
            error_message=f"Unexpected error loading audio: {exc}",
            disclaimers=Disclaimers(),
        )

    waveform = preprocessed.waveform  # 1D float32 @ 16 kHz
    metadata = preprocessed.metadata

    if max_duration_seconds is not None and metadata.duration_seconds > max_duration_seconds:
        return FileInferenceResult(
            file_path=path_str,
            status="error",
            error_message=(
                f"Audio duration ({metadata.duration_seconds:.2f}s) exceeds "
                f"maximum allowed limit of {max_duration_seconds:.2f}s."
            ),
            duration_seconds=round(metadata.duration_seconds, 3),
            original_sample_rate=metadata.original_sample_rate,
            original_channels=metadata.original_channels,
            disclaimers=Disclaimers(),
        )

    total_samples = len(waveform)
    window_samples = int(round(window_seconds * 16000))  # 64,000 samples
    hop_samples = int(round(hop_seconds * 16000))  # 16,000 samples

    window_slices: list[tuple[int, int]] = []
    if total_samples <= window_samples:
        window_slices.append((0, total_samples))
    else:
        start_idx = 0
        while start_idx + window_samples <= total_samples:
            window_slices.append((start_idx, start_idx + window_samples))
            start_idx += hop_samples

        # If significant audio remains after final hop (> 1.0s) and not covered, append last window
        if (
            total_samples - start_idx > 16000
            and (total_samples - window_samples) > window_slices[-1][0]
        ):
            window_slices.append((total_samples - window_samples, total_samples))

    windows: list[WindowResult] = []
    for idx, (s_idx, e_idx) in enumerate(window_slices):
        chunk = waveform[s_idx:e_idx]
        start_sec = round(s_idx / 16000.0, 3)
        end_sec = round(e_idx / 16000.0, 3)

        chunk_quality = compute_quality_metrics(chunk)
        speech_sec = _compute_active_speech_duration(chunk)

        # Run inference through BaseDetector
        det_output: DetectorOutput = detector.predict(chunk)

        windows.append(
            WindowResult(
                window_index=idx,
                start_seconds=start_sec,
                end_seconds=end_sec,
                raw_spoof_logit=det_output.raw_logits[0],
                raw_bonafide_logit=det_output.raw_logits[1],
                raw_bonafide_score=det_output.bonafide_score,
                calibrated_synthetic_likelihood=det_output.synthetic_likelihood,
                is_synthetic_suspected=det_output.is_synthetic_suspected,
                active_speech_seconds=speech_sec,
                quality_status=chunk_quality,
            )
        )

    # 2. File-level aggregation
    mean_likelihood = float(np.mean([w.calibrated_synthetic_likelihood for w in windows]))
    max_likelihood = float(np.max([w.calibrated_synthetic_likelihood for w in windows]))
    suspected_count = sum(1 for w in windows if w.is_synthetic_suspected)
    num_windows = len(windows)

    # Temporal decision policy:
    # Requires at least 2 consecutive/sustained suspected windows if multiple windows exist,
    # or the single window if audio duration <= 4.0s
    if num_windows == 1:
        is_suspected = bool(windows[0].is_synthetic_suspected)
    else:
        is_suspected = bool(suspected_count >= 2 and max_likelihood >= threshold)

    if is_suspected:
        explanation = (
            f"Advisory Warning: High synthetic speech likelihood sustained across "
            f"{suspected_count} of {num_windows} window(s) (Peak likelihood: {max_likelihood:.1%}). "
            "Recommend independent secondary verification. Do not automatically terminate or block calls."
        )
    else:
        explanation = (
            f"Consistent with authentic speech across all {num_windows} evaluated window(s) "
            f"(Mean synthetic likelihood: {mean_likelihood:.1%}). "
            "Advisory metric only; does NOT prove identity of caller."
        )

    # 3. Provenance and configuration hashes
    is_mock = getattr(detector, "is_mock", False)
    res_metadata: dict[str, Any] = {
        "model_id": getattr(detector, "model_id", detector.__class__.__name__),
        "device_used": str(getattr(detector, "device", "unknown")),
        "config_hash": config.compute_hash() if config else None,
        "pre_norm_peak": metadata.pre_norm_quality.peak_level,
        "pre_norm_rms": metadata.pre_norm_quality.rms,
        "pre_norm_status": metadata.pre_norm_quality.quality_status,
        "format": metadata.format,
        "subtype": metadata.subtype,
        "is_mock_detector": is_mock,
    }

    return FileInferenceResult(
        file_path=path_str,
        status="success",
        duration_seconds=round(metadata.duration_seconds, 3),
        original_sample_rate=metadata.original_sample_rate,
        original_channels=metadata.original_channels,
        total_windows=num_windows,
        mean_synthetic_likelihood=round(mean_likelihood, 4),
        max_synthetic_likelihood=round(max_likelihood, 4),
        is_synthetic_suspected=is_suspected,
        temporal_rule_explanation=explanation,
        windows=windows,
        is_mock_detector=is_mock,
        metadata=res_metadata,
        disclaimers=Disclaimers(),
    )
