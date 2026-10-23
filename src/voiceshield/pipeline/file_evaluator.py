"""Deterministic offline file inference engine around the BaseDetector interface."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, model_validator

from voiceshield.config.schema import VoiceShieldConfig
from voiceshield.core.schema import Disclaimers
from voiceshield.dsp.exceptions import AudioPreprocessingError
from voiceshield.dsp.preprocessor import load_audio
from voiceshield.dsp.quality import AudioQualityMetrics, compute_quality_metrics
from voiceshield.models.base import BaseDetector, DetectorOutput


class WindowResult(BaseModel):
    """Evaluation result for an individual temporal window."""

    window_index: int = Field(description="Zero-indexed window sequence number.")
    start_seconds: float = Field(description="Window start offset in seconds.")
    end_seconds: float = Field(description="Window end offset in seconds.")
    raw_spoof_logit: float = Field(description="Raw model logit for spoof (index 0).")
    raw_bonafide_logit: float = Field(description="Raw model logit for bonafide (index 1).")
    raw_bonafide_score: float = Field(description="Raw bona fide logit score (ASVspoof standard).")
    synthetic_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Uncalibrated acoustic anomaly score for synthetic/spoof speech in [0.0, 1.0].",
    )
    uncalibrated_synthetic_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Uncalibrated acoustic anomaly score for synthetic/spoof speech.",
    )
    score_is_calibrated: bool = Field(
        default=False,
        description="Whether score is calibrated against an empirical target dataset.",
    )
    calibrated_synthetic_likelihood: float | None = Field(
        default=None,
        description="Deprecated compatibility alias. Null when no calibrator is loaded.",
    )
    effective_threshold: float = Field(
        default=0.65,
        description="Effective decision threshold applied to this window.",
    )
    is_synthetic_suspected: bool = Field(description="Binary advisory decision for this window.")
    active_speech_seconds: float = Field(description="Estimated active speech duration in window.")
    is_eligible: bool = Field(
        default=True,
        description="Whether window has sufficient active speech duration for inference.",
    )
    is_partial: bool = Field(
        default=False,
        description="Whether window duration is partial / cyclically padded.",
    )
    abstain_reason: str | None = Field(
        default=None,
        description="Reason code if window was abstained from model evaluation.",
    )
    quality_status: AudioQualityMetrics = Field(
        description="Audio quality diagnostics for this window."
    )

    @model_validator(mode="before")
    @classmethod
    def sync_window_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            score = data.get("synthetic_score")
            if score is None:
                score = data.get("uncalibrated_synthetic_score")
            if score is None:
                score = data.get("calibrated_synthetic_likelihood")
            if score is not None:
                data.setdefault("synthetic_score", score)
                data.setdefault("uncalibrated_synthetic_score", score)
        return data


class FileInferenceResult(BaseModel):
    """Aggregated offline inference result for an entire audio file."""

    file_path: str = Field(description="Source file path or identifier.")
    status: str = Field(
        default="success",
        description="Status ('success', 'error', 'insufficient_evidence', 'unsupported').",
    )
    error_message: str | None = Field(
        default=None, description="Error details if status == 'error'."
    )
    abstain_reason: str | None = Field(
        default=None, description="Reason details if status == 'insufficient_evidence'."
    )
    duration_seconds: float = Field(default=0.0, description="Total duration of audio in seconds.")
    active_speech_duration_seconds: float = Field(
        default=0.0, description="Estimated total active speech duration in seconds."
    )
    original_sample_rate: int = Field(default=0, description="Native sampling rate in Hz.")
    original_channels: int = Field(default=0, description="Native channel count.")
    total_windows: int = Field(default=0, description="Number of windows evaluated.")
    eligible_windows: int = Field(
        default=0, description="Number of windows with sufficient active speech."
    )
    mean_synthetic_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Average synthetic score across eligible analyzed windows.",
    )
    max_synthetic_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Peak synthetic score observed across eligible analyzed windows.",
    )
    mean_synthetic_likelihood: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Deprecated alias for mean_synthetic_score. Uncalibrated score, not probability.",
    )
    max_synthetic_likelihood: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Deprecated alias for max_synthetic_score. Uncalibrated score, not probability.",
    )
    score_is_calibrated: bool = Field(
        default=False,
        description="Whether scores are calibrated against an empirical target dataset.",
    )
    effective_threshold: float = Field(
        default=0.65,
        description="Effective threshold applied for all window decisions.",
    )
    operating_threshold_source: str = Field(
        default="unvalidated_heuristic",
        description="Provenance of operating threshold.",
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

    @model_validator(mode="before")
    @classmethod
    def sync_file_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            mean_s = data.get("mean_synthetic_score")
            if mean_s is None:
                mean_s = data.get("mean_synthetic_likelihood", 0.0)
            data.setdefault("mean_synthetic_score", mean_s)
            data.setdefault("mean_synthetic_likelihood", mean_s)

            max_s = data.get("max_synthetic_score")
            if max_s is None:
                max_s = data.get("max_synthetic_likelihood", 0.0)
            data.setdefault("max_synthetic_score", max_s)
            data.setdefault("max_synthetic_likelihood", max_s)
        return data


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
    min_speech_seconds: float | None = None,
) -> FileInferenceResult:
    """Evaluate an audio file using sliding windows, reporting per-window scores and aggregation.

    Args:
        source: Audio file path, bytes, or buffer.
        detector: Instantiated BaseDetector implementation.
        config: Optional VoiceShieldConfig for provenance hashing.
        hop_seconds: Step size between consecutive windows in seconds (default: 1.0s).
        window_seconds: Duration of each analysis window in seconds (default: 4.0s).
        threshold: Request-level decision threshold for synthetic suspicion (default: 0.65).
        file_name: Optional explicit file name override for reporting.
        max_duration_seconds: Optional ceiling on acceptable audio duration in seconds.
        min_speech_seconds: Minimum active speech seconds required per window and file (default: 0.8s).

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

    effective_min_speech = (
        min_speech_seconds
        if min_speech_seconds is not None
        else (config.windowing.min_speech_seconds if config is not None else 0.8)
    )
    total_active_speech = round(_compute_active_speech_duration(waveform), 2)

    total_samples = len(waveform)
    window_samples = int(round(window_seconds * 16000))  # 64,000 samples
    hop_samples = int(round(hop_seconds * 16000))  # 16,000 samples

    window_slices: list[tuple[int, int, bool]] = []
    if total_samples <= window_samples:
        is_partial = total_samples < window_samples
        window_slices.append((0, total_samples, is_partial))
    else:
        start_idx = 0
        while start_idx + window_samples <= total_samples:
            window_slices.append((start_idx, start_idx + window_samples, False))
            start_idx += hop_samples

    windows: list[WindowResult] = []
    for idx, (s_idx, e_idx, is_partial) in enumerate(window_slices):
        chunk = waveform[s_idx:e_idx]
        start_sec = round(s_idx / 16000.0, 3)
        end_sec = round(e_idx / 16000.0, 3)

        chunk_quality = compute_quality_metrics(chunk)
        speech_sec = _compute_active_speech_duration(chunk)

        if speech_sec < effective_min_speech:
            # Task E: Insufficient speech: abstain window before inference/decision
            windows.append(
                WindowResult(
                    window_index=idx,
                    start_seconds=start_sec,
                    end_seconds=end_sec,
                    raw_spoof_logit=0.0,
                    raw_bonafide_logit=0.0,
                    raw_bonafide_score=0.0,
                    synthetic_score=0.0,
                    uncalibrated_synthetic_score=0.0,
                    score_is_calibrated=False,
                    calibrated_synthetic_likelihood=None,
                    effective_threshold=threshold,
                    is_synthetic_suspected=False,
                    active_speech_seconds=speech_sec,
                    is_eligible=False,
                    is_partial=is_partial,
                    abstain_reason="insufficient_speech_duration",
                    quality_status=chunk_quality,
                )
            )
        else:
            # Run inference through BaseDetector
            det_output: DetectorOutput = detector.predict(chunk)
            score = float(det_output.synthetic_score)

            # Task B: Calculate per-window decision using request effective threshold
            is_suspected = bool(score >= threshold)

            windows.append(
                WindowResult(
                    window_index=idx,
                    start_seconds=start_sec,
                    end_seconds=end_sec,
                    raw_spoof_logit=det_output.raw_logits[0],
                    raw_bonafide_logit=det_output.raw_logits[1],
                    raw_bonafide_score=det_output.bonafide_score,
                    synthetic_score=score,
                    uncalibrated_synthetic_score=score,
                    score_is_calibrated=False,
                    calibrated_synthetic_likelihood=None,
                    effective_threshold=threshold,
                    is_synthetic_suspected=is_suspected,
                    active_speech_seconds=speech_sec,
                    is_eligible=True,
                    is_partial=is_partial,
                    abstain_reason=None,
                    quality_status=chunk_quality,
                )
            )

    eligible_windows = [w for w in windows if w.is_eligible]
    num_windows = len(windows)
    num_eligible = len(eligible_windows)

    is_mock = getattr(detector, "is_mock", False)
    res_metadata: dict[str, Any] = {
        "model_id": getattr(detector, "model_id", detector.__class__.__name__),
        "device_used": str(getattr(detector, "device", "unknown")),
        "effective_threshold": threshold,
        "operating_threshold_source": "unvalidated_heuristic",
        "score_is_calibrated": False,
        "active_speech_duration_seconds": total_active_speech,
        "min_speech_threshold_seconds": effective_min_speech,
        "config_hash": config.compute_hash() if config else None,
        "pre_norm_peak": metadata.pre_norm_quality.peak_level,
        "pre_norm_rms": metadata.pre_norm_quality.rms,
        "pre_norm_status": metadata.pre_norm_quality.quality_status,
        "format": metadata.format,
        "subtype": metadata.subtype,
        "is_mock_detector": is_mock,
    }

    # Task E: If no windows have sufficient active speech, return insufficient_evidence / abstain
    if num_eligible == 0:
        explanation = (
            f"Insufficient active speech detected across file ({total_active_speech:.2f}s active speech "
            f"below required minimum threshold of {effective_min_speech:.2f}s). Decision abstained — does "
            "not evaluate synthetic anomalies without active speech. (Note: Energy VAD is a heuristic, "
            "not an infallible speech detector)."
        )
        return FileInferenceResult(
            file_path=path_str,
            status="insufficient_evidence",
            abstain_reason="insufficient_speech_duration",
            duration_seconds=round(metadata.duration_seconds, 3),
            active_speech_duration_seconds=total_active_speech,
            original_sample_rate=metadata.original_sample_rate,
            original_channels=metadata.original_channels,
            total_windows=num_windows,
            eligible_windows=0,
            mean_synthetic_score=0.0,
            max_synthetic_score=0.0,
            mean_synthetic_likelihood=0.0,
            max_synthetic_likelihood=0.0,
            score_is_calibrated=False,
            effective_threshold=threshold,
            operating_threshold_source="unvalidated_heuristic",
            is_synthetic_suspected=False,
            temporal_rule_explanation=explanation,
            windows=windows,
            is_mock_detector=is_mock,
            metadata=res_metadata,
            disclaimers=Disclaimers(),
        )

    mean_score = float(np.mean([w.synthetic_score for w in eligible_windows]))
    max_score = float(np.max([w.synthetic_score for w in eligible_windows]))

    # Task D: Actual consecutive-window logic
    # Requires at least 2 consecutive sustained hop-aligned windows if multiple windows exist,
    # or the single window if only 1 window exists.
    if num_windows == 1:
        is_suspected = bool(windows[0].is_eligible and windows[0].is_synthetic_suspected)
    else:
        max_consecutive = 0
        current_consecutive = 0
        prev_start_sec: float | None = None
        for w in windows:
            if w.is_eligible and w.is_synthetic_suspected:
                if current_consecutive > 0:
                    # Guard against near-duplicate or heavily overlapping non-hop windows
                    hop_diff = w.start_seconds - (prev_start_sec if prev_start_sec is not None else 0.0)
                    if hop_diff < (hop_seconds * 0.9):
                        continue
                current_consecutive += 1
                prev_start_sec = w.start_seconds
                if current_consecutive > max_consecutive:
                    max_consecutive = current_consecutive
            else:
                current_consecutive = 0
                prev_start_sec = None

        is_suspected = bool(max_consecutive >= 2)

    if is_suspected:
        explanation = (
            f"Advisory Warning: Elevated synthetic anomaly score sustained across at least 2 "
            f"consecutive windows (Peak score: {max_score:.3f} >= threshold {threshold:.3f}). "
            "Pause and verify through a known independent channel. Do not automatically terminate or block calls."
        )
    else:
        if any(w.is_synthetic_suspected for w in eligible_windows):
            explanation = (
                f"Isolated elevated window(s) detected, but synthetic anomaly was not sustained across "
                f"consecutive elevated windows (max consecutive = {max_consecutive if num_windows > 1 else 1}, required = 2). "
                f"Low anomaly overall across {num_eligible} evaluated window(s) (Mean score: {mean_score:.3f}, Peak score: {max_score:.3f} / threshold {threshold:.3f}). "
                "Advisory metric only; does NOT prove identity of caller."
            )
        else:
            explanation = (
                f"Low acoustic anomaly score — does not authenticate speech across {num_eligible} evaluated "
                f"window(s) (Mean score: {mean_score:.3f}, Peak score: {max_score:.3f} / threshold {threshold:.3f}). "
                "Advisory metric only; does NOT prove identity of caller."
            )

    return FileInferenceResult(
        file_path=path_str,
        status="success",
        duration_seconds=round(metadata.duration_seconds, 3),
        active_speech_duration_seconds=total_active_speech,
        original_sample_rate=metadata.original_sample_rate,
        original_channels=metadata.original_channels,
        total_windows=num_windows,
        eligible_windows=num_eligible,
        mean_synthetic_score=round(mean_score, 4),
        max_synthetic_score=round(max_score, 4),
        mean_synthetic_likelihood=round(mean_score, 4),
        max_synthetic_likelihood=round(max_score, 4),
        score_is_calibrated=False,
        effective_threshold=threshold,
        operating_threshold_source="unvalidated_heuristic",
        is_synthetic_suspected=is_suspected,
        temporal_rule_explanation=explanation,
        windows=windows,
        is_mock_detector=is_mock,
        metadata=res_metadata,
        disclaimers=Disclaimers(),
    )
