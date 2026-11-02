"""Deterministic offline file inference engine around the BaseDetector interface."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, model_validator

from voiceshield.config.overrides import apply_config_overrides
from voiceshield.config.schema import VoiceShieldConfig
from voiceshield.core.schema import Disclaimers
from voiceshield.dsp.exceptions import (
    AudioDecodingError,
    AudioDurationLimitError,
    AudioPreprocessingError,
    CorruptAudioError,
    EmptyAudioError,
    InvalidAudioValueError,
    UnsupportedAudioError,
)
from voiceshield.dsp.preprocessor import load_audio
from voiceshield.dsp.quality import AudioQualityMetrics, compute_quality_metrics
from voiceshield.dsp.reliability import (
    LIMITED_REASONS,
    InputReliabilityAssessment,
    ReliabilityState,
    assess_input_reliability,
)
from voiceshield.dsp.scheduler import CanonicalWindowScheduler
from voiceshield.models.base import BaseDetector, DetectorOutput
from voiceshield.streaming.policy import (
    DecisionState,
    ReasonCode,
    TemporalDecision,
    TemporalDecisionPolicy,
)
from voiceshield.streaming.vad import EnergyVAD

logger = logging.getLogger(__name__)


def _safe_file_audio_error(exc: AudioPreprocessingError) -> tuple[str, str]:
    """Map typed audio failures to actionable safe API messages."""
    if isinstance(exc, EmptyAudioError):
        return (
            "EMPTY_AUDIO_FILE",
            "Audio file is empty. Select a WAV or FLAC file containing audio.",
        )
    if isinstance(exc, AudioDurationLimitError):
        return "AUDIO_DURATION_EXCEEDED", str(exc)
    if isinstance(exc, UnsupportedAudioError):
        return "UNSUPPORTED_AUDIO", str(exc)
    if isinstance(exc, InvalidAudioValueError):
        return "INVALID_AUDIO_VALUE", str(exc)
    if isinstance(exc, (AudioDecodingError, CorruptAudioError)):
        return (
            "AUDIO_PROCESSING_ERROR",
            "Failed to decode audio. The selected file is not valid supported WAV or FLAC audio.",
        )
    return (
        "AUDIO_PROCESSING_ERROR",
        "Failed to decode audio. The selected file is not valid supported WAV or FLAC audio.",
    )


class WindowResult(BaseModel):
    """Evaluation result for an individual temporal window."""

    window_index: int = Field(description="Zero-indexed window sequence number.")
    start_seconds: float = Field(description="Window start offset in seconds.")
    end_seconds: float = Field(description="Window end offset in seconds.")
    raw_spoof_logit: float | None = Field(
        default=None, description="Raw model logit for spoof (index 0). Null when abstained."
    )
    raw_bonafide_logit: float | None = Field(
        default=None, description="Raw model logit for bonafide (index 1). Null when abstained."
    )
    raw_bonafide_score: float | None = Field(
        default=None,
        description="Raw bona fide logit score (ASVspoof standard). Null when abstained.",
    )
    synthetic_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Uncalibrated acoustic anomaly score for synthetic/spoof speech. Null when abstained.",
    )
    uncalibrated_synthetic_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Uncalibrated acoustic anomaly score for synthetic/spoof speech. Null when abstained.",
    )
    score_is_calibrated: bool = Field(
        default=False,
        description="Whether score is calibrated against an empirical target dataset.",
    )
    calibrated_synthetic_likelihood: float | None = Field(
        default=None,
        description="Deprecated compatibility alias. Null when no calibrator is loaded.",
    )
    synthetic_likelihood: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Deprecated compatibility alias for synthetic_score. Null when abstained.",
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
    input_reliability: InputReliabilityAssessment = Field(
        description="Input reliability assessment for this window."
    )
    decision: TemporalDecision | None = Field(
        default=None,
        description="Structured temporal decision emitted for this window.",
    )
    decision_state: str | None = Field(
        default=None,
        description="Advisory decision level (low, uncertain, high, insufficient_evidence).",
    )
    decision_reason: str | None = Field(
        default=None,
        description="Machine-readable decision explanation code.",
    )
    decision_reason_details: str | None = Field(
        default=None,
        description="Human-readable contextual explanation of the decision.",
    )
    smoothed_score: float | None = Field(
        default=None,
        description="Smoothed anomaly score (EWMA or median filtered).",
    )
    history_scores: list[float] = Field(
        default_factory=list,
        description="Recent smoothed score trajectory used for elevation and hysteresis.",
    )

    @model_validator(mode="before")
    @classmethod
    def sync_window_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            score = data.get("synthetic_score")
            if score is None and "uncalibrated_synthetic_score" in data:
                score = data.get("uncalibrated_synthetic_score")
            if score is None and "synthetic_likelihood" in data:
                score = data.get("synthetic_likelihood")
            if score is not None:
                data.setdefault("synthetic_score", score)
                data.setdefault("uncalibrated_synthetic_score", score)
                data.setdefault("synthetic_likelihood", score)
            else:
                data.setdefault("synthetic_score", None)
                data.setdefault("uncalibrated_synthetic_score", None)
                data.setdefault("synthetic_likelihood", None)
        return data


class FileInferenceResult(BaseModel):
    """Aggregated offline inference result for an entire audio file."""

    file_path: str = Field(description="Source file path or identifier.")
    status: str = Field(
        default="success",
        description="Status ('success', 'error', 'insufficient_evidence', 'limited_reliability', 'unsupported').",
    )
    error_message: str | None = Field(
        default=None, description="Error details if status == 'error'."
    )
    error_code: str | None = Field(
        default=None, description="Stable error code if status == 'error'."
    )
    abstain_reason: str | None = Field(
        default=None,
        description="Reason details if status in ('insufficient_evidence', 'limited_reliability').",
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
    reliable_window_count: int = Field(
        default=0,
        ge=0,
        description="Number of evaluated windows classified as reliable.",
    )
    limited_reliability_window_count: int = Field(
        default=0,
        ge=0,
        description="Number of evaluated windows classified as limited reliability.",
    )
    insufficient_reliability_window_count: int = Field(
        default=0,
        ge=0,
        description="Number of evaluated windows classified as insufficient reliability.",
    )
    degraded_quality_window_count: int = Field(
        default=0,
        ge=0,
        description="Number of evaluated windows whose reasons contain at least one degraded-quality (LIMITED_REASONS) member. Overlapping and reason-based.",
    )
    quality_origin_verdict_allowed: bool = Field(
        default=False,
        description="Whether overall input audio quality allows an origin verdict (synthetic vs human).",
    )
    raw_high_window_count: int = Field(
        default=0,
        ge=0,
        description="Number of eligible windows whose raw score reached the effective high threshold.",
    )
    uncertain_window_count: int = Field(
        default=0,
        ge=0,
        description="Number of eligible windows in the UNCERTAIN temporal state.",
    )
    high_window_count: int = Field(
        default=0,
        ge=0,
        description="Number of eligible windows in the sustained HIGH temporal state.",
    )
    mean_synthetic_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Average synthetic score across eligible analyzed windows. Null when abstained.",
    )
    max_synthetic_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Peak synthetic score observed across eligible analyzed windows. Null when abstained.",
    )
    mean_synthetic_likelihood: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Deprecated alias for mean_synthetic_score. Null when abstained.",
    )
    max_synthetic_likelihood: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Deprecated alias for max_synthetic_score. Null when abstained.",
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
        description="Conservative whole-file aggregate HIGH classification across all eligible windows.",
    )
    final_decision_state: str = Field(
        default="uncertain",
        description="Conservative whole-file aggregate advisory state across all eligible windows.",
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
    config_hash: str | None = Field(
        default=None,
        description="Deterministic SHA-256 hash of the effective configuration.",
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
            # Check mean_synthetic_score vs mean_synthetic_likelihood
            if "mean_synthetic_score" in data and data["mean_synthetic_score"] is None:
                data["mean_synthetic_likelihood"] = None
            elif "mean_synthetic_likelihood" in data and data["mean_synthetic_likelihood"] is None:
                data["mean_synthetic_score"] = None
            else:
                mean_s = data.get("mean_synthetic_score")
                if mean_s is None and "mean_synthetic_likelihood" in data:
                    mean_s = data.get("mean_synthetic_likelihood")
                if mean_s is not None:
                    data.setdefault("mean_synthetic_score", mean_s)
                    data.setdefault("mean_synthetic_likelihood", mean_s)

            # Check max_synthetic_score vs max_synthetic_likelihood
            if "max_synthetic_score" in data and data["max_synthetic_score"] is None:
                data["max_synthetic_likelihood"] = None
            elif "max_synthetic_likelihood" in data and data["max_synthetic_likelihood"] is None:
                data["max_synthetic_score"] = None
            else:
                max_s = data.get("max_synthetic_score")
                if max_s is None and "max_synthetic_likelihood" in data:
                    max_s = data.get("max_synthetic_likelihood")
                if max_s is not None:
                    data.setdefault("max_synthetic_score", max_s)
                    data.setdefault("max_synthetic_likelihood", max_s)
        return data


def _validate_window_safety_contracts(
    windows: list[WindowResult],
    num_windows: int,
    reliable_window_count: int,
    limited_reliability_window_count: int,
    insufficient_reliability_window_count: int,
    degraded_quality_window_count: int,
    num_eligible: int,
) -> None:
    """Validate runtime safety contracts for evaluated windows fail-closed."""
    total_reliability_counts = (
        reliable_window_count
        + limited_reliability_window_count
        + insufficient_reliability_window_count
    )
    if total_reliability_counts != num_windows:
        raise RuntimeError(
            f"Reliability window counters invariant violated: reliable ({reliable_window_count}) + "
            f"limited ({limited_reliability_window_count}) + insufficient ({insufficient_reliability_window_count}) "
            f"= {total_reliability_counts}, but total_windows is {num_windows}."
        )

    expected_degraded_quality_window_count = sum(
        1 for w in windows if any(r in LIMITED_REASONS for r in w.input_reliability.reasons)
    )
    if degraded_quality_window_count != expected_degraded_quality_window_count:
        raise RuntimeError(
            f"Degraded quality window count invariant violated: supplied ({degraded_quality_window_count}) "
            f"!= derived from window reasons ({expected_degraded_quality_window_count})."
        )

    if num_eligible != reliable_window_count:
        raise RuntimeError(
            f"Eligible windows count invariant violated: eligible_windows ({num_eligible}) != "
            f"reliable_window_count ({reliable_window_count})."
        )

    for w in windows:
        if w.is_eligible:
            if w.input_reliability.state != ReliabilityState.RELIABLE:
                raise RuntimeError(
                    f"Window {w.window_index} marked eligible but input_reliability.state is "
                    f"'{w.input_reliability.state.value}'; expected '{ReliabilityState.RELIABLE.value}'."
                )
            if not w.input_reliability.origin_verdict_allowed:
                raise RuntimeError(
                    f"Window {w.window_index} marked eligible but origin_verdict_allowed is False."
                )
            if w.decision_reason == ReasonCode.INSUFFICIENT_SPEECH_DURATION.value:
                raise RuntimeError(
                    f"Window {w.window_index} marked eligible but decision_reason is "
                    f"'{ReasonCode.INSUFFICIENT_SPEECH_DURATION.value}'."
                )
            if w.synthetic_score is None:
                raise RuntimeError(
                    f"Window {w.window_index} marked eligible but synthetic_score is None."
                )
            if w.raw_spoof_logit is None:
                raise RuntimeError(
                    f"Window {w.window_index} marked eligible but raw_spoof_logit is None."
                )
            if w.raw_bonafide_logit is None:
                raise RuntimeError(
                    f"Window {w.window_index} marked eligible but raw_bonafide_logit is None."
                )
            if w.raw_bonafide_score is None:
                raise RuntimeError(
                    f"Window {w.window_index} marked eligible but raw_bonafide_score is None."
                )
            if w.abstain_reason is not None:
                raise RuntimeError(
                    f"Window {w.window_index} marked eligible but abstain_reason is '{w.abstain_reason}' "
                    "(expected None)."
                )
        else:
            if w.input_reliability.origin_verdict_allowed is not False:
                raise RuntimeError(
                    f"Non-reliable window {w.window_index} has origin_verdict_allowed="
                    f"{w.input_reliability.origin_verdict_allowed} (expected False)."
                )
            if w.synthetic_score is not None:
                raise RuntimeError(
                    f"Non-reliable window {w.window_index} has synthetic_score="
                    f"{w.synthetic_score} (expected None)."
                )
            if w.uncalibrated_synthetic_score is not None:
                raise RuntimeError(
                    f"Non-reliable window {w.window_index} has uncalibrated_synthetic_score="
                    f"{w.uncalibrated_synthetic_score} (expected None)."
                )
            if w.synthetic_likelihood is not None:
                raise RuntimeError(
                    f"Non-reliable window {w.window_index} has synthetic_likelihood="
                    f"{w.synthetic_likelihood} (expected None)."
                )
            if w.calibrated_synthetic_likelihood is not None:
                raise RuntimeError(
                    f"Non-reliable window {w.window_index} has calibrated_synthetic_likelihood="
                    f"{w.calibrated_synthetic_likelihood} (expected None)."
                )
            if w.score_is_calibrated is not False:
                raise RuntimeError(
                    f"Non-reliable window {w.window_index} has score_is_calibrated=True "
                    "(expected False)."
                )
            if w.is_synthetic_suspected is not False:
                raise RuntimeError(
                    f"Non-reliable window {w.window_index} has is_synthetic_suspected=True "
                    "(expected False)."
                )
            if w.is_eligible is not False:
                raise RuntimeError(
                    f"Non-reliable window {w.window_index} has is_eligible=True (expected False)."
                )
            if w.raw_spoof_logit is not None:
                raise RuntimeError(
                    f"Non-reliable window {w.window_index} has raw_spoof_logit="
                    f"{w.raw_spoof_logit} (expected None)."
                )
            if w.raw_bonafide_logit is not None:
                raise RuntimeError(
                    f"Non-reliable window {w.window_index} has raw_bonafide_logit="
                    f"{w.raw_bonafide_logit} (expected None)."
                )
            if w.raw_bonafide_score is not None:
                raise RuntimeError(
                    f"Non-reliable window {w.window_index} has raw_bonafide_score="
                    f"{w.raw_bonafide_score} (expected None)."
                )
            if w.abstain_reason not in (
                ReasonCode.INSUFFICIENT_SPEECH_DURATION.value,
                ReasonCode.SIGNAL_QUALITY_DEGRADED.value,
            ):
                raise RuntimeError(
                    f"Non-reliable window {w.window_index} has invalid abstain_reason "
                    f"'{w.abstain_reason}'; expected one of "
                    f"{(ReasonCode.INSUFFICIENT_SPEECH_DURATION.value, ReasonCode.SIGNAL_QUALITY_DEGRADED.value)}."
                )


def evaluate_audio(
    source: str | Path | bytes | io.BytesIO,
    detector: BaseDetector,
    config: VoiceShieldConfig | None = None,
    hop_seconds: float | None = None,
    window_seconds: float | None = None,
    threshold: float | None = None,
    file_name: str | None = None,
    max_duration_seconds: float | None = None,
    min_speech_seconds: float | None = None,
) -> FileInferenceResult:
    """Evaluate an audio file using canonical windowing, shared EnergyVAD, and TemporalDecisionPolicy.

    Args:
        source: Audio file path, bytes, or buffer.
        detector: Instantiated BaseDetector implementation.
        config: Optional VoiceShieldConfig for provenance and policy configuration.
        hop_seconds: Optional step size between consecutive windows in seconds (uses config if omitted).
        window_seconds: Optional duration of each analysis window in seconds (uses config if omitted).
        threshold: Optional request-level decision threshold for synthetic suspicion (uses config if omitted).
        file_name: Optional explicit file name override for reporting.
        max_duration_seconds: Optional ceiling on acceptable audio duration in seconds.
        min_speech_seconds: Optional minimum active speech seconds required per window and file.

    Returns:
        Populated FileInferenceResult model.
    """
    path_str = file_name or (str(source) if isinstance(source, (str, Path)) else "in_memory_audio")

    # 1. Establish validated effective configuration
    base_cfg: VoiceShieldConfig = config if config is not None else VoiceShieldConfig()
    effective_cfg = apply_config_overrides(
        base_config=base_cfg,
        hop_seconds=hop_seconds,
        window_seconds=window_seconds,
        min_speech_seconds=min_speech_seconds,
        threshold=threshold,
    )

    effective_config_hash = effective_cfg.compute_hash()

    effective_window_cfg = effective_cfg.windowing
    effective_policy_cfg = effective_cfg.temporal_policy
    target_sr = effective_cfg.audio.sample_rate
    effective_threshold = effective_policy_cfg.threshold_high

    policy = TemporalDecisionPolicy(
        policy_config=effective_policy_cfg,
        windowing_config=effective_window_cfg,
    )

    error_metadata: dict[str, Any] = {
        "config_hash": effective_config_hash,
        "effective_threshold": effective_threshold,
    }
    if hasattr(detector, "provenance") and detector.provenance is not None:
        prov = detector.provenance
        error_metadata.update(
            {
                "model_id": prov.model_id,
                "provenance_status": prov.provenance_status,
                "checkpoint_verified": prov.checkpoint_verified,
                "checkpoint_sha256": prov.checkpoint_sha256,
                "model_provenance": prov.model_dump(),
            }
        )

    # 3. Load audio via canonical preprocessor with strict amplitude validation (no silent normalization)
    try:
        preprocessed = load_audio(
            source,
            target_sr=target_sr,
            normalize=False,
            max_duration_seconds=max_duration_seconds,
        )
    except AudioPreprocessingError as exc:
        logger.warning("Audio preprocessing failed for %s: %s", path_str, exc, exc_info=True)
        error_code, error_message = _safe_file_audio_error(exc)
        return FileInferenceResult(
            file_path=path_str,
            status="error",
            error_code=error_code,
            error_message=error_message,
            effective_threshold=effective_threshold,
            config_hash=effective_config_hash,
            metadata=error_metadata,
            disclaimers=Disclaimers(),
        )
    except Exception:
        logger.exception("Unexpected audio loading failure for %s", path_str)
        return FileInferenceResult(
            file_path=path_str,
            status="error",
            error_message="Failed to decode audio. The selected file is not valid supported WAV or FLAC audio.",
            effective_threshold=effective_threshold,
            config_hash=effective_config_hash,
            metadata=error_metadata,
            disclaimers=Disclaimers(),
        )

    waveform = preprocessed.waveform  # 1D float32 @ canonical sample rate
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
            effective_threshold=effective_threshold,
            config_hash=effective_config_hash,
            metadata=error_metadata,
            disclaimers=Disclaimers(),
        )

    effective_min_speech = effective_window_cfg.min_speech_seconds
    vad_energy_thresh = effective_window_cfg.energy_threshold
    vad_subframe_sec = getattr(effective_window_cfg, "subframe_duration_seconds", 0.02)
    vad = EnergyVAD(
        energy_threshold=vad_energy_thresh,
        subframe_duration_seconds=vad_subframe_sec,
    )
    # File-level total active speech calculated over the full canonical waveform for reporting
    total_active_speech = round(vad.compute_speech_duration(waveform, target_sr), 2)

    # 4. Canonical window scheduler
    window_samples = effective_window_cfg.window_samples(target_sr)
    hop_samples = effective_window_cfg.hop_samples(target_sr)
    scheduler = CanonicalWindowScheduler(
        sample_rate=target_sr,
        window_samples=window_samples,
        hop_samples=hop_samples,
    )
    logical_windows = scheduler.push_samples(waveform)

    is_mock = getattr(detector, "is_mock", False)
    res_metadata: dict[str, Any] = {
        "model_id": getattr(detector, "model_id", detector.__class__.__name__),
        "device_used": str(getattr(detector, "device", "unknown")),
        "effective_threshold": effective_threshold,
        "operating_threshold_source": "unvalidated_heuristic",
        "score_is_calibrated": False,
        "active_speech_duration_seconds": total_active_speech,
        "min_speech_threshold_seconds": effective_min_speech,
        "window_seconds": effective_window_cfg.window_seconds,
        "hop_seconds": effective_window_cfg.hop_seconds,
        "config_hash": effective_config_hash,
        "pre_norm_peak": metadata.pre_norm_quality.peak_level,
        "pre_norm_rms": metadata.pre_norm_quality.rms,
        "pre_norm_status": metadata.pre_norm_quality.quality_status,
        "format": metadata.format,
        "subtype": metadata.subtype,
        "is_mock_detector": is_mock,
    }
    if hasattr(detector, "provenance") and detector.provenance is not None:
        prov = detector.provenance
        res_metadata.update(
            {
                "model_id": prov.model_id,
                "provenance_status": prov.provenance_status,
                "checkpoint_verified": prov.checkpoint_verified,
                "checkpoint_sha256": prov.checkpoint_sha256,
                "expected_checkpoint_sha256": prov.expected_checkpoint_sha256,
                "model_provenance": prov.model_dump(),
            }
        )

    # 5. Deterministic full-window and end-of-stream policy:
    # If audio contains fewer than window_samples (< window_seconds), report insufficient evidence
    if len(logical_windows) == 0:
        req_samples = effective_window_cfg.window_samples(target_sr)
        req_seconds = effective_window_cfg.window_seconds
        explanation = (
            f"Audio duration ({metadata.duration_seconds:.2f}s) contains fewer than {req_samples:,} "
            f"canonical contiguous samples ({req_seconds:.1f}s) required for inference. Decision abstained."
        )
        return FileInferenceResult(
            file_path=path_str,
            status="insufficient_evidence",
            abstain_reason=ReasonCode.INSUFFICIENT_CONTIGUOUS_AUDIO.value,
            duration_seconds=round(metadata.duration_seconds, 3),
            active_speech_duration_seconds=total_active_speech,
            original_sample_rate=metadata.original_sample_rate,
            original_channels=metadata.original_channels,
            total_windows=0,
            eligible_windows=0,
            reliable_window_count=0,
            limited_reliability_window_count=0,
            insufficient_reliability_window_count=0,
            degraded_quality_window_count=0,
            quality_origin_verdict_allowed=False,
            mean_synthetic_score=None,
            max_synthetic_score=None,
            mean_synthetic_likelihood=None,
            max_synthetic_likelihood=None,
            score_is_calibrated=False,
            effective_threshold=effective_threshold,
            operating_threshold_source="unvalidated_heuristic",
            is_synthetic_suspected=False,
            final_decision_state=DecisionState.INSUFFICIENT_EVIDENCE.value,
            temporal_rule_explanation=explanation,
            windows=[],
            is_mock_detector=is_mock,
            config_hash=effective_config_hash,
            metadata=res_metadata,
            disclaimers=Disclaimers(),
        )

    # 6. Evaluate each logical window synchronously without drop semantics
    windows: list[WindowResult] = []
    for lw in logical_windows:
        chunk = lw.samples
        start_sec = lw.start_time
        end_sec = lw.end_time
        chunk_quality = compute_quality_metrics(chunk)
        speech_sec = vad.compute_speech_duration(chunk, target_sr)

        reliability = assess_input_reliability(
            metrics=chunk_quality,
            active_speech_seconds=speech_sec,
            required_min_speech_seconds=effective_min_speech,
        )

        if not reliability.origin_verdict_allowed:
            decision = policy.evaluate_reliability_abstention(
                assessment=reliability,
                window_index=lw.window_index,
                window_start_sec=start_sec,
                window_end_sec=end_sec,
                speech_seconds=speech_sec,
            )
            win_abstain = (
                ReasonCode.INSUFFICIENT_SPEECH_DURATION.value
                if speech_sec < effective_min_speech
                and not any(r in LIMITED_REASONS for r in reliability.reasons)
                else decision.reason.value
            )
            windows.append(
                WindowResult(
                    window_index=lw.window_index,
                    start_seconds=start_sec,
                    end_seconds=end_sec,
                    raw_spoof_logit=None,
                    raw_bonafide_logit=None,
                    raw_bonafide_score=None,
                    synthetic_score=None,
                    uncalibrated_synthetic_score=None,
                    synthetic_likelihood=None,
                    calibrated_synthetic_likelihood=None,
                    score_is_calibrated=False,
                    effective_threshold=effective_threshold,
                    is_synthetic_suspected=False,
                    active_speech_seconds=speech_sec,
                    is_eligible=False,
                    is_partial=False,
                    abstain_reason=win_abstain,
                    quality_status=chunk_quality,
                    input_reliability=reliability,
                    decision=decision,
                    decision_state=decision.state.value,
                    decision_reason=decision.reason.value,
                    decision_reason_details=decision.reason_details,
                    smoothed_score=decision.smoothed_score,
                    history_scores=decision.history_scores,
                )
            )
        else:
            det_output: DetectorOutput = detector.predict(chunk)
            score = float(det_output.synthetic_score)
            decision = policy.evaluate(
                synthetic_score=score,
                speech_seconds=speech_sec,
                window_index=lw.window_index,
                window_start_sec=start_sec,
                window_end_sec=end_sec,
            )
            is_suspected = bool(decision.state == DecisionState.HIGH)
            windows.append(
                WindowResult(
                    window_index=lw.window_index,
                    start_seconds=start_sec,
                    end_seconds=end_sec,
                    raw_spoof_logit=det_output.raw_logits[0],
                    raw_bonafide_logit=det_output.raw_logits[1],
                    raw_bonafide_score=det_output.bonafide_score,
                    synthetic_score=score,
                    uncalibrated_synthetic_score=score,
                    synthetic_likelihood=score,
                    calibrated_synthetic_likelihood=None,
                    score_is_calibrated=False,
                    effective_threshold=effective_threshold,
                    is_synthetic_suspected=is_suspected,
                    active_speech_seconds=speech_sec,
                    is_eligible=True,
                    is_partial=False,
                    abstain_reason=None,
                    quality_status=chunk_quality,
                    input_reliability=reliability,
                    decision=decision,
                    decision_state=decision.state.value,
                    decision_reason=decision.reason.value,
                    decision_reason_details=decision.reason_details,
                    smoothed_score=decision.smoothed_score,
                    history_scores=decision.history_scores,
                )
            )

    num_windows = len(windows)
    reliable_window_count = sum(
        1 for w in windows if w.input_reliability.state == ReliabilityState.RELIABLE
    )
    limited_reliability_window_count = sum(
        1 for w in windows if w.input_reliability.state == ReliabilityState.LIMITED
    )
    insufficient_reliability_window_count = sum(
        1 for w in windows if w.input_reliability.state == ReliabilityState.INSUFFICIENT
    )
    degraded_quality_window_count = sum(
        1 for w in windows if any(r in LIMITED_REASONS for r in w.input_reliability.reasons)
    )

    eligible_windows = [w for w in windows if w.is_eligible]
    num_eligible = len(eligible_windows)

    # Invariants
    _validate_window_safety_contracts(
        windows=windows,
        num_windows=num_windows,
        reliable_window_count=reliable_window_count,
        limited_reliability_window_count=limited_reliability_window_count,
        insufficient_reliability_window_count=insufficient_reliability_window_count,
        degraded_quality_window_count=degraded_quality_window_count,
        num_eligible=num_eligible,
    )

    # 7. Fail-closed aggregation
    if num_eligible == 0:
        if limited_reliability_window_count > 0:
            # Case D: No RELIABLE windows but one or more exact LIMITED windows
            explanation = (
                f"Offline advisory aggregation: no reliable speech windows were available and "
                f"{limited_reliability_window_count} of {num_windows} window(s) had degraded signal quality. "
                "Degraded input quality prevented a binary origin verdict; primary file verdict abstained. "
                "Audio quality is a separate reliability signal and does not constitute evidence of human or synthetic origin."
            )
            return FileInferenceResult(
                file_path=path_str,
                status="limited_reliability",
                abstain_reason=ReasonCode.SIGNAL_QUALITY_DEGRADED.value,
                duration_seconds=round(metadata.duration_seconds, 3),
                active_speech_duration_seconds=total_active_speech,
                original_sample_rate=metadata.original_sample_rate,
                original_channels=metadata.original_channels,
                total_windows=num_windows,
                eligible_windows=0,
                reliable_window_count=0,
                limited_reliability_window_count=limited_reliability_window_count,
                insufficient_reliability_window_count=insufficient_reliability_window_count,
                degraded_quality_window_count=degraded_quality_window_count,
                quality_origin_verdict_allowed=False,
                mean_synthetic_score=None,
                max_synthetic_score=None,
                mean_synthetic_likelihood=None,
                max_synthetic_likelihood=None,
                score_is_calibrated=False,
                effective_threshold=effective_threshold,
                operating_threshold_source="unvalidated_heuristic",
                is_synthetic_suspected=False,
                final_decision_state=DecisionState.UNCERTAIN.value,
                temporal_rule_explanation=explanation,
                windows=windows,
                is_mock_detector=is_mock,
                config_hash=effective_config_hash,
                metadata=res_metadata,
                disclaimers=Disclaimers(),
            )
        else:
            # Case C: No RELIABLE windows and no exact LIMITED windows (all windows INSUFFICIENT)
            explanation = (
                f"Insufficient active speech detected across file ({total_active_speech:.2f}s active speech "
                f"below required minimum threshold of {effective_min_speech:.2f}s). Decision abstained — does "
                "not evaluate synthetic anomalies without active speech. (Note: Energy VAD is a heuristic, "
                "not an infallible speech detector)."
            )
            if degraded_quality_window_count > 0:
                explanation += (
                    f" Note: {degraded_quality_window_count} of {num_windows} window(s) exhibited degraded signal quality "
                    "diagnostics, but insufficient active speech remains the primary status because no inference-eligible "
                    "evidence was available."
                )
            return FileInferenceResult(
                file_path=path_str,
                status="insufficient_evidence",
                abstain_reason=ReasonCode.INSUFFICIENT_SPEECH_DURATION.value,
                duration_seconds=round(metadata.duration_seconds, 3),
                active_speech_duration_seconds=total_active_speech,
                original_sample_rate=metadata.original_sample_rate,
                original_channels=metadata.original_channels,
                total_windows=num_windows,
                eligible_windows=0,
                reliable_window_count=0,
                limited_reliability_window_count=0,
                insufficient_reliability_window_count=insufficient_reliability_window_count,
                degraded_quality_window_count=degraded_quality_window_count,
                quality_origin_verdict_allowed=False,
                mean_synthetic_score=None,
                max_synthetic_score=None,
                mean_synthetic_likelihood=None,
                max_synthetic_likelihood=None,
                score_is_calibrated=False,
                effective_threshold=effective_threshold,
                operating_threshold_source="unvalidated_heuristic",
                is_synthetic_suspected=False,
                final_decision_state=DecisionState.INSUFFICIENT_EVIDENCE.value,
                temporal_rule_explanation=explanation,
                windows=windows,
                is_mock_detector=is_mock,
                config_hash=effective_config_hash,
                metadata=res_metadata,
                disclaimers=Disclaimers(),
            )

    valid_scores = [w.synthetic_score for w in eligible_windows if w.synthetic_score is not None]
    mean_score = float(np.mean(valid_scores)) if valid_scores else 0.0
    max_score = float(np.max(valid_scores)) if valid_scores else 0.0

    raw_high_window_count = sum(
        1
        for window in eligible_windows
        if window.synthetic_score is not None and window.synthetic_score >= effective_threshold
    )
    uncertain_window_count = sum(
        1 for window in eligible_windows if window.decision_state == DecisionState.UNCERTAIN.value
    )
    high_window_count = sum(
        1 for window in eligible_windows if window.decision_state == DecisionState.HIGH.value
    )

    if degraded_quality_window_count > 0:
        # Case A: At least one degraded quality window (with at least one RELIABLE window)
        file_status = "limited_reliability"
        aggregate_state_str = DecisionState.UNCERTAIN.value
        file_suspected = False
        quality_origin_verdict_allowed = False
        abstain_reason = ReasonCode.SIGNAL_QUALITY_DEGRADED.value
        explanation = (
            f"Offline advisory aggregation: {degraded_quality_window_count} of {num_windows} window(s) "
            f"had degraded signal quality ({reliable_window_count} reliable, {insufficient_reliability_window_count} insufficient). "
            f"Reliable window evidence: mean raw score {mean_score:.3f}, peak raw score {max_score:.3f}, "
            f"{raw_high_window_count} raw high threshold crossing(s), {high_window_count} sustained HIGH window(s). "
            "Degraded input quality prevented a binary origin verdict; primary file verdict abstained. "
            "Audio quality is a separate reliability signal and does not constitute evidence of human or synthetic origin."
        )
    else:
        # Case B: No degraded quality windows and at least one RELIABLE window
        file_status = "success"
        quality_origin_verdict_allowed = True
        abstain_reason = None
        if high_window_count > 0:
            aggregate_state = DecisionState.HIGH
        elif uncertain_window_count > 0 or raw_high_window_count > 0:
            aggregate_state = DecisionState.UNCERTAIN
        elif all(window.decision_state == DecisionState.LOW.value for window in eligible_windows):
            aggregate_state = DecisionState.LOW
        else:
            aggregate_state = DecisionState.UNCERTAIN
        aggregate_state_str = aggregate_state.value
        file_suspected = aggregate_state == DecisionState.HIGH
        explanation = (
            f"Offline advisory aggregation: mean raw score {mean_score:.3f}, peak raw score "
            f"{max_score:.3f}, effective threshold {effective_threshold:.3f}; "
            f"{raw_high_window_count} raw threshold crossing(s), {uncertain_window_count} "
            f"uncertain window(s), and {high_window_count} sustained HIGH window(s). "
            f"Sustained elevation occurred: {'yes' if file_suspected else 'no'}. "
        )
        if aggregate_state == DecisionState.HIGH:
            explanation += (
                "Sustained synthetic acoustic anomaly detected across temporal decision policy. "
                "Pause and verify through a known independent channel. Do not automatically "
                "terminate or block calls."
            )
        elif aggregate_state == DecisionState.UNCERTAIN:
            explanation += (
                "Isolated or ambiguous synthetic acoustic anomaly detected; evidence is mixed "
                "or not sustained. Advisory metric only and does NOT prove identity of caller."
            )
        else:
            explanation += (
                "All eligible windows remained LOW; advisory metric only and does NOT "
                "authenticate speech."
            )

    return FileInferenceResult(
        file_path=path_str,
        status=file_status,
        abstain_reason=abstain_reason,
        duration_seconds=round(metadata.duration_seconds, 3),
        active_speech_duration_seconds=total_active_speech,
        original_sample_rate=metadata.original_sample_rate,
        original_channels=metadata.original_channels,
        total_windows=num_windows,
        eligible_windows=num_eligible,
        reliable_window_count=reliable_window_count,
        limited_reliability_window_count=limited_reliability_window_count,
        insufficient_reliability_window_count=insufficient_reliability_window_count,
        degraded_quality_window_count=degraded_quality_window_count,
        quality_origin_verdict_allowed=quality_origin_verdict_allowed,
        raw_high_window_count=raw_high_window_count,
        uncertain_window_count=uncertain_window_count,
        high_window_count=high_window_count,
        mean_synthetic_score=round(mean_score, 4),
        max_synthetic_score=round(max_score, 4),
        mean_synthetic_likelihood=round(mean_score, 4),
        max_synthetic_likelihood=round(max_score, 4),
        score_is_calibrated=False,
        effective_threshold=effective_threshold,
        operating_threshold_source="unvalidated_heuristic",
        is_synthetic_suspected=file_suspected,
        final_decision_state=aggregate_state_str,
        temporal_rule_explanation=explanation,
        windows=windows,
        is_mock_detector=is_mock,
        config_hash=effective_config_hash,
        metadata=res_metadata,
        disclaimers=Disclaimers(),
    )
