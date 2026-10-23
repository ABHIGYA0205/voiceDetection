"""Deterministic offline file inference engine around the BaseDetector interface."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, model_validator

from voiceshield.config.schema import TemporalPolicyConfig, VoiceShieldConfig, WindowingConfig
from voiceshield.core.schema import Disclaimers
from voiceshield.dsp.exceptions import AudioPreprocessingError
from voiceshield.dsp.preprocessor import load_audio
from voiceshield.dsp.quality import AudioQualityMetrics, compute_quality_metrics
from voiceshield.dsp.scheduler import CanonicalWindowScheduler
from voiceshield.models.base import BaseDetector, DetectorOutput
from voiceshield.streaming.policy import (
    DecisionState,
    ReasonCode,
    TemporalDecision,
    TemporalDecisionPolicy,
)
from voiceshield.streaming.vad import EnergyVAD


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
        description="Overall file-level advisory classification derived from final temporal policy state.",
    )
    final_decision_state: str = Field(
        default="uncertain",
        description="Final categorical advisory decision state from the shared temporal policy.",
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
    """Evaluate an audio file using canonical windowing, shared EnergyVAD, and TemporalDecisionPolicy.

    Args:
        source: Audio file path, bytes, or buffer.
        detector: Instantiated BaseDetector implementation.
        config: Optional VoiceShieldConfig for provenance and policy configuration.
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

    # 1. Load audio via canonical preprocessor with strict amplitude validation (no silent normalization)
    try:
        preprocessed = load_audio(
            source,
            target_sr=16000,
            normalize=False,
            max_duration_seconds=max_duration_seconds,
        )
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

    # 2. Build validated effective WindowingConfig for the request
    if config is not None:
        effective_window_cfg = config.windowing.model_copy()
    else:
        effective_window_cfg = WindowingConfig()

    effective_window_cfg.window_seconds = float(window_seconds)
    effective_window_cfg.hop_seconds = float(hop_seconds)
    if min_speech_seconds is not None:
        effective_window_cfg.min_speech_seconds = float(min_speech_seconds)
    effective_window_cfg = WindowingConfig.model_validate(effective_window_cfg)

    effective_min_speech = effective_window_cfg.min_speech_seconds
    vad_energy_thresh = effective_window_cfg.energy_threshold
    vad_subframe_sec = getattr(effective_window_cfg, "subframe_duration_seconds", 0.02)
    vad = EnergyVAD(
        energy_threshold=vad_energy_thresh,
        subframe_duration_seconds=vad_subframe_sec,
    )
    # File-level total active speech calculated over the full canonical waveform for reporting
    total_active_speech = round(vad.compute_speech_duration(waveform, 16000), 2)

    # 3. Build validated effective TemporalPolicyConfig for the request
    if config is not None:
        effective_policy_cfg = config.temporal_policy.model_copy()
    else:
        effective_policy_cfg = TemporalPolicyConfig()

    thresh_high = float(threshold)
    thresh_low = effective_policy_cfg.threshold_low
    if thresh_low > thresh_high:
        thresh_low = thresh_high

    effective_policy_cfg.threshold_high = thresh_high
    effective_policy_cfg.threshold_low = thresh_low
    effective_policy_cfg = TemporalPolicyConfig.model_validate(effective_policy_cfg)

    policy = TemporalDecisionPolicy(
        policy_config=effective_policy_cfg,
        windowing_config=effective_window_cfg,
    )

    # 4. Canonical window scheduler
    window_samples = effective_window_cfg.window_samples(16000)
    hop_samples = effective_window_cfg.hop_samples(16000)
    scheduler = CanonicalWindowScheduler(
        sample_rate=16000,
        window_samples=window_samples,
        hop_samples=hop_samples,
    )
    logical_windows = scheduler.push_samples(waveform)

    is_mock = getattr(detector, "is_mock", False)
    res_metadata: dict[str, Any] = {
        "model_id": getattr(detector, "model_id", detector.__class__.__name__),
        "device_used": str(getattr(detector, "device", "unknown")),
        "effective_threshold": threshold,
        "operating_threshold_source": "unvalidated_heuristic",
        "score_is_calibrated": False,
        "active_speech_duration_seconds": total_active_speech,
        "min_speech_threshold_seconds": effective_min_speech,
        "window_seconds": effective_window_cfg.window_seconds,
        "hop_seconds": effective_window_cfg.hop_seconds,
        "config_hash": config.compute_hash() if config else None,
        "pre_norm_peak": metadata.pre_norm_quality.peak_level,
        "pre_norm_rms": metadata.pre_norm_quality.rms,
        "pre_norm_status": metadata.pre_norm_quality.quality_status,
        "format": metadata.format,
        "subtype": metadata.subtype,
        "is_mock_detector": is_mock,
    }

    # 5. Deterministic full-window and end-of-stream policy:
    # If audio contains fewer than 64,000 canonical contiguous samples (< 4.0s), report insufficient evidence
    if len(logical_windows) == 0:
        explanation = (
            f"Audio duration ({metadata.duration_seconds:.2f}s) contains fewer than 64,000 "
            "canonical contiguous samples (4.0s) required for inference. Decision abstained."
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
            mean_synthetic_score=None,
            max_synthetic_score=None,
            mean_synthetic_likelihood=None,
            max_synthetic_likelihood=None,
            score_is_calibrated=False,
            effective_threshold=threshold,
            operating_threshold_source="unvalidated_heuristic",
            is_synthetic_suspected=False,
            final_decision_state=DecisionState.INSUFFICIENT_EVIDENCE.value,
            temporal_rule_explanation=explanation,
            windows=[],
            is_mock_detector=is_mock,
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
        speech_sec = vad.compute_speech_duration(chunk, 16000)

        if speech_sec < effective_min_speech:
            decision = policy.evaluate_insufficient_speech(
                window_index=lw.window_index,
                window_start_sec=start_sec,
                window_end_sec=end_sec,
                speech_seconds=speech_sec,
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
                    score_is_calibrated=False,
                    calibrated_synthetic_likelihood=None,
                    effective_threshold=threshold,
                    is_synthetic_suspected=False,
                    active_speech_seconds=speech_sec,
                    is_eligible=False,
                    is_partial=False,
                    abstain_reason=ReasonCode.INSUFFICIENT_SPEECH_DURATION.value,
                    quality_status=chunk_quality,
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
                    score_is_calibrated=False,
                    calibrated_synthetic_likelihood=None,
                    effective_threshold=threshold,
                    is_synthetic_suspected=is_suspected,
                    active_speech_seconds=speech_sec,
                    is_eligible=True,
                    is_partial=False,
                    abstain_reason=None,
                    quality_status=chunk_quality,
                    decision=decision,
                    decision_state=decision.state.value,
                    decision_reason=decision.reason.value,
                    decision_reason_details=decision.reason_details,
                    smoothed_score=decision.smoothed_score,
                    history_scores=decision.history_scores,
                )
            )

    for w in windows:
        if w.is_eligible:
            assert w.decision_reason != ReasonCode.INSUFFICIENT_SPEECH_DURATION.value, (
                f"Window {w.window_index} marked eligible but policy returned insufficient_speech_duration"
            )
            assert w.synthetic_score is not None, (
                f"Window {w.window_index} eligible but score is None"
            )
            assert w.raw_spoof_logit is not None, (
                f"Window {w.window_index} eligible but raw_spoof_logit is None"
            )
        else:
            assert w.synthetic_score is None, (
                f"Window {w.window_index} abstained but score is not None"
            )
            assert w.raw_spoof_logit is None, (
                f"Window {w.window_index} abstained but raw_spoof_logit is not None"
            )
            assert w.abstain_reason == ReasonCode.INSUFFICIENT_SPEECH_DURATION.value

    eligible_windows = [w for w in windows if w.is_eligible]
    num_windows = len(windows)
    num_eligible = len(eligible_windows)

    # 7. If no windows have sufficient active speech, return insufficient_evidence / abstain
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
            abstain_reason=ReasonCode.INSUFFICIENT_SPEECH_DURATION.value,
            duration_seconds=round(metadata.duration_seconds, 3),
            active_speech_duration_seconds=total_active_speech,
            original_sample_rate=metadata.original_sample_rate,
            original_channels=metadata.original_channels,
            total_windows=num_windows,
            eligible_windows=0,
            mean_synthetic_score=None,
            max_synthetic_score=None,
            mean_synthetic_likelihood=None,
            max_synthetic_likelihood=None,
            score_is_calibrated=False,
            effective_threshold=threshold,
            operating_threshold_source="unvalidated_heuristic",
            is_synthetic_suspected=False,
            final_decision_state=DecisionState.INSUFFICIENT_EVIDENCE.value,
            temporal_rule_explanation=explanation,
            windows=windows,
            is_mock_detector=is_mock,
            metadata=res_metadata,
            disclaimers=Disclaimers(),
        )

    valid_scores = [w.synthetic_score for w in eligible_windows if w.synthetic_score is not None]
    mean_score = float(np.mean(valid_scores)) if valid_scores else 0.0
    max_score = float(np.max(valid_scores)) if valid_scores else 0.0

    # 8. File-level decision derived from final shared temporal-policy state
    file_suspected = bool(policy.current_state == DecisionState.HIGH)

    if file_suspected:
        explanation = (
            f"Advisory Warning: Sustained synthetic acoustic anomaly detected across temporal decision policy "
            f"(Peak score: {max_score:.3f} >= threshold {threshold:.3f}). "
            "Pause and verify through a known independent channel. Do not automatically terminate or block calls."
        )
    elif (
        any(w.is_synthetic_suspected for w in eligible_windows)
        or policy.current_state == DecisionState.UNCERTAIN
    ):
        explanation = (
            f"Isolated or ambiguous synthetic acoustic anomaly detected, but anomaly was not sustained across "
            f"temporal policy elevation criteria. "
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
        is_synthetic_suspected=file_suspected,
        final_decision_state=policy.current_state.value,
        temporal_rule_explanation=explanation,
        windows=windows,
        is_mock_detector=is_mock,
        metadata=res_metadata,
        disclaimers=Disclaimers(),
    )
