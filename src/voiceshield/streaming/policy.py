"""Temporal decision policy for multi-hop streaming synthetic speech detection."""

from __future__ import annotations

from enum import StrEnum

import numpy as np
from pydantic import BaseModel, Field

from voiceshield.config.schema import TemporalPolicyConfig, WindowingConfig


class DecisionState(StrEnum):
    """Categorical advisory decision state for a streaming session."""

    LOW = "low"
    UNCERTAIN = "uncertain"
    HIGH = "high"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class ReasonCode(StrEnum):
    """Machine-readable explanation identifier for temporal decisions."""

    INSUFFICIENT_SPEECH_DURATION = "insufficient_speech_duration"
    SUSTAINED_ANOMALY_ELEVATION = "sustained_anomaly_elevation"
    ELEVATION_IN_PROGRESS = "elevation_in_progress"
    AMBIGUOUS_SCORE_ZONE = "ambiguous_score_zone"
    HYSTERESIS_COOLDOWN = "hysteresis_cooldown"
    LOW_ANOMALY_CLEARED = "low_anomaly_cleared"
    NORMAL_ACOUSTICS = "normal_acoustics"
    SIGNAL_QUALITY_DEGRADED = "signal_quality_degraded"
    INITIALIZING = "initializing"


class TemporalDecision(BaseModel):
    """Structured temporal decision emitted for each streaming window."""

    state: DecisionState = Field(
        description="Advisory decision level (low, uncertain, high, insufficient_evidence)."
    )
    calibrated_score: float = Field(
        description="Calibrated probability of synthetic speech for this window."
    )
    smoothed_score: float = Field(
        description="Smoothed probability score (EWMA or median filtered)."
    )
    reason: ReasonCode = Field(description="Machine-readable decision explanation code.")
    reason_details: str = Field(
        description="Human-readable contextual explanation of the decision."
    )
    window_index: int = Field(description="Zero-indexed sequence number of this streaming window.")
    window_start_sec: float = Field(description="Stream timestamp of window start in seconds.")
    window_end_sec: float = Field(description="Stream timestamp of window end in seconds.")
    speech_seconds: float = Field(
        description="Active speech duration detected within the 4s window."
    )
    history_scores: list[float] = Field(
        default_factory=list,
        description="Recent smoothed score trajectory used for elevation and hysteresis.",
    )


class TemporalDecisionPolicy:
    """Stateful temporal policy implementing score smoothing, N-of-M elevation, and hysteresis.

    Advisory output only: indicates likelihood of synthetic acoustic artifacts; does not claim
    identity proof or fraud confirmation.
    """

    def __init__(
        self,
        policy_config: TemporalPolicyConfig | None = None,
        windowing_config: WindowingConfig | None = None,
    ) -> None:
        """Initialize temporal policy from configuration."""
        self.policy_cfg = policy_config or TemporalPolicyConfig()
        self.window_cfg = windowing_config or WindowingConfig()

        # State tracking
        self.current_state: DecisionState = DecisionState.UNCERTAIN
        self.raw_history: list[float] = []
        self.smoothed_history: list[float] = []
        self.consecutive_drops: int = 0
        self.window_count: int = 0

    def reset(self) -> None:
        """Reset temporal policy state and history."""
        self.current_state = DecisionState.UNCERTAIN
        self.raw_history.clear()
        self.smoothed_history.clear()
        self.consecutive_drops = 0
        self.window_count = 0

    def compute_smoothed_score(self, current_raw_score: float) -> float:
        """Apply configured smoothing (EWMA, median, or none) to the latest raw calibrated score."""
        method = self.policy_cfg.smoothing_method.lower()

        if method == "ewma":
            if not self.smoothed_history:
                return float(current_raw_score)
            prev_smoothed = self.smoothed_history[-1]
            alpha = self.policy_cfg.ewma_alpha
            return float(alpha * current_raw_score + (1.0 - alpha) * prev_smoothed)

        elif method == "median":
            k = self.policy_cfg.median_window_size
            window_slice = (self.raw_history + [current_raw_score])[-k:]
            return float(np.median(window_slice))

        elif method == "none":
            return float(current_raw_score)

        else:
            return float(current_raw_score)

    def evaluate_insufficient_speech(
        self,
        window_index: int,
        window_start_sec: float,
        window_end_sec: float,
        speech_seconds: float,
    ) -> TemporalDecision:
        """Emit an insufficient_evidence decision when active speech duration is below threshold."""
        self.window_count += 1
        return TemporalDecision(
            state=DecisionState.INSUFFICIENT_EVIDENCE,
            calibrated_score=0.0,
            smoothed_score=self.smoothed_history[-1] if self.smoothed_history else 0.0,
            reason=ReasonCode.INSUFFICIENT_SPEECH_DURATION,
            reason_details=(
                f"Active speech ({speech_seconds:.2f}s) is below required minimum threshold "
                f"({self.window_cfg.min_speech_seconds:.2f}s) for reliable inference."
            ),
            window_index=window_index,
            window_start_sec=window_start_sec,
            window_end_sec=window_end_sec,
            speech_seconds=speech_seconds,
            history_scores=list(self.smoothed_history[-5:]),
        )

    def evaluate(
        self,
        calibrated_score: float,
        speech_seconds: float,
        window_index: int,
        window_start_sec: float,
        window_end_sec: float,
    ) -> TemporalDecision:
        """Consume calibrated per-window score and determine temporal decision state.

        Args:
            calibrated_score: Calibrated synthetic speech probability for the current window [0.0, 1.0].
            speech_seconds: Duration of active speech in the window in seconds.
            window_index: Sequential index of the window.
            window_start_sec: Window start timestamp in seconds.
            window_end_sec: Window end timestamp in seconds.

        Returns:
            Structured TemporalDecision.
        """
        # 1. Check speech duration constraint
        if speech_seconds < self.window_cfg.min_speech_seconds:
            return self.evaluate_insufficient_speech(
                window_index=window_index,
                window_start_sec=window_start_sec,
                window_end_sec=window_end_sec,
                speech_seconds=speech_seconds,
            )

        self.window_count += 1
        clamped_score = float(np.clip(calibrated_score, 0.0, 1.0))
        smoothed = self.compute_smoothed_score(clamped_score)

        self.raw_history.append(clamped_score)
        self.smoothed_history.append(smoothed)

        thresh_high = self.policy_cfg.threshold_high
        thresh_low = self.policy_cfg.threshold_low
        n_req = self.policy_cfg.elevation_n
        m_horizon = self.policy_cfg.elevation_m
        h_drops = self.policy_cfg.hysteresis_consecutive_drops

        # Inspect last M smoothed scores for N-of-M elevation
        recent_horizon = self.smoothed_history[-m_horizon:]
        high_count = sum(1 for s in recent_horizon if s >= thresh_high)

        state: DecisionState
        reason: ReasonCode
        details: str

        if self.current_state == DecisionState.HIGH:
            # Currently in HIGH state: apply hysteresis
            if smoothed >= thresh_high:
                self.consecutive_drops = 0
                state = DecisionState.HIGH
                reason = ReasonCode.SUSTAINED_ANOMALY_ELEVATION
                details = (
                    f"Sustained synthetic acoustic anomaly maintained: "
                    f"{high_count} of last {len(recent_horizon)} windows exceeded threshold ({thresh_high:.2f})."
                )
            elif smoothed <= thresh_low:
                self.consecutive_drops += 1
                if self.consecutive_drops >= h_drops:
                    state = DecisionState.LOW
                    self.current_state = DecisionState.LOW
                    reason = ReasonCode.LOW_ANOMALY_CLEARED
                    details = (
                        f"Anomaly cleared: {self.consecutive_drops} consecutive windows below "
                        f"low threshold ({thresh_low:.2f}). Returned to LOW."
                    )
                else:
                    state = DecisionState.UNCERTAIN
                    reason = ReasonCode.HYSTERESIS_COOLDOWN
                    details = (
                        f"Anomaly cooling down ({self.consecutive_drops}/{h_drops} consecutive low windows). "
                        f"Awaiting sustained clearance."
                    )
            else:
                # In ambiguous intermediate zone (thresh_low < smoothed < thresh_high)
                self.consecutive_drops = 0
                state = DecisionState.UNCERTAIN
                reason = ReasonCode.AMBIGUOUS_SCORE_ZONE
                details = (
                    f"Score ({smoothed:.3f}) falls in intermediate zone between "
                    f"[{thresh_low:.2f}, {thresh_high:.2f}]."
                )

        else:
            # Currently in LOW or UNCERTAIN state
            if high_count >= n_req:
                state = DecisionState.HIGH
                self.current_state = DecisionState.HIGH
                self.consecutive_drops = 0
                reason = ReasonCode.SUSTAINED_ANOMALY_ELEVATION
                details = (
                    f"Elevated to HIGH: {high_count} of last {len(recent_horizon)} windows exceeded "
                    f"high threshold ({thresh_high:.2f}) (rule: {n_req}-of-{m_horizon})."
                )
            elif high_count > 0 or smoothed >= thresh_high:
                state = DecisionState.UNCERTAIN
                self.current_state = DecisionState.UNCERTAIN
                self.consecutive_drops = 0
                reason = ReasonCode.ELEVATION_IN_PROGRESS
                details = (
                    f"Potential anomaly observed ({high_count} of required {n_req} windows above "
                    f"{thresh_high:.2f}). Awaiting confirmation."
                )
            elif smoothed <= thresh_low:
                state = DecisionState.LOW
                self.current_state = DecisionState.LOW
                self.consecutive_drops = 0
                reason = ReasonCode.NORMAL_ACOUSTICS
                details = (
                    f"Acoustic features consistent with authentic speech "
                    f"(score: {smoothed:.3f} <= {thresh_low:.2f})."
                )
            else:
                state = DecisionState.UNCERTAIN
                self.current_state = DecisionState.UNCERTAIN
                self.consecutive_drops = 0
                reason = ReasonCode.AMBIGUOUS_SCORE_ZONE
                details = (
                    f"Ambiguous score ({smoothed:.3f}) between "
                    f"low ({thresh_low:.2f}) and high ({thresh_high:.2f}) thresholds."
                )

        return TemporalDecision(
            state=state,
            calibrated_score=clamped_score,
            smoothed_score=smoothed,
            reason=reason,
            reason_details=details,
            window_index=window_index,
            window_start_sec=window_start_sec,
            window_end_sec=window_end_sec,
            speech_seconds=speech_seconds,
            history_scores=list(self.smoothed_history[-5:]),
        )
