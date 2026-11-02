"""Input reliability assessment policy for audio quality and duration gating.

Audio quality is a separate reliability signal. It must never be interpreted as
evidence that audio is human or synthetic. This module defines the typed policy
foundation for evaluating whether audio input is reliable, limited, or insufficient
for downstream analysis, independent of model scores or origin classifications.
"""

from __future__ import annotations

import math
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from voiceshield.dsp.quality import AudioQualityMetrics


class ReliabilityState(StrEnum):
    """Categorical assessment of input audio reliability."""

    RELIABLE = "reliable"
    LIMITED = "limited"
    INSUFFICIENT = "insufficient"


class ReliabilityReason(StrEnum):
    """Machine-readable reason codes for input reliability classification."""

    EMPTY_AUDIO = "empty_audio"
    SILENT_AUDIO = "silent_audio"
    INSUFFICIENT_SPEECH = "insufficient_speech"
    SEVERE_CLIPPING = "severe_clipping"
    LOW_ENERGY = "low_energy"
    MARGINAL_SNR = "marginal_snr"


# Fixed deterministic ordering of reasons
REASON_ORDER: tuple[ReliabilityReason, ...] = (
    ReliabilityReason.EMPTY_AUDIO,
    ReliabilityReason.SILENT_AUDIO,
    ReliabilityReason.INSUFFICIENT_SPEECH,
    ReliabilityReason.SEVERE_CLIPPING,
    ReliabilityReason.LOW_ENERGY,
    ReliabilityReason.MARGINAL_SNR,
)

INSUFFICIENT_REASONS: frozenset[ReliabilityReason] = frozenset(
    {
        ReliabilityReason.EMPTY_AUDIO,
        ReliabilityReason.SILENT_AUDIO,
        ReliabilityReason.INSUFFICIENT_SPEECH,
    }
)

LIMITED_REASONS: frozenset[ReliabilityReason] = frozenset(
    {
        ReliabilityReason.SEVERE_CLIPPING,
        ReliabilityReason.LOW_ENERGY,
        ReliabilityReason.MARGINAL_SNR,
    }
)

ALLOWED_QUALITY_STATUSES: frozenset[str] = frozenset(
    {"empty", "silent", "clipped", "low_energy", "marginal_snr", "clean"}
)


class InputReliabilityAssessment(BaseModel):
    """Immutable assessment of input audio reliability."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    state: ReliabilityState = Field(
        description="Categorical reliability state (reliable, limited, insufficient)."
    )
    reasons: tuple[ReliabilityReason, ...] = Field(
        default_factory=tuple,
        description="Deterministic tuple of machine-readable reason codes explaining the reliability state.",
    )
    explanation: str = Field(description="Human-readable explanation of the assessment.")
    origin_verdict_allowed: bool = Field(
        description="Whether an origin verdict (synthetic/human) is permitted under this assessment."
    )

    @model_validator(mode="after")
    def _validate_invariants(self) -> InputReliabilityAssessment:
        """Validate structural and policy invariants across fields."""
        if not self.explanation.strip():
            raise ValueError("explanation must contain non-whitespace text")

        if len(self.reasons) != len(set(self.reasons)):
            raise ValueError("reasons cannot contain duplicate entries")

        expected_ordered = tuple(r for r in REASON_ORDER if r in set(self.reasons))
        if self.reasons != expected_ordered:
            raise ValueError(
                f"reasons must appear in exact REASON_ORDER order: expected {expected_ordered}, got {self.reasons}"
            )

        expected_origin_allowed = self.state == ReliabilityState.RELIABLE
        if self.origin_verdict_allowed != expected_origin_allowed:
            raise ValueError(
                f"origin_verdict_allowed must be {expected_origin_allowed} for state {self.state.value}"
            )

        if self.state == ReliabilityState.RELIABLE:
            if len(self.reasons) != 0:
                raise ValueError("state RELIABLE must have zero reasons")
        elif self.state == ReliabilityState.LIMITED:
            if any(r in INSUFFICIENT_REASONS for r in self.reasons):
                raise ValueError("state LIMITED cannot contain any INSUFFICIENT_REASONS member")
            if not any(r in LIMITED_REASONS for r in self.reasons):
                raise ValueError("state LIMITED must have at least one LIMITED_REASONS member")
        elif self.state == ReliabilityState.INSUFFICIENT:
            if not any(r in INSUFFICIENT_REASONS for r in self.reasons):
                raise ValueError(
                    "state INSUFFICIENT must have at least one INSUFFICIENT_REASONS member"
                )

        return self


def _format_reason_detail(
    reason: ReliabilityReason,
    active_speech_seconds: float,
    required_min_speech_seconds: float,
) -> str:
    """Format human-readable detail for a specific reliability reason."""
    if reason == ReliabilityReason.EMPTY_AUDIO:
        return "audio buffer is empty"
    if reason == ReliabilityReason.SILENT_AUDIO:
        return "audio energy is effectively silent (RMS < 1e-4)"
    if reason == ReliabilityReason.INSUFFICIENT_SPEECH:
        return (
            f"active speech duration ({active_speech_seconds:.2f}s) is below required minimum "
            f"({required_min_speech_seconds:.2f}s)"
        )
    if reason == ReliabilityReason.SEVERE_CLIPPING:
        return "severe digital clipping detected"
    if reason == ReliabilityReason.LOW_ENERGY:
        return "audio signal energy is low (RMS < 0.01)"
    if reason == ReliabilityReason.MARGINAL_SNR:
        return "estimated signal-to-noise ratio is marginal (SNR < 12 dB)"
    return str(reason.value)


def assess_input_reliability(
    metrics: AudioQualityMetrics,
    active_speech_seconds: float,
    required_min_speech_seconds: float,
) -> InputReliabilityAssessment:
    """Assess audio input reliability based on acoustic quality and speech duration.

    This function evaluates physical signal quality and speech presence to determine
    whether input audio is suitable for downstream analysis. It does not inspect
    or produce model scores or origin verdicts.

    Args:
        metrics: Physical audio quality metrics calculated from the input.
        active_speech_seconds: Duration of active speech detected in seconds.
        required_min_speech_seconds: Minimum active speech duration required.

    Returns:
        Immutable InputReliabilityAssessment instance.

    Raises:
        TypeError: If input types are invalid.
        ValueError: If quality status or physical metric ranges are invalid,
            or if duration values are NaN, infinite, negative, or invalid.
    """
    if not isinstance(metrics, AudioQualityMetrics):
        raise TypeError(
            f"metrics must be an AudioQualityMetrics instance, got {type(metrics).__name__}"
        )

    if metrics.quality_status not in ALLOWED_QUALITY_STATUSES:
        raise ValueError(
            f"Unknown AudioQualityMetrics.quality_status: {metrics.quality_status!r}. "
            f"Must be one of {sorted(ALLOWED_QUALITY_STATUSES)}."
        )

    if math.isnan(metrics.peak_level) or math.isinf(metrics.peak_level) or metrics.peak_level < 0.0:
        raise ValueError(
            f"AudioQualityMetrics.peak_level must be finite and >= 0, got {metrics.peak_level}"
        )

    if math.isnan(metrics.rms) or math.isinf(metrics.rms) or metrics.rms < 0.0:
        raise ValueError(f"AudioQualityMetrics.rms must be finite and >= 0, got {metrics.rms}")

    if (
        math.isnan(metrics.clipping_fraction)
        or math.isinf(metrics.clipping_fraction)
        or not (0.0 <= metrics.clipping_fraction <= 1.0)
    ):
        raise ValueError(
            "AudioQualityMetrics.clipping_fraction must be finite and in range [0, 1], "
            f"got {metrics.clipping_fraction}"
        )

    if math.isnan(metrics.snr_db_estimate) or math.isinf(metrics.snr_db_estimate):
        raise ValueError(
            f"AudioQualityMetrics.snr_db_estimate must be a finite float, got {metrics.snr_db_estimate}"
        )

    if not isinstance(active_speech_seconds, (int, float)) or isinstance(
        active_speech_seconds, bool
    ):
        raise TypeError(
            f"active_speech_seconds must be a real float, got {type(active_speech_seconds).__name__}"
        )

    active_speech_sec = float(active_speech_seconds)
    if math.isnan(active_speech_sec):
        raise ValueError("active_speech_seconds cannot be NaN")
    if math.isinf(active_speech_sec):
        raise ValueError("active_speech_seconds cannot be infinite")
    if active_speech_sec < 0.0:
        raise ValueError(f"active_speech_seconds must be non-negative, got {active_speech_sec}")

    if not isinstance(required_min_speech_seconds, (int, float)) or isinstance(
        required_min_speech_seconds, bool
    ):
        raise TypeError(
            f"required_min_speech_seconds must be a real float, got {type(required_min_speech_seconds).__name__}"
        )

    req_min_sec = float(required_min_speech_seconds)
    if math.isnan(req_min_sec):
        raise ValueError("required_min_speech_seconds cannot be NaN")
    if math.isinf(req_min_sec):
        raise ValueError("required_min_speech_seconds cannot be infinite")
    if req_min_sec <= 0.0:
        raise ValueError(
            f"required_min_speech_seconds must be strictly positive, got {req_min_sec}"
        )

    is_empty = metrics.quality_status == "empty"
    is_silent = (metrics.is_silent or metrics.quality_status == "silent") and not is_empty
    is_insufficient_speech = active_speech_sec < req_min_sec
    is_clipped = bool(metrics.is_clipped or metrics.quality_status == "clipped")
    is_low_energy = (metrics.quality_status == "low_energy") or (
        metrics.rms < 0.01 and not is_empty and not is_silent
    )
    is_marginal_snr = (metrics.quality_status == "marginal_snr") or (
        metrics.snr_db_estimate < 12.0 and not is_empty and not is_silent
    )

    triggered: set[ReliabilityReason] = set()
    if is_empty:
        triggered.add(ReliabilityReason.EMPTY_AUDIO)
    if is_silent:
        triggered.add(ReliabilityReason.SILENT_AUDIO)
    if is_insufficient_speech:
        triggered.add(ReliabilityReason.INSUFFICIENT_SPEECH)
    if is_clipped:
        triggered.add(ReliabilityReason.SEVERE_CLIPPING)
    if is_low_energy:
        triggered.add(ReliabilityReason.LOW_ENERGY)
    if is_marginal_snr:
        triggered.add(ReliabilityReason.MARGINAL_SNR)

    reasons: tuple[ReliabilityReason, ...] = tuple(r for r in REASON_ORDER if r in triggered)

    if any(r in INSUFFICIENT_REASONS for r in reasons):
        state = ReliabilityState.INSUFFICIENT
    elif any(r in LIMITED_REASONS for r in reasons):
        state = ReliabilityState.LIMITED
    else:
        state = ReliabilityState.RELIABLE

    origin_verdict_allowed = state == ReliabilityState.RELIABLE

    if state == ReliabilityState.RELIABLE:
        explanation = (
            "Input audio meets quality and speech duration requirements for reliable analysis."
        )
    elif state == ReliabilityState.INSUFFICIENT:
        details = "; ".join(
            _format_reason_detail(r, active_speech_sec, req_min_sec) for r in reasons
        )
        explanation = f"Input audio is insufficient for origin assessment: {details}."
    else:  # LIMITED
        details = "; ".join(
            _format_reason_detail(r, active_speech_sec, req_min_sec) for r in reasons
        )
        explanation = f"Input audio has limited reliability due to degraded acoustics: {details}."

    return InputReliabilityAssessment(
        state=state,
        reasons=reasons,
        explanation=explanation,
        origin_verdict_allowed=origin_verdict_allowed,
    )
