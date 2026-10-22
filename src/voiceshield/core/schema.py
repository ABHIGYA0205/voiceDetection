"""Pydantic schemas for the VoiceShield output contract and disclaimers."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Disclaimers(BaseModel):
    """Product and legal disclaimers accompanying all detection responses."""

    identity_proof: str = Field(
        default="Output does NOT prove identity of caller.",
        description="Authentic audio does not verify caller identity.",
    )
    fraud_probability: str = Field(
        default="Output is NOT a fraud probability.",
        description="Synthetic likelihood measures acoustic artifacts, not fraudulent intent.",
    )
    advisory_action: str = Field(
        default="Advisory only. Recommend independent verification; do not automatically block calls.",
        description="Systems must never automatically drop calls based on this output.",
    )
    capture_boundary: str = Field(
        default="Controlled microphone or media stream only; browser cannot capture arbitrary cellular calls.",
        description="Captures user-authorized microphone streams only.",
    )


class QualityStatus(BaseModel):
    """Audio quality diagnostics for the analyzed window."""

    is_clipping: bool = Field(default=False, description="Whether audio exhibits digital clipping.")
    snr_db_estimate: float = Field(
        default=30.0, description="Estimated signal-to-noise ratio in dB."
    )
    is_silent: bool = Field(default=False, description="Whether window is dominated by silence.")
    quality_warning: str | None = Field(default=None, description="Human-readable quality warning.")


class UncertaintyStatus(BaseModel):
    """Confidence and epistemic uncertainty status."""

    is_uncertain: bool = Field(default=False, description="Flag indicating unreliable prediction.")
    reasons: list[str] = Field(
        default_factory=list, description="Factors contributing to uncertainty."
    )


class DetectionResult(BaseModel):
    """Primary output schema contract for VoiceShield inference."""

    synthetic_likelihood: float = Field(
        ge=0.0,
        le=1.0,
        description="Calibrated probability of synthetic speech or voice conversion in [0.0, 1.0].",
    )
    is_synthetic_suspected: bool = Field(
        description="Binary advisory decision based on the calibrated threshold.",
    )
    window_start_seconds: float = Field(
        description="Start offset of the analyzed window in seconds."
    )
    window_end_seconds: float = Field(description="End offset of the analyzed window in seconds.")
    quality_status: QualityStatus = Field(description="Audio input quality metrics.")
    uncertainty_status: UncertaintyStatus = Field(description="Uncertainty diagnostics.")
    temporal_rule_explanation: str = Field(
        description="Human-readable explanation of multi-hop temporal aggregation decisions.",
    )
    disclaimers: Disclaimers = Field(
        default_factory=Disclaimers,
        description="Mandatory compliance and product scope disclaimers.",
    )
