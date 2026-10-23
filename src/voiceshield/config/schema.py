"""Typed configuration schemas and validation invariants for VoiceShield."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class AudioConfig(BaseModel):
    """Specification of strict input audio format."""

    sample_rate: int = Field(default=16000, description="Sampling rate in Hz (strictly 16,000 Hz).")
    channels: int = Field(default=1, description="Number of audio channels (strictly 1 mono).")
    dtype: str = Field(default="float32", description="Floating-point precision format.")
    normalize_range: tuple[float, float] = Field(
        default=(-1.0, 1.0),
        description="Normalization bounds for audio samples.",
    )

    @field_validator("sample_rate")
    @classmethod
    def validate_sample_rate(cls, v: int) -> int:
        if v != 16000:
            raise ValueError(
                f"VoiceShield input contract strictly requires 16000 Hz sample rate, got {v} Hz."
            )
        return v

    @field_validator("channels")
    @classmethod
    def validate_channels(cls, v: int) -> int:
        if v != 1:
            raise ValueError(
                f"VoiceShield input contract strictly requires single-channel (mono) audio, got {v}."
            )
        return v

    @field_validator("dtype")
    @classmethod
    def validate_dtype(cls, v: str) -> str:
        if v != "float32":
            raise ValueError(f"Audio dtype must be 'float32', got {v!r}.")
        return v


class WindowingConfig(BaseModel):
    """Sliding window and temporal segmentation parameters."""

    window_seconds: float = Field(
        default=4.0, gt=0, description="Model window duration in seconds."
    )
    hop_seconds: float = Field(default=1.0, gt=0, description="Streaming hop step in seconds.")
    min_speech_seconds: float = Field(
        default=0.8,
        gt=0,
        description="Minimum active speech duration required before triggering inference.",
    )
    energy_threshold: float = Field(
        default=0.015,
        ge=0,
        description="Root-mean-square (RMS) energy threshold for speech presence.",
    )

    @field_validator("window_seconds", "hop_seconds", "min_speech_seconds", "energy_threshold")
    @classmethod
    def validate_finite_floats(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError(f"Value must be a finite number, got {v}")
        return v

    @model_validator(mode="after")
    def validate_window_relationships(self) -> WindowingConfig:
        if self.hop_seconds > self.window_seconds:
            raise ValueError(
                f"Hop duration ({self.hop_seconds}s) cannot be greater than window duration ({self.window_seconds}s)."
            )
        if self.min_speech_seconds > self.window_seconds:
            raise ValueError(
                f"Minimum speech duration ({self.min_speech_seconds}s) cannot exceed window duration ({self.window_seconds}s)."
            )
        return self

    def window_samples(self, sample_rate: int = 16000) -> int:
        """Return the number of audio samples in one model window."""
        return int(round(self.window_seconds * sample_rate))

    def hop_samples(self, sample_rate: int = 16000) -> int:
        """Return the number of audio samples in one streaming hop."""
        return int(round(self.hop_seconds * sample_rate))


class ModelConfig(BaseModel):
    """Acoustic detector model architecture and hardware execution parameters."""

    architecture: str = Field(default="lcnn", description="Model architecture identifier.")
    checkpoint_path: str | None = Field(
        default=None, description="Path to model weights checkpoint."
    )
    num_classes: int = Field(default=2, ge=2, description="Output classification targets.")
    embedding_dim: int = Field(default=128, gt=0, description="Latent representation dimension.")
    device_preferred: str = Field(
        default="mps", description="Preferred acceleration device ('mps' or 'cpu')."
    )
    fallback_to_cpu: bool = Field(
        default=True, description="Gracefully fall back to CPU if MPS unavailable."
    )


class TrainingConfig(BaseModel):
    """Reproducible training and optimization hyperparameters."""

    batch_size: int = Field(default=32, gt=0, description="Mini-batch size.")
    learning_rate: float = Field(default=1e-4, gt=0, description="Initial learning rate.")
    weight_decay: float = Field(default=1e-5, ge=0, description="L2 weight regularization.")
    max_epochs: int = Field(default=50, gt=0, description="Maximum training epochs.")
    early_stopping_patience: int = Field(
        default=7, gt=0, description="Patience epochs before early stopping."
    )
    seed: int = Field(default=42, description="Global random seed.")
    deterministic: bool = Field(
        default=True, description="Enforce deterministic operations where supported."
    )


class CalibrationConfig(BaseModel):
    """Score calibration parameters."""

    method: str = Field(default="temperature_scaling", description="Calibration algorithm.")
    temperature: float = Field(default=1.0, gt=0, description="Temperature scaling factor.")
    threshold: float = Field(
        default=0.65, ge=0.0, le=1.0, description="Decision threshold for warning."
    )


class TemporalPolicyConfig(BaseModel):
    """Multi-hop temporal reasoning and explanation generation rules."""

    consecutive_hops_required: int = Field(
        default=2,
        ge=1,
        description="Number of consecutive high-anomaly hops required to trigger advisory alert.",
    )
    decay_factor: float = Field(
        default=0.85, gt=0.0, le=1.0, description="Exponential decay factor."
    )
    alert_threshold: float = Field(
        default=0.65, ge=0.0, le=1.0, description="Threshold for warning."
    )
    threshold_high: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Upper threshold for high synthetic speech likelihood.",
    )
    threshold_low: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Lower threshold for low synthetic speech likelihood.",
    )
    smoothing_method: str = Field(
        default="ewma",
        description="Smoothing method for temporal window scores ('ewma', 'median', or 'none').",
    )
    ewma_alpha: float = Field(
        default=0.3, gt=0.0, le=1.0, description="Smoothing factor for EWMA (alpha)."
    )
    median_window_size: int = Field(
        default=5, ge=1, description="Sliding window size for median smoothing."
    )
    elevation_n: int = Field(
        default=3, ge=1, description="Number of anomalous windows required in N-of-M elevation."
    )
    elevation_m: int = Field(
        default=5, ge=1, description="Window history horizon M for N-of-M elevation."
    )
    hysteresis_consecutive_drops: int = Field(
        default=3,
        ge=1,
        description="Consecutive low windows required before de-escalating from HIGH state.",
    )
    timing_gap_threshold_seconds: float = Field(
        default=0.25,
        gt=0.0,
        description="Timestamp jump threshold in seconds to flag timing discontinuity.",
    )
    explanation_rules: list[str] = Field(
        default_factory=lambda: [
            "REQUIRE_MINIMUM_SPEECH_DURATION",
            "SUSTAINED_ANOMALY_CONFIRMATION",
            "SNR_QUALITY_GATING",
        ],
        description="Active rule identifiers for temporal explanations.",
    )

    @field_validator(
        "alert_threshold",
        "threshold_high",
        "threshold_low",
        "ewma_alpha",
        "decay_factor",
        "timing_gap_threshold_seconds",
    )
    @classmethod
    def validate_finite_policy_floats(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError(f"Value must be a finite number, got {v}")
        return v

    @model_validator(mode="after")
    def validate_policy_parameters(self) -> TemporalPolicyConfig:
        if self.threshold_low > self.threshold_high:
            raise ValueError(
                f"threshold_low ({self.threshold_low}) cannot exceed threshold_high ({self.threshold_high})."
            )
        if self.elevation_n > self.elevation_m:
            raise ValueError(
                f"elevation_n ({self.elevation_n}) cannot exceed elevation_m ({self.elevation_m})."
            )
        if self.smoothing_method not in ("ewma", "median", "none"):
            raise ValueError(
                f"Invalid smoothing_method {self.smoothing_method!r}. Must be 'ewma', 'median', or 'none'."
            )
        return self


class APIConfig(BaseModel):
    """FastAPI service endpoints and network parameters."""

    host: str = Field(default="127.0.0.1", description="Service bind address.")
    port: int = Field(default=8000, gt=0, le=65535, description="Service bind port.")
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"],
        description="Allowed CORS origin URLs.",
    )


class LoggingConfig(BaseModel):
    """Logging verbosity and format configuration."""

    level: str = Field(default="INFO", description="Log level.")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Standard logger format string.",
    )


class DCFConfig(BaseModel):
    """Detection Cost Function (DCF) parameters for minDCF calculation."""

    c_miss: float = Field(default=1.0, gt=0, description="Cost of a false negative (missed spoof).")
    c_fa: float = Field(
        default=10.0, gt=0, description="Cost of a false alarm (false spoof alarm)."
    )
    p_target: float = Field(
        default=0.05, gt=0.0, lt=1.0, description="Prior probability of target class (spoof)."
    )


class EvaluationConfig(BaseModel):
    """Configuration parameters for evaluation and reporting toolkit."""

    dcf: DCFConfig = Field(default_factory=DCFConfig)
    target_tpr_levels: list[float] = Field(
        default_factory=lambda: [0.90, 0.95, 0.99],
        description="Target TPR levels for evaluating operating FPR.",
    )
    bootstrap_iterations: int = Field(
        default=500, ge=10, description="Number of clustered bootstrap resampling iterations."
    )
    bootstrap_ci_level: float = Field(
        default=0.95, gt=0.5, lt=1.0, description="Bootstrap confidence interval level."
    )
    ece_num_bins: int = Field(
        default=10, ge=2, description="Number of bins for Expected Calibration Error."
    )


class VoiceShieldConfig(BaseModel):
    """Canonical root configuration for VoiceShield."""

    audio: AudioConfig = Field(default_factory=AudioConfig)
    windowing: WindowingConfig = Field(default_factory=WindowingConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    temporal_policy: TemporalPolicyConfig = Field(default_factory=TemporalPolicyConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    def compute_hash(self) -> str:
        """Compute deterministic SHA-256 hash of this configuration."""
        canonical_json = json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to standard dictionary."""
        return self.model_dump()
