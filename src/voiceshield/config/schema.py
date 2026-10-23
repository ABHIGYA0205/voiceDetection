"""Typed configuration schemas and validation invariants for VoiceShield."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _check_finite(val: float, field_name: str) -> float:
    """Verify that a float value is finite (not NaN or Inf)."""
    if not math.isfinite(val):
        raise ValueError(f"Value for {field_name} must be a finite number, got {val}")
    return val


class StrictFrozenConfigModel(BaseModel):
    """Shared base model enforcing strict schema validation and deep immutability."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class AudioConfig(StrictFrozenConfigModel):
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

    @field_validator("normalize_range")
    @classmethod
    def validate_normalize_range(cls, v: tuple[float, float]) -> tuple[float, float]:
        if len(v) != 2:
            raise ValueError(f"normalize_range must contain exactly 2 bounds, got {len(v)}.")
        low, high = v
        _check_finite(low, "normalize_range[0]")
        _check_finite(high, "normalize_range[1]")
        if low >= high:
            raise ValueError(
                f"Lower normalize bound ({low}) must be strictly less than upper bound ({high})."
            )
        if low != -1.0 or high != 1.0:
            raise ValueError(
                f"VoiceShield input contract strictly requires normalize_range (-1.0, 1.0), got ({low}, {high})."
            )
        return v


class WindowingConfig(StrictFrozenConfigModel):
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
    def validate_finite_floats(cls, v: float, info: Any) -> float:
        return _check_finite(v, info.field_name)

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


class ModelConfig(StrictFrozenConfigModel):
    """Acoustic detector model architecture and hardware execution parameters."""

    architecture: Literal["aasist_l"] = Field(
        default="aasist_l",
        description="Supported model architecture identifier (strictly 'aasist_l').",
    )
    checkpoint_path: str = Field(
        default="models/weights/AASIST-L.pth",
        description="Path to AASIST-L model weights checkpoint.",
    )
    device_preferred: str = Field(
        default="mps", description="Preferred acceleration device ('mps' or 'cpu')."
    )
    fallback_to_cpu: bool = Field(
        default=True, description="Gracefully fall back to CPU if preferred device unavailable."
    )

    @field_validator("checkpoint_path")
    @classmethod
    def validate_checkpoint_path(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("checkpoint_path cannot be empty.")
        return v.strip()

    @field_validator("device_preferred")
    @classmethod
    def validate_device_preferred(cls, v: str) -> str:
        norm = v.strip().lower()
        if norm not in ("mps", "cpu"):
            raise ValueError(f"device_preferred must be 'mps' or 'cpu', got {v!r}.")
        return norm


class TemporalPolicyConfig(StrictFrozenConfigModel):
    """Multi-hop temporal reasoning and decision policy parameters."""

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

    @field_validator(
        "threshold_high",
        "threshold_low",
        "ewma_alpha",
        "timing_gap_threshold_seconds",
    )
    @classmethod
    def validate_finite_policy_floats(cls, v: float, info: Any) -> float:
        return _check_finite(v, info.field_name)

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


class DCFConfig(StrictFrozenConfigModel):
    """Detection Cost Function (DCF) parameters for minDCF calculation."""

    c_miss: float = Field(default=1.0, gt=0, description="Cost of a false negative (missed spoof).")
    c_fa: float = Field(
        default=10.0, gt=0, description="Cost of a false alarm (false spoof alarm)."
    )
    p_target: float = Field(
        default=0.05, gt=0.0, lt=1.0, description="Prior probability of target class (spoof)."
    )

    @field_validator("c_miss", "c_fa", "p_target")
    @classmethod
    def validate_finite_dcf_floats(cls, v: float, info: Any) -> float:
        return _check_finite(v, info.field_name)


class EvaluationConfig(StrictFrozenConfigModel):
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

    @field_validator("bootstrap_ci_level")
    @classmethod
    def validate_finite_ci_level(cls, v: float, info: Any) -> float:
        return _check_finite(v, info.field_name)

    @field_validator("target_tpr_levels")
    @classmethod
    def validate_target_tpr_levels(cls, v: list[float]) -> list[float]:
        for i, level in enumerate(v):
            _check_finite(level, f"target_tpr_levels[{i}]")
            if level <= 0.0 or level >= 1.0:
                raise ValueError(f"target_tpr_levels elements must be in (0, 1), got {level}")
        return v


class VoiceShieldConfig(StrictFrozenConfigModel):
    """Canonical root configuration for VoiceShield."""

    audio: AudioConfig = Field(default_factory=AudioConfig)
    windowing: WindowingConfig = Field(default_factory=WindowingConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    temporal_policy: TemporalPolicyConfig = Field(default_factory=TemporalPolicyConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    def compute_hash(self) -> str:
        """Compute deterministic SHA-256 hash of this configuration."""
        canonical_json = json.dumps(
            self.model_dump(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to standard dictionary."""
        return self.model_dump()
