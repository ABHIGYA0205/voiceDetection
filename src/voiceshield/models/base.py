"""Abstract base detector interface and standardized inference output schema."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, Field, model_validator


class DetectorOutput(BaseModel):
    """Inference output schema for all VoiceShield acoustic detectors."""

    @model_validator(mode="before")
    @classmethod
    def sync_score_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            score = data.get("synthetic_score")
            if score is None:
                score = data.get("uncalibrated_synthetic_score")
            if score is None:
                score = data.get("synthetic_likelihood")
            if score is not None:
                data.setdefault("synthetic_score", score)
                data.setdefault("uncalibrated_synthetic_score", score)
                data.setdefault("synthetic_likelihood", score)
        return data

    raw_logits: tuple[float, float] = Field(
        description="Raw output logits (index 0: spoof, index 1: bonafide).",
    )
    bonafide_score: float = Field(
        description="Bona fide score (higher = more likely authentic, standard ASVspoof metric).",
    )
    synthetic_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Uncalibrated acoustic anomaly score for synthetic/spoof speech (softmax index 0).",
    )
    uncalibrated_synthetic_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Uncalibrated acoustic anomaly score for synthetic/spoof speech.",
    )
    score_is_calibrated: bool = Field(
        default=False,
        description="Whether the score is calibrated against an empirical target dataset.",
    )
    operating_threshold: float = Field(
        default=0.65,
        description="Operating threshold used for suspected synthetic decision.",
    )
    operating_threshold_source: str = Field(
        default="unvalidated_heuristic",
        description="Provenance of operating threshold.",
    )
    synthetic_likelihood: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Deprecated compatibility alias for synthetic_score. Uncalibrated score, NOT a probability.",
    )
    is_synthetic_suspected: bool = Field(
        description="Binary decision flag based on operating_threshold.",
    )
    input_samples: int = Field(description="Sample count of the audio waveform analyzed.")
    device_used: str = Field(description="Execution device used for inference (e.g. 'mps', 'cpu').")
    model_id: str = Field(description="Model identifier or architecture tag.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Supplemental model diagnostics and intermediate representations.",
    )

    @classmethod
    def from_scores(
        cls,
        raw_logits: tuple[float, float],
        bonafide_score: float,
        synthetic_score: float,
        is_synthetic_suspected: bool,
        input_samples: int,
        device_used: str,
        model_id: str,
        metadata: dict[str, Any] | None = None,
        operating_threshold: float = 0.65,
        operating_threshold_source: str = "unvalidated_heuristic",
        score_is_calibrated: bool = False,
    ) -> DetectorOutput:
        """Convenience constructor populating canonical and compatibility fields."""
        return cls(
            raw_logits=raw_logits,
            bonafide_score=bonafide_score,
            synthetic_score=synthetic_score,
            uncalibrated_synthetic_score=synthetic_score,
            synthetic_likelihood=synthetic_score,
            score_is_calibrated=score_is_calibrated,
            operating_threshold=operating_threshold,
            operating_threshold_source=operating_threshold_source,
            is_synthetic_suspected=is_synthetic_suspected,
            input_samples=input_samples,
            device_used=device_used,
            model_id=model_id,
            metadata=metadata or {},
        )


class BaseDetector(ABC):
    """Common contract for all acoustic anti-spoofing and deepfake detectors."""

    is_mock: bool = False

    @abstractmethod
    def predict(
        self,
        audio: np.ndarray[Any, np.dtype[np.float32]] | torch.Tensor,
    ) -> DetectorOutput:
        """Run single-window inference on a 1D audio waveform.

        Args:
            audio: 1D float32 audio waveform.

        Returns:
            DetectorOutput containing raw logits, scores, and metadata.
        """

    @abstractmethod
    def predict_batch(
        self,
        batch: torch.Tensor,
    ) -> list[DetectorOutput]:
        """Run batched inference on a 2D tensor of shape (batch_size, n_samples).

        Args:
            batch: 2D float32 tensor of shape (B, N).

        Returns:
            List of DetectorOutput objects for each item in the batch.
        """
