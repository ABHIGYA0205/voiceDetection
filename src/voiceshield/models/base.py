"""Abstract base detector interface and standardized inference output schema."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, Field


class DetectorOutput(BaseModel):
    """Inference output schema for all VoiceShield acoustic detectors."""

    raw_logits: tuple[float, float] = Field(
        description="Raw output logits (index 0: spoof, index 1: bonafide).",
    )
    bonafide_score: float = Field(
        description="Bona fide score (higher = more likely authentic, standard ASVspoof metric).",
    )
    synthetic_likelihood: float = Field(
        ge=0.0,
        le=1.0,
        description="Calibrated or normalized probability of synthetic speech/spoof in [0.0, 1.0].",
    )
    is_synthetic_suspected: bool = Field(
        description="Binary decision flag based on calibrated threshold.",
    )
    input_samples: int = Field(description="Sample count of the audio waveform analyzed.")
    device_used: str = Field(description="Execution device used for inference (e.g. 'mps', 'cpu').")
    model_id: str = Field(description="Model identifier or architecture tag.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Supplemental model diagnostics and intermediate representations.",
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
