"""Adapter for the official Clova AI AASIST-L baseline detector.

Wraps the upstream AASIST architecture under a standardized BaseDetector contract,
enforcing 64,600-sample input framing, deterministic evaluation padding, explicit
checkpoint verification with actionable setup instructions, and MPS/CPU hardware fallback.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from voiceshield.core.device import get_device
from voiceshield.models.base import BaseDetector, DetectorOutput
from voiceshield.vendor.aasist.AASIST import Model as UpstreamAASISTModel

logger = logging.getLogger(__name__)

AASIST_L_DEFAULT_CONFIG: dict[str, Any] = {
    "architecture": "AASIST",
    "nb_samp": 64600,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 24], [24, 24]],
    "gat_dims": [24, 32],
    "pool_ratios": [0.4, 0.5, 0.7, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}

DEFAULT_CHECKPOINT_PATH = Path("models/weights/AASIST-L.pth")
DEFAULT_THRESHOLD = 0.65


class CheckpointNotFoundError(FileNotFoundError):
    """Raised when an expected model weights checkpoint is missing from the system."""


def pad_for_aasist(
    waveform: np.ndarray[Any, np.dtype[np.float32]] | torch.Tensor,
    max_len: int = 64600,
) -> torch.Tensor:
    """Deterministic evaluation padding matching clovaai/aasist data_utils.py.

    If input length < max_len, cyclically repeats the signal until reaching max_len.
    If input length >= max_len, crops the first max_len samples.

    Args:
        waveform: 1D audio sample array or Tensor.
        max_len: Target sample count (default: 64,600).

    Returns:
        1D float32 Tensor of shape (max_len,).
    """
    if isinstance(waveform, torch.Tensor):
        x = waveform.detach().cpu().numpy()
    else:
        x = np.asarray(waveform, dtype=np.float32)

    x_len = x.shape[0]
    if x_len == 0:
        raise ValueError("Cannot pad an empty audio waveform.")

    if x_len >= max_len:
        return torch.from_numpy(np.ascontiguousarray(x[:max_len], dtype=np.float32))

    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, num_repeats)[:max_len]
    return torch.from_numpy(np.ascontiguousarray(padded_x, dtype=np.float32))


def _verify_checkpoint_sha256(path: Path, expected_sha256: str) -> None:
    """Verify that a checkpoint matches an expected SHA-256 hash."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    actual_sha = hasher.hexdigest().lower()
    if actual_sha != expected_sha256.lower():
        raise ValueError(
            f"Checkpoint SHA-256 mismatch for {path}!\n"
            f"  Expected: {expected_sha256.lower()}\n"
            f"  Actual:   {actual_sha}"
        )


class AASISTLDetector(BaseDetector):
    """VoiceShield adapter for the official AASIST-L deepfake detector."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        expected_sha256: str | None = None,
        model_config: dict[str, Any] | None = None,
        preferred_device: str | None = None,
        fallback_to_cpu: bool = True,
        threshold: float = DEFAULT_THRESHOLD,
        _instantiated_model: UpstreamAASISTModel | None = None,
    ) -> None:
        """Initialize AASIST-L detector.

        Args:
            checkpoint_path: Path to weights file (defaults to models/weights/AASIST-L.pth).
            expected_sha256: Optional expected SHA-256 checksum to verify.
            model_config: Architecture hyperparameters dictionary (defaults to official AASIST-L).
            preferred_device: 'mps', 'cpu', or None to read environment.
            fallback_to_cpu: Whether to gracefully fall back to CPU if preferred device fails.
            threshold: Decision threshold for suspected synthetic classification.
            _instantiated_model: Internal parameter for unit-testing shape/control-flow with mocks.
        """
        self.threshold = threshold
        self.fallback_to_cpu = fallback_to_cpu
        self.config = model_config or AASIST_L_DEFAULT_CONFIG
        self.target_length = int(self.config.get("nb_samp", 64600))
        self.device = get_device(preferred=preferred_device, fallback_to_cpu=self.fallback_to_cpu)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT_PATH

        self.is_mock = _instantiated_model is not None
        if _instantiated_model is not None:
            # Used for shape & control-flow tests
            self.model = _instantiated_model
        else:
            if not self.checkpoint_path.is_file():
                setup_instructions = (
                    f"AASIST-L checkpoint not found at: {self.checkpoint_path}\n"
                    "VoiceShield forbids fabricating or inventing model weights.\n"
                    "Please download the official checkpoint using:\n\n"
                    "  mkdir -p models/weights\n"
                    f"  curl -L -o {self.checkpoint_path} "
                    "https://huggingface.co/SpeechAntiSpoofingBenchmarks/AASIST-L/resolve/main/AASIST-L.pth\n\n"
                    "Alternatively, install huggingface-hub and run:\n"
                    "  hf download SpeechAntiSpoofingBenchmarks/AASIST-L AASIST-L.pth --local-dir models/weights\n"
                )
                raise CheckpointNotFoundError(setup_instructions)

            target_sha = expected_sha256 or os.getenv("VOICESHIELD_EXPECTED_SHA256")
            if target_sha is not None and target_sha.strip():
                _verify_checkpoint_sha256(self.checkpoint_path, target_sha.strip())

            self.model = UpstreamAASISTModel(self.config)
            logger.info(
                "Loading AASIST-L weights from %s (weights_only=True)", self.checkpoint_path
            )
            try:
                state_dict = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
            except Exception as exc:
                raise ValueError(
                    f"Failed to load checkpoint safely via weights_only=True from {self.checkpoint_path}: {exc}"
                ) from exc
            self.model.load_state_dict(state_dict)

        self.model.eval()
        try:
            self.model.to(self.device)
        except Exception as exc:
            if not self.fallback_to_cpu:
                raise
            logger.warning(
                "Failed to place AASIST-L on device %s: %s. Falling back to CPU.", self.device, exc
            )
            self.device = torch.device("cpu")
            self.model.to(self.device)

    def predict(
        self,
        audio: np.ndarray[Any, np.dtype[np.float32]] | torch.Tensor,
    ) -> DetectorOutput:
        """Run inference on a single audio window.

        Args:
            audio: 1D float32 audio samples.

        Returns:
            DetectorOutput containing raw logits and synthetic likelihood.
        """
        padded_tensor = pad_for_aasist(audio, max_len=self.target_length)
        batch = padded_tensor.unsqueeze(0)  # Shape (1, 64600)
        outputs = self.predict_batch(batch)
        return outputs[0]

    def predict_batch(
        self,
        batch: torch.Tensor,
    ) -> list[DetectorOutput]:
        """Run batched inference on a 2D tensor of shape (B, N).

        Args:
            batch: Tensor of shape (B, N).

        Returns:
            List of DetectorOutput objects.
        """
        if batch.ndim != 2:
            raise ValueError(f"batch must be 2D of shape (batch_size, samples), got {batch.shape}.")

        batch_size = batch.shape[0]
        # Pad each item in batch to target length if needed
        processed_items: list[torch.Tensor] = []
        for i in range(batch_size):
            item = batch[i]
            if item.shape[0] != self.target_length:
                item = pad_for_aasist(item, max_len=self.target_length)
            processed_items.append(item)
        batch_cpu = torch.stack(processed_items, dim=0)

        fallback_used = False
        if self.device.type == "cpu":
            with torch.no_grad():
                _, logits = self.model(batch_cpu)
        else:
            try:
                tensor_in = batch_cpu.to(self.device)
                with torch.no_grad():
                    # Official forward returns: (last_hidden, output_logits)
                    _, logits = self.model(tensor_in)
            except Exception as exc:
                if not self.fallback_to_cpu:
                    raise
                logger.warning(
                    "AASIST execution failed on %s (%s). Executing CPU fallback.",
                    self.device,
                    exc,
                )
                fallback_used = True
                self.model.to("cpu")
                self.device = torch.device("cpu")
                with torch.no_grad():
                    _, logits = self.model(batch_cpu)

        # Label Mapping Invariant:
        # Index 0: spoof (synthetic)
        # Index 1: bonafide (authentic)
        logits_cpu = logits.detach().cpu()
        probs = F.softmax(logits_cpu, dim=-1)

        results: list[DetectorOutput] = []
        for i in range(batch_size):
            logit_spoof = float(logits_cpu[i, 0].item())
            logit_bonafide = float(logits_cpu[i, 1].item())
            prob_spoof = float(probs[i, 0].item())

            results.append(
                DetectorOutput.from_scores(
                    raw_logits=(logit_spoof, logit_bonafide),
                    bonafide_score=logit_bonafide,
                    synthetic_score=prob_spoof,
                    is_synthetic_suspected=bool(prob_spoof >= self.threshold),
                    input_samples=self.target_length,
                    device_used=self.device.type,
                    model_id="clovaai/aasist-l",
                    operating_threshold=self.threshold,
                    operating_threshold_source="unvalidated_heuristic",
                    score_is_calibrated=False,
                    metadata={
                        "upstream_repo": "https://github.com/clovaai/aasist",
                        "upstream_commit": "a04c9863f63d44471dde8a6abcb3b082b07cd1d1",
                        "label_mapping": {"0": "spoof", "1": "bonafide"},
                        "fallback_used": fallback_used,
                    },
                )
            )

        return results
