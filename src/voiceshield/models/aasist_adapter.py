"""Adapter for the official Clova AI AASIST-L baseline detector.

Wraps the upstream AASIST architecture under a standardized BaseDetector contract,
enforcing 64,600-sample input framing, deterministic evaluation padding, explicit
checkpoint verification with actionable setup instructions, and MPS/CPU hardware fallback.
"""

from __future__ import annotations

import copy
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
from voiceshield.models.provenance import ModelProvenance
from voiceshield.models.registry import (
    CheckpointIntegrityError,
    CheckpointLoadError,
    CheckpointNotFoundError,
    UnsupportedModelIdentityError,
    get_official_registry_entry,
)
from voiceshield.vendor.aasist.AASIST import Model as UpstreamAASISTModel

logger = logging.getLogger(__name__)


def get_canonical_aasist_l_config() -> dict[str, Any]:
    """Return a fresh deep copy of the canonical official AASIST-L architecture configuration."""
    return {
        "architecture": "AASIST",
        "nb_samp": 64600,
        "first_conv": 128,
        "filts": [70, [1, 32], [32, 32], [32, 24], [24, 24]],
        "gat_dims": [24, 32],
        "pool_ratios": [0.4, 0.5, 0.7, 0.5],
        "temperatures": [2.0, 2.0, 100.0, 100.0],
    }


# Backward-compatible public dictionary; official initialization does not rely on its mutable reference
AASIST_L_DEFAULT_CONFIG: dict[str, Any] = get_canonical_aasist_l_config()


def _normalize_config_to_tuples(c: Any) -> Any:
    if isinstance(c, dict):
        return tuple(sorted((k, _normalize_config_to_tuples(v)) for k, v in c.items()))
    if isinstance(c, (list, tuple)):
        return tuple(_normalize_config_to_tuples(x) for x in c)
    return c


def is_canonical_aasist_l_config(cfg: dict[str, Any]) -> bool:
    """Check if an architecture config matches canonical official AASIST-L in structure and values."""
    canonical = get_canonical_aasist_l_config()
    return bool(_normalize_config_to_tuples(cfg) == _normalize_config_to_tuples(canonical))


DEFAULT_CHECKPOINT_PATH = Path("models/weights/AASIST-L.pth")
DEFAULT_THRESHOLD = 0.65


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
        raise CheckpointIntegrityError(
            f"Checkpoint SHA-256 mismatch for {path}!\n"
            f"  Expected: {expected_sha256.lower()}\n"
            f"  Actual:   {actual_sha}"
        )


class AASISTLDetector(BaseDetector):
    """VoiceShield adapter for the official AASIST-L deepfake detector."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        model_id: str = "clovaai/aasist-l",
        expected_sha256: str | None = None,
        model_config: dict[str, Any] | None = None,
        preferred_device: str | None = None,
        fallback_to_cpu: bool = True,
        threshold: float = DEFAULT_THRESHOLD,
        _instantiated_model: torch.nn.Module | None = None,
    ) -> None:
        """Initialize AASIST-L detector.

        Args:
            checkpoint_path: Path to weights file (defaults to models/weights/AASIST-L.pth).
            model_id: Model identity ('clovaai/aasist-l', 'custom/aasist-l', or 'mock/aasist-l').
            expected_sha256: Optional expected SHA-256 checksum to verify.
            model_config: Architecture hyperparameters dictionary (defaults to official AASIST-L).
            preferred_device: 'cuda', 'mps', 'cpu', or None to read environment.
            fallback_to_cpu: Whether to gracefully fall back to CPU if preferred device fails (MPS only).
            threshold: Decision threshold for suspected synthetic classification.
            _instantiated_model: Internal parameter for unit-testing shape/control-flow with mocks.
        """
        self.threshold = threshold
        self.fallback_to_cpu = fallback_to_cpu
        self.device = get_device(preferred=preferred_device, fallback_to_cpu=self.fallback_to_cpu)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT_PATH
        self.is_mock = _instantiated_model is not None or model_id == "mock/aasist-l"

        if model_id == "clovaai/aasist-l":
            if model_config is not None and not is_canonical_aasist_l_config(model_config):
                raise CheckpointIntegrityError(
                    f"Noncanonical architecture configuration provided for official model '{model_id}'. "
                    "Official AASIST-L strictly requires canonical architecture configuration."
                )
            self.config = get_canonical_aasist_l_config()
            self.target_length = int(self.config["nb_samp"])
        else:
            self.config = (
                copy.deepcopy(model_config)
                if model_config is not None
                else get_canonical_aasist_l_config()
            )
            self.target_length = int(self.config.get("nb_samp", 64600))

        if _instantiated_model is not None:
            self.model = _instantiated_model
            try:
                raw_count = sum(p.numel() for p in self.model.parameters())
            except Exception as exc:
                raise ValueError(
                    f"Failed to enumerate parameters on injected model: {exc}"
                ) from exc
            if raw_count <= 0:
                raise ValueError(
                    f"Injected model parameter count must be positive, got {raw_count}. "
                    "Cannot instantiate mock provenance with zero parameters."
                )
            param_count = int(raw_count)
            self.provenance = ModelProvenance(
                model_id="mock/aasist-l",
                architecture="aasist_l",
                provenance_status="mock",
                checkpoint_sha256=None,
                checkpoint_verified=False,
                expected_checkpoint_sha256=None,
                upstream_repository=None,
                upstream_commit=None,
                parameter_count=param_count,
                is_mock=True,
                checkpoint_size_bytes=None,
                source_license_spdx=None,
                checkpoint_rights_note=None,
            )
        elif model_id == "mock/aasist-l":
            self.model = UpstreamAASISTModel(self.config)
            param_count = sum(p.numel() for p in self.model.parameters())
            self.provenance = ModelProvenance(
                model_id="mock/aasist-l",
                architecture="aasist_l",
                provenance_status="mock",
                checkpoint_sha256=None,
                checkpoint_verified=False,
                expected_checkpoint_sha256=None,
                upstream_repository=None,
                upstream_commit=None,
                parameter_count=param_count,
                is_mock=True,
                checkpoint_size_bytes=None,
                source_license_spdx=None,
                checkpoint_rights_note=None,
            )
        elif model_id == "clovaai/aasist-l":
            reg = get_official_registry_entry(model_id)

            if not self.checkpoint_path.is_file():
                setup_instructions = (
                    f"AASIST-L checkpoint not found at: {self.checkpoint_path}\n"
                    "VoiceShield forbids fabricating or inventing model weights.\n"
                    "Setup Instructions: Please download the official checkpoint from the pinned upstream release:\n\n"
                    "  mkdir -p models/weights\n"
                    f"  curl -L -o {self.checkpoint_path} \\\n"
                    f"    {reg.checkpoint_url}\n\n"
                    f"Expected SHA-256: {reg.checkpoint_sha256} ({reg.checkpoint_size_bytes} bytes)\n"
                    f"Pinned Upstream Commit: {reg.upstream_commit}"
                )
                raise CheckpointNotFoundError(setup_instructions)

            actual_size = self.checkpoint_path.stat().st_size
            hasher = hashlib.sha256()
            with open(self.checkpoint_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    hasher.update(chunk)
            actual_sha = hasher.hexdigest().lower()

            if actual_sha != reg.checkpoint_sha256:
                raise CheckpointIntegrityError(
                    f"Official AASIST-L checkpoint SHA-256 mismatch for {self.checkpoint_path}!\n"
                    f"  Expected (registry): {reg.checkpoint_sha256}\n"
                    f"  Actual:              {actual_sha}\n"
                    "Official model identity 'clovaai/aasist-l' strictly requires byte verification against the immutable registry."
                )

            if expected_sha256 is not None:
                exp_norm = expected_sha256.strip().lower()
                if exp_norm != reg.checkpoint_sha256:
                    raise CheckpointIntegrityError(
                        f"Supplied expected_sha256 conflicts with official registry SHA-256 for {model_id}!\n"
                        f"  Registry: {reg.checkpoint_sha256}\n"
                        f"  Supplied: {exp_norm}"
                    )

            env_expected = os.getenv("VOICESHIELD_EXPECTED_SHA256")
            if env_expected and env_expected.strip().lower() != reg.checkpoint_sha256:
                logger.warning(
                    "VOICESHIELD_EXPECTED_SHA256 is ignored in official mode; official registry SHA-256 takes precedence."
                )

            self.model = UpstreamAASISTModel(self.config)
            logger.info(
                "Loading official AASIST-L weights from %s (weights_only=True)",
                self.checkpoint_path,
            )
            try:
                state_dict = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
            except Exception as exc:
                raise CheckpointLoadError(
                    f"Failed to load checkpoint safely via weights_only=True from {self.checkpoint_path}: {exc}"
                ) from exc

            try:
                self.model.load_state_dict(state_dict, strict=True)
            except Exception as exc:
                raise CheckpointIntegrityError(
                    f"AASIST-L state dictionary in {self.checkpoint_path} is incompatible with architecture: {exc}"
                ) from exc

            param_count = sum(p.numel() for p in self.model.parameters())
            if param_count != reg.parameter_count:
                raise CheckpointIntegrityError(
                    f"AASIST-L parameter count mismatch for {self.checkpoint_path}!\n"
                    f"  Expected: {reg.parameter_count}\n"
                    f"  Actual:   {param_count}"
                )

            self.provenance = ModelProvenance(
                model_id="clovaai/aasist-l",
                architecture="aasist_l",
                provenance_status="official_verified",
                checkpoint_sha256=actual_sha,
                checkpoint_verified=True,
                expected_checkpoint_sha256=reg.checkpoint_sha256,
                upstream_repository=reg.upstream_repository,
                upstream_commit=reg.upstream_commit,
                parameter_count=param_count,
                is_mock=False,
                checkpoint_size_bytes=actual_size,
                source_license_spdx=reg.source_license_spdx,
                checkpoint_rights_note=reg.checkpoint_rights_note,
            )
        elif model_id == "custom/aasist-l":
            if not self.checkpoint_path.is_file():
                raise CheckpointNotFoundError(
                    f"Custom AASIST-L checkpoint not found at: {self.checkpoint_path}"
                )

            actual_size = self.checkpoint_path.stat().st_size
            hasher = hashlib.sha256()
            with open(self.checkpoint_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    hasher.update(chunk)
            actual_sha = hasher.hexdigest().lower()

            if expected_sha256 is not None:
                exp_norm = expected_sha256.strip().lower()
                if exp_norm != actual_sha:
                    raise CheckpointIntegrityError(
                        f"Custom checkpoint SHA-256 mismatch for {self.checkpoint_path}!\n"
                        f"  Expected: {exp_norm}\n"
                        f"  Actual:   {actual_sha}"
                    )

            self.model = UpstreamAASISTModel(self.config)
            logger.info(
                "Loading custom AASIST-L weights from %s (weights_only=True)", self.checkpoint_path
            )
            try:
                state_dict = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
            except Exception as exc:
                raise CheckpointLoadError(
                    f"Failed to load custom checkpoint safely via weights_only=True from {self.checkpoint_path}: {exc}"
                ) from exc

            try:
                self.model.load_state_dict(state_dict, strict=True)
            except Exception as exc:
                raise CheckpointIntegrityError(
                    f"Custom state dictionary in {self.checkpoint_path} is incompatible with architecture: {exc}"
                ) from exc

            param_count = sum(p.numel() for p in self.model.parameters())
            self.provenance = ModelProvenance(
                model_id="custom/aasist-l",
                architecture="aasist_l",
                provenance_status="custom_unverified",
                checkpoint_sha256=actual_sha,
                checkpoint_verified=False,
                expected_checkpoint_sha256=None,
                upstream_repository=None,
                upstream_commit=None,
                parameter_count=param_count,
                is_mock=False,
                checkpoint_size_bytes=actual_size,
                source_license_spdx=None,
                checkpoint_rights_note=None,
            )
        else:
            raise UnsupportedModelIdentityError(
                f"Unsupported model identity: {model_id!r}. "
                "Supported identities: 'clovaai/aasist-l', 'custom/aasist-l', 'mock/aasist-l'"
            )

        self.model.eval()
        try:
            self.model.to(self.device)
        except Exception as exc:
            if not self.fallback_to_cpu or self.device.type == "cuda":
                raise
            logger.warning(
                "Failed to place AASIST-L on device %s: %s. Falling back to CPU.", self.device, exc
            )
            self.device = torch.device("cpu")
            self.model.to(self.device)

    @property
    def model_id(self) -> str:
        """Return canonical model identifier from central provenance."""
        return self.provenance.model_id

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
        # Detach and place each item on CPU, padding or cropping to target length if needed
        processed_items: list[torch.Tensor] = []
        for i in range(batch_size):
            item = batch[i].detach().to("cpu", dtype=torch.float32)
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
                if not self.fallback_to_cpu or self.device.type == "cuda":
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
                    model_id=self.provenance.model_id,
                    operating_threshold=self.threshold,
                    operating_threshold_source="unvalidated_heuristic",
                    score_is_calibrated=False,
                    metadata={
                        "model_id": self.provenance.model_id,
                        "provenance_status": self.provenance.provenance_status,
                        "checkpoint_verified": self.provenance.checkpoint_verified,
                        "checkpoint_sha256": self.provenance.checkpoint_sha256,
                        "parameter_count": self.provenance.parameter_count,
                        "upstream_repo": self.provenance.upstream_repository,
                        "upstream_commit": self.provenance.upstream_commit,
                        "label_mapping": {"0": "spoof", "1": "bonafide"},
                        "fallback_used": fallback_used,
                        "model_provenance": self.provenance.model_dump(),
                    },
                )
            )

        return results
