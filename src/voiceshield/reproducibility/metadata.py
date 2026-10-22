"""Collection of provenance and run metadata for VoiceShield experiments and runs."""

from __future__ import annotations

import hashlib
import logging
import platform
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import torch
from pydantic import BaseModel, Field

from voiceshield.config.schema import VoiceShieldConfig
from voiceshield.core.device import resolve_device_status

logger = logging.getLogger(__name__)


def _compute_file_sha256(path: Path) -> str | None:
    """Compute SHA-256 hash of a file if it exists."""
    if not path.is_file():
        return None
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _get_git_commit() -> str | None:
    """Retrieve current Git commit hash or None if repository has no commits yet."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5.0,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception as exc:
        logger.debug("Failed to resolve git commit: %s", exc)
    return None


def _is_git_dirty() -> bool:
    """Check if Git working tree has uncommitted modifications."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5.0,
        )
        if result.returncode == 0:
            return bool(result.stdout.strip())
    except Exception as exc:
        logger.debug("Failed to check git dirty status: %s", exc)
    return False


class RunMetadata(BaseModel):
    """Provenance and runtime execution environment metadata."""

    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    git_commit: str | None = Field(default=None, description="Current Git commit SHA if committed.")
    git_dirty: bool = Field(
        default=False, description="Whether working tree has uncommitted changes."
    )
    config_hash: str = Field(description="Deterministic SHA-256 hash of the run configuration.")
    manifest_hash: str | None = Field(
        default=None, description="SHA-256 hash of dataset manifest file."
    )
    model_checkpoint_hash: str | None = Field(
        default=None,
        description="SHA-256 hash of loaded model weights checkpoint.",
    )
    platform: str = Field(default_factory=platform.platform)
    python_version: str = Field(default_factory=lambda: sys.version.split()[0])
    torch_version: str = Field(default_factory=lambda: str(torch.__version__))
    selected_device: str = Field(description="Target device name (e.g. 'mps' or 'cpu').")
    device_is_mps: bool = Field(
        description="Whether computation is executing on Apple Silicon MPS."
    )
    device_explanation: str = Field(description="Device diagnostic explanation.")


def collect_run_metadata(
    config: VoiceShieldConfig,
    manifest_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> RunMetadata:
    """Collect runtime environment metadata and provenance hashes.

    Args:
        config: Loaded VoiceShield configuration.
        manifest_path: Optional path to dataset manifest file.
        checkpoint_path: Optional path to model weights checkpoint.

    Returns:
        Populated RunMetadata model.
    """
    dev_status = resolve_device_status(
        preferred=config.model.device_preferred,
        fallback_to_cpu=config.model.fallback_to_cpu,
    )

    manifest_hash: str | None = None
    if manifest_path is not None:
        manifest_hash = _compute_file_sha256(Path(manifest_path))

    ckpt_target = checkpoint_path or config.model.checkpoint_path
    ckpt_hash: str | None = None
    if ckpt_target is not None:
        ckpt_hash = _compute_file_sha256(Path(ckpt_target))

    return RunMetadata(
        git_commit=_get_git_commit(),
        git_dirty=_is_git_dirty(),
        config_hash=config.compute_hash(),
        manifest_hash=manifest_hash,
        model_checkpoint_hash=ckpt_hash,
        selected_device=dev_status.device_name,
        device_is_mps=dev_status.is_mps,
        device_explanation=dev_status.explanation,
    )
