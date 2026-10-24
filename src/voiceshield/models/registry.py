"""Immutable central model registry for VoiceShield official models.

Provides authoritative identity, pinned upstream metadata, checkpoint checksums,
and architecture parameters for verified official models.
"""

from __future__ import annotations

import re
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelRegistryError(Exception):
    """Base exception for model registry errors."""


class UnsupportedModelIdentityError(ModelRegistryError):
    """Raised when an unknown or unsupported model identity is requested."""


class CheckpointError(ValueError):
    """Base exception for model checkpoint errors."""


class CheckpointNotFoundError(CheckpointError, FileNotFoundError):
    """Raised when a required checkpoint file does not exist."""


class CheckpointIntegrityError(CheckpointError):
    """Raised when checkpoint SHA-256, size, or parameter count validation fails."""


class CheckpointLoadError(CheckpointError):
    """Raised when checkpoint deserialization or state-dict loading fails."""


class ModelRegistryEntry(BaseModel):
    """Immutable metadata and integrity specification for an official model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str = Field(description="Canonical model identifier (e.g. 'clovaai/aasist-l').")
    architecture: str = Field(description="Supported architecture identifier ('aasist_l').")
    upstream_repository: str = Field(description="Upstream source code repository URL.")
    upstream_commit: str = Field(description="Pinned upstream git commit SHA.")
    checkpoint_url: str = Field(description="Direct pinned URL to download official checkpoint.")
    checkpoint_filename: str = Field(description="Canonical checkpoint file name.")
    checkpoint_sha256: str = Field(
        description="Expected SHA-256 hex digest (64 lowercase hex characters)."
    )
    checkpoint_size_bytes: int = Field(
        description="Exact size of the official checkpoint in bytes."
    )
    parameter_count: int = Field(description="Exact parameter count of the architecture.")
    source_license_spdx: str = Field(
        description="SPDX license identifier of the upstream repository."
    )
    checkpoint_rights_note: str = Field(
        description="Neutral notice regarding checkpoint redistribution rights."
    )

    @field_validator("checkpoint_sha256")
    @classmethod
    def validate_sha256(cls, v: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", v):
            raise ValueError(
                f"checkpoint_sha256 must be exactly 64 lowercase hexadecimal characters, got {v!r}"
            )
        return v


# Authoritative official AASIST-L registry entry
OFFICIAL_AASIST_L = ModelRegistryEntry(
    model_id="clovaai/aasist-l",
    architecture="aasist_l",
    upstream_repository="https://github.com/clovaai/aasist",
    upstream_commit="a04c9863f63d44471dde8a6abcb3b082b07cd1d1",
    checkpoint_url="https://raw.githubusercontent.com/clovaai/aasist/a04c9863f63d44471dde8a6abcb3b082b07cd1d1/models/weights/AASIST-L.pth",
    checkpoint_filename="AASIST-L.pth",
    checkpoint_sha256="814331d088032bb4c3fa61cc014789eadeed464209dd094ab3a2dd6ffbdce27a",
    checkpoint_size_bytes=426428,
    parameter_count=85306,
    source_license_spdx="MIT",
    checkpoint_rights_note=(
        "The upstream source repository is MIT licensed. Checkpoint licensing and redistribution "
        "rights are not separately documented or independently verified by VoiceShield; obtain "
        "checkpoint bytes from the pinned upstream source."
    ),
)

# Immutable mapping proxy prevents runtime mutation of registered models.
# Constructed directly from a dict literal with no module-level mutable dictionary reference.
OFFICIAL_MODEL_REGISTRY: MappingProxyType[str, ModelRegistryEntry] = MappingProxyType(
    {
        OFFICIAL_AASIST_L.model_id: OFFICIAL_AASIST_L,
    }
)


def get_official_registry_entry(model_id: str) -> ModelRegistryEntry:
    """Retrieve the immutable registry entry for an official model ID.

    Args:
        model_id: The model identifier to look up.

    Returns:
        The matching ModelRegistryEntry.

    Raises:
        UnsupportedModelIdentityError: If model_id is not registered in the official registry.
    """
    if model_id not in OFFICIAL_MODEL_REGISTRY:
        raise UnsupportedModelIdentityError(
            f"Unknown or unsupported official model ID: {model_id!r}. "
            f"Registered official model IDs: {list(OFFICIAL_MODEL_REGISTRY.keys())}"
        )
    return OFFICIAL_MODEL_REGISTRY[model_id]


def get_official_model_ids() -> tuple[str, ...]:
    """Return immutable tuple of all registered official model IDs."""
    return tuple(OFFICIAL_MODEL_REGISTRY.keys())
