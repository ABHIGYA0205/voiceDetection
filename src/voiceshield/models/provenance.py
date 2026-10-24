"""Central typed model provenance contract for VoiceShield.

Defines the single source of truth for runtime model identification,
verification status, checkpoint fingerprint, upstream provenance, and mock status.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ProvenanceStatus = Literal["official_verified", "custom_unverified", "mock"]


class ModelProvenance(BaseModel):
    """Immutable runtime model identity and integrity provenance specification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str = Field(
        description="Model identifier (e.g. 'clovaai/aasist-l', 'custom/aasist-l', 'mock/aasist-l')."
    )
    architecture: str = Field(description="Architecture identifier ('aasist_l').")
    provenance_status: ProvenanceStatus = Field(
        description="Verification provenance status: 'official_verified', 'custom_unverified', or 'mock'."
    )
    checkpoint_sha256: str | None = Field(
        default=None,
        description="Actual SHA-256 fingerprint of the loaded weights file. Null for mock mode.",
    )
    checkpoint_verified: bool = Field(
        default=False,
        description="True strictly when official checkpoint was verified against the central registry.",
    )
    expected_checkpoint_sha256: str | None = Field(
        default=None,
        description="Expected registry SHA-256 for official mode. Null for custom or mock mode.",
    )
    upstream_repository: str | None = Field(
        default=None, description="Upstream source repository URL if applicable."
    )
    upstream_commit: str | None = Field(
        default=None, description="Pinned upstream git commit SHA if applicable."
    )
    parameter_count: int = Field(description="Exact parameter count of the architecture.")
    is_mock: bool = Field(
        default=False, description="True if running with synthetic or mock architecture weights."
    )
    checkpoint_size_bytes: int | None = Field(
        default=None, description="Actual checkpoint file size in bytes if available."
    )
    source_license_spdx: str | None = Field(
        default=None, description="Upstream source repository license identifier."
    )
    checkpoint_rights_note: str | None = Field(
        default=None, description="Statement regarding checkpoint redistribution rights."
    )
