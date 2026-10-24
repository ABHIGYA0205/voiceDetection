"""Central typed model provenance contract for VoiceShield.

Defines the single source of truth for runtime model identification,
verification status, checkpoint fingerprint, upstream provenance, and mock status.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from voiceshield.models.registry import OFFICIAL_AASIST_L

ModelId = Literal["clovaai/aasist-l", "custom/aasist-l", "mock/aasist-l"]
ArchitectureId = Literal["aasist_l"]
ProvenanceStatus = Literal["official_verified", "custom_unverified", "mock"]

_SHA256_REGEX = re.compile(r"^[0-9a-f]{64}$")


def _is_valid_sha256(val: str | None) -> bool:
    """Validate 64 lowercase hexadecimal character syntax."""
    if not isinstance(val, str):
        return False
    return bool(_SHA256_REGEX.fullmatch(val))


class ModelProvenance(BaseModel):
    """Immutable runtime model identity and integrity provenance specification."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    model_id: ModelId = Field(
        description="Model identifier ('clovaai/aasist-l', 'custom/aasist-l', or 'mock/aasist-l')."
    )
    architecture: ArchitectureId = Field(
        default="aasist_l",
        description="Architecture identifier (strictly 'aasist_l').",
    )
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

    @model_validator(mode="after")
    def validate_truth_table_invariants(self) -> ModelProvenance:
        """Enforce strict truth-table integrity invariants across all provenance fields."""
        if self.provenance_status == "official_verified":
            if self.model_id != "clovaai/aasist-l":
                raise ValueError(
                    f"official_verified provenance status requires model_id='clovaai/aasist-l', got {self.model_id!r}"
                )
            if self.architecture != "aasist_l":
                raise ValueError(
                    f"official_verified provenance requires architecture='aasist_l', got {self.architecture!r}"
                )
            if self.checkpoint_verified is not True:
                raise ValueError("official_verified provenance requires checkpoint_verified=True")
            if self.is_mock is not False:
                raise ValueError("official_verified provenance requires is_mock=False")
            if not _is_valid_sha256(self.checkpoint_sha256):
                raise ValueError(
                    f"official_verified provenance requires a valid 64-hex lowercase checkpoint_sha256, got {self.checkpoint_sha256!r}"
                )
            if not _is_valid_sha256(self.expected_checkpoint_sha256):
                raise ValueError(
                    f"official_verified provenance requires a valid 64-hex lowercase expected_checkpoint_sha256, got {self.expected_checkpoint_sha256!r}"
                )
            if self.checkpoint_sha256 != self.expected_checkpoint_sha256:
                raise ValueError(
                    "official_verified provenance requires checkpoint_sha256 to equal expected_checkpoint_sha256"
                )
            if self.checkpoint_sha256 != OFFICIAL_AASIST_L.checkpoint_sha256:
                raise ValueError(
                    f"official_verified checkpoint_sha256 must equal official registry SHA ({OFFICIAL_AASIST_L.checkpoint_sha256}), got {self.checkpoint_sha256}"
                )
            if not self.upstream_repository:
                raise ValueError(
                    "official_verified provenance requires non-empty upstream_repository"
                )
            if self.upstream_commit != OFFICIAL_AASIST_L.upstream_commit:
                raise ValueError(
                    f"official_verified upstream_commit must equal official registry commit ({OFFICIAL_AASIST_L.upstream_commit}), got {self.upstream_commit!r}"
                )
            if self.parameter_count != OFFICIAL_AASIST_L.parameter_count:
                raise ValueError(
                    f"official_verified parameter_count must equal {OFFICIAL_AASIST_L.parameter_count}, got {self.parameter_count}"
                )
            if self.checkpoint_size_bytes != OFFICIAL_AASIST_L.checkpoint_size_bytes:
                raise ValueError(
                    f"official_verified checkpoint_size_bytes must equal {OFFICIAL_AASIST_L.checkpoint_size_bytes}, got {self.checkpoint_size_bytes}"
                )
            if self.source_license_spdx != OFFICIAL_AASIST_L.source_license_spdx:
                raise ValueError(
                    f"official_verified source_license_spdx must equal {OFFICIAL_AASIST_L.source_license_spdx!r}, got {self.source_license_spdx!r}"
                )
            if self.checkpoint_rights_note != OFFICIAL_AASIST_L.checkpoint_rights_note:
                raise ValueError(
                    "official_verified checkpoint_rights_note must equal official registry note"
                )

        elif self.provenance_status == "custom_unverified":
            if self.model_id != "custom/aasist-l":
                raise ValueError(
                    f"custom_unverified provenance status requires model_id='custom/aasist-l', got {self.model_id!r}"
                )
            if self.architecture != "aasist_l":
                raise ValueError(
                    f"custom_unverified provenance requires architecture='aasist_l', got {self.architecture!r}"
                )
            if self.checkpoint_verified is not False:
                raise ValueError("custom_unverified provenance requires checkpoint_verified=False")
            if self.is_mock is not False:
                raise ValueError("custom_unverified provenance requires is_mock=False")
            if not _is_valid_sha256(self.checkpoint_sha256):
                raise ValueError(
                    f"custom_unverified provenance requires a valid 64-hex lowercase checkpoint_sha256 fingerprint, got {self.checkpoint_sha256!r}"
                )
            if self.expected_checkpoint_sha256 is not None:
                raise ValueError(
                    "custom_unverified provenance must not specify expected_checkpoint_sha256"
                )
            if self.upstream_repository is not None or self.upstream_commit is not None:
                raise ValueError(
                    "custom_unverified provenance must not attach official upstream repository or commit"
                )
            if self.source_license_spdx is not None:
                raise ValueError("custom_unverified provenance must have source_license_spdx=None")
            if self.checkpoint_rights_note is not None:
                raise ValueError(
                    "custom_unverified provenance must have checkpoint_rights_note=None"
                )
            if self.parameter_count <= 0:
                raise ValueError(
                    f"custom_unverified parameter_count must be positive, got {self.parameter_count}"
                )

        elif self.provenance_status == "mock":
            if self.model_id != "mock/aasist-l":
                raise ValueError(
                    f"mock provenance status requires model_id='mock/aasist-l', got {self.model_id!r}"
                )
            if self.architecture != "aasist_l":
                raise ValueError(
                    f"mock provenance requires architecture='aasist_l', got {self.architecture!r}"
                )
            if self.checkpoint_verified is not False:
                raise ValueError("mock provenance requires checkpoint_verified=False")
            if self.is_mock is not True:
                raise ValueError("mock provenance requires is_mock=True")
            if self.checkpoint_sha256 is not None:
                raise ValueError("mock provenance must have checkpoint_sha256=None")
            if self.expected_checkpoint_sha256 is not None:
                raise ValueError("mock provenance must have expected_checkpoint_sha256=None")
            if self.checkpoint_size_bytes is not None:
                raise ValueError("mock provenance must have checkpoint_size_bytes=None")
            if self.upstream_repository is not None or self.upstream_commit is not None:
                raise ValueError(
                    "mock provenance must not attach upstream repository or commit fields"
                )
            if self.source_license_spdx is not None:
                raise ValueError("mock provenance must have source_license_spdx=None")
            if self.checkpoint_rights_note is not None:
                raise ValueError("mock provenance must have checkpoint_rights_note=None")
            if self.parameter_count <= 0:
                raise ValueError(
                    f"mock parameter_count must be positive, got {self.parameter_count}"
                )

        return self
