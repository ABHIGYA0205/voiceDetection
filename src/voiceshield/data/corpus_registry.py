"""Rights-aware corpus registry and fail-closed governance gate."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    TypeAdapter,
    field_validator,
    model_validator,
)

REGISTRY_SCHEMA_VERSION = "1.0.0"

CORPUS_ID_REGEX = re.compile(r"^[a-z0-9_-]+$")
SHA256_REGEX = re.compile(r"^[a-f0-9]{64}$")

_HTTP_URL_ADAPTER: TypeAdapter[HttpUrl] = TypeAdapter(HttpUrl)


class CorpusRegistryError(Exception):
    """Base exception for corpus registry validation and governance failures."""


class CorpusValidationError(CorpusRegistryError):
    """Raised when registry structure or invariant validation fails."""


class CorpusAuthorizationError(CorpusRegistryError):
    """Raised when a corpus fails use-gate authorization."""

    def __init__(self, message: str, denial_reasons: list[str] | None = None) -> None:
        super().__init__(message)
        self.denial_reasons: list[str] = denial_reasons or []


class DataState(StrEnum):
    """Lifecycle acquisition and verification state of a dataset snapshot."""

    PLANNED = "planned"
    ACQUIRED = "acquired"
    VERIFIED = "verified"


class ContentType(StrEnum):
    """Primary acoustic content classification of the corpus."""

    BONAFIDE = "bonafide"
    SPOOF = "spoof"
    MIXED = "mixed"


class RightsDecision(StrEnum):
    """Project-specific legal/governance review decision."""

    APPROVED = "approved"
    RESTRICTED = "restricted"
    PENDING = "pending"
    BLOCKED = "blocked"


class AllowedPurpose(StrEnum):
    """Specific authorized purposes permitted by governance review."""

    RESEARCH_EVALUATION = "research_evaluation"
    RESEARCH_TRAINING = "research_training"
    CALIBRATION = "calibration"
    PRODUCT_EVALUATION = "product_evaluation"
    PRODUCT_TRAINING = "product_training"
    RAW_REDISTRIBUTION = "raw_redistribution"


class RedistributionStatus(StrEnum):
    """Policy on redistribution of raw audio files."""

    ALLOWED = "allowed"
    PROHIBITED = "prohibited"
    UNKNOWN = "unknown"


def _validate_corpus_id(v: str) -> str:
    if not isinstance(v, str) or not v.strip():
        raise ValueError("corpus_id must be a non-empty string.")
    v = v.strip()
    if not CORPUS_ID_REGEX.match(v):
        raise ValueError(
            f"corpus_id must be a stable lowercase identifier containing only [a-z0-9_-] (got '{v}')."
        )
    return v


def _validate_https_url(v: str | None, field_name: str) -> str | None:
    if v is None:
        return None
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"{field_name} must be a non-empty string when provided.")
    v = v.strip()

    # Reject whitespace anywhere in URL
    if any(c.isspace() for c in v):
        raise ValueError(f"{field_name} must not contain whitespace (got '{v}').")

    # 1. Structural check via urllib.parse.urlsplit
    try:
        split_res = urlsplit(v)
    except Exception as exc:
        raise ValueError(f"{field_name} is not a valid URL: {exc}") from exc

    if split_res.scheme != "https":
        raise ValueError(f"{field_name} must use the 'https' scheme (got '{split_res.scheme}').")

    if not split_res.netloc:
        raise ValueError(f"{field_name} must have a valid non-empty authority/netloc (got '{v}').")

    if "@" in split_res.netloc or split_res.username or split_res.password:
        raise ValueError(f"{field_name} must not contain embedded username or password.")

    if not split_res.hostname or not split_res.hostname.strip():
        raise ValueError(f"{field_name} must contain a valid non-empty hostname.")

    # 2. Semantic parsing via Pydantic HttpUrl
    try:
        parsed = _HTTP_URL_ADAPTER.validate_python(v)
    except Exception as exc:
        raise ValueError(f"{field_name} is not a valid HTTPS URL: {exc}") from exc

    if parsed.scheme != "https":
        raise ValueError(f"{field_name} must use the 'https' scheme.")

    if parsed.username or parsed.password:
        raise ValueError(f"{field_name} must not contain embedded user credentials.")

    if not parsed.host:
        raise ValueError(f"{field_name} must contain a valid host.")

    return v


def _validate_sha256(v: str | None, field_name: str) -> str | None:
    if v is None:
        return None
    if not isinstance(v, str):
        raise ValueError(f"{field_name} must be a string.")
    if not SHA256_REGEX.match(v):
        raise ValueError(
            f"{field_name} must be a lowercase 64-character hexadecimal SHA-256 hash (got '{v}')."
        )
    return v


class CorpusEntry(BaseModel):
    """Structured, immutable metadata for a single dataset snapshot in VoiceShield."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    corpus_id: str = Field(description="Stable lowercase identifier for the corpus.")
    display_name: str = Field(description="Human-readable display name of the dataset.")
    version_ref: str | None = Field(
        default=None,
        description="Immutable version/revision when known; nullable only while planned.",
    )
    data_state: DataState = Field(
        description="Lifecycle acquisition and verification state (planned, acquired, verified)."
    )
    content_type: ContentType = Field(
        description="Primary audio classification (bonafide, spoof, mixed)."
    )
    source_url: str = Field(description="HTTPS origin or download repository URL.")
    documentation_url: str = Field(description="HTTPS documentation, paper, or dataset card URL.")
    declared_license_spdx: str | None = Field(
        default=None,
        description="Declared SPDX license identifier, or null if genuinely unknown.",
    )
    license_url: str | None = Field(
        default=None,
        description="HTTPS URL to license terms or legal deed, or null.",
    )
    rights_decision: RightsDecision = Field(
        description="Project governance rights decision (approved, restricted, pending, blocked)."
    )
    allowed_purposes: tuple[AllowedPurpose, ...] = Field(
        default_factory=tuple,
        description="Explicit immutable tuple of permitted usage purposes authorized by rights review.",
    )
    rights_reviewed_at: date | None = Field(
        default=None,
        description="ISO date when project rights review was recorded (YYYY-MM-DD).",
    )
    rights_reviewed_by: str | None = Field(
        default=None,
        description="Non-empty reviewer name or identity who conducted the rights review.",
    )
    attribution_required: bool | None = Field(
        default=None,
        description="Whether attribution must be preserved in documentation/releases.",
    )
    raw_redistribution_status: RedistributionStatus = Field(
        default=RedistributionStatus.UNKNOWN,
        description="Redistribution permission status for raw corpus files.",
    )
    archive_sha256: str | None = Field(
        default=None,
        description="Lowercase 64-character SHA-256 hash of the exact downloaded archive.",
    )
    manifest_sha256: str | None = Field(
        default=None,
        description="Lowercase 64-character SHA-256 hash of the canonical ManifestRecord JSONL.",
    )
    sample_count: int | None = Field(
        default=None,
        description="Exact count of verified audio samples in the converted manifest.",
    )
    notes: str = Field(
        description="Non-empty conservative provenance, rights, or segregation note."
    )

    @field_validator("corpus_id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        return _validate_corpus_id(v)

    @field_validator("display_name")
    @classmethod
    def validate_display_name(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("display_name must be a non-empty string.")
        return v.strip()

    @field_validator("version_ref")
    @classmethod
    def validate_version(cls, v: str | None) -> str | None:
        if v is not None:
            if not isinstance(v, str) or not v.strip():
                raise ValueError("version_ref must be a non-empty string when provided.")
            return v.strip()
        return None

    @field_validator("source_url")
    @classmethod
    def validate_source_url(cls, v: str) -> str:
        res = _validate_https_url(v, "source_url")
        assert res is not None
        return res

    @field_validator("documentation_url")
    @classmethod
    def validate_documentation_url(cls, v: str) -> str:
        res = _validate_https_url(v, "documentation_url")
        assert res is not None
        return res

    @field_validator("license_url")
    @classmethod
    def validate_license_url(cls, v: str | None) -> str | None:
        return _validate_https_url(v, "license_url")

    @field_validator("archive_sha256")
    @classmethod
    def validate_archive_hash(cls, v: str | None) -> str | None:
        return _validate_sha256(v, "archive_sha256")

    @field_validator("manifest_sha256")
    @classmethod
    def validate_manifest_hash(cls, v: str | None) -> str | None:
        return _validate_sha256(v, "manifest_sha256")

    @field_validator("sample_count", mode="before")
    @classmethod
    def validate_sample_count_strict(cls, v: Any) -> Any:
        if v is None:
            return None
        # Disallow boolean values (since bool is a subclass of int in Python)
        if isinstance(v, bool):
            raise ValueError("sample_count must be a positive integer, not a boolean.")
        if not isinstance(v, int):
            raise ValueError(f"sample_count must be an integer, got {type(v).__name__}.")
        if v <= 0:
            raise ValueError(f"sample_count must be a positive integer (>0), got {v}.")
        return v

    @field_validator("attribution_required", mode="before")
    @classmethod
    def validate_attribution_required_strict(cls, v: Any) -> Any:
        if v is None:
            return None
        # Disallow arbitrary integers (0, 1) or string coercion ("true", "false")
        if not isinstance(v, bool):
            raise ValueError(
                f"attribution_required must be a boolean (True/False) or null, got {type(v).__name__} ({v!r})."
            )
        return v

    @field_validator("allowed_purposes", mode="after")
    @classmethod
    def validate_unique_purposes(cls, v: tuple[AllowedPurpose, ...]) -> tuple[AllowedPurpose, ...]:
        if len(v) != len(set(v)):
            raise ValueError(
                f"allowed_purposes contains duplicate values: {[p.value for p in v]}. "
                "Each allowed purpose must be unique."
            )
        return v

    @field_validator("rights_reviewed_by")
    @classmethod
    def validate_reviewer(cls, v: str | None) -> str | None:
        if v is not None:
            if not isinstance(v, str) or not v.strip():
                raise ValueError("rights_reviewed_by must be a non-empty string when provided.")
            return v.strip()
        return None

    @field_validator("declared_license_spdx")
    @classmethod
    def validate_license_spdx(cls, v: str | None) -> str | None:
        if v is not None:
            if not isinstance(v, str) or not v.strip():
                raise ValueError("declared_license_spdx must be a non-empty string when provided.")
            return v.strip()
        return None

    @field_validator("notes")
    @classmethod
    def validate_notes(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("notes must be a non-empty string.")
        return v.strip()

    @model_validator(mode="after")
    def validate_invariants(self) -> CorpusEntry:
        # Invariant: Rights review before acquisition
        # data_state=acquired or verified requires rights_decision=approved or restricted
        if self.data_state in (DataState.ACQUIRED, DataState.VERIFIED):
            if self.rights_decision not in (RightsDecision.APPROVED, RightsDecision.RESTRICTED):
                raise ValueError(
                    f"Corpus '{self.corpus_id}' has data_state='{self.data_state.value}' "
                    f"but rights_decision='{self.rights_decision.value}'. "
                    "Acquired or verified entries require a prior rights review decision of 'approved' or 'restricted'. "
                    "Pending or blocked entries cannot be acquired or verified."
                )

        # Invariant: Pending or blocked entries must have an empty allowed_purposes list
        if self.rights_decision in (RightsDecision.PENDING, RightsDecision.BLOCKED):
            if len(self.allowed_purposes) > 0:
                raise ValueError(
                    f"Corpus '{self.corpus_id}' has rights_decision='{self.rights_decision.value}' "
                    f"but non-empty allowed_purposes: {[p.value for p in self.allowed_purposes]}. "
                    "Pending or blocked entries must have an empty allowed_purposes list."
                )

        # Invariant: Planned entries must not claim archive or manifest hashes or sample counts
        if self.data_state == DataState.PLANNED:
            if self.archive_sha256 is not None:
                raise ValueError(
                    f"Corpus '{self.corpus_id}' has data_state='planned' but specifies archive_sha256. "
                    "Planned entries must not claim archive hashes."
                )
            if self.manifest_sha256 is not None:
                raise ValueError(
                    f"Corpus '{self.corpus_id}' has data_state='planned' but specifies manifest_sha256. "
                    "Planned entries must not claim manifest hashes."
                )
            if self.sample_count is not None:
                raise ValueError(
                    f"Corpus '{self.corpus_id}' has data_state='planned' but specifies sample_count. "
                    "Planned entries must not claim sample counts."
                )

        # Invariant: Acquired entries require version_ref and archive_sha256
        if self.data_state == DataState.ACQUIRED:
            if not self.version_ref:
                raise ValueError(
                    f"Corpus '{self.corpus_id}' has data_state='acquired' but missing version_ref. "
                    "Acquired entries require an immutable version_ref."
                )
            if not self.archive_sha256:
                raise ValueError(
                    f"Corpus '{self.corpus_id}' has data_state='acquired' but missing archive_sha256. "
                    "Acquired entries require archive_sha256."
                )

        # Invariant: Verified entries require version_ref, archive_sha256, manifest_sha256, and positive sample_count
        if self.data_state == DataState.VERIFIED:
            if not self.version_ref:
                raise ValueError(
                    f"Corpus '{self.corpus_id}' has data_state='verified' but missing version_ref. "
                    "Verified entries require an immutable version_ref."
                )
            if not self.archive_sha256:
                raise ValueError(
                    f"Corpus '{self.corpus_id}' has data_state='verified' but missing archive_sha256. "
                    "Verified entries require archive_sha256."
                )
            if not self.manifest_sha256:
                raise ValueError(
                    f"Corpus '{self.corpus_id}' has data_state='verified' but missing manifest_sha256. "
                    "Verified entries require manifest_sha256."
                )
            if self.sample_count is None or self.sample_count <= 0:
                raise ValueError(
                    f"Corpus '{self.corpus_id}' has data_state='verified' but invalid sample_count: {self.sample_count}. "
                    "Verified entries require a positive sample_count."
                )

        # Invariant: Non-planned entries require version_ref
        if self.data_state != DataState.PLANNED and not self.version_ref:
            raise ValueError(
                f"Corpus '{self.corpus_id}' in state '{self.data_state.value}' requires a non-null version_ref."
            )

        # Invariant: Approved or restricted entries require a rights reviewer, review date, declared licence, and licence URL
        if self.rights_decision in (RightsDecision.APPROVED, RightsDecision.RESTRICTED):
            if not self.rights_reviewed_by:
                raise ValueError(
                    f"Corpus '{self.corpus_id}' has rights_decision='{self.rights_decision.value}' "
                    "but missing rights_reviewed_by. Approved or restricted entries require rights_reviewed_by."
                )
            if self.rights_reviewed_at is None:
                raise ValueError(
                    f"Corpus '{self.corpus_id}' has rights_decision='{self.rights_decision.value}' "
                    "but missing rights_reviewed_at. Approved or restricted entries require rights_reviewed_at."
                )
            if not self.declared_license_spdx:
                raise ValueError(
                    f"Corpus '{self.corpus_id}' has rights_decision='{self.rights_decision.value}' "
                    "but missing declared_license_spdx. Approved or restricted entries require declared_license_spdx."
                )
            if not self.license_url:
                raise ValueError(
                    f"Corpus '{self.corpus_id}' has rights_decision='{self.rights_decision.value}' "
                    "but missing license_url. Approved or restricted entries require license_url."
                )

        # Invariant: raw_redistribution may be an allowed purpose only when raw_redistribution_status is allowed
        if AllowedPurpose.RAW_REDISTRIBUTION in self.allowed_purposes:
            if self.raw_redistribution_status != RedistributionStatus.ALLOWED:
                raise ValueError(
                    f"Corpus '{self.corpus_id}' includes raw_redistribution in allowed_purposes, "
                    f"but raw_redistribution_status is '{self.raw_redistribution_status.value}'. "
                    "raw_redistribution may be an allowed purpose only when raw_redistribution_status is allowed."
                )

        return self


class CorpusRegistry(BaseModel):
    """Top-level schema for the VoiceShield rights-aware corpus registry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"] = "1.0.0"
    corpora: tuple[CorpusEntry, ...] = Field(
        default_factory=tuple,
        description="Immutable tuple of registered corpus entries.",
    )

    @model_validator(mode="after")
    def validate_unique_identities(self) -> CorpusRegistry:
        seen_identities: set[tuple[str, str | None]] = set()
        for entry in self.corpora:
            identity = (entry.corpus_id, entry.version_ref)
            if identity in seen_identities:
                raise ValueError(
                    f"Duplicate corpus identity detected: corpus_id='{entry.corpus_id}', "
                    f"version_ref='{entry.version_ref}'. Each (corpus_id, version_ref) identity must be unique."
                )
            seen_identities.add(identity)
        return self


class CorpusIdentity(BaseModel):
    """Exact corpus identity used for authorization queries."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    corpus_id: str
    version_ref: str | None = None

    @field_validator("corpus_id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        return _validate_corpus_id(v)

    @field_validator("version_ref")
    @classmethod
    def validate_version_ref(cls, v: str | None) -> str | None:
        if v is not None:
            if not isinstance(v, str) or not v.strip():
                raise ValueError("version_ref must be a non-empty string when provided.")
            return v.strip()
        return None

    @classmethod
    def parse(cls, identifier: str | CorpusIdentity) -> CorpusIdentity:
        if isinstance(identifier, cls):
            return identifier
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(
                f"Invalid corpus identifier: '{identifier}'. Must be non-empty string."
            )
        s = identifier.strip()
        if "@" in s:
            parts = s.split("@", 1)
            cid = parts[0].strip()
            vref = parts[1].strip()
            if not cid:
                raise ValueError(f"Corpus ID cannot be empty in '{s}'.")
            if not vref:
                raise ValueError(f"Version reference cannot be empty in '{s}'.")
            return cls(corpus_id=cid, version_ref=vref)
        return cls(corpus_id=s, version_ref=None)


class AuthorizedCorpus(BaseModel):
    """An authorized, verified corpus identity with immutable integrity hashes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    corpus_id: str
    version_ref: str
    archive_sha256: str
    manifest_sha256: str
    sample_count: int
    rights_decision: RightsDecision
    content_type: ContentType


class CorpusAuthorizationResult(BaseModel):
    """Deterministic result of a successful corpus use-gate authorization check."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    purpose: AllowedPurpose
    authorized_corpora: tuple[AuthorizedCorpus, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "purpose": self.purpose.value,
            "authorized_corpora": [
                {
                    "corpus_id": c.corpus_id,
                    "version_ref": c.version_ref,
                    "archive_sha256": c.archive_sha256,
                    "manifest_sha256": c.manifest_sha256,
                    "sample_count": c.sample_count,
                    "rights_decision": c.rights_decision.value,
                    "content_type": c.content_type.value,
                }
                for c in self.authorized_corpora
            ],
        }


class _UniqueKeyLoader(yaml.SafeLoader):
    """YAML SafeLoader subclass that rejects duplicate mapping keys."""


def _construct_mapping_unique(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        try:
            key = loader.construct_object(key_node, deep=deep)
            if key in mapping:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate YAML key '{key}'",
                    key_node.start_mark,
                )
            mapping[key] = loader.construct_object(value_node, deep=deep)
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"unhashable mapping key: {exc}",
                key_node.start_mark,
            ) from exc
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping_unique,
)


def compute_registry_bytes_hash(data: bytes) -> str:
    """Calculate the lowercase SHA-256 hex digest of exact registry file bytes."""
    return hashlib.sha256(data).hexdigest().lower()


def compute_registry_file_hash(path: str | Path) -> str:
    """Calculate the lowercase SHA-256 hex digest of the exact registry file on disk."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Corpus registry file not found: {p}")
    return compute_registry_bytes_hash(p.read_bytes())


# Canonical alias for registry hash
compute_registry_hash = compute_registry_file_hash


def load_corpus_registry(source: str | Path | bytes) -> CorpusRegistry:
    """Load and strictly validate a CorpusRegistry from YAML file or raw bytes.

    Enforces unique YAML keys, non-empty content, extra-fields-forbidden,
    and all fail-closed governance invariants. Consistently raises CorpusValidationError.
    """
    raw_bytes: bytes
    try:
        if isinstance(source, (str, Path)):
            p = Path(source)
            if not p.is_file():
                raise CorpusValidationError(f"Corpus registry file not found: {p}")
            raw_bytes = p.read_bytes()
        elif isinstance(source, bytes):
            raw_bytes = source
        else:
            raise CorpusValidationError(
                f"Unsupported source type for registry: {type(source).__name__}"
            )
    except CorpusValidationError:
        raise
    except Exception as exc:
        raise CorpusValidationError(f"Failed to read corpus registry source: {exc}") from exc

    if not raw_bytes or not raw_bytes.strip():
        raise CorpusValidationError("Corpus registry source is empty or contains only whitespace.")

    try:
        parsed: Any = yaml.load(raw_bytes, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise CorpusValidationError(f"Malformed YAML in corpus registry: {exc}") from exc
    except Exception as exc:
        raise CorpusValidationError(f"Failed to parse corpus registry YAML: {exc}") from exc

    if not isinstance(parsed, dict):
        raise CorpusValidationError(
            f"Corpus registry root must be a YAML mapping/dictionary, got {type(parsed).__name__}."
        )

    try:
        return CorpusRegistry.model_validate(parsed)
    except CorpusValidationError:
        raise
    except Exception as exc:
        raise CorpusValidationError(f"Corpus registry invariant validation failed: {exc}") from exc


def check_corpus_authorization(
    registry: CorpusRegistry,
    purpose: AllowedPurpose | str,
    corpus_identifiers: Sequence[str | CorpusIdentity],
) -> tuple[bool, CorpusAuthorizationResult | None, list[str]]:
    """Check authorization for one or more exact corpus identities for a requested purpose.

    Returns:
        tuple of (is_authorized, result_or_none, list_of_denial_reasons)
    """
    # Revalidate the complete model graph at the authorization boundary. This
    # prevents unvalidated ``model_copy`` / ``model_construct`` instances from
    # bypassing schema and cross-field invariants.
    try:
        if not isinstance(registry, CorpusRegistry):
            raise TypeError(f"expected CorpusRegistry, got {type(registry).__name__}")
        registry = CorpusRegistry.model_validate(registry.model_dump(mode="python"))
    except Exception as exc:
        return False, None, [f"Invalid corpus registry supplied for authorization: {exc}"]

    # 1. Parse and validate purpose
    req_purpose: AllowedPurpose
    if isinstance(purpose, AllowedPurpose):
        req_purpose = purpose
    elif isinstance(purpose, str):
        try:
            req_purpose = AllowedPurpose(purpose)
        except ValueError:
            valid_purposes = [p.value for p in AllowedPurpose]
            return (
                False,
                None,
                [f"Invalid requested purpose '{purpose}'. Allowed purposes are: {valid_purposes}."],
            )
    else:
        return (
            False,
            None,
            [f"Invalid purpose type '{type(purpose).__name__}'. Must be string or AllowedPurpose."],
        )

    # 2. Require a non-empty sequence, while rejecting bare strings/bytes that
    # would otherwise be iterated character by character.
    if isinstance(corpus_identifiers, (str, bytes)) or not isinstance(corpus_identifiers, Sequence):
        return (
            False,
            None,
            ["Corpus identities must be supplied as a sequence, not a bare string or bytes value."],
        )
    if not corpus_identifiers:
        return (
            False,
            None,
            ["At least one corpus identity must be specified for authorization."],
        )

    # 3. Index registry entries by exact identity and by corpus_id
    exact_index: dict[tuple[str, str | None], CorpusEntry] = {}
    id_index: dict[str, list[CorpusEntry]] = {}
    for entry in registry.corpora:
        exact_index[(entry.corpus_id, entry.version_ref)] = entry
        id_index.setdefault(entry.corpus_id, []).append(entry)

    denial_reasons: list[str] = []
    authorized_list: list[AuthorizedCorpus] = []
    seen_requests: set[tuple[str, str | None]] = set()

    # 4. Evaluate each requested corpus identity fail-closed
    for raw_id in corpus_identifiers:
        try:
            ident = CorpusIdentity.parse(raw_id)
        except ValueError as err:
            denial_reasons.append(str(err))
            continue

        # Reject duplicate requested corpus identities in a single authorization call
        req_key = (ident.corpus_id, ident.version_ref)
        if req_key in seen_requests:
            denial_reasons.append(
                f"Duplicate corpus identity request detected for '{raw_id}'. "
                "Each requested corpus identity must be unique within an authorization request."
            )
            continue
        seen_requests.add(req_key)

        # Exact version reference is required for authorization
        if ident.version_ref is None:
            denial_reasons.append(
                f"Corpus '{ident.corpus_id}' is missing an exact version reference. "
                f"Authorization requires an explicit version (format: '{ident.corpus_id}@<version>')."
            )
            continue

        key = (ident.corpus_id, ident.version_ref)
        if key not in exact_index:
            if ident.corpus_id not in id_index:
                denial_reasons.append(f"Unknown corpus ID '{ident.corpus_id}'.")
            else:
                existing_entries = id_index[ident.corpus_id]
                available_versions = [
                    e.version_ref for e in existing_entries if e.version_ref is not None
                ]
                if not available_versions:
                    denial_reasons.append(
                        f"Corpus '{ident.corpus_id}' has no acquired or verified version snapshots "
                        f"(current data_state: {[e.data_state.value for e in existing_entries]})."
                    )
                else:
                    denial_reasons.append(
                        f"Corpus '{ident.corpus_id}' version '{ident.version_ref}' not found in registry "
                        f"(available versions: {available_versions})."
                    )
            continue

        entry = exact_index[key]

        # Fail-closed check: data_state must be verified
        if entry.data_state != DataState.VERIFIED:
            denial_reasons.append(
                f"Corpus '{ident.corpus_id}@{ident.version_ref}' has data_state='{entry.data_state.value}'. "
                "Only 'verified' datasets may be authorized for use."
            )
            continue

        # Fail-closed check: rights_decision must be approved or restricted
        if entry.rights_decision not in (RightsDecision.APPROVED, RightsDecision.RESTRICTED):
            denial_reasons.append(
                f"Corpus '{ident.corpus_id}@{ident.version_ref}' has rights_decision='{entry.rights_decision.value}'. "
                "Use is prohibited unless rights_decision is 'approved' or 'restricted'."
            )
            continue

        # Fail-closed check: requested purpose must be explicitly listed in allowed_purposes
        if req_purpose not in entry.allowed_purposes:
            permitted = [p.value for p in entry.allowed_purposes]
            denial_reasons.append(
                f"Corpus '{ident.corpus_id}@{ident.version_ref}' does not permit purpose '{req_purpose.value}' "
                f"(explicitly allowed purposes: {permitted})."
            )
            continue

        # Defense in depth for the highest-risk purpose. The schema invariant
        # already enforces this, and the use gate independently fails closed.
        if (
            req_purpose == AllowedPurpose.RAW_REDISTRIBUTION
            and entry.raw_redistribution_status != RedistributionStatus.ALLOWED
        ):
            denial_reasons.append(
                f"Corpus '{ident.corpus_id}@{ident.version_ref}' does not permit raw redistribution "
                f"(raw_redistribution_status='{entry.raw_redistribution_status.value}')."
            )
            continue

        # Fail-closed check: verified integrity metadata must be present
        if (
            not entry.archive_sha256
            or not entry.manifest_sha256
            or entry.sample_count is None
            or entry.sample_count <= 0
        ):
            denial_reasons.append(
                f"Corpus '{ident.corpus_id}@{ident.version_ref}' lacks verified integrity metadata "
                f"(archive_sha256={entry.archive_sha256}, manifest_sha256={entry.manifest_sha256}, sample_count={entry.sample_count})."
            )
            continue

        # Verified entries have non-null integrity metadata
        assert entry.version_ref is not None
        assert entry.archive_sha256 is not None
        assert entry.manifest_sha256 is not None
        assert entry.sample_count is not None

        authorized_list.append(
            AuthorizedCorpus(
                corpus_id=entry.corpus_id,
                version_ref=entry.version_ref,
                archive_sha256=entry.archive_sha256,
                manifest_sha256=entry.manifest_sha256,
                sample_count=entry.sample_count,
                rights_decision=entry.rights_decision,
                content_type=entry.content_type,
            )
        )

    if denial_reasons:
        return False, None, denial_reasons

    # Construct CorpusAuthorizationResult with deterministic tuple sorted by (corpus_id, version_ref)
    sorted_authorized = tuple(sorted(authorized_list, key=lambda x: (x.corpus_id, x.version_ref)))
    result = CorpusAuthorizationResult(
        purpose=req_purpose,
        authorized_corpora=sorted_authorized,
    )
    return True, result, []


def authorize_corpus_usage(
    registry: CorpusRegistry,
    purpose: AllowedPurpose | str,
    corpus_identifiers: Sequence[str | CorpusIdentity],
) -> CorpusAuthorizationResult:
    """Authorize one or more exact corpus identities for one requested purpose.

    Fails closed by raising CorpusAuthorizationError if any corpus is denied.
    """
    is_authorized, result, denial_reasons = check_corpus_authorization(
        registry, purpose, corpus_identifiers
    )
    if not is_authorized or result is None:
        p_str = purpose.value if isinstance(purpose, AllowedPurpose) else str(purpose)
        reasons_msg = "\n".join(f"  - {r}" for r in denial_reasons)
        raise CorpusAuthorizationError(
            f"Corpus authorization denied for purpose '{p_str}':\n{reasons_msg}",
            denial_reasons=denial_reasons,
        )
    return result
