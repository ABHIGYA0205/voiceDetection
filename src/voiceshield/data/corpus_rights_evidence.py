"""Immutable corpus rights-evidence dossiers and research/product track segregation."""

from __future__ import annotations

import hashlib
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
    ValidationError,
    field_validator,
    model_validator,
)

from voiceshield.data.corpus_registry import (
    CORPUS_ID_REGEX,
    SHA256_REGEX,
    CorpusRegistry,
    DataState,
    RightsDecision,
    load_corpus_registry,
)

EVIDENCE_SCHEMA_VERSION = "1.0.0"

_HTTP_URL_ADAPTER: TypeAdapter[HttpUrl] = TypeAdapter(HttpUrl)


class CorpusEvidenceError(Exception):
    """Base exception for corpus rights-evidence errors."""


class CorpusEvidenceValidationError(CorpusEvidenceError):
    """Raised when evidence dossier structure or invariant validation fails."""


class CorpusEvidenceCrossValidationError(CorpusEvidenceError):
    """Raised when cross-validation between evidence and registry fails."""

    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        super().__init__(message)
        self.errors: list[str] = errors or []


class EvidenceSignal(StrEnum):
    """Conservative, non-authorizing evidence signal derived from source disclosures.

    These signals represent factual source disclosures and do NOT constitute
    a project rights decision, authorization, or legal clearance.
    """

    SOURCE_DECLARES_PROHIBITED = "source_declares_prohibited"
    SOURCE_DECLARES_RESTRICTED = "source_declares_restricted"
    POTENTIALLY_COMPATIBLE_NOT_REVIEWED = "potentially_compatible_not_reviewed"
    UNCLEAR = "unclear"
    NOT_APPLICABLE = "not_applicable"


class AccessMode(StrEnum):
    """Corpus distribution access mode as officially stated."""

    PUBLIC_DIRECT = "public_direct"
    ACCOUNT_REQUIRED = "account_required"
    GATED_APPROVAL = "gated_approval"
    UNCLEAR = "unclear"


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


def normalize_source_url(url: str) -> str:
    """Normalize an HTTPS evidence URL for canonical comparison and duplicate detection.

    Normalization rules:
    1. Scheme is lowercased ('https').
    2. Hostname is lowercased.
    3. Default HTTPS port (:443) is omitted from authority.
    4. Trailing-slash policy: All trailing slashes are stripped from the path component
       (e.g., 'https://example.com/' -> 'https://example.com' and
       'https://example.com/path/' -> 'https://example.com/path'). If the path is empty
       or consists only of '/', it normalizes to empty path ''.
    5. Query string and fragment components are preserved as-is.
    """
    split_res = urlsplit(url)
    scheme = split_res.scheme.lower()

    # Hostname and port handling
    hostname = (split_res.hostname or "").lower()
    port = split_res.port
    if port is None or port == 443:
        netloc = hostname
    else:
        netloc = f"{hostname}:{port}"

    # Trailing-slash policy: strip trailing slashes from path
    path = split_res.path.rstrip("/")

    # Reconstruct normalized URL
    query_part = f"?{split_res.query}" if split_res.query else ""
    fragment_part = f"#{split_res.fragment}" if split_res.fragment else ""
    return f"{scheme}://{netloc}{path}{query_part}{fragment_part}"


def _validate_sha256_hash(v: str | None, field_name: str) -> str | None:
    if v is None:
        return None
    if not isinstance(v, str):
        raise ValueError(f"{field_name} must be a string, got {type(v).__name__}.")
    v = v.strip()
    if not SHA256_REGEX.match(v):
        raise ValueError(
            f"{field_name} must be a lowercase 64-character hexadecimal SHA-256 string (got '{v}')."
        )
    return v


class EvidenceSource(BaseModel):
    """Documented official evidence source (e.g. project page, license text, paper)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str = Field(
        description="Unique source identifier within dossier, e.g. 'zenodo_record'."
    )
    url: str = Field(description="Strict HTTPS URL pointing to the official source.")
    title: str = Field(description="Human-readable title of the evidence document.")
    revision_ref: str | None = Field(
        default=None,
        description="Non-empty immutable upstream revision reference if officially documented.",
    )
    effective_date: date | None = Field(
        default=None,
        description="Official effective date or last updated date if published by source.",
    )
    content_sha256: str | None = Field(
        default=None,
        description="Lowercase SHA-256 of the captured source document if immutably captured.",
    )
    immutable_capture_required: bool = Field(
        description="Must be True if content_sha256 is null and no immutable revision is pinned.",
    )
    notes: str = Field(description="Detailed notes explaining the source and capture status.")

    @field_validator("source_id", mode="after")
    @classmethod
    def validate_source_id(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("source_id must be a non-empty string.")
        v = v.strip()
        if not CORPUS_ID_REGEX.match(v):
            raise ValueError(
                f"source_id must be a stable lowercase identifier containing only [a-z0-9_-] (got '{v}')."
            )
        return v

    @field_validator("url", mode="after")
    @classmethod
    def validate_url(cls, v: str) -> str:
        res = _validate_https_url(v, "url")
        assert res is not None
        return res

    @field_validator("title", mode="after")
    @classmethod
    def validate_title(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("title must be a non-empty string.")
        return v.strip()

    @field_validator("revision_ref", mode="after")
    @classmethod
    def validate_revision_ref(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not isinstance(v, str) or not v.strip():
            raise ValueError("revision_ref must be a non-empty string when provided.")
        return v.strip()

    @field_validator("content_sha256", mode="after")
    @classmethod
    def validate_content_sha256(cls, v: str | None) -> str | None:
        return _validate_sha256_hash(v, "content_sha256")

    @field_validator("immutable_capture_required", mode="before")
    @classmethod
    def validate_immutable_capture_required_strict(cls, v: Any) -> bool:
        if not isinstance(v, bool):
            raise ValueError(
                f"immutable_capture_required must be a strict boolean (True or False), got {type(v).__name__}."
            )
        return v

    @field_validator("notes", mode="after")
    @classmethod
    def validate_notes(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("notes must be a non-empty string.")
        return v.strip()

    @model_validator(mode="after")
    def validate_capture_policy(self) -> EvidenceSource:
        # Coherent capture-state semantics:
        # immutable_capture_required == (content_sha256 is null and revision_ref is null)
        # A source with a content hash or immutable revision must set immutable_capture_required=False.
        # A mutable source without content hash and without revision must set immutable_capture_required=True.
        expected_capture_required = self.content_sha256 is None and self.revision_ref is None
        if self.immutable_capture_required != expected_capture_required:
            if expected_capture_required:
                raise ValueError(
                    f"Evidence source '{self.source_id}' has neither content_sha256 nor revision_ref, "
                    "so immutable_capture_required must be True."
                )
            raise ValueError(
                f"Evidence source '{self.source_id}' has a pinned revision_ref or content_sha256, "
                "so immutable_capture_required must be False."
            )
        return self


class SignalAssessment(BaseModel):
    """Fact-based evidence signal for an activity track with rationale and source references."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    signal: EvidenceSignal = Field(
        description="Factual evidence signal derived strictly from source disclosures.",
    )
    rationale: str = Field(
        description="Non-empty explanation detailing factual disclosures from referenced sources.",
    )
    evidence_source_ids: tuple[str, ...] = Field(
        description="Immutable tuple of referenced source_ids in the same dossier.",
    )
    is_not_project_rights_decision: Literal[True] = Field(
        default=True,
        description="Explicit confirmation that this signal is NOT a project rights decision.",
    )

    @field_validator("rationale", mode="after")
    @classmethod
    def validate_rationale(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("rationale must be a non-empty string.")
        return v.strip()

    @field_validator("evidence_source_ids", mode="before")
    @classmethod
    def coerce_and_validate_evidence_source_ids(cls, v: Any) -> tuple[str, ...]:
        if not isinstance(v, (list, tuple)):
            raise ValueError(
                f"evidence_source_ids must be a sequence of strings, got {type(v).__name__}."
            )
        if not v:
            raise ValueError(
                "evidence_source_ids must contain at least one referenced evidence source ID."
            )
        cleaned: list[str] = []
        seen: set[str] = set()
        for idx, item in enumerate(v):
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"evidence_source_ids[{idx}] must be a non-empty string.")
            s = item.strip()
            if s in seen:
                raise ValueError(f"Duplicate evidence source ID '{s}' in evidence_source_ids.")
            seen.add(s)
            cleaned.append(s)
        return tuple(cleaned)

    @field_validator("is_not_project_rights_decision", mode="before")
    @classmethod
    def validate_is_not_project_rights_decision(cls, v: Any) -> bool:
        if v is not True:
            raise ValueError("is_not_project_rights_decision must be strictly True.")
        return True


class CorpusEvidenceDossier(BaseModel):
    """Comprehensive rights-evidence dossier for one registry corpus snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    corpus_id: str = Field(description="Registry corpus identifier, e.g. 'asvspoof5'.")
    evidence_as_of: date = Field(description="Date when evidence was compiled or audited.")
    declared_license_spdx: str = Field(description="SPDX license identifier declared by source.")
    official_release_label: str | None = Field(
        default=None,
        description="Official release label (e.g. 'v0.1', '18.0') without implying acquired version.",
    )
    access_mode: AccessMode = Field(description="Observed access and delivery model.")
    source_revision: str | None = Field(
        default=None,
        description="Official immutable upstream commit or snapshot revision, if publicly documented.",
    )
    official_sources: tuple[EvidenceSource, ...] = Field(
        description="Immutable tuple of official sources examined for this dossier (minimum 2).",
    )

    # Activity track signal assessments
    research_evaluation: SignalAssessment = Field(
        description="Evidence signal for research benchmarking and evaluation.",
    )
    research_training: SignalAssessment = Field(
        description="Evidence signal for research training and model fitting.",
    )
    product_evaluation: SignalAssessment = Field(
        description="Evidence signal for commercial or product evaluation.",
    )
    product_training: SignalAssessment = Field(
        description="Evidence signal for commercial or product detector training.",
    )
    raw_redistribution: SignalAssessment = Field(
        description="Evidence signal regarding redistribution of raw audio files.",
    )
    attribution: SignalAssessment = Field(
        description="Evidence signal regarding attribution, notices, and crediting requirements.",
    )

    # Governance and risk fields
    access_and_account_obligations: str = Field(
        description="Documented terms of service, registration, or portal access obligations.",
    )
    data_subject_privacy_considerations: str = Field(
        description="Privacy, consent, biometric, or personal data considerations disclosed by source.",
    )
    data_minimization_requirements: str = Field(
        description="Strict filtering and field minimization rules required before any future ingestion.",
    )
    unresolved_questions: tuple[str, ...] = Field(
        description="Immutable tuple of legal, technical, or licensing questions requiring human review.",
    )
    research_product_segregation_notes: str = Field(
        description="Specific boundaries separating non-commercial research exploration from product use.",
    )
    conservative_summary: str = Field(
        description="Factual summary highlighting primary restrictions and pending decision status.",
    )

    @field_validator("corpus_id", mode="after")
    @classmethod
    def validate_corpus_id(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("corpus_id must be a non-empty string.")
        v = v.strip()
        if not CORPUS_ID_REGEX.match(v):
            raise ValueError(
                f"corpus_id must be a stable lowercase identifier containing only [a-z0-9_-] (got '{v}')."
            )
        return v

    @field_validator("declared_license_spdx", mode="after")
    @classmethod
    def validate_declared_license_spdx(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("declared_license_spdx must be a non-empty string.")
        return v.strip()

    @field_validator("official_release_label", mode="after")
    @classmethod
    def validate_official_release_label(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not isinstance(v, str) or not v.strip():
            raise ValueError("official_release_label must be a non-empty string when provided.")
        return v.strip()

    @field_validator("source_revision", mode="after")
    @classmethod
    def validate_source_revision(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not isinstance(v, str) or not v.strip():
            raise ValueError("source_revision must be a non-empty string when provided.")
        return v.strip()

    @field_validator("official_sources", mode="before")
    @classmethod
    def coerce_official_sources(cls, v: Any) -> tuple[Any, ...]:
        if not isinstance(v, (list, tuple)):
            raise ValueError(f"official_sources must be a sequence, got {type(v).__name__}.")
        return tuple(v)

    @field_validator("unresolved_questions", mode="before")
    @classmethod
    def coerce_unresolved_questions(cls, v: Any) -> tuple[str, ...]:
        if not isinstance(v, (list, tuple)):
            raise ValueError(f"unresolved_questions must be a sequence, got {type(v).__name__}.")
        if not v:
            raise ValueError("unresolved_questions must contain at least one question.")
        cleaned: list[str] = []
        seen: set[str] = set()
        for idx, item in enumerate(v):
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"unresolved_questions[{idx}] must be a non-empty string.")
            q = item.strip()
            if q in seen:
                raise ValueError(f"Duplicate unresolved question detected: '{q}'.")
            seen.add(q)
            cleaned.append(q)
        return tuple(cleaned)

    @field_validator(
        "access_and_account_obligations",
        "data_subject_privacy_considerations",
        "data_minimization_requirements",
        "research_product_segregation_notes",
        "conservative_summary",
        mode="after",
    )
    @classmethod
    def validate_text_fields(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Field must be a non-empty string.")
        return v.strip()

    @model_validator(mode="after")
    def validate_dossier_invariants(self) -> CorpusEvidenceDossier:
        # 1. At least two official sources required
        if len(self.official_sources) < 2:
            raise ValueError(
                f"Corpus '{self.corpus_id}' has {len(self.official_sources)} official sources. "
                "Every evidence dossier must cite at least two distinct official evidence sources."
            )

        # 2. Source IDs must be unique
        source_id_set: set[str] = set()
        for src in self.official_sources:
            if src.source_id in source_id_set:
                raise ValueError(
                    f"Corpus '{self.corpus_id}' has duplicate official source_id '{src.source_id}'."
                )
            source_id_set.add(src.source_id)

        # 3. Normalized source URLs must be unique even when source IDs differ
        normalized_url_set: set[str] = set()
        for src in self.official_sources:
            norm_url = normalize_source_url(src.url)
            if norm_url in normalized_url_set:
                raise ValueError(
                    f"Corpus '{self.corpus_id}' has duplicate official source URL (normalized: '{norm_url}')."
                )
            normalized_url_set.add(norm_url)

        # At least two distinct normalized source URLs required
        if len(normalized_url_set) < 2:
            raise ValueError(
                f"Corpus '{self.corpus_id}' must cite at least two distinct official source URLs."
            )

        # 4. Source revision coherence: if source_revision is non-null, it must match an official-source revision_ref
        if self.source_revision is not None:
            official_revs = {
                src.revision_ref for src in self.official_sources if src.revision_ref is not None
            }
            if self.source_revision not in official_revs:
                raise ValueError(
                    f"Corpus '{self.corpus_id}' source_revision '{self.source_revision}' must equal "
                    f"at least one non-null official-source revision_ref. Found: {sorted(official_revs)}"
                )

        # 5. Effective date sanity: reject any source effective_date later than evidence_as_of
        for src in self.official_sources:
            if src.effective_date is not None and src.effective_date > self.evidence_as_of:
                raise ValueError(
                    f"Evidence source '{src.source_id}' in corpus '{self.corpus_id}' has effective_date "
                    f"({src.effective_date}) later than dossier evidence_as_of ({self.evidence_as_of})."
                )

        # 6. All referenced evidence_source_ids in signals must resolve to official_sources
        assessments = [
            ("research_evaluation", self.research_evaluation),
            ("research_training", self.research_training),
            ("product_evaluation", self.product_evaluation),
            ("product_training", self.product_training),
            ("raw_redistribution", self.raw_redistribution),
            ("attribution", self.attribution),
        ]
        for name, assess in assessments:
            for ref_id in assess.evidence_source_ids:
                if ref_id not in source_id_set:
                    raise ValueError(
                        f"Signal '{name}' in dossier '{self.corpus_id}' references unknown source_id '{ref_id}'. "
                        f"Available sources: {sorted(source_id_set)}"
                    )

        # 7. Unresolved questions must not be empty
        if not self.unresolved_questions:
            raise ValueError(
                f"Corpus '{self.corpus_id}' must record at least one unresolved legal or governance question."
            )

        return self


class CorpusRightsEvidence(BaseModel):
    """Root model for corpus rights-evidence configuration file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0.0"] = Field(
        default="1.0.0",
        description="Semantic version of the rights-evidence schema.",
    )
    dossiers: tuple[CorpusEvidenceDossier, ...] = Field(
        description="Immutable tuple of corpus evidence dossiers.",
    )

    @field_validator("dossiers", mode="before")
    @classmethod
    def coerce_dossiers(cls, v: Any) -> tuple[Any, ...]:
        if not isinstance(v, (list, tuple)):
            raise ValueError(f"dossiers must be a sequence, got {type(v).__name__}.")
        return tuple(v)

    @model_validator(mode="after")
    def validate_unique_dossiers(self) -> CorpusRightsEvidence:
        if not self.dossiers:
            raise ValueError("CorpusRightsEvidence must contain at least one dossier.")
        seen: set[str] = set()
        for d in self.dossiers:
            if d.corpus_id in seen:
                raise ValueError(f"Duplicate dossier for corpus_id '{d.corpus_id}' detected.")
            seen.add(d.corpus_id)
        return self


class CorpusEvidenceCrossValidationResult(BaseModel):
    """Result of cross-validation between evidence dossiers and corpus registry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    is_valid: bool = Field(description="Whether cross-validation passed completely.")
    evidence_corpus_ids: tuple[str, ...] = Field(
        description="Sorted list of corpus IDs in evidence."
    )
    registry_corpus_ids: tuple[str, ...] = Field(
        description="Sorted list of corpus IDs in registry."
    )
    checked_dossiers_count: int = Field(description="Number of dossiers cross-validated.")
    errors: tuple[str, ...] = Field(
        default=(), description="Tuple of error messages if validation failed."
    )


# =========================================================================
# YAML Loading with Duplicate-Key Detection
# =========================================================================


class _EvidenceUniqueKeyLoader(yaml.SafeLoader):
    """YAML SafeLoader subclass that rejects duplicate mapping keys."""


def _construct_evidence_mapping_unique(
    loader: _EvidenceUniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
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


_EvidenceUniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_evidence_mapping_unique,
)


def compute_evidence_bytes_hash(data: bytes) -> str:
    """Calculate the lowercase SHA-256 hex digest of exact evidence file bytes."""
    return hashlib.sha256(data).hexdigest().lower()


def compute_evidence_file_hash(path: str | Path) -> str:
    """Calculate the lowercase SHA-256 hex digest of the exact evidence file on disk."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Corpus rights evidence file not found: {p}")
    return compute_evidence_bytes_hash(p.read_bytes())


compute_evidence_hash = compute_evidence_file_hash


def load_corpus_rights_evidence(source: str | Path | bytes) -> CorpusRightsEvidence:
    """Load and strictly validate CorpusRightsEvidence from YAML file or raw bytes.

    Consistently raises CorpusEvidenceValidationError on syntax, schema, or invariant failures.
    """
    raw_bytes: bytes
    try:
        if isinstance(source, (str, Path)):
            p = Path(source)
            if not p.is_file():
                raise CorpusEvidenceValidationError(f"Corpus rights evidence file not found: {p}")
            raw_bytes = p.read_bytes()
        elif isinstance(source, bytes):
            raw_bytes = source
        else:
            raise CorpusEvidenceValidationError(
                f"Unsupported source type for evidence: {type(source).__name__}"
            )
    except CorpusEvidenceValidationError:
        raise
    except Exception as exc:
        raise CorpusEvidenceValidationError(
            f"Failed to read corpus rights evidence source: {exc}"
        ) from exc

    if not raw_bytes or not raw_bytes.strip():
        raise CorpusEvidenceValidationError(
            "Corpus rights evidence source is empty or contains only whitespace."
        )

    try:
        parsed: Any = yaml.load(raw_bytes, Loader=_EvidenceUniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise CorpusEvidenceValidationError(
            f"Malformed YAML in corpus rights evidence: {exc}"
        ) from exc
    except Exception as exc:
        raise CorpusEvidenceValidationError(
            f"Failed to parse corpus rights evidence YAML: {exc}"
        ) from exc

    if not isinstance(parsed, dict):
        raise CorpusEvidenceValidationError(
            f"Corpus rights evidence root must be a YAML mapping/dictionary, got {type(parsed).__name__}."
        )

    try:
        return CorpusRightsEvidence.model_validate(parsed)
    except CorpusEvidenceValidationError:
        raise
    except Exception as exc:
        raise CorpusEvidenceValidationError(
            f"Corpus rights evidence invariant validation failed: {exc}"
        ) from exc


# =========================================================================
# Cross-Validation between Evidence Dossiers and Frozen Registry
# =========================================================================


def cross_validate_evidence_and_registry(
    evidence: CorpusRightsEvidence,
    registry: CorpusRegistry,
) -> CorpusEvidenceCrossValidationResult:
    """Cross-validate corpus rights evidence against the frozen corpus registry.

    Fails closed if corpus IDs diverge, licenses mismatch, registry entries are not
    pending/planned, review/integrity metadata is fabricated, questions are missing,
    fewer than 2 official sources are cited, or dumped graph re-validation fails.
    """
    errors: list[str] = []

    # 1. Re-validate dumped copies of both complete model graphs before trusting nested instances
    validated_evidence: CorpusRightsEvidence
    try:
        if hasattr(evidence, "model_dump"):
            evidence_data = evidence.model_dump()
        elif isinstance(evidence, dict):
            evidence_data = evidence
        else:
            errors.append(f"Invalid evidence type: {type(evidence).__name__}")
            return CorpusEvidenceCrossValidationResult(
                is_valid=False,
                evidence_corpus_ids=(),
                registry_corpus_ids=(),
                checked_dossiers_count=0,
                errors=tuple(errors),
            )
        validated_evidence = CorpusRightsEvidence.model_validate(evidence_data)
    except ValidationError as exc:
        errors.append(f"Evidence graph re-validation failed: {exc}")
        return CorpusEvidenceCrossValidationResult(
            is_valid=False,
            evidence_corpus_ids=(),
            registry_corpus_ids=(),
            checked_dossiers_count=0,
            errors=tuple(errors),
        )

    validated_registry: CorpusRegistry
    try:
        if hasattr(registry, "model_dump"):
            registry_data = registry.model_dump()
        elif isinstance(registry, dict):
            registry_data = registry
        else:
            errors.append(f"Invalid registry type: {type(registry).__name__}")
            return CorpusEvidenceCrossValidationResult(
                is_valid=False,
                evidence_corpus_ids=(),
                registry_corpus_ids=(),
                checked_dossiers_count=0,
                errors=tuple(errors),
            )
        validated_registry = CorpusRegistry.model_validate(registry_data)
    except ValidationError as exc:
        errors.append(f"Registry graph re-validation failed: {exc}")
        return CorpusEvidenceCrossValidationResult(
            is_valid=False,
            evidence_corpus_ids=(),
            registry_corpus_ids=(),
            checked_dossiers_count=0,
            errors=tuple(errors),
        )

    evidence_ids = tuple(sorted(d.corpus_id for d in validated_evidence.dossiers))
    registry_ids = tuple(sorted(c.corpus_id for c in validated_registry.corpora))

    # 2. Exact corpus set equality
    if set(evidence_ids) != set(registry_ids):
        diff_missing_in_evidence = set(registry_ids) - set(evidence_ids)
        diff_extra_in_evidence = set(evidence_ids) - set(registry_ids)
        msg_parts = []
        if diff_missing_in_evidence:
            msg_parts.append(f"Missing in evidence: {sorted(diff_missing_in_evidence)}")
        if diff_extra_in_evidence:
            msg_parts.append(f"Extra in evidence: {sorted(diff_extra_in_evidence)}")
        errors.append(
            f"Evidence corpus IDs do not match registry corpus IDs. {'; '.join(msg_parts)}"
        )

    # Index registry entries by corpus_id
    registry_by_id = {c.corpus_id: c for c in validated_registry.corpora}

    for dossier in validated_evidence.dossiers:
        reg_entry = registry_by_id.get(dossier.corpus_id)
        if reg_entry is None:
            # Already flagged in set comparison
            continue

        # 3. Declared license must match registry
        if dossier.declared_license_spdx != reg_entry.declared_license_spdx:
            errors.append(
                f"Corpus '{dossier.corpus_id}' license mismatch: "
                f"evidence states '{dossier.declared_license_spdx}', "
                f"registry states '{reg_entry.declared_license_spdx}'."
            )

        # 4. Linked registry entry must remain pending
        if reg_entry.rights_decision != RightsDecision.PENDING:
            errors.append(
                f"Corpus '{dossier.corpus_id}' registry rights_decision is '{reg_entry.rights_decision.value}'. "
                "All linked registry entries must remain 'pending' during evidence stage."
            )

        # 5. Linked registry entry must remain planned
        if reg_entry.data_state != DataState.PLANNED:
            errors.append(
                f"Corpus '{dossier.corpus_id}' registry data_state is '{reg_entry.data_state.value}'. "
                "All linked registry entries must remain 'planned' during evidence stage."
            )

        # 6. All allowed_purposes in registry must remain empty
        if reg_entry.allowed_purposes:
            errors.append(
                f"Corpus '{dossier.corpus_id}' registry has allowed_purposes={reg_entry.allowed_purposes}. "
                "All allowed_purposes must remain empty in evidence stage."
            )

        # 7. No review identity or review date may appear in registry
        if reg_entry.rights_reviewed_by is not None:
            errors.append(
                f"Corpus '{dossier.corpus_id}' has rights_reviewed_by='{reg_entry.rights_reviewed_by}'. "
                "Reviewer identity must remain null in unreviewed registry."
            )
        if reg_entry.rights_reviewed_at is not None:
            errors.append(
                f"Corpus '{dossier.corpus_id}' has rights_reviewed_at='{reg_entry.rights_reviewed_at}'. "
                "Review date must remain null in unreviewed registry."
            )

        # 8. No integrity metadata may appear in registry
        if reg_entry.archive_sha256 is not None:
            errors.append(
                f"Corpus '{dossier.corpus_id}' has archive_sha256='{reg_entry.archive_sha256}'. "
                "Archive hash must remain null in unacquired planned registry."
            )
        if reg_entry.manifest_sha256 is not None:
            errors.append(
                f"Corpus '{dossier.corpus_id}' has manifest_sha256='{reg_entry.manifest_sha256}'. "
                "Manifest hash must remain null in unacquired planned registry."
            )
        if reg_entry.sample_count is not None:
            errors.append(
                f"Corpus '{dossier.corpus_id}' has sample_count={reg_entry.sample_count}. "
                "Sample count must remain null in unacquired planned registry."
            )

        # 9. Unresolved questions requirement
        if not dossier.unresolved_questions:
            errors.append(
                f"Corpus '{dossier.corpus_id}' dossier has no unresolved questions. "
                "Every dossier must document unresolved governance/rights questions."
            )

        # 10. At least two official sources requirement
        if len(dossier.official_sources) < 2:
            errors.append(
                f"Corpus '{dossier.corpus_id}' dossier has only {len(dossier.official_sources)} official sources. "
                "At least two official evidence sources are required."
            )

        # 11. Confirmation of non-authorization
        assessments = [
            ("research_evaluation", dossier.research_evaluation),
            ("research_training", dossier.research_training),
            ("product_evaluation", dossier.product_evaluation),
            ("product_training", dossier.product_training),
            ("raw_redistribution", dossier.raw_redistribution),
            ("attribution", dossier.attribution),
        ]
        for name, assess in assessments:
            if not assess.is_not_project_rights_decision:
                errors.append(
                    f"Signal '{name}' in dossier '{dossier.corpus_id}' fails to confirm non-authorization."
                )

    is_valid = len(errors) == 0
    return CorpusEvidenceCrossValidationResult(
        is_valid=is_valid,
        evidence_corpus_ids=evidence_ids,
        registry_corpus_ids=registry_ids,
        checked_dossiers_count=len(validated_evidence.dossiers),
        errors=tuple(errors),
    )


def verify_corpus_rights_evidence_compliance(
    evidence_source: str | Path | bytes,
    registry_source: str | Path | bytes = "configs/corpus_registry.yaml",
) -> tuple[CorpusRightsEvidence, CorpusRegistry, CorpusEvidenceCrossValidationResult]:
    """Load both evidence and registry, and strictly verify fail-closed compliance.

    Raises CorpusEvidenceValidationError or CorpusEvidenceCrossValidationError on failure.
    """
    evidence = load_corpus_rights_evidence(evidence_source)
    registry = load_corpus_registry(registry_source)

    result = cross_validate_evidence_and_registry(evidence, registry)
    if not result.is_valid:
        err_msg = "\n".join(f"  - {e}" for e in result.errors)
        raise CorpusEvidenceCrossValidationError(
            f"Corpus rights evidence cross-validation failed:\n{err_msg}",
            errors=list(result.errors),
        )

    return evidence, registry, result
