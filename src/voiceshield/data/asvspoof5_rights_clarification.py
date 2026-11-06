"""Strict, non-authorizing ASVspoof 5 rights-clarification packets.

Stage C4-A records public evidence, immutable capture metadata, unresolved
questions, and a maintainer-question draft. It does not make a rights decision
or authorize contact, account creation, terms acceptance, data acquisition,
evaluation, training, product use, or redistribution.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from voiceshield.data.corpus_registry import (
    SHA256_REGEX,
    DataState,
    RightsDecision,
    load_corpus_registry,
)
from voiceshield.data.corpus_rights_recommendations import (
    RecommendationDisposition,
    compute_recommendation_bytes_hash,
    load_corpus_rights_recommendations,
)

ASVSPOOF5_CLARIFICATION_SCHEMA_VERSION = "1.0.0"
ASVSPOOF5_CLARIFICATION_BASE_COMMIT = "4d4af273acb16e28569c0573f68fa65272e013ec"
_SOURCE_ID_REGEX = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_GIT_SHA_REGEX = re.compile(r"^[0-9a-f]{40}$")
_FORBIDDEN_CAPTURE_SUFFIXES = (
    ".tar",
    ".tar.gz",
    ".tgz",
    ".zip",
    ".flac",
    ".wav",
    ".mp3",
    ".pt",
    ".pth",
    ".ckpt",
    ".bin",
    ".safetensors",
)
_FORBIDDEN_CAPTURE_FRAGMENTS = ("flac_t_", "flac_d_", "flac_e_", "asvspoof5_protocols")
_MAX_SINGLE_CAPTURE_BYTES = 10_000_000


class ASVspoof5ClarificationError(Exception):
    """Base exception for Stage C4-A packet failures."""


class ASVspoof5ClarificationValidationError(ASVspoof5ClarificationError):
    """Raised when packet, manifest, or capture invariants fail."""


class ASVspoof5ClarificationCrossValidationError(ASVspoof5ClarificationError):
    """Raised when linked frozen governance files do not match the packet."""

    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        super().__init__(message)
        self.errors: list[str] = errors or []


class ClarificationPacketStatus(StrEnum):
    """The only status allowed for this evidence-only packet."""

    NON_AUTHORIZING_CLARIFICATION = "non_authorizing_clarification"


class ComponentClarificationStatus(StrEnum):
    """Evidence state for one rights component, not a legal conclusion."""

    PARTIALLY_CLARIFIED = "partially_clarified"
    UNRESOLVED = "unresolved"


class PreconditionStatus(StrEnum):
    """Progress against one frozen C3 precondition."""

    PARTIALLY_SATISFIED = "partially_satisfied"
    SATISFIED_FOR_CURRENT_PUBLIC_PACKET = "satisfied_for_current_public_packet"
    UNRESOLVED = "unresolved"


class ClarificationDisposition(StrEnum):
    """Non-authorizing operational disposition."""

    DEFER_PENDING_CONDITIONS = "defer_pending_conditions"


def _clean_text(value: Any, field_name: str = "text") -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a non-empty string.")
    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return cleaned


def _strict_false(value: Any) -> bool:
    if value is not False:
        raise ValueError("Non-authorizing boundary flags must be strictly false.")
    return False


def _strict_true(value: Any) -> bool:
    if value is not True:
        raise ValueError("Non-decision acknowledgement flags must be strictly true.")
    return True


def _strict_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a strict positive integer.")
    return value


def _clean_unique_strings(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a sequence of non-empty strings.")
    if not value:
        raise ValueError(f"{field_name} must contain at least one entry.")
    cleaned: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        text = _clean_text(item, f"{field_name}[{index}]")
        if text in seen:
            raise ValueError(f"{field_name} contains duplicate entry '{text}'.")
        seen.add(text)
        cleaned.append(text)
    return tuple(cleaned)


def _validate_sha256(value: Any, field_name: str = "hash") -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string.")
    cleaned = str(value).strip()
    if not SHA256_REGEX.fullmatch(cleaned):
        raise ValueError(f"{field_name} must be a lowercase 64-character SHA-256 value.")
    return cleaned


class LinkedIntegrity(BaseModel):
    """Exact-byte anchors to frozen C2, C3, and registry files."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    c2_evidence_sha256: str
    c3_recommendations_sha256: str
    registry_sha256: str

    @field_validator(
        "c2_evidence_sha256", "c3_recommendations_sha256", "registry_sha256", mode="before"
    )
    @classmethod
    def validate_hash(cls, value: Any, info: Any) -> str:
        return _validate_sha256(value, info.field_name)


class AuthorizationBoundary(BaseModel):
    """Every action that remains unauthorized by Stage C4-A."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    registry_changes_authorized: Literal[False]
    acquisition_authorized: Literal[False]
    account_creation_authorized: Literal[False]
    terms_acceptance_authorized: Literal[False]
    outbound_contact_authorized: Literal[False]
    evaluation_authorized: Literal[False]
    training_authorized: Literal[False]
    product_use_authorized: Literal[False]
    redistribution_authorized: Literal[False]

    @field_validator("*", mode="before")
    @classmethod
    def validate_false(cls, value: Any) -> bool:
        return _strict_false(value)


class CaptureBoundary(BaseModel):
    """Negative acquisition assertions and the permitted capture class."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    corpus_archives_downloaded: Literal[False]
    protocol_archive_downloaded: Literal[False]
    audio_downloaded: Literal[False]
    model_weights_downloaded: Literal[False]
    accounts_created: Literal[False]
    terms_accepted: Literal[False]
    captured_material: str

    @field_validator(
        "corpus_archives_downloaded",
        "protocol_archive_downloaded",
        "audio_downloaded",
        "model_weights_downloaded",
        "accounts_created",
        "terms_accepted",
        mode="before",
    )
    @classmethod
    def validate_false(cls, value: Any) -> bool:
        return _strict_false(value)

    @field_validator("captured_material", mode="before")
    @classmethod
    def validate_text(cls, value: Any) -> str:
        return _clean_text(value, "captured_material")


class SourceCapture(BaseModel):
    """Immutable metadata for one captured public documentation artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    source_authority: str
    category: str
    url: str
    revision_ref: str
    retrieved_at_utc: datetime
    byte_size: int = Field(gt=0, le=_MAX_SINGLE_CAPTURE_BYTES)
    sha256: str
    capture_filename: str
    source_state: str
    corpus_payload: Literal[False]

    @field_validator("source_id", mode="before")
    @classmethod
    def validate_source_id(cls, value: Any) -> str:
        cleaned = _clean_text(value, "source_id")
        if not _SOURCE_ID_REGEX.fullmatch(cleaned):
            raise ValueError("source_id must use lowercase [a-z0-9_] characters.")
        return cleaned

    @field_validator("source_authority", "category", "revision_ref", "source_state", mode="before")
    @classmethod
    def validate_text(cls, value: Any, info: Any) -> str:
        return _clean_text(value, info.field_name)

    @field_validator("url", mode="before")
    @classmethod
    def validate_url(cls, value: Any) -> str:
        cleaned = _clean_text(value, "url")
        if not cleaned.startswith("https://") or any(char.isspace() for char in cleaned):
            raise ValueError("Source capture URL must be an absolute HTTPS URL without whitespace.")
        return cleaned

    @field_validator("retrieved_at_utc", mode="after")
    @classmethod
    def validate_retrieved_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("retrieved_at_utc must be timezone-aware.")
        if value.utcoffset() != UTC.utcoffset(value):
            raise ValueError("retrieved_at_utc must be expressed in UTC.")
        return value

    @field_validator("byte_size", mode="before")
    @classmethod
    def validate_byte_size(cls, value: Any) -> int:
        return _strict_positive_int(value, "byte_size")

    @field_validator("sha256", mode="before")
    @classmethod
    def validate_hash(cls, value: Any) -> str:
        return _validate_sha256(value, "sha256")

    @field_validator("capture_filename", mode="before")
    @classmethod
    def validate_filename(cls, value: Any) -> str:
        cleaned = _clean_text(value, "capture_filename")
        path = Path(cleaned)
        if path.name != cleaned or cleaned in {".", ".."}:
            raise ValueError("capture_filename must be a safe basename.")
        lowered = cleaned.lower()
        if lowered.endswith(_FORBIDDEN_CAPTURE_SUFFIXES) or any(
            fragment in lowered for fragment in _FORBIDDEN_CAPTURE_FRAGMENTS
        ):
            raise ValueError(
                "Corpus, audio, model-weight, and protocol payload filenames are forbidden."
            )
        return cleaned

    @field_validator("corpus_payload", mode="before")
    @classmethod
    def validate_no_payload(cls, value: Any) -> bool:
        return _strict_false(value)


class EvidenceStatement(BaseModel):
    """A public-source fact, with no inferred permission."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    statement_id: str
    statement: str
    source_ids: tuple[str, ...]

    @field_validator("statement_id", mode="before")
    @classmethod
    def validate_id(cls, value: Any) -> str:
        cleaned = _clean_text(value, "statement_id")
        if not _SOURCE_ID_REGEX.fullmatch(cleaned):
            raise ValueError("statement_id must use lowercase [a-z0-9_] characters.")
        return cleaned

    @field_validator("statement", mode="before")
    @classmethod
    def validate_statement(cls, value: Any) -> str:
        return _clean_text(value, "statement")

    @field_validator("source_ids", mode="before")
    @classmethod
    def validate_sources(cls, value: Any) -> tuple[str, ...]:
        return _clean_unique_strings(value, "source_ids")


class ComponentAssessment(BaseModel):
    """Evidence status for one separately traced corpus component."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    component_id: str
    role: str
    status: ComponentClarificationStatus
    evidence_summary: str
    unresolved_items: str
    source_ids: tuple[str, ...]

    @field_validator("component_id", mode="before")
    @classmethod
    def validate_id(cls, value: Any) -> str:
        cleaned = _clean_text(value, "component_id")
        if not _SOURCE_ID_REGEX.fullmatch(cleaned):
            raise ValueError("component_id must use lowercase [a-z0-9_] characters.")
        return cleaned

    @field_validator("role", "evidence_summary", "unresolved_items", mode="before")
    @classmethod
    def validate_text(cls, value: Any, info: Any) -> str:
        return _clean_text(value, info.field_name)

    @field_validator("source_ids", mode="before")
    @classmethod
    def validate_sources(cls, value: Any) -> tuple[str, ...]:
        return _clean_unique_strings(value, "source_ids")


class PreconditionAssessment(BaseModel):
    """Evidence-only progress against an exact frozen C3 precondition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    precondition_id: str
    c3_precondition: str
    status: PreconditionStatus
    evidence_update: str
    remaining_work: str
    source_ids: tuple[str, ...]

    @field_validator("precondition_id", mode="before")
    @classmethod
    def validate_id(cls, value: Any) -> str:
        cleaned = _clean_text(value, "precondition_id")
        if not _SOURCE_ID_REGEX.fullmatch(cleaned):
            raise ValueError("precondition_id must use lowercase [a-z0-9_] characters.")
        return cleaned

    @field_validator("c3_precondition", "evidence_update", "remaining_work", mode="before")
    @classmethod
    def validate_text(cls, value: Any, info: Any) -> str:
        return _clean_text(value, info.field_name)

    @field_validator("source_ids", mode="before")
    @classmethod
    def validate_sources(cls, value: Any) -> tuple[str, ...]:
        return _clean_unique_strings(value, "source_ids")


class QualifiedReviewItem(BaseModel):
    """An interpretation expressly reserved for qualified human review."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    item_id: str
    statement: str
    source_ids: tuple[str, ...]

    @field_validator("item_id", mode="before")
    @classmethod
    def validate_id(cls, value: Any) -> str:
        cleaned = _clean_text(value, "item_id")
        if not _SOURCE_ID_REGEX.fullmatch(cleaned):
            raise ValueError("item_id must use lowercase [a-z0-9_] characters.")
        return cleaned

    @field_validator("statement", mode="before")
    @classmethod
    def validate_statement(cls, value: Any) -> str:
        return _clean_text(value, "statement")

    @field_validator("source_ids", mode="before")
    @classmethod
    def validate_sources(cls, value: Any) -> tuple[str, ...]:
        return _clean_unique_strings(value, "source_ids")


class MaintainerQuestion(BaseModel):
    """One unsent question for authoritative maintainer clarification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question_id: str
    topic: str
    question: str
    reason: str
    source_ids: tuple[str, ...]

    @field_validator("question_id", mode="before")
    @classmethod
    def validate_id(cls, value: Any) -> str:
        cleaned = _clean_text(value, "question_id")
        if not _SOURCE_ID_REGEX.fullmatch(cleaned):
            raise ValueError("question_id must use lowercase [a-z0-9_] characters.")
        return cleaned

    @field_validator("topic", "question", "reason", mode="before")
    @classmethod
    def validate_text(cls, value: Any, info: Any) -> str:
        return _clean_text(value, info.field_name)

    @field_validator("source_ids", mode="before")
    @classmethod
    def validate_sources(cls, value: Any) -> tuple[str, ...]:
        return _clean_unique_strings(value, "source_ids")


class ASVspoof5RightsClarificationPacket(BaseModel):
    """Immutable root object for Stage C4-A public clarification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0.0"]
    corpus_id: Literal["asvspoof5"]
    clarification_as_of: date
    base_source_commit: str
    scoped_purpose: Literal["research_evaluation"]
    packet_status: Literal[ClarificationPacketStatus.NON_AUTHORIZING_CLARIFICATION]
    capture_manifest_path: Literal["evidence/asvspoof5/CAPTURE-MANIFEST.json"]
    capture_manifest_sha256: str
    capture_manifest_source_count: int
    capture_manifest_source_bytes: int
    linked_integrity: LinkedIntegrity
    authorization_boundary: AuthorizationBoundary
    capture_boundary: CaptureBoundary
    source_captures: tuple[SourceCapture, ...]
    verified_facts: tuple[EvidenceStatement, ...]
    component_assessments: tuple[ComponentAssessment, ...]
    c3_precondition_assessments: tuple[PreconditionAssessment, ...]
    qualified_review_items: tuple[QualifiedReviewItem, ...]
    maintainer_questions: tuple[MaintainerQuestion, ...]
    remaining_blockers: tuple[str, ...]
    recommended_disposition: Literal[ClarificationDisposition.DEFER_PENDING_CONDITIONS]
    proposed_restrictions_if_later_approved: tuple[str, ...]
    next_review_action: str
    is_not_legal_advice: Literal[True]
    is_not_project_rights_decision: Literal[True]

    @field_validator("base_source_commit", mode="before")
    @classmethod
    def validate_commit(cls, value: Any) -> str:
        cleaned = _clean_text(value, "base_source_commit")
        if not _GIT_SHA_REGEX.fullmatch(cleaned):
            raise ValueError("base_source_commit must be a lowercase 40-character Git SHA.")
        if cleaned != ASVSPOOF5_CLARIFICATION_BASE_COMMIT:
            raise ValueError(
                "base_source_commit must remain pinned to the accepted C3 commit "
                f"{ASVSPOOF5_CLARIFICATION_BASE_COMMIT}."
            )
        return cleaned

    @field_validator("capture_manifest_sha256", mode="before")
    @classmethod
    def validate_manifest_hash(cls, value: Any) -> str:
        return _validate_sha256(value, "capture_manifest_sha256")

    @field_validator(
        "capture_manifest_source_count", "capture_manifest_source_bytes", mode="before"
    )
    @classmethod
    def validate_counts(cls, value: Any, info: Any) -> int:
        return _strict_positive_int(value, info.field_name)

    @field_validator(
        "source_captures",
        "verified_facts",
        "component_assessments",
        "c3_precondition_assessments",
        "qualified_review_items",
        "maintainer_questions",
        mode="before",
    )
    @classmethod
    def validate_model_sequences(cls, value: Any, info: Any) -> tuple[Any, ...]:
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError(f"{info.field_name} must be a non-empty sequence.")
        return tuple(value)

    @field_validator("remaining_blockers", "proposed_restrictions_if_later_approved", mode="before")
    @classmethod
    def validate_text_sequences(cls, value: Any, info: Any) -> tuple[str, ...]:
        return _clean_unique_strings(value, info.field_name)

    @field_validator("next_review_action", mode="before")
    @classmethod
    def validate_next_action(cls, value: Any) -> str:
        return _clean_text(value, "next_review_action")

    @field_validator("is_not_legal_advice", "is_not_project_rights_decision", mode="before")
    @classmethod
    def validate_true(cls, value: Any) -> bool:
        return _strict_true(value)

    @model_validator(mode="after")
    def validate_graph(self) -> ASVspoof5RightsClarificationPacket:
        if self.capture_manifest_source_count != len(self.source_captures):
            raise ValueError("capture_manifest_source_count must match source_captures length.")
        if self.capture_manifest_source_bytes != sum(
            item.byte_size for item in self.source_captures
        ):
            raise ValueError("capture_manifest_source_bytes must equal captured source byte total.")

        source_ids = [source.source_id for source in self.source_captures]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source_captures must use unique source_id values.")
        filenames = [source.capture_filename for source in self.source_captures]
        if len(filenames) != len(set(filenames)):
            raise ValueError("source_captures must use unique capture_filename values.")
        if any(
            source.retrieved_at_utc.date() > self.clarification_as_of
            for source in self.source_captures
        ):
            raise ValueError("Source retrieval timestamps cannot postdate clarification_as_of.")

        required_sources = {
            "zenodo_record_14498691_api",
            "zenodo_readme_14498691",
            "zenodo_license_14498691",
            "odc_by_1_0_legal_code",
            "cc_by_4_0_legal_code",
            "openslr_mls_slr94",
            "librivox_public_domain",
            "asvspoof5_design_paper_arxiv_v4",
            "asvspoof5_evaluation_plan_phase2_v0_6",
            "coqui_xtts_v2_license",
        }
        missing_required = sorted(required_sources - set(source_ids))
        if missing_required:
            raise ValueError(f"Required public captures are missing: {missing_required}.")

        identifiers: list[str] = []
        referenced: set[str] = set()
        for collection in (
            self.verified_facts,
            self.component_assessments,
            self.c3_precondition_assessments,
            self.qualified_review_items,
            self.maintainer_questions,
        ):
            for entry in collection:
                entry_id = next(
                    getattr(entry, name)
                    for name in (
                        "statement_id",
                        "component_id",
                        "precondition_id",
                        "item_id",
                        "question_id",
                    )
                    if hasattr(entry, name)
                )
                identifiers.append(entry_id)
                referenced.update(entry.source_ids)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(
                "All finding, component, precondition, review, and question IDs must be unique."
            )
        unknown_sources = sorted(referenced - set(source_ids))
        if unknown_sources:
            raise ValueError(f"Packet references unknown source IDs: {unknown_sources}.")

        if len(self.c3_precondition_assessments) != 4:
            raise ValueError("Exactly four frozen ASVspoof 5 C3 preconditions must be assessed.")
        precondition_ids = tuple(
            assessment.precondition_id for assessment in self.c3_precondition_assessments
        )
        if precondition_ids != ("c3_p1", "c3_p2", "c3_p3", "c3_p4"):
            raise ValueError("C3 precondition IDs must be exactly c3_p1 through c3_p4 in order.")
        if not any(
            component.status == ComponentClarificationStatus.UNRESOLVED
            for component in self.component_assessments
        ):
            raise ValueError("At least one component must remain unresolved in Stage C4-A.")
        if not any(
            precondition.status == PreconditionStatus.UNRESOLVED
            for precondition in self.c3_precondition_assessments
        ):
            raise ValueError("At least one C3 precondition must remain unresolved.")
        if self.recommended_disposition != ClarificationDisposition.DEFER_PENDING_CONDITIONS:
            raise ValueError("Stage C4-A disposition must remain defer_pending_conditions.")
        return self


class ClarificationCrossValidationResult(BaseModel):
    """Deterministic verification result for C4-A and frozen governance anchors."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    is_valid: bool
    checked_source_count: int
    checked_source_bytes: int
    registry_state: str
    c3_disposition: str
    errors: tuple[str, ...] = ()


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
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
    _construct_unique_mapping,
)


def _read_bytes(source: str | Path | bytes, description: str) -> bytes:
    if isinstance(source, bytes):
        raw = source
    elif isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise ASVspoof5ClarificationValidationError(f"{description} file not found: {path}")
        try:
            raw = path.read_bytes()
        except Exception as exc:
            raise ASVspoof5ClarificationValidationError(
                f"Failed to read {description} file '{path}': {exc}"
            ) from exc
    else:
        raise ASVspoof5ClarificationValidationError(
            f"Unsupported {description} source type: {type(source).__name__}"
        )
    if not raw or not raw.strip():
        raise ASVspoof5ClarificationValidationError(
            f"{description} source is empty or contains only whitespace."
        )
    return raw


def compute_clarification_bytes_hash(data: bytes) -> str:
    """Return lowercase SHA-256 for exact packet bytes."""
    return hashlib.sha256(data).hexdigest().lower()


def compute_clarification_file_hash(path: str | Path) -> str:
    """Return lowercase SHA-256 for an exact packet file."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"ASVspoof 5 clarification packet not found: {file_path}")
    return compute_clarification_bytes_hash(file_path.read_bytes())


compute_clarification_hash = compute_clarification_file_hash


def load_asvspoof5_rights_clarification(
    source: str | Path | bytes,
) -> ASVspoof5RightsClarificationPacket:
    """Load and strictly validate a C4-A YAML packet."""
    raw = _read_bytes(source, "Clarification packet")
    try:
        parsed: Any = yaml.load(raw, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ASVspoof5ClarificationValidationError(
            f"Malformed YAML in ASVspoof 5 clarification packet: {exc}"
        ) from exc
    except Exception as exc:
        raise ASVspoof5ClarificationValidationError(
            f"Failed to parse ASVspoof 5 clarification YAML: {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ASVspoof5ClarificationValidationError(
            "ASVspoof 5 clarification root must be a YAML mapping/dictionary, "
            f"got {type(parsed).__name__}."
        )
    try:
        return ASVspoof5RightsClarificationPacket.model_validate(parsed)
    except ValidationError as exc:
        raise ASVspoof5ClarificationValidationError(
            f"ASVspoof 5 clarification invariant validation failed: {exc}"
        ) from exc


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key '{key}'")
        result[key] = value
    return result


def verify_capture_manifest(
    packet: ASVspoof5RightsClarificationPacket,
    manifest_source: str | Path | bytes,
) -> None:
    """Verify the committed capture manifest against packet metadata."""
    raw = _read_bytes(manifest_source, "Capture manifest")
    actual_hash = hashlib.sha256(raw).hexdigest().lower()
    if actual_hash != packet.capture_manifest_sha256:
        raise ASVspoof5ClarificationValidationError(
            "Capture manifest hash mismatch: "
            f"declared={packet.capture_manifest_sha256}, actual={actual_hash}."
        )
    try:
        parsed = json.loads(raw, object_pairs_hook=_unique_json_object)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ASVspoof5ClarificationValidationError(
            f"Malformed or duplicate-key capture manifest JSON: {exc}"
        ) from exc
    if not isinstance(parsed, dict) or not isinstance(parsed.get("files"), list):
        raise ASVspoof5ClarificationValidationError(
            "Capture manifest root must contain a files array."
        )
    for flag in (
        "corpus_archives_downloaded",
        "protocol_archive_downloaded",
        "account_created",
        "terms_accepted",
    ):
        if parsed.get(flag) is not False:
            raise ASVspoof5ClarificationValidationError(
                f"Capture manifest boundary '{flag}' must be strictly false."
            )

    expected = {
        item.source_id: {
            "source_id": item.source_id,
            "local_filename": item.capture_filename,
            "requested_url": item.url,
            "source_authority": item.source_authority,
            "revision_ref": item.revision_ref,
            "category": item.category,
            "source_state": item.source_state,
            "retrieved_at_utc": item.retrieved_at_utc.isoformat(timespec="microseconds").replace(
                "+00:00", "Z"
            ),
            "byte_size": item.byte_size,
            "sha256": item.sha256,
            "corpus_payload": False,
        }
        for item in packet.source_captures
    }
    actual: dict[str, Any] = {}
    for entry in parsed["files"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("source_id"), str):
            raise ASVspoof5ClarificationValidationError(
                "Every capture manifest file entry must contain a string source_id."
            )
        source_id = entry["source_id"]
        if source_id in actual:
            raise ASVspoof5ClarificationValidationError(
                f"Duplicate source_id '{source_id}' in capture manifest."
            )
        actual[source_id] = entry
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(
            source_id
            for source_id in set(actual) & set(expected)
            if actual[source_id] != expected[source_id]
        )
        raise ASVspoof5ClarificationValidationError(
            "Capture manifest entries do not match packet metadata: "
            f"missing={missing}, extra={extra}, changed={changed}."
        )


def verify_capture_directory(
    packet: ASVspoof5RightsClarificationPacket,
    capture_directory: str | Path,
) -> None:
    """Optionally verify captured artifacts by size and exact SHA-256."""
    directory = Path(capture_directory)
    if not directory.is_dir():
        raise ASVspoof5ClarificationValidationError(f"Capture directory not found: {directory}")
    errors: list[str] = []
    for capture in packet.source_captures:
        path = directory / capture.capture_filename
        if not path.is_file():
            errors.append(f"missing capture: {capture.capture_filename}")
            continue
        data = path.read_bytes()
        actual_hash = hashlib.sha256(data).hexdigest().lower()
        if len(data) != capture.byte_size:
            errors.append(
                f"size mismatch for {capture.capture_filename}: "
                f"declared={capture.byte_size}, actual={len(data)}"
            )
        if actual_hash != capture.sha256:
            errors.append(
                f"hash mismatch for {capture.capture_filename}: "
                f"declared={capture.sha256}, actual={actual_hash}"
            )
    if errors:
        raise ASVspoof5ClarificationValidationError(
            "Capture directory verification failed: " + "; ".join(errors)
        )


def cross_validate_asvspoof5_clarification(
    packet: ASVspoof5RightsClarificationPacket,
    *,
    evidence_bytes: bytes,
    recommendations_bytes: bytes,
    registry_bytes: bytes,
) -> ClarificationCrossValidationResult:
    """Cross-check C4-A against exact frozen C2, C3, and registry bytes."""
    errors: list[str] = []
    evidence_hash = hashlib.sha256(evidence_bytes).hexdigest().lower()
    recommendations_hash = compute_recommendation_bytes_hash(recommendations_bytes)
    registry_hash = hashlib.sha256(registry_bytes).hexdigest().lower()
    if evidence_hash != packet.linked_integrity.c2_evidence_sha256:
        errors.append(
            "C2 evidence hash mismatch: "
            f"declared={packet.linked_integrity.c2_evidence_sha256}, actual={evidence_hash}."
        )
    if recommendations_hash != packet.linked_integrity.c3_recommendations_sha256:
        errors.append(
            "C3 recommendation hash mismatch: "
            f"declared={packet.linked_integrity.c3_recommendations_sha256}, "
            f"actual={recommendations_hash}."
        )
    if registry_hash != packet.linked_integrity.registry_sha256:
        errors.append(
            "Registry hash mismatch: "
            f"declared={packet.linked_integrity.registry_sha256}, actual={registry_hash}."
        )

    c3_disposition = "unavailable"
    registry_state = "unavailable"
    try:
        recommendations = load_corpus_rights_recommendations(recommendations_bytes)
        matching = [
            item for item in recommendations.recommendations if item.corpus_id == "asvspoof5"
        ]
        if len(matching) != 1:
            errors.append("Frozen C3 packet must contain exactly one ASVspoof 5 recommendation.")
        else:
            c3_disposition = matching[0].disposition.value
            if matching[0].disposition != RecommendationDisposition.DEFER_PENDING_CONDITIONS:
                errors.append(
                    "Frozen C3 ASVspoof 5 disposition must remain defer_pending_conditions."
                )
            mapped_preconditions = tuple(
                item.c3_precondition for item in packet.c3_precondition_assessments
            )
            if mapped_preconditions != matching[0].required_preconditions:
                errors.append(
                    "C4-A must map the exact frozen C3 ASVspoof 5 preconditions in order."
                )
    except Exception as exc:
        errors.append(f"Failed to validate frozen C3 recommendations: {exc}")

    try:
        registry = load_corpus_registry(registry_bytes)
        matching_entries = [entry for entry in registry.corpora if entry.corpus_id == "asvspoof5"]
        if len(matching_entries) != 1:
            errors.append("Frozen registry must contain exactly one ASVspoof 5 entry.")
        else:
            entry = matching_entries[0]
            registry_state = f"{entry.data_state.value}/{entry.rights_decision.value}"
            if entry.data_state != DataState.PLANNED:
                errors.append("ASVspoof 5 data_state must remain planned.")
            if entry.rights_decision != RightsDecision.PENDING:
                errors.append("ASVspoof 5 rights_decision must remain pending.")
            if entry.allowed_purposes:
                errors.append("ASVspoof 5 allowed_purposes must remain empty.")
            if any(
                value is not None
                for value in (
                    entry.rights_reviewed_at,
                    entry.rights_reviewed_by,
                    entry.archive_sha256,
                    entry.manifest_sha256,
                    entry.sample_count,
                )
            ):
                errors.append("ASVspoof 5 review and acquisition fields must remain unset.")
    except Exception as exc:
        errors.append(f"Failed to validate frozen registry: {exc}")

    return ClarificationCrossValidationResult(
        is_valid=not errors,
        checked_source_count=len(packet.source_captures),
        checked_source_bytes=sum(item.byte_size for item in packet.source_captures),
        registry_state=registry_state,
        c3_disposition=c3_disposition,
        errors=tuple(errors),
    )


def verify_asvspoof5_rights_clarification(
    clarification_source: str | Path | bytes,
    capture_manifest_source: str | Path | bytes,
    evidence_source: str | Path | bytes,
    recommendations_source: str | Path | bytes,
    registry_source: str | Path | bytes,
    *,
    capture_directory: str | Path | None = None,
) -> tuple[ASVspoof5RightsClarificationPacket, ClarificationCrossValidationResult]:
    """Perform strict packet, capture-manifest, and frozen-governance verification."""
    packet = load_asvspoof5_rights_clarification(clarification_source)
    verify_capture_manifest(packet, capture_manifest_source)
    if capture_directory is not None:
        verify_capture_directory(packet, capture_directory)
    evidence_bytes = _read_bytes(evidence_source, "C2 evidence")
    recommendations_bytes = _read_bytes(recommendations_source, "C3 recommendations")
    registry_bytes = _read_bytes(registry_source, "Registry")
    result = cross_validate_asvspoof5_clarification(
        packet,
        evidence_bytes=evidence_bytes,
        recommendations_bytes=recommendations_bytes,
        registry_bytes=registry_bytes,
    )
    if not result.is_valid:
        raise ASVspoof5ClarificationCrossValidationError(
            "ASVspoof 5 clarification cross-validation failed.", list(result.errors)
        )
    return packet, result
