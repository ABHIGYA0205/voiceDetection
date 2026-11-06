"""Non-authorizing Stage C3 corpus-rights recommendation packets.

This module records conservative operational recommendations for a narrowly
scoped future rights review. Recommendations are not project rights decisions,
do not mutate the corpus registry, and cannot authorize account creation,
terms acceptance, acquisition, evaluation, training, product use, or
redistribution.
"""

from __future__ import annotations

import hashlib
import re
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from voiceshield.data.corpus_registry import (
    CORPUS_ID_REGEX,
    SHA256_REGEX,
    CorpusRegistry,
    DataState,
    RightsDecision,
    load_corpus_registry,
)
from voiceshield.data.corpus_rights_evidence import (
    CorpusRightsEvidence,
    cross_validate_evidence_and_registry,
    load_corpus_rights_evidence,
)

RECOMMENDATIONS_SCHEMA_VERSION = "1.0.0"
_SOURCE_COMMIT_REGEX = re.compile(r"^[0-9a-f]{40}$")


class CorpusRecommendationError(Exception):
    """Base exception for corpus recommendation packet errors."""


class CorpusRecommendationValidationError(CorpusRecommendationError):
    """Raised when packet structure or invariants fail validation."""


class CorpusRecommendationCrossValidationError(CorpusRecommendationError):
    """Raised when a packet does not match its evidence or frozen registry."""

    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        super().__init__(message)
        self.errors: list[str] = errors or []


class RecommendationDisposition(StrEnum):
    """Non-authorizing operational recommendation for the current scope."""

    DEFER_PENDING_CONDITIONS = "defer_pending_conditions"
    DO_NOT_PURSUE_CURRENT_SCOPE = "do_not_pursue_current_scope"


class ScopedPurpose(StrEnum):
    """Only the purpose selected for this recommendation packet."""

    RESEARCH_EVALUATION = "research_evaluation"


class CorpusRightsRecommendation(BaseModel):
    """A conservative, non-authorizing recommendation for one corpus."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    corpus_id: str = Field(description="Corpus identifier from the frozen registry.")
    review_priority: int = Field(
        ge=1,
        description="Unique one-based order for resolving preconditions; not an approval ranking.",
    )
    disposition: RecommendationDisposition = Field(
        description="Current non-authorizing disposition for the scoped purpose."
    )
    rationale: str = Field(description="Why this disposition is recommended.")
    evidence_source_ids: tuple[str, ...] = Field(
        description="Official source IDs from the linked C2 evidence dossier."
    )
    required_preconditions: tuple[str, ...] = Field(
        description="Conditions that must be resolved before a human rights decision."
    )
    restrictions_if_later_approved: tuple[str, ...] = Field(
        description="Minimum restrictions proposed if a later human review approves use."
    )
    next_review_action: str = Field(
        description="The next review action; it must not create an account, accept terms, or acquire data."
    )
    is_not_project_rights_decision: Literal[True] = Field(
        default=True,
        description="Mandatory acknowledgement that this record is not an authorization.",
    )

    @field_validator("corpus_id", mode="after")
    @classmethod
    def validate_corpus_id(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("corpus_id must be a non-empty string.")
        value = value.strip()
        if not CORPUS_ID_REGEX.match(value):
            raise ValueError(
                "corpus_id must contain only stable lowercase [a-z0-9_-] characters "
                f"(got '{value}')."
            )
        return value

    @field_validator("review_priority", mode="before")
    @classmethod
    def validate_review_priority_strict(cls, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("review_priority must be a strict positive integer.")
        return int(value)

    @field_validator("rationale", "next_review_action", mode="after")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Text fields must be non-empty strings.")
        return value.strip()

    @field_validator(
        "evidence_source_ids",
        "required_preconditions",
        "restrictions_if_later_approved",
        mode="before",
    )
    @classmethod
    def validate_non_empty_unique_sequence(cls, value: Any) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("Field must be a sequence of non-empty strings.")
        if not value:
            raise ValueError("Field must contain at least one entry.")
        cleaned: list[str] = []
        seen: set[str] = set()
        for index, item in enumerate(value):
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"Entry at index {index} must be a non-empty string.")
            item = item.strip()
            if item in seen:
                raise ValueError(f"Duplicate sequence entry detected: '{item}'.")
            seen.add(item)
            cleaned.append(item)
        return tuple(cleaned)

    @field_validator("is_not_project_rights_decision", mode="before")
    @classmethod
    def validate_non_decision_literal(cls, value: Any) -> bool:
        if value is not True:
            raise ValueError("is_not_project_rights_decision must be strictly True.")
        return True


class CorpusRightsRecommendationPacket(BaseModel):
    """Immutable root object for Stage C3 recommendations-only review."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0.0"] = Field(default="1.0.0")
    base_source_commit: str = Field(
        description="Exact VoiceShield commit whose evidence and registry were reviewed."
    )
    recommendation_as_of: date = Field(description="Date of this recommendation analysis.")
    scoped_purpose: Literal[ScopedPurpose.RESEARCH_EVALUATION] = Field(
        default=ScopedPurpose.RESEARCH_EVALUATION,
        description="The sole purpose considered by this packet.",
    )
    evidence_sha256: str = Field(description="Exact-byte SHA-256 of the C2 evidence YAML.")
    registry_sha256: str = Field(description="Exact-byte SHA-256 of the frozen registry YAML.")
    registry_changes_authorized: Literal[False] = Field(default=False)
    acquisition_authorized: Literal[False] = Field(default=False)
    account_creation_authorized: Literal[False] = Field(default=False)
    terms_acceptance_authorized: Literal[False] = Field(default=False)
    recommendations: tuple[CorpusRightsRecommendation, ...] = Field(
        description="One recommendation for every linked candidate corpus."
    )

    @field_validator("base_source_commit", mode="after")
    @classmethod
    def validate_base_source_commit(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("base_source_commit must be a string.")
        value = value.strip()
        if not _SOURCE_COMMIT_REGEX.fullmatch(value):
            raise ValueError("base_source_commit must be a lowercase 40-character Git SHA.")
        return value

    @field_validator("evidence_sha256", "registry_sha256", mode="before")
    @classmethod
    def validate_hashes(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("Hash fields must be strings.")
        value = value.strip()
        if not SHA256_REGEX.fullmatch(value):
            raise ValueError("Hash fields must be lowercase 64-character SHA-256 strings.")
        return str(value)

    @field_validator(
        "registry_changes_authorized",
        "acquisition_authorized",
        "account_creation_authorized",
        "terms_acceptance_authorized",
        mode="before",
    )
    @classmethod
    def validate_authorization_flags(cls, value: Any) -> bool:
        if value is not False:
            raise ValueError(
                "Every authorization flag in a recommendations-only packet must be false."
            )
        return False

    @field_validator("recommendations", mode="before")
    @classmethod
    def coerce_recommendations(cls, value: Any) -> tuple[Any, ...]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("recommendations must be a sequence.")
        return tuple(value)

    @model_validator(mode="after")
    def validate_packet_invariants(self) -> CorpusRightsRecommendationPacket:
        if not self.recommendations:
            raise ValueError("Recommendation packet must contain at least one recommendation.")

        corpus_ids: set[str] = set()
        priorities: set[int] = set()
        for recommendation in self.recommendations:
            if recommendation.corpus_id in corpus_ids:
                raise ValueError(
                    f"Duplicate recommendation for corpus_id '{recommendation.corpus_id}'."
                )
            corpus_ids.add(recommendation.corpus_id)

            if recommendation.review_priority in priorities:
                raise ValueError(
                    f"Duplicate review_priority {recommendation.review_priority} detected."
                )
            priorities.add(recommendation.review_priority)

        expected_priorities = set(range(1, len(self.recommendations) + 1))
        if priorities != expected_priorities:
            raise ValueError(
                "review_priority values must form a contiguous one-based sequence; "
                f"expected {sorted(expected_priorities)}, got {sorted(priorities)}."
            )
        return self


class CorpusRecommendationCrossValidationResult(BaseModel):
    """Deterministic cross-validation result for a recommendation packet."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    is_valid: bool
    scoped_purpose: ScopedPurpose
    recommendation_corpus_ids: tuple[str, ...]
    evidence_corpus_ids: tuple[str, ...]
    registry_corpus_ids: tuple[str, ...]
    checked_recommendations_count: int
    errors: tuple[str, ...] = ()


class _RecommendationUniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _RecommendationUniqueKeyLoader,
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


_RecommendationUniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def compute_recommendation_bytes_hash(data: bytes) -> str:
    """Return the lowercase SHA-256 digest of exact packet bytes."""
    return hashlib.sha256(data).hexdigest().lower()


def compute_recommendation_file_hash(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of an exact packet file."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Corpus recommendation packet not found: {file_path}")
    return compute_recommendation_bytes_hash(file_path.read_bytes())


compute_recommendation_hash = compute_recommendation_file_hash


def _read_source_bytes(source: str | Path | bytes, description: str) -> bytes:
    if isinstance(source, bytes):
        raw_bytes = source
    elif isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise CorpusRecommendationValidationError(f"{description} file not found: {path}")
        try:
            raw_bytes = path.read_bytes()
        except Exception as exc:
            raise CorpusRecommendationValidationError(
                f"Failed to read {description} file '{path}': {exc}"
            ) from exc
    else:
        raise CorpusRecommendationValidationError(
            f"Unsupported {description} source type: {type(source).__name__}"
        )

    if not raw_bytes or not raw_bytes.strip():
        raise CorpusRecommendationValidationError(
            f"{description} source is empty or contains only whitespace."
        )
    return raw_bytes


def load_corpus_rights_recommendations(
    source: str | Path | bytes,
) -> CorpusRightsRecommendationPacket:
    """Load and strictly validate a recommendation packet from YAML or bytes."""
    raw_bytes = _read_source_bytes(source, "Recommendation packet")
    try:
        parsed: Any = yaml.load(raw_bytes, Loader=_RecommendationUniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise CorpusRecommendationValidationError(
            f"Malformed YAML in corpus recommendation packet: {exc}"
        ) from exc
    except Exception as exc:
        raise CorpusRecommendationValidationError(
            f"Failed to parse corpus recommendation packet YAML: {exc}"
        ) from exc

    if not isinstance(parsed, dict):
        raise CorpusRecommendationValidationError(
            "Corpus recommendation packet root must be a YAML mapping/dictionary, "
            f"got {type(parsed).__name__}."
        )

    try:
        return CorpusRightsRecommendationPacket.model_validate(parsed)
    except ValidationError as exc:
        raise CorpusRecommendationValidationError(
            f"Corpus recommendation packet invariant validation failed: {exc}"
        ) from exc


def cross_validate_recommendations(
    packet: CorpusRightsRecommendationPacket,
    evidence: CorpusRightsEvidence,
    registry: CorpusRegistry,
    *,
    actual_evidence_sha256: str,
    actual_registry_sha256: str,
) -> CorpusRecommendationCrossValidationResult:
    """Cross-validate recommendations against exact C2 evidence and registry bytes."""
    errors: list[str] = []

    try:
        validated_packet = CorpusRightsRecommendationPacket.model_validate(packet.model_dump())
    except (AttributeError, ValidationError) as exc:
        errors.append(f"Recommendation graph re-validation failed: {exc}")
        return CorpusRecommendationCrossValidationResult(
            is_valid=False,
            scoped_purpose=ScopedPurpose.RESEARCH_EVALUATION,
            recommendation_corpus_ids=(),
            evidence_corpus_ids=(),
            registry_corpus_ids=(),
            checked_recommendations_count=0,
            errors=tuple(errors),
        )

    try:
        validated_evidence = CorpusRightsEvidence.model_validate(evidence.model_dump())
    except (AttributeError, ValidationError) as exc:
        errors.append(f"Evidence graph re-validation failed: {exc}")
        return CorpusRecommendationCrossValidationResult(
            is_valid=False,
            scoped_purpose=ScopedPurpose.RESEARCH_EVALUATION,
            recommendation_corpus_ids=(),
            evidence_corpus_ids=(),
            registry_corpus_ids=(),
            checked_recommendations_count=0,
            errors=tuple(errors),
        )

    try:
        validated_registry = CorpusRegistry.model_validate(registry.model_dump())
    except (AttributeError, ValidationError) as exc:
        errors.append(f"Registry graph re-validation failed: {exc}")
        return CorpusRecommendationCrossValidationResult(
            is_valid=False,
            scoped_purpose=ScopedPurpose.RESEARCH_EVALUATION,
            recommendation_corpus_ids=(),
            evidence_corpus_ids=(),
            registry_corpus_ids=(),
            checked_recommendations_count=0,
            errors=tuple(errors),
        )

    evidence_registry_result = cross_validate_evidence_and_registry(
        validated_evidence, validated_registry
    )
    if not evidence_registry_result.is_valid:
        errors.extend(
            f"C2 evidence/registry invariant failed: {error}"
            for error in evidence_registry_result.errors
        )

    recommendation_ids = tuple(
        sorted(recommendation.corpus_id for recommendation in validated_packet.recommendations)
    )
    evidence_ids = tuple(sorted(dossier.corpus_id for dossier in validated_evidence.dossiers))
    registry_ids = tuple(sorted(entry.corpus_id for entry in validated_registry.corpora))

    if set(recommendation_ids) != set(evidence_ids):
        missing = sorted(set(evidence_ids) - set(recommendation_ids))
        extra = sorted(set(recommendation_ids) - set(evidence_ids))
        errors.append(
            "Recommendation corpus IDs do not exactly match evidence corpus IDs: "
            f"missing={missing}, extra={extra}."
        )
    if set(recommendation_ids) != set(registry_ids):
        missing = sorted(set(registry_ids) - set(recommendation_ids))
        extra = sorted(set(recommendation_ids) - set(registry_ids))
        errors.append(
            "Recommendation corpus IDs do not exactly match registry corpus IDs: "
            f"missing={missing}, extra={extra}."
        )

    if validated_packet.evidence_sha256 != actual_evidence_sha256:
        errors.append(
            "Recommendation packet evidence_sha256 does not match exact evidence bytes: "
            f"declared={validated_packet.evidence_sha256}, actual={actual_evidence_sha256}."
        )
    if validated_packet.registry_sha256 != actual_registry_sha256:
        errors.append(
            "Recommendation packet registry_sha256 does not match exact registry bytes: "
            f"declared={validated_packet.registry_sha256}, actual={actual_registry_sha256}."
        )

    evidence_by_id = {dossier.corpus_id: dossier for dossier in validated_evidence.dossiers}
    registry_by_id = {entry.corpus_id: entry for entry in validated_registry.corpora}

    for recommendation in validated_packet.recommendations:
        dossier = evidence_by_id.get(recommendation.corpus_id)
        entry = registry_by_id.get(recommendation.corpus_id)
        if dossier is None or entry is None:
            continue

        if recommendation.disposition != RecommendationDisposition.DEFER_PENDING_CONDITIONS:
            errors.append(
                f"Corpus '{recommendation.corpus_id}' disposition must remain "
                "'defer_pending_conditions' in this recommendations-only packet."
            )

        official_source_ids = {source.source_id for source in dossier.official_sources}
        unknown_source_ids = sorted(set(recommendation.evidence_source_ids) - official_source_ids)
        if unknown_source_ids:
            errors.append(
                f"Corpus '{recommendation.corpus_id}' recommendation references unknown "
                f"evidence_source_ids={unknown_source_ids}."
            )

        evaluation_source_ids = set(dossier.research_evaluation.evidence_source_ids)
        if not evaluation_source_ids.issubset(set(recommendation.evidence_source_ids)):
            missing_sources = sorted(
                evaluation_source_ids - set(recommendation.evidence_source_ids)
            )
            errors.append(
                f"Corpus '{recommendation.corpus_id}' recommendation omits C2 research-evaluation "
                f"source IDs: {missing_sources}."
            )

        if validated_packet.recommendation_as_of < dossier.evidence_as_of:
            errors.append(
                f"Corpus '{recommendation.corpus_id}' recommendation date "
                f"{validated_packet.recommendation_as_of} predates evidence date "
                f"{dossier.evidence_as_of}."
            )

        if entry.rights_decision != RightsDecision.PENDING:
            errors.append(
                f"Corpus '{recommendation.corpus_id}' registry rights_decision must remain pending."
            )
        if entry.data_state != DataState.PLANNED:
            errors.append(
                f"Corpus '{recommendation.corpus_id}' registry data_state must remain planned."
            )
        if entry.allowed_purposes:
            errors.append(
                f"Corpus '{recommendation.corpus_id}' registry allowed_purposes must remain empty."
            )

    return CorpusRecommendationCrossValidationResult(
        is_valid=not errors,
        scoped_purpose=ScopedPurpose.RESEARCH_EVALUATION,
        recommendation_corpus_ids=recommendation_ids,
        evidence_corpus_ids=evidence_ids,
        registry_corpus_ids=registry_ids,
        checked_recommendations_count=len(validated_packet.recommendations),
        errors=tuple(errors),
    )


def verify_corpus_rights_recommendations(
    recommendation_source: str | Path | bytes,
    evidence_source: str | Path | bytes = "configs/corpus_rights_evidence.yaml",
    registry_source: str | Path | bytes = "configs/corpus_registry.yaml",
) -> tuple[
    CorpusRightsRecommendationPacket,
    CorpusRightsEvidence,
    CorpusRegistry,
    CorpusRecommendationCrossValidationResult,
]:
    """Load exact files and fail closed unless every C3 invariant passes."""
    recommendation_bytes = _read_source_bytes(recommendation_source, "Recommendation packet")
    evidence_bytes = _read_source_bytes(evidence_source, "Evidence")
    registry_bytes = _read_source_bytes(registry_source, "Registry")

    packet = load_corpus_rights_recommendations(recommendation_bytes)
    evidence = load_corpus_rights_evidence(evidence_bytes)
    registry = load_corpus_registry(registry_bytes)

    result = cross_validate_recommendations(
        packet,
        evidence,
        registry,
        actual_evidence_sha256=hashlib.sha256(evidence_bytes).hexdigest().lower(),
        actual_registry_sha256=hashlib.sha256(registry_bytes).hexdigest().lower(),
    )
    if not result.is_valid:
        formatted_errors = "\n".join(f"  - {error}" for error in result.errors)
        raise CorpusRecommendationCrossValidationError(
            f"Corpus recommendation cross-validation failed:\n{formatted_errors}",
            errors=list(result.errors),
        )

    return packet, evidence, registry, result
