"""Tests for Stage C3 non-authorizing corpus-rights recommendations."""

from __future__ import annotations

import hashlib
import json
import socket
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from voiceshield.data import (
    RECOMMENDATIONS_SCHEMA_VERSION,
    CorpusRecommendationCrossValidationError,
    CorpusRecommendationCrossValidationResult,
    CorpusRecommendationValidationError,
    CorpusRightsRecommendation,
    CorpusRightsRecommendationPacket,
    RecommendationDisposition,
    ScopedPurpose,
    compute_recommendation_bytes_hash,
    compute_recommendation_file_hash,
    compute_recommendation_hash,
    cross_validate_recommendations,
    load_corpus_rights_recommendations,
    verify_corpus_rights_recommendations,
)
from voiceshield.data.corpus_registry import (
    CorpusRegistry,
    DataState,
    RightsDecision,
    load_corpus_registry,
)
from voiceshield.data.corpus_rights_evidence import (
    CorpusRightsEvidence,
    load_corpus_rights_evidence,
)

RECOMMENDATION_PATH = Path("configs/corpus_rights_recommendations.yaml")
EVIDENCE_PATH = Path("configs/corpus_rights_evidence.yaml")
REGISTRY_PATH = Path("configs/corpus_registry.yaml")
SCRIPT_PATH = Path("scripts/validate_corpus_rights_recommendations.py")
DOC_PATH = Path("docs/corpus_rights_recommendations.md")
EXPECTED_BASE_COMMIT = "76124adfde57b4624667ee632cd0908a01a1991f"
EXPECTED_EVIDENCE_HASH = "5017f75d282b71ec6b05e589af08b02e541cbc3c27953363a03601fb78993197"
EXPECTED_REGISTRY_HASH = "686ad25c1f0e390f67d59d6b8da817d0f7009c82d7a08d8f0d9a75aeaa87ce85"


def _load_all() -> tuple[CorpusRightsRecommendationPacket, CorpusRightsEvidence, CorpusRegistry]:
    return (
        load_corpus_rights_recommendations(RECOMMENDATION_PATH),
        load_corpus_rights_evidence(EVIDENCE_PATH),
        load_corpus_registry(REGISTRY_PATH),
    )


def _cross_validate(
    packet: CorpusRightsRecommendationPacket,
    evidence: CorpusRightsEvidence,
    registry: CorpusRegistry,
    *,
    evidence_hash: str = EXPECTED_EVIDENCE_HASH,
    registry_hash: str = EXPECTED_REGISTRY_HASH,
) -> CorpusRecommendationCrossValidationResult:
    return cross_validate_recommendations(
        packet,
        evidence,
        registry,
        actual_evidence_sha256=evidence_hash,
        actual_registry_sha256=registry_hash,
    )


def _packet_dict() -> dict[str, Any]:
    return load_corpus_rights_recommendations(RECOMMENDATION_PATH).model_dump(mode="json")


def _minimal_recommendation(**updates: Any) -> CorpusRightsRecommendation:
    data: dict[str, Any] = {
        "corpus_id": "sample_corpus",
        "review_priority": 1,
        "disposition": "defer_pending_conditions",
        "rationale": "A conservative recommendation remains deferred pending review.",
        "evidence_source_ids": ["source_one"],
        "required_preconditions": ["Resolve source terms before any decision."],
        "restrictions_if_later_approved": ["Research evaluation only."],
        "next_review_action": "Seek public clarification without acquiring data.",
        "is_not_project_rights_decision": True,
    }
    data.update(updates)
    return CorpusRightsRecommendation.model_validate(data)


# ---------------------------------------------------------------------------
# Accepted packet and governance boundaries
# ---------------------------------------------------------------------------


def test_accepted_packet_and_cross_validation() -> None:
    packet, evidence, registry, result = verify_corpus_rights_recommendations(
        RECOMMENDATION_PATH, EVIDENCE_PATH, REGISTRY_PATH
    )
    assert result.is_valid is True
    assert result.errors == ()
    assert result.checked_recommendations_count == 4
    assert packet.schema_version == RECOMMENDATIONS_SCHEMA_VERSION
    assert packet.base_source_commit == EXPECTED_BASE_COMMIT
    assert packet.recommendation_as_of == date(2026, 11, 6)
    assert packet.scoped_purpose == ScopedPurpose.RESEARCH_EVALUATION
    assert result.scoped_purpose == ScopedPurpose.RESEARCH_EVALUATION
    assert result.recommendation_corpus_ids == (
        "asvspoof5",
        "common_voice_18",
        "indicvoices",
        "mlaad_v9",
    )
    assert result.recommendation_corpus_ids == result.evidence_corpus_ids
    assert result.recommendation_corpus_ids == result.registry_corpus_ids
    assert len(evidence.dossiers) == len(registry.corpora) == 4


def test_all_authorizations_false_and_registry_frozen() -> None:
    packet, _evidence, registry = _load_all()
    assert packet.registry_changes_authorized is False
    assert packet.acquisition_authorized is False
    assert packet.account_creation_authorized is False
    assert packet.terms_acceptance_authorized is False
    for recommendation in packet.recommendations:
        assert recommendation.is_not_project_rights_decision is True
        assert recommendation.disposition == RecommendationDisposition.DEFER_PENDING_CONDITIONS
    for entry in registry.corpora:
        assert entry.rights_decision == RightsDecision.PENDING
        assert entry.data_state == DataState.PLANNED
        assert entry.allowed_purposes == ()
        assert entry.rights_reviewed_by is None
        assert entry.rights_reviewed_at is None
        assert entry.version_ref is None
        assert entry.archive_sha256 is None
        assert entry.manifest_sha256 is None
        assert entry.sample_count is None


def test_integrity_anchors_match_exact_bytes() -> None:
    packet = load_corpus_rights_recommendations(RECOMMENDATION_PATH)
    assert packet.evidence_sha256 == hashlib.sha256(EVIDENCE_PATH.read_bytes()).hexdigest()
    assert packet.registry_sha256 == hashlib.sha256(REGISTRY_PATH.read_bytes()).hexdigest()
    assert packet.evidence_sha256 == EXPECTED_EVIDENCE_HASH
    assert packet.registry_sha256 == EXPECTED_REGISTRY_HASH


def test_priorities_are_exact_and_contiguous() -> None:
    packet = load_corpus_rights_recommendations(RECOMMENDATION_PATH)
    priorities = {item.corpus_id: item.review_priority for item in packet.recommendations}
    assert priorities == {
        "asvspoof5": 1,
        "mlaad_v9": 2,
        "common_voice_18": 3,
        "indicvoices": 4,
    }


def test_corpus_specific_recommendation_controls() -> None:
    packet = load_corpus_rights_recommendations(RECOMMENDATION_PATH)
    by_id = {item.corpus_id: item for item in packet.recommendations}

    asv = by_id["asvspoof5"]
    assert "Section 2.4" in asv.rationale
    assert "individual audio Contents" in asv.rationale
    assert any("Sections 2.2, 2.4, 4.2, and 4.3" in item for item in asv.required_preconditions)

    mlaad = by_id["mlaad_v9"]
    assert "M-AILABS" in mlaad.rationale
    assert "NonCommercial" in mlaad.rationale
    assert any("product" in item.lower() for item in mlaad.restrictions_if_later_approved)

    common_voice = by_id["common_voice_18"]
    assert "anti-re-identification" in common_voice.rationale
    assert any("voiceprint" in item.lower() for item in common_voice.restrictions_if_later_approved)
    assert "mdc_consumer_terms_of_service" in common_voice.evidence_source_ids

    indicvoices = by_id["indicvoices"]
    assert "gated" in indicvoices.rationale.lower()
    assert any("default-deny" in item for item in indicvoices.required_preconditions)
    assert any("phone numbers" in item for item in indicvoices.required_preconditions)


def test_documentation_preserves_non_authorization_boundaries() -> None:
    text = DOC_PATH.read_text()
    assert "recommendations-only" in text
    assert "not legal advice" in text
    assert "No decision for research evaluation automatically permits" in text
    assert EXPECTED_BASE_COMMIT in text
    assert EXPECTED_EVIDENCE_HASH in text
    assert EXPECTED_REGISTRY_HASH in text


# ---------------------------------------------------------------------------
# Strict model validation
# ---------------------------------------------------------------------------


def test_extra_fields_forbidden() -> None:
    recommendation = _minimal_recommendation()
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CorpusRightsRecommendation.model_validate({**recommendation.model_dump(), "approved": True})

    packet = _packet_dict()
    packet["decision"] = "approved"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CorpusRightsRecommendationPacket.model_validate(packet)


@pytest.mark.parametrize("corpus_id", ["", "   ", "Bad Corpus", "UPPERCASE", "bad.id"])
def test_invalid_corpus_ids_rejected(corpus_id: str) -> None:
    with pytest.raises(ValidationError, match="corpus_id"):
        _minimal_recommendation(corpus_id=corpus_id)


@pytest.mark.parametrize("priority", [True, False, "1", 1.0, None])
def test_review_priority_requires_strict_integer(priority: Any) -> None:
    with pytest.raises(ValidationError, match="strict positive integer"):
        _minimal_recommendation(review_priority=priority)


def test_review_priority_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        _minimal_recommendation(review_priority=0)


@pytest.mark.parametrize("field", ["rationale", "next_review_action"])
def test_blank_text_fields_rejected(field: str) -> None:
    with pytest.raises(ValidationError, match="non-empty strings"):
        _minimal_recommendation(**{field: "  "})


@pytest.mark.parametrize(
    "field",
    ["evidence_source_ids", "required_preconditions", "restrictions_if_later_approved"],
)
def test_sequence_fields_require_non_empty_sequences(field: str) -> None:
    with pytest.raises(ValidationError, match="must be a sequence"):
        _minimal_recommendation(**{field: "not-a-list"})
    with pytest.raises(ValidationError, match="at least one"):
        _minimal_recommendation(**{field: []})
    with pytest.raises(ValidationError, match="non-empty string"):
        _minimal_recommendation(**{field: [" "]})
    with pytest.raises(ValidationError, match="Duplicate sequence entry"):
        _minimal_recommendation(**{field: ["same", "same"]})


def test_non_decision_acknowledgement_must_be_true() -> None:
    with pytest.raises(ValidationError, match="strictly True"):
        _minimal_recommendation(is_not_project_rights_decision=False)
    with pytest.raises(ValidationError, match="strictly True"):
        _minimal_recommendation(is_not_project_rights_decision=1)


@pytest.mark.parametrize("commit", ["", "a" * 39, "A" * 40, "g" * 40, 123])
def test_invalid_base_commit_rejected(commit: Any) -> None:
    packet = _packet_dict()
    packet["base_source_commit"] = commit
    with pytest.raises(ValidationError, match="base_source_commit"):
        CorpusRightsRecommendationPacket.model_validate(packet)


@pytest.mark.parametrize("field", ["evidence_sha256", "registry_sha256"])
@pytest.mark.parametrize("value", ["", "a" * 63, "A" * 64, "z" * 64, 123])
def test_invalid_hashes_rejected(field: str, value: Any) -> None:
    packet = _packet_dict()
    packet[field] = value
    with pytest.raises(ValidationError, match="Hash fields"):
        CorpusRightsRecommendationPacket.model_validate(packet)


@pytest.mark.parametrize(
    "flag",
    [
        "registry_changes_authorized",
        "acquisition_authorized",
        "account_creation_authorized",
        "terms_acceptance_authorized",
    ],
)
@pytest.mark.parametrize("value", [True, 0, "false", None])
def test_authorization_flags_must_be_literal_false(flag: str, value: Any) -> None:
    packet = _packet_dict()
    packet[flag] = value
    with pytest.raises(ValidationError, match="authorization flag"):
        CorpusRightsRecommendationPacket.model_validate(packet)


def test_recommendations_sequence_and_uniqueness_invariants() -> None:
    packet = _packet_dict()
    packet["recommendations"] = "bad"
    with pytest.raises(ValidationError, match="must be a sequence"):
        CorpusRightsRecommendationPacket.model_validate(packet)

    packet = _packet_dict()
    packet["recommendations"] = []
    with pytest.raises(ValidationError, match="at least one recommendation"):
        CorpusRightsRecommendationPacket.model_validate(packet)

    packet = _packet_dict()
    duplicate = dict(packet["recommendations"][0])
    duplicate["review_priority"] = 5
    packet["recommendations"].append(duplicate)
    with pytest.raises(ValidationError, match="Duplicate recommendation"):
        CorpusRightsRecommendationPacket.model_validate(packet)

    packet = _packet_dict()
    packet["recommendations"][1]["review_priority"] = 1
    with pytest.raises(ValidationError, match="Duplicate review_priority"):
        CorpusRightsRecommendationPacket.model_validate(packet)

    packet = _packet_dict()
    packet["recommendations"][3]["review_priority"] = 9
    with pytest.raises(ValidationError, match="contiguous one-based sequence"):
        CorpusRightsRecommendationPacket.model_validate(packet)


# ---------------------------------------------------------------------------
# YAML loading and hashing
# ---------------------------------------------------------------------------


def test_load_from_bytes_and_deep_immutability() -> None:
    packet = load_corpus_rights_recommendations(RECOMMENDATION_PATH.read_bytes())
    assert isinstance(packet.recommendations, tuple)
    assert isinstance(packet.recommendations[0].required_preconditions, tuple)
    with pytest.raises(AttributeError):
        packet.recommendations.append(packet.recommendations[0])  # type: ignore[attr-defined]


def test_hash_helpers(tmp_path: Path) -> None:
    raw = RECOMMENDATION_PATH.read_bytes()
    expected = hashlib.sha256(raw).hexdigest()
    assert compute_recommendation_bytes_hash(raw) == expected
    assert compute_recommendation_file_hash(RECOMMENDATION_PATH) == expected
    assert compute_recommendation_hash(RECOMMENDATION_PATH) == expected
    with pytest.raises(FileNotFoundError, match="not found"):
        compute_recommendation_file_hash(tmp_path / "missing.yaml")


def test_missing_empty_and_unsupported_sources(tmp_path: Path) -> None:
    with pytest.raises(CorpusRecommendationValidationError, match="file not found"):
        load_corpus_rights_recommendations(tmp_path / "missing.yaml")
    with pytest.raises(CorpusRecommendationValidationError, match="empty"):
        load_corpus_rights_recommendations(b"")
    with pytest.raises(CorpusRecommendationValidationError, match="Unsupported"):
        load_corpus_rights_recommendations(123)  # type: ignore[arg-type]


def test_malformed_non_mapping_duplicate_and_unhashable_yaml(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("schema_version: [")
    with pytest.raises(CorpusRecommendationValidationError, match="Malformed YAML"):
        load_corpus_rights_recommendations(malformed)

    non_mapping = tmp_path / "list.yaml"
    non_mapping.write_text("- one\n- two\n")
    with pytest.raises(CorpusRecommendationValidationError, match="root must be a YAML mapping"):
        load_corpus_rights_recommendations(non_mapping)

    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text("schema_version: '1.0.0'\nschema_version: '1.0.0'\n")
    with pytest.raises(CorpusRecommendationValidationError, match="duplicate YAML key"):
        load_corpus_rights_recommendations(duplicate)

    unhashable = tmp_path / "unhashable.yaml"
    unhashable.write_text("{[one, two]: value}\n")
    with pytest.raises(CorpusRecommendationValidationError, match="unhashable mapping key"):
        load_corpus_rights_recommendations(unhashable)


def test_schema_failure_wrapped_as_domain_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    data = _packet_dict()
    data["acquisition_authorized"] = True
    bad.write_text(yaml.safe_dump(data, sort_keys=False))
    with pytest.raises(CorpusRecommendationValidationError, match="invariant validation failed"):
        load_corpus_rights_recommendations(bad)


# ---------------------------------------------------------------------------
# Cross-validation failure modes
# ---------------------------------------------------------------------------


def test_hash_mismatch_failures() -> None:
    packet, evidence, registry = _load_all()
    result = _cross_validate(packet, evidence, registry, evidence_hash="a" * 64)
    assert not result.is_valid
    assert any("evidence_sha256" in error for error in result.errors)

    result = _cross_validate(packet, evidence, registry, registry_hash="b" * 64)
    assert not result.is_valid
    assert any("registry_sha256" in error for error in result.errors)


def test_recommendation_set_mismatch_against_evidence_and_registry() -> None:
    packet, evidence, registry = _load_all()
    reduced = packet.model_copy(update={"recommendations": packet.recommendations[:-1]})
    result = _cross_validate(reduced, evidence, registry)
    assert not result.is_valid
    assert any("evidence corpus IDs" in error for error in result.errors)
    assert any("registry corpus IDs" in error for error in result.errors)


def test_non_defer_disposition_rejected_by_current_packet_policy() -> None:
    packet, evidence, registry = _load_all()
    changed = packet.recommendations[0].model_copy(
        update={"disposition": RecommendationDisposition.DO_NOT_PURSUE_CURRENT_SCOPE}
    )
    modified = packet.model_copy(update={"recommendations": (changed, *packet.recommendations[1:])})
    result = _cross_validate(modified, evidence, registry)
    assert not result.is_valid
    assert any("defer_pending_conditions" in error for error in result.errors)


def test_unknown_and_omitted_evidence_source_ids_rejected() -> None:
    packet, evidence, registry = _load_all()
    first = packet.recommendations[0]

    unknown = first.model_copy(update={"evidence_source_ids": ("unknown_source",)})
    modified = packet.model_copy(update={"recommendations": (unknown, *packet.recommendations[1:])})
    result = _cross_validate(modified, evidence, registry)
    assert not result.is_valid
    assert any("unknown evidence_source_ids" in error for error in result.errors)
    assert any("omits C2 research-evaluation" in error for error in result.errors)

    omitted = first.model_copy(update={"evidence_source_ids": ("zenodo_record_14498691",)})
    modified = packet.model_copy(update={"recommendations": (omitted, *packet.recommendations[1:])})
    result = _cross_validate(modified, evidence, registry)
    assert not result.is_valid
    assert any("odc_by_1_0_license_text" in error for error in result.errors)


def test_recommendation_date_cannot_predate_evidence() -> None:
    packet, evidence, registry = _load_all()
    modified = packet.model_copy(update={"recommendation_as_of": date(2026, 1, 1)})
    result = _cross_validate(modified, evidence, registry)
    assert not result.is_valid
    assert sum("predates evidence date" in error for error in result.errors) == 4


def test_registry_must_remain_pending() -> None:
    packet, evidence, registry = _load_all()
    data = registry.model_dump(mode="json")
    data["corpora"][0]["rights_decision"] = "approved"
    data["corpora"][0]["rights_reviewed_by"] = "Named internal reviewer"
    data["corpora"][0]["rights_reviewed_at"] = "2026-11-06"
    data["corpora"][0]["allowed_purposes"] = ["research_evaluation"]
    modified_registry = CorpusRegistry.model_validate(data)
    result = _cross_validate(packet, evidence, modified_registry)
    assert not result.is_valid
    assert any("must remain pending" in error for error in result.errors)
    assert any("allowed_purposes must remain empty" in error for error in result.errors)
    assert any("C2 evidence/registry invariant failed" in error for error in result.errors)


def test_registry_must_remain_planned() -> None:
    packet, evidence, registry = _load_all()
    data = registry.model_dump(mode="json")
    entry = data["corpora"][0]
    entry["rights_decision"] = "approved"
    entry["rights_reviewed_by"] = "Named internal reviewer"
    entry["rights_reviewed_at"] = "2026-11-06"
    entry["allowed_purposes"] = ["research_evaluation"]
    entry["data_state"] = "acquired"
    entry["version_ref"] = "candidate-version"
    entry["archive_sha256"] = "a" * 64
    modified_registry = CorpusRegistry.model_validate(data)
    result = _cross_validate(packet, evidence, modified_registry)
    assert not result.is_valid
    assert any("must remain planned" in error for error in result.errors)


def test_graph_revalidation_rejects_model_copy_bypasses() -> None:
    packet, evidence, registry = _load_all()

    bad_recommendation = packet.recommendations[0].model_copy(
        update={"is_not_project_rights_decision": False}
    )
    bad_packet = packet.model_copy(
        update={"recommendations": (bad_recommendation, *packet.recommendations[1:])}
    )
    result = _cross_validate(bad_packet, evidence, registry)
    assert not result.is_valid
    assert any("Recommendation graph re-validation failed" in error for error in result.errors)

    bad_dossier = evidence.dossiers[0].model_copy(update={"unresolved_questions": ()})
    bad_evidence = evidence.model_copy(update={"dossiers": (bad_dossier, *evidence.dossiers[1:])})
    result = _cross_validate(packet, bad_evidence, registry)
    assert not result.is_valid
    assert any("Evidence graph re-validation failed" in error for error in result.errors)

    bad_entry = registry.corpora[0].model_copy(update={"corpus_id": "INVALID ID"})
    bad_registry = registry.model_copy(update={"corpora": (bad_entry, *registry.corpora[1:])})
    result = _cross_validate(packet, evidence, bad_registry)
    assert not result.is_valid
    assert any("Registry graph re-validation failed" in error for error in result.errors)


def test_verify_raises_cross_validation_error(tmp_path: Path) -> None:
    bad_packet = _packet_dict()
    bad_packet["evidence_sha256"] = "a" * 64
    path = tmp_path / "bad-hash.yaml"
    path.write_text(yaml.safe_dump(bad_packet, sort_keys=False))
    with pytest.raises(CorpusRecommendationCrossValidationError) as exc_info:
        verify_corpus_rights_recommendations(path, EVIDENCE_PATH, REGISTRY_PATH)
    assert any("evidence_sha256" in error for error in exc_info.value.errors)


def test_verify_missing_linked_files(tmp_path: Path) -> None:
    with pytest.raises(CorpusRecommendationValidationError, match="Evidence file not found"):
        verify_corpus_rights_recommendations(
            RECOMMENDATION_PATH, tmp_path / "missing-evidence.yaml", REGISTRY_PATH
        )
    with pytest.raises(CorpusRecommendationValidationError, match="Registry file not found"):
        verify_corpus_rights_recommendations(
            RECOMMENDATION_PATH, EVIDENCE_PATH, tmp_path / "missing-registry.yaml"
        )


def test_validation_is_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Network access attempted")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    packet, _evidence, _registry, result = verify_corpus_rights_recommendations(
        RECOMMENDATION_PATH, EVIDENCE_PATH, REGISTRY_PATH
    )
    assert result.is_valid
    assert len(packet.recommendations) == 4


# ---------------------------------------------------------------------------
# CLI behavior
# ---------------------------------------------------------------------------


def test_cli_success_and_deterministic_output() -> None:
    command = [sys.executable, str(SCRIPT_PATH)]
    first = subprocess.run(command, capture_output=True, text=True, check=False)
    second = subprocess.run(command, capture_output=True, text=True, check=False)
    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == ""
    assert first.stdout == second.stdout

    result = json.loads(first.stdout)
    assert result["status"] == "valid_recommendations_only"
    assert result["checked_recommendations_count"] == 4
    assert result["counts_by_disposition"] == {
        "defer_pending_conditions": 4,
        "do_not_pursue_current_scope": 0,
    }
    assert result["scoped_purpose"] == "research_evaluation"
    assert result["base_source_commit"] == EXPECTED_BASE_COMMIT
    assert result["evidence_sha256"] == EXPECTED_EVIDENCE_HASH
    assert result["registry_sha256"] == EXPECTED_REGISTRY_HASH
    assert result["recommendation_sha256"] == compute_recommendation_file_hash(RECOMMENDATION_PATH)
    assert result["registry_changes_authorized"] is False
    assert result["acquisition_authorized"] is False
    assert result["account_creation_authorized"] is False
    assert result["terms_acceptance_authorized"] is False


def test_cli_missing_file_failure(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(tmp_path / "missing.yaml")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "Input file not found" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_validation_failure(tmp_path: Path) -> None:
    bad = tmp_path / "invalid.yaml"
    bad.write_text("schema_version: [")
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(bad), str(EVIDENCE_PATH), str(REGISTRY_PATH)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "Recommendation Validation Failed" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_cross_validation_failure(tmp_path: Path) -> None:
    data = _packet_dict()
    data["registry_sha256"] = "b" * 64
    bad = tmp_path / "bad-hash.yaml"
    bad.write_text(yaml.safe_dump(data, sort_keys=False))
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(bad), str(EVIDENCE_PATH), str(REGISTRY_PATH)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "Recommendation Cross-Validation Failed" in result.stderr
    assert "registry_sha256" in result.stderr
    assert "Traceback" not in result.stderr
