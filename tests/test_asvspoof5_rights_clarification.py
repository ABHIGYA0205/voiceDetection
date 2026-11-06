"""Tests for Stage C4-A non-authorizing ASVspoof 5 clarification."""

from __future__ import annotations

import hashlib
import json
import socket
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from voiceshield.data import (
    ASVSPOOF5_CLARIFICATION_BASE_COMMIT,
    ASVSPOOF5_CLARIFICATION_SCHEMA_VERSION,
    ASVspoof5ClarificationCrossValidationError,
    ASVspoof5ClarificationValidationError,
    ASVspoof5RightsClarificationPacket,
    ClarificationDisposition,
    ClarificationPacketStatus,
    ComponentClarificationStatus,
    PreconditionStatus,
    SourceCapture,
    compute_clarification_bytes_hash,
    compute_clarification_file_hash,
    compute_clarification_hash,
    cross_validate_asvspoof5_clarification,
    load_asvspoof5_rights_clarification,
    verify_asvspoof5_rights_clarification,
    verify_capture_directory,
    verify_capture_manifest,
)

PACKET_PATH = Path("configs/asvspoof5_rights_clarification.yaml")
MANIFEST_PATH = Path("evidence/asvspoof5/CAPTURE-MANIFEST.json")
EVIDENCE_PATH = Path("configs/corpus_rights_evidence.yaml")
RECOMMENDATIONS_PATH = Path("configs/corpus_rights_recommendations.yaml")
REGISTRY_PATH = Path("configs/corpus_registry.yaml")
DOC_PATH = Path("docs/asvspoof5_rights_clarification.md")
SCRIPT_PATH = Path("scripts/validate_asvspoof5_rights_clarification.py")
EXPECTED_MANIFEST_HASH = "382b2ae9646a287e84110f6dcf92c6e17358a05bb003572e7255cd473a0da845"
EXPECTED_EVIDENCE_HASH = "5017f75d282b71ec6b05e589af08b02e541cbc3c27953363a03601fb78993197"
EXPECTED_RECOMMENDATIONS_HASH = "d80ed9ab61d4e97ff5bca9201179e281e725c875f03399704b1986cd6a1cebb3"
EXPECTED_REGISTRY_HASH = "686ad25c1f0e390f67d59d6b8da817d0f7009c82d7a08d8f0d9a75aeaa87ce85"


def _packet() -> ASVspoof5RightsClarificationPacket:
    return load_asvspoof5_rights_clarification(PACKET_PATH)


def _packet_dict() -> dict[str, Any]:
    return _packet().model_dump(mode="json")


def _verify() -> tuple[ASVspoof5RightsClarificationPacket, Any]:
    return verify_asvspoof5_rights_clarification(
        PACKET_PATH,
        MANIFEST_PATH,
        EVIDENCE_PATH,
        RECOMMENDATIONS_PATH,
        REGISTRY_PATH,
    )


# ---------------------------------------------------------------------------
# Accepted packet and non-authorization boundaries
# ---------------------------------------------------------------------------


def test_accepted_packet_manifest_and_frozen_governance() -> None:
    packet, result = _verify()
    assert result.is_valid is True
    assert result.errors == ()
    assert packet.schema_version == ASVSPOOF5_CLARIFICATION_SCHEMA_VERSION
    assert packet.base_source_commit == ASVSPOOF5_CLARIFICATION_BASE_COMMIT
    assert packet.corpus_id == "asvspoof5"
    assert packet.clarification_as_of == date(2026, 11, 7)
    assert packet.packet_status == ClarificationPacketStatus.NON_AUTHORIZING_CLARIFICATION
    assert packet.scoped_purpose == "research_evaluation"
    assert result.registry_state == "planned/pending"
    assert result.c3_disposition == "defer_pending_conditions"


def test_all_authorization_and_acquisition_flags_are_false() -> None:
    packet = _packet()
    assert all(value is False for value in packet.authorization_boundary.model_dump().values())
    capture = packet.capture_boundary.model_dump()
    capture.pop("captured_material")
    assert all(value is False for value in capture.values())
    assert packet.is_not_legal_advice is True
    assert packet.is_not_project_rights_decision is True


def test_linked_integrity_matches_exact_frozen_bytes() -> None:
    packet = _packet()
    assert hashlib.sha256(EVIDENCE_PATH.read_bytes()).hexdigest() == EXPECTED_EVIDENCE_HASH
    assert (
        hashlib.sha256(RECOMMENDATIONS_PATH.read_bytes()).hexdigest()
        == EXPECTED_RECOMMENDATIONS_HASH
    )
    assert hashlib.sha256(REGISTRY_PATH.read_bytes()).hexdigest() == EXPECTED_REGISTRY_HASH
    assert packet.linked_integrity.c2_evidence_sha256 == EXPECTED_EVIDENCE_HASH
    assert packet.linked_integrity.c3_recommendations_sha256 == EXPECTED_RECOMMENDATIONS_HASH
    assert packet.linked_integrity.registry_sha256 == EXPECTED_REGISTRY_HASH


def test_capture_manifest_integrity_and_counts() -> None:
    packet = _packet()
    assert hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest() == EXPECTED_MANIFEST_HASH
    assert packet.capture_manifest_sha256 == EXPECTED_MANIFEST_HASH
    assert packet.capture_manifest_path == "evidence/asvspoof5/CAPTURE-MANIFEST.json"
    assert packet.capture_manifest_source_count == len(packet.source_captures) == 19
    assert packet.capture_manifest_source_bytes == 9_477_682
    assert packet.capture_manifest_source_bytes == sum(x.byte_size for x in packet.source_captures)
    verify_capture_manifest(packet, MANIFEST_PATH)


def test_capture_metadata_is_complete_unique_and_payload_free() -> None:
    packet = _packet()
    ids = [item.source_id for item in packet.source_captures]
    names = [item.capture_filename for item in packet.source_captures]
    assert len(ids) == len(set(ids))
    assert len(names) == len(set(names))
    assert all(item.corpus_payload is False for item in packet.source_captures)
    assert all(item.url.startswith("https://") for item in packet.source_captures)
    assert all(item.retrieved_at_utc.utcoffset() == timedelta(0) for item in packet.source_captures)
    assert all(len(item.sha256) == 64 for item in packet.source_captures)


def test_required_authoritative_sources_are_present() -> None:
    source_ids = {item.source_id for item in _packet().source_captures}
    assert {
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
    }.issubset(source_ids)


def test_verified_facts_preserve_source_fact_boundary() -> None:
    packet = _packet()
    statements = {item.statement_id: item.statement for item in packet.verified_facts}
    assert "CC BY 4.0" in statements["vf_database_and_bonafide_terms"]
    assert "does not state a separate content licence" in statements["vf_spoof_generation"]
    assert "do not establish which exact versions" in statements["vf_coqui_public_metadata"]
    assert "do not expressly classify every rule" in statements["vf_challenge_rules"]


def test_components_separate_bonafide_spoof_models_and_privacy() -> None:
    packet = _packet()
    components = {item.component_id: item for item in packet.component_assessments}
    assert components["bonafide_audio"].status == ComponentClarificationStatus.PARTIALLY_CLARIFIED
    assert components["spoof_tts_vc_audio"].status == ComponentClarificationStatus.UNRESOLVED
    assert components["adversarial_spoof_audio"].status == ComponentClarificationStatus.UNRESOLVED
    assert components["pretrained_attack_models"].status == ComponentClarificationStatus.UNRESOLVED
    assert components["privacy_voice_personality"].status == ComponentClarificationStatus.UNRESOLVED
    assert "contributor assignment" in components["spoof_tts_vc_audio"].unresolved_items


def test_all_c3_preconditions_are_mapped_exactly() -> None:
    packet = _packet()
    assessments = {item.precondition_id: item for item in packet.c3_precondition_assessments}
    assert set(assessments) == {"c3_p1", "c3_p2", "c3_p3", "c3_p4"}
    assert assessments["c3_p1"].status == PreconditionStatus.PARTIALLY_SATISFIED
    assert assessments["c3_p2"].status == PreconditionStatus.UNRESOLVED
    assert assessments["c3_p3"].status == PreconditionStatus.SATISFIED_FOR_CURRENT_PUBLIC_PACKET
    assert assessments["c3_p4"].status == PreconditionStatus.UNRESOLVED
    assert "No qualified human review" in assessments["c3_p2"].remaining_work
    assert "not permission" not in assessments["c3_p3"].evidence_update


def test_disposition_and_blockers_remain_conservative() -> None:
    packet = _packet()
    assert packet.recommended_disposition == ClarificationDisposition.DEFER_PENDING_CONDITIONS
    blockers = " ".join(packet.remaining_blockers)
    assert "spoof" in blockers.lower()
    assert "A17, A28, and A29" in blockers
    assert "Qualified review" in blockers
    assert "Phase 2" in blockers


def test_maintainer_questions_are_unsent_and_cover_open_components() -> None:
    packet = _packet()
    questions = {item.question_id: item for item in packet.maintainer_questions}
    assert len(questions) == 9
    assert packet.authorization_boundary.outbound_contact_authorized is False
    assert "content-level licence" in questions["mq_spoof_content_terms"].question
    assert "versions and checkpoints" in questions["mq_model_versions"].question
    assert "which Phase 2 rules" in questions["mq_post_challenge_rules"].question
    assert "consent" in questions["mq_voice_privacy"].question


def test_proposed_approval_boundary_is_track1_only_and_minimized() -> None:
    restrictions = " ".join(_packet().proposed_restrictions_if_later_approved)
    assert "Track 1" in restrictions
    assert "do not infer authorization for Track 2" in restrictions
    assert "Do not authorize training" in restrictions
    assert "exclude gender" in restrictions
    assert "Prohibit identity inference" in restrictions
    assert "aggregate metrics" in restrictions


def test_documentation_states_findings_and_unsent_contact_boundary() -> None:
    text = DOC_PATH.read_text()
    assert "non-authorizing" in text
    assert "not legal advice" in text
    assert "Bona fide content is materially clearer" in text
    assert "Spoof content remains the main rights blocker" in text
    assert "Maintainer-question draft — not sent" in text
    assert "No message should be sent" in text
    assert ASVSPOOF5_CLARIFICATION_BASE_COMMIT in text
    assert EXPECTED_MANIFEST_HASH in text


# ---------------------------------------------------------------------------
# Strict model and YAML validation
# ---------------------------------------------------------------------------


def test_extra_root_and_capture_fields_are_forbidden() -> None:
    data = _packet_dict()
    data["approved"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ASVspoof5RightsClarificationPacket.model_validate(data)

    capture = data = _packet_dict()["source_captures"][0]
    capture["downloaded_audio"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SourceCapture.model_validate(capture)


@pytest.mark.parametrize("value", ["", "a" * 39, "A" * 40, "g" * 40, 123])
def test_base_commit_requires_exact_lowercase_git_sha(value: Any) -> None:
    data = _packet_dict()
    data["base_source_commit"] = value
    with pytest.raises(ValidationError, match="base_source_commit"):
        ASVspoof5RightsClarificationPacket.model_validate(data)


def test_base_commit_rejects_other_valid_sha() -> None:
    data = _packet_dict()
    data["base_source_commit"] = "a" * 40
    with pytest.raises(ValidationError, match="accepted C3 commit"):
        ASVspoof5RightsClarificationPacket.model_validate(data)


@pytest.mark.parametrize("value", ["", "a" * 63, "A" * 64, "z" * 64, 123])
def test_manifest_hash_requires_lowercase_sha256(value: Any) -> None:
    data = _packet_dict()
    data["capture_manifest_sha256"] = value
    with pytest.raises(ValidationError, match="capture_manifest_sha256"):
        ASVspoof5RightsClarificationPacket.model_validate(data)


@pytest.mark.parametrize(
    "flag",
    [
        "registry_changes_authorized",
        "acquisition_authorized",
        "account_creation_authorized",
        "terms_acceptance_authorized",
        "outbound_contact_authorized",
        "evaluation_authorized",
        "training_authorized",
        "product_use_authorized",
        "redistribution_authorized",
    ],
)
@pytest.mark.parametrize("value", [True, 0, "false", None])
def test_authorization_flags_must_be_literal_false(flag: str, value: Any) -> None:
    data = _packet_dict()
    data["authorization_boundary"][flag] = value
    with pytest.raises(ValidationError, match="strictly false"):
        ASVspoof5RightsClarificationPacket.model_validate(data)


@pytest.mark.parametrize(
    "flag",
    [
        "corpus_archives_downloaded",
        "protocol_archive_downloaded",
        "audio_downloaded",
        "model_weights_downloaded",
        "accounts_created",
        "terms_accepted",
    ],
)
def test_capture_boundary_flags_cannot_claim_acquisition(flag: str) -> None:
    data = _packet_dict()
    data["capture_boundary"][flag] = True
    with pytest.raises(ValidationError, match="strictly false"):
        ASVspoof5RightsClarificationPacket.model_validate(data)


@pytest.mark.parametrize("field", ["is_not_legal_advice", "is_not_project_rights_decision"])
@pytest.mark.parametrize("value", [False, 1, "true", None])
def test_non_decision_acknowledgements_must_be_literal_true(field: str, value: Any) -> None:
    data = _packet_dict()
    data[field] = value
    with pytest.raises(ValidationError, match="strictly true"):
        ASVspoof5RightsClarificationPacket.model_validate(data)


@pytest.mark.parametrize("source_id", ["", "Bad-ID", "has.dot", "has space", "_leading"])
def test_source_id_format_is_strict(source_id: str) -> None:
    data = _packet_dict()["source_captures"][0]
    data["source_id"] = source_id
    with pytest.raises(ValidationError, match="source_id"):
        SourceCapture.model_validate(data)


@pytest.mark.parametrize("url", ["http://example.test", "example.test", "https://bad url", ""])
def test_source_urls_require_https_without_whitespace(url: str) -> None:
    data = _packet_dict()["source_captures"][0]
    data["url"] = url
    with pytest.raises(ValidationError, match="HTTPS URL|non-empty string"):
        SourceCapture.model_validate(data)


@pytest.mark.parametrize(
    "filename",
    [
        "../README.txt",
        "flac_T_aa.tar",
        "ASVspoof5_protocols.tar.gz",
        "sample.flac",
        "model.zip",
        "clip.wav",
        "checkpoint.pth",
        "weights.safetensors",
    ],
)
def test_payload_capture_filenames_are_rejected(filename: str) -> None:
    data = _packet_dict()["source_captures"][0]
    data["capture_filename"] = filename
    with pytest.raises(ValidationError, match="forbidden|safe basename"):
        SourceCapture.model_validate(data)


def test_source_byte_size_is_strict_and_bounded() -> None:
    data = _packet_dict()["source_captures"][0]
    for value in (True, 0, -1, "10", 10_000_001):
        data["byte_size"] = value
        with pytest.raises(ValidationError, match="byte_size|less than or equal"):
            SourceCapture.model_validate(data)


def test_packet_sequence_count_and_total_invariants() -> None:
    data = _packet_dict()
    data["capture_manifest_source_count"] += 1
    with pytest.raises(ValidationError, match="source_count"):
        ASVspoof5RightsClarificationPacket.model_validate(data)

    data = _packet_dict()
    data["capture_manifest_source_bytes"] += 1
    with pytest.raises(ValidationError, match="source_bytes"):
        ASVspoof5RightsClarificationPacket.model_validate(data)


def test_source_retrieval_cannot_postdate_clarification() -> None:
    data = _packet_dict()
    data["source_captures"][0]["retrieved_at_utc"] = "2026-11-08T00:00:00Z"
    with pytest.raises(ValidationError, match="cannot postdate"):
        ASVspoof5RightsClarificationPacket.model_validate(data)


def test_duplicate_source_ids_and_filenames_rejected() -> None:
    data = _packet_dict()
    data["source_captures"][1]["source_id"] = data["source_captures"][0]["source_id"]
    with pytest.raises(ValidationError, match="unique source_id"):
        ASVspoof5RightsClarificationPacket.model_validate(data)

    data = _packet_dict()
    data["source_captures"][1]["capture_filename"] = data["source_captures"][0]["capture_filename"]
    with pytest.raises(ValidationError, match="unique capture_filename"):
        ASVspoof5RightsClarificationPacket.model_validate(data)


def test_unknown_source_reference_and_duplicate_graph_id_rejected() -> None:
    data = _packet_dict()
    data["verified_facts"][0]["source_ids"] = ["unknown_source"]
    with pytest.raises(ValidationError, match="unknown source IDs"):
        ASVspoof5RightsClarificationPacket.model_validate(data)

    data = _packet_dict()
    data["maintainer_questions"][0]["question_id"] = data["qualified_review_items"][0]["item_id"]
    with pytest.raises(ValidationError, match="must be unique"):
        ASVspoof5RightsClarificationPacket.model_validate(data)


def test_required_source_and_unresolved_invariants() -> None:
    data = _packet_dict()
    removed = data["source_captures"].pop(0)
    data["capture_manifest_source_count"] -= 1
    data["capture_manifest_source_bytes"] -= removed["byte_size"]
    with pytest.raises(ValidationError, match="Required public captures"):
        ASVspoof5RightsClarificationPacket.model_validate(data)

    data = _packet_dict()
    for component in data["component_assessments"]:
        component["status"] = "partially_clarified"
    with pytest.raises(ValidationError, match="must remain unresolved"):
        ASVspoof5RightsClarificationPacket.model_validate(data)


def test_exactly_four_ordered_preconditions_required() -> None:
    data = _packet_dict()
    data["c3_precondition_assessments"].pop()
    with pytest.raises(ValidationError, match="Exactly four"):
        ASVspoof5RightsClarificationPacket.model_validate(data)

    data = _packet_dict()
    data["c3_precondition_assessments"][0]["precondition_id"] = "c3_px"
    with pytest.raises(ValidationError, match="exactly c3_p1 through c3_p4"):
        ASVspoof5RightsClarificationPacket.model_validate(data)


def test_load_from_bytes_is_deeply_immutable() -> None:
    packet = load_asvspoof5_rights_clarification(PACKET_PATH.read_bytes())
    assert isinstance(packet.source_captures, tuple)
    assert isinstance(packet.verified_facts[0].source_ids, tuple)
    with pytest.raises(AttributeError):
        packet.source_captures.append(packet.source_captures[0])  # type: ignore[attr-defined]


def test_hash_helpers(tmp_path: Path) -> None:
    raw = PACKET_PATH.read_bytes()
    expected = hashlib.sha256(raw).hexdigest()
    assert compute_clarification_bytes_hash(raw) == expected
    assert compute_clarification_file_hash(PACKET_PATH) == expected
    assert compute_clarification_hash(PACKET_PATH) == expected
    with pytest.raises(FileNotFoundError, match="not found"):
        compute_clarification_file_hash(tmp_path / "missing.yaml")


def test_missing_empty_and_unsupported_packet_sources(tmp_path: Path) -> None:
    with pytest.raises(ASVspoof5ClarificationValidationError, match="file not found"):
        load_asvspoof5_rights_clarification(tmp_path / "missing.yaml")
    with pytest.raises(ASVspoof5ClarificationValidationError, match="empty"):
        load_asvspoof5_rights_clarification(b"")
    with pytest.raises(ASVspoof5ClarificationValidationError, match="Unsupported"):
        load_asvspoof5_rights_clarification(123)  # type: ignore[arg-type]


def test_malformed_non_mapping_duplicate_and_unhashable_yaml(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("schema_version: [")
    with pytest.raises(ASVspoof5ClarificationValidationError, match="Malformed YAML"):
        load_asvspoof5_rights_clarification(malformed)

    non_mapping = tmp_path / "list.yaml"
    non_mapping.write_text("- one\n- two\n")
    with pytest.raises(ASVspoof5ClarificationValidationError, match="root must be a YAML mapping"):
        load_asvspoof5_rights_clarification(non_mapping)

    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text("schema_version: 1.0.0\nschema_version: 1.0.0\n")
    with pytest.raises(ASVspoof5ClarificationValidationError, match="duplicate YAML key"):
        load_asvspoof5_rights_clarification(duplicate)

    unhashable = tmp_path / "unhashable.yaml"
    unhashable.write_text("{[one, two]: value}\n")
    with pytest.raises(ASVspoof5ClarificationValidationError, match="unhashable mapping key"):
        load_asvspoof5_rights_clarification(unhashable)


# ---------------------------------------------------------------------------
# Manifest, retained capture, and frozen governance failure modes
# ---------------------------------------------------------------------------


def test_manifest_hash_mismatch_duplicate_key_and_boundary_failures(tmp_path: Path) -> None:
    packet = _packet()
    with pytest.raises(ASVspoof5ClarificationValidationError, match="hash mismatch"):
        verify_capture_manifest(packet, MANIFEST_PATH.read_bytes() + b"\n")

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"files": [], "files": []}')
    changed = packet.model_copy(
        update={"capture_manifest_sha256": hashlib.sha256(duplicate.read_bytes()).hexdigest()}
    )
    with pytest.raises(ASVspoof5ClarificationValidationError, match="duplicate-key"):
        verify_capture_manifest(changed, duplicate)

    manifest = json.loads(MANIFEST_PATH.read_text())
    manifest["audio_downloaded"] = False
    manifest["corpus_archives_downloaded"] = True
    raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    changed = packet.model_copy(update={"capture_manifest_sha256": hashlib.sha256(raw).hexdigest()})
    with pytest.raises(ASVspoof5ClarificationValidationError, match="strictly false"):
        verify_capture_manifest(changed, raw)


def test_manifest_entry_drift_is_rejected() -> None:
    packet = _packet()
    manifest = json.loads(MANIFEST_PATH.read_text())
    manifest["files"][0]["sha256"] = "a" * 64
    raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    changed = packet.model_copy(update={"capture_manifest_sha256": hashlib.sha256(raw).hexdigest()})
    with pytest.raises(ASVspoof5ClarificationValidationError, match="changed="):
        verify_capture_manifest(changed, raw)


def test_optional_capture_directory_verification(tmp_path: Path) -> None:
    packet = _packet()
    content = b"public documentation"
    replacement = packet.source_captures[0].model_copy(
        update={
            "capture_filename": "public-doc.txt",
            "byte_size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    )
    captures = (replacement, *packet.source_captures[1:])
    total = sum(item.byte_size for item in captures)
    changed = packet.model_copy(
        update={"source_captures": captures, "capture_manifest_source_bytes": total}
    )
    for item in changed.source_captures[1:]:
        (tmp_path / item.capture_filename).write_bytes(b"missing on purpose")
    (tmp_path / "public-doc.txt").write_bytes(content)
    with pytest.raises(ASVspoof5ClarificationValidationError, match="verification failed"):
        verify_capture_directory(changed, tmp_path)

    # Root policy requires the full public packet, so exercise the helper through a model copy.
    single = packet.model_copy(
        update={
            "source_captures": (replacement,),
            "capture_manifest_source_count": 1,
            "capture_manifest_source_bytes": len(content),
        }
    )
    verify_capture_directory(single, tmp_path)
    (tmp_path / "public-doc.txt").write_bytes(content + b"!")
    with pytest.raises(ASVspoof5ClarificationValidationError, match="size mismatch|hash mismatch"):
        verify_capture_directory(single, tmp_path)


def test_cross_validation_hash_mismatches() -> None:
    packet = _packet()
    result = cross_validate_asvspoof5_clarification(
        packet,
        evidence_bytes=b"changed",
        recommendations_bytes=RECOMMENDATIONS_PATH.read_bytes(),
        registry_bytes=REGISTRY_PATH.read_bytes(),
    )
    assert result.is_valid is False
    assert any("C2 evidence hash mismatch" in error for error in result.errors)

    result = cross_validate_asvspoof5_clarification(
        packet,
        evidence_bytes=EVIDENCE_PATH.read_bytes(),
        recommendations_bytes=b"changed",
        registry_bytes=b"changed",
    )
    assert result.is_valid is False
    assert any("C3 recommendation hash mismatch" in error for error in result.errors)
    assert any("Registry hash mismatch" in error for error in result.errors)


def test_cross_validation_rejects_changed_c3_precondition_text() -> None:
    packet = _packet()
    assessments = list(packet.c3_precondition_assessments)
    assessments[0] = assessments[0].model_copy(update={"c3_precondition": "Changed text."})
    result = cross_validate_asvspoof5_clarification(
        packet.model_copy(update={"c3_precondition_assessments": tuple(assessments)}),
        evidence_bytes=EVIDENCE_PATH.read_bytes(),
        recommendations_bytes=RECOMMENDATIONS_PATH.read_bytes(),
        registry_bytes=REGISTRY_PATH.read_bytes(),
    )
    assert result.is_valid is False
    assert any("exact frozen C3" in error for error in result.errors)


def test_registry_mutation_remains_rejected() -> None:
    packet = _packet()
    registry = yaml.safe_load(REGISTRY_PATH.read_text())
    entry = next(item for item in registry["corpora"] if item["corpus_id"] == "asvspoof5")
    entry["rights_decision"] = "approved"
    entry["data_state"] = "acquired"
    entry["allowed_purposes"] = ["research_evaluation"]
    entry["rights_reviewed_by"] = "Named reviewer"
    entry["rights_reviewed_at"] = "2026-11-07"
    entry["version_ref"] = "v0.1"
    entry["archive_sha256"] = "a" * 64
    raw = yaml.safe_dump(registry, sort_keys=False).encode()
    result = cross_validate_asvspoof5_clarification(
        packet.model_copy(
            update={
                "linked_integrity": packet.linked_integrity.model_copy(
                    update={"registry_sha256": hashlib.sha256(raw).hexdigest()}
                )
            }
        ),
        evidence_bytes=EVIDENCE_PATH.read_bytes(),
        recommendations_bytes=RECOMMENDATIONS_PATH.read_bytes(),
        registry_bytes=raw,
    )
    assert result.is_valid is False
    assert any("must remain planned" in error for error in result.errors)
    assert any("must remain pending" in error for error in result.errors)
    assert any("allowed_purposes must remain empty" in error for error in result.errors)


def test_verify_raises_cross_validation_error_for_linked_drift() -> None:
    data = _packet_dict()
    data["linked_integrity"]["registry_sha256"] = "a" * 64
    raw = yaml.safe_dump(data, sort_keys=False).encode()
    with pytest.raises(ASVspoof5ClarificationCrossValidationError) as exc_info:
        verify_asvspoof5_rights_clarification(
            raw,
            MANIFEST_PATH,
            EVIDENCE_PATH,
            RECOMMENDATIONS_PATH,
            REGISTRY_PATH,
        )
    assert any("Registry hash mismatch" in error for error in exc_info.value.errors)


def test_validation_is_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Network access attempted")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    packet, result = _verify()
    assert result.is_valid
    assert len(packet.source_captures) == 19


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

    output = json.loads(first.stdout)
    assert output["status"] == "valid_non_authorizing_clarification"
    assert output["packet_status"] == "non_authorizing_clarification"
    assert output["recommended_disposition"] == "defer_pending_conditions"
    assert output["captured_source_count"] == 19
    assert output["captured_source_bytes"] == 9_477_682
    assert output["registry_state"] == "planned/pending"
    assert output["c3_disposition"] == "defer_pending_conditions"
    assert output["maintainer_question_count"] == 9
    assert output["capture_manifest_sha256"] == EXPECTED_MANIFEST_HASH
    assert output["registry_changes_authorized"] is False
    assert output["acquisition_authorized"] is False
    assert output["account_creation_authorized"] is False
    assert output["terms_acceptance_authorized"] is False
    assert output["outbound_contact_authorized"] is False


def test_cli_missing_and_invalid_input_fail_without_traceback(tmp_path: Path) -> None:
    missing = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(tmp_path / "missing.yaml")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing.returncode == 1
    assert "Input file not found" in missing.stderr
    assert "Traceback" not in missing.stderr

    bad = tmp_path / "bad.yaml"
    bad.write_text("schema_version: [")
    invalid = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            str(bad),
            str(MANIFEST_PATH),
            str(EVIDENCE_PATH),
            str(RECOMMENDATIONS_PATH),
            str(REGISTRY_PATH),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert invalid.returncode == 1
    assert "Clarification Validation Failed" in invalid.stderr
    assert "Traceback" not in invalid.stderr
