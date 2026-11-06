"""Comprehensive test suite for VoiceShield Stage C2 rights-evidence dossiers and track segregation."""

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

from voiceshield.data.corpus_registry import (
    AllowedPurpose,
    CorpusRegistry,
    DataState,
    RightsDecision,
    load_corpus_registry,
)
from voiceshield.data.corpus_rights_evidence import (
    EVIDENCE_SCHEMA_VERSION,
    AccessMode,
    CorpusEvidenceCrossValidationError,
    CorpusEvidenceDossier,
    CorpusEvidenceValidationError,
    CorpusRightsEvidence,
    EvidenceSignal,
    EvidenceSource,
    SignalAssessment,
    compute_evidence_bytes_hash,
    compute_evidence_file_hash,
    compute_evidence_hash,
    cross_validate_evidence_and_registry,
    load_corpus_rights_evidence,
    normalize_source_url,
    verify_corpus_rights_evidence_compliance,
)

EVIDENCE_CONFIG_PATH = Path("configs/corpus_rights_evidence.yaml")
REGISTRY_CONFIG_PATH = Path("configs/corpus_registry.yaml")
SCRIPT_PATH = Path("scripts/validate_corpus_rights_evidence.py")
DOSSIERS_DOC_PATH = Path("docs/corpus_rights_review_dossiers.md")


def create_minimal_valid_dossier(
    corpus_id: str = "test_corpus",
    declared_license_spdx: str = "ODC-By-1.0",
    access_mode: AccessMode = AccessMode.PUBLIC_DIRECT,
) -> CorpusEvidenceDossier:
    """Helper fixture to create a valid CorpusEvidenceDossier."""
    source1 = EvidenceSource(
        source_id="src_official_page",
        url="https://example.com/corpus",
        title="Official Corpus Landing Page",
        revision_ref=None,
        effective_date=None,
        content_sha256=None,
        immutable_capture_required=True,
        notes="Official distribution page. Immutable capture required prior to project decisions.",
    )
    source2 = EvidenceSource(
        source_id="src_license_text",
        url="https://example.com/license",
        title="Official License Legal Text",
        revision_ref="v1.0",
        effective_date=None,
        content_sha256=None,
        immutable_capture_required=False,
        notes="Canonical license text.",
    )

    assessment = SignalAssessment(
        signal=EvidenceSignal.POTENTIALLY_COMPATIBLE_NOT_REVIEWED,
        rationale="Source terms disclose potentially compatible conditions, but formal review is pending.",
        evidence_source_ids=("src_official_page", "src_license_text"),
        is_not_project_rights_decision=True,
    )

    return CorpusEvidenceDossier(
        corpus_id=corpus_id,
        evidence_as_of=date(2026, 11, 3),
        declared_license_spdx=declared_license_spdx,
        official_release_label="v1.0",
        access_mode=access_mode,
        source_revision="v1.0",
        official_sources=(source1, source2),
        research_evaluation=assessment,
        research_training=assessment,
        product_evaluation=assessment,
        product_training=assessment,
        raw_redistribution=assessment,
        attribution=assessment,
        access_and_account_obligations="Public direct download with no authentication requirement.",
        data_subject_privacy_considerations="Speech data with participant consent disclosed by authors.",
        data_minimization_requirements="Extract only audio waveforms and language tags.",
        unresolved_questions=("Are underlying audio copyrights cleared by original authors?",),
        research_product_segregation_notes="Research and product tracks remain strictly pending.",
        conservative_summary="Conservative test dossier summary.",
    )


# =========================================================================
# 1. Structural Validation & YAML Loading Tests
# =========================================================================


def test_valid_evidence_configuration() -> None:
    """Verify that configs/corpus_rights_evidence.yaml loads cleanly and passes all invariants."""
    assert EVIDENCE_CONFIG_PATH.is_file(), f"Config file not found at {EVIDENCE_CONFIG_PATH}"
    evidence = load_corpus_rights_evidence(EVIDENCE_CONFIG_PATH)

    assert evidence.schema_version == EVIDENCE_SCHEMA_VERSION
    assert len(evidence.dossiers) == 4

    expected_ids = {"asvspoof5", "mlaad_v9", "common_voice_18", "indicvoices"}
    found_ids = {d.corpus_id for d in evidence.dossiers}
    assert found_ids == expected_ids

    for dossier in evidence.dossiers:
        assert dossier.evidence_as_of == date(2026, 11, 3)
        assert len(dossier.declared_license_spdx.strip()) > 0
        assert dossier.access_mode in AccessMode
        assert len(dossier.official_sources) >= 2
        assert len(dossier.unresolved_questions) >= 1

        # Check all 6 signal assessments
        for sig_name in [
            "research_evaluation",
            "research_training",
            "product_evaluation",
            "product_training",
            "raw_redistribution",
            "attribution",
        ]:
            assess: SignalAssessment = getattr(dossier, sig_name)
            assert assess.signal in EvidenceSignal
            assert len(assess.rationale.strip()) > 0
            assert len(assess.evidence_source_ids) >= 1
            assert assess.is_not_project_rights_decision is True


def test_cross_validation_with_registry() -> None:
    """Verify cross-validation between configs/corpus_rights_evidence.yaml and configs/corpus_registry.yaml."""
    evidence, registry, result = verify_corpus_rights_evidence_compliance(
        EVIDENCE_CONFIG_PATH, REGISTRY_CONFIG_PATH
    )
    assert result.is_valid is True
    assert result.checked_dossiers_count == 4
    assert result.errors == ()
    assert result.evidence_corpus_ids == ("asvspoof5", "common_voice_18", "indicvoices", "mlaad_v9")
    assert result.registry_corpus_ids == ("asvspoof5", "common_voice_18", "indicvoices", "mlaad_v9")


# =========================================================================
# 2. Strict Model & Schema Rejection Tests
# =========================================================================


def test_extra_fields_forbidden() -> None:
    """Verify extra/unknown fields are strictly forbidden on all models."""
    base_dossier = create_minimal_valid_dossier()

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CorpusRightsEvidence(
            schema_version="1.0.0",
            dossiers=(base_dossier,),
            unknown_root_field="malicious_payload",  # type: ignore[call-arg]
        )

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EvidenceSource(
            source_id="src_extra",
            url="https://example.com",
            title="Extra",
            immutable_capture_required=True,
            notes="Note",
            unrecognized_field=123,  # type: ignore[call-arg]
        )

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SignalAssessment(
            signal=EvidenceSignal.UNCLEAR,
            rationale="Rationale",
            evidence_source_ids=("src_official_page",),
            is_not_project_rights_decision=True,
            approved=True,  # type: ignore[call-arg]
        )


def test_duplicate_dossier_rejection() -> None:
    """Verify duplicate dossier corpus_id entries in evidence are rejected."""
    d1 = create_minimal_valid_dossier(corpus_id="dup_id")
    d2 = create_minimal_valid_dossier(corpus_id="dup_id")

    with pytest.raises(ValidationError, match="Duplicate dossier for corpus_id 'dup_id' detected"):
        CorpusRightsEvidence(schema_version="1.0.0", dossiers=(d1, d2))


def test_duplicate_source_rejection() -> None:
    """Verify duplicate source_id within a dossier is rejected."""
    base = create_minimal_valid_dossier().model_dump()
    src_dup = [
        {
            "source_id": "duplicate_src",
            "url": "https://example.com/1",
            "title": "Source 1",
            "immutable_capture_required": True,
            "notes": "Note 1",
        },
        {
            "source_id": "duplicate_src",
            "url": "https://example.com/2",
            "title": "Source 2",
            "immutable_capture_required": True,
            "notes": "Note 2",
        },
    ]
    base["official_sources"] = src_dup
    with pytest.raises(ValidationError, match="duplicate official source_id 'duplicate_src'"):
        CorpusEvidenceDossier.model_validate(base)


def test_duplicate_source_url_rejection() -> None:
    """Verify duplicate source URLs within a dossier are rejected even when source IDs differ."""
    base = create_minimal_valid_dossier().model_dump()
    src_dup_url = [
        {
            "source_id": "src_one",
            "url": "https://example.com/same-url",
            "title": "Source 1",
            "immutable_capture_required": True,
            "notes": "Note 1",
        },
        {
            "source_id": "src_two",
            "url": "https://example.com/same-url",
            "title": "Source 2",
            "immutable_capture_required": True,
            "notes": "Note 2",
        },
    ]
    base["official_sources"] = src_dup_url
    with pytest.raises(ValidationError, match="duplicate official source URL"):
        CorpusEvidenceDossier.model_validate(base)


def test_normalize_source_url() -> None:
    """Verify normalize_source_url handles scheme/host case, port 443, and trailing slashes."""
    # Host and scheme lowercasing
    assert normalize_source_url("HTTPS://EXAMPLE.COM/path") == "https://example.com/path"

    # Default HTTPS port 443 stripped
    assert normalize_source_url("https://example.com:443/path") == "https://example.com/path"

    # Non-standard port preserved
    assert normalize_source_url("https://example.com:8443/path") == "https://example.com:8443/path"

    # Trailing slash policy: trailing slashes stripped from path
    assert normalize_source_url("https://example.com/") == "https://example.com"
    assert normalize_source_url("https://example.com/path/") == "https://example.com/path"
    assert normalize_source_url("https://example.com/path") == "https://example.com/path"
    assert (
        normalize_source_url("https://example.com/path/?q=1#frag")
        == "https://example.com/path?q=1#frag"
    )


def test_duplicate_normalized_source_url_rejection() -> None:
    """Verify source URLs differing only in case, port 443, or trailing slashes are rejected as duplicates."""
    base = create_minimal_valid_dossier().model_dump()

    # Differing only in trailing slash: /path/ vs /path
    dup_slash = [
        {
            "source_id": "src_one",
            "url": "https://example.com/path/",
            "title": "Source 1",
            "immutable_capture_required": True,
            "notes": "Note 1",
        },
        {
            "source_id": "src_two",
            "url": "https://example.com/path",
            "title": "Source 2",
            "immutable_capture_required": True,
            "notes": "Note 2",
        },
    ]
    base["official_sources"] = dup_slash
    with pytest.raises(ValidationError, match="duplicate official source URL"):
        CorpusEvidenceDossier.model_validate(base)

    # Differing only in host casing and default port 443
    dup_case_port = [
        {
            "source_id": "src_one",
            "url": "https://EXAMPLE.COM:443/path",
            "title": "Source 1",
            "immutable_capture_required": True,
            "notes": "Note 1",
        },
        {
            "source_id": "src_two",
            "url": "https://example.com/path",
            "title": "Source 2",
            "immutable_capture_required": True,
            "notes": "Note 2",
        },
    ]
    base["official_sources"] = dup_case_port
    with pytest.raises(ValidationError, match="duplicate official source URL"):
        CorpusEvidenceDossier.model_validate(base)


def test_blank_release_and_revision_fields_rejection() -> None:
    """Verify blank or whitespace-only optional release and revision fields are rejected."""
    base = create_minimal_valid_dossier().model_dump()

    # Blank official_release_label
    with pytest.raises(
        ValidationError, match="official_release_label must be a non-empty string when provided"
    ):
        CorpusEvidenceDossier.model_validate({**base, "official_release_label": "   "})

    # Blank source_revision
    with pytest.raises(
        ValidationError, match="source_revision must be a non-empty string when provided"
    ):
        CorpusEvidenceDossier.model_validate({**base, "source_revision": "   "})

    # Blank revision_ref on EvidenceSource
    with pytest.raises(
        ValidationError, match="revision_ref must be a non-empty string when provided"
    ):
        EvidenceSource(
            source_id="src_blank_rev",
            url="https://example.com",
            title="Title",
            revision_ref="   ",
            immutable_capture_required=True,
            notes="Note",
        )


def test_source_revision_must_match_official_source_revision_ref() -> None:
    """Verify that a non-null dossier source_revision must equal at least one non-null official-source revision_ref."""
    base = create_minimal_valid_dossier().model_dump()
    # Dossier has source_revision="v2.0", but sources only provide revision_ref="v1.0" or None
    base["source_revision"] = "v2.0"

    with pytest.raises(
        ValidationError,
        match="must equal at least one non-null official-source revision_ref",
    ):
        CorpusEvidenceDossier.model_validate(base)

    # When source_revision is None, validation passes even if sources have revisions
    base["source_revision"] = None
    d = CorpusEvidenceDossier.model_validate(base)
    assert d.source_revision is None


def test_effective_date_later_than_evidence_as_of_rejection() -> None:
    """Verify that any official source effective_date later than the dossier evidence_as_of is rejected."""
    base = create_minimal_valid_dossier().model_dump()
    base["evidence_as_of"] = date(2026, 11, 3)
    # Source has effective date in future (2026-11-04 > 2026-11-03)
    base["official_sources"][0]["effective_date"] = date(2026, 11, 4)

    with pytest.raises(ValidationError, match="later than dossier evidence_as_of"):
        CorpusEvidenceDossier.model_validate(base)


def test_duplicate_unresolved_questions_rejection() -> None:
    """Verify duplicate unresolved questions after trimming are rejected."""
    base = create_minimal_valid_dossier().model_dump()
    base["unresolved_questions"] = ["Are rights cleared?", "  Are rights cleared?  "]

    with pytest.raises(ValidationError, match="Duplicate unresolved question detected"):
        CorpusEvidenceDossier.model_validate(base)


def test_duplicate_yaml_key_rejection(tmp_path: Path) -> None:
    """Verify duplicate YAML keys in evidence file are rejected by custom loader."""
    dup_yaml = tmp_path / "dup.yaml"
    dup_yaml.write_text("schema_version: '1.0.0'\nschema_version: '1.0.0'\ndossiers: []\n")
    with pytest.raises(CorpusEvidenceValidationError, match="found duplicate YAML key"):
        load_corpus_rights_evidence(dup_yaml)


def test_malformed_and_non_https_urls() -> None:
    """Verify adversarial, non-HTTPS, or relative URLs are rejected in EvidenceSource."""
    bad_urls = [
        "https://bad host",
        "https://?",
        "https://#",
        "https:///path",
        "https://user:password@example.com",
        "http://example.com",
        "ftp://ftp.example.com",
        "relative/path",
        "/absolute/path",
        "   ",
    ]
    for bad_url in bad_urls:
        with pytest.raises(ValidationError):
            EvidenceSource(
                source_id="bad_src",
                url=bad_url,
                title="Bad URL Source",
                immutable_capture_required=True,
                notes="Note",
            )


def test_strict_scalar_types() -> None:
    """Verify strict scalar validation: disallowing boolean-to-integer and integer-to-boolean coercion."""
    # immutable_capture_required must not coerce integers or strings
    with pytest.raises(
        ValidationError, match="immutable_capture_required must be a strict boolean"
    ):
        EvidenceSource(
            source_id="src_bool",
            url="https://example.com",
            title="Title",
            immutable_capture_required=1,  # type: ignore[arg-type]
            notes="Note",
        )

    with pytest.raises(
        ValidationError, match="immutable_capture_required must be a strict boolean"
    ):
        EvidenceSource(
            source_id="src_bool",
            url="https://example.com",
            title="Title",
            immutable_capture_required="true",  # type: ignore[arg-type]
            notes="Note",
        )

    # is_not_project_rights_decision must strictly be True
    with pytest.raises(
        ValidationError, match="is_not_project_rights_decision must be strictly True"
    ):
        SignalAssessment(
            signal=EvidenceSignal.UNCLEAR,
            rationale="Note",
            evidence_source_ids=("src_1",),
            is_not_project_rights_decision=False,  # type: ignore[arg-type]
        )


def test_deep_immutability() -> None:
    """Verify nested containers in dossiers and evidence are immutable tuples."""
    dossier = create_minimal_valid_dossier()
    evidence = CorpusRightsEvidence(schema_version="1.0.0", dossiers=(dossier,))

    # 1. Evidence dossiers container
    assert isinstance(evidence.dossiers, tuple)
    with pytest.raises(AttributeError):
        evidence.dossiers.append(dossier)  # type: ignore[attr-defined]

    # 2. Official sources container
    assert isinstance(dossier.official_sources, tuple)
    with pytest.raises(AttributeError):
        dossier.official_sources.append(dossier.official_sources[0])  # type: ignore[attr-defined]

    # 3. Unresolved questions container
    assert isinstance(dossier.unresolved_questions, tuple)
    with pytest.raises(AttributeError):
        dossier.unresolved_questions.append("New question?")  # type: ignore[attr-defined]

    # 4. Evidence source IDs container
    assert isinstance(dossier.research_evaluation.evidence_source_ids, tuple)
    with pytest.raises(AttributeError):
        dossier.research_evaluation.evidence_source_ids.append("injected_src")  # type: ignore[attr-defined]

    # Pydantic accepts YAML lists and casts to immutable tuples
    d_dict = dossier.model_dump()
    d_dict["unresolved_questions"] = ["Question 1", "Question 2"]
    d_parsed = CorpusEvidenceDossier.model_validate(d_dict)
    assert isinstance(d_parsed.unresolved_questions, tuple)


def test_unresolved_question_requirement() -> None:
    """Verify that dossiers without unresolved questions are rejected."""
    base = create_minimal_valid_dossier().model_dump()
    base["unresolved_questions"] = []

    with pytest.raises(
        ValidationError, match="unresolved_questions must contain at least one question"
    ):
        CorpusEvidenceDossier.model_validate(base)

    base["unresolved_questions"] = ["   "]
    with pytest.raises(ValidationError, match="must be a non-empty string"):
        CorpusEvidenceDossier.model_validate(base)


def test_minimum_official_source_requirement() -> None:
    """Verify that dossiers must cite at least two distinct official evidence sources."""
    base = create_minimal_valid_dossier().model_dump()
    # Only 1 source
    base["official_sources"] = [base["official_sources"][0]]
    for sig in [
        "research_evaluation",
        "research_training",
        "product_evaluation",
        "product_training",
        "raw_redistribution",
        "attribution",
    ]:
        base[sig]["evidence_source_ids"] = ["src_official_page"]

    with pytest.raises(
        ValidationError,
        match="Every evidence dossier must cite at least two distinct official evidence sources",
    ):
        CorpusEvidenceDossier.model_validate(base)


def test_invalid_evidence_references() -> None:
    """Verify that referencing a non-existent source_id in a signal assessment fails validation."""
    base = create_minimal_valid_dossier().model_dump()
    base["research_evaluation"]["evidence_source_ids"] = ["nonexistent_source_id"]

    with pytest.raises(
        ValidationError, match="references unknown source_id 'nonexistent_source_id'"
    ):
        CorpusEvidenceDossier.model_validate(base)


def test_capture_policy_consistency() -> None:
    """Verify coherent capture-state semantics: immutable_capture_required == (content_sha256 is null and revision_ref is null)."""
    # 1. content_sha256 is None and revision_ref is None, but immutable_capture_required is False -> REJECT
    with pytest.raises(ValidationError, match="immutable_capture_required must be True"):
        EvidenceSource(
            source_id="src_bad_policy",
            url="https://example.com",
            title="Title",
            content_sha256=None,
            revision_ref=None,
            immutable_capture_required=False,
            notes="Note",
        )

    # 2. When revision_ref is provided, immutable_capture_required must be False (True is REJECTED)
    with pytest.raises(ValidationError, match="immutable_capture_required must be False"):
        EvidenceSource(
            source_id="src_bad_policy2",
            url="https://example.com",
            title="Title",
            content_sha256=None,
            revision_ref="commit-12345",
            immutable_capture_required=True,
            notes="Note",
        )

    # 3. When content_sha256 is provided, immutable_capture_required must be False (True is REJECTED)
    with pytest.raises(ValidationError, match="immutable_capture_required must be False"):
        EvidenceSource(
            source_id="src_bad_policy3",
            url="https://example.com",
            title="Title",
            content_sha256="a" * 64,
            revision_ref=None,
            immutable_capture_required=True,
            notes="Note",
        )

    # 4. Valid pinned revision
    src_versioned = EvidenceSource(
        source_id="src_versioned",
        url="https://example.com/v1",
        title="Versioned Source",
        revision_ref="commit-12345",
        content_sha256=None,
        immutable_capture_required=False,
        notes="Note",
    )
    assert src_versioned.revision_ref == "commit-12345"
    assert src_versioned.immutable_capture_required is False


# =========================================================================
# 3. Cross-Validation Gate Failure & Re-validation Hardening Tests
# =========================================================================


def test_exact_corpus_set_equality_failure() -> None:
    """Verify cross-validation fails if evidence and registry corpus ID sets diverge."""
    evidence = load_corpus_rights_evidence(EVIDENCE_CONFIG_PATH)
    registry = load_corpus_registry(REGISTRY_CONFIG_PATH)

    # 1. Missing corpus in evidence
    subset_evidence = CorpusRightsEvidence(
        schema_version="1.0.0",
        dossiers=evidence.dossiers[:3],  # only 3 of 4
    )
    res1 = cross_validate_evidence_and_registry(subset_evidence, registry)
    assert not res1.is_valid
    assert any("Missing in evidence" in err for err in res1.errors)

    # 2. Extra corpus in evidence
    extra_dossier = create_minimal_valid_dossier(corpus_id="extra_corpus")
    superset_evidence = CorpusRightsEvidence(
        schema_version="1.0.0",
        dossiers=(*evidence.dossiers, extra_dossier),
    )
    res2 = cross_validate_evidence_and_registry(superset_evidence, registry)
    assert not res2.is_valid
    assert any("Extra in evidence" in err for err in res2.errors)


def test_license_mismatch_rejection() -> None:
    """Verify cross-validation fails if dossier license differs from registry declared license."""
    evidence = load_corpus_rights_evidence(EVIDENCE_CONFIG_PATH)
    registry = load_corpus_registry(REGISTRY_CONFIG_PATH)

    # Modify license of asvspoof5 in evidence
    modified_dossiers = []
    for d in evidence.dossiers:
        if d.corpus_id == "asvspoof5":
            m_dict = d.model_dump()
            m_dict["declared_license_spdx"] = "MIT"
            modified_dossiers.append(CorpusEvidenceDossier.model_validate(m_dict))
        else:
            modified_dossiers.append(d)

    bad_evidence = CorpusRightsEvidence(schema_version="1.0.0", dossiers=tuple(modified_dossiers))
    res = cross_validate_evidence_and_registry(bad_evidence, registry)
    assert not res.is_valid
    assert any("asvspoof5" in err and "license mismatch" in err for err in res.errors)


def test_non_pending_registry_rejection() -> None:
    """Verify cross-validation fails if any linked registry entry is not 'pending'."""
    evidence = load_corpus_rights_evidence(EVIDENCE_CONFIG_PATH)
    registry_dict = load_corpus_registry(REGISTRY_CONFIG_PATH).model_dump()

    # Change first entry rights_decision to 'approved'
    registry_dict["corpora"][0]["rights_decision"] = RightsDecision.APPROVED
    registry_dict["corpora"][0]["rights_reviewed_at"] = date(2026, 1, 1)
    registry_dict["corpora"][0]["rights_reviewed_by"] = "Auditor"

    mod_reg = CorpusRegistry.model_validate(registry_dict)
    res = cross_validate_evidence_and_registry(evidence, mod_reg)
    assert not res.is_valid
    assert any("must remain 'pending'" in err for err in res.errors)


def test_non_planned_registry_rejection() -> None:
    """Verify cross-validation fails if any linked registry entry is not 'planned'."""
    evidence = load_corpus_rights_evidence(EVIDENCE_CONFIG_PATH)
    registry_dict = load_corpus_registry(REGISTRY_CONFIG_PATH).model_dump()

    # Change entry to 'acquired' with required fields
    registry_dict["corpora"][0]["data_state"] = DataState.ACQUIRED
    registry_dict["corpora"][0]["version_ref"] = "v1"
    registry_dict["corpora"][0]["rights_decision"] = RightsDecision.APPROVED
    registry_dict["corpora"][0]["rights_reviewed_at"] = date(2026, 1, 1)
    registry_dict["corpora"][0]["rights_reviewed_by"] = "Auditor"
    registry_dict["corpora"][0]["archive_sha256"] = "a" * 64

    mod_reg = CorpusRegistry.model_validate(registry_dict)
    res = cross_validate_evidence_and_registry(evidence, mod_reg)
    assert not res.is_valid
    assert any("must remain 'planned'" in err for err in res.errors)


def test_non_empty_allowed_purposes_rejection() -> None:
    """Verify cross-validation fails if any registry entry has non-empty allowed_purposes."""
    evidence = load_corpus_rights_evidence(EVIDENCE_CONFIG_PATH)
    registry_dict = load_corpus_registry(REGISTRY_CONFIG_PATH).model_dump()

    # Set rights_decision=approved (required to have allowed_purposes)
    registry_dict["corpora"][0]["rights_decision"] = RightsDecision.APPROVED
    registry_dict["corpora"][0]["rights_reviewed_at"] = date(2026, 1, 1)
    registry_dict["corpora"][0]["rights_reviewed_by"] = "Auditor"
    registry_dict["corpora"][0]["allowed_purposes"] = [AllowedPurpose.RESEARCH_EVALUATION]

    mod_reg = CorpusRegistry.model_validate(registry_dict)
    res = cross_validate_evidence_and_registry(evidence, mod_reg)
    assert not res.is_valid
    assert any("All allowed_purposes must remain empty" in err for err in res.errors)


def test_fabricated_review_and_integrity_metadata_rejection() -> None:
    """Verify cross-validation fails if review or integrity metadata appears prematurely in registry."""
    evidence = load_corpus_rights_evidence(EVIDENCE_CONFIG_PATH)

    # 1. Fabricated reviewer name
    reg_dict = load_corpus_registry(REGISTRY_CONFIG_PATH).model_dump()
    reg_dict["corpora"][0]["rights_reviewed_by"] = "Unauthorized Reviewer"
    res1 = cross_validate_evidence_and_registry(evidence, CorpusRegistry.model_validate(reg_dict))
    assert not res1.is_valid
    assert any("Reviewer identity must remain null" in err for err in res1.errors)

    # 2. Fabricated review date
    reg_dict2 = load_corpus_registry(REGISTRY_CONFIG_PATH).model_dump()
    reg_dict2["corpora"][0]["rights_reviewed_at"] = date(2026, 1, 1)
    res2 = cross_validate_evidence_and_registry(evidence, CorpusRegistry.model_validate(reg_dict2))
    assert not res2.is_valid
    assert any("Review date must remain null" in err for err in res2.errors)


def test_adversarial_model_copy_bypass_rejection() -> None:
    """Verify cross_validate_evidence_and_registry fails closed against model_copy or model_construct bypasses."""
    evidence = load_corpus_rights_evidence(EVIDENCE_CONFIG_PATH)
    registry = load_corpus_registry(REGISTRY_CONFIG_PATH)

    # 1. model_construct bypassing EvidenceSource URL validator
    invalid_src = EvidenceSource.model_construct(
        source_id="invalid_src",
        url="http://insecure-http-url.com",  # HTTP rejected by _validate_https_url
        title="Insecure Source",
        revision_ref=None,
        effective_date=None,
        content_sha256=None,
        immutable_capture_required=True,
        notes="Constructed invalid source.",
    )
    bypassed_dossier = evidence.dossiers[0].model_copy(
        update={"official_sources": (invalid_src, evidence.dossiers[0].official_sources[1])}
    )
    bypassed_evidence = evidence.model_copy(
        update={"dossiers": (bypassed_dossier, *evidence.dossiers[1:])}
    )

    res = cross_validate_evidence_and_registry(bypassed_evidence, registry)
    assert not res.is_valid
    assert any("Evidence graph re-validation failed" in err for err in res.errors)

    # 2. model_copy on registry causing license mismatch
    invalid_entry = registry.corpora[0].model_copy(
        update={"declared_license_spdx": "MALICIOUS-LICENSE"}
    )
    bypassed_registry = registry.model_copy(
        update={"corpora": (invalid_entry, *registry.corpora[1:])}
    )

    res2 = cross_validate_evidence_and_registry(evidence, bypassed_registry)
    assert not res2.is_valid
    assert any("license mismatch" in err for err in res2.errors)


def test_adversarial_injected_source_id_bypass_rejection() -> None:
    """Verify cross_validate_evidence_and_registry fails closed when model_copy injects a nonexistent evidence_source_id."""
    evidence = load_corpus_rights_evidence(EVIDENCE_CONFIG_PATH)
    registry = load_corpus_registry(REGISTRY_CONFIG_PATH)

    # Injected unknown evidence_source_id into SignalAssessment bypassing validator
    injected_assess = evidence.dossiers[0].research_evaluation.model_copy(
        update={"evidence_source_ids": ("completely_unknown_injected_source_id",)}
    )
    bypassed_dossier = evidence.dossiers[0].model_copy(
        update={"research_evaluation": injected_assess}
    )
    bypassed_evidence = evidence.model_copy(
        update={"dossiers": (bypassed_dossier, *evidence.dossiers[1:])}
    )

    res = cross_validate_evidence_and_registry(bypassed_evidence, registry)
    assert not res.is_valid
    assert any("Evidence graph re-validation failed" in err for err in res.errors)
    assert any(
        "unknown source_id 'completely_unknown_injected_source_id'" in err for err in res.errors
    )


# =========================================================================
# 4. Domain-Specific Conservative Facts & Pinned Revision Tests (Revision 3)
# =========================================================================


def test_asvspoof_factual_correctness_revision_3() -> None:
    """Verify ASVspoof 5 facts: exact title, §2.2 / §2.4 Database-vs-Contents distinction, no participation rule claims."""
    evidence = load_corpus_rights_evidence(EVIDENCE_CONFIG_PATH)
    asv = next(d for d in evidence.dossiers if d.corpus_id == "asvspoof5")

    # Exact title, record ID, and publication date
    zenodo_src = next(s for s in asv.official_sources if s.source_id == "zenodo_record_14498691")
    assert zenodo_src.title == (
        "ASVspoof 5: Design, Collection and Validation of Resources for Spoofing, "
        "Deepfake, and Adversarial Attack Detection Using Crowdsourced Speech"
    )
    assert zenodo_src.revision_ref == "14498691"
    assert zenodo_src.effective_date == date(2024, 12, 18)
    assert zenodo_src.immutable_capture_required is False

    # License and access mode
    assert asv.declared_license_spdx == "ODC-By-1.0"
    assert asv.official_release_label == "v0.1"
    assert asv.access_mode == AccessMode.PUBLIC_DIRECT
    assert asv.source_revision == "14498691"

    # Precise Database-versus-Contents characterization (§2.2 vs §2.4)
    odc_src = next(s for s in asv.official_sources if s.source_id == "odc_by_1_0_license_text")
    assert "2.2" in odc_src.notes
    assert "2.4" in odc_src.notes
    assert "Section 2.2" in asv.conservative_summary
    assert "Section 2.4" in asv.conservative_summary
    assert "does not cover rights in individual Contents" in asv.conservative_summary
    assert "Section 4.2" in asv.attribution.rationale
    assert "Section 4.3" in asv.attribution.rationale

    # Ensure "licenses only the database structure" phrase was replaced
    assert "licenses only the database structure" not in asv.conservative_summary.lower()
    assert "licenses only the database structure" not in odc_src.notes.lower()

    # Audio copyright, privacy, and consent remain unresolved
    assert "unresolved" in asv.conservative_summary.lower()
    assert any("Database vs. Contents" in q for q in asv.unresolved_questions)

    # Ensure unsupported challenge participation rule assertions are absent
    assert "challenge participation" not in asv.access_and_account_obligations.lower()
    assert "challenge participation" not in asv.conservative_summary.lower()


def test_mlaad_factual_correctness_revision_3() -> None:
    """Verify MLAAD v9 facts: account_required, pinned commit, CC BY-NC §1(i) and §2(a)(1), no 'web-scraped' claims."""
    evidence = load_corpus_rights_evidence(EVIDENCE_CONFIG_PATH)
    mlaad = next(d for d in evidence.dossiers if d.corpus_id == "mlaad_v9")

    assert mlaad.declared_license_spdx == "CC-BY-NC-4.0"
    assert mlaad.access_mode == AccessMode.ACCOUNT_REQUIRED
    assert mlaad.source_revision == "30c3dec763fa4803f11c2df1557d3b0828395366"

    hf_src = next(s for s in mlaad.official_sources if s.source_id == "mlaad_huggingface_card")
    assert hf_src.revision_ref == "30c3dec763fa4803f11c2df1557d3b0828395366"
    assert hf_src.immutable_capture_required is False

    # Prohibited signals citing Section 1(i) and Section 2(a)(1)
    assert mlaad.product_evaluation.signal == EvidenceSignal.SOURCE_DECLARES_PROHIBITED
    assert mlaad.product_training.signal == EvidenceSignal.SOURCE_DECLARES_PROHIBITED
    assert "1(i)" in mlaad.product_evaluation.rationale
    assert "2(a)(1)" in mlaad.product_evaluation.rationale
    assert "1(i)" in mlaad.product_training.rationale
    assert "2(a)(1)" in mlaad.product_training.rationale
    assert "4.2" not in mlaad.product_evaluation.rationale
    assert "4.2" not in mlaad.product_training.rationale

    # Absolute absence of "web-scraped" or "web scraped" in YAML and documentation
    full_yaml_text = EVIDENCE_CONFIG_PATH.read_text().lower()
    full_doc_text = DOSSIERS_DOC_PATH.read_text().lower()
    assert "web-scraped" not in full_yaml_text
    assert "web scraped" not in full_yaml_text
    assert "web-scraped" not in full_doc_text
    assert "web scraped" not in full_doc_text

    # No project conclusion that M-AILABS is public domain
    assert "public domain" not in mlaad.data_subject_privacy_considerations.lower()
    assert "external bona fide dependency" in mlaad.data_subject_privacy_considerations.lower()
    assert "component-level provenance" in mlaad.data_subject_privacy_considerations.lower()

    # Conditional product exclusion rather than permanent prohibition
    assert "permanently prohibited" not in mlaad.research_product_segregation_notes.lower()
    assert "excluded" in mlaad.research_product_segregation_notes.lower()


def test_common_voice_factual_correctness_revision_3() -> None:
    """Verify Common Voice 18 facts: CC0 limited effect, CV terms ask not to mirror, MDC §2(b)(viii) re-identification."""
    evidence = load_corpus_rights_evidence(EVIDENCE_CONFIG_PATH)
    cv = next(d for d in evidence.dossiers if d.corpus_id == "common_voice_18")

    assert cv.declared_license_spdx == "CC0-1.0"
    assert cv.access_mode == AccessMode.ACCOUNT_REQUIRED
    assert cv.official_release_label == "18.0"

    cv_terms = next(s for s in cv.official_sources if s.source_id == "common_voice_terms_of_use")
    assert cv_terms.effective_date == date(2025, 10, 31)
    assert cv_terms.immutable_capture_required is True
    # Common Voice Terms only ask not to post/distribute/mirror; NO re-identification covenant in CV terms
    assert "mirror" in cv_terms.notes.lower() or "post" in cv_terms.notes.lower()
    assert "re-identification" not in cv_terms.notes.lower()
    assert "re-identify" not in cv_terms.notes.lower()

    mdc_terms = next(
        s for s in cv.official_sources if s.source_id == "mdc_consumer_terms_of_service"
    )
    assert mdc_terms.effective_date == date(2026, 10, 28)
    assert mdc_terms.immutable_capture_required is True
    # Binding re-identification restriction attributed to MDC Consumer Terms §2(b)(viii)
    assert "2(b)(viii)" in mdc_terms.notes
    assert "Data Consumer License" in mdc_terms.notes
    assert "individual contributor or source of data" in mdc_terms.notes
    assert "author, contributor, or data subject" not in mdc_terms.notes

    cc0_terms = next(s for s in cv.official_sources if s.source_id == "cc0_1_0_legal_code")
    assert "creativecommons.org/publicdomain/zero/1.0/legalcode.en" in cc0_terms.url
    assert cc0_terms.immutable_capture_required is True

    # Mirroring/redistribution restrictions described together with Data Consumer License exceptions
    assert cv.raw_redistribution.signal == EvidenceSignal.SOURCE_DECLARES_RESTRICTED
    assert "Data Consumer License" in cv.raw_redistribution.rationale

    # CC0 limited effect documented
    assert "affirmer" in cv.research_evaluation.rationale.lower()
    assert (
        "consent" in cv.research_evaluation.rationale.lower()
        or "third-party" in cv.research_evaluation.rationale.lower()
    )
    assert "2(b)(viii)" in cv.conservative_summary


def test_indicvoices_factual_correctness_revision_3() -> None:
    """Verify IndicVoices facts: gated access without research justification claim, deleted videos, default-deny minimization."""
    evidence = load_corpus_rights_evidence(EVIDENCE_CONFIG_PATH)
    iv = next(d for d in evidence.dossiers if d.corpus_id == "indicvoices")

    assert iv.declared_license_spdx == "CC-BY-4.0"
    assert iv.access_mode == AccessMode.GATED_APPROVAL
    assert iv.source_revision == "c96f9088f138cf89d419da7e8e643e1f05c00a87"

    hf_src = next(s for s in iv.official_sources if s.source_id == "indicvoices_huggingface_card")
    assert hf_src.revision_ref == "c96f9088f138cf89d419da7e8e643e1f05c00a87"
    assert hf_src.immutable_capture_required is False

    # Does not claim research justification or academic justification
    full_yaml_text = EVIDENCE_CONFIG_PATH.read_text().lower()
    full_doc_text = DOSSIERS_DOC_PATH.read_text().lower()
    assert "research justification" not in full_yaml_text
    assert "academic justification" not in full_yaml_text
    assert "research justification" not in full_doc_text
    assert "academic justification" not in full_doc_text

    # Describes HF gate as requiring authentication, sharing info, accepting repository conditions
    assert "authentication" in iv.access_and_account_obligations.lower()
    assert "user information" in iv.access_and_account_obligations.lower()
    assert "unseen conditions" in iv.access_and_account_obligations.lower()

    # Preserve source-accurate registration and video handling without asserting sanitized distributed content.
    privacy_text = iv.data_subject_privacy_considerations.lower()
    assert "deleted after verification" in privacy_text
    assert "could not be downloaded or shared" in privacy_text
    assert "signed consent form" in privacy_text
    assert "phone numbers" in privacy_text
    assert "would not be shared with third parties" in privacy_text
    assert "could not be downloaded or shared" in privacy_text
    assert "unsanitized transcript fields" in privacy_text
    assert "no assumption" in privacy_text

    # Audio/language default-deny minimization
    assert "default-deny" in iv.data_minimization_requirements.lower()
    assert "protocol" in iv.data_minimization_requirements.lower()
    assert "no such exception is approved in c2" in iv.data_minimization_requirements.lower()


# =========================================================================
# 5. CLI Execution & Deterministic Output Tests
# =========================================================================


def test_cli_success_and_deterministic_json() -> None:
    """Verify validate_corpus_rights_evidence.py produces deterministic sort_keys=True JSON and exit 0."""
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        str(EVIDENCE_CONFIG_PATH),
        str(REGISTRY_CONFIG_PATH),
    ]
    res1 = subprocess.run(cmd, capture_output=True, text=True, check=False)
    res2 = subprocess.run(cmd, capture_output=True, text=True, check=False)

    assert res1.returncode == 0, f"CLI exited with {res1.returncode}: {res1.stderr}"
    assert res2.returncode == 0

    # Bit-for-bit identical stdout
    assert res1.stdout == res2.stdout

    data = json.loads(res1.stdout)
    assert data["status"] == "valid"
    assert data["schema_version"] == "1.0.0"
    assert data["checked_dossiers_count"] == 4
    assert data["evidence_sha256"] == compute_evidence_file_hash(EVIDENCE_CONFIG_PATH)
    assert (
        data["registry_sha256"]
        == hashlib.sha256(REGISTRY_CONFIG_PATH.read_bytes()).hexdigest().lower()
    )
    assert "counts_by_access_mode" in data
    assert "counts_by_signal" in data
    assert "access_modes" in data


def test_cli_failure_clean_error_no_traceback(tmp_path: Path) -> None:
    """Verify CLI failures emit clean stderr errors with zero Python tracebacks."""
    # 1. Missing evidence file
    res1 = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(tmp_path / "missing.yaml")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res1.returncode != 0
    assert "Evidence file not found" in res1.stderr
    assert "Traceback (most recent call last)" not in res1.stderr

    # 2. Malformed evidence YAML
    bad_yaml = tmp_path / "bad_syntax.yaml"
    bad_yaml.write_text("schema_version: '1.0.0'\ndossiers: [unclosed")
    res2 = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(bad_yaml)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res2.returncode != 0
    assert "Evidence Validation Failed" in res2.stderr
    assert "Traceback (most recent call last)" not in res2.stderr

    # 3. Cross-validation failure
    mismatch_yaml = tmp_path / "mismatch.yaml"
    m_data = yaml.safe_load(EVIDENCE_CONFIG_PATH.read_text())
    m_data["dossiers"][0]["declared_license_spdx"] = "GPL-3.0"
    mismatch_yaml.write_text(yaml.dump(m_data))
    res3 = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(mismatch_yaml), str(REGISTRY_CONFIG_PATH)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res3.returncode != 0
    assert "Cross-Validation Failed" in res3.stderr
    assert "license mismatch" in res3.stderr
    assert "Traceback (most recent call last)" not in res3.stderr


def test_no_network_access(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify evidence loading, validation, hashing, cross-validation, and CLI make zero network requests."""

    def guarded_connect(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Network connection attempted during evidence validation!")

    monkeypatch.setattr(socket, "create_connection", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)

    evidence, registry, result = verify_corpus_rights_evidence_compliance(
        EVIDENCE_CONFIG_PATH, REGISTRY_CONFIG_PATH
    )
    assert result.is_valid is True

    # Hashing
    h = compute_evidence_file_hash(EVIDENCE_CONFIG_PATH)
    assert len(h) == 64


# =========================================================================
# 6. Additional Coverage & Boundary Tests
# =========================================================================


def test_bytes_source_and_unsupported_source_loader(tmp_path: Path) -> None:
    """Verify loading from raw bytes, alias hashing, and rejection of invalid source types."""
    raw_bytes = EVIDENCE_CONFIG_PATH.read_bytes()
    evidence = load_corpus_rights_evidence(raw_bytes)
    assert len(evidence.dossiers) == 4

    # Unsupported source type
    with pytest.raises(CorpusEvidenceValidationError, match="Unsupported source type"):
        load_corpus_rights_evidence(12345)  # type: ignore[arg-type]

    # Empty bytes
    with pytest.raises(CorpusEvidenceValidationError, match="empty or contains only whitespace"):
        load_corpus_rights_evidence(b"")

    # Non-existent file hash
    with pytest.raises(FileNotFoundError):
        compute_evidence_file_hash(tmp_path / "nonexistent.yaml")

    # compute_evidence_bytes_hash and canonical alias
    computed_b = compute_evidence_bytes_hash(raw_bytes)
    computed_f = compute_evidence_file_hash(EVIDENCE_CONFIG_PATH)
    computed_alias = compute_evidence_hash(EVIDENCE_CONFIG_PATH)
    assert computed_b == computed_f == computed_alias


def test_field_validators_and_coercions() -> None:
    """Verify type coercion, string trimming, and rejection of invalid inputs across models."""
    base_dossier = create_minimal_valid_dossier().model_dump()

    # Empty title
    with pytest.raises(ValidationError, match="title must be a non-empty string"):
        EvidenceSource(
            source_id="src_test",
            url="https://example.com",
            title="   ",
            immutable_capture_required=True,
            notes="Note",
        )

    # Empty notes
    with pytest.raises(ValidationError, match="notes must be a non-empty string"):
        EvidenceSource(
            source_id="src_test",
            url="https://example.com",
            title="Title",
            immutable_capture_required=True,
            notes="",
        )

    # Invalid source_id regex
    with pytest.raises(ValidationError, match="source_id must be a stable lowercase identifier"):
        EvidenceSource(
            source_id="INVALID_ID!",
            url="https://example.com",
            title="Title",
            immutable_capture_required=True,
            notes="Note",
        )

    # Invalid corpus_id regex
    with pytest.raises(ValidationError, match="corpus_id must be a stable lowercase identifier"):
        CorpusEvidenceDossier.model_validate({**base_dossier, "corpus_id": "Bad_Corpus_ID!"})

    # Empty declared_license_spdx
    with pytest.raises(ValidationError, match="declared_license_spdx must be a non-empty string"):
        CorpusEvidenceDossier.model_validate({**base_dossier, "declared_license_spdx": "  "})

    # Non-sequence official_sources
    with pytest.raises(ValidationError, match="official_sources must be a sequence"):
        CorpusEvidenceDossier.model_validate({**base_dossier, "official_sources": "not_a_seq"})

    # Non-sequence unresolved_questions
    with pytest.raises(ValidationError, match="unresolved_questions must be a sequence"):
        CorpusEvidenceDossier.model_validate({**base_dossier, "unresolved_questions": 1234})

    # Non-sequence evidence_source_ids in SignalAssessment
    with pytest.raises(ValidationError, match="evidence_source_ids must be a sequence of strings"):
        SignalAssessment(
            signal=EvidenceSignal.UNCLEAR,
            rationale="Note",
            evidence_source_ids="not_a_seq",  # type: ignore[arg-type]
            is_not_project_rights_decision=True,
        )

    # Empty string inside evidence_source_ids
    with pytest.raises(ValidationError, match="evidence_source_ids.*must be a non-empty string"):
        SignalAssessment(
            signal=EvidenceSignal.UNCLEAR,
            rationale="Note",
            evidence_source_ids=("",),
            is_not_project_rights_decision=True,
        )

    # Duplicate IDs in evidence_source_ids
    with pytest.raises(ValidationError, match="Duplicate evidence source ID"):
        SignalAssessment(
            signal=EvidenceSignal.UNCLEAR,
            rationale="Note",
            evidence_source_ids=("src1", "src1"),
            is_not_project_rights_decision=True,
        )

    # Empty dossier list in CorpusRightsEvidence
    with pytest.raises(
        ValidationError, match="CorpusRightsEvidence must contain at least one dossier"
    ):
        CorpusRightsEvidence(schema_version="1.0.0", dossiers=())

    # Non-sequence dossiers in CorpusRightsEvidence
    with pytest.raises(ValidationError, match="dossiers must be a sequence"):
        CorpusRightsEvidence(schema_version="1.0.0", dossiers="not_a_seq")  # type: ignore[arg-type]


def test_verify_compliance_raises_cross_validation_error(tmp_path: Path) -> None:
    """Verify verify_corpus_rights_evidence_compliance raises CorpusEvidenceCrossValidationError."""
    bad_yaml = tmp_path / "mismatch.yaml"
    m_data = yaml.safe_load(EVIDENCE_CONFIG_PATH.read_text())
    m_data["dossiers"][0]["declared_license_spdx"] = "APACHE-2.0"
    bad_yaml.write_text(yaml.dump(m_data))

    with pytest.raises(CorpusEvidenceCrossValidationError) as exc_info:
        verify_corpus_rights_evidence_compliance(bad_yaml, REGISTRY_CONFIG_PATH)

    assert "cross-validation failed" in str(exc_info.value)
    assert len(exc_info.value.errors) > 0


def test_unhashable_and_non_dict_yaml_loader(tmp_path: Path) -> None:
    """Verify unhashable keys and non-dict YAML roots are rejected with CorpusEvidenceValidationError."""
    # Non-dict root
    list_yaml = tmp_path / "list_root.yaml"
    list_yaml.write_text("- item1\n- item2\n")
    with pytest.raises(CorpusEvidenceValidationError, match="must be a YAML mapping/dictionary"):
        load_corpus_rights_evidence(list_yaml)

    # Unhashable key
    unhashable_yaml = tmp_path / "unhashable.yaml"
    unhashable_yaml.write_text("{[1, 2]: 'val'}\n")
    with pytest.raises(CorpusEvidenceValidationError, match="unhashable mapping key"):
        load_corpus_rights_evidence(unhashable_yaml)


def test_cross_validation_integrity_metadata_rejection() -> None:
    """Verify cross-validation fails if premature acquired/verified state or non-null integrity metadata appears in registry."""
    evidence = load_corpus_rights_evidence(EVIDENCE_CONFIG_PATH)

    # 1. Non-planned acquired state with archive_sha256
    reg_dict1 = load_corpus_registry(REGISTRY_CONFIG_PATH).model_dump()
    reg_dict1["corpora"][0]["data_state"] = DataState.ACQUIRED
    reg_dict1["corpora"][0]["version_ref"] = "v0.1"
    reg_dict1["corpora"][0]["rights_decision"] = RightsDecision.APPROVED
    reg_dict1["corpora"][0]["rights_reviewed_at"] = date(2026, 1, 1)
    reg_dict1["corpora"][0]["rights_reviewed_by"] = "Auditor"
    reg_dict1["corpora"][0]["archive_sha256"] = "a" * 64

    res1 = cross_validate_evidence_and_registry(evidence, CorpusRegistry.model_validate(reg_dict1))
    assert not res1.is_valid
    assert any("must remain 'planned'" in err for err in res1.errors)
    assert any("Archive hash must remain null" in err for err in res1.errors)

    # 2. Verified state with manifest_sha256 and sample_count
    reg_dict2 = load_corpus_registry(REGISTRY_CONFIG_PATH).model_dump()
    reg_dict2["corpora"][0]["data_state"] = DataState.VERIFIED
    reg_dict2["corpora"][0]["version_ref"] = "v0.1"
    reg_dict2["corpora"][0]["rights_decision"] = RightsDecision.APPROVED
    reg_dict2["corpora"][0]["rights_reviewed_at"] = date(2026, 1, 1)
    reg_dict2["corpora"][0]["rights_reviewed_by"] = "Auditor"
    reg_dict2["corpora"][0]["archive_sha256"] = "a" * 64
    reg_dict2["corpora"][0]["manifest_sha256"] = "b" * 64
    reg_dict2["corpora"][0]["sample_count"] = 1000

    res2 = cross_validate_evidence_and_registry(evidence, CorpusRegistry.model_validate(reg_dict2))
    assert not res2.is_valid
    assert any("Manifest hash must remain null" in err for err in res2.errors)
    assert any("Sample count must remain null" in err for err in res2.errors)
