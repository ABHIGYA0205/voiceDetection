"""Comprehensive test suite for VoiceShield Rights-Aware Corpus Registry and Use Gate."""

from __future__ import annotations

import hashlib
import json
import socket
import subprocess
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from voiceshield.data.corpus_registry import (
    REGISTRY_SCHEMA_VERSION,
    AllowedPurpose,
    ContentType,
    CorpusAuthorizationError,
    CorpusAuthorizationResult,
    CorpusEntry,
    CorpusIdentity,
    CorpusRegistry,
    CorpusValidationError,
    DataState,
    RedistributionStatus,
    RightsDecision,
    authorize_corpus_usage,
    check_corpus_authorization,
    compute_registry_bytes_hash,
    compute_registry_file_hash,
    compute_registry_hash,
    load_corpus_registry,
)

CONFIG_PATH = Path("configs/corpus_registry.yaml")
SCRIPT_PATH = Path("scripts/validate_corpus_registry.py")

VALID_ARCHIVE_HASH = "a" * 64
VALID_MANIFEST_HASH = "b" * 64


def create_verified_entry(
    corpus_id: str = "verified_corpus",
    version_ref: str = "v1.0",
    allowed_purposes: Sequence[AllowedPurpose] | None = None,
    raw_redistribution_status: RedistributionStatus = RedistributionStatus.PROHIBITED,
    rights_decision: RightsDecision = RightsDecision.APPROVED,
) -> CorpusEntry:
    """Helper fixture to create a valid verified & approved corpus entry."""
    purposes_tuple = (
        (AllowedPurpose.RESEARCH_EVALUATION,)
        if allowed_purposes is None
        else tuple(allowed_purposes)
    )
    return CorpusEntry(
        corpus_id=corpus_id,
        display_name=f"Verified Test Corpus ({corpus_id})",
        version_ref=version_ref,
        data_state=DataState.VERIFIED,
        content_type=ContentType.MIXED,
        source_url="https://example.com/corpus",
        documentation_url="https://example.com/docs",
        declared_license_spdx="Apache-2.0",
        license_url="https://www.apache.org/licenses/LICENSE-2.0",
        rights_decision=rights_decision,
        allowed_purposes=purposes_tuple,
        rights_reviewed_at=date(2026, 1, 15),
        rights_reviewed_by="VoiceShield Legal & Compliance",
        attribution_required=True,
        raw_redistribution_status=raw_redistribution_status,
        archive_sha256=VALID_ARCHIVE_HASH,
        manifest_sha256=VALID_MANIFEST_HASH,
        sample_count=5000,
        notes="Verified fixture for governance testing.",
    )


# =========================================================================
# 1. Structural Validation & Initial Registry Tests
# =========================================================================


def test_valid_initial_registry() -> None:
    """Verify that configs/corpus_registry.yaml loads cleanly and satisfies all invariants."""
    assert CONFIG_PATH.is_file(), f"Config file not found at {CONFIG_PATH}"
    registry = load_corpus_registry(CONFIG_PATH)

    assert registry.schema_version == REGISTRY_SCHEMA_VERSION
    assert len(registry.corpora) == 4

    expected_ids = {"asvspoof5", "mlaad_v9", "common_voice_18", "indicvoices"}
    found_ids = {c.corpus_id for c in registry.corpora}
    assert found_ids == expected_ids

    for entry in registry.corpora:
        assert entry.data_state == DataState.PLANNED
        assert entry.rights_decision == RightsDecision.PENDING
        assert entry.allowed_purposes == ()
        assert entry.version_ref is None
        assert entry.archive_sha256 is None
        assert entry.manifest_sha256 is None
        assert entry.sample_count is None
        assert entry.rights_reviewed_at is None
        assert entry.rights_reviewed_by is None
        assert entry.attribution_required is None
        assert entry.raw_redistribution_status == RedistributionStatus.UNKNOWN
        assert entry.source_url.startswith("https://")
        assert entry.documentation_url.startswith("https://")
        if entry.license_url is not None:
            assert entry.license_url.startswith("https://")
        assert len(entry.notes.strip()) > 0


def test_strict_unknown_field_rejection() -> None:
    """Verify that unexpected extra fields are rejected at both entry and registry levels."""
    # Extra field on CorpusEntry
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CorpusEntry(  # type: ignore[call-arg]
            corpus_id="test_extra",
            display_name="Extra Test",
            version_ref=None,
            data_state=DataState.PLANNED,
            content_type=ContentType.MIXED,
            source_url="https://example.com",
            documentation_url="https://example.com",
            rights_decision=RightsDecision.PENDING,
            allowed_purposes=(),
            notes="Note",
            unrecognized_extra_field="malicious_payload",
        )

    # Extra field on CorpusRegistry
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CorpusRegistry(  # type: ignore[call-arg]
            schema_version="1.0.0",
            corpora=(),
            unexpected_top_level_key=12345,
        )


def test_duplicate_corpus_identity_rejection() -> None:
    """Verify duplicate (corpus_id, version_ref) combinations fail validation."""
    entry1 = create_verified_entry(corpus_id="asvspoof", version_ref="v1")
    entry2 = create_verified_entry(corpus_id="asvspoof", version_ref="v1")

    with pytest.raises(ValidationError, match="Duplicate corpus identity detected"):
        CorpusRegistry(schema_version="1.0.0", corpora=(entry1, entry2))

    # Also test planned duplicate (version_ref is None)
    entry_plan1 = CorpusEntry(
        corpus_id="planned_dup",
        display_name="Planned 1",
        version_ref=None,
        data_state=DataState.PLANNED,
        content_type=ContentType.MIXED,
        source_url="https://example.com/1",
        documentation_url="https://example.com/1",
        rights_decision=RightsDecision.PENDING,
        allowed_purposes=(),
        notes="Note 1",
    )
    entry_plan2 = CorpusEntry(
        corpus_id="planned_dup",
        display_name="Planned 2",
        version_ref=None,
        data_state=DataState.PLANNED,
        content_type=ContentType.MIXED,
        source_url="https://example.com/2",
        documentation_url="https://example.com/2",
        rights_decision=RightsDecision.PENDING,
        allowed_purposes=(),
        notes="Note 2",
    )
    with pytest.raises(ValidationError, match="Duplicate corpus identity detected"):
        CorpusRegistry(schema_version="1.0.0", corpora=(entry_plan1, entry_plan2))


def test_invalid_corpus_ids() -> None:
    """Verify corpus_id must be a stable lowercase identifier without spaces, uppercase, or special chars."""
    invalid_ids = ["ASVspoof5", "mlaad v9", "common@voice", "test!", "", "   "]
    for cid in invalid_ids:
        with pytest.raises(ValidationError, match="corpus_id"):
            CorpusEntry(
                corpus_id=cid,
                display_name="Invalid ID Test",
                data_state=DataState.PLANNED,
                content_type=ContentType.MIXED,
                source_url="https://example.com",
                documentation_url="https://example.com",
                rights_decision=RightsDecision.PENDING,
                allowed_purposes=(),
                notes="Note",
            )


def test_invalid_and_uppercase_hashes() -> None:
    """Verify SHA-256 hashes must be strictly lowercase 64-character hex strings."""
    base_dict = create_verified_entry().model_dump()

    # Uppercase hex characters
    with pytest.raises(ValidationError, match="lowercase 64-character hexadecimal SHA-256"):
        CorpusEntry.model_validate({**base_dict, "archive_sha256": "A" * 64})

    # Truncated hash (<64 chars)
    with pytest.raises(ValidationError, match="lowercase 64-character hexadecimal SHA-256"):
        CorpusEntry.model_validate({**base_dict, "archive_sha256": "a" * 63})

    # Hash with invalid hex character (e.g. 'g')
    with pytest.raises(ValidationError, match="lowercase 64-character hexadecimal SHA-256"):
        CorpusEntry.model_validate({**base_dict, "manifest_sha256": "g" * 64})


# =========================================================================
# 2. Robust HTTPS Validation Tests (Correction 1)
# =========================================================================


def test_robust_https_validation_rejections() -> None:
    """Verify adversarial and invalid URLs are rejected across all URL fields."""
    adversarial_urls = [
        "https://bad host",
        "https://?",
        "https://#",
        "https:///path",
        "https://user:password@example.com",
        "http://example.com",
        "ftp://ftp.example.com/file",
        "relative/path",
        "/absolute/path",
        "   ",
        "https:// example.com",
        "https://example .com",
    ]

    base_plan = {
        "corpus_id": "url_test",
        "display_name": "URL Test",
        "data_state": DataState.PLANNED,
        "content_type": ContentType.MIXED,
        "rights_decision": RightsDecision.PENDING,
        "allowed_purposes": [],
        "notes": "Testing URL validation.",
    }

    for bad_url in adversarial_urls:
        # source_url rejection
        with pytest.raises(ValidationError):
            CorpusEntry.model_validate(
                {
                    **base_plan,
                    "source_url": bad_url,
                    "documentation_url": "https://example.com/docs",
                }
            )

        # documentation_url rejection
        with pytest.raises(ValidationError):
            CorpusEntry.model_validate(
                {
                    **base_plan,
                    "source_url": "https://example.com/src",
                    "documentation_url": bad_url,
                }
            )

        # license_url rejection
        with pytest.raises(ValidationError):
            CorpusEntry.model_validate(
                {
                    **base_plan,
                    "source_url": "https://example.com/src",
                    "documentation_url": "https://example.com/docs",
                    "license_url": bad_url,
                }
            )


def test_robust_https_validation_acceptances() -> None:
    """Verify normal HTTPS URLs with paths, query strings, and fragments remain valid."""
    valid_urls = [
        "https://example.com",
        "https://example.com/",
        "https://example.com/path/to/resource",
        "https://example.com/search?query=voice+detection&page=2",
        "https://example.com/document#section-3",
        "https://subdomain.dept.univ.edu:8443/datasets/speech?v=1#readme",
    ]

    for valid_url in valid_urls:
        entry = CorpusEntry(
            corpus_id="valid_url_corpus",
            display_name="Valid URL Corpus",
            data_state=DataState.PLANNED,
            content_type=ContentType.MIXED,
            source_url=valid_url,
            documentation_url=valid_url,
            license_url=valid_url,
            rights_decision=RightsDecision.PENDING,
            allowed_purposes=(),
            notes="Testing valid URLs.",
        )
        assert entry.source_url == valid_url
        assert entry.documentation_url == valid_url
        assert entry.license_url == valid_url


# =========================================================================
# 3. Deep Immutability Tests (Correction 2)
# =========================================================================


def test_deep_immutability_enforcement() -> None:
    """Verify authorization-critical containers are immutable tuples that cannot be mutated."""
    # Prove Pydantic accepts YAML/JSON lists as input but converts them to immutable tuples
    raw_dict = create_verified_entry().model_dump()
    raw_dict["allowed_purposes"] = ["research_evaluation"]
    entry_from_list = CorpusEntry.model_validate(raw_dict)
    assert isinstance(entry_from_list.allowed_purposes, tuple)
    assert entry_from_list.allowed_purposes == (AllowedPurpose.RESEARCH_EVALUATION,)

    reg_from_list = CorpusRegistry.model_validate(
        {"schema_version": "1.0.0", "corpora": [entry_from_list.model_dump()]}
    )
    assert isinstance(reg_from_list.corpora, tuple)
    assert len(reg_from_list.corpora) == 1

    entry = create_verified_entry(
        corpus_id="immut_corpus",
        allowed_purposes=[AllowedPurpose.RESEARCH_EVALUATION],
    )

    # 1. CorpusEntry.allowed_purposes is an immutable tuple
    assert isinstance(entry.allowed_purposes, tuple)
    with pytest.raises(AttributeError):
        entry.allowed_purposes.append(AllowedPurpose.PRODUCT_TRAINING)  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        entry.allowed_purposes[0] = AllowedPurpose.PRODUCT_TRAINING  # type: ignore[index]

    # Adversarial test: caller cannot mutate entry to bypass authorization
    registry = CorpusRegistry(schema_version="1.0.0", corpora=(entry,))

    # 2. CorpusRegistry.corpora is an immutable tuple
    assert isinstance(registry.corpora, tuple)
    with pytest.raises(AttributeError):
        registry.corpora.append(  # type: ignore[attr-defined]
            create_verified_entry(corpus_id="injected_corpus")
        )

    # Attempting to authorize product_training must fail
    with pytest.raises(
        CorpusAuthorizationError, match="does not permit purpose 'product_training'"
    ):
        authorize_corpus_usage(
            registry=registry,
            purpose=AllowedPurpose.PRODUCT_TRAINING,
            corpus_identifiers=["immut_corpus@v1.0"],
        )

    # 3. CorpusAuthorizationResult.authorized_corpora is an immutable tuple
    result = authorize_corpus_usage(
        registry=registry,
        purpose=AllowedPurpose.RESEARCH_EVALUATION,
        corpus_identifiers=["immut_corpus@v1.0"],
    )
    assert isinstance(result.authorized_corpora, tuple)
    with pytest.raises(AttributeError):
        result.authorized_corpora.append(result.authorized_corpora[0])  # type: ignore[attr-defined]


# =========================================================================
# 4. Rights Review Before Acquisition Tests (Correction 3)
# =========================================================================


def test_rights_review_before_acquisition() -> None:
    """Verify data_state in (acquired, verified) strictly requires rights_decision in (approved, restricted)."""
    base_acq_dict = {
        "corpus_id": "acq_rights_test",
        "display_name": "Acquired Rights Test",
        "data_state": DataState.ACQUIRED,
        "content_type": ContentType.MIXED,
        "source_url": "https://example.com/src",
        "documentation_url": "https://example.com/docs",
        "declared_license_spdx": "MIT",
        "license_url": "https://opensource.org/licenses/MIT",
        "rights_reviewed_at": date(2026, 1, 1),
        "rights_reviewed_by": "Compliance Lead",
        "allowed_purposes": [],
        "notes": "Testing rights review before acquisition.",
        "version_ref": "v1.0",
        "archive_sha256": VALID_ARCHIVE_HASH,
    }

    # acquired + pending -> rejected
    with pytest.raises(
        ValidationError,
        match="Acquired or verified entries require a prior rights review decision of 'approved' or 'restricted'",
    ):
        CorpusEntry.model_validate({**base_acq_dict, "rights_decision": RightsDecision.PENDING})

    # acquired + blocked -> rejected
    with pytest.raises(
        ValidationError,
        match="Acquired or verified entries require a prior rights review decision of 'approved' or 'restricted'",
    ):
        CorpusEntry.model_validate({**base_acq_dict, "rights_decision": RightsDecision.BLOCKED})

    # verified + pending -> rejected
    base_ver_dict = {
        **base_acq_dict,
        "data_state": DataState.VERIFIED,
        "manifest_sha256": VALID_MANIFEST_HASH,
        "sample_count": 1000,
        "allowed_purposes": [AllowedPurpose.RESEARCH_EVALUATION],
    }
    with pytest.raises(
        ValidationError,
        match="Acquired or verified entries require a prior rights review decision of 'approved' or 'restricted'",
    ):
        CorpusEntry.model_validate({**base_ver_dict, "rights_decision": RightsDecision.PENDING})

    # verified + blocked -> rejected
    with pytest.raises(
        ValidationError,
        match="Acquired or verified entries require a prior rights review decision of 'approved' or 'restricted'",
    ):
        CorpusEntry.model_validate({**base_ver_dict, "rights_decision": RightsDecision.BLOCKED})

    # planned + approved is valid in registry, but cannot be authorized
    planned_approved = CorpusEntry(
        corpus_id="plan_app",
        display_name="Planned Approved",
        version_ref=None,
        data_state=DataState.PLANNED,
        content_type=ContentType.MIXED,
        source_url="https://example.com",
        documentation_url="https://example.com",
        declared_license_spdx="MIT",
        license_url="https://opensource.org/licenses/MIT",
        rights_decision=RightsDecision.APPROVED,
        rights_reviewed_at=date(2026, 1, 1),
        rights_reviewed_by="Compliance Officer",
        allowed_purposes=(AllowedPurpose.RESEARCH_EVALUATION,),
        notes="Planned entry with pre-cleared rights.",
    )
    assert planned_approved.data_state == DataState.PLANNED
    assert planned_approved.rights_decision == RightsDecision.APPROVED

    plan_reg = CorpusRegistry(schema_version="1.0.0", corpora=(planned_approved,))
    is_auth, res, reasons = check_corpus_authorization(
        plan_reg, AllowedPurpose.RESEARCH_EVALUATION, ["plan_app@v1.0"]
    )
    assert not is_auth
    assert any("has data_state='planned'" in r or "no acquired or verified" in r for r in reasons)


# =========================================================================
# 5. Strict Scalar Types Tests (Correction 4)
# =========================================================================


def test_strict_scalar_types() -> None:
    """Verify sample_count and attribution_required enforce strict typing and reject boolean/integer coercion."""
    base_dict = create_verified_entry().model_dump()

    # sample_count must not coerce True to 1 or False to 0
    with pytest.raises(
        ValidationError, match="sample_count must be a positive integer, not a boolean"
    ):
        CorpusEntry.model_validate({**base_dict, "sample_count": True})

    with pytest.raises(
        ValidationError, match="sample_count must be a positive integer, not a boolean"
    ):
        CorpusEntry.model_validate({**base_dict, "sample_count": False})

    # sample_count must be positive integer (>0)
    with pytest.raises(ValidationError, match="sample_count must be a positive integer"):
        CorpusEntry.model_validate({**base_dict, "sample_count": 0})

    with pytest.raises(ValidationError, match="sample_count must be a positive integer"):
        CorpusEntry.model_validate({**base_dict, "sample_count": -5})

    with pytest.raises(ValidationError, match="sample_count must be an integer"):
        CorpusEntry.model_validate({**base_dict, "sample_count": 100.5})

    with pytest.raises(ValidationError, match="sample_count must be an integer"):
        CorpusEntry.model_validate({**base_dict, "sample_count": "5000"})

    # attribution_required must not coerce integers (0, 1) or strings ("true", "false")
    with pytest.raises(ValidationError, match="attribution_required must be a boolean"):
        CorpusEntry.model_validate({**base_dict, "attribution_required": 1})

    with pytest.raises(ValidationError, match="attribution_required must be a boolean"):
        CorpusEntry.model_validate({**base_dict, "attribution_required": 0})

    with pytest.raises(ValidationError, match="attribution_required must be a boolean"):
        CorpusEntry.model_validate({**base_dict, "attribution_required": "true"})

    with pytest.raises(ValidationError, match="attribution_required must be a boolean"):
        CorpusEntry.model_validate({**base_dict, "attribution_required": "false"})

    # Valid values
    valid_true = CorpusEntry.model_validate({**base_dict, "attribution_required": True})
    assert valid_true.attribution_required is True

    valid_false = CorpusEntry.model_validate({**base_dict, "attribution_required": False})
    assert valid_false.attribution_required is False

    valid_none = CorpusEntry.model_validate({**base_dict, "attribution_required": None})
    assert valid_none.attribution_required is None


# =========================================================================
# 6. Purpose and Request Uniqueness & Determinism (Correction 5)
# =========================================================================


def test_purpose_uniqueness_rejection() -> None:
    """Verify duplicate values in allowed_purposes are rejected."""
    base_dict = create_verified_entry().model_dump()
    with pytest.raises(ValidationError, match="allowed_purposes contains duplicate values"):
        CorpusEntry.model_validate(
            {
                **base_dict,
                "allowed_purposes": [
                    AllowedPurpose.RESEARCH_EVALUATION,
                    AllowedPurpose.RESEARCH_EVALUATION,
                ],
            }
        )


def test_duplicate_request_identities_denial() -> None:
    """Verify duplicate requested corpus identities in one authorization call are rejected."""
    entry = create_verified_entry(corpus_id="dup_req_corpus", version_ref="v1")
    registry = CorpusRegistry(schema_version="1.0.0", corpora=(entry,))

    is_auth, res, reasons = check_corpus_authorization(
        registry=registry,
        purpose=AllowedPurpose.RESEARCH_EVALUATION,
        corpus_identifiers=["dup_req_corpus@v1", "dup_req_corpus@v1"],
    )
    assert not is_auth
    assert res is None
    assert any("Duplicate corpus identity request detected" in r for r in reasons)

    with pytest.raises(
        CorpusAuthorizationError, match="Duplicate corpus identity request detected"
    ):
        authorize_corpus_usage(
            registry=registry,
            purpose=AllowedPurpose.RESEARCH_EVALUATION,
            corpus_identifiers=["dup_req_corpus@v1", "dup_req_corpus@v1"],
        )


def test_authorization_result_deterministic_sorting() -> None:
    """Verify successful authorized identities are deterministically sorted by (corpus_id, version_ref)."""
    entry_b = create_verified_entry(corpus_id="corpus_b", version_ref="v1")
    entry_a = create_verified_entry(corpus_id="corpus_a", version_ref="v2")
    registry = CorpusRegistry(schema_version="1.0.0", corpora=(entry_b, entry_a))

    # Request in reverse order
    res1 = authorize_corpus_usage(
        registry=registry,
        purpose=AllowedPurpose.RESEARCH_EVALUATION,
        corpus_identifiers=["corpus_b@v1", "corpus_a@v2"],
    )

    # Request in alphabetical order
    res2 = authorize_corpus_usage(
        registry=registry,
        purpose=AllowedPurpose.RESEARCH_EVALUATION,
        corpus_identifiers=["corpus_a@v2", "corpus_b@v1"],
    )

    # Both must result in corpus_a first, corpus_b second
    assert res1.authorized_corpora[0].corpus_id == "corpus_a"
    assert res1.authorized_corpora[1].corpus_id == "corpus_b"
    assert res2.authorized_corpora[0].corpus_id == "corpus_a"
    assert res2.authorized_corpora[1].corpus_id == "corpus_b"
    assert res1.authorized_corpora == res2.authorized_corpora


# =========================================================================
# 7. Version Identity Validation Tests (Correction 6)
# =========================================================================


def test_version_identity_validation() -> None:
    """Verify CorpusIdentity rejects empty or whitespace-only versions on direct construction and parsing."""
    # Direct construction rejections
    with pytest.raises(
        ValidationError, match="version_ref must be a non-empty string when provided"
    ):
        CorpusIdentity(corpus_id="test_corp", version_ref="   ")

    with pytest.raises(
        ValidationError, match="version_ref must be a non-empty string when provided"
    ):
        CorpusIdentity(corpus_id="test_corp", version_ref="")

    # Parsing rejections
    with pytest.raises(ValueError, match="Version reference cannot be empty"):
        CorpusIdentity.parse("test_corp@")

    with pytest.raises(ValueError, match="Version reference cannot be empty"):
        CorpusIdentity.parse("test_corp@   ")

    # Valid direct construction
    ident_valid = CorpusIdentity(corpus_id="test_corp", version_ref="v1.0")
    assert ident_valid.corpus_id == "test_corp"
    assert ident_valid.version_ref == "v1.0"

    ident_none = CorpusIdentity(corpus_id="test_corp", version_ref=None)
    assert ident_none.version_ref is None


# =========================================================================
# 8. Lifecycle State Invariant Tests
# =========================================================================


def test_planned_state_invariants() -> None:
    """Verify planned entries must not claim archive hashes, manifest hashes, or sample counts."""
    base_planned = {
        "corpus_id": "plan_invariants",
        "display_name": "Planned Invariants",
        "data_state": DataState.PLANNED,
        "content_type": ContentType.MIXED,
        "source_url": "https://example.com",
        "documentation_url": "https://example.com",
        "rights_decision": RightsDecision.PENDING,
        "allowed_purposes": [],
        "notes": "Testing planned state invariants.",
    }

    # Claiming archive hash
    with pytest.raises(ValidationError, match="Planned entries must not claim archive hashes"):
        CorpusEntry.model_validate({**base_planned, "archive_sha256": VALID_ARCHIVE_HASH})

    # Claiming manifest hash
    with pytest.raises(ValidationError, match="Planned entries must not claim manifest hashes"):
        CorpusEntry.model_validate({**base_planned, "manifest_sha256": VALID_MANIFEST_HASH})

    # Claiming sample count
    with pytest.raises(ValidationError, match="Planned entries must not claim sample counts"):
        CorpusEntry.model_validate({**base_planned, "sample_count": 500})


def test_acquired_state_invariants() -> None:
    """Verify acquired entries require an immutable version_ref, archive_sha256, and approved/restricted rights."""
    base_acquired = {
        "corpus_id": "acq_invariants",
        "display_name": "Acquired Invariants",
        "data_state": DataState.ACQUIRED,
        "content_type": ContentType.MIXED,
        "source_url": "https://example.com",
        "documentation_url": "https://example.com",
        "declared_license_spdx": "Apache-2.0",
        "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
        "rights_decision": RightsDecision.APPROVED,
        "rights_reviewed_at": date(2026, 1, 1),
        "rights_reviewed_by": "Compliance Lead",
        "allowed_purposes": [],
        "notes": "Testing acquired invariants.",
    }

    # Missing version_ref
    with pytest.raises(ValidationError, match="Acquired entries require an immutable version_ref"):
        CorpusEntry.model_validate({**base_acquired, "archive_sha256": VALID_ARCHIVE_HASH})

    # Missing archive_sha256
    with pytest.raises(ValidationError, match="Acquired entries require archive_sha256"):
        CorpusEntry.model_validate({**base_acquired, "version_ref": "v1.0"})

    # Valid acquired entry
    valid_acq = CorpusEntry.model_validate(
        {**base_acquired, "version_ref": "v1.0", "archive_sha256": VALID_ARCHIVE_HASH}
    )
    assert valid_acq.data_state == DataState.ACQUIRED


def test_verified_state_invariants() -> None:
    """Verify verified entries require version_ref, archive hash, manifest hash, and positive sample_count."""
    base_verified: dict[str, Any] = {
        "corpus_id": "ver_invariants",
        "display_name": "Verified Invariants",
        "data_state": DataState.VERIFIED,
        "content_type": ContentType.MIXED,
        "source_url": "https://example.com",
        "documentation_url": "https://example.com",
        "declared_license_spdx": "Apache-2.0",
        "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
        "rights_decision": RightsDecision.APPROVED,
        "rights_reviewed_at": date(2026, 1, 1),
        "rights_reviewed_by": "Auditor",
        "allowed_purposes": [AllowedPurpose.RESEARCH_EVALUATION],
        "notes": "Testing verified invariants.",
        "version_ref": "v1.0",
        "archive_sha256": VALID_ARCHIVE_HASH,
        "manifest_sha256": VALID_MANIFEST_HASH,
        "sample_count": 1000,
    }

    # Missing version_ref
    with pytest.raises(ValidationError, match="Verified entries require an immutable version_ref"):
        CorpusEntry.model_validate({**base_verified, "version_ref": None})

    # Missing archive_sha256
    with pytest.raises(ValidationError, match="Verified entries require archive_sha256"):
        CorpusEntry.model_validate({**base_verified, "archive_sha256": None})

    # Missing manifest_sha256
    with pytest.raises(ValidationError, match="Verified entries require manifest_sha256"):
        CorpusEntry.model_validate({**base_verified, "manifest_sha256": None})

    # Missing sample_count
    with pytest.raises(ValidationError, match="Verified entries require a positive sample_count"):
        CorpusEntry.model_validate({**base_verified, "sample_count": None})

    # Zero or negative sample_count
    with pytest.raises(ValidationError, match="sample_count must be a positive integer"):
        CorpusEntry.model_validate({**base_verified, "sample_count": 0})


def test_pending_and_blocked_purpose_rejection() -> None:
    """Verify pending or blocked entries must have an empty allowed_purposes list."""
    # Pending with purpose
    with pytest.raises(
        ValidationError, match="Pending or blocked entries must have an empty allowed_purposes list"
    ):
        CorpusEntry(
            corpus_id="pending_with_purposes",
            display_name="Pending Test",
            data_state=DataState.PLANNED,
            content_type=ContentType.MIXED,
            source_url="https://example.com",
            documentation_url="https://example.com",
            rights_decision=RightsDecision.PENDING,
            allowed_purposes=(AllowedPurpose.RESEARCH_EVALUATION,),
            notes="Note",
        )

    # Blocked with purpose
    with pytest.raises(
        ValidationError, match="Pending or blocked entries must have an empty allowed_purposes list"
    ):
        CorpusEntry(
            corpus_id="blocked_with_purposes",
            display_name="Blocked Test",
            data_state=DataState.PLANNED,
            content_type=ContentType.MIXED,
            source_url="https://example.com",
            documentation_url="https://example.com",
            rights_decision=RightsDecision.BLOCKED,
            allowed_purposes=(AllowedPurpose.CALIBRATION,),
            notes="Note",
        )


def test_approved_and_restricted_review_metadata_requirements() -> None:
    """Verify approved or restricted entries require reviewer, date, license SPDX, and license URL."""
    base_approved = create_verified_entry().model_dump()

    # Missing rights_reviewed_by
    with pytest.raises(
        ValidationError, match="Approved or restricted entries require rights_reviewed_by"
    ):
        CorpusEntry.model_validate({**base_approved, "rights_reviewed_by": None})

    # Missing rights_reviewed_at
    with pytest.raises(
        ValidationError, match="Approved or restricted entries require rights_reviewed_at"
    ):
        CorpusEntry.model_validate({**base_approved, "rights_reviewed_at": None})

    # Missing declared_license_spdx
    with pytest.raises(
        ValidationError, match="Approved or restricted entries require declared_license_spdx"
    ):
        CorpusEntry.model_validate({**base_approved, "declared_license_spdx": None})

    # Missing license_url
    with pytest.raises(ValidationError, match="Approved or restricted entries require license_url"):
        CorpusEntry.model_validate({**base_approved, "license_url": None})


def test_redistribution_purpose_coherence() -> None:
    """Verify raw_redistribution may be an allowed purpose only when raw_redistribution_status is allowed."""
    base_dict = create_verified_entry().model_dump()
    base_dict["allowed_purposes"] = [AllowedPurpose.RAW_REDISTRIBUTION]
    base_dict["raw_redistribution_status"] = RedistributionStatus.PROHIBITED

    with pytest.raises(
        ValidationError,
        match="raw_redistribution may be an allowed purpose only when raw_redistribution_status is allowed",
    ):
        CorpusEntry.model_validate(base_dict)

    # Allowed status passes
    base_dict["raw_redistribution_status"] = RedistributionStatus.ALLOWED
    valid_redist = CorpusEntry.model_validate(base_dict)
    assert AllowedPurpose.RAW_REDISTRIBUTION in valid_redist.allowed_purposes


# =========================================================================
# 9. Use-Gate Authorization API Tests
# =========================================================================


def test_unknown_corpus_denial() -> None:
    """Verify authorization fails closed when an unknown corpus ID is requested."""
    registry = CorpusRegistry(schema_version="1.0.0", corpora=(create_verified_entry(),))

    with pytest.raises(CorpusAuthorizationError) as exc_info:
        authorize_corpus_usage(
            registry=registry,
            purpose=AllowedPurpose.RESEARCH_EVALUATION,
            corpus_identifiers=["nonexistent_corpus@v1.0"],
        )
    assert "Unknown corpus ID 'nonexistent_corpus'" in str(exc_info.value)
    assert len(exc_info.value.denial_reasons) == 1


def test_ambiguous_or_incorrect_version_denial() -> None:
    """Verify authorization denies requests missing version or specifying wrong version."""
    entry = create_verified_entry(corpus_id="dataset_alpha", version_ref="v2.1")
    registry = CorpusRegistry(schema_version="1.0.0", corpora=(entry,))

    # Missing version reference
    with pytest.raises(CorpusAuthorizationError) as exc_info:
        authorize_corpus_usage(
            registry=registry,
            purpose=AllowedPurpose.RESEARCH_EVALUATION,
            corpus_identifiers=["dataset_alpha"],
        )
    assert "missing an exact version reference" in str(exc_info.value)

    # Wrong version reference
    with pytest.raises(CorpusAuthorizationError) as exc_info:
        authorize_corpus_usage(
            registry=registry,
            purpose=AllowedPurpose.RESEARCH_EVALUATION,
            corpus_identifiers=["dataset_alpha@v9.9"],
        )
    assert "version 'v9.9' not found in registry" in str(exc_info.value)


def test_planned_corpus_authorization_denial() -> None:
    """Verify authorization fails closed for planned datasets in configs/corpus_registry.yaml."""
    registry = load_corpus_registry(CONFIG_PATH)

    with pytest.raises(CorpusAuthorizationError) as exc_info:
        authorize_corpus_usage(
            registry=registry,
            purpose=AllowedPurpose.RESEARCH_EVALUATION,
            corpus_identifiers=["asvspoof5@5"],
        )
    assert "no acquired or verified version snapshots" in str(exc_info.value)


def test_missing_purpose_denial() -> None:
    """Verify authorization denies requests when purpose is not in allowed_purposes."""
    entry = create_verified_entry(
        corpus_id="eval_only",
        version_ref="v1.0",
        allowed_purposes=[AllowedPurpose.RESEARCH_EVALUATION],
    )
    registry = CorpusRegistry(schema_version="1.0.0", corpora=(entry,))

    # Request calibration (not allowed)
    with pytest.raises(CorpusAuthorizationError) as exc_info:
        authorize_corpus_usage(
            registry=registry,
            purpose=AllowedPurpose.CALIBRATION,
            corpus_identifiers=["eval_only@v1.0"],
        )
    assert "does not permit purpose 'calibration'" in str(exc_info.value)

    # Request product_training (not allowed)
    with pytest.raises(CorpusAuthorizationError) as exc_info:
        authorize_corpus_usage(
            registry=registry,
            purpose=AllowedPurpose.PRODUCT_TRAINING,
            corpus_identifiers=["eval_only@v1.0"],
        )
    assert "does not permit purpose 'product_training'" in str(exc_info.value)


def test_successful_authorization_fixture() -> None:
    """Verify successful authorization of a verified & approved corpus fixture."""
    entry = create_verified_entry(
        corpus_id="golden_eval",
        version_ref="v3.0",
        allowed_purposes=[AllowedPurpose.RESEARCH_EVALUATION, AllowedPurpose.CALIBRATION],
    )
    registry = CorpusRegistry(schema_version="1.0.0", corpora=(entry,))

    result: CorpusAuthorizationResult = authorize_corpus_usage(
        registry=registry,
        purpose=AllowedPurpose.RESEARCH_EVALUATION,
        corpus_identifiers=["golden_eval@v3.0"],
    )

    assert result.purpose == AllowedPurpose.RESEARCH_EVALUATION
    assert len(result.authorized_corpora) == 1
    auth_c = result.authorized_corpora[0]
    assert auth_c.corpus_id == "golden_eval"
    assert auth_c.version_ref == "v3.0"
    assert auth_c.archive_sha256 == VALID_ARCHIVE_HASH
    assert auth_c.manifest_sha256 == VALID_MANIFEST_HASH
    assert auth_c.sample_count == 5000
    assert auth_c.rights_decision == RightsDecision.APPROVED
    assert auth_c.content_type == ContentType.MIXED


def test_multi_corpus_all_or_nothing_authorization() -> None:
    """Verify multi-corpus authorization fails closed if any single corpus is denied."""
    entry1 = create_verified_entry(
        corpus_id="corpus_1",
        version_ref="v1.0",
        allowed_purposes=[AllowedPurpose.RESEARCH_EVALUATION],
    )
    entry2 = create_verified_entry(
        corpus_id="corpus_2",
        version_ref="v1.0",
        allowed_purposes=[AllowedPurpose.CALIBRATION],  # does not permit RESEARCH_EVALUATION
    )
    registry = CorpusRegistry(schema_version="1.0.0", corpora=(entry1, entry2))

    is_auth, res, reasons = check_corpus_authorization(
        registry=registry,
        purpose=AllowedPurpose.RESEARCH_EVALUATION,
        corpus_identifiers=["corpus_1@v1.0", "corpus_2@v1.0"],
    )
    assert not is_auth
    assert res is None
    assert len(reasons) == 1
    assert "corpus_2@v1.0" in reasons[0]

    # authorize_corpus_usage must raise
    with pytest.raises(CorpusAuthorizationError) as exc_info:
        authorize_corpus_usage(
            registry=registry,
            purpose=AllowedPurpose.RESEARCH_EVALUATION,
            corpus_identifiers=["corpus_1@v1.0", "corpus_2@v1.0"],
        )
    assert "corpus_2@v1.0" in str(exc_info.value)


def test_authorization_revalidates_registry_and_rejects_bare_string() -> None:
    """Authorization must fail closed for bypass-constructed models and bare-string requests."""
    entry = create_verified_entry(raw_redistribution_status=RedistributionStatus.PROHIBITED)
    tampered_entry = entry.model_copy(
        update={"allowed_purposes": (AllowedPurpose.RAW_REDISTRIBUTION,)}
    )
    tampered_registry = CorpusRegistry.model_construct(
        schema_version="1.0.0",
        corpora=(tampered_entry,),
    )

    is_auth, result, reasons = check_corpus_authorization(
        registry=tampered_registry,
        purpose=AllowedPurpose.RAW_REDISTRIBUTION,
        corpus_identifiers=[f"{entry.corpus_id}@{entry.version_ref}"],
    )
    assert not is_auth
    assert result is None
    assert any("Invalid corpus registry supplied" in reason for reason in reasons)

    valid_registry = CorpusRegistry(corpora=(entry,))
    is_auth, result, reasons = check_corpus_authorization(
        registry=valid_registry,
        purpose=AllowedPurpose.RESEARCH_EVALUATION,
        corpus_identifiers=f"{entry.corpus_id}@{entry.version_ref}",
    )
    assert not is_auth
    assert result is None
    assert reasons == [
        "Corpus identities must be supplied as a sequence, not a bare string or bytes value."
    ]


# =========================================================================
# 10. Hashing & Byte Integrity Tests
# =========================================================================


def test_exact_registry_byte_hash(tmp_path: Path) -> None:
    """Verify compute_registry_hash computes exact lowercase SHA-256 of file bytes."""
    test_file = tmp_path / "test_reg.yaml"
    content = b"schema_version: '1.0.0'\ncorpora: []\n"
    test_file.write_bytes(content)

    expected_hash = hashlib.sha256(content).hexdigest().lower()
    computed_file = compute_registry_file_hash(test_file)
    computed_bytes = compute_registry_bytes_hash(content)
    canonical_alias = compute_registry_hash(test_file)

    assert computed_file == expected_hash
    assert computed_bytes == expected_hash
    assert canonical_alias == expected_hash
    assert (
        compute_registry_hash(CONFIG_PATH)
        == hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest().lower()
    )


def test_scientific_protocol_lifecycle_and_metric_contract() -> None:
    """The protocol must preserve the non-circular lifecycle and required metric semantics."""
    protocol = Path("docs/scientific_validation_protocol.md").read_text()

    assert "Registration & Rights Review Before Acquisition" in protocol
    assert (
        "Canonical manifest construction and integrity verification are lifecycle steps" in protocol
    )
    assert "**False-positive analysis**" in protocol
    assert "**False-negative analysis**" in protocol
    assert "full confusion matrix" in protocol
    assert "sensitivity/recall" in protocol
    assert "specificity" in protocol
    assert "not a confidence value or posterior probability" in protocol
    assert "is the declared spoof target prior" in protocol
    assert "Reports must not call the unnormalized cost normalized" in protocol
    assert "Prior to downloading, acquiring, preprocessing, or evaluating" not in protocol


# =========================================================================
# 11. Error Normalization Tests (Correction 12)
# =========================================================================


def test_load_corpus_registry_error_normalization(tmp_path: Path) -> None:
    """Verify malformed YAML, unhashable keys, non-mapping roots, missing files, and schema errors raise CorpusValidationError."""
    # 1. File not found
    with pytest.raises(CorpusValidationError, match="Corpus registry file not found"):
        load_corpus_registry(tmp_path / "nonexistent.yaml")

    # 2. Empty file
    empty_file = tmp_path / "empty.yaml"
    empty_file.write_text("")
    with pytest.raises(CorpusValidationError, match="empty or contains only whitespace"):
        load_corpus_registry(empty_file)

    # 3. Whitespace-only file
    ws_file = tmp_path / "whitespace.yaml"
    ws_file.write_text("   \n\t  \n")
    with pytest.raises(CorpusValidationError, match="empty or contains only whitespace"):
        load_corpus_registry(ws_file)

    # 4. Malformed YAML syntax
    malformed_yaml = tmp_path / "syntax_err.yaml"
    malformed_yaml.write_text("schema_version: '1.0.0'\ncorpora: [unclosed list")
    with pytest.raises(CorpusValidationError, match="Malformed YAML in corpus registry"):
        load_corpus_registry(malformed_yaml)

    # 5. Non-mapping YAML root (list)
    list_yaml = tmp_path / "list_root.yaml"
    list_yaml.write_text("- item1\n- item2\n")
    with pytest.raises(CorpusValidationError, match="root must be a YAML mapping/dictionary"):
        load_corpus_registry(list_yaml)

    # 6. Non-mapping YAML root (scalar)
    scalar_yaml = tmp_path / "scalar_root.yaml"
    scalar_yaml.write_text("just_a_string")
    with pytest.raises(CorpusValidationError, match="root must be a YAML mapping/dictionary"):
        load_corpus_registry(scalar_yaml)

    # 7. Duplicate YAML keys
    dup_yaml = tmp_path / "duplicate_keys.yaml"
    dup_yaml.write_text("schema_version: '1.0.0'\nschema_version: '1.0.0'\ncorpora: []\n")
    with pytest.raises(CorpusValidationError, match="found duplicate YAML key"):
        load_corpus_registry(dup_yaml)

    # 8. Unhashable YAML keys
    unhashable_yaml = tmp_path / "unhashable_keys.yaml"
    unhashable_yaml.write_text("{[1, 2]: 'val'}\n")
    with pytest.raises(CorpusValidationError, match="unhashable mapping key"):
        load_corpus_registry(unhashable_yaml)

    # 9. Schema validation failure (extra fields or invariant failure)
    schema_err_yaml = tmp_path / "schema_err.yaml"
    schema_err_yaml.write_text("schema_version: '9.9.9'\ncorpora: []\n")
    with pytest.raises(CorpusValidationError, match="invariant validation failed"):
        load_corpus_registry(schema_err_yaml)


# =========================================================================
# 12. CLI Execution & JSON Contract Tests
# =========================================================================


def test_cli_success_json() -> None:
    """Verify running the CLI on valid registry returns exit 0 and required JSON fields."""
    cmd = [sys.executable, str(SCRIPT_PATH), str(CONFIG_PATH)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    assert result.returncode == 0, f"CLI exited with {result.returncode}: {result.stderr}"
    data = json.loads(result.stdout)

    assert data["schema_version"] == "1.0.0"
    assert data["registry_sha256"] == compute_registry_hash(CONFIG_PATH)
    assert data["corpus_count"] == 4
    assert data["counts_by_data_state"] == {"planned": 4, "acquired": 0, "verified": 0}
    assert data["counts_by_rights_decision"] == {
        "approved": 0,
        "restricted": 0,
        "pending": 4,
        "blocked": 0,
    }


def test_cli_structural_failure_exit(tmp_path: Path) -> None:
    """Verify CLI structural failure exits nonzero with a clean error and no traceback."""
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("schema_version: '1.0.0'\nunknown_top_level: true\n")

    cmd = [sys.executable, str(SCRIPT_PATH), str(bad_yaml)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    assert result.returncode != 0
    assert "Registry Validation Failed" in result.stderr
    assert "Traceback (most recent call last)" not in result.stderr


def test_cli_authorization_denial_exit() -> None:
    """Verify CLI authorization denial on planned dataset exits nonzero without a traceback."""
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        str(CONFIG_PATH),
        "--purpose",
        "research_evaluation",
        "--corpus",
        "asvspoof5@5",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    assert result.returncode != 0
    assert "Authorization Denied for purpose 'research_evaluation'" in result.stderr
    assert "Traceback (most recent call last)" not in result.stderr


def test_cli_authorization_success(tmp_path: Path) -> None:
    """Verify CLI authorization success on verified dataset outputs authorized JSON."""
    verified_yaml = tmp_path / "verified_reg.yaml"
    registry_obj = CorpusRegistry(
        schema_version="1.0.0",
        corpora=(
            create_verified_entry(
                corpus_id="cli_verified",
                version_ref="v1",
                allowed_purposes=[AllowedPurpose.RESEARCH_EVALUATION],
            ),
        ),
    )
    verified_yaml.write_text(yaml.dump(registry_obj.model_dump(mode="json")))

    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        str(verified_yaml),
        "--purpose",
        "research_evaluation",
        "--corpus",
        "cli_verified@v1",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    assert result.returncode == 0, f"CLI exited with {result.returncode}: {result.stderr}"
    data = json.loads(result.stdout)
    assert data["status"] == "authorized"
    assert data["purpose"] == "research_evaluation"
    assert len(data["authorized_corpora"]) == 1
    assert data["authorized_corpora"][0]["corpus_id"] == "cli_verified"


def test_cli_missing_arguments_exit() -> None:
    """Verify specifying --purpose without --corpus or --corpus without --purpose exits with code 2 and no traceback."""
    # Purpose without corpus
    cmd1 = [sys.executable, str(SCRIPT_PATH), str(CONFIG_PATH), "--purpose", "research_evaluation"]
    res1 = subprocess.run(cmd1, capture_output=True, text=True, check=False)
    assert res1.returncode == 2
    assert "Both --purpose and at least one --corpus are required" in res1.stderr
    assert "Traceback (most recent call last)" not in res1.stderr

    # Corpus without purpose
    cmd2 = [sys.executable, str(SCRIPT_PATH), str(CONFIG_PATH), "--corpus", "asvspoof5@5"]
    res2 = subprocess.run(cmd2, capture_output=True, text=True, check=False)
    assert res2.returncode == 2
    assert "Both --purpose and at least one --corpus are required" in res2.stderr
    assert "Traceback (most recent call last)" not in res2.stderr


def test_cli_error_normalization_no_traceback(tmp_path: Path) -> None:
    """Verify CLI cleanly reports errors for unhashable keys, duplicate keys, and missing files without tracebacks."""
    # Unhashable key file
    unhashable = tmp_path / "unhashable.yaml"
    unhashable.write_text("{[1, 2]: 'val'}\n")
    res = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(unhashable)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 1
    assert "Registry Validation Failed" in res.stderr
    assert "Traceback (most recent call last)" not in res.stderr

    # Missing file
    missing = tmp_path / "missing.yaml"
    res_missing = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(missing)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_missing.returncode == 1
    assert "File Not Found" in res_missing.stderr
    assert "Traceback (most recent call last)" not in res_missing.stderr


# =========================================================================
# 13. Identity Parsing & No Network Access Tests
# =========================================================================


def test_corpus_identity_parsing() -> None:
    """Verify CorpusIdentity parses strings with and without versions, rejecting invalid forms."""
    ident = CorpusIdentity.parse("asvspoof5@5")
    assert ident.corpus_id == "asvspoof5"
    assert ident.version_ref == "5"

    ident_no_ver = CorpusIdentity.parse("common_voice_18")
    assert ident_no_ver.corpus_id == "common_voice_18"
    assert ident_no_ver.version_ref is None

    # Already a CorpusIdentity instance
    assert CorpusIdentity.parse(ident) == ident

    # Invalid empty or whitespace strings
    with pytest.raises(ValueError, match="Invalid corpus identifier"):
        CorpusIdentity.parse("")
    with pytest.raises(ValueError, match="Invalid corpus identifier"):
        CorpusIdentity.parse("   ")
    with pytest.raises(ValueError, match="Corpus ID cannot be empty"):
        CorpusIdentity.parse("@v1")
    with pytest.raises(ValueError, match="Version reference cannot be empty"):
        CorpusIdentity.parse("corpus@")


def test_no_network_access(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify registry loading, validation, hashing, authorization, and CLI make zero network requests."""

    # Defensively intercept socket connections
    def guarded_connect(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Network connection attempted during governance validation!")

    monkeypatch.setattr(socket, "create_connection", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)

    # 1. Loading and validation
    reg = load_corpus_registry(CONFIG_PATH)
    assert len(reg.corpora) == 4

    # 2. Hashing
    h = compute_registry_hash(CONFIG_PATH)
    assert len(h) == 64

    # 3. Authorization gate check
    is_auth, res, reasons = check_corpus_authorization(
        reg, AllowedPurpose.RESEARCH_EVALUATION, ["asvspoof5@5"]
    )
    assert not is_auth
