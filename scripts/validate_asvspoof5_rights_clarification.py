#!/usr/bin/env python3
"""Validate Stage C4-A ASVspoof 5 public rights clarification."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from voiceshield.data.asvspoof5_rights_clarification import (
    ASVspoof5ClarificationCrossValidationError,
    ASVspoof5ClarificationValidationError,
    compute_clarification_bytes_hash,
    verify_asvspoof5_rights_clarification,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate VoiceShield Stage C4-A ASVspoof 5 clarification against "
            "its capture manifest and exact frozen governance files."
        )
    )
    parser.add_argument(
        "clarification_file",
        nargs="?",
        default="configs/asvspoof5_rights_clarification.yaml",
    )
    parser.add_argument(
        "capture_manifest_file",
        nargs="?",
        default="evidence/asvspoof5/CAPTURE-MANIFEST.json",
    )
    parser.add_argument(
        "evidence_file",
        nargs="?",
        default="configs/corpus_rights_evidence.yaml",
    )
    parser.add_argument(
        "recommendations_file",
        nargs="?",
        default="configs/corpus_rights_recommendations.yaml",
    )
    parser.add_argument(
        "registry_file",
        nargs="?",
        default="configs/corpus_registry.yaml",
    )
    parser.add_argument(
        "--capture-directory",
        default=None,
        help="Optional local source-capture directory for exact file hash verification.",
    )
    args = parser.parse_args()

    paths = {
        "clarification": Path(args.clarification_file),
        "capture manifest": Path(args.capture_manifest_file),
        "C2 evidence": Path(args.evidence_file),
        "C3 recommendations": Path(args.recommendations_file),
        "registry": Path(args.registry_file),
    }
    try:
        raw = {name: path.read_bytes() for name, path in paths.items()}
    except FileNotFoundError as exc:
        sys.stderr.write(f"Input file not found: {exc.filename}\n")
        return 1
    except Exception as exc:
        sys.stderr.write(f"Input read failed: {exc}\n")
        return 1

    try:
        packet, result = verify_asvspoof5_rights_clarification(
            raw["clarification"],
            raw["capture manifest"],
            raw["C2 evidence"],
            raw["C3 recommendations"],
            raw["registry"],
            capture_directory=args.capture_directory,
        )
    except ASVspoof5ClarificationValidationError as exc:
        sys.stderr.write(f"Clarification Validation Failed: {exc}\n")
        return 1
    except ASVspoof5ClarificationCrossValidationError as exc:
        sys.stderr.write("Clarification Cross-Validation Failed:\n")
        for error in exc.errors:
            sys.stderr.write(f"  - {error}\n")
        return 1
    except Exception as exc:
        sys.stderr.write(f"Unexpected Clarification Error: {exc}\n")
        return 1

    component_counts = Counter(item.status.value for item in packet.component_assessments)
    precondition_counts = Counter(item.status.value for item in packet.c3_precondition_assessments)
    summary = {
        "account_creation_authorized": packet.authorization_boundary.account_creation_authorized,
        "acquisition_authorized": packet.authorization_boundary.acquisition_authorized,
        "base_source_commit": packet.base_source_commit,
        "capture_manifest_sha256": hashlib.sha256(raw["capture manifest"]).hexdigest().lower(),
        "captured_source_bytes": result.checked_source_bytes,
        "captured_source_count": result.checked_source_count,
        "clarification_sha256": compute_clarification_bytes_hash(raw["clarification"]),
        "component_counts": dict(sorted(component_counts.items())),
        "c3_disposition": result.c3_disposition,
        "maintainer_question_count": len(packet.maintainer_questions),
        "outbound_contact_authorized": packet.authorization_boundary.outbound_contact_authorized,
        "packet_status": packet.packet_status,
        "precondition_counts": dict(sorted(precondition_counts.items())),
        "recommended_disposition": packet.recommended_disposition,
        "registry_changes_authorized": packet.authorization_boundary.registry_changes_authorized,
        "registry_state": result.registry_state,
        "schema_version": packet.schema_version,
        "scoped_purpose": packet.scoped_purpose,
        "status": "valid_non_authorizing_clarification",
        "terms_acceptance_authorized": packet.authorization_boundary.terms_acceptance_authorized,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
