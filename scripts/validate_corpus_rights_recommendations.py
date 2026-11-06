#!/usr/bin/env python3
"""Validate Stage C3 non-authorizing corpus-rights recommendations."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from voiceshield.data.corpus_rights_recommendations import (
    CorpusRecommendationCrossValidationError,
    CorpusRecommendationValidationError,
    RecommendationDisposition,
    compute_recommendation_bytes_hash,
    verify_corpus_rights_recommendations,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate VoiceShield Stage C3 research-evaluation recommendations "
            "against exact C2 evidence and frozen registry bytes."
        )
    )
    parser.add_argument(
        "recommendations_file",
        nargs="?",
        default="configs/corpus_rights_recommendations.yaml",
    )
    parser.add_argument(
        "evidence_file",
        nargs="?",
        default="configs/corpus_rights_evidence.yaml",
    )
    parser.add_argument(
        "registry_file",
        nargs="?",
        default="configs/corpus_registry.yaml",
    )
    args = parser.parse_args()

    recommendation_path = Path(args.recommendations_file)
    evidence_path = Path(args.evidence_file)
    registry_path = Path(args.registry_file)

    try:
        recommendation_bytes = recommendation_path.read_bytes()
        evidence_bytes = evidence_path.read_bytes()
        registry_bytes = registry_path.read_bytes()
    except FileNotFoundError as exc:
        sys.stderr.write(f"Input file not found: {exc.filename}\n")
        return 1
    except Exception as exc:
        sys.stderr.write(f"Input read failed: {exc}\n")
        return 1

    try:
        packet, _evidence, _registry, result = verify_corpus_rights_recommendations(
            recommendation_bytes,
            evidence_bytes,
            registry_bytes,
        )
    except CorpusRecommendationValidationError as exc:
        sys.stderr.write(f"Recommendation Validation Failed: {exc}\n")
        return 1
    except CorpusRecommendationCrossValidationError as exc:
        sys.stderr.write("Recommendation Cross-Validation Failed:\n")
        for error in exc.errors:
            sys.stderr.write(f"  - {error}\n")
        return 1
    except Exception as exc:
        sys.stderr.write(f"Unexpected Recommendation Error: {exc}\n")
        return 1

    counts_by_disposition = {disposition.value: 0 for disposition in RecommendationDisposition}
    priorities: dict[str, int] = {}
    for recommendation in packet.recommendations:
        counts_by_disposition[recommendation.disposition.value] += 1
        priorities[recommendation.corpus_id] = recommendation.review_priority

    summary = {
        "account_creation_authorized": packet.account_creation_authorized,
        "acquisition_authorized": packet.acquisition_authorized,
        "base_source_commit": packet.base_source_commit,
        "checked_recommendations_count": result.checked_recommendations_count,
        "counts_by_disposition": counts_by_disposition,
        "evidence_sha256": hashlib.sha256(evidence_bytes).hexdigest().lower(),
        "priorities": dict(sorted(priorities.items())),
        "recommendation_sha256": compute_recommendation_bytes_hash(recommendation_bytes),
        "registry_changes_authorized": packet.registry_changes_authorized,
        "registry_sha256": hashlib.sha256(registry_bytes).hexdigest().lower(),
        "schema_version": packet.schema_version,
        "scoped_purpose": packet.scoped_purpose,
        "status": "valid_recommendations_only",
        "terms_acceptance_authorized": packet.terms_acceptance_authorized,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
