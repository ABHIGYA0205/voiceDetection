#!/usr/bin/env python3
"""Validation and authorization CLI for VoiceShield rights-aware corpus registry.

Performs structural schema validation, duplicate key checking, governance invariant
verification, exact byte hashing, and fail-closed use-gate authorization checks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from voiceshield.data.corpus_registry import (
    CorpusValidationError,
    DataState,
    RightsDecision,
    check_corpus_authorization,
    compute_registry_bytes_hash,
    load_corpus_registry,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate VoiceShield rights-aware corpus registry."
    )
    parser.add_argument(
        "registry_path",
        type=Path,
        help="Path to corpus_registry.yaml file.",
    )
    parser.add_argument(
        "--purpose",
        type=str,
        default=None,
        help="Requested purpose to authorize (e.g. 'research_evaluation').",
    )
    parser.add_argument(
        "--corpus",
        dest="corpora",
        action="append",
        default=[],
        help="Corpus identity to authorize (format: 'corpus_id@version_ref'). Can be repeated.",
    )

    args = parser.parse_args()

    # Check for partially specified authorization arguments
    if args.purpose is not None and not args.corpora:
        sys.stderr.write(
            "Error: --purpose was specified without any --corpus arguments. "
            "Both --purpose and at least one --corpus are required for authorization checks.\n"
        )
        return 2

    if args.corpora and args.purpose is None:
        sys.stderr.write(
            "Error: --corpus was specified without --purpose. "
            "Both --purpose and at least one --corpus are required for authorization checks.\n"
        )
        return 2

    # 1. Read the file once so the reported hash and validated registry refer to
    # the same exact byte sequence, even if the path changes concurrently.
    try:
        registry_bytes = args.registry_path.read_bytes()
        registry_sha256 = compute_registry_bytes_hash(registry_bytes)
        registry = load_corpus_registry(registry_bytes)
    except CorpusValidationError as err:
        sys.stderr.write(f"Registry Validation Failed: {err}\n")
        return 1
    except FileNotFoundError as err:
        sys.stderr.write(f"File Not Found: {err}\n")
        return 1
    except Exception as err:
        sys.stderr.write(f"Unexpected Registry Error: {err}\n")
        return 1

    # 2. Authorization Check (if requested)
    if args.purpose is not None and args.corpora:
        is_auth, auth_result, denial_reasons = check_corpus_authorization(
            registry=registry,
            purpose=args.purpose,
            corpus_identifiers=args.corpora,
        )
        if not is_auth or auth_result is None:
            sys.stderr.write(f"Authorization Denied for purpose '{args.purpose}':\n")
            for reason in denial_reasons:
                sys.stderr.write(f"  - {reason}\n")
            return 1

        auth_output = {
            "authorized_corpora": [
                {
                    "archive_sha256": c.archive_sha256,
                    "content_type": c.content_type.value,
                    "corpus_id": c.corpus_id,
                    "manifest_sha256": c.manifest_sha256,
                    "rights_decision": c.rights_decision.value,
                    "sample_count": c.sample_count,
                    "version_ref": c.version_ref,
                }
                for c in sorted(
                    auth_result.authorized_corpora, key=lambda x: (x.corpus_id, x.version_ref)
                )
            ],
            "purpose": auth_result.purpose.value,
            "registry_sha256": registry_sha256,
            "schema_version": registry.schema_version,
            "status": "authorized",
        }
        print(json.dumps(auth_output, indent=2, sort_keys=True))
        return 0

    # 3. Pure Structural Validation Summary
    data_state_counts = {s.value: 0 for s in DataState}
    rights_decision_counts = {r.value: 0 for r in RightsDecision}
    for entry in registry.corpora:
        data_state_counts[entry.data_state.value] += 1
        rights_decision_counts[entry.rights_decision.value] += 1

    summary = {
        "corpus_count": len(registry.corpora),
        "counts_by_data_state": data_state_counts,
        "counts_by_rights_decision": rights_decision_counts,
        "registry_sha256": registry_sha256,
        "schema_version": registry.schema_version,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
