#!/usr/bin/env python3
"""Validation and cross-validation CLI for VoiceShield corpus rights-evidence dossiers.

Validates immutable evidence schema, enforces non-authorizing signal constraints,
computes exact byte SHA-256 hashes, and cross-validates with configs/corpus_registry.yaml.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from voiceshield.data.corpus_registry import (
    CorpusValidationError,
    compute_registry_bytes_hash,
    load_corpus_registry,
)
from voiceshield.data.corpus_rights_evidence import (
    AccessMode,
    CorpusEvidenceCrossValidationError,
    CorpusEvidenceValidationError,
    EvidenceSignal,
    compute_evidence_bytes_hash,
    cross_validate_evidence_and_registry,
    load_corpus_rights_evidence,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate VoiceShield corpus rights-evidence dossiers and cross-validate with registry."
    )
    parser.add_argument(
        "evidence_file",
        nargs="?",
        default=None,
        help="Path to corpus_rights_evidence.yaml (positional).",
    )
    parser.add_argument(
        "registry_file",
        nargs="?",
        default=None,
        help="Path to corpus_registry.yaml (positional, default: configs/corpus_registry.yaml).",
    )
    parser.add_argument(
        "--evidence",
        dest="opt_evidence",
        default=None,
        help="Path to corpus_rights_evidence.yaml (option).",
    )
    parser.add_argument(
        "--registry",
        dest="opt_registry",
        default=None,
        help="Path to corpus_registry.yaml (option).",
    )

    args = parser.parse_args()

    # Resolve paths (flags override positional)
    evidence_arg = args.opt_evidence or args.evidence_file or "configs/corpus_rights_evidence.yaml"
    registry_arg = args.opt_registry or args.registry_file or "configs/corpus_registry.yaml"

    evidence_path = Path(evidence_arg)
    registry_path = Path(registry_arg)

    # 1. Read each file exactly once and compute exact-byte hashes
    try:
        if not evidence_path.is_file():
            sys.stderr.write(f"Evidence file not found: {evidence_path}\n")
            return 1
        evidence_bytes = evidence_path.read_bytes()
        evidence_sha256 = compute_evidence_bytes_hash(evidence_bytes)

        if not registry_path.is_file():
            sys.stderr.write(f"Registry file not found: {registry_path}\n")
            return 1
        registry_bytes = registry_path.read_bytes()
        registry_sha256 = compute_registry_bytes_hash(registry_bytes)
    except Exception as exc:
        sys.stderr.write(f"IO Error reading files: {exc}\n")
        return 1

    # 2. Structural Parsing and Invariant Validation
    try:
        evidence = load_corpus_rights_evidence(evidence_bytes)
    except CorpusEvidenceValidationError as err:
        sys.stderr.write(f"Evidence Validation Failed: {err}\n")
        return 1
    except Exception as err:
        sys.stderr.write(f"Unexpected Evidence Error: {err}\n")
        return 1

    try:
        registry = load_corpus_registry(registry_bytes)
    except CorpusValidationError as err:
        sys.stderr.write(f"Registry Validation Failed: {err}\n")
        return 1
    except Exception as err:
        sys.stderr.write(f"Unexpected Registry Error: {err}\n")
        return 1

    # 3. Cross-Validation against Frozen Registry
    try:
        result = cross_validate_evidence_and_registry(evidence, registry)
        if not result.is_valid:
            sys.stderr.write("Cross-Validation Failed:\n")
            for reason in result.errors:
                sys.stderr.write(f"  - {reason}\n")
            return 1
    except CorpusEvidenceCrossValidationError as err:
        sys.stderr.write(f"Cross-Validation Error: {err}\n")
        return 1
    except Exception as err:
        sys.stderr.write(f"Unexpected Cross-Validation Error: {err}\n")
        return 1

    # 4. Summarize access modes and evidence signals
    access_modes = {
        d.corpus_id: d.access_mode.value
        for d in sorted(evidence.dossiers, key=lambda x: x.corpus_id)
    }
    counts_by_access_mode = {mode.value: 0 for mode in AccessMode}
    for d in evidence.dossiers:
        counts_by_access_mode[d.access_mode.value] += 1

    counts_by_signal = {sig.value: 0 for sig in EvidenceSignal}
    for d in evidence.dossiers:
        for sig in (
            d.research_evaluation.signal,
            d.research_training.signal,
            d.product_evaluation.signal,
            d.product_training.signal,
            d.raw_redistribution.signal,
            d.attribution.signal,
        ):
            counts_by_signal[sig.value] += 1

    summary = {
        "access_modes": access_modes,
        "checked_dossiers_count": len(evidence.dossiers),
        "counts_by_access_mode": counts_by_access_mode,
        "counts_by_signal": counts_by_signal,
        "evidence_corpus_ids": list(result.evidence_corpus_ids),
        "evidence_sha256": evidence_sha256,
        "registry_corpus_ids": list(result.registry_corpus_ids),
        "registry_sha256": registry_sha256,
        "schema_version": evidence.schema_version,
        "status": "valid",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
