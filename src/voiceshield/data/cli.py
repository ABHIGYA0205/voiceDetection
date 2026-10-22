"""Command-line interface for dataset manifest validation, leakage checking, splitting, and summaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from voiceshield.data.io import ManifestIOError, load_manifest, save_manifest
from voiceshield.data.leakage import check_leakage
from voiceshield.data.splitter import (
    codec_holdout_split,
    generator_holdout_split,
    speaker_disjoint_split,
)
from voiceshield.data.summary import format_summary_tables, summarize_manifest


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser for manifest commands."""
    parser = argparse.ArgumentParser(
        prog="voiceshield-data",
        description="VoiceShield manifest validation, leakage detection, and partitioning tool.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. validate command
    val_parser = subparsers.add_parser(
        "validate", help="Validate manifest schema and check for cross-split leakage."
    )
    val_parser.add_argument(
        "--manifest", required=True, type=Path, help="Path to .csv or .jsonl manifest."
    )
    val_parser.add_argument(
        "--holdout-generators",
        nargs="*",
        default=None,
        help="Generator families that must NOT appear in training split.",
    )
    val_parser.add_argument(
        "--holdout-codecs",
        nargs="*",
        default=None,
        help="Codec conditions that must NOT appear in training split.",
    )

    # 2. split command
    split_parser = subparsers.add_parser(
        "split", help="Partition an un-split manifest deterministically."
    )
    split_parser.add_argument(
        "--manifest", required=True, type=Path, help="Path to input manifest."
    )
    split_parser.add_argument("--output", required=True, type=Path, help="Destination output path.")
    split_parser.add_argument(
        "--mode",
        choices=["speaker_disjoint", "generator_holdout", "codec_holdout"],
        default="speaker_disjoint",
        help="Partitioning strategy.",
    )
    split_parser.add_argument(
        "--holdouts",
        nargs="*",
        default=[],
        help="Holdout generator or codec names when using holdout modes.",
    )
    split_parser.add_argument("--train-ratio", type=float, default=0.70)
    split_parser.add_argument("--dev-ratio", type=float, default=0.15)
    split_parser.add_argument("--test-ratio", type=float, default=0.15)
    split_parser.add_argument("--seed", type=int, default=42)

    # 3. summary command
    sum_parser = subparsers.add_parser(
        "summary", help="Print summary breakdown tables for a manifest."
    )
    sum_parser.add_argument("--manifest", required=True, type=Path, help="Path to manifest file.")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute manifest CLI command."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        records = load_manifest(args.manifest)
    except ManifestIOError as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return 1

    if args.command == "validate":
        print(f"Loaded {len(records)} records from {args.manifest}. Running leakage analysis...")
        report = check_leakage(
            records=records,
            holdout_generators=args.holdout_generators,
            holdout_codecs=args.holdout_codecs,
        )

        if report.has_leakage:
            sys.stderr.write("FAILED: Data leakage or duplicate checks violated!\n")
            for msg in report.summary_messages:
                sys.stderr.write(f"  - {msg}\n")
            return 1

        print("SUCCESS: Manifest is schema-valid and zero leakage was detected.")
        return 0

    if args.command == "split":
        print(f"Partitioning {len(records)} records using mode='{args.mode}'...")
        if args.mode == "speaker_disjoint":
            split_records = speaker_disjoint_split(
                records,
                train_ratio=args.train_ratio,
                dev_ratio=args.dev_ratio,
                test_ratio=args.test_ratio,
                seed=args.seed,
            )
        elif args.mode == "generator_holdout":
            split_records = generator_holdout_split(
                records,
                holdout_generators=args.holdouts,
                train_ratio=args.train_ratio,
                dev_ratio=args.dev_ratio,
                test_ratio=args.test_ratio,
                seed=args.seed,
            )
        elif args.mode == "codec_holdout":
            split_records = codec_holdout_split(
                records,
                holdout_codecs=args.holdouts,
                train_ratio=args.train_ratio,
                dev_ratio=args.dev_ratio,
                test_ratio=args.test_ratio,
                seed=args.seed,
            )
        else:
            sys.stderr.write(f"Unknown split mode: {args.mode}\n")
            return 1

        # Check leakage immediately on generated partition
        report = check_leakage(
            split_records,
            holdout_generators=args.holdouts if args.mode == "generator_holdout" else None,
            holdout_codecs=args.holdouts if args.mode == "codec_holdout" else None,
        )
        if report.has_leakage:
            sys.stderr.write("ERROR: Partition generated with leakage!\n")
            for msg in report.summary_messages:
                sys.stderr.write(f"  - {msg}\n")
            return 1

        save_manifest(split_records, args.output)
        print(f"Successfully saved {len(split_records)} partitioned records to {args.output}.")
        return 0

    if args.command == "summary":
        summary = summarize_manifest(records)
        print(format_summary_tables(summary))
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
