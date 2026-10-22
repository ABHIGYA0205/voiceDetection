"""Command-line interface for offline audio file and directory inference."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from voiceshield.config.loader import load_config
from voiceshield.models.aasist_adapter import (
    AASIST_L_DEFAULT_CONFIG,
    AASISTLDetector,
    CheckpointNotFoundError,
)
from voiceshield.models.base import BaseDetector
from voiceshield.pipeline.file_evaluator import FileInferenceResult, evaluate_audio
from voiceshield.vendor.aasist import AASISTModel

SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac"}


def build_parser() -> argparse.ArgumentParser:
    """Construct argument parser for offline inference CLI."""
    parser = argparse.ArgumentParser(
        prog="voiceshield-infer",
        description="Run offline VoiceShield deepfake voice detection on audio files or directories.",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to an audio file (WAV/FLAC) or directory containing audio files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional destination path for detailed results (.jsonl or .csv).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to VoiceShield YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to AASIST-L checkpoint (.pth).",
    )
    parser.add_argument(
        "--hop-seconds",
        type=float,
        default=1.0,
        help="Sliding window hop step in seconds (default: 1.0s).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Decision threshold for synthetic speech warning (default: 0.65).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["mps", "cpu"],
        help="Preferred computation device ('mps' or 'cpu').",
    )
    parser.add_argument(
        "--mock-model",
        action="store_true",
        help="Instantiate architecture with mock weights for offline testing without checkpoint.",
    )
    return parser


def format_cli_summary_table(results: list[FileInferenceResult]) -> str:
    """Format human-readable table of file inference results."""
    lines: list[str] = [
        "=========================================================================================================",
        " VoiceShield Offline File Evaluation Summary",
        " (Advisory output only. Does NOT prove caller identity and is NOT a fraud probability)",
        "=========================================================================================================",
        f"{'File':<35} | {'Status':<10} | {'Duration':<8} | {'Windows':<7} | {'Mean Prob':<10} | {'Peak Prob':<10} | {'Decision'}",
        f"{'-' * 35}-|-{'-' * 10}-|-{'-' * 8}-|-{'-' * 7}-|-{'-' * 10}-|-{'-' * 10}-|-{'-' * 12}",
    ]

    for res in results:
        fname = Path(res.file_path).name
        if len(fname) > 35:
            fname = fname[:32] + "..."
        if res.status == "success":
            decision = "SUSPECTED" if res.is_synthetic_suspected else "Authentic"
            lines.append(
                f"{fname:<35} | {res.status:<10} | {res.duration_seconds:<8.1f} | "
                f"{res.total_windows:<7} | {res.mean_synthetic_likelihood:<10.1%} | "
                f"{res.max_synthetic_likelihood:<10.1%} | {decision}"
            )
        else:
            err = res.error_message or "Unknown"
            if len(err) > 30:
                err = err[:27] + "..."
            lines.append(
                f"{fname:<35} | {res.status:<10} | {'-':<8} | {'-':<7} | {'-':<10} | {'-':<10} | {err}"
            )

    return "\n".join(lines)


def save_cli_results(results: list[FileInferenceResult], output_path: Path) -> None:
    """Save batch results to JSONL or CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix == ".jsonl":
        with open(output_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(r.model_dump_json() + "\n")
    elif suffix == ".csv":
        fieldnames = [
            "file_path",
            "status",
            "duration_seconds",
            "total_windows",
            "mean_synthetic_likelihood",
            "max_synthetic_likelihood",
            "is_synthetic_suspected",
            "temporal_rule_explanation",
            "error_message",
        ]
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(
                    {
                        "file_path": r.file_path,
                        "status": r.status,
                        "duration_seconds": r.duration_seconds,
                        "total_windows": r.total_windows,
                        "mean_synthetic_likelihood": r.mean_synthetic_likelihood,
                        "max_synthetic_likelihood": r.max_synthetic_likelihood,
                        "is_synthetic_suspected": r.is_synthetic_suspected,
                        "temporal_rule_explanation": r.temporal_rule_explanation,
                        "error_message": r.error_message or "",
                    }
                )


def main(argv: list[str] | None = None) -> int:
    """Main CLI entrypoint for file inference."""
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config) if args.config else None

    # 1. Initialize detector
    detector: BaseDetector
    if args.mock_model:
        mock_raw = AASISTModel(AASIST_L_DEFAULT_CONFIG)
        detector = AASISTLDetector(_instantiated_model=mock_raw, preferred_device=args.device)
    else:
        try:
            detector = AASISTLDetector(
                checkpoint_path=args.checkpoint,
                preferred_device=args.device,
                threshold=args.threshold,
            )
        except CheckpointNotFoundError as exc:
            sys.stderr.write(f"\nERROR: {exc}\n")
            return 1

    # 2. Collect target files
    input_path = Path(args.input)
    if not input_path.exists():
        sys.stderr.write(f"ERROR: Input path does not exist: {input_path}\n")
        return 1

    audio_files: list[Path] = []
    unsupported_files: list[Path] = []

    if input_path.is_file():
        if input_path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
            audio_files.append(input_path)
        else:
            unsupported_files.append(input_path)
    else:
        for p in sorted(input_path.rglob("*")):
            if p.is_file():
                if p.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
                    audio_files.append(p)
                else:
                    unsupported_files.append(p)

    results: list[FileInferenceResult] = []

    # Record unsupported files cleanly
    for u in unsupported_files:
        results.append(
            FileInferenceResult(
                file_path=str(u),
                status="unsupported",
                error_message=f"Unsupported format '{u.suffix}'. Only WAV and FLAC are supported.",
            )
        )

    # 3. Evaluate each audio file, continuing past individual corrupt files
    for a in audio_files:
        res = evaluate_audio(
            source=a,
            detector=detector,
            config=cfg,
            hop_seconds=args.hop_seconds,
            threshold=args.threshold,
        )
        results.append(res)

    # 4. Print human-readable summary
    print(format_cli_summary_table(results))

    # 5. Save detailed results if requested
    if args.output:
        save_cli_results(results, args.output)
        print(f"\nSaved detailed inference records to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
