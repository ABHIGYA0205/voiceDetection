"""Command-line interface for offline audio file and directory inference."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

from voiceshield.config.loader import ConfigError, load_config
from voiceshield.config.overrides import apply_config_overrides
from voiceshield.models import (
    AASIST_L_DEFAULT_CONFIG,
    AASISTLDetector,
    CheckpointError,
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
        help="Path to AASIST-L checkpoint (.pth) (uses config if omitted).",
    )
    parser.add_argument(
        "--hop-seconds",
        type=float,
        default=None,
        help="Sliding window hop step in seconds (uses config if omitted).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold for synthetic speech warning (uses config if omitted).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["mps", "cpu"],
        help="Preferred computation device ('mps' or 'cpu') (uses config if omitted).",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        choices=["clovaai/aasist-l", "custom/aasist-l"],
        help="Model identity override ('clovaai/aasist-l' or 'custom/aasist-l').",
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
        f"{'File':<35} | {'Status':<10} | {'Duration':<8} | {'Windows':<7} | {'Mean Score':<10} | {'Peak Score':<10} | {'Decision'}",
        f"{'-' * 35}-|-{'-' * 10}-|-{'-' * 8}-|-{'-' * 7}-|-{'-' * 10}-|-{'-' * 10}-|-{'-' * 12}",
    ]

    for res in results:
        fname = Path(res.file_path).name
        if len(fname) > 35:
            fname = fname[:32] + "..."
        if res.status == "success":
            decision_state = res.final_decision_state
            if decision_state not in {"high", "uncertain", "low"}:
                decision_state = "high" if res.is_synthetic_suspected else "low"
            if decision_state == "high":
                decision = "SUSPECTED"
            elif decision_state == "uncertain":
                decision = "MIXED / UNCERTAIN"
            else:
                decision = "LOW ANOMALY"
            mean_str = (
                f"{res.mean_synthetic_score:.3f}" if res.mean_synthetic_score is not None else "N/A"
            )
            max_str = (
                f"{res.max_synthetic_score:.3f}" if res.max_synthetic_score is not None else "N/A"
            )
            lines.append(
                f"{fname:<35} | {res.status:<10} | {res.duration_seconds:<8.1f} | "
                f"{res.total_windows:<7} | {mean_str:<10} | "
                f"{max_str:<10} | {decision}"
            )
        elif res.status == "insufficient_evidence":
            lines.append(
                f"{fname:<35} | {res.status:<10} | {res.duration_seconds:<8.1f} | "
                f"{res.total_windows:<7} | {'N/A':<10} | {'N/A':<10} | ABSTAIN"
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
            "config_hash",
            "model_id",
            "provenance_status",
            "checkpoint_verified",
            "checkpoint_sha256",
            "duration_seconds",
            "total_windows",
            "mean_synthetic_score",
            "max_synthetic_score",
            "mean_synthetic_likelihood",
            "max_synthetic_likelihood",
            "raw_high_window_count",
            "uncertain_window_count",
            "high_window_count",
            "is_synthetic_suspected",
            "final_decision_state",
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
                        "config_hash": r.config_hash or "",
                        "model_id": r.metadata.get("model_id") or "",
                        "provenance_status": r.metadata.get("provenance_status") or "",
                        "checkpoint_verified": r.metadata.get("checkpoint_verified", False),
                        "checkpoint_sha256": r.metadata.get("checkpoint_sha256") or "",
                        "duration_seconds": r.duration_seconds,
                        "total_windows": r.total_windows,
                        "mean_synthetic_score": r.mean_synthetic_score
                        if r.mean_synthetic_score is not None
                        else "N/A",
                        "max_synthetic_score": r.max_synthetic_score
                        if r.max_synthetic_score is not None
                        else "N/A",
                        "mean_synthetic_likelihood": r.mean_synthetic_likelihood
                        if r.mean_synthetic_likelihood is not None
                        else "N/A",
                        "max_synthetic_likelihood": r.max_synthetic_likelihood
                        if r.max_synthetic_likelihood is not None
                        else "N/A",
                        "raw_high_window_count": r.raw_high_window_count,
                        "uncertain_window_count": r.uncertain_window_count,
                        "high_window_count": r.high_window_count,
                        "is_synthetic_suspected": r.is_synthetic_suspected,
                        "final_decision_state": r.final_decision_state,
                        "temporal_rule_explanation": r.temporal_rule_explanation,
                        "error_message": r.error_message or "",
                    }
                )


def main(argv: list[str] | None = None) -> int:
    """Main CLI entrypoint for file inference."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        base_config = load_config(args.config)
        effective_config = apply_config_overrides(
            base_config=base_config,
            model_id=args.model_id,
            checkpoint_path=str(args.checkpoint) if args.checkpoint is not None else None,
            device_preferred=args.device,
            hop_seconds=args.hop_seconds,
            threshold=args.threshold,
        )
    except ConfigError as exc:
        sys.stderr.write(f"CONFIG ERROR: {exc}\n")
        return 2

    effective_config_hash = effective_config.compute_hash()

    # 1. Initialize detector with effective config
    detector: BaseDetector
    if args.mock_model:
        mock_raw = AASISTModel(AASIST_L_DEFAULT_CONFIG)
        detector = AASISTLDetector(
            _instantiated_model=mock_raw,
            model_id="mock/aasist-l",
            preferred_device=effective_config.model.device_preferred,
            fallback_to_cpu=effective_config.model.fallback_to_cpu,
            threshold=effective_config.temporal_policy.threshold_high,
        )
    else:
        try:
            detector = AASISTLDetector(
                checkpoint_path=effective_config.model.checkpoint_path,
                model_id=effective_config.model.model_id,
                preferred_device=effective_config.model.device_preferred,
                fallback_to_cpu=effective_config.model.fallback_to_cpu,
                threshold=effective_config.temporal_policy.threshold_high,
            )
        except CheckpointError as exc:
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
    unsupported_meta: dict[str, Any] = {"config_hash": effective_config_hash}
    if hasattr(detector, "provenance"):
        prov = detector.provenance
        unsupported_meta.update(
            {
                "model_id": prov.model_id,
                "provenance_status": prov.provenance_status,
                "checkpoint_verified": prov.checkpoint_verified,
                "checkpoint_sha256": prov.checkpoint_sha256,
            }
        )

    for u in unsupported_files:
        results.append(
            FileInferenceResult(
                file_path=str(u),
                status="unsupported",
                error_message=f"Unsupported format '{u.suffix}'. Only WAV and FLAC are supported.",
                config_hash=effective_config_hash,
                metadata=unsupported_meta,
            )
        )

    # 3. Evaluate each audio file with effective config
    for a in audio_files:
        res = evaluate_audio(
            source=a,
            detector=detector,
            config=effective_config,
        )
        results.append(res)

    # 4. Print human-readable summary and config provenance
    print(f"Effective Config Hash: {effective_config_hash}")
    print(format_cli_summary_table(results))

    # 5. Save detailed results if requested
    if args.output:
        save_cli_results(results, args.output)
        print(f"\nSaved detailed inference records to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
