"""Command-line interface for the scientific evaluation toolkit."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from voiceshield.config.loader import load_config
from voiceshield.evaluation.aggregation import AggregationRule, WindowScoreRecord
from voiceshield.evaluation.evaluator import (
    ThresholdOptimizationError,
    evaluate_deepfake_detector,
)
from voiceshield.evaluation.metrics import ScoreDirection
from voiceshield.evaluation.reporting import save_evaluation_report


def build_parser() -> argparse.ArgumentParser:
    """Construct argument parser for evaluation CLI."""
    parser = argparse.ArgumentParser(
        prog="voiceshield-eval",
        description="Scientifically evaluate audio deepfake detection predictions against a canonical manifest.",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Path to canonical manifest CSV or JSONL file.",
    )
    parser.add_argument(
        "--scores",
        required=True,
        type=Path,
        help="Path to predictions JSONL, JSON, or CSV file containing sample_id and score.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/evaluation"),
        help="Destination directory for evaluation report and plots.",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="higher_is_spoof",
        choices=["higher_is_spoof", "higher_is_bonafide"],
        help="Score direction convention (default: higher_is_spoof).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional explicit deployment decision threshold.",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default="mean",
        choices=["mean", "max", "top_k_mean", "temporal_consecutive", "pass_through"],
        help="Aggregation rule if window scores are provided (default: mean).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to VoiceShield configuration YAML file.",
    )
    return parser


def load_scores_file(
    path: Path,
) -> tuple[dict[str, float] | list[WindowScoreRecord], dict[str, float] | None]:
    """Parse scores and optional latency timings from JSONL, JSON, or CSV."""
    suffix = path.suffix.lower()
    sample_scores: dict[str, float] = {}
    window_records: list[WindowScoreRecord] = []
    timing_data: dict[str, float] = {}

    if suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                sid = obj.get("sample_id") or obj.get("file_path") or obj.get("path")
                if not sid:
                    continue
                # Check if it has window records
                if "windows" in obj and isinstance(obj["windows"], list):
                    for w in obj["windows"]:
                        window_records.append(
                            WindowScoreRecord(
                                sample_id=sid,
                                window_index=w.get("window_index", 0),
                                score=float(
                                    w.get(
                                        "calibrated_synthetic_likelihood",
                                        w.get("raw_spoof_logit", 0.0),
                                    )
                                ),
                                duration_seconds=float(w.get("end_seconds", 4.0))
                                - float(w.get("start_seconds", 0.0)),
                            )
                        )
                # Check sample-level score
                score = (
                    obj.get("mean_synthetic_likelihood")
                    or obj.get("synthetic_likelihood")
                    or obj.get("score")
                    or obj.get("calibrated_synthetic_likelihood")
                )
                if score is not None:
                    sample_scores[sid] = float(score)

                if "duration_seconds" in obj and "latency_seconds" in obj:
                    timing_data[sid] = float(obj["latency_seconds"])

    elif suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                sample_scores = {k: float(v) for k, v in data.items()}
            elif isinstance(data, list):
                for item in data:
                    sid = item.get("sample_id") or item.get("file_path")
                    if sid and "score" in item:
                        sample_scores[sid] = float(item["score"])
    elif suffix == ".csv":
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row.get("sample_id") or row.get("file_path")
                score_str = (
                    row.get("score")
                    or row.get("mean_synthetic_likelihood")
                    or row.get("synthetic_likelihood")
                )
                if sid and score_str:
                    sample_scores[sid] = float(score_str)

    scores_out: dict[str, float] | list[WindowScoreRecord]
    if len(window_records) > 0 and len(sample_scores) == 0:
        scores_out = window_records
    else:
        scores_out = sample_scores

    return scores_out, timing_data if timing_data else None


def main(argv: list[str] | None = None) -> int:
    """Execute evaluation CLI workflow."""
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config) if args.config else None

    # Load scores
    if not args.scores.exists():
        sys.stderr.write(f"ERROR: Scores file not found: {args.scores}\n")
        return 1

    scores_obj, timings = load_scores_file(args.scores)
    dir_enum = ScoreDirection(args.direction)
    agg_enum = AggregationRule(args.aggregation)

    # Versioned output directory
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    target_out_dir = args.output_dir / f"eval_{timestamp}"

    try:
        report = evaluate_deepfake_detector(
            manifest=args.manifest,
            scores=scores_obj,
            direction=dir_enum,
            deployment_threshold=args.threshold,
            aggregation_rule=agg_enum,
            config=cfg,
            timing_data=timings,
        )
    except ThresholdOptimizationError as exc:
        sys.stderr.write(f"\nGUARDRAIL ERROR: {exc}\n")
        return 2
    except ValueError as exc:
        sys.stderr.write(f"\nVALIDATION ERROR: {exc}\n")
        return 1

    save_evaluation_report(
        report=report,
        output_dir=target_out_dir,
    )

    print(f"\nSuccessfully generated evaluation report in: {target_out_dir}")
    print(
        f"- Overall EER: {report.overall.curves.eer:.2%}"
        if report.overall.curves.eer is not None
        else "- Overall EER: N/A"
    )
    print(
        f"- Overall ROC-AUC: {report.overall.curves.roc_auc:.4f}"
        if report.overall.curves.roc_auc is not None
        else "- Overall ROC-AUC: N/A"
    )
    print(
        f"- Operating Threshold: {report.operating_threshold} ({report.deployment_threshold_source})"
    )
    print(f"- Markdown Summary: {target_out_dir / 'summary.md'}")
    print(f"- Tabular CSV: {target_out_dir / 'metrics.csv'}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
