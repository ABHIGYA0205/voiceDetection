"""Command-line interface for the scientific evaluation toolkit."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
        "--allow-partial-scores",
        action="store_true",
        default=False,
        help="Permit partial score evaluation when some manifest sample IDs are missing from scores (default: False).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to VoiceShield configuration YAML file.",
    )
    return parser


SCORE_CANDIDATE_KEYS = [
    "synthetic_score",
    "uncalibrated_synthetic_score",
    "score",
    "mean_synthetic_score",
    "mean_synthetic_likelihood",
    "synthetic_likelihood",
    "calibrated_synthetic_likelihood",
]
ID_CANDIDATE_KEYS = ["sample_id", "file_path", "path"]
WINDOW_SCORE_KEYS = [
    "synthetic_score",
    "uncalibrated_synthetic_score",
    "score",
    "calibrated_synthetic_likelihood",
    "raw_spoof_logit",
]


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Hook for json.loads/json.load that rejects duplicate keys in objects."""
    res: dict[str, Any] = {}
    for k, v in pairs:
        if k in res:
            raise ValueError(f"Duplicate JSON object key found: {k!r}")
        res[k] = v
    return res


def _extract_sample_id(data: dict[str, Any], context: str = "") -> str:
    for k in ID_CANDIDATE_KEYS:
        if k in data:
            val = data[k]
            if val is None or (isinstance(val, str) and not val.strip()):
                raise ValueError(
                    f"Empty or missing sample identifier for key '{k}' {context}".strip()
                )
            if isinstance(val, str):
                return val.strip()
            return str(val).strip()
    raise ValueError(
        f"Missing sample identifier {context} (checked {ID_CANDIDATE_KEYS}) in: {data}".strip()
    )


def _extract_numeric_value(val: Any, key_name: str, context: str = "") -> float:
    if val is None:
        raise ValueError(f"Null score value for key '{key_name}' {context}".strip())
    if isinstance(val, str) and not val.strip():
        raise ValueError(f"Empty score value for key '{key_name}' {context}".strip())
    if isinstance(val, bool):
        raise ValueError(
            f"Boolean score values are not permitted for key '{key_name}' {context}: {val}".strip()
        )
    try:
        num = float(val)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Non-numeric score value for key '{key_name}' {context}: {val}".strip()
        ) from exc
    import numpy as np

    if not np.isfinite(num):
        raise ValueError(
            f"Non-finite score value (NaN or Inf) for key '{key_name}' {context}: {num}".strip()
        )
    return num


def _extract_score(data: dict[str, Any], context: str = "") -> float:
    extracted_scores: list[tuple[str, float]] = []
    for k in SCORE_CANDIDATE_KEYS:
        if k in data:
            val = data[k]
            if val is not None and not (isinstance(val, str) and not val.strip()):
                sc = _extract_numeric_value(val, key_name=k, context=context)
                extracted_scores.append((k, sc))

    if not extracted_scores:
        for k in SCORE_CANDIDATE_KEYS:
            if k in data:
                _extract_numeric_value(data[k], key_name=k, context=context)
        raise ValueError(
            f"Missing score {context} (checked keys: {SCORE_CANDIDATE_KEYS}) in: {data}".strip()
        )

    first_key, first_val = extracted_scores[0]
    for key, val in extracted_scores[1:]:
        if val != first_val:
            raise ValueError(
                f"Conflicting score fields {context}: '{first_key}'={first_val} and '{key}'={val} have different values."
            )

    return first_val


def _extract_window_score(data: dict[str, Any], context: str = "") -> float:
    extracted_scores: list[tuple[str, float]] = []
    for k in WINDOW_SCORE_KEYS:
        if k in data:
            val = data[k]
            if val is not None and not (isinstance(val, str) and not val.strip()):
                sc = _extract_numeric_value(val, key_name=k, context=context)
                extracted_scores.append((k, sc))

    if not extracted_scores:
        for k in WINDOW_SCORE_KEYS:
            if k in data:
                _extract_numeric_value(data[k], key_name=k, context=context)
        raise ValueError(
            f"Missing window score {context} (checked keys: {WINDOW_SCORE_KEYS}) in: {data}".strip()
        )

    first_key, first_val = extracted_scores[0]
    for key, val in extracted_scores[1:]:
        if val != first_val:
            raise ValueError(
                f"Conflicting window score fields {context}: '{first_key}'={first_val} and '{key}'={val} have different values."
            )

    return first_val


def load_scores_file(
    path: Path,
) -> tuple[dict[str, float] | list[WindowScoreRecord], dict[str, float] | None]:
    """Parse scores and optional latency timings from JSONL, JSON, or CSV with strict validation."""
    import numpy as np

    suffix = path.suffix.lower()
    sample_scores: dict[str, float] = {}
    window_records: list[WindowScoreRecord] = []
    timing_data: dict[str, float] = {}
    seen_window_keys: set[tuple[str, int]] = set()

    if suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    obj = json.loads(line_str, object_pairs_hook=_reject_duplicate_pairs)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at line {line_num} in {path}: {exc}") from exc
                if not isinstance(obj, dict):
                    raise ValueError(
                        f"JSONL line {line_num} in {path} must be a JSON object, got {type(obj)}"
                    )

                sid = _extract_sample_id(obj, context=f"at line {line_num}")

                has_windows = "windows" in obj and isinstance(obj["windows"], list)
                has_sample_score = any(k in obj for k in SCORE_CANDIDATE_KEYS)

                if has_windows and has_sample_score:
                    raise ValueError(
                        f"Ambiguous mixture at line {line_num}: sample '{sid}' contains both "
                        "sample-level score and window-level scores ('windows')."
                    )

                if has_windows:
                    win_list = obj["windows"]
                    if not win_list:
                        raise ValueError(
                            f"Empty 'windows' list for sample '{sid}' at line {line_num}"
                        )
                    for w_idx, w in enumerate(win_list):
                        if not isinstance(w, dict):
                            raise ValueError(
                                f"Window item {w_idx} for sample '{sid}' at line {line_num} must be a dict"
                            )
                        actual_idx = w.get("window_index", w_idx)
                        if isinstance(actual_idx, bool) or not isinstance(actual_idx, int):
                            raise ValueError(
                                f"Invalid window_index for sample '{sid}' at line {line_num}: {actual_idx}"
                            )
                        win_key = (sid, actual_idx)
                        if win_key in seen_window_keys:
                            raise ValueError(
                                f"Duplicate (sample_id, window_index) window record: {win_key}"
                            )
                        seen_window_keys.add(win_key)

                        w_score = _extract_window_score(
                            w, context=f"for sample '{sid}', window {actual_idx}"
                        )
                        start_sec = float(w.get("start_seconds", 0.0))
                        end_sec = float(w.get("end_seconds", 4.0))
                        window_records.append(
                            WindowScoreRecord(
                                sample_id=sid,
                                window_index=actual_idx,
                                score=w_score,
                                duration_seconds=max(0.0, end_sec - start_sec),
                            )
                        )
                elif has_sample_score:
                    score_val = _extract_score(
                        obj, context=f"for sample '{sid}' at line {line_num}"
                    )
                    if sid in sample_scores:
                        raise ValueError(
                            f"Duplicate sample-level score ID '{sid}' at line {line_num}"
                        )
                    sample_scores[sid] = score_val
                else:
                    raise ValueError(f"Missing score for sample '{sid}' at line {line_num}")

                if "duration_seconds" in obj and "latency_seconds" in obj:
                    lat = obj["latency_seconds"]
                    if lat is not None and not isinstance(lat, bool):
                        try:
                            lat_f = float(lat)
                            if np.isfinite(lat_f):
                                timing_data[sid] = lat_f
                        except (ValueError, TypeError):
                            pass

        if len(window_records) > 0 and len(sample_scores) > 0:
            raise ValueError(
                f"Ambiguous mixture of sample-level scores ({len(sample_scores)} samples) and "
                f"window-level scores ({len(window_records)} windows) in {path}."
            )

    elif suffix == ".json":
        with open(path, encoding="utf-8") as f:
            try:
                data = json.load(f, object_pairs_hook=_reject_duplicate_pairs)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
            if isinstance(data, dict):
                for k, v in data.items():
                    if not isinstance(k, str) or not k.strip():
                        raise ValueError(f"Empty or invalid sample ID key in JSON mapping: {k!r}")
                    sc = _extract_numeric_value(v, key_name=k, context="in JSON dictionary")
                    sample_scores[k.strip()] = sc
            elif isinstance(data, list):
                for idx, item in enumerate(data, start=1):
                    if not isinstance(item, dict):
                        raise ValueError(
                            f"Item {idx} in JSON list must be an object, got {type(item)}"
                        )
                    sid = _extract_sample_id(item, context=f"at item {idx}")
                    has_windows = "windows" in item and isinstance(item["windows"], list)
                    has_sample_score = any(k in item for k in SCORE_CANDIDATE_KEYS)
                    if has_windows and has_sample_score:
                        raise ValueError(
                            f"Ambiguous mixture at item {idx}: sample '{sid}' contains both "
                            "sample score and windows."
                        )
                    if has_windows:
                        win_list = item["windows"]
                        if not win_list:
                            raise ValueError(
                                f"Empty 'windows' list for sample '{sid}' at item {idx}"
                            )
                        for w_idx, w in enumerate(win_list):
                            if not isinstance(w, dict):
                                raise ValueError(
                                    f"Window item {w_idx} for sample '{sid}' must be a dict"
                                )
                            actual_idx = w.get("window_index", w_idx)
                            if isinstance(actual_idx, bool) or not isinstance(actual_idx, int):
                                raise ValueError(
                                    f"Invalid window_index for sample '{sid}': {actual_idx}"
                                )
                            win_key = (sid, actual_idx)
                            if win_key in seen_window_keys:
                                raise ValueError(
                                    f"Duplicate (sample_id, window_index) window record: {win_key}"
                                )
                            seen_window_keys.add(win_key)
                            w_score = _extract_window_score(
                                w, context=f"for sample '{sid}', window {actual_idx}"
                            )
                            start_sec = float(w.get("start_seconds", 0.0))
                            end_sec = float(w.get("end_seconds", 4.0))
                            window_records.append(
                                WindowScoreRecord(
                                    sample_id=sid,
                                    window_index=actual_idx,
                                    score=w_score,
                                    duration_seconds=max(0.0, end_sec - start_sec),
                                )
                            )
                    elif has_sample_score:
                        score_val = _extract_score(
                            item, context=f"for sample '{sid}' at item {idx}"
                        )
                        if sid in sample_scores:
                            raise ValueError(
                                f"Duplicate sample-level score ID '{sid}' at item {idx}"
                            )
                        sample_scores[sid] = score_val
                    else:
                        raise ValueError(f"Missing score for sample '{sid}' at item {idx}")
            else:
                raise ValueError(
                    f"JSON file {path} must contain a JSON object or list, got {type(data)}"
                )

        if len(window_records) > 0 and len(sample_scores) > 0:
            raise ValueError(
                f"Ambiguous mixture of sample-level scores ({len(sample_scores)} samples) and "
                f"window-level scores ({len(window_records)} windows) in {path}."
            )

    elif suffix == ".csv":
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError(f"CSV file is empty or missing headers: {path}")
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ValueError(f"Duplicate column headers found in CSV: {reader.fieldnames}")
            for row_num, row in enumerate(reader, start=2):
                sid = _extract_sample_id(row, context=f"at CSV row {row_num}")
                if sid in sample_scores:
                    raise ValueError(
                        f"Duplicate sample-level score ID: '{sid}' at CSV row {row_num}"
                    )
                score_val = _extract_score(row, context=f"at CSV row {row_num}")
                sample_scores[sid] = score_val
                if "latency_seconds" in row and row["latency_seconds"]:
                    try:
                        lat_val = float(row["latency_seconds"])
                        if np.isfinite(lat_val):
                            timing_data[sid] = lat_val
                    except ValueError:
                        pass
    else:
        raise ValueError(
            f"Unsupported score file extension '{suffix}'. Expected .jsonl, .json, or .csv"
        )

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

    cfg = load_config(args.config)

    # Load scores inside validation try-catch
    if not args.scores.exists():
        sys.stderr.write(f"ERROR: Scores file not found: {args.scores}\n")
        return 1

    try:
        scores_obj, timings = load_scores_file(args.scores)
    except (ValueError, csv.Error, json.JSONDecodeError) as exc:
        sys.stderr.write(f"\nVALIDATION ERROR: {exc}\n")
        return 1

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
            allow_partial_scores=args.allow_partial_scores,
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
