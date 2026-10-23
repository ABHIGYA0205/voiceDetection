"""Report generation in Markdown, CSV, and publication-quality plots."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

import numpy as np

# Ensure matplotlib uses headless Agg backend and writable config directory
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn import metrics as sk_metrics

from voiceshield.data.schema import ManifestRecord
from voiceshield.evaluation.evaluator import FullEvaluationReport, GroupEvaluationResult
from voiceshield.evaluation.metrics import (
    ScoreDirection,
    compute_eer_interpolated,
    normalize_scores_to_higher_is_spoof,
)


def _fmt(val: float | None, pct: bool = False, decimals: int = 4) -> str:
    """Format floating point numbers or report N/A for undefined values."""
    if val is None or np.isnan(val):
        return "N/A"
    if pct:
        return f"{val:.1%}"
    return f"{val:.{decimals}f}"


def _fmt_ci(ci: tuple[float, float] | None, pct: bool = False) -> str:
    """Format confidence interval tuple."""
    if ci is None:
        return "—"
    low, high = ci
    if pct:
        return f"[{low:.1%}, {high:.1%}]"
    return f"[{low:.4f}, {high:.4f}]"


def generate_markdown_summary(report: FullEvaluationReport) -> str:
    """Generate human-readable Markdown summary report with comprehensive tables."""
    cov = report.score_coverage
    partial_warn_block = ""
    if cov.partial_scores_allowed and cov.missing_score_count > 0:
        partial_warn_block = (
            "\n> ⚠️ **WARNING**: Evaluation was conducted with partial score coverage "
            f"({cov.coverage_fraction:.1%}). "
            f"{cov.missing_score_count} manifest record(s) were missing scores. "
            "Reported metrics DO NOT represent the complete manifest.\n"
        )

    brier_val = (
        _fmt(report.overall.curves.brier_score)
        if report.overall.curves.brier_score is not None
        else "N/A (no calibrator artifact supplied)"
    )
    ece_val = (
        _fmt(report.overall.curves.ece, pct=True)
        if report.overall.curves.ece is not None
        else "N/A (no calibrator artifact supplied)"
    )

    lines: list[str] = [
        "# VoiceShield Scientific Detection Evaluation Report",
        "",
        "> **Notice**: VoiceShield produces acoustic anomaly and synthetic speech likelihood metrics. "
        "Outputs do **NOT** prove caller identity and do **NOT** constitute fraud probabilities. "
        "Recommend independent secondary verification before taking advisory action.",
        partial_warn_block,
        "## 1. Score Coverage & Manifest Accounting",
        "",
        f"- **Manifest Total Records**: `{cov.manifest_record_count}`",
        f"- **Manifest Unique Sample IDs**: `{cov.manifest_unique_id_count}`",
        f"- **Provided Scores**: `{cov.provided_score_count}`",
        f"- **Matched Scores**: `{cov.matched_score_count}`",
        f"- **Missing Scores**: `{cov.missing_score_count}`",
        f"- **Unknown Scores**: `{cov.unknown_score_count}`",
        f"- **Coverage Fraction**: `{cov.coverage_fraction:.2%}`",
        f"- **Partial Scores Allowed**: `{cov.partial_scores_allowed}`",
        "",
        "## 2. Provenance & Evaluation Metadata",
        "",
        f"- **Operating Decision Threshold**: `{report.operating_threshold}` ({report.deployment_threshold_source})",
        f"- **Score Direction**: `{report.metadata.get('score_direction', 'unknown')}`",
        f"- **Aggregation Strategy**: `{report.metadata.get('aggregation_rule', 'none')}`",
        f"- **Total Manifest Records Evaluated**: `{report.overall.sample_count}` (Spoofs: `{report.overall.spoof_count}`, Bona Fide: `{report.overall.bonafide_count}`)",
        f"- **Config SHA-256 Hash**: `{report.metadata.get('config_hash', 'N/A')}`",
        f"- **Manifest SHA-256 Hash**: `{report.metadata.get('manifest_hash', 'N/A')}`",
        f"- **Model / Checkpoint Hash**: `{report.metadata.get('model_checkpoint_hash', 'N/A')}`",
        "",
        "---",
        "",
        "## 3. Overall Discrimination & Calibration Performance",
        "",
        "| Metric | Value | 95% Clustered Bootstrap CI | Description |",
        "| :--- | :--- | :--- | :--- |",
        f"| **EER (Equal Error Rate)** | `{_fmt(report.overall.curves.eer, pct=True)}` | `{_fmt_ci(report.overall.bootstrap_cis_95.get('eer'), pct=True)}` | Linear-interpolated crossover error |",
        f"| **ROC-AUC** | `{_fmt(report.overall.curves.roc_auc)}` | `{_fmt_ci(report.overall.bootstrap_cis_95.get('roc_auc'))}` | Area under ROC curve |",
        f"| **PR-AUC** | `{_fmt(report.overall.curves.pr_auc)}` | — | Area under Precision-Recall curve |",
        f"| **minDCF** | `{_fmt(report.overall.curves.min_dcf)}` | — | Normalized minimum Detection Cost Function |",
        f"| **F1 Score** | `{_fmt(report.overall.classification.f1)}` | `{_fmt_ci(report.overall.bootstrap_cis_95.get('f1'))}` | Harmonic mean of Precision & Recall |",
        f"| **Balanced Accuracy** | `{_fmt(report.overall.classification.balanced_accuracy, pct=True)}` | `{_fmt_ci(report.overall.bootstrap_cis_95.get('balanced_accuracy'), pct=True)}` | (Sensitivity + Specificity) / 2 |",
        f"| **Precision (PPV)** | `{_fmt(report.overall.classification.precision, pct=True)}` | — | True spoofs / all flagged spoofs |",
        f"| **Recall (TPR)** | `{_fmt(report.overall.classification.recall, pct=True)}` | — | Correctly detected spoofs |",
        f"| **Specificity (TNR)** | `{_fmt(report.overall.classification.specificity, pct=True)}` | — | Correctly accepted bona fide |",
        f"| **False Positive Rate** | `{_fmt(report.overall.classification.fpr, pct=True)}` | — | False alarms on authentic speech |",
        f"| **Brier Score** | `{brier_val}` | — | Probability calibration accuracy (N/A: no calibrator artifact supplied) |",
        f"| **Expected Calibration Error** | `{ece_val}` | — | Expected calibration deviation (N/A: no calibrator artifact supplied) |",
        "",
    ]

    # FPR @ target TPRs
    if report.overall.curves.fpr_at_tpr:
        lines.extend(
            [
                "### Operating Sensitivity Points",
                "",
                "| Target TPR (Sensitivity) | Operating FPR (False Alarm Rate) |",
                "| :--- | :--- |",
            ]
        )
        for tpr_key, fpr_val in sorted(report.overall.curves.fpr_at_tpr.items()):
            lines.append(f"| **{float(tpr_key):.0%} TPR** | `{_fmt(fpr_val, pct=True)}` |")
        lines.append("")

    # Confusion Matrix
    cm = report.overall.classification.confusion_matrix
    lines.extend(
        [
            "### Confusion Matrix (Operating Threshold)",
            "",
            "| | Predicted Bonafide | Predicted Spoof (Warning) |",
            "| :--- | :--- | :--- |",
            f"| **Actual Bonafide** | **TN**: `{cm.tn}` | **FP**: `{cm.fp}` |",
            f"| **Actual Spoof** | **FN**: `{cm.fn}` | **TP**: `{cm.tp}` |",
            "",
            "---",
            "",
        ]
    )

    # Sliced Tables Helper
    def append_slice_table(title: str, groups: list[GroupEvaluationResult]) -> None:
        if not groups:
            return
        lines.extend(
            [
                f"## {title}",
                "",
                "| Subset | Total | Spoof | Bonafide | EER | ROC-AUC | minDCF | Recall | Specificity | F1 | Status |",
                "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
            ]
        )
        for g in groups:
            warn = g.curves.single_class_warning or g.classification.single_class_warning or "Valid"
            if len(warn) > 25:
                warn = warn[:22] + "..."
            lines.append(
                f"| **{g.group_name}** | {g.sample_count} | {g.spoof_count} | {g.bonafide_count} | "
                f"`{_fmt(g.curves.eer, pct=True)}` | `{_fmt(g.curves.roc_auc)}` | `{_fmt(g.curves.min_dcf)}` | "
                f"`{_fmt(g.classification.recall, pct=True)}` | `{_fmt(g.classification.specificity, pct=True)}` | "
                f"`{_fmt(g.classification.f1)}` | *{warn}* |"
            )
        lines.extend(["", "---", ""])

    append_slice_table(
        "3. Breakdown by Synthesis / Voice Conversion Generator", report.by_generator
    )
    append_slice_table("4. Breakdown by Source Dataset", report.by_source_dataset)
    append_slice_table("5. Breakdown by Language", report.by_language)
    append_slice_table("6. Breakdown by Codec / Acoustic Channel", report.by_codec)
    append_slice_table("7. Breakdown by Replay Condition", report.by_replay)
    append_slice_table("8. Breakdown by Utterance Duration Bucket", report.by_duration_bucket)

    # Cross Condition Analysis
    if report.cross_conditions:
        lines.extend(
            [
                "## 9. Cross-Condition & Generalization Performance",
                "",
                "| Condition | Samples | Spoofs | EER | ROC-AUC | minDCF | F1 | Balanced Acc |",
                "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
            ]
        )
        for c_res in report.cross_conditions.values():
            lines.append(
                f"| **{c_res.group_name}** | {c_res.sample_count} | {c_res.spoof_count} | "
                f"`{_fmt(c_res.curves.eer, pct=True)}` | `{_fmt(c_res.curves.roc_auc)}` | "
                f"`{_fmt(c_res.curves.min_dcf)}` | `{_fmt(c_res.classification.f1)}` | "
                f"`{_fmt(c_res.classification.balanced_accuracy, pct=True)}` |"
            )
        lines.extend(["", "---", ""])

    # Latency / RTF diagnostics
    if report.latency:
        lines.extend(
            [
                "## 10. Inference Speed & Real-Time Factor (RTF)",
                "",
                f"- **Samples Timed**: `{report.latency.total_samples_timed}`",
                f"- **Median Latency (p50)**: `{report.latency.p50_latency_ms:.1f} ms`",
                f"- **95th Percentile Latency (p95)**: `{report.latency.p95_latency_ms:.1f} ms`",
                f"- **Mean Latency**: `{report.latency.mean_latency_ms:.1f} ms`",
                f"- **Total Audio Processed**: `{report.latency.total_audio_seconds:.1f} s`",
                f"- **Total Compute Time**: `{report.latency.total_inference_seconds:.2f} s`",
                f"- **Real-Time Factor (RTF)**: `{report.latency.real_time_factor:.4f}` "
                f"({'Faster than real-time' if report.latency.real_time_factor < 1.0 else 'Slower than real-time'})",
                "",
                "---",
                "",
            ]
        )

    return "\n".join(lines)


def export_metrics_csv(report: FullEvaluationReport, output_csv_path: Path) -> None:
    """Export tabular metrics for overall and all slices to CSV."""
    all_groups: list[GroupEvaluationResult] = [report.overall]
    all_groups.extend(report.by_generator)
    all_groups.extend(report.by_source_dataset)
    all_groups.extend(report.by_language)
    all_groups.extend(report.by_codec)
    all_groups.extend(report.by_replay)
    all_groups.extend(report.by_duration_bucket)
    all_groups.extend(report.cross_conditions.values())

    fieldnames = [
        "group_name",
        "sample_count",
        "spoof_count",
        "bonafide_count",
        "operating_threshold",
        "tp",
        "fp",
        "tn",
        "fn",
        "precision",
        "recall",
        "specificity",
        "fpr",
        "f1",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc",
        "eer",
        "min_dcf",
        "brier_score",
        "ece",
        "single_class_warning",
    ]

    with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for g in all_groups:
            cm = g.classification.confusion_matrix
            writer.writerow(
                {
                    "group_name": g.group_name,
                    "sample_count": g.sample_count,
                    "spoof_count": g.spoof_count,
                    "bonafide_count": g.bonafide_count,
                    "operating_threshold": report.operating_threshold,
                    "tp": cm.tp,
                    "fp": cm.fp,
                    "tn": cm.tn,
                    "fn": cm.fn,
                    "precision": g.classification.precision
                    if g.classification.precision is not None
                    else "",
                    "recall": g.classification.recall
                    if g.classification.recall is not None
                    else "",
                    "specificity": g.classification.specificity
                    if g.classification.specificity is not None
                    else "",
                    "fpr": g.classification.fpr if g.classification.fpr is not None else "",
                    "f1": g.classification.f1 if g.classification.f1 is not None else "",
                    "balanced_accuracy": g.classification.balanced_accuracy
                    if g.classification.balanced_accuracy is not None
                    else "",
                    "roc_auc": g.curves.roc_auc if g.curves.roc_auc is not None else "",
                    "pr_auc": g.curves.pr_auc if g.curves.pr_auc is not None else "",
                    "eer": g.curves.eer if g.curves.eer is not None else "",
                    "min_dcf": g.curves.min_dcf if g.curves.min_dcf is not None else "",
                    "brier_score": g.curves.brier_score if g.curves.brier_score is not None else "",
                    "ece": g.curves.ece if g.curves.ece is not None else "",
                    "single_class_warning": g.curves.single_class_warning
                    or g.classification.single_class_warning
                    or "",
                }
            )


def generate_evaluation_plots(
    records: list[ManifestRecord],
    scores_dict: dict[str, float],
    direction: ScoreDirection,
    output_dir: Path,
) -> dict[str, Path]:
    """Generate and save publication-quality evaluation plots (ROC, DET, Score Distributions)."""
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    generated_plots: dict[str, Path] = {}

    y_list: list[int] = []
    raw_score_list: list[float] = []
    for r in records:
        if r.sample_id in scores_dict:
            y_list.append(1 if r.label == "spoof" else 0)
            raw_score_list.append(scores_dict[r.sample_id])

    y_arr = np.array(y_list, dtype=int)
    raw_scores = np.array(raw_score_list, dtype=np.float64)

    if len(np.unique(y_arr)) < 2:
        return generated_plots

    scores_norm = normalize_scores_to_higher_is_spoof(raw_scores, direction)

    # 1. ROC Curve
    fprs, tprs, _ = sk_metrics.roc_curve(y_arr, scores_norm, pos_label=1)
    auc_val = sk_metrics.roc_auc_score(y_arr, scores_norm)
    eer_val, _ = compute_eer_interpolated(y_arr, scores_norm)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    ax.plot(fprs, tprs, color="#2563eb", lw=2, label=f"ROC (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="#94a3b8", linestyle="--", label="Random Chance")
    if eer_val is not None:
        ax.plot(
            [eer_val],
            [1.0 - eer_val],
            marker="o",
            markersize=6,
            color="#dc2626",
            label=f"EER = {eer_val:.2%}",
        )
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.05))
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title("Receiver Operating Characteristic (ROC)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    roc_path = plot_dir / "roc_curve.png"
    fig.tight_layout()
    fig.savefig(roc_path)
    plt.close(fig)
    generated_plots["roc_curve"] = roc_path

    # 2. Score Distributions (Bonafide vs Spoof)
    spoof_scores = scores_norm[y_arr == 1]
    bonafide_scores = scores_norm[y_arr == 0]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.hist(
        bonafide_scores,
        bins=30,
        alpha=0.6,
        color="#10b981",
        label="Bonafide (Authentic)",
        density=True,
    )
    ax.hist(
        spoof_scores, bins=30, alpha=0.6, color="#ef4444", label="Spoof (Synthetic)", density=True
    )
    ax.set_xlabel("Normalized Score (Higher = Spoof)")
    ax.set_ylabel("Empirical Density")
    ax.set_title("Detection Score Separation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    dist_path = plot_dir / "score_distributions.png"
    fig.tight_layout()
    fig.savefig(dist_path)
    plt.close(fig)
    generated_plots["score_distributions"] = dist_path

    return generated_plots


def save_evaluation_report(
    report: FullEvaluationReport,
    output_dir: Path | str,
    records: list[ManifestRecord] | None = None,
    scores_dict: dict[str, float] | None = None,
    direction: ScoreDirection = ScoreDirection.HIGHER_IS_SPOOF,
) -> Path:
    """Save complete evaluation report (summary.md, metrics.csv, and plots) to a versioned directory."""
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save Markdown summary
    summary_md = generate_markdown_summary(report)
    md_path = target_dir / "summary.md"
    md_path.write_text(summary_md, encoding="utf-8")

    # 2. Save Metrics CSV
    csv_path = target_dir / "metrics.csv"
    export_metrics_csv(report, csv_path)

    # 3. Save plots if raw data provided
    if records is not None and scores_dict is not None:
        generate_evaluation_plots(
            records=records,
            scores_dict=scores_dict,
            direction=direction,
            output_dir=target_dir,
        )

    return target_dir
