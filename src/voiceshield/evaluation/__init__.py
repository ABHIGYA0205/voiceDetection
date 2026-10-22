"""Scientifically rigorous evaluation toolkit for binary audio deepfake detection."""

from __future__ import annotations

from voiceshield.evaluation.aggregation import (
    AggregationRule,
    WindowScoreRecord,
    aggregate_window_scores,
)
from voiceshield.evaluation.bootstrap import compute_clustered_bootstrap_cis
from voiceshield.evaluation.evaluator import (
    FullEvaluationReport,
    GroupEvaluationResult,
    LatencyMetrics,
    ThresholdOptimizationError,
    evaluate_deepfake_detector,
)
from voiceshield.evaluation.metrics import (
    ClassificationMetrics,
    ConfusionMatrix,
    CurveMetrics,
    ScoreDirection,
    compute_all_curve_metrics,
    compute_brier_and_ece,
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_eer_interpolated,
    compute_fpr_at_target_tprs,
    compute_min_dcf,
    normalize_scores_to_higher_is_spoof,
    validate_score_direction,
)
from voiceshield.evaluation.reporting import (
    export_metrics_csv,
    generate_evaluation_plots,
    generate_markdown_summary,
    save_evaluation_report,
)

__all__ = [
    "AggregationRule",
    "ClassificationMetrics",
    "ConfusionMatrix",
    "CurveMetrics",
    "FullEvaluationReport",
    "GroupEvaluationResult",
    "LatencyMetrics",
    "ScoreDirection",
    "ThresholdOptimizationError",
    "WindowScoreRecord",
    "aggregate_window_scores",
    "compute_all_curve_metrics",
    "compute_brier_and_ece",
    "compute_classification_metrics",
    "compute_clustered_bootstrap_cis",
    "compute_confusion_matrix",
    "compute_eer_interpolated",
    "compute_fpr_at_target_tprs",
    "compute_min_dcf",
    "evaluate_deepfake_detector",
    "export_metrics_csv",
    "generate_evaluation_plots",
    "generate_markdown_summary",
    "normalize_scores_to_higher_is_spoof",
    "save_evaluation_report",
    "validate_score_direction",
]
