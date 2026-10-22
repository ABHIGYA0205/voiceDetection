"""Scientifically rigorous evaluation metrics for binary audio deepfake detection."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from sklearn import metrics as sk_metrics


class ScoreDirection(StrEnum):
    """Directional convention for detection scores."""

    HIGHER_IS_SPOOF = "higher_is_spoof"
    HIGHER_IS_BONAFIDE = "higher_is_bonafide"


class ConfusionMatrix(BaseModel):
    """Confusion matrix counts for binary classification."""

    tp: int = Field(description="True Positives: synthetic speech correctly flagged.")
    fp: int = Field(description="False Positives: authentic speech incorrectly flagged.")
    tn: int = Field(description="True Negatives: authentic speech correctly cleared.")
    fn: int = Field(description="False Negatives: synthetic speech missed.")
    total: int = Field(description="Total sample count.")


class ClassificationMetrics(BaseModel):
    """Operating point metrics at a discrete decision threshold."""

    threshold: float = Field(description="Decision threshold applied to scores.")
    confusion_matrix: ConfusionMatrix
    precision: float | None = Field(description="Precision (PPV) or None if undefined.")
    recall: float | None = Field(description="Recall (TPR / Sensitivity) or None if undefined.")
    specificity: float | None = Field(description="Specificity (TNR) or None if undefined.")
    fpr: float | None = Field(description="False Positive Rate (Fall-out) or None if undefined.")
    f1: float | None = Field(description="F1 Score (harmonic mean) or None if undefined.")
    balanced_accuracy: float | None = Field(
        description="Balanced Accuracy ((TPR + TNR) / 2) or None if undefined."
    )
    single_class_warning: str | None = Field(
        default=None, description="Explanation if certain metrics are mathematically undefined."
    )


class CurveMetrics(BaseModel):
    """Threshold-independent discrimination and calibration metrics."""

    roc_auc: float | None = Field(description="Area under the ROC curve (None if single-class).")
    pr_auc: float | None = Field(
        description="Area under Precision-Recall curve (None if single-class)."
    )
    eer: float | None = Field(
        description="Equal Error Rate via linear interpolation (None if single-class)."
    )
    eer_threshold: float | None = Field(
        description="Interpolated threshold achieving Equal Error Rate."
    )
    fpr_at_tpr: dict[str, float | None] = Field(
        default_factory=dict,
        description="FPR measured at specified target TPR levels (e.g. '0.95', '0.99').",
    )
    min_dcf: float | None = Field(
        description="Normalized minimum Detection Cost Function (minDCF)."
    )
    min_dcf_threshold: float | None = Field(
        description="Decision threshold minimizing the normalized DCF."
    )
    brier_score: float | None = Field(
        description="Brier score for calibrated probabilities (None if uncalibrated/logits)."
    )
    ece: float | None = Field(
        description="Expected Calibration Error (None if uncalibrated/logits)."
    )
    single_class_warning: str | None = Field(
        default=None, description="Explanation if curves are mathematically undefined."
    )


def validate_score_direction(
    y_true: np.ndarray[Any, np.dtype[np.int_]],
    scores: np.ndarray[Any, np.dtype[np.float64]],
    direction: ScoreDirection,
    tolerance: float = 0.05,
) -> None:
    """Validate that empirical scores align with the declared score direction.

    Args:
        y_true: Binary ground-truth labels (0 = bonafide, 1 = spoof).
        scores: Continuous raw or calibrated scores.
        direction: Declared ScoreDirection.
        tolerance: Allowable margin before raising an inversion error.

    Raises:
        ValueError: If scores contradict the declared direction.
    """
    unique = np.unique(y_true)
    if len(unique) < 2:
        return  # Cannot test direction on single-class data

    spoof_scores = scores[y_true == 1]
    bonafide_scores = scores[y_true == 0]

    if len(spoof_scores) == 0 or len(bonafide_scores) == 0:
        return

    mean_spoof = float(np.mean(spoof_scores))
    mean_bonafide = float(np.mean(bonafide_scores))

    if direction == ScoreDirection.HIGHER_IS_SPOOF:
        if mean_bonafide - mean_spoof > tolerance:
            raise ValueError(
                f"Declared direction is '{direction.value}', but empirical mean(bonafide) ({mean_bonafide:.4f}) "
                f"exceeds mean(spoof) ({mean_spoof:.4f}). Check if score convention is inverted."
            )
    elif direction == ScoreDirection.HIGHER_IS_BONAFIDE:
        if mean_spoof - mean_bonafide > tolerance:
            raise ValueError(
                f"Declared direction is '{direction.value}', but empirical mean(spoof) ({mean_spoof:.4f}) "
                f"exceeds mean(bonafide) ({mean_bonafide:.4f}). Check if score convention is inverted."
            )


def normalize_scores_to_higher_is_spoof(
    scores: np.ndarray[Any, np.dtype[np.float64]],
    direction: ScoreDirection,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Convert scores so that higher values consistently indicate higher spoof likelihood."""
    if direction == ScoreDirection.HIGHER_IS_BONAFIDE:
        # If bounded in [0, 1] probability
        if float(np.min(scores)) >= 0.0 and float(np.max(scores)) <= 1.0:
            return 1.0 - scores
        return -scores
    return scores.copy()


def compute_confusion_matrix(
    y_true: np.ndarray[Any, np.dtype[np.int_]],
    scores_higher_spoof: np.ndarray[Any, np.dtype[np.float64]],
    threshold: float,
) -> ConfusionMatrix:
    """Compute binary confusion matrix at the given threshold."""
    y_pred = (scores_higher_spoof >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return ConfusionMatrix(tp=tp, fp=fp, tn=tn, fn=fn, total=len(y_true))


def compute_classification_metrics(
    y_true: np.ndarray[Any, np.dtype[np.int_]],
    scores_higher_spoof: np.ndarray[Any, np.dtype[np.float64]],
    threshold: float,
) -> ClassificationMetrics:
    """Compute threshold-dependent metrics (precision, recall, specificity, F1, balanced accuracy)."""
    cm = compute_confusion_matrix(y_true, scores_higher_spoof, threshold)

    unique_classes = set(np.unique(y_true))
    single_class_msg: str | None = None
    if len(unique_classes) < 2:
        present_name = "spoof" if 1 in unique_classes else "bonafide"
        single_class_msg = (
            f"Single-class group ({present_name} only). Complementary rate metrics are undefined."
        )

    # Precision = TP / (TP + FP)
    precision: float | None = None
    if (cm.tp + cm.fp) > 0:
        precision = round(float(cm.tp / (cm.tp + cm.fp)), 4)

    # Recall (TPR) = TP / (TP + FN)
    recall: float | None = None
    if (cm.tp + cm.fn) > 0:
        recall = round(float(cm.tp / (cm.tp + cm.fn)), 4)

    # Specificity (TNR) = TN / (TN + FP)
    specificity: float | None = None
    if (cm.tn + cm.fp) > 0:
        specificity = round(float(cm.tn / (cm.tn + cm.fp)), 4)

    # FPR = FP / (TN + FP)
    fpr: float | None = None
    if (cm.tn + cm.fp) > 0:
        fpr = round(float(cm.fp / (cm.tn + cm.fp)), 4)

    # F1 = 2 * P * R / (P + R)
    f1: float | None = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = round(float(2 * (precision * recall) / (precision + recall)), 4)

    # Balanced Accuracy = (TPR + TNR) / 2
    balanced_acc: float | None = None
    if recall is not None and specificity is not None:
        balanced_acc = round(float((recall + specificity) / 2.0), 4)

    return ClassificationMetrics(
        threshold=round(float(threshold), 4),
        confusion_matrix=cm,
        precision=precision,
        recall=recall,
        specificity=specificity,
        fpr=fpr,
        f1=f1,
        balanced_accuracy=balanced_acc,
        single_class_warning=single_class_msg,
    )


def compute_eer_interpolated(
    y_true: np.ndarray[Any, np.dtype[np.int_]],
    scores_higher_spoof: np.ndarray[Any, np.dtype[np.float64]],
) -> tuple[float | None, float | None]:
    """Compute Equal Error Rate (EER) with exact linear interpolation across crossover thresholds.

    Returns:
        tuple of (eer, eer_threshold) or (None, None) if data contains only one class.
    """
    if len(np.unique(y_true)) < 2:
        return None, None

    # Compute ROC curve thresholds
    fprs, tprs, thresholds = sk_metrics.roc_curve(y_true, scores_higher_spoof, pos_label=1)
    fnrs = 1.0 - tprs

    # Difference between FPR and FNR
    # For high thresholds: FPR is small, FNR is large -> (FPR - FNR) < 0
    # For low thresholds: FPR is large, FNR is small -> (FPR - FNR) > 0
    diff = fprs - fnrs

    # Find where diff crosses zero
    crossover_indices = np.where(np.diff(np.sign(diff)) != 0)[0]

    if len(crossover_indices) == 0:
        # If diff never crosses zero (rare extreme separation), pick min absolute difference
        idx = int(np.argmin(np.abs(diff)))
        eer_val = float((fprs[idx] + fnrs[idx]) / 2.0)
        return round(eer_val, 6), round(float(thresholds[idx]), 6)

    idx = int(crossover_indices[0])
    # Linear interpolation between idx and idx + 1:
    # d1 = diff[idx], d2 = diff[idx+1]
    d1 = float(diff[idx])
    d2 = float(diff[idx + 1])
    denominator = d1 - d2

    if abs(denominator) < 1e-12:
        alpha = 0.5
    else:
        alpha = d1 / denominator

    alpha = float(np.clip(alpha, 0.0, 1.0))

    interp_fpr = float(fprs[idx] + alpha * (fprs[idx + 1] - fprs[idx]))
    interp_fnr = float(fnrs[idx] + alpha * (fnrs[idx + 1] - fnrs[idx]))
    eer_val = (interp_fpr + interp_fnr) / 2.0

    t1 = float(thresholds[idx])
    t2 = float(thresholds[idx + 1])
    if np.isinf(t1):
        interp_thresh = t2
    elif np.isinf(t2):
        interp_thresh = t1
    else:
        interp_thresh = float(t1 + alpha * (t2 - t1))

    return round(float(eer_val), 6), round(float(interp_thresh), 6)


def compute_fpr_at_target_tprs(
    y_true: np.ndarray[Any, np.dtype[np.int_]],
    scores_higher_spoof: np.ndarray[Any, np.dtype[np.float64]],
    target_tprs: list[float],
) -> dict[str, float | None]:
    """Measure False Positive Rate at target True Positive Rates (e.g. 0.90, 0.95, 0.99)."""
    if len(np.unique(y_true)) < 2:
        return {f"{t:.2f}": None for t in target_tprs}

    fprs, tprs, _ = sk_metrics.roc_curve(y_true, scores_higher_spoof, pos_label=1)

    result: dict[str, float | None] = {}
    for target in target_tprs:
        # Find points where TPR >= target
        valid_indices = np.where(tprs >= target)[0]
        if len(valid_indices) == 0:
            result[f"{target:.2f}"] = 1.0
        else:
            # First index where TPR >= target gives the minimum FPR at that sensitivity
            idx = int(valid_indices[0])
            result[f"{target:.2f}"] = round(float(fprs[idx]), 4)

    return result


def compute_min_dcf(
    y_true: np.ndarray[Any, np.dtype[np.int_]],
    scores_higher_spoof: np.ndarray[Any, np.dtype[np.float64]],
    c_miss: float = 1.0,
    c_fa: float = 10.0,
    p_target: float = 0.05,
) -> tuple[float | None, float | None]:
    """Compute normalized minimum Detection Cost Function (minDCF).

    Formula (NIST/ASVspoof standard):
        C_det(theta) = C_miss * P_target * P_miss(theta) + C_fa * (1 - P_target) * P_fa(theta)
        C_norm(theta) = C_det(theta) / min(C_miss * P_target, C_fa * (1 - P_target))
        minDCF = min_theta C_norm(theta)

    Returns:
        tuple of (min_dcf, optimal_threshold) or (None, None) if single-class.
    """
    if len(np.unique(y_true)) < 2:
        return None, None

    fprs, tprs, thresholds = sk_metrics.roc_curve(y_true, scores_higher_spoof, pos_label=1)
    fnrs = 1.0 - tprs

    cost_miss = c_miss * p_target * fnrs
    cost_fa = c_fa * (1.0 - p_target) * fprs
    c_det = cost_miss + cost_fa

    c_default = min(c_miss * p_target, c_fa * (1.0 - p_target))
    if c_default <= 0:
        return None, None

    c_norm = c_det / c_default
    best_idx = int(np.argmin(c_norm))

    return round(float(c_norm[best_idx]), 4), round(float(thresholds[best_idx]), 6)


def compute_brier_and_ece(
    y_true: np.ndarray[Any, np.dtype[np.int_]],
    scores: np.ndarray[Any, np.dtype[np.float64]],
    num_bins: int = 10,
) -> tuple[float | None, float | None]:
    """Compute Brier score and Expected Calibration Error (ECE) for calibrated probabilities.

    Returns (None, None) if scores are outside the valid probability bounds [0.0, 1.0].
    """
    if len(scores) == 0:
        return None, None

    min_s = float(np.min(scores))
    max_s = float(np.max(scores))

    # Guardrail: Only evaluate on calibrated probabilities
    if min_s < -1e-6 or max_s > 1.0 + 1e-6:
        return None, None

    probs = np.clip(scores, 0.0, 1.0)
    brier = float(np.mean((probs - y_true) ** 2))

    # Expected Calibration Error (ECE)
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    total_samples = len(y_true)
    ece_val = 0.0

    for i in range(num_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == num_bins - 1:
            in_bin = (probs >= low) & (probs <= high)
        else:
            in_bin = (probs >= low) & (probs < high)

        bin_count = int(np.sum(in_bin))
        if bin_count > 0:
            bin_conf = float(np.mean(probs[in_bin]))
            bin_acc = float(np.mean(y_true[in_bin]))
            ece_val += (bin_count / total_samples) * abs(bin_acc - bin_conf)

    return round(brier, 4), round(float(ece_val), 4)


def compute_all_curve_metrics(
    y_true: np.ndarray[Any, np.dtype[np.int_]],
    scores_higher_spoof: np.ndarray[Any, np.dtype[np.float64]],
    raw_scores: np.ndarray[Any, np.dtype[np.float64]],
    target_tprs: list[float] | None = None,
    c_miss: float = 1.0,
    c_fa: float = 10.0,
    p_target: float = 0.05,
    ece_bins: int = 10,
) -> CurveMetrics:
    """Compute complete set of threshold-independent discrimination and calibration metrics."""
    if target_tprs is None:
        target_tprs = [0.90, 0.95, 0.99]

    unique_classes = set(np.unique(y_true))
    if len(unique_classes) < 2:
        present_name = "spoof" if 1 in unique_classes else "bonafide"
        return CurveMetrics(
            roc_auc=None,
            pr_auc=None,
            eer=None,
            eer_threshold=None,
            fpr_at_tpr={f"{t:.2f}": None for t in target_tprs},
            min_dcf=None,
            min_dcf_threshold=None,
            brier_score=None,
            ece=None,
            single_class_warning=f"Single-class group ({present_name} only). Discrimination curves are undefined.",
        )

    # ROC-AUC
    roc_auc = round(float(sk_metrics.roc_auc_score(y_true, scores_higher_spoof)), 4)

    # PR-AUC
    pr_auc = round(float(sk_metrics.average_precision_score(y_true, scores_higher_spoof)), 4)

    # EER
    eer_val, eer_thresh = compute_eer_interpolated(y_true, scores_higher_spoof)

    # FPR @ target TPRs
    fpr_at_tpr = compute_fpr_at_target_tprs(y_true, scores_higher_spoof, target_tprs)

    # minDCF
    min_dcf_val, min_dcf_thresh = compute_min_dcf(
        y_true, scores_higher_spoof, c_miss=c_miss, c_fa=c_fa, p_target=p_target
    )

    # Calibration: check if raw_scores is a valid probability distribution
    brier, ece = compute_brier_and_ece(y_true, raw_scores, num_bins=ece_bins)

    return CurveMetrics(
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        eer=eer_val,
        eer_threshold=eer_thresh,
        fpr_at_tpr=fpr_at_tpr,
        min_dcf=min_dcf_val,
        min_dcf_threshold=min_dcf_thresh,
        brier_score=brier,
        ece=ece,
    )
