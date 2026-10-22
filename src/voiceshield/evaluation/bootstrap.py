"""Clustered bootstrap confidence interval estimation for audio deepfake metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn import metrics as sk_metrics

from voiceshield.evaluation.metrics import (
    compute_classification_metrics,
    compute_eer_interpolated,
)


def compute_clustered_bootstrap_cis(
    y_true: np.ndarray[Any, np.dtype[np.int_]],
    scores_higher_spoof: np.ndarray[Any, np.dtype[np.float64]],
    cluster_ids: np.ndarray[Any, np.dtype[Any]],
    threshold: float = 0.65,
    n_iterations: int = 500,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict[str, tuple[float, float] | None]:
    """Calculate non-parametric bootstrap confidence intervals clustered by speaker or source sample.

    Clustering ensures that non-independent samples (such as overlapping windows from the
    same speaker or utterance) do not produce spuriously narrow confidence intervals.

    Args:
        y_true: Ground truth binary labels (0 = bonafide, 1 = spoof).
        scores_higher_spoof: Detection scores where higher indicates spoof.
        cluster_ids: Array of cluster identifiers (e.g. speaker_id or source_utterance_id).
        threshold: Decision threshold for operating point metrics.
        n_iterations: Number of bootstrap resamples.
        ci_level: Nominal confidence level (default: 0.95 for 95% CI).
        seed: Random seed for deterministic reproducibility.

    Returns:
        Dictionary mapping metric names ('roc_auc', 'eer', 'f1', 'balanced_accuracy') to
        (ci_lower, ci_upper) tuples or None if undefined.
    """
    if len(np.unique(y_true)) < 2:
        return {
            "roc_auc": None,
            "eer": None,
            "f1": None,
            "balanced_accuracy": None,
        }

    unique_clusters = np.unique(cluster_ids)
    num_clusters = len(unique_clusters)
    if num_clusters < 2:
        return {
            "roc_auc": None,
            "eer": None,
            "f1": None,
            "balanced_accuracy": None,
        }

    # Pre-index cluster rows for fast vectorized lookup
    cluster_to_idx: dict[Any, np.ndarray[Any, np.dtype[np.int_]]] = {}
    for c in unique_clusters:
        cluster_to_idx[c] = np.where(cluster_ids == c)[0]

    rng = np.random.default_rng(seed)

    boot_roc_auc: list[float] = []
    boot_eer: list[float] = []
    boot_f1: list[float] = []
    boot_bacc: list[float] = []

    for _ in range(n_iterations):
        # Sample clusters with replacement
        sampled_clusters = rng.choice(unique_clusters, size=num_clusters, replace=True)
        sample_indices = np.concatenate([cluster_to_idx[c] for c in sampled_clusters])

        sub_y = y_true[sample_indices]
        sub_scores = scores_higher_spoof[sample_indices]

        # Guard: must have both classes in sample
        if len(np.unique(sub_y)) < 2:
            continue

        # ROC-AUC
        auc_val = float(sk_metrics.roc_auc_score(sub_y, sub_scores))
        boot_roc_auc.append(auc_val)

        # EER
        eer_val, _ = compute_eer_interpolated(sub_y, sub_scores)
        if eer_val is not None:
            boot_eer.append(eer_val)

        # Operating point metrics
        cm = compute_classification_metrics(sub_y, sub_scores, threshold)
        if cm.f1 is not None:
            boot_f1.append(cm.f1)
        if cm.balanced_accuracy is not None:
            boot_bacc.append(cm.balanced_accuracy)

    alpha = (1.0 - ci_level) / 2.0
    results: dict[str, tuple[float, float] | None] = {}

    def _extract_ci(vals: list[float]) -> tuple[float, float] | None:
        if len(vals) < 10:
            return None
        low = float(np.percentile(vals, 100.0 * alpha))
        high = float(np.percentile(vals, 100.0 * (1.0 - alpha)))
        return round(low, 4), round(high, 4)

    results["roc_auc"] = _extract_ci(boot_roc_auc)
    results["eer"] = _extract_ci(boot_eer)
    results["f1"] = _extract_ci(boot_f1)
    results["balanced_accuracy"] = _extract_ci(boot_bacc)

    return results
