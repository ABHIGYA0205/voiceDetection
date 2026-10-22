"""Rules for aggregating multi-window audio scores into file-level detection scores."""

from __future__ import annotations

from enum import StrEnum

import numpy as np
from pydantic import BaseModel, Field


class AggregationRule(StrEnum):
    """Aggregation function applied across multiple temporal windows of an utterance."""

    MEAN = "mean"
    MAX = "max"
    TOP_K_MEAN = "top_k_mean"
    TEMPORAL_CONSECUTIVE = "temporal_consecutive"
    PASS_THROUGH = "pass_through"


class WindowScoreRecord(BaseModel):
    """An individual evaluated window score belonging to an utterance."""

    sample_id: str = Field(description="Parent utterance sample identifier.")
    window_index: int = Field(description="Zero-indexed window sequence number.")
    score: float = Field(description="Window detection score or probability.")
    duration_seconds: float = Field(default=4.0, description="Window duration in seconds.")
    active_speech_seconds: float = Field(
        default=4.0, description="Active speech duration in window."
    )


def aggregate_window_scores(
    records: list[WindowScoreRecord],
    rule: AggregationRule = AggregationRule.MEAN,
    top_k: int = 2,
    temporal_threshold: float = 0.65,
) -> dict[str, float]:
    """Aggregate window scores into a single scalar score per sample_id.

    Args:
        records: Collection of WindowScoreRecords across utterances.
        rule: Aggregation strategy to apply.
        top_k: Parameter k when rule is TOP_K_MEAN (default: 2).
        temporal_threshold: Threshold for TEMPORAL_CONSECUTIVE rule.

    Returns:
        Dictionary mapping sample_id to aggregated scalar score.
    """
    grouped: dict[str, list[WindowScoreRecord]] = {}
    for r in records:
        grouped.setdefault(r.sample_id, []).append(r)

    results: dict[str, float] = {}

    for sample_id, win_list in grouped.items():
        # Sort by window index
        sorted_wins = sorted(win_list, key=lambda w: w.window_index)
        scores = np.array([w.score for w in sorted_wins], dtype=np.float64)

        if len(scores) == 0:
            continue

        if rule == AggregationRule.MEAN:
            results[sample_id] = float(np.mean(scores))
        elif rule == AggregationRule.MAX:
            results[sample_id] = float(np.max(scores))
        elif rule == AggregationRule.TOP_K_MEAN:
            k = min(len(scores), top_k)
            # Top k largest scores
            top_scores = np.sort(scores)[-k:]
            results[sample_id] = float(np.mean(top_scores))
        elif rule == AggregationRule.TEMPORAL_CONSECUTIVE:
            # Check for at least 2 consecutive windows >= temporal_threshold
            has_consecutive = False
            for i in range(len(scores) - 1):
                if scores[i] >= temporal_threshold and scores[i + 1] >= temporal_threshold:
                    has_consecutive = True
                    break
            if has_consecutive:
                results[sample_id] = float(np.max(scores))
            else:
                results[sample_id] = float(np.mean(scores))
        elif rule == AggregationRule.PASS_THROUGH:
            # Use first/single score directly
            results[sample_id] = float(scores[0])
        else:
            results[sample_id] = float(np.mean(scores))

    return results
