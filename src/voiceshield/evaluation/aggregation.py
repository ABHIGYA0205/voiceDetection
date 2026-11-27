"""Rules for aggregating multi-window audio scores into file-level detection scores."""

from __future__ import annotations

from enum import StrEnum
from itertools import pairwise

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
    window_index: int = Field(ge=0, description="Zero-indexed window sequence number.")
    continuity_segment_id: int = Field(default=0, ge=0, description="Audio continuity segment.")
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
            has_consecutive = any(
                right.window_index == left.window_index + 1
                and right.continuity_segment_id == left.continuity_segment_id
                and left.score >= temporal_threshold
                and right.score >= temporal_threshold
                for left, right in pairwise(sorted_wins)
            )
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
