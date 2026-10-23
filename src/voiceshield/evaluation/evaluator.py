"""Orchestrator for audio deepfake detection evaluation with scientific guardrails."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from voiceshield.config.schema import VoiceShieldConfig
from voiceshield.core.schema import Disclaimers
from voiceshield.data.io import load_manifest
from voiceshield.data.schema import ManifestRecord
from voiceshield.evaluation.aggregation import (
    AggregationRule,
    WindowScoreRecord,
    aggregate_window_scores,
)
from voiceshield.evaluation.bootstrap import compute_clustered_bootstrap_cis
from voiceshield.evaluation.metrics import (
    ClassificationMetrics,
    ConfusionMatrix,
    CurveMetrics,
    ScoreDirection,
    compute_all_curve_metrics,
    compute_classification_metrics,
    compute_eer_interpolated,
    normalize_scores_to_higher_is_spoof,
    validate_score_direction,
)


class ThresholdOptimizationError(RuntimeError):
    """Raised when threshold optimization is improperly requested on a test split."""


class LatencyMetrics(BaseModel):
    """Inference speed, latency percentiles, and real-time factor diagnostics."""

    total_samples_timed: int
    p50_latency_ms: float
    p95_latency_ms: float
    mean_latency_ms: float
    total_audio_seconds: float
    total_inference_seconds: float
    real_time_factor: float = Field(
        description="Ratio of inference time to audio duration (RTF < 1.0 means faster than real-time)."
    )


class GroupEvaluationResult(BaseModel):
    """Evaluation metrics for a specific slice or subset."""

    group_name: str
    sample_count: int
    spoof_count: int
    bonafide_count: int
    classification: ClassificationMetrics
    curves: CurveMetrics
    bootstrap_cis_95: dict[str, tuple[float, float] | None] = Field(default_factory=dict)


class ScoreCoverage(BaseModel):
    """Structured accounting of manifest records and ingested prediction scores."""

    manifest_record_count: int
    manifest_unique_id_count: int
    provided_score_count: int
    matched_score_count: int
    missing_score_count: int
    unknown_score_count: int
    coverage_fraction: float
    missing_sample_ids: list[str] = Field(default_factory=list)
    unknown_sample_ids: list[str] = Field(default_factory=list)
    partial_scores_allowed: bool = False


class FullEvaluationReport(BaseModel):
    """Complete evaluation report containing overall, sliced, cross-condition, and latency results."""

    overall: GroupEvaluationResult
    score_coverage: ScoreCoverage
    by_generator: list[GroupEvaluationResult] = Field(default_factory=list)
    by_source_dataset: list[GroupEvaluationResult] = Field(default_factory=list)
    by_language: list[GroupEvaluationResult] = Field(default_factory=list)
    by_codec: list[GroupEvaluationResult] = Field(default_factory=list)
    by_replay: list[GroupEvaluationResult] = Field(default_factory=list)
    by_duration_bucket: list[GroupEvaluationResult] = Field(default_factory=list)
    cross_conditions: dict[str, GroupEvaluationResult] = Field(
        default_factory=dict,
        description="Cross-condition results: 'matched', 'unseen_generator', 'unseen_channel', 'external_test'.",
    )
    deployment_threshold_source: str = Field(
        description="Source of deployment threshold ('explicit', 'selected_on_dev_eer', 'selected_on_train_eer')."
    )
    operating_threshold: float
    latency: LatencyMetrics | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    disclaimers: Disclaimers = Field(default_factory=Disclaimers)


def get_duration_bucket(duration: float) -> str:
    """Categorize audio duration into standard duration buckets."""
    if duration < 2.0:
        return "[0s - 2s)"
    elif duration < 4.0:
        return "[2s - 4s)"
    elif duration < 8.0:
        return "[4s - 8s)"
    else:
        return "[8s+]"


def evaluate_sample_group(
    group_name: str,
    records: list[ManifestRecord],
    scores_dict: dict[str, float],
    direction: ScoreDirection,
    threshold: float,
    config: VoiceShieldConfig | None = None,
    compute_bootstrap: bool = True,
) -> GroupEvaluationResult:
    """Evaluate discrimination and calibration metrics for a specific subset of manifest records."""
    effective_cfg = config if config is not None else VoiceShieldConfig()
    eval_cfg = effective_cfg.evaluation
    c_miss = eval_cfg.dcf.c_miss
    c_fa = eval_cfg.dcf.c_fa
    p_target = eval_cfg.dcf.p_target
    target_tprs = eval_cfg.target_tpr_levels
    boot_iters = eval_cfg.bootstrap_iterations
    boot_ci = eval_cfg.bootstrap_ci_level
    ece_bins = eval_cfg.ece_num_bins

    # Collect labels, raw scores, and cluster IDs
    y_list: list[int] = []
    raw_score_list: list[float] = []
    cluster_list: list[str] = []

    for r in records:
        if r.sample_id in scores_dict:
            y_list.append(1 if r.label == "spoof" else 0)
            raw_score_list.append(scores_dict[r.sample_id])
            # Grouping key for bootstrap: speaker_id if available, else sample_id
            cluster_list.append(r.speaker_id if r.speaker_id else r.sample_id)

    y_arr = np.array(y_list, dtype=int)
    raw_scores = np.array(raw_score_list, dtype=np.float64)
    cluster_arr = np.array(cluster_list)

    if len(y_arr) == 0:
        # Empty group
        cm = ClassificationMetrics(
            threshold=threshold,
            confusion_matrix=ConfusionMatrix(tp=0, fp=0, tn=0, fn=0, total=0),
            precision=None,
            recall=None,
            specificity=None,
            fpr=None,
            f1=None,
            balanced_accuracy=None,
            single_class_warning="Empty group.",
        )
        curves = CurveMetrics(
            roc_auc=None,
            pr_auc=None,
            eer=None,
            eer_threshold=None,
            min_dcf=None,
            min_dcf_threshold=None,
            brier_score=None,
            ece=None,
            single_class_warning="Empty group.",
        )
        return GroupEvaluationResult(
            group_name=group_name,
            sample_count=0,
            spoof_count=0,
            bonafide_count=0,
            classification=cm,
            curves=curves,
            bootstrap_cis_95={},
        )

    # Normalize scores so higher means spoof
    scores_norm = normalize_scores_to_higher_is_spoof(raw_scores, direction)

    # 1. Classification metrics at operating threshold
    cls_metrics = compute_classification_metrics(y_arr, scores_norm, threshold)

    # 2. Curve metrics
    curve_metrics = compute_all_curve_metrics(
        y_true=y_arr,
        scores_higher_spoof=scores_norm,
        raw_scores=raw_scores,
        target_tprs=target_tprs,
        c_miss=c_miss,
        c_fa=c_fa,
        p_target=p_target,
        ece_bins=ece_bins,
    )

    # 3. Bootstrap CIs
    boot_cis: dict[str, tuple[float, float] | None] = {}
    if compute_bootstrap and len(np.unique(y_arr)) >= 2:
        boot_cis = compute_clustered_bootstrap_cis(
            y_true=y_arr,
            scores_higher_spoof=scores_norm,
            cluster_ids=cluster_arr,
            threshold=threshold,
            n_iterations=boot_iters,
            ci_level=boot_ci,
        )

    return GroupEvaluationResult(
        group_name=group_name,
        sample_count=len(y_arr),
        spoof_count=int(np.sum(y_arr == 1)),
        bonafide_count=int(np.sum(y_arr == 0)),
        classification=cls_metrics,
        curves=curve_metrics,
        bootstrap_cis_95=boot_cis,
    )


def compute_latency_metrics(
    records: list[ManifestRecord],
    timing_data: dict[str, float],
) -> LatencyMetrics | None:
    """Compute latency percentiles and real-time factor."""
    latencies: list[float] = []
    total_audio = 0.0

    record_map = {r.sample_id: r for r in records}
    for sid, elapsed_sec in timing_data.items():
        if sid in record_map:
            latencies.append(elapsed_sec * 1000.0)  # ms
            total_audio += record_map[sid].duration

    if len(latencies) == 0:
        return None

    p50 = float(np.percentile(latencies, 50.0))
    p95 = float(np.percentile(latencies, 95.0))
    mean_lat = float(np.mean(latencies))
    total_infer = float(np.sum(latencies) / 1000.0)
    rtf = float(total_infer / total_audio) if total_audio > 0 else 0.0

    return LatencyMetrics(
        total_samples_timed=len(latencies),
        p50_latency_ms=round(p50, 2),
        p95_latency_ms=round(p95, 2),
        mean_latency_ms=round(mean_lat, 2),
        total_audio_seconds=round(total_audio, 2),
        total_inference_seconds=round(total_infer, 2),
        real_time_factor=round(rtf, 4),
    )


def evaluate_deepfake_detector(
    manifest: list[ManifestRecord] | str | Path,
    scores: dict[str, float] | list[WindowScoreRecord],
    direction: ScoreDirection = ScoreDirection.HIGHER_IS_SPOOF,
    deployment_threshold: float | None = None,
    aggregation_rule: AggregationRule = AggregationRule.MEAN,
    allow_partial_scores: bool = False,
    config: VoiceShieldConfig | None = None,
    timing_data: dict[str, float] | None = None,
    metadata: dict[str, Any] | None = None,
) -> FullEvaluationReport:
    """Evaluate deepfake detector with scientific guardrails and detailed slicing.

    Args:
        manifest: List of ManifestRecords or path to manifest CSV/JSONL file.
        scores: Dict mapping sample_id to score, OR list of WindowScoreRecords to aggregate.
        direction: Declared score direction (higher_is_spoof or higher_is_bonafide).
        deployment_threshold: Optional external deployment threshold.
        aggregation_rule: Aggregation strategy if window scores are supplied.
        allow_partial_scores: If True, allow missing scores for manifest records (default: False).
        config: VoiceShieldConfig object with evaluation and DCF settings.
        timing_data: Optional mapping of sample_id to elapsed inference seconds.
        metadata: Optional dictionary with run/model/commit hashes.

    Returns:
        FullEvaluationReport with overall, score coverage, sliced, cross-condition, and latency metrics.

    Raises:
        ThresholdOptimizationError: If asked to optimize a threshold on a test split, an unassigned manifest,
            or if the optimization split lacks classes or defined EER.
        ValueError: If manifest or scores violate integrity, coverage, or empirical direction constraints.
    """
    # 0. Validate explicit deployment threshold if provided
    if deployment_threshold is not None:
        if (
            isinstance(deployment_threshold, bool)
            or not isinstance(deployment_threshold, (int, float))
            or not np.isfinite(deployment_threshold)
        ):
            raise ValueError(
                f"Explicit deployment threshold must be a finite number, got {deployment_threshold}"
            )

    # 1. Load manifest records if path provided and check uniqueness
    records: list[ManifestRecord]
    if isinstance(manifest, (str, Path)):
        records = load_manifest(Path(manifest))
    else:
        records = list(manifest)

    if not records:
        raise ValueError("Cannot evaluate an empty manifest.")

    seen_manifest_ids: set[str] = set()
    dup_manifest_ids: list[str] = []
    for r in records:
        if r.sample_id in seen_manifest_ids:
            dup_manifest_ids.append(r.sample_id)
        seen_manifest_ids.add(r.sample_id)
    if dup_manifest_ids:
        raise ValueError(f"Duplicate sample_id found in manifest: {sorted(set(dup_manifest_ids))}")

    # 2. Ingest and validate scores
    scores_dict: dict[str, float]
    if isinstance(scores, list):
        # Validate window records
        seen_win_keys: set[tuple[str, int]] = set()
        for w in scores:
            if not isinstance(w.sample_id, str) or not w.sample_id.strip():
                raise ValueError("Window record missing or empty sample_id.")
            if (
                isinstance(w.score, bool)
                or not isinstance(w.score, (int, float))
                or not np.isfinite(w.score)
            ):
                raise ValueError(
                    f"Window score for sample '{w.sample_id}', window {w.window_index} "
                    f"must be a finite number, got {w.score}"
                )
            win_key = (w.sample_id, w.window_index)
            if win_key in seen_win_keys:
                raise ValueError(f"Duplicate (sample_id, window_index) window record: {win_key}")
            seen_win_keys.add(win_key)
        scores_dict = aggregate_window_scores(scores, rule=aggregation_rule)
    elif isinstance(scores, dict):
        scores_dict = {}
        for sid, sc in scores.items():
            if not isinstance(sid, str) or not sid.strip():
                raise ValueError("Missing or empty sample_id in scores mapping.")
            if isinstance(sc, bool) or not isinstance(sc, (int, float)) or not np.isfinite(sc):
                raise ValueError(f"Score for sample '{sid}' must be a finite number, got {sc}")
            scores_dict[sid] = float(sc)
    else:
        raise ValueError(
            f"Scores must be a dict mapping sample_id to float or a list of WindowScoreRecord, got {type(scores)}"
        )

    # 3. Score Coverage Accounting
    manifest_unique_ids = seen_manifest_ids
    provided_ids = set(scores_dict.keys())
    matched_ids = manifest_unique_ids.intersection(provided_ids)
    missing_ids = sorted(manifest_unique_ids - matched_ids)
    unknown_ids = sorted(provided_ids - manifest_unique_ids)

    manifest_record_count = len(records)
    manifest_unique_id_count = len(manifest_unique_ids)
    provided_score_count = len(scores_dict)
    matched_score_count = len(matched_ids)
    missing_score_count = len(missing_ids)
    unknown_score_count = len(unknown_ids)
    coverage_fraction = (
        matched_score_count / manifest_unique_id_count if manifest_unique_id_count > 0 else 0.0
    )

    # Unknown score IDs are strictly rejected in all modes
    if unknown_score_count > 0:
        raise ValueError(
            f"Provided scores contain {unknown_score_count} unknown sample ID(s) not present in manifest: "
            f"{unknown_ids[:10]}"
        )

    # Incomplete score coverage fails by default
    if not allow_partial_scores and missing_score_count > 0:
        raise ValueError(
            f"Incomplete score coverage: {missing_score_count} of {manifest_unique_id_count} manifest sample(s) "
            f"are missing scores ({coverage_fraction:.1%} coverage). Missing sample IDs: {missing_ids[:10]}. "
            "Use allow_partial_scores=True / --allow-partial-scores to explicitly permit partial evaluation."
        )

    if matched_score_count == 0:
        raise ValueError("No manifest samples matched the provided scores.")

    score_coverage = ScoreCoverage(
        manifest_record_count=manifest_record_count,
        manifest_unique_id_count=manifest_unique_id_count,
        provided_score_count=provided_score_count,
        matched_score_count=matched_score_count,
        missing_score_count=missing_score_count,
        unknown_score_count=unknown_score_count,
        coverage_fraction=coverage_fraction,
        missing_sample_ids=missing_ids,
        unknown_sample_ids=unknown_ids,
        partial_scores_allowed=allow_partial_scores,
    )

    # 4. Test score direction on matched records
    matched_y: list[int] = []
    matched_scores: list[float] = []
    for r in records:
        if r.sample_id in scores_dict:
            matched_y.append(1 if r.label == "spoof" else 0)
            matched_scores.append(scores_dict[r.sample_id])

    validate_score_direction(
        y_true=np.array(matched_y, dtype=int),
        scores=np.array(matched_scores, dtype=np.float64),
        direction=direction,
    )

    # 5. Enforce Threshold Guardrail: Test-Set Isolation and Split Policies
    operating_threshold: float
    threshold_source: str

    if deployment_threshold is not None:
        operating_threshold = float(deployment_threshold)
        threshold_source = "explicit_deployment_threshold"
    else:
        splits_present = {r.split for r in records}
        non_none_splits = {s for s in splits_present if s is not None}

        # 5.a. All records unassigned: refuse optimization
        if not non_none_splits:
            raise ThresholdOptimizationError(
                "Scientific Guardrail Violation: All records in manifest are unassigned (split=None). "
                "Refusing to optimize decision threshold over an unknown mixture. "
                "An explicit deployment threshold must be supplied."
            )

        # 5.b. Pure test manifest: refuse optimization
        if non_none_splits == {"test"}:
            raise ThresholdOptimizationError(
                "Scientific Guardrail Violation: Refusing to optimize decision threshold on split marked 'test'. "
                "The deployment threshold must be supplied explicitly or selected on a separate dev/train split."
            )

        candidate_records: list[ManifestRecord]
        source_prefix: str
        source_split_name: str

        if "dev" in splits_present:
            candidate_records = [r for r in records if r.split == "dev"]
            source_prefix = "selected_on_dev_eer"
            source_split_name = "dev"
        elif "train" in splits_present:
            candidate_records = [r for r in records if r.split == "train"]
            source_prefix = "selected_on_train_eer"
            source_split_name = "train"
        else:
            raise ThresholdOptimizationError(
                f"No valid 'dev' or 'train' split available for threshold optimization (found: {sorted(str(s) for s in non_none_splits)}). "
                "An explicit deployment threshold must be supplied."
            )

        # 5.c. Selection split must have complete score coverage
        missing_in_thresh = [
            r.sample_id for r in candidate_records if r.sample_id not in scores_dict
        ]
        if missing_in_thresh:
            raise ThresholdOptimizationError(
                f"Threshold selection split '{source_split_name}' has incomplete score coverage: "
                f"{len(missing_in_thresh)} of {len(candidate_records)} records missing scores. "
                "Threshold optimization requires complete score coverage on the selection split."
            )

        # 5.d. Selection split must contain both classes
        thresh_labels = {r.label for r in candidate_records}
        if len(thresh_labels) < 2:
            raise ThresholdOptimizationError(
                f"Threshold selection split '{source_split_name}' contains only one class ({thresh_labels}). "
                "Both bona-fide and spoof classes are required to optimize an operating threshold."
            )

        # 5.e. Compute interpolated EER on selection split
        thresh_y = np.array([1 if r.label == "spoof" else 0 for r in candidate_records], dtype=int)
        thresh_s = np.array([scores_dict[r.sample_id] for r in candidate_records], dtype=np.float64)
        thresh_norm = normalize_scores_to_higher_is_spoof(thresh_s, direction)

        eer_val, eer_th = compute_eer_interpolated(thresh_y, thresh_norm)
        if eer_th is None or not np.isfinite(eer_th):
            raise ThresholdOptimizationError(
                f"EER threshold optimization is undefined on '{source_split_name}' split. "
                "Cannot determine operating threshold."
            )

        operating_threshold = float(eer_th)
        threshold_source = f"{source_prefix} (threshold={operating_threshold:.4f})"

    # 6. Overall evaluation
    overall_res = evaluate_sample_group(
        group_name="Overall",
        records=records,
        scores_dict=scores_dict,
        direction=direction,
        threshold=operating_threshold,
        config=config,
        compute_bootstrap=True,
    )

    # 6. Breakdown by Generator
    by_generator: list[GroupEvaluationResult] = []
    generator_groups: dict[str, list[ManifestRecord]] = {}
    for r in records:
        gen = r.generator_family or "Authentic (None)"
        generator_groups.setdefault(gen, []).append(r)

    for gen_name in sorted(generator_groups.keys()):
        by_generator.append(
            evaluate_sample_group(
                group_name=f"Generator: {gen_name}",
                records=generator_groups[gen_name],
                scores_dict=scores_dict,
                direction=direction,
                threshold=operating_threshold,
                config=config,
                compute_bootstrap=False,
            )
        )

    # 7. Breakdown by Source Dataset
    by_source: list[GroupEvaluationResult] = []
    source_groups: dict[str, list[ManifestRecord]] = {}
    for r in records:
        source_groups.setdefault(r.source_dataset, []).append(r)

    for src_name in sorted(source_groups.keys()):
        by_source.append(
            evaluate_sample_group(
                group_name=f"Dataset: {src_name}",
                records=source_groups[src_name],
                scores_dict=scores_dict,
                direction=direction,
                threshold=operating_threshold,
                config=config,
                compute_bootstrap=False,
            )
        )

    # 8. Breakdown by Language
    by_language: list[GroupEvaluationResult] = []
    lang_groups: dict[str, list[ManifestRecord]] = {}
    for r in records:
        lang_groups.setdefault(r.language, []).append(r)

    for lang in sorted(lang_groups.keys()):
        by_language.append(
            evaluate_sample_group(
                group_name=f"Language: {lang}",
                records=lang_groups[lang],
                scores_dict=scores_dict,
                direction=direction,
                threshold=operating_threshold,
                config=config,
                compute_bootstrap=False,
            )
        )

    # 9. Breakdown by Codec Condition
    by_codec: list[GroupEvaluationResult] = []
    codec_groups: dict[str, list[ManifestRecord]] = {}
    for r in records:
        codec_groups.setdefault(r.codec_channel_condition, []).append(r)

    for codec in sorted(codec_groups.keys()):
        by_codec.append(
            evaluate_sample_group(
                group_name=f"Codec: {codec}",
                records=codec_groups[codec],
                scores_dict=scores_dict,
                direction=direction,
                threshold=operating_threshold,
                config=config,
                compute_bootstrap=False,
            )
        )

    # 10. Breakdown by Replay Condition
    by_replay: list[GroupEvaluationResult] = []
    replay_groups: dict[str, list[ManifestRecord]] = {}
    for r in records:
        replay_groups.setdefault(r.replay_condition, []).append(r)

    for replay in sorted(replay_groups.keys()):
        by_replay.append(
            evaluate_sample_group(
                group_name=f"Replay: {replay}",
                records=replay_groups[replay],
                scores_dict=scores_dict,
                direction=direction,
                threshold=operating_threshold,
                config=config,
                compute_bootstrap=False,
            )
        )

    # 11. Breakdown by Duration Bucket
    by_duration: list[GroupEvaluationResult] = []
    dur_groups: dict[str, list[ManifestRecord]] = {}
    for r in records:
        bucket = get_duration_bucket(r.duration)
        dur_groups.setdefault(bucket, []).append(r)

    for bucket in sorted(dur_groups.keys()):
        by_duration.append(
            evaluate_sample_group(
                group_name=f"Duration: {bucket}",
                records=dur_groups[bucket],
                scores_dict=scores_dict,
                direction=direction,
                threshold=operating_threshold,
                config=config,
                compute_bootstrap=False,
            )
        )

    # 12. Cross-condition slices: matched, unseen-generator, unseen-channel, external-test
    cross_conditions: dict[str, GroupEvaluationResult] = {}

    matched_recs = [
        r
        for r in records
        if "holdout" not in (r.generator_family or "")
        and "holdout" not in r.codec_channel_condition
        and "external" not in r.source_dataset.lower()
    ]
    unseen_gen_recs = [
        r
        for r in records
        if "holdout" in (r.generator_family or "").lower()
        or "unseen" in (r.generator_family or "").lower()
    ]
    unseen_chan_recs = [
        r
        for r in records
        if "holdout" in r.codec_channel_condition.lower()
        or "unseen" in r.codec_channel_condition.lower()
    ]
    ext_test_recs = [
        r
        for r in records
        if "external" in r.source_dataset.lower() or "wild" in r.source_dataset.lower()
    ]

    if matched_recs:
        cross_conditions["matched"] = evaluate_sample_group(
            "Cross-Condition: Matched",
            matched_recs,
            scores_dict,
            direction,
            operating_threshold,
            config,
        )
    if unseen_gen_recs:
        cross_conditions["unseen_generator"] = evaluate_sample_group(
            "Cross-Condition: Unseen Generator",
            unseen_gen_recs,
            scores_dict,
            direction,
            operating_threshold,
            config,
        )
    if unseen_chan_recs:
        cross_conditions["unseen_channel"] = evaluate_sample_group(
            "Cross-Condition: Unseen Channel",
            unseen_chan_recs,
            scores_dict,
            direction,
            operating_threshold,
            config,
        )
    if ext_test_recs:
        cross_conditions["external_test"] = evaluate_sample_group(
            "Cross-Condition: External Test",
            ext_test_recs,
            scores_dict,
            direction,
            operating_threshold,
            config,
        )

    # 13. Latency diagnostics
    latency_res = compute_latency_metrics(records, timing_data) if timing_data else None

    # Merge provenance metadata
    effective_cfg = config if config is not None else VoiceShieldConfig()
    res_metadata: dict[str, Any] = {
        "score_direction": direction.value,
        "aggregation_rule": aggregation_rule.value,
        "total_records_in_manifest": len(records),
        "evaluated_records": len(matched_y),
        "config_hash": effective_cfg.compute_hash(),
    }
    if metadata:
        res_metadata.update(metadata)

    return FullEvaluationReport(
        overall=overall_res,
        score_coverage=score_coverage,
        by_generator=by_generator,
        by_source_dataset=by_source,
        by_language=by_language,
        by_codec=by_codec,
        by_replay=by_replay,
        by_duration_bucket=by_duration,
        cross_conditions=cross_conditions,
        deployment_threshold_source=threshold_source,
        operating_threshold=round(operating_threshold, 4),
        latency=latency_res,
        metadata=res_metadata,
        disclaimers=Disclaimers(),
    )
