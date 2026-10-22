"""Comprehensive unit test suite for the scientific evaluation toolkit.

Verifies:
- Confusion matrix and threshold metrics (precision, recall, specificity, F1, balanced accuracy) on known synthetic fixtures.
- ROC-AUC and PR-AUC calculations.
- EER computation with documented linear interpolation across crossovers.
- minDCF calculation with configurable costs and priors.
- Brier score and ECE for calibrated probabilities vs rejection for uncalibrated logits.
- Clustered bootstrap confidence intervals grouped by speaker.
- Window aggregation strategies (mean, max, top-k mean, temporal consecutive).
- Sliced and cross-condition reporting (generator, dataset, language, codec, replay, duration bucket).
- Latency diagnostics (p50/p95, real-time factor).
- Guardrail: Refusal to optimize a threshold on a test split.
- Guardrail: Undefined metrics for single-class groups clearly labeled and never fabricated as zeros.
- Guardrail: Testing score direction and detecting inverted conventions.
- Artifact generation: Markdown report, metrics CSV, and publication plots.
- Product compliance: No identity or fraud probability claims.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voiceshield.config.schema import VoiceShieldConfig
from voiceshield.data.schema import ManifestRecord
from voiceshield.evaluation.aggregation import (
    AggregationRule,
    WindowScoreRecord,
    aggregate_window_scores,
)
from voiceshield.evaluation.bootstrap import compute_clustered_bootstrap_cis
from voiceshield.evaluation.evaluator import (
    FullEvaluationReport,
    ThresholdOptimizationError,
    evaluate_deepfake_detector,
)
from voiceshield.evaluation.metrics import (
    ScoreDirection,
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
    save_evaluation_report,
)

# ---------------------------------------------------------------------------
# Synthetic fixtures with known expected values
# ---------------------------------------------------------------------------


def create_synthetic_manifest_records() -> list[ManifestRecord]:
    """Create a diverse synthetic manifest with known speakers, generators, and splits."""
    records: list[ManifestRecord] = [
        # Matched train/dev/test bonafide samples
        ManifestRecord(
            sample_id="bonafide_01",
            path="data/bonafide_01.wav",
            label="bonafide",
            speaker_id="spk_A",
            source_dataset="InHouseClean",
            attack_type="none",
            language="en",
            original_sample_rate=16000,
            codec_channel_condition="pcm_clean",
            replay_condition="none",
            duration=3.5,
            consent_licence_ref="internal_pilot",
            split="dev",
            sha256="a" * 64,
        ),
        ManifestRecord(
            sample_id="bonafide_02",
            path="data/bonafide_02.wav",
            label="bonafide",
            speaker_id="spk_A",
            source_dataset="InHouseClean",
            attack_type="none",
            language="en",
            original_sample_rate=16000,
            codec_channel_condition="pcm_clean",
            replay_condition="none",
            duration=5.0,
            consent_licence_ref="internal_pilot",
            split="test",
            sha256="b" * 64,
        ),
        ManifestRecord(
            sample_id="bonafide_03",
            path="data/bonafide_03.wav",
            label="bonafide",
            speaker_id="spk_B",
            source_dataset="InHouseClean",
            attack_type="none",
            language="es",
            original_sample_rate=16000,
            codec_channel_condition="g711_alaw",
            replay_condition="none",
            duration=1.8,
            consent_licence_ref="internal_pilot",
            split="test",
            sha256="c" * 64,
        ),
        ManifestRecord(
            sample_id="bonafide_04",
            path="data/bonafide_04.wav",
            label="bonafide",
            speaker_id="spk_C",
            source_dataset="InHouseClean",
            attack_type="none",
            language="en",
            original_sample_rate=16000,
            codec_channel_condition="pcm_clean",
            replay_condition="simulated_room",
            duration=9.0,
            consent_licence_ref="internal_pilot",
            split="test",
            sha256="d" * 64,
        ),
        # Matched spoof samples
        ManifestRecord(
            sample_id="spoof_01",
            path="data/spoof_01.wav",
            label="spoof",
            speaker_id="spk_A",
            source_dataset="InHouseClean",
            generator_family="ElevenLabs",
            generator_version="v2",
            attack_type="TTS",
            language="en",
            original_sample_rate=16000,
            codec_channel_condition="pcm_clean",
            replay_condition="none",
            duration=4.2,
            consent_licence_ref="internal_pilot",
            split="dev",
            sha256="e" * 64,
        ),
        ManifestRecord(
            sample_id="spoof_02",
            path="data/spoof_02.wav",
            label="spoof",
            speaker_id="spk_B",
            source_dataset="InHouseClean",
            generator_family="ElevenLabs",
            generator_version="v2",
            attack_type="TTS",
            language="en",
            original_sample_rate=16000,
            codec_channel_condition="pcm_clean",
            replay_condition="none",
            duration=3.8,
            consent_licence_ref="internal_pilot",
            split="test",
            sha256="f" * 64,
        ),
        # Unseen generator holdout sample
        ManifestRecord(
            sample_id="spoof_unseen_gen",
            path="data/spoof_unseen_gen.wav",
            label="spoof",
            speaker_id="spk_D",
            source_dataset="InHouseClean",
            generator_family="Holdout-DiffTTS",
            generator_version="v1",
            attack_type="TTS",
            language="en",
            original_sample_rate=16000,
            codec_channel_condition="pcm_clean",
            replay_condition="none",
            duration=6.5,
            consent_licence_ref="internal_pilot",
            split="test",
            sha256="1" * 64,
        ),
        # Unseen channel holdout sample
        ManifestRecord(
            sample_id="spoof_unseen_chan",
            path="data/spoof_unseen_chan.wav",
            label="spoof",
            speaker_id="spk_E",
            source_dataset="InHouseClean",
            generator_family="VITS",
            generator_version="v1",
            attack_type="TTS",
            language="es",
            original_sample_rate=16000,
            codec_channel_condition="holdout_opus_8k",
            replay_condition="none",
            duration=2.5,
            consent_licence_ref="internal_pilot",
            split="test",
            sha256="2" * 64,
        ),
        # External test sample
        ManifestRecord(
            sample_id="spoof_external",
            path="data/spoof_external.wav",
            label="spoof",
            speaker_id="spk_F",
            source_dataset="ExternalWildBenchmark",
            generator_family="StyleTTS",
            generator_version="v1",
            attack_type="TTS",
            language="en",
            original_sample_rate=16000,
            codec_channel_condition="pcm_clean",
            replay_condition="physical_recording",
            duration=8.2,
            consent_licence_ref="external_benchmark",
            split="test",
            sha256="3" * 64,
        ),
    ]
    return records


# ---------------------------------------------------------------------------
# Unit tests: Metrics with exact mathematical verification
# ---------------------------------------------------------------------------


def test_confusion_matrix_and_classification_metrics_exact() -> None:
    """Verify confusion matrix, precision, recall, specificity, and F1 on known values."""
    # 5 spoofs (1), 5 bonafide (0)
    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=int)
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.5, 0.3, 0.2, 0.1, 0.05], dtype=float)
    threshold = 0.55

    # Scores >= 0.55:
    # Spoof indices: 0, 1, 2, 3 -> TP = 4, FN = 1
    # Bonafide indices: none >= 0.55 (0.5 < 0.55) -> FP = 0, TN = 5
    cm = compute_confusion_matrix(y_true, scores, threshold=threshold)
    assert cm.tp == 4
    assert cm.fp == 0
    assert cm.tn == 5
    assert cm.fn == 1
    assert cm.total == 10

    m = compute_classification_metrics(y_true, scores, threshold=threshold)
    assert m.precision == 1.0  # 4 / (4 + 0)
    assert m.recall == 0.8  # 4 / (4 + 1)
    assert m.specificity == 1.0  # 5 / (5 + 0)
    assert m.fpr == 0.0  # 0 / (5 + 0)
    # F1 = 2 * (1.0 * 0.8) / (1.0 + 0.8) = 1.6 / 1.8 = 0.8889
    assert m.f1 == 0.8889
    # Balanced Accuracy = (0.8 + 1.0) / 2 = 0.9000
    assert m.balanced_accuracy == 0.9


def test_eer_linear_interpolation_exact() -> None:
    """Verify exact linear interpolation of Equal Error Rate across crossover point."""
    # 2 spoofs, 2 bonafide:
    # Bonafide: 0.2, 0.6
    # Spoof:    0.4, 0.8
    y_true = np.array([1, 1, 0, 0], dtype=int)
    scores = np.array([0.8, 0.4, 0.6, 0.2], dtype=float)

    # At threshold 0.5:
    # Spoofs >= 0.5: [0.8] -> TPR = 0.5 -> FNR = 0.5
    # Bonafide >= 0.5: [0.6] -> FPR = 0.5
    # Equal error point is exactly 0.50 (50%) at threshold 0.50
    eer_val, eer_th = compute_eer_interpolated(y_true, scores)
    assert eer_val is not None
    assert eer_th is not None
    assert pytest.approx(eer_val, rel=1e-3) == 0.50
    assert pytest.approx(eer_th, rel=1e-3) == 0.60


def test_min_dcf_exact() -> None:
    """Verify minDCF calculation with explicit costs and priors."""
    y_true = np.array([1, 1, 1, 0, 0, 0], dtype=int)
    scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1], dtype=float)

    # Perfectly separable: minDCF should be 0.0 at threshold between 0.3 and 0.7
    min_dcf, best_th = compute_min_dcf(y_true, scores, c_miss=1.0, c_fa=10.0, p_target=0.05)
    assert min_dcf is not None
    assert min_dcf == 0.0
    assert best_th is not None
    assert 0.3 <= best_th <= 0.8


def test_fpr_at_target_tprs() -> None:
    """Verify evaluation of operating FPR at target TPR levels."""
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=int)
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2, 0.1], dtype=float)

    # At TPR >= 0.75 (3 out of 4 spoofs detected):
    # Threshold = 0.7 selects spoofs [0.9, 0.8, 0.7].
    # Bonafide >= 0.7: none -> FPR = 0.0
    res = compute_fpr_at_target_tprs(y_true, scores, target_tprs=[0.75, 0.99])
    assert res["0.75"] == 0.0
    # At TPR >= 0.99 (all 4 spoofs detected):
    # Threshold = 0.6. Bonafide >= 0.6: none -> FPR = 0.0
    assert res["0.99"] == 0.0


def test_brier_and_ece_for_calibrated_probabilities() -> None:
    """Verify Brier score and ECE on bounded probabilities and refusal on uncalibrated logits."""
    y_true = np.array([1, 0], dtype=int)
    calibrated_probs = np.array([0.8, 0.2], dtype=float)

    # Brier: ((0.8 - 1)^2 + (0.2 - 0)^2) / 2 = (0.04 + 0.04) / 2 = 0.04
    brier, ece = compute_brier_and_ece(y_true, calibrated_probs, num_bins=10)
    assert brier is not None
    assert pytest.approx(brier, rel=1e-3) == 0.04
    assert ece is not None

    # Uncalibrated raw logits (outside [0, 1]) must return None
    uncalibrated_logits = np.array([-3.2, 4.5], dtype=float)
    brier_none, ece_none = compute_brier_and_ece(y_true, uncalibrated_logits)
    assert brier_none is None
    assert ece_none is None


# ---------------------------------------------------------------------------
# Unit tests: Guardrails
# ---------------------------------------------------------------------------


def test_guardrail_refuse_threshold_optimization_on_test_split() -> None:
    """Verify that attempting to optimize threshold on a test split raises ThresholdOptimizationError."""
    # All records marked split='test'
    records = [
        ManifestRecord(
            sample_id="test_01",
            path="data/test_01.wav",
            label="bonafide",
            speaker_id="spk_A",
            source_dataset="ASVspoof",
            attack_type="none",
            original_sample_rate=16000,
            duration=3.0,
            consent_licence_ref="ref",
            split="test",
            sha256="0" * 64,
        ),
        ManifestRecord(
            sample_id="test_02",
            path="data/test_02.wav",
            label="spoof",
            speaker_id="spk_B",
            source_dataset="ASVspoof",
            generator_family="TTS",
            attack_type="TTS",
            original_sample_rate=16000,
            duration=3.0,
            consent_licence_ref="ref",
            split="test",
            sha256="1" * 64,
        ),
    ]
    scores = {"test_01": 0.2, "test_02": 0.8}

    # Calling without deployment_threshold must be refused
    with pytest.raises(ThresholdOptimizationError, match="Refusing to optimize decision threshold"):
        evaluate_deepfake_detector(
            manifest=records,
            scores=scores,
            deployment_threshold=None,
        )

    # Calling with explicit deployment threshold must be accepted
    report = evaluate_deepfake_detector(
        manifest=records,
        scores=scores,
        deployment_threshold=0.65,
    )
    assert report.operating_threshold == 0.65
    assert "explicit_deployment_threshold" in report.deployment_threshold_source


def test_guardrail_single_class_group_metrics_undefined() -> None:
    """Verify that single-class slices clearly label undefined metrics rather than fabricating zeros."""
    # Only spoofs:
    y_true = np.array([1, 1, 1, 1], dtype=int)
    scores = np.array([0.9, 0.8, 0.7, 0.6], dtype=float)

    cm = compute_classification_metrics(y_true, scores, threshold=0.65)
    # Specificity and FPR are undefined because there are 0 bonafide (TN + FP = 0)
    assert cm.specificity is None
    assert cm.fpr is None
    assert cm.single_class_warning is not None
    assert "Single-class group" in cm.single_class_warning

    # EER and minDCF are undefined
    eer_val, eer_th = compute_eer_interpolated(y_true, scores)
    assert eer_val is None
    assert eer_th is None

    min_dcf, _ = compute_min_dcf(y_true, scores)
    assert min_dcf is None


def test_guardrail_score_direction_validation() -> None:
    """Verify that score direction is tested and inverted score conventions are rejected."""
    y_true = np.array([1, 1, 0, 0], dtype=int)
    # Bonafide scores = 10.0, Spoof scores = 0.1 -> strongly inverted for higher_is_spoof
    inverted_scores = np.array([0.1, 0.1, 10.0, 10.0], dtype=float)

    with pytest.raises(ValueError, match="Check if score convention is inverted"):
        validate_score_direction(y_true, inverted_scores, direction=ScoreDirection.HIGHER_IS_SPOOF)

    # Score inversion normalization works
    norm_scores = normalize_scores_to_higher_is_spoof(
        inverted_scores, ScoreDirection.HIGHER_IS_BONAFIDE
    )
    assert norm_scores[0] > norm_scores[2]  # Now spoof > bonafide


# ---------------------------------------------------------------------------
# Unit tests: Window aggregation & Clustered Bootstrap
# ---------------------------------------------------------------------------


def test_window_score_aggregation_strategies() -> None:
    """Verify MEAN, MAX, TOP_K_MEAN, and TEMPORAL_CONSECUTIVE aggregation."""
    wins = [
        WindowScoreRecord(sample_id="utt_1", window_index=0, score=0.4),
        WindowScoreRecord(sample_id="utt_1", window_index=1, score=0.8),
        WindowScoreRecord(sample_id="utt_1", window_index=2, score=0.9),
    ]

    # Mean: (0.4 + 0.8 + 0.9) / 3 = 0.70
    assert (
        pytest.approx(aggregate_window_scores(wins, AggregationRule.MEAN)["utt_1"], rel=1e-3)
        == 0.70
    )

    # Max: 0.90
    assert aggregate_window_scores(wins, AggregationRule.MAX)["utt_1"] == 0.90

    # Top-2 Mean: (0.8 + 0.9) / 2 = 0.85
    assert (
        pytest.approx(
            aggregate_window_scores(wins, AggregationRule.TOP_K_MEAN, top_k=2)["utt_1"], rel=1e-3
        )
        == 0.85
    )

    # Temporal consecutive with threshold 0.75:
    # Windows 1 and 2 are 0.8 and 0.9 (>= 0.75) -> consecutive sustained -> max score (0.90)
    assert (
        aggregate_window_scores(
            wins, AggregationRule.TEMPORAL_CONSECUTIVE, temporal_threshold=0.75
        )["utt_1"]
        == 0.90
    )


def test_clustered_bootstrap_confidence_intervals() -> None:
    """Verify grouped bootstrap CI calculation produces valid bounds."""
    # 20 samples from 4 distinct speakers
    speakers = ["spk_1"] * 5 + ["spk_2"] * 5 + ["spk_3"] * 5 + ["spk_4"] * 5
    y_true = np.array([1, 1, 1, 0, 0] * 4, dtype=int)
    scores = np.array([0.9, 0.85, 0.8, 0.2, 0.1] * 4, dtype=float)

    cis = compute_clustered_bootstrap_cis(
        y_true=y_true,
        scores_higher_spoof=scores,
        cluster_ids=np.array(speakers),
        threshold=0.5,
        n_iterations=50,
        ci_level=0.95,
        seed=42,
    )

    assert "roc_auc" in cis
    assert "eer" in cis
    assert "f1" in cis
    assert "balanced_accuracy" in cis

    # AUC should be very high (near 1.0)
    auc_ci = cis["roc_auc"]
    assert auc_ci is not None
    assert 0.90 <= auc_ci[0] <= auc_ci[1] <= 1.0


# ---------------------------------------------------------------------------
# Integration tests: Full evaluation workflow, reports, and plots
# ---------------------------------------------------------------------------


def test_full_evaluation_report_and_export(tmp_path: Path) -> None:
    """Test full evaluation pipeline with slicing, cross-conditions, latency, and reporting."""
    records = create_synthetic_manifest_records()
    scores = {
        "bonafide_01": 0.15,
        "bonafide_02": 0.25,
        "bonafide_03": 0.35,
        "bonafide_04": 0.10,
        "spoof_01": 0.85,
        "spoof_02": 0.95,
        "spoof_unseen_gen": 0.75,
        "spoof_unseen_chan": 0.65,
        "spoof_external": 0.80,
    }
    timings = dict.fromkeys(scores, 0.045)  # 45 ms per sample

    cfg = VoiceShieldConfig()

    report: FullEvaluationReport = evaluate_deepfake_detector(
        manifest=records,
        scores=scores,
        direction=ScoreDirection.HIGHER_IS_SPOOF,
        deployment_threshold=0.60,
        config=cfg,
        timing_data=timings,
        metadata={"git_commit": "abcdef123", "config_hash": cfg.compute_hash()},
    )

    # Verify overall result
    assert report.overall.sample_count == 9
    assert report.overall.spoof_count == 5
    assert report.overall.bonafide_count == 4
    assert report.overall.curves.roc_auc is not None
    assert report.overall.curves.roc_auc == 1.0  # Perfect separation
    assert report.overall.curves.eer == 0.0

    # Verify slices
    assert len(report.by_generator) >= 2
    assert len(report.by_codec) >= 2
    assert len(report.by_duration_bucket) >= 2
    assert "unseen_generator" in report.cross_conditions
    assert "unseen_channel" in report.cross_conditions
    assert "external_test" in report.cross_conditions

    # Verify latency
    assert report.latency is not None
    assert report.latency.p50_latency_ms == 45.0
    assert report.latency.real_time_factor < 1.0  # Real-time

    # Verify disclaimers compliance
    assert "identity of caller" in report.disclaimers.identity_proof.lower()
    assert "fraud probability" in report.disclaimers.fraud_probability.lower()

    # Save reports and plots
    out_dir = tmp_path / "eval_report"
    save_evaluation_report(
        report=report,
        output_dir=out_dir,
        records=records,
        scores_dict=scores,
        direction=ScoreDirection.HIGHER_IS_SPOOF,
    )

    assert (out_dir / "summary.md").exists()
    assert (out_dir / "metrics.csv").exists()
    assert (out_dir / "plots" / "roc_curve.png").exists()
    assert (out_dir / "plots" / "score_distributions.png").exists()
    assert (out_dir / "plots" / "calibration_curve.png").exists()

    md_content = (out_dir / "summary.md").read_text(encoding="utf-8")
    assert "VoiceShield Scientific Detection Evaluation Report" in md_content
    assert "ROC-AUC" in md_content
    assert "minDCF" in md_content
    assert "Breakdown by Synthesis / Voice Conversion Generator" in md_content
    assert "Cross-Condition & Generalization Performance" in md_content
    assert "Real-Time Factor" in md_content
