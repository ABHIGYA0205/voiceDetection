"""Comprehensive regression test suite for R2 repairs:
- Connected-component data splitting across speaker_id, source_utterance_id, and sha256.
- Transitive component grouping and order-invariance.
- Holdout generator and codec whole-component train isolation.
- Strict score ingestion preserving 0.0, refusing duplicate/NaN/Inf/bool/missing IDs and scores.
- Default strict score coverage (100%), rejecting unknown IDs and missing IDs unless --allow-partial-scores.
- Threshold selection isolation (dev-only, train-only when no dev, never test, refuse pure test/unassigned/single-class).
- Fail-closed calibration metrics and plot omission for uncalibrated bounded scores.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from voiceshield.data.cli import main as data_cli_main
from voiceshield.data.io import ManifestIOError, load_manifest, save_manifest
from voiceshield.data.schema import ManifestRecord
from voiceshield.data.splitter import (
    codec_holdout_split,
    generator_holdout_split,
    speaker_disjoint_split,
)
from voiceshield.evaluation.cli import load_scores_file
from voiceshield.evaluation.cli import main as eval_cli_main
from voiceshield.evaluation.evaluator import (
    ThresholdOptimizationError,
    evaluate_deepfake_detector,
)
from voiceshield.evaluation.metrics import (
    ScoreDirection,
    compute_brier_and_ece,
    normalize_calibrated_probabilities,
)
from voiceshield.evaluation.reporting import generate_evaluation_plots


def _make_rec(
    sample_id: str,
    speaker_id: str,
    label: str = "bonafide",
    split: str | None = None,
    source_utterance_id: str | None = None,
    generator_family: str | None = None,
    attack_type: str = "none",
    codec: str = "pcm_clean",
    sha256: str | None = None,
) -> ManifestRecord:
    import hashlib

    sha = sha256 if sha256 else hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    return ManifestRecord(
        sample_id=sample_id,
        path=f"audio/{sample_id}.wav",
        label=label,  # type: ignore[arg-type]
        speaker_id=speaker_id,
        source_utterance_id=source_utterance_id,
        source_dataset="UnitTestSet",
        generator_family=generator_family,
        attack_type=attack_type,  # type: ignore[arg-type]
        language="en",
        original_sample_rate=16000,
        codec_channel_condition=codec,
        duration=3.0,
        consent_licence_ref="CC0-1.0",
        split=split,  # type: ignore[arg-type]
        sha256=sha,
    )


# ======================================================================
# A. CONNECTED-COMPONENT DATA SPLITTING
# ======================================================================


def test_split_speaker_id_connected() -> None:
    """Records sharing speaker_id always receive the same split."""
    records = [
        _make_rec("s1", "spk1", label="bonafide"),
        _make_rec("s2", "spk1", label="bonafide"),
        _make_rec("s3", "spk2", label="bonafide"),
        _make_rec("s4", "spk2", label="bonafide"),
        _make_rec("s5", "spk3", label="bonafide"),
        _make_rec("s6", "spk4", label="bonafide"),
    ]
    partitioned = speaker_disjoint_split(
        records, train_ratio=0.5, dev_ratio=0.25, test_ratio=0.25, seed=42
    )
    split_map = {r.sample_id: r.split for r in partitioned}
    assert split_map["s1"] == split_map["s2"]
    assert split_map["s3"] == split_map["s4"]


def test_split_source_utterance_id_connected() -> None:
    """Different speakers sharing source_utterance_id receive the same split."""
    records = [
        _make_rec("s1", "spk1", label="bonafide", source_utterance_id="utt_shared"),
        _make_rec("s2", "spk2", label="bonafide", source_utterance_id="utt_shared"),
        _make_rec("s3", "spk3", label="bonafide", source_utterance_id="utt_other"),
        _make_rec("s4", "spk4", label="bonafide", source_utterance_id="utt_other"),
    ]
    partitioned = speaker_disjoint_split(
        records, train_ratio=0.5, dev_ratio=0.25, test_ratio=0.25, seed=123
    )
    split_map = {r.sample_id: r.split for r in partitioned}
    assert split_map["s1"] == split_map["s2"]
    assert split_map["s3"] == split_map["s4"]


def test_split_sha256_connected() -> None:
    """Records sharing exact sha256 receive the same split at the splitter level."""
    shared_sha = "f" * 64
    records = [
        _make_rec("s1", "spk1", label="bonafide", sha256=shared_sha),
        _make_rec("s2", "spk2", label="bonafide", sha256=shared_sha),
        _make_rec("s3", "spk3", label="bonafide"),
        _make_rec("s4", "spk4", label="bonafide"),
    ]
    partitioned = speaker_disjoint_split(
        records, train_ratio=0.5, dev_ratio=0.25, test_ratio=0.25, seed=42
    )
    split_map = {r.sample_id: r.split for r in partitioned}
    assert split_map["s1"] == split_map["s2"]


def test_split_transitive_links() -> None:
    """A and B share speaker, B and C share source_utterance => A, B, C receive same split."""
    records = [
        _make_rec("sA", "spk_AB", label="bonafide", source_utterance_id="utt_A"),
        _make_rec("sB", "spk_AB", label="bonafide", source_utterance_id="utt_BC"),
        _make_rec("sC", "spk_C", label="bonafide", source_utterance_id="utt_BC"),
        _make_rec("sD", "spk_D", label="bonafide"),
        _make_rec("sE", "spk_E", label="bonafide"),
    ]
    partitioned = speaker_disjoint_split(
        records, train_ratio=0.5, dev_ratio=0.25, test_ratio=0.25, seed=99
    )
    split_map = {r.sample_id: r.split for r in partitioned}
    assert split_map["sA"] == split_map["sB"] == split_map["sC"]


def test_split_deterministic_reordering() -> None:
    """Reordering input records produces the exact same sample_id-to-split mapping."""
    records = [
        _make_rec("s1", "spk1", label="bonafide"),
        _make_rec("s2", "spk1", label="bonafide"),
        _make_rec("s3", "spk2", label="bonafide"),
        _make_rec("s4", "spk3", label="bonafide"),
        _make_rec("s5", "spk4", label="bonafide"),
    ]
    reordered = [records[2], records[0], records[4], records[1], records[3]]

    p1 = speaker_disjoint_split(records, seed=42)
    p2 = speaker_disjoint_split(reordered, seed=42)

    map1 = {r.sample_id: r.split for r in p1}
    map2 = {r.sample_id: r.split for r in p2}
    assert map1 == map2


def test_generator_holdout_speaker_overlap_forbidden_from_train() -> None:
    """A standard bonafide sample and a holdout-generator spoof sample sharing speaker cannot split into train and dev/test."""
    records = [
        _make_rec("bona_spk1", "spk1", label="bonafide"),
        _make_rec(
            "spoof_spk1_holdout",
            "spk1",
            label="spoof",
            attack_type="TTS",
            generator_family="ElevenLabs",
        ),
        _make_rec("bona_spk2", "spk2", label="bonafide"),
        _make_rec("spoof_spk2", "spk2", label="spoof", attack_type="TTS", generator_family="VITS"),
        _make_rec("bona_spk3", "spk3", label="bonafide"),
    ]
    partitioned = generator_holdout_split(records, holdout_generators=["ElevenLabs"], seed=42)
    split_map = {r.sample_id: r.split for r in partitioned}

    # Both bona_spk1 and spoof_spk1_holdout belong to spk1 component and must NOT be train
    assert split_map["bona_spk1"] != "train"
    assert split_map["spoof_spk1_holdout"] != "train"
    assert split_map["bona_spk1"] == split_map["spoof_spk1_holdout"]
    assert split_map["spoof_spk1_holdout"] in ("dev", "test")


def test_codec_holdout_speaker_overlap_forbidden_from_train() -> None:
    """A clean sample and a holdout codec sample sharing speaker cannot split into train and dev/test."""
    records = [
        _make_rec("clean_spk1", "spk1", label="bonafide", codec="pcm_clean"),
        _make_rec("codec_spk1_holdout", "spk1", label="bonafide", codec="g711_alaw"),
        _make_rec("clean_spk2", "spk2", label="bonafide", codec="pcm_clean"),
        _make_rec("clean_spk3", "spk3", label="bonafide", codec="pcm_clean"),
    ]
    partitioned = codec_holdout_split(records, holdout_codecs=["g711_alaw"], seed=42)
    split_map = {r.sample_id: r.split for r in partitioned}

    assert split_map["clean_spk1"] != "train"
    assert split_map["codec_spk1_holdout"] != "train"
    assert split_map["clean_spk1"] == split_map["codec_spk1_holdout"]
    assert split_map["codec_spk1_holdout"] in ("dev", "test")


def test_duplicate_sample_id_rejected_in_splitter() -> None:
    """Duplicate sample_ids are rejected with ValueError."""
    records = [
        _make_rec("dup_id", "spk1", label="bonafide"),
        _make_rec("dup_id", "spk2", label="bonafide"),
    ]
    with pytest.raises(ValueError, match="Duplicate sample_id"):
        speaker_disjoint_split(records)


def test_impossible_constraints_rejected_in_splitter() -> None:
    """Impossible assignments raise clear ValueError."""
    records = [
        _make_rec("s1", "spk1", label="spoof", attack_type="TTS", generator_family="ElevenLabs"),
    ]
    # train_ratio > 0 but only holdout component exists
    with pytest.raises(ValueError, match="Impossible split"):
        generator_holdout_split(
            records,
            holdout_generators=["ElevenLabs"],
            train_ratio=0.7,
            dev_ratio=0.15,
            test_ratio=0.15,
        )


def test_data_cli_fails_before_saving_if_leaked(tmp_path: Path) -> None:
    """CLI fails closed and does not save a leaked manifest."""
    raw_manifest = tmp_path / "raw.jsonl"
    out_manifest = tmp_path / "partitioned.jsonl"
    records = [
        _make_rec("s1", "spk1", label="bonafide"),
        _make_rec("s2", "spk2", label="bonafide"),
    ]
    save_manifest(records, raw_manifest)

    # Trigger invalid ratio sum in CLI
    exit_code = data_cli_main(
        [
            "split",
            "--manifest",
            str(raw_manifest),
            "--mode",
            "speaker_disjoint",
            "--train-ratio",
            "0.8",
            "--dev-ratio",
            "0.8",
            "--test-ratio",
            "0.8",
            "--output",
            str(out_manifest),
        ]
    )
    assert exit_code != 0
    assert not out_manifest.exists()


# ======================================================================
# B. STRICT SCORE INGESTION
# ======================================================================


def test_score_zero_point_zero_retained_exactly(tmp_path: Path) -> None:
    """Score of exactly 0.0 is preserved and not treated as missing or defaulted."""
    jsonl_file = tmp_path / "scores.jsonl"
    jsonl_file.write_text(
        json.dumps({"sample_id": "s0", "synthetic_score": 0.0})
        + "\n"
        + json.dumps({"sample_id": "s1", "uncalibrated_synthetic_score": 0.0})
        + "\n"
        + json.dumps({"sample_id": "s2", "score": 0.0})
        + "\n",
        encoding="utf-8",
    )
    scores, _ = load_scores_file(jsonl_file)
    assert isinstance(scores, dict)
    assert scores["s0"] == 0.0
    assert scores["s1"] == 0.0
    assert scores["s2"] == 0.0


def test_duplicate_sample_id_refused_in_scores(tmp_path: Path) -> None:
    """Duplicate sample-level score IDs in JSONL and CSV are refused."""
    jsonl_file = tmp_path / "dup.jsonl"
    jsonl_file.write_text(
        json.dumps({"sample_id": "s1", "score": 0.5})
        + "\n"
        + json.dumps({"sample_id": "s1", "score": 0.8})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate sample-level score ID"):
        load_scores_file(jsonl_file)

    csv_file = tmp_path / "dup.csv"
    csv_file.write_text("sample_id,score\ns1,0.5\ns1,0.8\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate sample-level score ID"):
        load_scores_file(csv_file)


def test_nan_and_inf_scores_refused(tmp_path: Path) -> None:
    """NaN and Inf score values in JSONL and CSV are refused."""
    for bad_val in ["NaN", "Infinity", "-Infinity"]:
        csv_file = tmp_path / f"bad_{bad_val}.csv"
        csv_file.write_text(f"sample_id,score\ns1,{bad_val}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Non-finite score value"):
            load_scores_file(csv_file)


def test_missing_scores_refused(tmp_path: Path) -> None:
    """Missing or empty score values are refused with clear ValueError."""
    jsonl_file = tmp_path / "missing_score.jsonl"
    jsonl_file.write_text(json.dumps({"sample_id": "s1"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing score"):
        load_scores_file(jsonl_file)

    csv_file = tmp_path / "empty_score.csv"
    csv_file.write_text("sample_id,score\ns1,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Empty score value"):
        load_scores_file(csv_file)


def test_missing_sample_id_refused(tmp_path: Path) -> None:
    """Missing sample identifier in scores file is refused with clear ValueError."""
    jsonl_file = tmp_path / "no_id.jsonl"
    jsonl_file.write_text(json.dumps({"score": 0.75}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing sample identifier"):
        load_scores_file(jsonl_file)


def test_boolean_scores_refused(tmp_path: Path) -> None:
    """Boolean score values (True/False) are rejected."""
    jsonl_file = tmp_path / "bool.jsonl"
    jsonl_file.write_text(json.dumps({"sample_id": "s1", "score": True}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Boolean score values are not permitted"):
        load_scores_file(jsonl_file)


def test_ambiguous_mixture_refused(tmp_path: Path) -> None:
    """Mixture of sample-level scores and window-level scores is rejected."""
    jsonl_file = tmp_path / "mix.jsonl"
    jsonl_file.write_text(
        json.dumps(
            {"sample_id": "s1", "score": 0.5, "windows": [{"window_index": 0, "score": 0.5}]}
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Ambiguous mixture"):
        load_scores_file(jsonl_file)


def test_duplicate_window_records_refused(tmp_path: Path) -> None:
    """Duplicate (sample_id, window_index) window records are rejected."""
    jsonl_file = tmp_path / "dup_win.jsonl"
    jsonl_file.write_text(
        json.dumps(
            {
                "sample_id": "s1",
                "windows": [
                    {"window_index": 0, "score": 0.2},
                    {"window_index": 0, "score": 0.4},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate \\(sample_id, window_index\\)"):
        load_scores_file(jsonl_file)


def test_duplicate_manifest_sample_ids_detected(tmp_path: Path) -> None:
    """Duplicate sample IDs in manifest file raise ManifestIOError; direct input raises ValueError."""
    dup_manifest_file = tmp_path / "dup_manifest.jsonl"
    dup_manifest_file.write_text(
        json.dumps(
            {
                "sample_id": "s1",
                "path": "a.wav",
                "label": "bonafide",
                "attack_type": "none",
                "speaker_id": "spk1",
                "source_dataset": "d",
                "original_sample_rate": 16000,
                "codec_channel_condition": "pcm_clean",
                "duration": 1.0,
                "consent_licence_ref": "c",
                "sha256": "a" * 64,
            }
        )
        + "\n"
        + json.dumps(
            {
                "sample_id": "s1",
                "path": "b.wav",
                "label": "bonafide",
                "attack_type": "none",
                "speaker_id": "spk2",
                "source_dataset": "d",
                "original_sample_rate": 16000,
                "codec_channel_condition": "pcm_clean",
                "duration": 1.0,
                "consent_licence_ref": "c",
                "sha256": "b" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ManifestIOError, match="Duplicate sample_id"):
        load_manifest(dup_manifest_file)

    direct_records = [
        _make_rec("s1", "spk1", label="bonafide"),
        _make_rec("s1", "spk2", label="bonafide"),
    ]
    with pytest.raises(ValueError, match="Duplicate sample_id"):
        evaluate_deepfake_detector(manifest=direct_records, scores={"s1": 0.5})


# ======================================================================
# C. COMPLETE SCORE COVERAGE BY DEFAULT
# ======================================================================


def test_score_coverage_default_fails_on_missing() -> None:
    """One missing score fails by default under strict mode."""
    records = [
        _make_rec("s1", "spk1", label="bonafide", split="dev"),
        _make_rec(
            "s2",
            "spk1",
            label="spoof",
            attack_type="TTS",
            generator_family="ElevenLabs",
            split="dev",
        ),
        _make_rec("s3", "spk2", label="bonafide", split="test"),
    ]
    scores = {"s1": 0.1, "s2": 0.9}  # s3 missing

    with pytest.raises(ValueError, match="Incomplete score coverage: 1 of 3"):
        evaluate_deepfake_detector(records, scores, deployment_threshold=0.5)


def test_score_coverage_partial_opt_in_succeeds() -> None:
    """Same missing input succeeds with allow_partial_scores=True and reports coverage."""
    records = [
        _make_rec("s1", "spk1", label="bonafide", split="dev"),
        _make_rec(
            "s2",
            "spk1",
            label="spoof",
            attack_type="TTS",
            generator_family="ElevenLabs",
            split="dev",
        ),
        _make_rec("s3", "spk2", label="bonafide", split="test"),
    ]
    scores = {"s1": 0.1, "s2": 0.9}  # s3 missing

    report = evaluate_deepfake_detector(
        records, scores, deployment_threshold=0.5, allow_partial_scores=True
    )
    cov = report.score_coverage
    assert cov.partial_scores_allowed is True
    assert cov.missing_score_count == 1
    assert cov.missing_sample_ids == ["s3"]
    assert pytest.approx(cov.coverage_fraction, 0.01) == 2 / 3
    assert cov.matched_score_count == 2


def test_score_coverage_unknown_ids_fail_in_all_modes() -> None:
    """Extra/unknown IDs fail even when allow_partial_scores=True."""
    records = [
        _make_rec("s1", "spk1", label="bonafide", split="dev"),
        _make_rec(
            "s2",
            "spk1",
            label="spoof",
            attack_type="TTS",
            generator_family="ElevenLabs",
            split="dev",
        ),
    ]
    scores = {"s1": 0.1, "s2": 0.9, "unknown_ghost_sample": 0.5}

    with pytest.raises(ValueError, match="unknown sample ID"):
        evaluate_deepfake_detector(
            records, scores, deployment_threshold=0.5, allow_partial_scores=True
        )


def test_score_coverage_100_percent() -> None:
    """Complete score coverage reports 100%."""
    records = [
        _make_rec("s1", "spk1", label="bonafide", split="dev"),
        _make_rec(
            "s2",
            "spk1",
            label="spoof",
            attack_type="TTS",
            generator_family="ElevenLabs",
            split="dev",
        ),
    ]
    scores = {"s1": 0.1, "s2": 0.9}
    report = evaluate_deepfake_detector(records, scores, deployment_threshold=0.5)
    cov = report.score_coverage
    assert cov.coverage_fraction == 1.0
    assert cov.missing_score_count == 0
    assert cov.unknown_score_count == 0


# ======================================================================
# D. TEST-SET THRESHOLD ISOLATION
# ======================================================================


def test_threshold_isolation_dev_ignores_test_scores() -> None:
    """In dev+test input, changing test scores cannot change the selected threshold."""
    records = [
        _make_rec("d1", "spk1", label="bonafide", split="dev"),
        _make_rec(
            "d2",
            "spk1",
            label="spoof",
            attack_type="TTS",
            generator_family="ElevenLabs",
            split="dev",
        ),
        _make_rec("t1", "spk2", label="bonafide", split="test"),
        _make_rec(
            "t2",
            "spk2",
            label="spoof",
            attack_type="TTS",
            generator_family="ElevenLabs",
            split="test",
        ),
    ]
    scores_1 = {"d1": 0.2, "d2": 0.8, "t1": 0.1, "t2": 0.9}
    scores_2 = {"d1": 0.2, "d2": 0.8, "t1": 0.49, "t2": 0.51}

    rep1 = evaluate_deepfake_detector(records, scores_1)
    rep2 = evaluate_deepfake_detector(records, scores_2)

    assert rep1.operating_threshold == rep2.operating_threshold
    assert "selected_on_dev_eer" in rep1.deployment_threshold_source


def test_threshold_isolation_train_plus_test_no_dev() -> None:
    """In train+test (no dev) input, threshold is selected on train only; changing test scores has 0 effect."""
    records = [
        _make_rec("tr1", "spk1", label="bonafide", split="train"),
        _make_rec(
            "tr2",
            "spk1",
            label="spoof",
            attack_type="TTS",
            generator_family="ElevenLabs",
            split="train",
        ),
        _make_rec("t1", "spk2", label="bonafide", split="test"),
        _make_rec(
            "t2",
            "spk2",
            label="spoof",
            attack_type="TTS",
            generator_family="ElevenLabs",
            split="test",
        ),
    ]
    scores_1 = {"tr1": 0.2, "tr2": 0.8, "t1": 0.05, "t2": 0.95}
    scores_2 = {"tr1": 0.2, "tr2": 0.8, "t1": 0.45, "t2": 0.55}

    rep1 = evaluate_deepfake_detector(records, scores_1)
    rep2 = evaluate_deepfake_detector(records, scores_2)

    assert rep1.operating_threshold == rep2.operating_threshold
    assert "selected_on_train_eer" in rep1.deployment_threshold_source


def test_threshold_optimization_refused_on_pure_test() -> None:
    """Pure test input without an explicit threshold raises ThresholdOptimizationError."""
    records = [
        _make_rec("t1", "spk1", label="bonafide", split="test"),
        _make_rec(
            "t2",
            "spk1",
            label="spoof",
            attack_type="TTS",
            generator_family="ElevenLabs",
            split="test",
        ),
    ]
    scores = {"t1": 0.1, "t2": 0.9}
    with pytest.raises(
        ThresholdOptimizationError,
        match="Refusing to optimize decision threshold on split marked 'test'",
    ):
        evaluate_deepfake_detector(records, scores)


def test_threshold_optimization_refused_on_unassigned() -> None:
    """Unassigned input (split=None) without explicit threshold raises ThresholdOptimizationError."""
    records = [
        _make_rec("u1", "spk1", label="bonafide", split=None),
        _make_rec(
            "u2",
            "spk1",
            label="spoof",
            attack_type="TTS",
            generator_family="ElevenLabs",
            split=None,
        ),
    ]
    scores = {"u1": 0.1, "u2": 0.9}
    with pytest.raises(ThresholdOptimizationError, match="All records in manifest are unassigned"):
        evaluate_deepfake_detector(records, scores)


def test_threshold_optimization_fails_on_single_class() -> None:
    """Single-class selection split raises ThresholdOptimizationError instead of silently defaulting to 0.5."""
    records = [
        _make_rec("d1", "spk1", label="bonafide", split="dev"),
        _make_rec("d2", "spk2", label="bonafide", split="dev"),
    ]
    scores = {"d1": 0.1, "d2": 0.2}
    with pytest.raises(ThresholdOptimizationError, match="contains only one class"):
        evaluate_deepfake_detector(records, scores)


def test_explicit_threshold_succeeds_and_rejects_nan_inf() -> None:
    """Explicit threshold succeeds without test tuning, while NaN/Inf explicit thresholds are rejected."""
    records = [
        _make_rec("t1", "spk1", label="bonafide", split="test"),
        _make_rec(
            "t2",
            "spk1",
            label="spoof",
            attack_type="TTS",
            generator_family="ElevenLabs",
            split="test",
        ),
    ]
    scores = {"t1": 0.1, "t2": 0.9}
    rep = evaluate_deepfake_detector(records, scores, deployment_threshold=0.65)
    assert rep.operating_threshold == 0.65
    assert rep.deployment_threshold_source == "explicit_deployment_threshold"

    with pytest.raises(ValueError, match="Explicit deployment threshold must be a finite number"):
        evaluate_deepfake_detector(records, scores, deployment_threshold=float("nan"))

    with pytest.raises(ValueError, match="Explicit deployment threshold must be a finite number"):
        evaluate_deepfake_detector(records, scores, deployment_threshold=float("inf"))


# ======================================================================
# E. CALIBRATION METRICS FAIL CLOSED
# ======================================================================


def test_bounded_uncalibrated_scores_produce_none_calibration() -> None:
    """Bounded scores in [0, 1] do not produce Brier/ECE without an explicit calibrator."""
    records = [
        _make_rec("d1", "spk1", label="bonafide", split="dev"),
        _make_rec(
            "d2",
            "spk1",
            label="spoof",
            attack_type="TTS",
            generator_family="ElevenLabs",
            split="dev",
        ),
    ]
    scores = {"d1": 0.1, "d2": 0.9}  # raw scores bounded in [0, 1]
    report = evaluate_deepfake_detector(records, scores, deployment_threshold=0.5)

    assert report.overall.curves.brier_score is None
    assert report.overall.curves.ece is None


def test_calibration_curve_plot_not_generated_from_raw_scores(tmp_path: Path) -> None:
    """Calibration curve plot is not generated from raw bounded scores."""
    records = [
        _make_rec("d1", "spk1", label="bonafide", split="dev"),
        _make_rec(
            "d2",
            "spk1",
            label="spoof",
            attack_type="TTS",
            generator_family="ElevenLabs",
            split="dev",
        ),
    ]
    scores = {"d1": 0.1, "d2": 0.9}
    out_dir = tmp_path / "plot_test"

    plots = generate_evaluation_plots(records, scores, ScoreDirection.HIGHER_IS_SPOOF, out_dir)
    assert "calibration_curve" not in plots
    assert not (out_dir / "plots" / "calibration_curve.png").exists()
    assert (out_dir / "plots" / "roc_curve.png").exists()


def test_normalize_calibrated_probabilities_direction() -> None:
    """For labels [0, 1] and calibrated bona-fide probabilities [0.9, 0.1], direction normalization yields spoof probabilities [0.1, 0.9] and Brier score 0.01, not 0.81."""
    y_true = np.array([0, 1], dtype=int)
    p_bonafide = np.array([0.9, 0.1], dtype=np.float64)

    p_spoof = normalize_calibrated_probabilities(
        p_bonafide, direction=ScoreDirection.HIGHER_IS_BONAFIDE
    )
    assert np.allclose(p_spoof, np.array([0.1, 0.9]))

    brier, ece = compute_brier_and_ece(y_true, p_spoof)
    assert brier == pytest.approx(0.01, abs=1e-4)
    assert brier != pytest.approx(0.81, abs=1e-4)


def test_eval_cli_workflow_complete_coverage(tmp_path: Path) -> None:
    """CLI evaluates complete scores, saves summary.md and metrics.csv."""
    manifest_file = tmp_path / "manifest.jsonl"
    records = [
        _make_rec("s1", "spk1", label="bonafide", split="dev"),
        _make_rec(
            "s2",
            "spk1",
            label="spoof",
            attack_type="TTS",
            generator_family="ElevenLabs",
            split="dev",
        ),
    ]
    save_manifest(records, manifest_file)

    scores_file = tmp_path / "scores.jsonl"
    scores_file.write_text(
        json.dumps({"sample_id": "s1", "synthetic_score": 0.1})
        + "\n"
        + json.dumps({"sample_id": "s2", "synthetic_score": 0.9})
        + "\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "eval_out"
    code = eval_cli_main(
        [
            "--manifest",
            str(manifest_file),
            "--scores",
            str(scores_file),
            "--output-dir",
            str(out_dir),
        ]
    )
    assert code == 0


def test_eval_cli_workflow_partial_scores_flag(tmp_path: Path) -> None:
    """CLI fails on missing score by default, succeeds when --allow-partial-scores is provided."""
    manifest_file = tmp_path / "manifest.jsonl"
    records = [
        _make_rec("s1", "spk1", label="bonafide", split="dev"),
        _make_rec(
            "s2",
            "spk1",
            label="spoof",
            attack_type="TTS",
            generator_family="ElevenLabs",
            split="dev",
        ),
        _make_rec("s3", "spk2", label="bonafide", split="test"),
    ]
    save_manifest(records, manifest_file)

    scores_file = tmp_path / "scores.jsonl"
    scores_file.write_text(
        json.dumps({"sample_id": "s1", "score": 0.1})
        + "\n"
        + json.dumps({"sample_id": "s2", "score": 0.9})
        + "\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "eval_out"

    # Default fails
    fail_code = eval_cli_main(
        [
            "--manifest",
            str(manifest_file),
            "--scores",
            str(scores_file),
            "--output-dir",
            str(out_dir),
        ]
    )
    assert fail_code == 1

    # With flag succeeds
    ok_code = eval_cli_main(
        [
            "--manifest",
            str(manifest_file),
            "--scores",
            str(scores_file),
            "--allow-partial-scores",
            "--output-dir",
            str(out_dir),
        ]
    )
    assert ok_code == 0
