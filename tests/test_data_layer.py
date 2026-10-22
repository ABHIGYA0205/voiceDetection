"""Unit tests for the manifest-driven data layer, partition splitters, and leakage detection."""

from pathlib import Path

import pytest

from voiceshield.data import (
    ManifestRecord,
    check_leakage,
    codec_holdout_split,
    format_summary_tables,
    generator_holdout_split,
    load_manifest,
    save_manifest,
    speaker_disjoint_split,
    summarize_manifest,
)
from voiceshield.data.cli import main as cli_main


def _make_sample(
    sample_id: str,
    speaker_id: str,
    label: str = "bonafide",
    split: str = "train",
    source_utterance_id: str | None = None,
    generator_family: str | None = None,
    attack_type: str = "none",
    codec: str = "pcm_clean",
    sha256: str | None = None,
) -> ManifestRecord:
    """Helper to create valid synthetic manifest records for testing."""
    import hashlib

    if sha256 is None:
        sha = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    else:
        sha = sha256

    return ManifestRecord(
        sample_id=sample_id,
        path=f"audio/{sample_id}.wav",
        label=label,  # type: ignore[arg-type]
        speaker_id=speaker_id,
        source_utterance_id=source_utterance_id,
        source_dataset="SyntheticTestSet",
        generator_family=generator_family,
        attack_type=attack_type,  # type: ignore[arg-type]
        language="en",
        original_sample_rate=16000,
        codec_channel_condition=codec,
        duration=3.5,
        consent_licence_ref="CC0-1.0",
        split=split,  # type: ignore[arg-type]
        sha256=sha,
    )


def test_manifest_schema_validation_bonafide_and_spoof() -> None:
    """Verify that bonafide and spoof records enforce label and attack_type consistency."""
    # Valid bonafide
    bonafide = _make_sample("s1", "spk1", label="bonafide", attack_type="none")
    assert bonafide.label == "bonafide"

    # Invalid: bonafide with TTS attack_type raises ValueError
    with pytest.raises(ValueError, match="Bona fide speech must have attack_type='none'"):
        _make_sample("s2", "spk1", label="bonafide", attack_type="TTS")

    # Invalid: spoof with attack_type='none' raises ValueError
    with pytest.raises(ValueError, match="Spoof speech must specify attack_type"):
        _make_sample("s3", "spk1", label="spoof", attack_type="none")

    # Invalid: spoof TTS with missing generator_family raises ValueError
    with pytest.raises(ValueError, match="missing generator_family"):
        _make_sample("s4", "spk1", label="spoof", attack_type="TTS", generator_family=None)


def test_manifest_csv_jsonl_roundtrip(tmp_path: Path) -> None:
    """Verify serialization and re-loading in both CSV and JSONL formats."""
    samples = [
        _make_sample("s1", "spk1", label="bonafide", source_utterance_id="utt_1"),
        _make_sample("s2", "spk2", label="spoof", attack_type="TTS", generator_family="ElevenLabs"),
    ]

    # JSONL roundtrip
    jsonl_path = tmp_path / "manifest.jsonl"
    save_manifest(samples, jsonl_path)
    loaded_jsonl = load_manifest(jsonl_path)
    assert len(loaded_jsonl) == 2
    assert loaded_jsonl[0].sample_id == "s1"
    assert loaded_jsonl[1].generator_family == "ElevenLabs"

    # CSV roundtrip
    csv_path = tmp_path / "manifest.csv"
    save_manifest(samples, csv_path)
    loaded_csv = load_manifest(csv_path)
    assert len(loaded_csv) == 2
    assert loaded_csv[0].source_utterance_id == "utt_1"
    assert loaded_csv[1].attack_type == "TTS"


def test_speaker_disjoint_splitting() -> None:
    """Verify that speaker_disjoint_split assigns each speaker strictly to one split."""
    records: list[ManifestRecord] = []
    # 10 speakers, 4 samples each
    for i in range(10):
        spk = f"spk_{i:02d}"
        for j in range(4):
            records.append(_make_sample(f"{spk}_{j}", spk, split="train"))

    partitioned = speaker_disjoint_split(
        records, train_ratio=0.7, dev_ratio=0.15, test_ratio=0.15, seed=123
    )

    report = check_leakage(partitioned)
    assert not report.has_leakage
    assert len(report.speaker_leakage) == 0

    # Verify split distribution
    splits = {r.split for r in partitioned}
    assert splits == {"train", "dev", "test"}


def test_intentional_speaker_leakage_detected() -> None:
    """Demonstrate that intentional speaker leakage across splits is caught."""
    records = [
        _make_sample("s1", speaker_id="spk_A", split="train"),
        _make_sample("s2", speaker_id="spk_A", split="test"),  # Leaked into test!
        _make_sample("s3", speaker_id="spk_B", split="train"),
    ]

    report = check_leakage(records)
    assert report.has_leakage
    assert "train_vs_test" in report.speaker_leakage
    assert "spk_A" in report.speaker_leakage["train_vs_test"]


def test_intentional_source_utterance_leakage_detected() -> None:
    """Demonstrate that intentional prompt/utterance leakage across splits is caught."""
    records = [
        _make_sample("s1", speaker_id="spk_A", split="train", source_utterance_id="utt_weather_01"),
        _make_sample(
            "s2",
            speaker_id="spk_B",  # Different speaker, but identical prompt!
            split="test",
            source_utterance_id="utt_weather_01",
        ),
    ]

    report = check_leakage(records)
    assert report.has_leakage
    assert "train_vs_test" in report.source_utterance_leakage
    assert "utt_weather_01" in report.source_utterance_leakage["train_vs_test"]


def test_generator_holdout_splitting_and_leakage_detection() -> None:
    """Verify generator holdout partitioning and ensure leakage is flagged if violated."""
    records = [
        _make_sample("s1", "spk1", label="spoof", attack_type="TTS", generator_family="ElevenLabs"),
        _make_sample("s2", "spk2", label="spoof", attack_type="TTS", generator_family="VITS"),
        _make_sample("s3", "spk3", label="bonafide"),
        _make_sample("s4", "spk4", label="bonafide"),
    ]

    # Hold out ElevenLabs
    partitioned = generator_holdout_split(records, holdout_generators=["ElevenLabs"], seed=42)
    report = check_leakage(partitioned, holdout_generators=["ElevenLabs"])
    assert not report.has_leakage

    # Verify ElevenLabs never appears in train
    for rec in partitioned:
        if rec.generator_family == "ElevenLabs":
            assert rec.split in ("dev", "test")

    # Now intentionally violate holdout by placing ElevenLabs in train
    violating = [
        _make_sample(
            "s1",
            "spk1",
            label="spoof",
            attack_type="TTS",
            generator_family="ElevenLabs",
            split="train",
        )
    ]
    violation_report = check_leakage(violating, holdout_generators=["ElevenLabs"])
    assert violation_report.has_leakage
    assert "ElevenLabs" in violation_report.generator_leakage["train_holdout_violation"]


def test_codec_holdout_splitting_and_leakage_detection() -> None:
    """Verify codec holdout partitioning and ensure leakage is flagged if violated."""
    records = [
        _make_sample("s1", "spk1", codec="g711_alaw"),
        _make_sample("s2", "spk2", codec="opus_16k"),
        _make_sample("s3", "spk3", codec="pcm_clean"),
    ]

    partitioned = codec_holdout_split(records, holdout_codecs=["g711_alaw"], seed=42)
    report = check_leakage(partitioned, holdout_codecs=["g711_alaw"])
    assert not report.has_leakage

    # Intentionally violate
    violating = [_make_sample("s1", "spk1", codec="g711_alaw", split="train")]
    report_violation = check_leakage(violating, holdout_codecs=["g711_alaw"])
    assert report_violation.has_leakage
    assert "g711_alaw" in report_violation.codec_leakage["train_holdout_violation"]


def test_duplicate_checksum_reporting() -> None:
    """Verify that multiple records sharing identical sha256 checksums are flagged."""
    dup_sha = "f" * 64
    records = [
        _make_sample("s1", "spk1", sha256=dup_sha),
        _make_sample("s2", "spk2", sha256=dup_sha),  # Duplicate checksum!
    ]

    report = check_leakage(records)
    assert report.has_leakage
    assert dup_sha in report.duplicate_checksums


def test_manifest_summary_tables() -> None:
    """Verify manifest summary generates all required dimension tables."""
    records = [
        _make_sample("s1", "spk1", label="bonafide", split="train"),
        _make_sample(
            "s2",
            "spk2",
            label="spoof",
            attack_type="TTS",
            generator_family="ElevenLabs",
            split="dev",
        ),
    ]

    summary = summarize_manifest(records)
    assert summary.total_samples == 2
    assert summary.num_unique_speakers == 2
    assert "bonafide" in summary.by_label.categories
    assert "spoof" in summary.by_label.categories

    formatted = format_summary_tables(summary)
    assert "Label Distribution" in formatted
    assert "Split Breakdown" in formatted
    assert "Generator Families" in formatted
    assert "Codecs & Channel Conditions" in formatted


def test_cli_validate_success_and_leakage_failure(tmp_path: Path) -> None:
    """Verify CLI exits with 0 on valid clean manifest and non-zero on prohibited leakage."""
    clean_manifest = tmp_path / "clean.jsonl"
    clean_records = [
        _make_sample("s1", "spk1", split="train", sha256="1" * 64),
        _make_sample("s2", "spk2", split="test", sha256="2" * 64),
    ]
    save_manifest(clean_records, clean_manifest)

    # CLI validate clean -> exit code 0
    exit_code = cli_main(["validate", "--manifest", str(clean_manifest)])
    assert exit_code == 0

    # CLI validate with speaker leakage -> exit code 1
    leaked_manifest = tmp_path / "leaked.jsonl"
    leaked_records = [
        _make_sample("s1", "spk_shared", split="train", sha256="1" * 64),
        _make_sample("s2", "spk_shared", split="test", sha256="2" * 64),
    ]
    save_manifest(leaked_records, leaked_manifest)
    exit_code_leaked = cli_main(["validate", "--manifest", str(leaked_manifest)])
    assert exit_code_leaked == 1
