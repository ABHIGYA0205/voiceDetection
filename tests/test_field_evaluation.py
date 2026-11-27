"""Regression coverage for field replay and short upload scoring."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

import pytest

from tests.test_file_reliability_integration import (
    CountingSpyDetector,
    make_clean_wav,
    make_clipped_wav,
    make_silence_wav,
)
from voiceshield.config.schema import VoiceShieldConfig, WindowingConfig
from voiceshield.data.io import save_manifest
from voiceshield.data.schema import ManifestRecord
from voiceshield.evaluation import field
from voiceshield.pipeline.file_evaluator import evaluate_audio


class WavFactory(Protocol):
    def __call__(self, path: Path, duration_s: float) -> Path: ...


@pytest.mark.parametrize(
    ("make_audio", "duration", "status", "calls"),
    [
        (make_clean_wav, 2.0, "success", 1),
        (make_clean_wav, 0.5, "insufficient_evidence", 0),
        (make_silence_wav, 2.0, "insufficient_evidence", 0),
        (make_clipped_wav, 2.0, "limited_reliability", 0),
    ],
)
def test_short_uploads_use_unpadded_audio_for_reliability(
    tmp_path: Path, make_audio: WavFactory, duration: float, status: str, calls: int
) -> None:
    wav = make_audio(tmp_path / "short.wav", duration_s=duration)
    detector = CountingSpyDetector()
    assert evaluate_audio(wav, detector).total_windows == 0

    config = VoiceShieldConfig(windowing=WindowingConfig(allow_short_files=True))
    result = evaluate_audio(wav, detector, config)

    assert result.status == status
    assert detector.predict_calls == calls
    assert len(result.windows) == 1
    assert result.windows[0].is_partial is True
    assert result.windows[0].end_seconds == pytest.approx(duration)
    if calls:
        assert detector.input_windows[0].size == round(16_000 * duration)


def make_manifest(directory: Path) -> Path:
    records = []
    for index, (split, label, duration) in enumerate(
        [
            ("dev", "bonafide", 6),
            ("dev", "spoof", 7),
            ("test", "bonafide", 8),
            ("test", "spoof", 2),
        ]
    ):
        path = make_clean_wav(directory / f"{index}.wav", duration_s=duration)
        spoof = label == "spoof"
        records.append(
            ManifestRecord.model_validate(
                {
                    "sample_id": f"trial-{index}",
                    "path": path.name,
                    "label": label,
                    "speaker_id": f"speaker-{index}",
                    "source_utterance_id": f"source-{index}",
                    "source_dataset": "test-fixtures",
                    "generator_family": f"synthetic-{index}" if spoof else None,
                    "attack_type": "TTS" if spoof else "none",
                    "original_sample_rate": 16000,
                    "duration": duration,
                    "consent_licence_ref": "generated-testing-fixtures",
                    "split": split,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        )
    manifest = directory / "manifest.jsonl"
    save_manifest(records, manifest)
    return manifest


class FixtureProvenance:
    model_id = "fixture"
    provenance_status = "mock"
    checkpoint_verified = False
    checkpoint_sha256 = None
    expected_checkpoint_sha256 = None

    def model_dump(self, *, mode: str = "python") -> dict[str, str]:
        return {"model_id": self.model_id}


class FieldDetector(CountingSpyDetector):
    provenance = FixtureProvenance()


def test_field_report_replays_both_paths_and_uses_dev_threshold(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    detector = FieldDetector(default_score=0.9)
    monkeypatch.setattr(field, "AASISTLDetector", lambda **kwargs: detector)
    manifest = make_manifest(tmp_path)

    assert field.main(["--manifest", str(manifest), "--target-fpr", "0"]) == 0
    report = json.loads(capsys.readouterr().out)
    policies = report["policies"]
    assert report["dev_bonafide_trials"] == 1
    assert report["held_out_generators"] == ["synthetic-3"]
    assert policies["stream_current"]["test"]["false_alerts_per_bonafide_hour"] == 450
    assert policies["stream_current"]["test"]["abstention_rate"] == 0.5
    assert policies["file_current"]["test"]["abstention_rate"] == 0.5
    assert policies["file_top_2"]["test"]["abstention_rate"] == 0.5
    assert policies["file_top_2_with_short_files"]["test"]["abstention_rate"] == 0
    assert policies["file_top_2_with_short_files"]["test"]["bonafide_clip_fpr"] == 0
    assert (
        policies["file_top_2_with_short_files"]["test_slices"]["generator"]["synthetic-3"]["trials"]
        == 1
    )
    assert report["candidate_thresholds"]["file_top_2_with_short_files"] > 0.9
    assert set(report["candidate_dev_bonafide_clip_fpr"].values()) == {0.0}

    (tmp_path / "2.wav").write_bytes(b"changed audio")
    assert field.main(["--manifest", str(manifest)]) == 1
    assert "SHA-256 mismatch" in capsys.readouterr().err


def test_candidate_threshold_keeps_ties_below_false_alert_budget() -> None:
    assert field.select_threshold([0.9, 0.9, 0.3, -1.0], 0.25) > 0.9
    assert field.select_threshold([0.9, -1.0], 0.5) == 0.0
    assert field.select_threshold([0.9], 1.0) == 0.0
    trace = ((0.9, 4.0), (0.91, 5.0), (0.5, 6.0), (0.9, 7.0))
    assert field.candidate_observation(trace, 0.9).alert_times == (4.0, 7.0)
    assert field.candidate_observation(((-1.0, 4.0),), 0.0) == field.Observation(False, ())
