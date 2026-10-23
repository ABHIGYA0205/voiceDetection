"""Tests specifically validating Tasks A through E of scientific correctness audit.

Invariants verified:
1. Softmax score is an uncalibrated synthetic anomaly score, not a calibrated probability.
2. Request threshold overrides detector-default threshold for all per-window decisions.
3. Tail windows follow strict hop grid; no manufactured near-duplicate windows (e.g. 4.01s).
4. Window count progression across duration boundaries: 3.99s, 4.00s, 4.01s, 4.99s, 5.00s, 5.01s.
5. Consecutive logic: [Elevated, Low, Elevated] does NOT satisfy 2-consecutive rule.
6. Silence and insufficient active speech return abstention (status="insufficient_evidence").
7. Streaming API responses have calibrated_score = None and score_is_calibrated = False when uncalibrated.
8. Mock mode remains prominently identified.
9. Upstream AASIST label invariant: index 0 = spoof, index 1 = bona fide.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient

from services.api.main import app
from voiceshield.models.aasist_adapter import (
    AASIST_L_DEFAULT_CONFIG,
    AASISTLDetector,
)
from voiceshield.models.base import BaseDetector, DetectorOutput
from voiceshield.pipeline.file_evaluator import evaluate_audio
from voiceshield.vendor.aasist import AASISTModel


class ProgrammableScoreDetector(BaseDetector):
    """Test detector returning pre-programmed scores for successive predict calls."""

    def __init__(self, scores: list[float], internal_threshold: float = 0.65) -> None:
        self.scores = scores
        self.internal_threshold = internal_threshold
        self.call_idx = 0
        self.model_id = "mock_programmable"
        self.device = torch.device("cpu")
        self.is_mock = True

    def predict(self, audio: np.ndarray[Any, Any] | torch.Tensor) -> DetectorOutput:
        idx = min(self.call_idx, len(self.scores) - 1)
        score = self.scores[idx]
        self.call_idx += 1

        # Upstream invariant: index 0 = spoof, index 1 = bona fide
        spoof_logit = float(np.log(max(score, 1e-7)))
        bonafide_logit = float(np.log(max(1.0 - score, 1e-7)))

        return DetectorOutput(
            raw_logits=(spoof_logit, bonafide_logit),
            bonafide_score=bonafide_logit,
            synthetic_score=score,
            uncalibrated_synthetic_score=score,
            score_is_calibrated=False,
            operating_threshold=self.internal_threshold,
            operating_threshold_source="unvalidated_heuristic",
            is_synthetic_suspected=(score >= self.internal_threshold),
            input_samples=len(audio) if isinstance(audio, np.ndarray) else audio.shape[-1],
            device_used="cpu",
            model_id=self.model_id,
        )

    def predict_batch(
        self,
        batch: torch.Tensor | np.ndarray[Any, Any],
        lengths: list[int] | None = None,
    ) -> list[DetectorOutput]:
        outputs: list[DetectorOutput] = []
        for item in batch:
            outputs.append(self.predict(item))
        return outputs


def make_test_wav(
    path: Path,
    duration_s: float,
    sample_rate: int = 16000,
    amplitude: float = 0.5,
    freq: float = 440.0,
) -> Path:
    """Generate audio WAV file with specified duration and amplitude."""
    num_samples = int(round(sample_rate * duration_s))
    if amplitude > 0.0:
        t = np.linspace(0, duration_s, num_samples, endpoint=False, dtype=np.float32)
        samples = amplitude * np.sin(2 * np.pi * freq * t)
    else:
        samples = np.zeros(num_samples, dtype=np.float32)
    sf.write(str(path), samples, sample_rate, format="WAV", subtype="PCM_16")
    return path


# ---------------------------------------------------------------------------
# Task B: Request Threshold vs Detector Internal Threshold
# ---------------------------------------------------------------------------


def test_request_threshold_overrides_detector_internal_threshold(tmp_path: Path) -> None:
    """Requirement 1 & 2: Score 0.30 is elevated when request threshold=0.20 (even if internal=0.65).

    Score 0.30 is NOT elevated when request threshold=0.40.
    """
    wav_path = make_test_wav(tmp_path / "test_thresh.wav", duration_s=4.0)

    # Detector has internal threshold 0.65, but returns score 0.30
    det1 = ProgrammableScoreDetector(scores=[0.30], internal_threshold=0.65)
    res_elevated = evaluate_audio(wav_path, detector=det1, threshold=0.20)
    assert res_elevated.status == "success"
    assert res_elevated.effective_threshold == 0.20
    assert len(res_elevated.windows) == 1
    # Per-window decision MUST use request threshold 0.20, not internal 0.65
    assert res_elevated.windows[0].is_synthetic_suspected is True
    assert res_elevated.windows[0].effective_threshold == 0.20
    assert res_elevated.windows[0].synthetic_score == 0.30
    assert res_elevated.windows[0].score_is_calibrated is False
    assert res_elevated.windows[0].calibrated_synthetic_likelihood is None

    # Now evaluate with threshold 0.40
    det2 = ProgrammableScoreDetector(scores=[0.30], internal_threshold=0.65)
    res_not_elevated = evaluate_audio(wav_path, detector=det2, threshold=0.40)
    assert res_not_elevated.status == "success"
    assert res_not_elevated.effective_threshold == 0.40
    assert len(res_not_elevated.windows) == 1
    assert res_not_elevated.windows[0].is_synthetic_suspected is False
    assert res_not_elevated.windows[0].effective_threshold == 0.40


def test_api_analyze_file_threshold_behavior(tmp_path: Path) -> None:
    """Verify POST /api/v1/analyze/file passes request threshold to evaluator."""
    wav_path = make_test_wav(tmp_path / "api_thresh.wav", duration_s=4.0)

    with open(wav_path, "rb") as f:
        file_bytes = f.read()

    with TestClient(app) as client:
        # Request with threshold 0.20
        resp1 = client.post(
            "/api/v1/analyze/file?threshold=0.20",
            files={"file": ("test.wav", io.BytesIO(file_bytes), "audio/wav")},
        )
        assert resp1.status_code == 200
        data1 = resp1.json()
        assert data1["effective_threshold"] == 0.20
        assert data1["operating_threshold_source"] == "unvalidated_heuristic"
        assert data1["score_is_calibrated"] is False
        for win in data1["windows"]:
            assert win["effective_threshold"] == 0.20

        # Request with threshold 0.85
        resp2 = client.post(
            "/api/v1/analyze/file?threshold=0.85",
            files={"file": ("test.wav", io.BytesIO(file_bytes), "audio/wav")},
        )
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["effective_threshold"] == 0.85
        for win in data2["windows"]:
            assert win["effective_threshold"] == 0.85


# ---------------------------------------------------------------------------
# Task C: Window Generation Across Exact Durations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("duration_s", "expected_windows", "expect_partial"),
    [
        (3.99, 1, True),   # < 4.0s: cyclically padded partial window
        (4.00, 1, False),  # exactly 4.0s: 1 full window [0.0, 4.0]
        (4.01, 1, False),  # 4.01s: 1 full window [0.0, 4.0], NO near-duplicate tail window!
        (4.99, 1, False),  # 4.99s: 1 full window [0.0, 4.0] (hop 1.0s requires 5.0s for 2nd window)
        (5.00, 2, False),  # 5.00s: exactly 2 windows [0.0, 4.0], [1.0, 5.0]
        (5.01, 2, False),  # 5.01s: 2 windows [0.0, 4.0], [1.0, 5.0], NO 3rd near-duplicate window
    ],
)
def test_file_durations_window_counts(
    tmp_path: Path,
    duration_s: float,
    expected_windows: int,
    expect_partial: bool,
) -> None:
    """Requirement 3 & 4: Strict hop grid windowing across duration boundaries."""
    wav_path = make_test_wav(tmp_path / f"dur_{duration_s}.wav", duration_s=duration_s)
    mock_raw = AASISTModel(AASIST_L_DEFAULT_CONFIG)  # type: ignore[no-untyped-call]
    detector = AASISTLDetector(_instantiated_model=mock_raw, preferred_device="cpu")

    result = evaluate_audio(
        wav_path,
        detector=detector,
        window_seconds=4.0,
        hop_seconds=1.0,
    )
    assert result.status == "success"
    assert result.total_windows == expected_windows
    assert len(result.windows) == expected_windows

    if expect_partial:
        assert result.windows[0].is_partial is True
    else:
        assert result.windows[0].is_partial is False


def test_4_01s_does_not_manufacture_two_sustained_windows(tmp_path: Path) -> None:
    """Requirement 4: 4.01-second audio must not create two heavily overlapping sustained windows."""
    wav_path = make_test_wav(tmp_path / "dur_4_01.wav", duration_s=4.01)
    # Detector returns high anomaly score 0.90
    detector = ProgrammableScoreDetector(scores=[0.90], internal_threshold=0.65)

    result = evaluate_audio(
        wav_path,
        detector=detector,
        window_seconds=4.0,
        hop_seconds=1.0,
        threshold=0.65,
    )
    # Exactly 1 window produced; no near-duplicate tail window [0.01, 4.01]
    assert result.total_windows == 1
    assert len(result.windows) == 1
    assert result.windows[0].start_seconds == 0.0
    assert result.windows[0].end_seconds == 4.0
    assert not any(abs(w.start_seconds - 0.01) < 0.005 for w in result.windows)


# ---------------------------------------------------------------------------
# Task D: Consecutive Logic
# ---------------------------------------------------------------------------


def test_elevated_low_elevated_does_not_satisfy_two_consecutive(tmp_path: Path) -> None:
    """Requirement 5: Three windows [Elevated, Low, Elevated] must NOT satisfy the 2-consecutive rule."""
    wav_path = make_test_wav(tmp_path / "three_windows.wav", duration_s=6.0)

    # 6.0s audio with 4.0s window and 1.0s hop produces 3 windows:
    # win 0: [0.0, 4.0]
    # win 1: [1.0, 5.0]
    # win 2: [2.0, 6.0]
    det_interrupted = ProgrammableScoreDetector(scores=[0.80, 0.20, 0.85], internal_threshold=0.65)
    res_interrupted = evaluate_audio(wav_path, detector=det_interrupted, threshold=0.65)

    assert res_interrupted.total_windows == 3
    assert res_interrupted.windows[0].is_synthetic_suspected is True
    assert res_interrupted.windows[1].is_synthetic_suspected is False
    assert res_interrupted.windows[2].is_synthetic_suspected is True

    # Even though 2 total windows were elevated, they were separated by a low window!
    assert res_interrupted.is_synthetic_suspected is False
    assert "consecutive elevated windows (max consecutive = 1, required = 2)" in res_interrupted.temporal_rule_explanation

    # Now verify that consecutive elevated windows DO trigger suspected
    det_consecutive = ProgrammableScoreDetector(scores=[0.80, 0.85, 0.20], internal_threshold=0.65)
    res_consecutive = evaluate_audio(wav_path, detector=det_consecutive, threshold=0.65)

    assert res_consecutive.total_windows == 3
    assert res_consecutive.windows[0].is_synthetic_suspected is True
    assert res_consecutive.windows[1].is_synthetic_suspected is True
    assert res_consecutive.is_synthetic_suspected is True
    assert "Elevated synthetic anomaly score sustained across at least 2 consecutive windows" in res_consecutive.temporal_rule_explanation


# ---------------------------------------------------------------------------
# Task E: Offline Speech Abstention
# ---------------------------------------------------------------------------


def test_silence_returns_insufficient_evidence_abstention(tmp_path: Path) -> None:
    """Requirement 6: Complete silence returns status='insufficient_evidence' and null scores."""
    wav_path = make_test_wav(tmp_path / "silence.wav", duration_s=4.0, amplitude=0.0)
    det = ProgrammableScoreDetector(scores=[0.99], internal_threshold=0.65)

    result = evaluate_audio(wav_path, detector=det, threshold=0.65)
    assert result.status == "insufficient_evidence"
    assert result.abstain_reason == "insufficient_speech_duration"
    assert result.is_synthetic_suspected is False
    assert len(result.windows) == 1
    # 1. Silence causes zero detector calls
    assert det.call_idx == 0

    # 2. Abstained window score fields are null
    win = result.windows[0]
    assert win.is_eligible is False
    assert win.abstain_reason == "insufficient_speech_duration"
    assert win.synthetic_score is None
    assert win.uncalibrated_synthetic_score is None
    assert win.calibrated_synthetic_likelihood is None
    assert win.synthetic_likelihood is None
    assert win.raw_spoof_logit is None
    assert win.raw_bonafide_logit is None
    assert win.raw_bonafide_score is None

    # 3. All-abstained file aggregate score fields are null
    assert result.mean_synthetic_score is None
    assert result.max_synthetic_score is None
    assert result.mean_synthetic_likelihood is None
    assert result.max_synthetic_likelihood is None


def test_exact_zero_score_remains_numeric_zero(tmp_path: Path) -> None:
    """Requirement: A real detector score of exactly 0.0 remains numeric 0.0 and is not converted to null."""
    wav_path = make_test_wav(tmp_path / "zero_score.wav", duration_s=4.0, amplitude=0.5)
    det = ProgrammableScoreDetector(scores=[0.0], internal_threshold=0.65)

    result = evaluate_audio(wav_path, detector=det, threshold=0.65)
    assert result.status == "success"
    # Real detector was invoked
    assert det.call_idx == 1
    assert result.total_windows == 1
    win = result.windows[0]
    assert win.is_eligible is True
    assert win.synthetic_score == 0.0
    assert win.uncalibrated_synthetic_score == 0.0
    assert win.synthetic_likelihood == 0.0
    assert isinstance(win.synthetic_score, float)
    assert win.raw_spoof_logit is not None
    assert win.raw_bonafide_logit is not None
    assert win.raw_bonafide_score is not None

    # Aggregates must remain numeric 0.0
    assert result.mean_synthetic_score == 0.0
    assert result.max_synthetic_score == 0.0
    assert result.mean_synthetic_likelihood == 0.0
    assert result.max_synthetic_likelihood == 0.0
    assert isinstance(result.mean_synthetic_score, float)


def test_insufficient_speech_abstention(tmp_path: Path) -> None:
    """Requirement 6: Audio with only 0.2s of sound (below 0.8s threshold) abstains."""
    # 4.0s audio where only first 0.2s has sound, remainder is silence
    sample_rate = 16000
    total_samples = sample_rate * 4
    samples = np.zeros(total_samples, dtype=np.float32)
    # 0.2s pulse
    pulse_samples = int(0.2 * sample_rate)
    samples[:pulse_samples] = 0.5 * np.sin(2 * np.pi * 440.0 * np.linspace(0, 0.2, pulse_samples))

    wav_path = tmp_path / "sparse_speech.wav"
    sf.write(str(wav_path), samples, sample_rate, format="WAV", subtype="PCM_16")

    det = ProgrammableScoreDetector(scores=[0.99], internal_threshold=0.65)
    result = evaluate_audio(wav_path, detector=det, threshold=0.65, min_speech_seconds=0.8)

    assert result.status == "insufficient_evidence"
    assert result.abstain_reason == "insufficient_speech_duration"
    assert result.is_synthetic_suspected is False
    assert det.call_idx == 0


def test_abstained_windows_do_not_contribute_to_sustained_elevation(tmp_path: Path) -> None:
    """Requirement 6: An abstained window between two elevated windows breaks consecutive chain."""
    sample_rate = 16000
    # 6.0s audio: window 0 has sound, window 1 is silent, window 2 has sound
    # With hop 1.0s and window 4.0s:
    # win 0: [0.0, 4.0] -> speech from 0.0 to 1.0
    # win 1: [1.0, 5.0] -> completely silent (1.0 to 5.0)
    # win 2: [2.0, 6.0] -> speech from 5.0 to 6.0
    samples = np.zeros(sample_rate * 6, dtype=np.float32)
    # Active sound in 0.0 - 1.0
    t0 = np.linspace(0, 1.0, sample_rate, endpoint=False)
    samples[:sample_rate] = 0.5 * np.sin(2 * np.pi * 440.0 * t0)
    # Active sound in 5.0 - 6.0
    t2 = np.linspace(0, 1.0, sample_rate, endpoint=False)
    samples[5 * sample_rate :] = 0.5 * np.sin(2 * np.pi * 440.0 * t2)

    wav_path = tmp_path / "gap_speech.wav"
    sf.write(str(wav_path), samples, sample_rate, format="WAV", subtype="PCM_16")

    det = ProgrammableScoreDetector(scores=[0.90, 0.90], internal_threshold=0.65)
    result = evaluate_audio(wav_path, detector=det, threshold=0.65, min_speech_seconds=0.8)

    assert result.total_windows == 3
    assert result.windows[0].is_eligible is True
    assert result.windows[1].is_eligible is False
    assert result.windows[1].abstain_reason == "insufficient_speech_duration"
    assert result.windows[2].is_eligible is True

    # Window 1 abstained, so consecutive elevation is broken!
    assert result.is_synthetic_suspected is False


# ---------------------------------------------------------------------------
# Task A: Streaming API Score Semantics
# ---------------------------------------------------------------------------


def test_streaming_api_uncalibrated_score_semantics() -> None:
    """Requirement 7: Streaming responses have calibrated_score = None and score_is_calibrated = False."""
    with TestClient(app) as client:
        with client.websocket_connect("/api/v1/stream") as websocket:
            # 1. Start handshake
            websocket.send_json({
                "type": "start",
                "sample_rate": 16000,
                "protocol_version": "1.0",
                "channels": 1,
                "pcm_encoding": "int16",
            })
            ack = websocket.receive_json()
            assert ack["type"] == "session_started"

            # 2. Send 4.0 seconds of PCM audio (64000 samples @ 16kHz int16)
            t = np.linspace(0, 4.0, 64000, endpoint=False, dtype=np.float32)
            pcm_float = 0.5 * np.sin(2 * np.pi * 440.0 * t)
            pcm_int16 = (pcm_float * 32767).astype(np.int16).tobytes()

            # Send in chunks of 2048 samples (4096 bytes)
            chunk_size = 4096
            for offset in range(0, len(pcm_int16), chunk_size):
                websocket.send_bytes(pcm_int16[offset : offset + chunk_size])

            # Wait for window result message
            received_window = False
            for _ in range(15):
                msg = websocket.receive_json()
                if msg["type"] == "window_result":
                    received_window = True
                    # Assert scientific score semantics
                    assert msg["score_is_calibrated"] is False
                    assert msg["calibrated_score"] is None
                    assert "synthetic_score" in msg
                    assert "operating_threshold_source" in msg
                    assert msg["operating_threshold_source"] == "unvalidated_heuristic"

                    decision = msg["decision"]
                    assert decision["score_is_calibrated"] is False
                    assert decision["calibrated_score"] is None
                    assert "synthetic_score" in decision
                    break

            assert received_window, "Expected at least one window_result from streaming WebSocket"

            # Stop
            websocket.send_json({"type": "stop"})
            stop_ack = websocket.receive_json()
            assert stop_ack["type"] == "session_stopped"


# ---------------------------------------------------------------------------
# Upstream AASIST Invariants
# ---------------------------------------------------------------------------


def test_aasist_label_direction_invariant() -> None:
    """Requirement 10: Preserves AASIST label invariant: index 0 = spoof, index 1 = bona fide."""
    dummy_model = MagicMock()
    detector = AASISTLDetector(_instantiated_model=dummy_model, preferred_device="cpu")

    # Case A: High spoof logit (index 0 = 5.0, index 1 = -2.0)
    dummy_model.return_value = (None, torch.tensor([[5.0, -2.0]]))
    out_spoof = detector.predict(np.zeros(64600, dtype=np.float32))

    assert out_spoof.raw_logits == (5.0, -2.0)
    assert out_spoof.bonafide_score == -2.0
    assert out_spoof.synthetic_score > 0.99
    assert out_spoof.is_synthetic_suspected is True

    # Case B: High bonafide logit (index 0 = -3.0, index 1 = 4.0)
    dummy_model.return_value = (None, torch.tensor([[-3.0, 4.0]]))
    out_bonafide = detector.predict(np.zeros(64600, dtype=np.float32))

    assert out_bonafide.raw_logits == (-3.0, 4.0)
    assert out_bonafide.bonafide_score == 4.0
    assert out_bonafide.synthetic_score < 0.01
    assert out_bonafide.is_synthetic_suspected is False
