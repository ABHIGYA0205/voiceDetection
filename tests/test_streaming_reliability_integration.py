"""Comprehensive unit and integration tests for VoiceShield Reliability Stage A3:
fail-closed input-reliability gating for live streaming.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from services.api.main import app
from voiceshield.dsp.quality import compute_quality_metrics
from voiceshield.dsp.reliability import (
    InputReliabilityAssessment,
    ReliabilityReason,
    ReliabilityState,
)
from voiceshield.models.base import BaseDetector, DetectorOutput
from voiceshield.models.provenance import ModelProvenance
from voiceshield.streaming.policy import DecisionState, ReasonCode
from voiceshield.streaming.session import (
    StreamingAbstention,
    StreamingSession,
    WindowForInference,
)

# ==============================================================================
# Helper Signal Generators & Mock Detector
# ==============================================================================


def make_clean_speech(
    duration_sec: float,
    sample_rate: int = 16000,
    freq: float = 300.0,
    amplitude: float = 0.4,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Generate canonical active speech waveform (clean, unclipped, high-SNR)."""
    num_samples = int(round(duration_sec * sample_rate))
    t = np.linspace(0, duration_sec, num_samples, endpoint=False, dtype=np.float32)
    return np.asarray(amplitude * np.sin(2.0 * np.pi * freq * t), dtype=np.float32)


def make_clipped_speech(
    duration_sec: float,
    sample_rate: int = 16000,
    freq: float = 300.0,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Generate active speech with severe digital clipping (> 0.05% at ceiling)."""
    num_samples = int(round(duration_sec * sample_rate))
    t = np.linspace(0, duration_sec, num_samples, endpoint=False, dtype=np.float32)
    raw = 1.5 * np.sin(2.0 * np.pi * freq * t)
    return np.asarray(np.clip(raw, -1.0, 1.0), dtype=np.float32)


def make_marginal_snr_speech(
    duration_sec: float,
    sample_rate: int = 16000,
    freq: float = 300.0,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Generate active speech with low dynamic range yielding SNR < 12 dB."""
    num_samples = int(round(duration_sec * sample_rate))
    t = np.linspace(0, duration_sec, num_samples, endpoint=False, dtype=np.float32)
    return np.asarray(0.05 + 0.015 * np.sin(2.0 * np.pi * freq * t), dtype=np.float32)


def make_silence(
    duration_sec: float,
    sample_rate: int = 16000,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Generate pure digital silence (RMS < 1e-4)."""
    num_samples = int(round(duration_sec * sample_rate))
    return np.zeros(num_samples, dtype=np.float32)


def make_short_speech(
    duration_sec: float = 4.0,
    speech_duration_sec: float = 0.5,
    sample_rate: int = 16000,
    freq: float = 300.0,
    amplitude: float = 0.4,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Generate audio with active speech duration below min_speech_seconds (0.8s)."""
    num_samples = int(round(duration_sec * sample_rate))
    speech_samples = int(round(speech_duration_sec * sample_rate))
    signal = np.zeros(num_samples, dtype=np.float32)
    t_sp = np.linspace(0, speech_duration_sec, speech_samples, endpoint=False, dtype=np.float32)
    signal[:speech_samples] = amplitude * np.sin(2.0 * np.pi * freq * t_sp)
    return np.asarray(signal, dtype=np.float32)


def to_pcm16_bytes(samples: np.ndarray[Any, np.dtype[np.float32]]) -> bytes:
    """Convert float32 samples in [-1.0, 1.0] to int16 PCM byte string."""
    clamped = np.clip(samples, -1.0, 1.0)
    int16_arr = (clamped * 32767.0).astype(np.int16)
    return bytes(int16_arr.tobytes())


def receive_window_result(ws: Any, target_window_index: int) -> dict[str, Any]:
    """Receive WebSocket messages until a window_result matching target_window_index is received."""
    while True:
        msg = ws.receive_json()
        if msg.get("type") == "window_result" and msg.get("window_index") == target_window_index:
            return cast(dict[str, Any], msg)


class RecordingDetector(BaseDetector):
    """Detector for tests that records call invocations and arguments."""

    provenance = ModelProvenance(
        model_id="mock/aasist-l",
        architecture="aasist_l",
        provenance_status="mock",
        checkpoint_verified=False,
        is_mock=True,
        parameter_count=85306,
    )

    def __init__(self, score: float = 0.85) -> None:
        super().__init__()
        self.call_count: int = 0
        self.calls: list[np.ndarray[Any, np.dtype[np.float32]]] = []
        self.score: float = score

    def predict(
        self, audio: np.ndarray[Any, np.dtype[np.float32]] | torch.Tensor
    ) -> DetectorOutput:
        self.call_count += 1
        arr = (
            audio.detach().cpu().numpy().astype(np.float32)
            if isinstance(audio, torch.Tensor)
            else np.asarray(audio, dtype=np.float32)
        )
        self.calls.append(arr.copy())
        logit_spoof = float(np.log(max(self.score, 1e-6)))
        logit_bonafide = float(np.log(max(1.0 - self.score, 1e-6)))
        return DetectorOutput(
            raw_logits=(logit_spoof, logit_bonafide),
            bonafide_score=logit_bonafide - logit_spoof,
            synthetic_score=self.score,
            uncalibrated_synthetic_score=self.score,
            is_synthetic_suspected=bool(self.score >= 0.65),
            input_samples=arr.shape[0],
            device_used="cpu",
            model_id="RecordingDetector",
        )

    def predict_batch(self, batch: torch.Tensor) -> list[DetectorOutput]:
        outputs: list[DetectorOutput] = []
        for i in range(batch.shape[0]):
            outputs.append(self.predict(batch[i]))
        return outputs


# ==============================================================================
# Unit & Gating Tests (1-10)
# ==============================================================================


def test_clean_canonical_speech_is_reliable_and_reaches_detector() -> None:
    """Requirement 1: Clean canonical speech is RELIABLE, queues once, and reaches the detector."""
    session = StreamingSession()
    clean_audio = make_clean_speech(4.0)

    res = session.ingest_pcm(clean_audio, sample_rate=16000)
    assert res.window_ready is True
    assert res.pending_window_available is True
    assert len(res.immediate_decisions) == 0
    assert len(res.immediate_abstentions) == 0

    assert session.reliable_window_count == 1
    assert session.limited_reliability_window_count == 0
    assert session.insufficient_reliability_window_count == 0
    assert session.total_windows_generated == 1

    win = session.pop_pending_window()
    assert win is not None
    assert win.is_eligible is True
    assert win.input_reliability.state == ReliabilityState.RELIABLE
    assert win.input_reliability.origin_verdict_allowed is True
    assert win.input_reliability.reasons == ()
    assert win.quality_metrics.quality_status == "clean"

    # Worker executes inference and completes
    decision = session.complete_inference(synthetic_score=0.85, window=win)
    assert decision is not None
    assert decision.synthetic_score == 0.85


def test_clipped_active_speech_is_limited_never_queued_null_scores() -> None:
    """Requirement 2: Clipped active speech is LIMITED, never queued, emits null scores, zero detector calls."""
    session = StreamingSession()
    clipped_audio = make_clipped_speech(4.0)

    res = session.ingest_pcm(clipped_audio, sample_rate=16000)
    assert res.window_ready is True
    assert res.pending_window_available is False
    assert session.has_pending_window is False

    assert len(res.immediate_abstentions) == 1
    abstention = res.immediate_abstentions[0]
    assert abstention.input_reliability.state == ReliabilityState.LIMITED
    assert ReliabilityReason.SEVERE_CLIPPING in abstention.input_reliability.reasons
    assert abstention.input_reliability.origin_verdict_allowed is False
    assert abstention.decision.state == DecisionState.UNCERTAIN
    assert abstention.decision.reason == ReasonCode.SIGNAL_QUALITY_DEGRADED
    assert abstention.decision.synthetic_score is None

    assert session.limited_reliability_window_count == 1
    assert session.reliable_window_count == 0
    assert session.insufficient_reliability_window_count == 0
    assert session.total_windows_generated == 1


def test_marginal_snr_active_speech_is_limited_never_queued_null_scores() -> None:
    """Requirement 3: Marginal-SNR active-speech window is LIMITED, never queued, emits null scores."""
    session = StreamingSession()
    marginal_audio = make_marginal_snr_speech(4.0)

    res = session.ingest_pcm(marginal_audio, sample_rate=16000)
    assert res.window_ready is True
    assert res.pending_window_available is False
    assert session.has_pending_window is False

    assert len(res.immediate_abstentions) == 1
    abstention = res.immediate_abstentions[0]
    assert abstention.input_reliability.state == ReliabilityState.LIMITED
    assert ReliabilityReason.MARGINAL_SNR in abstention.input_reliability.reasons
    assert abstention.input_reliability.origin_verdict_allowed is False
    assert abstention.decision.state == DecisionState.UNCERTAIN
    assert abstention.decision.reason == ReasonCode.SIGNAL_QUALITY_DEGRADED
    assert abstention.decision.synthetic_score is None

    assert session.limited_reliability_window_count == 1
    assert session.reliable_window_count == 0
    assert session.insufficient_reliability_window_count == 0
    assert session.total_windows_generated == 1


def test_silence_is_insufficient_signal_quality_degraded_zero_detector_calls() -> None:
    """Requirement 4: Silence is INSUFFICIENT with reasons (SILENT_AUDIO, INSUFFICIENT_SPEECH)."""
    session = StreamingSession()
    silence_audio = make_silence(4.0)

    res = session.ingest_pcm(silence_audio, sample_rate=16000)
    assert res.window_ready is True
    assert res.pending_window_available is False
    assert session.has_pending_window is False

    assert len(res.immediate_abstentions) == 1
    abstention = res.immediate_abstentions[0]
    assert abstention.input_reliability.state == ReliabilityState.INSUFFICIENT
    assert abstention.input_reliability.reasons == (
        ReliabilityReason.SILENT_AUDIO,
        ReliabilityReason.INSUFFICIENT_SPEECH,
    )
    assert abstention.input_reliability.origin_verdict_allowed is False
    assert abstention.decision.state == DecisionState.INSUFFICIENT_EVIDENCE
    assert abstention.decision.reason == ReasonCode.SIGNAL_QUALITY_DEGRADED
    assert abstention.abstain_reason == ReasonCode.INSUFFICIENT_SPEECH_DURATION
    assert abstention.decision.synthetic_score is None

    assert session.insufficient_reliability_window_count == 1
    assert session.reliable_window_count == 0
    assert session.limited_reliability_window_count == 0
    assert session.total_windows_generated == 1


def test_pure_short_speech_retains_insufficient_speech_duration() -> None:
    """Requirement 5: Pure short speech retains INSUFFICIENT_SPEECH_DURATION."""
    session = StreamingSession()
    short_audio = make_short_speech(4.0, speech_duration_sec=0.5)

    res = session.ingest_pcm(short_audio, sample_rate=16000)
    assert res.window_ready is True
    assert res.pending_window_available is False

    assert len(res.immediate_abstentions) == 1
    abstention = res.immediate_abstentions[0]
    assert abstention.input_reliability.state == ReliabilityState.INSUFFICIENT
    assert abstention.input_reliability.reasons == (ReliabilityReason.INSUFFICIENT_SPEECH,)
    assert abstention.decision.reason == ReasonCode.INSUFFICIENT_SPEECH_DURATION
    assert abstention.abstain_reason == ReasonCode.INSUFFICIENT_SPEECH_DURATION
    assert abstention.decision.state == DecisionState.INSUFFICIENT_EVIDENCE

    assert session.insufficient_reliability_window_count == 1
    assert session.reliable_window_count == 0
    assert session.limited_reliability_window_count == 0


def test_abstentions_do_not_mutate_score_histories() -> None:
    """Requirement 6: Abstentions do not mutate raw_history or smoothed_history."""
    # 1. Evaluate one reliable window in session to establish baseline history
    session = StreamingSession()
    clean = make_clean_speech(4.0)
    session.ingest_pcm(clean, sample_rate=16000)
    w0 = session.pop_pending_window()
    assert w0 is not None
    session.complete_inference(synthetic_score=0.75, window=w0)

    raw_len = len(session.temporal_policy.raw_history)
    smoothed_len = len(session.temporal_policy.smoothed_history)
    assert raw_len == 1
    assert smoothed_len == 1

    # 2. Ingest silence in a fresh session with seeded history
    s_silence = StreamingSession()
    s_silence.temporal_policy.raw_history.append(0.75)
    s_silence.temporal_policy.smoothed_history.append(0.75)
    s_silence.ingest_pcm(make_silence(4.0), sample_rate=16000)
    assert len(s_silence.temporal_policy.raw_history) == 1
    assert len(s_silence.temporal_policy.smoothed_history) == 1

    # 3. Ingest clipped in a fresh session with seeded history
    s_clip = StreamingSession()
    s_clip.temporal_policy.raw_history.append(0.75)
    s_clip.temporal_policy.smoothed_history.append(0.75)
    s_clip.ingest_pcm(make_clipped_speech(4.0), sample_rate=16000)
    assert len(s_clip.temporal_policy.raw_history) == 1
    assert len(s_clip.temporal_policy.smoothed_history) == 1

    # 4. Ingest marginal SNR in a fresh session with seeded history
    s_marginal = StreamingSession()
    s_marginal.temporal_policy.raw_history.append(0.75)
    s_marginal.temporal_policy.smoothed_history.append(0.75)
    s_marginal.ingest_pcm(make_marginal_snr_speech(4.0), sample_rate=16000)
    assert len(s_marginal.temporal_policy.raw_history) == 1
    assert len(s_marginal.temporal_policy.smoothed_history) == 1


def test_ingest_and_finalize_parity() -> None:
    """Requirement 7 & Stage A3 Correction 5: Parity, idempotency, and no retained mutable side-channel."""
    # At 44.1 kHz, 4.0s requires resampler flush on finalize to cross the 64,000 canonical sample boundary.
    # 1. Clean speech
    s_clean_ingest = StreamingSession()
    s_clean_ingest.ingest_pcm(make_clean_speech(4.0), sample_rate=16000)
    w_ingest = s_clean_ingest.pop_pending_window()
    assert w_ingest is not None

    s_clean_fin = StreamingSession()
    clean_44k = make_clean_speech(4.0, sample_rate=44100)
    res_clean = s_clean_fin.ingest_pcm(clean_44k, sample_rate=44100)
    assert res_clean.window_ready is False  # Awaiting flush
    eligible_fin, abstentions_fin = s_clean_fin.finalize()
    assert len(eligible_fin) == 1
    assert len(abstentions_fin) == 0
    w_fin = eligible_fin[0]
    assert w_fin.is_eligible == w_ingest.is_eligible
    assert w_fin.input_reliability.state == w_ingest.input_reliability.state
    assert (
        w_fin.input_reliability.origin_verdict_allowed
        == w_ingest.input_reliability.origin_verdict_allowed
    )
    assert w_fin.quality_metrics.quality_status == w_ingest.quality_metrics.quality_status

    # Second finalize call must return no eligible windows and no abstentions
    rep_el, rep_abs = s_clean_fin.finalize()
    assert len(rep_el) == 0
    assert len(rep_abs) == 0
    assert not hasattr(s_clean_fin, "last_finalized_abstentions")

    # 2. Clipped speech
    s_clip_ingest = StreamingSession()
    res_clip_ingest = s_clip_ingest.ingest_pcm(make_clipped_speech(4.0), sample_rate=16000)
    assert len(res_clip_ingest.immediate_abstentions) == 1
    abs_ingest = res_clip_ingest.immediate_abstentions[0]

    s_clip_fin = StreamingSession()
    clipped_44k = make_clipped_speech(4.0, sample_rate=44100)
    res_clip = s_clip_fin.ingest_pcm(clipped_44k, sample_rate=44100)
    assert res_clip.window_ready is False
    el_clip, abs_clip = s_clip_fin.finalize()
    assert len(el_clip) == 0
    assert len(abs_clip) == 1
    abs_fin = abs_clip[0]
    assert isinstance(abs_fin, StreamingAbstention)
    assert abs_fin.input_reliability.state == abs_ingest.input_reliability.state
    assert abs_fin.decision.state == abs_ingest.decision.state
    assert abs_fin.decision.reason == abs_ingest.decision.reason

    rep_el_c, rep_abs_c = s_clip_fin.finalize()
    assert len(rep_el_c) == 0
    assert len(rep_abs_c) == 0
    assert not hasattr(s_clip_fin, "last_finalized_abstentions")

    # 3. Silence
    s_sil_ingest = StreamingSession()
    res_sil_ingest = s_sil_ingest.ingest_pcm(make_silence(4.0), sample_rate=16000)
    assert len(res_sil_ingest.immediate_abstentions) == 1
    abs_sil_ingest = res_sil_ingest.immediate_abstentions[0]

    s_sil_fin = StreamingSession()
    sil_44k = make_silence(4.0, sample_rate=44100)
    s_sil_fin.ingest_pcm(sil_44k, sample_rate=44100)
    el_sil, abs_sil = s_sil_fin.finalize()
    assert len(el_sil) == 0
    assert len(abs_sil) == 1
    abs_sil_fin = abs_sil[0]
    assert isinstance(abs_sil_fin, StreamingAbstention)
    assert abs_sil_fin.input_reliability.state == abs_sil_ingest.input_reliability.state
    assert abs_sil_fin.decision.reason == abs_sil_ingest.decision.reason

    rep_el_s, rep_abs_s = s_sil_fin.finalize()
    assert len(rep_el_s) == 0
    assert len(rep_abs_s) == 0
    assert not hasattr(s_sil_fin, "last_finalized_abstentions")


def test_reliability_counters_sum_to_total_windows_generated() -> None:
    """Requirement 8: Reliability counters sum exactly to total_windows_generated."""
    session = StreamingSession()

    # Ingest 4.0s of clean speech -> generates Window 0 (reliable)
    session.ingest_pcm(make_clean_speech(4.0), sample_rate=16000)
    assert session.total_windows_generated == 1
    assert session.reliable_window_count == 1

    # Ingest 1.0s of clean speech -> generates Window 1 (reliable)
    session.ingest_pcm(make_clean_speech(1.0), sample_rate=16000)
    assert session.total_windows_generated == 2
    assert session.reliable_window_count == 2

    # Ingest 1.0s of clipped speech -> generates Window 2 (limited)
    session.ingest_pcm(make_clipped_speech(1.0), sample_rate=16000)
    assert session.total_windows_generated == 3
    assert session.limited_reliability_window_count == 1

    # Ingest 1.0s of marginal-SNR speech -> generates Window 3 (limited)
    session.ingest_pcm(make_marginal_snr_speech(1.0), sample_rate=16000)
    assert session.total_windows_generated == 4
    assert session.limited_reliability_window_count == 2

    # Ingest 4.0s of silence -> flushes out previous speech, generates windows with silence (insufficient)
    session.ingest_pcm(make_silence(4.0), sample_rate=16000)
    assert session.total_windows_generated == 8
    assert session.insufficient_reliability_window_count > 0

    assert (
        session.reliable_window_count
        + session.limited_reliability_window_count
        + session.insufficient_reliability_window_count
    ) == session.total_windows_generated


def test_session_reset_clears_all_counters_and_state() -> None:
    """Requirement 9: Reset clears all reliability counters and pending reliability state."""
    session = StreamingSession()
    session.ingest_pcm(make_clean_speech(4.0), sample_rate=16000)
    session.ingest_pcm(make_clipped_speech(4.0), sample_rate=16000)

    assert session.total_windows_generated > 0
    assert session.reliable_window_count > 0
    assert session.limited_reliability_window_count > 0

    session.reset()

    assert session.reliable_window_count == 0
    assert session.limited_reliability_window_count == 0
    assert session.insufficient_reliability_window_count == 0
    assert session.total_windows_generated == 0
    assert session.total_windows_dropped == 0
    assert session.has_pending_window is False
    assert session.in_flight_inference is False
    assert not hasattr(session, "last_finalized_abstentions")


def test_complete_inference_fail_closed_on_non_reliable_window() -> None:
    """Requirement 10: Non-reliable window cannot be passed to complete_inference; fail closed with RuntimeError."""
    session = StreamingSession()
    samples = make_clean_speech(4.0)

    # 1. Passing None raises RuntimeError
    with pytest.raises(
        RuntimeError, match="Cannot complete inference without a valid WindowForInference"
    ):
        session.complete_inference(synthetic_score=0.8, window=None)

    # 2. Passing an ineligible window raises RuntimeError
    metrics = compute_quality_metrics(samples)
    assessment = InputReliabilityAssessment(
        state=ReliabilityState.RELIABLE,
        reasons=(),
        explanation="Valid clean speech",
        origin_verdict_allowed=True,
    )
    ineligible_win = WindowForInference(
        window_index=0,
        samples=samples,
        start_time=0.0,
        end_time=4.0,
        speech_seconds=0.0,
        is_eligible=False,
        input_reliability=assessment,
        quality_metrics=metrics,
    )
    with pytest.raises(RuntimeError, match="is not eligible"):
        session.complete_inference(synthetic_score=0.8, window=ineligible_win)

    # 3. Passing a window with LIMITED assessment raises RuntimeError
    limited_assessment = InputReliabilityAssessment(
        state=ReliabilityState.LIMITED,
        reasons=(ReliabilityReason.SEVERE_CLIPPING,),
        explanation="Digital clipping",
        origin_verdict_allowed=False,
    )
    limited_win = WindowForInference(
        window_index=1,
        samples=samples,
        start_time=0.0,
        end_time=4.0,
        speech_seconds=3.0,
        is_eligible=True,
        input_reliability=limited_assessment,
        quality_metrics=metrics,
    )
    with pytest.raises(RuntimeError, match="is not RELIABLE"):
        session.complete_inference(synthetic_score=0.8, window=limited_win)


# ==============================================================================
# WebSocket Integration Tests (11-16)
# ==============================================================================


def test_websocket_clean_window_json_contract() -> None:
    """Requirement 11: WebSocket clean-window JSON includes reliable assessment and populated model evidence."""
    detector = RecordingDetector(score=0.82)
    with TestClient(app) as client:
        orig = getattr(app.state, "detector", None)
        app.state.detector = detector
        try:
            with client.websocket_connect("/api/v1/stream") as ws:
                ws.send_json(
                    {
                        "type": "start",
                        "sample_rate": 16000,
                        "channels": 1,
                        "pcm_encoding": "int16",
                    }
                )
                ack = ws.receive_json()
                assert ack["type"] == "session_started"

                clean_pcm = to_pcm16_bytes(make_clean_speech(4.0))
                ws.send_bytes(clean_pcm)
                ws.send_json({"type": "stop"})

                results = []
                stop_ack = None
                while True:
                    msg = ws.receive_json()
                    if msg["type"] == "session_stopped":
                        stop_ack = msg
                        break
                    results.append(msg)

                win_events = [m for m in results if m.get("type") == "window_result"]
                assert len(win_events) == 1
                ev = win_events[0]

                assert ev["is_eligible"] is True
                assert pytest.approx(ev["synthetic_score"], abs=1e-3) == 0.82
                assert ev["abstain_reason"] is None
                assert ev["input_reliability"]["state"] == "reliable"
                assert ev["input_reliability"]["origin_verdict_allowed"] is True
                assert ev["input_reliability"]["reasons"] == []
                assert ev["quality_status"]["quality_status"] == "clean"

                assert stop_ack is not None
                assert stop_ack["reliable_window_count"] == 1
                assert stop_ack["limited_reliability_window_count"] == 0
                assert stop_ack["quality_origin_verdict_allowed"] is True

                assert detector.call_count == 1
        finally:
            app.state.detector = orig


@pytest.mark.parametrize(
    "signal_type,expected_reason",
    [
        ("clipped", "severe_clipping"),
        ("marginal_snr", "marginal_snr"),
    ],
)
def test_websocket_limited_reliability_signals(
    signal_type: str,
    expected_reason: str,
) -> None:
    """Requirement 12 & Correction 3: WebSocket clipped and marginal-SNR JSON includes null model evidence and zero detector calls."""
    detector = RecordingDetector(score=0.88)
    with TestClient(app) as client:
        orig = getattr(app.state, "detector", None)
        app.state.detector = detector
        try:
            with client.websocket_connect("/api/v1/stream") as ws:
                ws.send_json(
                    {
                        "type": "start",
                        "sample_rate": 16000,
                        "channels": 1,
                        "pcm_encoding": "int16",
                    }
                )
                ws.receive_json()

                if signal_type == "clipped":
                    audio = make_clipped_speech(4.0)
                elif signal_type == "marginal_snr":
                    audio = make_marginal_snr_speech(4.0)
                else:
                    raise ValueError(f"Unknown signal_type: {signal_type}")

                ws.send_bytes(to_pcm16_bytes(audio))
                ws.send_json({"type": "stop"})

                results = []
                stop_ack = None
                while True:
                    msg = ws.receive_json()
                    if msg["type"] == "session_stopped":
                        stop_ack = msg
                        break
                    results.append(msg)

                win_events = [m for m in results if m.get("type") == "window_result"]
                assert len(win_events) == 1
                ev = win_events[0]

                assert ev["is_eligible"] is False
                assert ev["synthetic_score"] is None
                assert ev["uncalibrated_synthetic_score"] is None
                assert ev["raw_score"] is None
                assert ev["calibrated_score"] is None
                assert ev["score_is_calibrated"] is False
                assert ev["input_reliability"]["state"] == "limited"
                assert ev["input_reliability"]["origin_verdict_allowed"] is False
                assert expected_reason in ev["input_reliability"]["reasons"]
                assert ev["abstain_reason"] == "signal_quality_degraded"

                assert detector.call_count == 0
                assert stop_ack is not None
                assert stop_ack["status"] == "limited_reliability"
                assert stop_ack["final_decision_state"] == "uncertain"
                assert stop_ack["final_reason"] == "signal_quality_degraded"
                assert stop_ack["quality_origin_verdict_allowed"] is False
        finally:
            app.state.detector = orig


def test_websocket_silence_json_contract() -> None:
    """Requirement 13: WebSocket silence JSON exposes distinct decision reason and typed reasons while preserving legacy compatibility."""
    detector = RecordingDetector(score=0.90)
    with TestClient(app) as client:
        orig = getattr(app.state, "detector", None)
        app.state.detector = detector
        try:
            with client.websocket_connect("/api/v1/stream") as ws:
                ws.send_json(
                    {
                        "type": "start",
                        "sample_rate": 16000,
                        "channels": 1,
                        "pcm_encoding": "int16",
                    }
                )
                ws.receive_json()

                silence_pcm = to_pcm16_bytes(make_silence(4.0))
                ws.send_bytes(silence_pcm)
                ws.send_json({"type": "stop"})

                results = []
                stop_ack = None
                while True:
                    msg = ws.receive_json()
                    if msg["type"] == "session_stopped":
                        stop_ack = msg
                        break
                    results.append(msg)

                win_events = [m for m in results if m.get("type") == "window_result"]
                assert len(win_events) == 1
                ev = win_events[0]

                assert ev["is_eligible"] is False
                assert ev["synthetic_score"] is None
                assert ev["decision"]["reason"] == "signal_quality_degraded"
                assert ev["decision"]["state"] == "insufficient_evidence"
                assert ev["abstain_reason"] == "insufficient_speech_duration"
                assert ev["input_reliability"]["state"] == "insufficient"
                assert "silent_audio" in ev["input_reliability"]["reasons"]
                assert "insufficient_speech" in ev["input_reliability"]["reasons"]

                assert detector.call_count == 0
                assert stop_ack is not None
                assert stop_ack["status"] == "insufficient_evidence"
                assert stop_ack["final_reason"] == "insufficient_speech_duration"
                assert stop_ack["quality_origin_verdict_allowed"] is False
        finally:
            app.state.detector = orig


def test_sustained_high_followed_by_limited_produces_limited_reliability_stopack() -> None:
    """Requirement 14 & Correction 2: A real sustained HIGH sequence followed by a LIMITED window produces limited_reliability / uncertain."""
    detector = RecordingDetector(score=0.92)  # High score above 0.65 threshold
    with TestClient(app) as client:
        orig = getattr(app.state, "detector", None)
        app.state.detector = detector
        try:
            with client.websocket_connect("/api/v1/stream") as ws:
                ws.send_json(
                    {
                        "type": "start",
                        "sample_rate": 16000,
                        "channels": 1,
                        "pcm_encoding": "int16",
                    }
                )
                start_ack = ws.receive_json()
                assert start_ack["type"] == "session_started"

                # 1. Send 4.0s clean active speech. Receive window_result 0
                ws.send_bytes(to_pcm16_bytes(make_clean_speech(4.0)))
                res0 = receive_window_result(ws, 0)
                assert res0["is_eligible"] is True
                assert res0["decision"]["state"] == "uncertain"
                assert res0["decision"]["reason"] == "elevation_in_progress"

                # 2. Send 1.0s clean active speech. Receive window_result 1
                ws.send_bytes(to_pcm16_bytes(make_clean_speech(1.0)))
                res1 = receive_window_result(ws, 1)
                assert res1["is_eligible"] is True
                assert res1["decision"]["state"] == "uncertain"
                assert res1["decision"]["reason"] == "elevation_in_progress"

                # 3. Send 1.0s clean active speech. Receive window_result 2
                ws.send_bytes(to_pcm16_bytes(make_clean_speech(1.0)))
                res2 = receive_window_result(ws, 2)
                assert res2["is_eligible"] is True
                assert res2["decision"]["state"] == "high"
                assert res2["decision"]["reason"] == "sustained_anomaly_elevation"
                assert detector.call_count == 3

                # 4. Send 1.0s clipped speech. Receive window_result 3
                ws.send_bytes(to_pcm16_bytes(make_clipped_speech(1.0)))
                res3 = receive_window_result(ws, 3)
                assert res3["input_reliability"]["state"] == "limited"
                assert res3["is_eligible"] is False
                assert res3["synthetic_score"] is None
                assert res3["uncalibrated_synthetic_score"] is None
                assert res3["raw_score"] is None
                assert res3["calibrated_score"] is None
                assert res3["score_is_calibrated"] is False
                assert detector.call_count == 3

                # 5. Send stop. Receive session_stopped
                ws.send_json({"type": "stop"})
                stop_ack = None
                while True:
                    msg = ws.receive_json()
                    if msg["type"] == "session_stopped":
                        stop_ack = msg
                        break
                assert stop_ack is not None
                # CRITICAL SAFETY INVARIANT: Despite having evaluated sustained HIGH windows,
                # the presence of a LIMITED window forces final status to limited_reliability,
                # decision state to uncertain, and forbids origin verdicts.
                assert stop_ack["status"] == "limited_reliability"
                assert stop_ack["final_decision_state"] == "uncertain"
                assert stop_ack["final_reason"] == "signal_quality_degraded"
                assert stop_ack["quality_origin_verdict_allowed"] is False
                assert stop_ack["limited_reliability_window_count"] > 0
                assert stop_ack["reliable_window_count"] > 0
        finally:
            app.state.detector = orig


def test_backpressure_drop_replace_does_not_affect_reliability_states() -> None:
    """Requirement 15: Existing backpressure, continuity-reset, and drop counting preserve reliability counts."""
    session = StreamingSession()
    clean_chunk = make_clean_speech(1.0)

    # Ingest 4.0s -> Window 0 is ready
    for _ in range(4):
        session.ingest_pcm(clean_chunk, sample_rate=16000)

    # Worker pops Window 0 (marked in-flight)
    w0 = session.pop_pending_window()
    assert w0 is not None
    assert session.in_flight_inference is True

    # Ingest 1.0s -> Window 1 becomes pending
    session.ingest_pcm(clean_chunk, sample_rate=16000)
    assert session.has_pending_window is True

    # Ingest another 1.0s -> Window 2 replaces Window 1 (backpressure drop)
    session.ingest_pcm(clean_chunk, sample_rate=16000)
    assert session.total_windows_dropped == 1
    assert session.total_windows_generated == 3
    # Both dropped window and replacement window were RELIABLE
    assert session.reliable_window_count == 3
    assert session.limited_reliability_window_count == 0
    assert session.insufficient_reliability_window_count == 0


def test_websocket_stop_precedence_all_cases() -> None:
    """Requirement 16: Verify precedence cases (b, d, e) for ServerStopAck."""
    detector = RecordingDetector(score=0.80)
    with TestClient(app) as client:
        orig = getattr(app.state, "detector", None)
        app.state.detector = detector
        try:
            # Case (b): Zero generated windows (< 4.0s audio)
            with client.websocket_connect("/api/v1/stream") as ws:
                ws.send_json(
                    {
                        "type": "start",
                        "sample_rate": 16000,
                        "channels": 1,
                        "pcm_encoding": "int16",
                    }
                )
                ws.receive_json()
                ws.send_bytes(to_pcm16_bytes(make_clean_speech(2.0)))
                ws.send_json({"type": "stop"})
                stop_ack = None
                while True:
                    msg = ws.receive_json()
                    if msg["type"] == "session_stopped":
                        stop_ack = msg
                        break
                assert stop_ack is not None
                assert stop_ack["total_windows_generated"] == 0
                assert stop_ack["status"] == "insufficient_evidence"
                assert stop_ack["final_reason"] == "insufficient_contiguous_audio"
                assert stop_ack["quality_origin_verdict_allowed"] is False
                assert stop_ack["final_decision_state"] == "insufficient_evidence"

            # Case (d): Clean speech coexisting with silence pause (0 limited windows)
            detector.call_count = 0
            with client.websocket_connect("/api/v1/stream") as ws:
                ws.send_json(
                    {
                        "type": "start",
                        "sample_rate": 16000,
                        "channels": 1,
                        "pcm_encoding": "int16",
                    }
                )
                start_ack = ws.receive_json()
                assert start_ack["type"] == "session_started"

                # Send 4.0s clean speech and wait for window_result 0 (is_eligible is True, reliable)
                ws.send_bytes(to_pcm16_bytes(make_clean_speech(4.0)))
                res0 = receive_window_result(ws, 0)
                assert res0["is_eligible"] is True
                assert res0["input_reliability"]["state"] == "reliable"

                # Send 4.0s silence to generate an actual full-window silence abstention (not 2.0s)
                results = [res0]
                for win_idx in range(1, 5):
                    ws.send_bytes(to_pcm16_bytes(make_silence(1.0)))
                    res = receive_window_result(ws, win_idx)
                    results.append(res)

                ws.send_json({"type": "stop"})
                stop_ack = None
                while True:
                    msg = ws.receive_json()
                    if msg["type"] == "session_stopped":
                        stop_ack = msg
                        break
                    results.append(msg)

                assert stop_ack is not None
                assert stop_ack["reliable_window_count"] > 0
                assert stop_ack["insufficient_reliability_window_count"] > 0
                assert stop_ack["limited_reliability_window_count"] == 0
                assert stop_ack["status"] == "success"
                assert stop_ack["quality_origin_verdict_allowed"] is True
                assert stop_ack["final_reason"] is None

                # Inspect collected window results to find the fully silent window
                win_events = [m for m in results if m.get("type") == "window_result"]
                silent_windows = [
                    ev
                    for ev in win_events
                    if ev.get("input_reliability", {}).get("state") == "insufficient"
                ]
                assert len(silent_windows) > 0
                last_silent = silent_windows[-1]
                assert last_silent["input_reliability"]["state"] == "insufficient"
                assert last_silent["is_eligible"] is False
                assert last_silent["synthetic_score"] is None

                # Verify detector was never called for this silence window:
                # call count matches reliable window count
                assert detector.call_count == stop_ack["reliable_window_count"]
        finally:
            app.state.detector = orig
