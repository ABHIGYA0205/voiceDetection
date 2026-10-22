"""Comprehensive unit tests for VoiceShield framework-independent streaming core."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from voiceshield.config.schema import TemporalPolicyConfig
from voiceshield.dsp.exceptions import UnsupportedAudioError
from voiceshield.streaming.policy import (
    DecisionState,
    ReasonCode,
    TemporalDecisionPolicy,
)
from voiceshield.streaming.resampler import StatefulResampler
from voiceshield.streaming.ring_buffer import AudioRingBuffer
from voiceshield.streaming.session import StreamingSession


def generate_sine(
    duration_sec: float,
    sample_rate: int,
    freq: float = 440.0,
    amplitude: float = 0.5,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Generate a pure float32 sine wave."""
    num_samples = int(round(duration_sec * sample_rate))
    t = np.linspace(0, duration_sec, num_samples, endpoint=False, dtype=np.float32)
    arr: np.ndarray[Any, np.dtype[np.float32]] = np.asarray(
        amplitude * np.sin(2.0 * np.pi * freq * t), dtype=np.float32
    )
    return arr


# ==============================================================================
# 1. Chunk-Boundary Invariance Tests
# ==============================================================================


def test_chunk_boundary_invariance_resampling_48k_to_16k() -> None:
    """Verify that resampling across arbitrary small chunks produces output identical to a single batch."""
    sr_in = 48000
    sr_out = 16000
    duration = 1.0  # 1 second
    full_audio = generate_sine(duration, sr_in, freq=440.0, amplitude=0.5)

    # 1. Process in single batch
    resampler_batch = StatefulResampler(orig_sr=sr_in, target_sr=sr_out)
    batch_output = resampler_batch.process(full_audio)

    # 2. Process in arbitrary chunks (e.g., 237, 512, 1024 samples)
    resampler_stream = StatefulResampler(orig_sr=sr_in, target_sr=sr_out)
    chunk_outputs = []
    idx = 0
    chunk_sizes = [237, 512, 1024, 300, 777]
    c_idx = 0
    while idx < len(full_audio):
        step = min(len(full_audio) - idx, chunk_sizes[c_idx % len(chunk_sizes)])
        chunk = full_audio[idx : idx + step]
        out_chunk = resampler_stream.process(chunk)
        if out_chunk.size > 0:
            chunk_outputs.append(out_chunk)
        idx += step
        c_idx += 1

    streamed_output = np.concatenate(chunk_outputs)

    # The lengths must match and the maximum sample difference must be near float32 precision
    assert len(batch_output) == len(streamed_output)
    max_diff = float(np.max(np.abs(batch_output - streamed_output)))
    assert max_diff < 1e-5, f"Max difference across chunk boundaries exceeded tolerance: {max_diff}"


def test_chunk_boundary_invariance_stereo_downmix() -> None:
    """Verify stereo downmix and chunking match batch processing."""
    sr = 16000
    duration = 0.5
    left = generate_sine(duration, sr, freq=440.0, amplitude=0.4)
    right = generate_sine(duration, sr, freq=880.0, amplitude=0.2)
    stereo = np.column_stack((left, right))  # shape: (8000, 2)

    resampler_batch = StatefulResampler(orig_sr=sr, target_sr=sr, channels=2)
    batch_out = resampler_batch.process(stereo)

    resampler_stream = StatefulResampler(orig_sr=sr, target_sr=sr, channels=2)
    chunk1 = resampler_stream.process(stereo[:3000])
    chunk2 = resampler_stream.process(stereo[3000:])
    stream_out = np.concatenate([chunk1, chunk2])

    assert np.allclose(batch_out, stream_out, atol=1e-6)


# ==============================================================================
# 2. Bounded Memory Tests
# ==============================================================================


def test_ring_buffer_bounded_memory() -> None:
    """Verify that circular ring buffer retains bounded memory under prolonged streaming."""
    capacity = 160000  # 10 seconds @ 16 kHz
    ring = AudioRingBuffer(capacity_samples=capacity)

    # Stream 60 seconds (960,000 samples) in 100ms (1600 sample) chunks
    chunk = np.ones(1600, dtype=np.float32) * 0.5
    total_chunks = 600

    for _ in range(total_chunks):
        ring.write(chunk)

    # Verify memory invariants
    assert ring._buffer.shape == (capacity,)
    assert ring.total_samples_written == total_chunks * 1600
    assert ring.available_samples == capacity
    assert ring.is_full is True

    # Read the 4-second tail (64,000 samples)
    tail = ring.read_tail(64000)
    assert tail.shape == (64000,)
    assert np.allclose(tail, 0.5)


def test_ring_buffer_invalid_read_raises() -> None:
    """Verify that requesting more samples than available or negative samples raises ValueError."""
    ring = AudioRingBuffer(capacity_samples=1000)
    ring.write(np.ones(200, dtype=np.float32))

    with pytest.raises(ValueError, match="Requested 300 samples, but only 200 are available"):
        ring.read_tail(300)

    with pytest.raises(ValueError, match="num_samples must be positive"):
        ring.read_tail(0)


# ==============================================================================
# 3. Timing Gaps Detection Tests
# ==============================================================================


def test_timing_gap_detection() -> None:
    """Verify that timestamp jumps exceeding the timing gap threshold flag a discontinuity."""
    session = StreamingSession()
    sr = 16000
    chunk = generate_sine(0.1, sr, amplitude=0.3)  # 100ms chunk

    # Normal ingestion: consecutive timestamps
    res1 = session.ingest_pcm(chunk, sample_rate=sr, timestamp=10.0)
    assert res1.quality_flags.has_timing_gap is False
    assert res1.quality_flags.gap_count == 0

    res2 = session.ingest_pcm(chunk, sample_rate=sr, timestamp=10.1)
    assert res2.quality_flags.has_timing_gap is False
    assert res2.quality_flags.gap_count == 0

    # Discontinuity: jump from 10.2 to 11.5 (1.3s gap > 0.25s threshold)
    res3 = session.ingest_pcm(chunk, sample_rate=sr, timestamp=11.5)
    assert res3.quality_flags.has_timing_gap is True
    assert res3.quality_flags.gap_count == 1
    assert session.quality_flags.gap_count == 1


# ==============================================================================
# 4. Insufficient Speech Duration Tests
# ==============================================================================


def test_insufficient_speech_abstention() -> None:
    """Verify that windows with insufficient active speech are abstained from inference."""
    session = StreamingSession()
    sr = 16000
    silence_chunk = np.zeros(16000, dtype=np.float32)  # 1.0 second of silence

    # Feed 4 seconds of silence to trigger first window
    for _ in range(4):
        res = session.ingest_pcm(silence_chunk, sample_rate=sr)

    assert res.window_ready is True
    assert res.pending_window_available is False  # Abstained: NOT queued for model inference
    assert res.immediate_decision is not None
    assert res.immediate_decision.state == DecisionState.INSUFFICIENT_EVIDENCE
    assert res.immediate_decision.reason == ReasonCode.INSUFFICIENT_SPEECH_DURATION
    assert session.has_pending_window is False


def test_sufficient_speech_becomes_eligible() -> None:
    """Verify that windows with active speech exceed min_speech_seconds and become eligible."""
    session = StreamingSession()
    sr = 16000
    speech_chunk = generate_sine(1.0, sr, freq=300.0, amplitude=0.4)

    # Ingest 4 seconds of active speech
    for _ in range(4):
        res = session.ingest_pcm(speech_chunk, sample_rate=sr)

    assert res.window_ready is True
    assert res.pending_window_available is True
    assert res.immediate_decision is None
    assert session.has_pending_window is True

    window = session.pop_pending_window()
    assert window is not None
    assert window.is_eligible is True
    assert window.speech_seconds >= 3.8
    assert session.in_flight_inference is True


# ==============================================================================
# 5. Backpressure and Drop/Replace Tests
# ==============================================================================


def test_backpressure_and_drop_replace_stale_windows() -> None:
    """Verify at most 1 in-flight inference, and that new windows drop/replace stale pending windows."""
    session = StreamingSession()
    sr = 16000
    chunk = generate_sine(1.0, sr, amplitude=0.4)

    # 1. Ingest 4 seconds -> Window 0 ready
    for _ in range(4):
        session.ingest_pcm(chunk, sample_rate=sr)

    # Worker pops Window 0
    w0 = session.pop_pending_window()
    assert w0 is not None
    assert w0.window_index == 0
    assert session.in_flight_inference is True
    assert session.has_pending_window is False

    # 2. Ingest 1 more second while inference is in-flight -> Window 1 generated
    session.ingest_pcm(chunk, sample_rate=sr)
    assert session.has_pending_window is True
    # Worker tries to pop while busy -> must return None
    assert session.pop_pending_window() is None
    assert session._pending_window.window_index == 1

    # 3. Ingest another second while inference is STILL in-flight -> Window 2 generated
    # This must replace stale Window 1!
    session.ingest_pcm(chunk, sample_rate=sr)
    assert session.has_pending_window is True
    assert session._pending_window.window_index == 2  # Window 1 was replaced

    # 4. Worker completes inference for Window 0
    decision = session.complete_inference(calibrated_score=0.85, window=w0)
    assert session.in_flight_inference is False
    assert decision.state in (DecisionState.UNCERTAIN, DecisionState.HIGH)

    # 5. Worker pops next window -> retrieves the freshest (Window 2)
    w_next = session.pop_pending_window()
    assert w_next is not None
    assert w_next.window_index == 2
    assert session.has_pending_window is False
    assert session.in_flight_inference is True


# ==============================================================================
# 6. Disconnect Reset Tests
# ==============================================================================


def test_disconnect_reset_cleans_all_state() -> None:
    """Verify session.reset() cleanly wipes buffers, resampler state, and counters."""
    session = StreamingSession()
    sr = 48000
    chunk = generate_sine(1.0, sr, amplitude=0.9995)  # Clipped

    session.ingest_pcm(chunk, sample_rate=sr, timestamp=5.0)
    assert session.ring_buffer.available_samples > 0
    assert session.quality_flags.is_clipped is True
    assert session.quality_flags.clipping_count > 0

    session.reset()

    assert session.ring_buffer.available_samples == 0
    assert session.ring_buffer.total_samples_written == 0
    assert session.in_flight_inference is False
    assert session.has_pending_window is False
    assert session.window_index == 0
    assert session.quality_flags.clipping_count == 0
    assert session.quality_flags.is_clipped is False
    assert session._stream_start_timestamp is None


# ==============================================================================
# 7. Concurrent Sessions Isolation Tests
# ==============================================================================


def test_concurrent_sessions_isolation() -> None:
    """Verify that multiple concurrent StreamingSession instances maintain strict isolation."""
    session_a = StreamingSession()
    session_b = StreamingSession()
    session_c = StreamingSession()

    sr = 16000
    sine = generate_sine(1.0, sr, freq=440.0, amplitude=0.5)
    silence = np.zeros(16000, dtype=np.float32)
    clipped = np.ones(16000, dtype=np.float32)  # amplitude 1.0 (clipped)

    # Interleave chunk ingestion
    for _ in range(4):
        session_a.ingest_pcm(sine, sample_rate=sr)
        session_b.ingest_pcm(silence, sample_rate=sr)
        session_c.ingest_pcm(clipped, sample_rate=sr)

    # Session A: Active speech, eligible window ready
    assert session_a.has_pending_window is True
    assert session_a.quality_flags.is_clipped is False

    # Session B: Silence, abstained, no pending window
    assert session_b.has_pending_window is False
    assert session_b.quality_flags.is_silent is True

    # Session C: Clipped signal
    assert session_c.quality_flags.is_clipped is True
    assert session_c.quality_flags.clipping_count > 0


# ==============================================================================
# 8. Deterministic Temporal Decisions Tests
# ==============================================================================


def test_deterministic_ewma_smoothing() -> None:
    """Verify exact deterministic EWMA progression according to alpha formula."""
    cfg = TemporalPolicyConfig(
        smoothing_method="ewma",
        ewma_alpha=0.3,
        threshold_high=0.7,
        threshold_low=0.3,
    )
    policy = TemporalDecisionPolicy(policy_config=cfg)

    # Score 1: raw = 0.8 -> smoothed = 0.8
    d1 = policy.evaluate(
        0.8, speech_seconds=2.0, window_index=0, window_start_sec=0, window_end_sec=4
    )
    assert pytest.approx(d1.smoothed_score, abs=1e-5) == 0.8

    # Score 2: raw = 0.4 -> smoothed = 0.3 * 0.4 + 0.7 * 0.8 = 0.12 + 0.56 = 0.68
    d2 = policy.evaluate(
        0.4, speech_seconds=2.0, window_index=1, window_start_sec=1, window_end_sec=5
    )
    assert pytest.approx(d2.smoothed_score, abs=1e-5) == 0.68


def test_deterministic_median_smoothing() -> None:
    """Verify median smoothing filters out transient spikes."""
    cfg = TemporalPolicyConfig(
        smoothing_method="median",
        median_window_size=3,
        threshold_high=0.7,
        threshold_low=0.3,
    )
    policy = TemporalDecisionPolicy(policy_config=cfg)

    policy.evaluate(0.1, speech_seconds=2.0, window_index=0, window_start_sec=0, window_end_sec=4)
    policy.evaluate(0.9, speech_seconds=2.0, window_index=1, window_start_sec=1, window_end_sec=5)
    d3 = policy.evaluate(
        0.1, speech_seconds=2.0, window_index=2, window_start_sec=2, window_end_sec=6
    )

    # Median of [0.1, 0.9, 0.1] is 0.1
    assert pytest.approx(d3.smoothed_score, abs=1e-5) == 0.1
    assert d3.state == DecisionState.LOW


def test_n_of_m_elevation_and_hysteresis_cooldown() -> None:
    """Verify 3-of-5 elevation transition to HIGH and 3-drop hysteresis transition to LOW."""
    cfg = TemporalPolicyConfig(
        smoothing_method="none",  # Raw score directly controls thresholding
        threshold_high=0.65,
        threshold_low=0.35,
        elevation_n=3,
        elevation_m=5,
        hysteresis_consecutive_drops=3,
    )
    policy = TemporalDecisionPolicy(policy_config=cfg)

    # Window 0: High score (1 of 3 required) -> UNCERTAIN (ELEVATION_IN_PROGRESS)
    d0 = policy.evaluate(
        0.8, speech_seconds=2.0, window_index=0, window_start_sec=0, window_end_sec=4
    )
    assert d0.state == DecisionState.UNCERTAIN
    assert d0.reason == ReasonCode.ELEVATION_IN_PROGRESS

    # Window 1: High score (2 of 3 required) -> UNCERTAIN (ELEVATION_IN_PROGRESS)
    d1 = policy.evaluate(
        0.8, speech_seconds=2.0, window_index=1, window_start_sec=1, window_end_sec=5
    )
    assert d1.state == DecisionState.UNCERTAIN
    assert d1.reason == ReasonCode.ELEVATION_IN_PROGRESS

    # Window 2: High score (3 of 3 required) -> HIGH (SUSTAINED_ANOMALY_ELEVATION)
    d2 = policy.evaluate(
        0.8, speech_seconds=2.0, window_index=2, window_start_sec=2, window_end_sec=6
    )
    assert d2.state == DecisionState.HIGH
    assert d2.reason == ReasonCode.SUSTAINED_ANOMALY_ELEVATION

    # --- Hysteresis De-escalation ---
    # Drop 1: Low score 0.2 (< 0.35) -> UNCERTAIN (HYSTERESIS_COOLDOWN, 1 of 3 drops)
    d3 = policy.evaluate(
        0.2, speech_seconds=2.0, window_index=3, window_start_sec=3, window_end_sec=7
    )
    assert d3.state == DecisionState.UNCERTAIN
    assert d3.reason == ReasonCode.HYSTERESIS_COOLDOWN

    # Drop 2: Low score 0.2 -> UNCERTAIN (HYSTERESIS_COOLDOWN, 2 of 3 drops)
    d4 = policy.evaluate(
        0.2, speech_seconds=2.0, window_index=4, window_start_sec=4, window_end_sec=8
    )
    assert d4.state == DecisionState.UNCERTAIN
    assert d4.reason == ReasonCode.HYSTERESIS_COOLDOWN

    # Drop 3: Low score 0.2 -> LOW (LOW_ANOMALY_CLEARED, 3 of 3 drops satisfied!)
    d5 = policy.evaluate(
        0.2, speech_seconds=2.0, window_index=5, window_start_sec=5, window_end_sec=9
    )
    assert d5.state == DecisionState.LOW
    assert d5.reason == ReasonCode.LOW_ANOMALY_CLEARED


def test_invalid_audio_input_raises() -> None:
    """Verify that invalid audio parameters or non-finite inputs raise UnsupportedAudioError."""
    session = StreamingSession()
    with pytest.raises(UnsupportedAudioError):
        session.ingest_pcm(b"\x00\x00", sample_rate=0)

    with pytest.raises(UnsupportedAudioError):
        session.ingest_pcm(b"\x00\x00", sample_rate=16000, channels=0)
