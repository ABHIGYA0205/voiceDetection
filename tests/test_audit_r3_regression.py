"""Comprehensive regression tests for R3 audit repairs:
- Hop cadence and sample-accurate window scheduling (H-02)
- Continuity segments, timing discontinuities, and stale result suppression (H-05)
- Strict PCM decoding, alignment, and range validation (M-01)
- Sample-rate contract consistency (M-02)
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient

from services.api.main import app
from voiceshield.dsp.exceptions import (
    CorruptAudioError,
    EmptyAudioError,
    InvalidAudioValueError,
    UnsupportedAudioError,
)
from voiceshield.dsp.preprocessor import load_audio
from voiceshield.streaming.policy import DecisionState
from voiceshield.streaming.resampler import (
    StatefulResampler,
    compute_raw_frame_count,
)
from voiceshield.streaming.session import StreamingSession


def _generate_sine_pcm(
    duration_sec: float,
    sample_rate: int = 16000,
    channels: int = 1,
    freq: float = 440.0,
    amplitude: float = 0.5,
    dtype: str = "int16",
) -> bytes:
    """Generate deterministic synthetic PCM sine wave bytes."""
    num_samples = int(round(duration_sec * sample_rate))
    t = np.linspace(0, duration_sec, num_samples, endpoint=False, dtype=np.float32)
    mono = amplitude * np.sin(2.0 * np.pi * freq * t)

    if channels == 2:
        interleaved = np.column_stack([mono, mono]).reshape(-1)
    else:
        interleaved = mono

    if dtype == "int16":
        scaled = np.clip(interleaved * 32767.0, -32768.0, 32767.0).astype("<i2")
        return scaled.tobytes()
    elif dtype == "int32":
        scaled = np.clip(interleaved * 2147483647.0, -2147483648.0, 2147483647.0).astype("<i4")
        return scaled.tobytes()
    elif dtype == "float32":
        return interleaved.astype("<f4").tobytes()
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


# ==============================================================================
# SECTION A & F: Cadence and Window Scheduler
# ==============================================================================


def test_cadence_two_four_second_frames_silence_abstention() -> None:
    """Reproduction of H-02: two 4-second frames must produce logical windows ending at 4, 5, 6, 7, 8s."""
    session = StreamingSession()
    sr = 16000
    # Two 4-second silence frames (128,000 bytes each)
    silence_frame = np.zeros(sr * 4, dtype="<i2").tobytes()
    assert len(silence_frame) == 128000

    res1 = session.ingest_pcm(silence_frame, sample_rate=sr, timestamp=0.0, dtype="int16")
    assert res1.windows_generated_this_ingest == 1
    assert res1.generated_window_indices == [0]
    assert res1.generated_window_end_times == [4.0]
    assert len(res1.immediate_decisions) == 1
    assert res1.immediate_decisions[0].synthetic_score is None
    assert res1.immediate_decisions[0].state == DecisionState.INSUFFICIENT_EVIDENCE

    res2 = session.ingest_pcm(silence_frame, sample_rate=sr, timestamp=4.0, dtype="int16")
    assert res2.windows_generated_this_ingest == 4
    assert res2.generated_window_indices == [1, 2, 3, 4]
    assert res2.generated_window_end_times == [5.0, 6.0, 7.0, 8.0]
    assert len(res2.immediate_decisions) == 4
    for dec in res2.immediate_decisions:
        assert dec.synthetic_score is None
        assert dec.state == DecisionState.INSUFFICIENT_EVIDENCE

    assert session.total_windows_generated == 5
    assert session.total_windows_dropped == 0


def test_cadence_two_four_second_frames_active_speech_backpressure_drop() -> None:
    """Two 4-second speech frames produce 5 logical windows; stale pending windows are counted as dropped."""
    session = StreamingSession()
    sr = 16000
    speech_frame = _generate_sine_pcm(4.0, sample_rate=sr, freq=440.0, amplitude=0.5, dtype="int16")

    # Ingest first 4s frame -> generates Window 0 (ends at 4.0s)
    res1 = session.ingest_pcm(speech_frame, sample_rate=sr, timestamp=0.0, dtype="int16")
    assert res1.windows_generated_this_ingest == 1
    assert res1.generated_window_indices == [0]
    assert res1.generated_window_end_times == [4.0]
    assert session.has_pending_window is True

    # Pop Window 0 for simulated in-flight inference
    win0 = session.pop_pending_window()
    assert win0 is not None
    assert win0.window_index == 0
    assert session.in_flight_inference is True

    # Ingest second 4s frame while Window 0 is still in flight
    # Generates Windows 1, 2, 3, 4 (ends at 5, 6, 7, 8s)
    # Window 1 becomes pending, Window 2 replaces 1 (drop), Window 3 replaces 2 (drop), Window 4 replaces 3 (drop)
    res2 = session.ingest_pcm(speech_frame, sample_rate=sr, timestamp=4.0, dtype="int16")
    assert res2.windows_generated_this_ingest == 4
    assert res2.generated_window_indices == [1, 2, 3, 4]
    assert res2.generated_window_end_times == [5.0, 6.0, 7.0, 8.0]
    assert res2.windows_dropped_this_ingest == 3
    assert session.total_windows_generated == 5
    assert session.total_windows_dropped == 3

    # Freshest pending window is Window 4
    assert session.has_pending_window is True
    assert session._pending_window is not None
    assert session._pending_window.window_index == 4


@pytest.mark.parametrize("frame_ms", [20, 100, 500, 1000, 4000])
def test_cadence_various_chunk_durations_16k(frame_ms: int) -> None:
    """Verify sample-accurate hop boundaries regardless of incoming chunk size at 16 kHz."""
    session = StreamingSession()
    sr = 16000
    duration_sec = frame_ms / 1000.0
    pcm_chunk = _generate_sine_pcm(duration_sec, sample_rate=sr, freq=400.0, amplitude=0.4)

    total_stream_sec = 12.0
    num_frames = int(round(total_stream_sec / duration_sec))

    all_end_times: list[float] = []
    all_indices: list[int] = []

    for i in range(num_frames):
        ts = i * duration_sec
        res = session.ingest_pcm(pcm_chunk, sample_rate=sr, timestamp=ts, dtype="int16")
        all_end_times.extend(res.generated_window_end_times)
        all_indices.extend(res.generated_window_indices)

    # In 12.0 seconds of audio: windows end at 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 (9 windows)
    assert len(all_end_times) == 9
    expected_ends = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    assert [round(t, 2) for t in all_end_times] == expected_ends
    assert all_indices == list(range(9))


@pytest.mark.parametrize("sr", [8000, 16000, 44100, 48000])
def test_cadence_non_16k_rates_remain_on_logical_grid(sr: int) -> None:
    """Non-16kHz rates generate canonical windows strictly along the 1-second logical grid."""
    session = StreamingSession()
    chunk_sec = 0.5  # 500ms chunks
    chunk_bytes = _generate_sine_pcm(chunk_sec, sample_rate=sr, freq=400.0, amplitude=0.4)

    all_end_times: list[float] = []
    all_indices: list[int] = []
    total_sec = 10.0
    num_chunks = int(round(total_sec / chunk_sec))

    for i in range(num_chunks):
        ts = i * chunk_sec
        res = session.ingest_pcm(chunk_bytes, sample_rate=sr, timestamp=ts, dtype="int16")
        all_end_times.extend(res.generated_window_end_times)
        all_indices.extend(res.generated_window_indices)

    # While stream continues, generated windows remain on exact 1-second logical grid
    assert len(all_end_times) >= 6
    assert all_indices == list(range(len(all_indices)))
    for idx, end_t in enumerate(all_end_times):
        expected_t = 4.0 + idx * 1.0
        assert abs(end_t - expected_t) < 1e-4


def test_cadence_long_stream_no_timestamp_drift() -> None:
    """Long stream (30 seconds in 100ms frames) exhibits zero cumulative timestamp drift."""
    session = StreamingSession()
    sr = 16000
    chunk_sec = 0.1
    chunk_bytes = _generate_sine_pcm(chunk_sec, sample_rate=sr, freq=300.0, amplitude=0.3)

    all_end_times: list[float] = []
    for i in range(300):  # 300 * 0.1s = 30.0s
        ts = i * chunk_sec
        res = session.ingest_pcm(chunk_bytes, sample_rate=sr, timestamp=ts, dtype="int16")
        all_end_times.extend(res.generated_window_end_times)

    # 30.0s should produce windows from 4.0 to 30.0 (27 windows)
    assert len(all_end_times) == 27
    for idx, end_t in enumerate(all_end_times):
        expected_t = 4.0 + idx * 1.0
        assert abs(end_t - expected_t) < 1e-6


# ==============================================================================
# SECTION C: Discontinuity & Continuity Segments
# ==============================================================================


def test_discontinuity_gap_before_first_complete_window() -> None:
    """A timing gap occurring before the first complete 4s window resets the scheduler."""
    session = StreamingSession()
    sr = 16000
    # Ingest 2 seconds of audio
    c2 = _generate_sine_pcm(2.0, sample_rate=sr, freq=440.0)
    session.ingest_pcm(c2, sample_rate=sr, timestamp=0.0)
    assert session.ring_buffer.available_samples == 32000
    assert session.total_windows_generated == 0

    # Ingest next chunk after a 2.0s gap (expected next was 2.0, actual is 4.0)
    res = session.ingest_pcm(c2, sample_rate=sr, timestamp=4.0)
    assert res.continuity_reset is True
    assert res.continuity_segment_id == 1
    assert session.quality_flags.gap_count == 1
    # Pre-gap 2 seconds were cleared; only the post-gap 2 seconds are in the buffer
    assert session.ring_buffer.available_samples == 32000
    assert session.total_windows_generated == 0


def test_discontinuity_gap_after_valid_window() -> None:
    """Gap after valid window resets temporal policy history and requires 4 contiguous post-gap seconds."""
    session = StreamingSession()
    sr = 16000
    # Ingest 4.0s -> generates Window 0
    c4 = _generate_sine_pcm(4.0, sample_rate=sr, freq=440.0)
    res1 = session.ingest_pcm(c4, sample_rate=sr, timestamp=0.0)
    assert res1.windows_generated_this_ingest == 1
    assert res1.generated_window_indices == [0]

    # Pop Window 0 and evaluate with high synthetic score
    w0 = session.pop_pending_window()
    assert w0 is not None
    dec0 = session.complete_inference(synthetic_score=0.9, window=w0)
    assert dec0 is not None
    assert len(session.temporal_policy.smoothed_history) == 1

    # Induce timing gap at t = 6.0s (expected was 4.0s)
    c1 = _generate_sine_pcm(1.0, sample_rate=sr, freq=440.0)
    res2 = session.ingest_pcm(c1, sample_rate=sr, timestamp=6.0)
    assert res2.continuity_reset is True
    assert res2.continuity_segment_id == 1
    # Temporal history wiped
    assert len(session.temporal_policy.smoothed_history) == 0
    assert len(session.temporal_policy.raw_history) == 0

    # Ingest 3 more seconds post-gap (total 4.0s contiguous in segment 1)
    c3 = _generate_sine_pcm(3.0, sample_rate=sr, freq=440.0)
    res3 = session.ingest_pcm(c3, sample_rate=sr, timestamp=7.0)
    assert res3.windows_generated_this_ingest == 1
    # Window index is globally monotonic (window 1, segment 1)
    assert res3.generated_window_indices == [1]
    assert session._pending_window is not None
    assert session._pending_window.window_index == 1
    assert session._pending_window.continuity_segment_id == 1
    # Timestamps are relative to segment start (6.0s + 4.0s = 10.0s end)
    assert round(res3.generated_window_end_times[0], 2) == 10.0


def test_discontinuity_stale_pre_gap_inference_suppression() -> None:
    """In-flight inference from pre-gap segment completing after continuity reset is suppressed."""
    session = StreamingSession()
    sr = 16000
    c4 = _generate_sine_pcm(4.0, sample_rate=sr, freq=440.0)
    session.ingest_pcm(c4, sample_rate=sr, timestamp=0.0)

    # In-flight window from segment 0
    stale_win = session.pop_pending_window()
    assert stale_win is not None
    assert stale_win.continuity_segment_id == 0
    assert session.in_flight_inference is True

    # Gap occurs while inference is in flight
    session.ingest_pcm(c4, sample_rate=sr, timestamp=8.0)
    assert session.continuity_segment_id == 1

    # Stale inference returns: complete_inference must return None (suppressed)
    stale_dec = session.complete_inference(synthetic_score=0.95, window=stale_win)
    assert stale_dec is None
    # Temporal policy remains clean (unpolluted by stale pre-gap score)
    assert len(session.temporal_policy.smoothed_history) == 0
    assert session.in_flight_inference is False


def test_discontinuity_materially_overlapping_and_out_of_order() -> None:
    """Out-of-order and materially overlapping timestamps trigger continuity reset."""
    session = StreamingSession()
    sr = 16000
    c1 = _generate_sine_pcm(1.0, sample_rate=sr)

    # Frame 1 at t = 1.0 (ends at 2.0)
    session.ingest_pcm(c1, sample_rate=sr, timestamp=1.0)
    assert session.continuity_segment_id == 0

    # Out-of-order frame at t = 0.5 (< 1.0)
    res_ooo = session.ingest_pcm(c1, sample_rate=sr, timestamp=0.5)
    assert res_ooo.continuity_reset is True
    assert res_ooo.continuity_segment_id == 1

    # Frame at t = 1.5 (contiguous following frame 2 which was [0.5, 1.5], ends at 2.5)
    res_cont = session.ingest_pcm(c1, sample_rate=sr, timestamp=1.5)
    assert res_cont.continuity_reset is False
    assert res_cont.continuity_segment_id == 1

    # Overlapping frame at t = 1.7 (expected was 2.5; overlap of 0.8s > 0.25s threshold)
    res_overlap = session.ingest_pcm(c1, sample_rate=sr, timestamp=1.7)
    assert res_overlap.continuity_reset is True
    assert res_overlap.continuity_segment_id == 2


def test_discontinuity_non_finite_timestamp() -> None:
    """Non-finite timestamp (NaN / Inf) triggers continuity reset."""
    session = StreamingSession()
    sr = 16000
    c1 = _generate_sine_pcm(1.0, sample_rate=sr)

    session.ingest_pcm(c1, sample_rate=sr, timestamp=0.0)
    res_nan = session.ingest_pcm(c1, sample_rate=sr, timestamp=float("nan"))
    assert res_nan.continuity_reset is True
    assert res_nan.continuity_segment_id == 1


# ==============================================================================
# SECTION D: Strict PCM Decoding and Frame Validation
# ==============================================================================


def test_pcm_validation_empty_frames_rejected() -> None:
    """Empty binary frames and zero-sample arrays are rejected."""
    resampler = StatefulResampler(orig_sr=16000)
    with pytest.raises(EmptyAudioError):
        resampler.process(b"")

    with pytest.raises(EmptyAudioError):
        resampler.process(np.empty(0, dtype=np.float32))

    with pytest.raises(EmptyAudioError):
        resampler.process(torch.empty(0, dtype=torch.float32))


def test_pcm_validation_odd_length_and_misaligned_rejected() -> None:
    """Odd-length int16 and misaligned int32/float32 buffers are rejected."""
    resampler = StatefulResampler(orig_sr=16000, dtype="int16")
    with pytest.raises(CorruptAudioError) as exc_info:
        resampler.process(b"\x00\x01\x02")  # 3 bytes (odd)
    assert "Odd-length int16 buffer rejected" in str(exc_info.value)

    resampler_f32 = StatefulResampler(orig_sr=16000, dtype="float32")
    with pytest.raises(CorruptAudioError) as exc_info2:
        resampler_f32.process(b"\x00\x01\x02\x03\x04")  # 5 bytes (not divisible by 4)
    assert "not divisible by 4" in str(exc_info2.value)


def test_pcm_validation_stereo_interleaved_alignment_rejected() -> None:
    """Stereo buffer not divisible by complete interleaved frame width is rejected."""
    # Stereo int16: frame width = 2 ch * 2 bytes = 4 bytes
    resampler_stereo = StatefulResampler(orig_sr=16000, channels=2, dtype="int16")
    with pytest.raises(CorruptAudioError) as exc_info:
        resampler_stereo.process(b"\x00\x00\x01\x01\x02\x02")  # 6 bytes (divisible by 2, but not 4)
    assert "complete frame width" in str(exc_info.value)


def test_pcm_validation_non_finite_and_out_of_bounds_floats_rejected() -> None:
    """NaN, Inf, and floats outside [-1.0, 1.0] are strictly rejected."""
    resampler = StatefulResampler(orig_sr=16000, dtype="float32")

    # NaN float32 bytes
    nan_bytes = np.array([0.0, np.nan, 0.5], dtype="<f4").tobytes()
    with pytest.raises(InvalidAudioValueError) as exc1:
        resampler.process(nan_bytes)
    assert "non-finite" in str(exc1.value)

    # Inf float32 bytes
    inf_bytes = np.array([0.0, np.inf], dtype="<f4").tobytes()
    with pytest.raises(InvalidAudioValueError) as exc2:
        resampler.process(inf_bytes)
    assert "non-finite" in str(exc2.value)

    # Out-of-bounds float (> 1.0)
    oob_bytes = np.array([0.0, 1.05], dtype="<f4").tobytes()
    with pytest.raises(InvalidAudioValueError) as exc3:
        resampler.process(oob_bytes)
    assert "out of valid range" in str(exc3.value)

    # Out-of-bounds float (< -1.0)
    oob_neg = np.array([-1.001, 0.0], dtype="<f4").tobytes()
    with pytest.raises(InvalidAudioValueError) as exc4:
        resampler.process(oob_neg)
    assert "out of valid range" in str(exc4.value)


def test_pcm_validation_boundaries_accepted() -> None:
    """Values exactly at -1.0, +1.0 and integer boundaries are accepted."""
    resampler_f = StatefulResampler(orig_sr=16000, dtype="float32")
    bound_f32 = np.array([-1.0, 0.0, 1.0], dtype="<f4").tobytes()
    out = resampler_f.process(bound_f32)
    assert out.size == 3
    assert out[0] == pytest.approx(-1.0, abs=1e-6)
    assert out[2] == pytest.approx(1.0, abs=1e-6)

    # Int16 min/max
    resampler_i16 = StatefulResampler(orig_sr=16000, dtype="int16")
    bound_i16 = np.array([-32768, 32767], dtype="<i2").tobytes()
    out_i16 = resampler_i16.process(bound_i16)
    assert out_i16.size == 2
    assert out_i16[0] == pytest.approx(-1.0, abs=1e-4)


def test_pcm_validation_midstream_format_immutability() -> None:
    """Mid-stream sample rate, channel, or dtype changes are rejected."""
    session = StreamingSession()
    sr = 16000
    c1 = _generate_sine_pcm(0.5, sample_rate=sr, dtype="int16")
    session.ingest_pcm(c1, sample_rate=sr, dtype="int16")

    # Mismatched sample rate
    with pytest.raises(UnsupportedAudioError) as exc_sr:
        session.ingest_pcm(c1, sample_rate=48000, dtype="int16")
    assert "Mid-stream audio format change rejected" in str(exc_sr.value)

    # Mismatched channel count
    with pytest.raises(UnsupportedAudioError) as exc_ch:
        session.ingest_pcm(c1, sample_rate=sr, channels=2, dtype="int16")
    assert "Mid-stream audio format change rejected" in str(exc_ch.value)

    # Mismatched dtype
    with pytest.raises(UnsupportedAudioError) as exc_dt:
        session.ingest_pcm(c1, sample_rate=sr, dtype="float32")
    assert "Mid-stream audio format change rejected" in str(exc_dt.value)


def test_compute_raw_frame_count_shapes() -> None:
    """Frame count is computed accurately across shapes and multichannel layouts."""
    # Bytes mono int16: 3200 bytes = 1600 samples
    assert compute_raw_frame_count(b"\x00" * 3200, channels=1, dtype="int16") == 1600
    # Bytes stereo int16: 3200 bytes = 800 frames
    assert compute_raw_frame_count(b"\x00" * 3200, channels=2, dtype="int16") == 800

    # 1D array mono
    arr1d = np.zeros(1000, dtype=np.float32)
    assert compute_raw_frame_count(arr1d, channels=1) == 1000
    # 1D array stereo interleaved
    assert compute_raw_frame_count(arr1d, channels=2) == 500

    # 2D array (samples, channels): (1000, 2)
    arr_sc = np.zeros((1000, 2), dtype=np.float32)
    assert compute_raw_frame_count(arr_sc, channels=2) == 1000

    # 2D array (channels, samples): (2, 1000)
    arr_cs = np.zeros((2, 1000), dtype=np.float32)
    assert compute_raw_frame_count(arr_cs, channels=2) == 1000


# ==============================================================================
# SECTION E: Sample Rate Contract
# ==============================================================================


@pytest.mark.parametrize("bad_sr", [4000, 7999, 192001])
def test_sample_rate_contract_rejected(bad_sr: int) -> None:
    """Rates below 8,000 or above 192,000 Hz are rejected consistently across components."""
    # 1. StatefulResampler
    with pytest.raises(UnsupportedAudioError) as exc_res:
        StatefulResampler(orig_sr=bad_sr)
    assert "Permitted range is 8,000 Hz to 192,000 Hz." in str(exc_res.value)

    # 2. StreamingSession
    session = StreamingSession()
    dummy_bytes = b"\x00\x00" * 100
    with pytest.raises(UnsupportedAudioError) as exc_sess:
        session.ingest_pcm(dummy_bytes, sample_rate=bad_sr)
    assert "Permitted range is 8,000 Hz to 192,000 Hz." in str(exc_sess.value)

    # 3. load_audio
    buf = io.BytesIO()
    sf.write(buf, np.zeros(100, dtype=np.float32), bad_sr, format="WAV", subtype="PCM_16")
    with pytest.raises(UnsupportedAudioError) as exc_load:
        load_audio(buf.getvalue())
    assert "Permitted range is 8,000 Hz to 192,000 Hz." in str(exc_load.value)


@pytest.mark.parametrize("valid_sr", [8000, 16000, 44100, 48000, 192000])
def test_sample_rate_contract_accepted(valid_sr: int) -> None:
    """Valid boundary and standard rates (8k through 192k) are accepted."""
    resampler = StatefulResampler(orig_sr=valid_sr)
    assert resampler.orig_sr == valid_sr

    session = StreamingSession()
    chunk = _generate_sine_pcm(0.1, sample_rate=valid_sr)
    res = session.ingest_pcm(chunk, sample_rate=valid_sr)
    assert res.samples_ingested > 0


# ==============================================================================
# SECTION D (WebSocket regression): Malformed Frame Resilience
# ==============================================================================


def test_websocket_malformed_frame_emits_audio_decode_error_and_recovers() -> None:
    """Verify WebSocket emits AUDIO_DECODE_ERROR on malformed frame and continues accepting valid audio."""
    with TestClient(app) as client:
        with client.websocket_connect("/api/v1/stream") as ws:
            ws.send_json(
                {"type": "start", "sample_rate": 16000, "channels": 1, "pcm_encoding": "int16"}
            )
            ack = ws.receive_json()
            assert ack["type"] == "session_started"

            # 1. Send invalid odd-length int16 frame (3 bytes)
            ws.send_bytes(b"\x00\x01\x02")

            err = ws.receive_json()
            assert err["type"] == "error"
            assert err["error_code"] == "AUDIO_DECODE_ERROR"
            assert "Odd-length int16 buffer rejected" in err["message"]

            # 2. Connection and session remain alive: send valid 4.0s audio
            c4 = _generate_sine_pcm(4.0, sample_rate=16000, freq=350.0, amplitude=0.5)
            ws.send_bytes(c4)

            # Receive valid window result
            win_msg = ws.receive_json()
            assert win_msg["type"] == "window_result"
            assert win_msg["window_index"] == 0
            assert win_msg["is_eligible"] is True

            ws.send_json({"type": "stop"})
            stop_ack = ws.receive_json()
            assert stop_ack["type"] == "session_stopped"
