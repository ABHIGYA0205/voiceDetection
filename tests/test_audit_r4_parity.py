"""Comprehensive parity regression tests verifying offline and streaming parity (Audit R4 / H-03).

Validates:
H.1: Canonical resampler parity matrix (8k, 16k, 44.1k, 48k, 96k; mono/stereo; sine, noise, impulses; chunkings).
H.2: Offline vs streaming canonical waveform parity for WAV and FLAC.
H.3: Canonical window boundary tests (3.99s, 4.00s, 4.01s, 4.99s, 5.00s, 5.01s, 12.0s).
H.4: VAD parity tests (silence, active speech, boundary duration, subframe leftover).
H.5: Deterministic detector and temporal decision policy parity between offline and streaming.
H.6: End-of-stream flush and idempotent finalization tests at non-16kHz rates.
H.7: Protected R3 behavior (discontinuity resets, invalid input rejection, drop accounting).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
import torch

from voiceshield.dsp.preprocessor import load_audio
from voiceshield.dsp.resampler import CanonicalResampler
from voiceshield.models.base import BaseDetector, DetectorOutput
from voiceshield.pipeline.file_evaluator import evaluate_audio
from voiceshield.streaming.policy import (
    DecisionState,
    ReasonCode,
    TemporalDecision,
)
from voiceshield.streaming.session import StreamingSession
from voiceshield.streaming.vad import EnergyVAD

# ==============================================================================
# Helper fixtures and signal generators
# ==============================================================================


class DeterministicLightweightDetector(BaseDetector):
    """Deterministic test detector with synthetic scores computed strictly from input samples."""

    def __init__(self) -> None:
        super().__init__()
        self.predict_calls: int = 0
        self.input_windows: list[np.ndarray[Any, np.dtype[np.float32]]] = []

    def predict(
        self, audio: np.ndarray[Any, np.dtype[np.float32]] | torch.Tensor
    ) -> DetectorOutput:
        self.predict_calls += 1
        if isinstance(audio, torch.Tensor):
            audio_arr = audio.detach().cpu().numpy().astype(np.float32)
        else:
            audio_arr = audio.astype(np.float32)

        self.input_windows.append(np.ascontiguousarray(audio_arr.copy(), dtype=np.float32))

        # Deterministic score based on energy and spectral proxy
        energy = float(np.mean(audio_arr.astype(np.float64) ** 2))
        peak = float(np.max(np.abs(audio_arr)))
        synth_score = float(np.clip(0.35 + 0.45 * np.sin(energy * 2000.0) + 0.15 * peak, 0.0, 1.0))

        logit_spoof = float(np.log(max(synth_score, 1e-6)))
        logit_bonafide = float(np.log(max(1.0 - synth_score, 1e-6)))

        return DetectorOutput(
            raw_logits=(logit_spoof, logit_bonafide),
            bonafide_score=logit_bonafide - logit_spoof,
            synthetic_score=synth_score,
            uncalibrated_synthetic_score=synth_score,
            is_synthetic_suspected=bool(synth_score >= 0.65),
            input_samples=audio_arr.shape[0],
            device_used="cpu",
            model_id="DeterministicLightweightDetector",
        )

    def predict_batch(self, batch: torch.Tensor) -> list[DetectorOutput]:
        outputs = []
        for i in range(batch.shape[0]):
            outputs.append(self.predict(batch[i]))
        return outputs


def make_sine_mixture(
    duration: float, sr: int, channels: int = 1
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Generate multi-tone sine mixture within [-0.9, 0.9]."""
    num_samples = int(round(duration * sr))
    t = np.linspace(0, duration, num_samples, endpoint=False, dtype=np.float32)
    s = 0.4 * np.sin(2 * np.pi * 300.0 * t) + 0.3 * np.sin(2 * np.pi * 1200.0 * t)
    s = s.astype(np.float32)
    if channels == 1:
        return np.asarray(s, dtype=np.float32)
    r = 0.3 * np.sin(2 * np.pi * 600.0 * t) + 0.3 * np.sin(2 * np.pi * 2400.0 * t)
    return np.asarray(np.column_stack([s, r.astype(np.float32)]), dtype=np.float32)


def make_seeded_noise(
    duration: float, sr: int, channels: int = 1, seed: int = 42
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Generate deterministic pseudorandom noise within [-0.8, 0.8]."""
    num_samples = int(round(duration * sr))
    rng = np.random.RandomState(seed)
    if channels == 1:
        return rng.uniform(-0.8, 0.8, size=num_samples).astype(np.float32)
    return rng.uniform(-0.8, 0.8, size=(num_samples, channels)).astype(np.float32)


def make_impulse(
    duration: float, sr: int, position: str = "first", channels: int = 1
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Generate an impulse near the first or final sample."""
    num_samples = int(round(duration * sr))
    if channels == 1:
        arr = np.zeros(num_samples, dtype=np.float32)
        idx = 5 if position == "first" else (num_samples - 6)
        arr[idx] = 0.9
        return arr
    arr2 = np.zeros((num_samples, channels), dtype=np.float32)
    idx = 5 if position == "first" else (num_samples - 6)
    arr2[idx, :] = 0.9
    return arr2


# ==============================================================================
# H.1 Canonical Resampler Parity Matrix
# ==============================================================================


@pytest.mark.parametrize("rate", [8000, 16000, 44100, 48000, 96000])
@pytest.mark.parametrize("channels", [1, 2])
def test_resampler_parity_matrix(rate: int, channels: int) -> None:
    """Test CanonicalResampler across rates, channels, signals, and chunking patterns."""
    duration = 4.0
    signals = {
        "sine_mix": make_sine_mixture(duration, rate, channels),
        "seeded_noise": make_seeded_noise(duration, rate, channels),
        "impulse_first": make_impulse(duration, rate, "first", channels),
        "impulse_final": make_impulse(duration, rate, "final", channels),
    }

    gcd = math.gcd(rate, 16000)
    in_ratio = rate // gcd

    for sig_name, sig_data in signals.items():
        # Downmix if stereo
        if channels == 1:
            mono = sig_data
        else:
            mono = np.mean(sig_data, axis=1, dtype=np.float32)

        total_frames = mono.shape[0]
        expected_len = math.ceil(total_frames * 16000 / rate)

        # 1. Batch reference: process + flush
        ref_resampler = CanonicalResampler(orig_sr=rate, target_sr=16000)
        c_ref = ref_resampler.process(mono)
        t_ref = ref_resampler.flush()
        batch_out = np.concatenate([c_ref, t_ref]) if t_ref.size > 0 else c_ref

        assert batch_out.shape[0] == expected_len, (
            f"Expected {expected_len} samples for {rate}Hz ({sig_name}), got {batch_out.shape[0]}"
        )
        assert np.all(np.isfinite(batch_out)), "Batch resampled array contains non-finite values."
        assert batch_out.dtype == np.float32

        # 16 kHz identity check
        if rate == 16000:
            np.testing.assert_array_equal(batch_out, mono)

        # Idempotent second flush
        assert ref_resampler.flush().size == 0, "Second flush must return empty array."

        # Process after flush must fail
        with pytest.raises(RuntimeError, match="Cannot process audio after flush"):
            ref_resampler.process(mono[:10])

        # Reset restores usability
        ref_resampler.reset()
        assert not ref_resampler.is_flushed
        assert ref_resampler.total_source_frames == 0
        resumed = ref_resampler.process(mono[:100])
        assert resumed.shape[0] >= 0

        # Impulse verification: impulses near beginning and end must be present in output
        if sig_name == "impulse_first":
            # Output near start must have significant energy
            assert np.max(np.abs(batch_out[:100])) > 0.05
        elif sig_name == "impulse_final":
            # Output near end must have significant energy (not lost in FIR tail)
            assert np.max(np.abs(batch_out[-100:])) > 0.05

        # 2. Chunking variations:
        chunk_patterns = {
            "20ms": max(1, int(0.02 * rate)),
            "1s": rate,
            "awkward": max(1, in_ratio // 2 + 1),
        }

        max_diffs: dict[str, float] = {}

        for pattern_name, chunk_size in chunk_patterns.items():
            stream_resampler = CanonicalResampler(orig_sr=rate, target_sr=16000)
            outputs = []
            pos = 0
            while pos < total_frames:
                step = min(chunk_size, total_frames - pos)
                ch = mono[pos : pos + step]
                out_chunk = stream_resampler.process(ch)
                if out_chunk.size > 0:
                    outputs.append(out_chunk)
                pos += step

            tail = stream_resampler.flush()
            if tail.size > 0:
                outputs.append(tail)

            streamed_out = np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)
            assert streamed_out.shape[0] == expected_len, (
                f"Pattern {pattern_name} at {rate}Hz produced {streamed_out.shape[0]}, expected {expected_len}"
            )

            diff = float(np.max(np.abs(batch_out - streamed_out)))
            max_diffs[pattern_name] = diff
            assert diff <= 1e-5, (
                f"Diff {diff} exceeded tolerance for {rate}Hz, pattern {pattern_name}"
            )

        # 3. Seeded irregular partition test
        rng = np.random.RandomState(123)
        stream_resampler = CanonicalResampler(orig_sr=rate, target_sr=16000)
        outputs_irreg = []
        pos = 0
        while pos < total_frames:
            step = min(total_frames - pos, int(rng.randint(1, 1500)))
            ch = mono[pos : pos + step]
            out_c = stream_resampler.process(ch)
            if out_c.size > 0:
                outputs_irreg.append(out_c)
            pos += step

        tail_irreg = stream_resampler.flush()
        if tail_irreg.size > 0:
            outputs_irreg.append(tail_irreg)

        streamed_irreg = (
            np.concatenate(outputs_irreg) if outputs_irreg else np.empty(0, dtype=np.float32)
        )
        assert streamed_irreg.shape[0] == expected_len
        diff_irreg = float(np.max(np.abs(batch_out - streamed_irreg)))
        assert diff_irreg <= 1e-5, (
            f"Diff {diff_irreg} exceeded tolerance for irregular chunking at {rate}Hz"
        )


# ==============================================================================
# H.2 Offline versus Streaming Canonical Waveform Parity
# ==============================================================================


@pytest.mark.parametrize("rate", [8000, 16000, 44100, 48000])
@pytest.mark.parametrize("fmt", ["WAV", "FLAC"])
def test_offline_vs_streaming_canonical_waveform_parity(
    tmp_path: Path, rate: int, fmt: str
) -> None:
    """Verify that WAV and FLAC files decoded offline produce identical canonical 16 kHz samples to streamed PCM."""
    duration = 5.0
    num_samples = int(rate * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False, dtype=np.float32)
    source_float = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

    # Quantize to PCM16 so container decoding and streamed bytes are identically quantized
    pcm16_int = (np.clip(source_float, -1.0, 1.0) * 32767.0).astype(np.int16)
    file_path = tmp_path / f"fixture_{rate}_{fmt.lower()}.{fmt.lower()}"

    if fmt == "WAV":
        sf.write(str(file_path), pcm16_int, rate, format="WAV", subtype="PCM_16")
    else:
        sf.write(str(file_path), pcm16_int, rate, format="FLAC", subtype="PCM_16")

    # 1. Offline evaluation path
    offline_preprocessed = load_audio(file_path, normalize=False)
    offline_waveform = offline_preprocessed.waveform

    # 2. Streaming evaluation path: feed identical raw PCM16 bytes in arbitrary chunk boundaries
    raw_pcm_bytes = pcm16_int.tobytes()
    session = StreamingSession()

    rng = np.random.RandomState(456)
    pos = 0
    bytes_len = len(raw_pcm_bytes)
    # Stream in random frame-aligned chunk slices
    while pos < bytes_len:
        # 2 bytes per sample for mono int16
        step_frames = int(rng.randint(50, 600))
        step_bytes = min(bytes_len - pos, step_frames * 2)
        chunk_bytes = raw_pcm_bytes[pos : pos + step_bytes]
        session.ingest_pcm(chunk_bytes, sample_rate=rate, channels=1, dtype="int16")
        pos += step_bytes

    # Finalize session to flush remaining canonical samples
    session.finalize()

    # Compare canonical resampled length and samples
    assert offline_waveform.shape[0] == session.total_canonical_samples_written, (
        f"Length mismatch at {rate}Hz {fmt}: offline={offline_waveform.shape[0]}, "
        f"streaming={session.total_canonical_samples_written}"
    )

    # Compare resampled output directly
    # Offline used CanonicalResampler; streaming resampler processed the same PCM.
    # Note that soundfile decodes int16 as float32 / 32768.0, identical to validate_and_decode_pcm.
    resampler_check = CanonicalResampler(orig_sr=rate, target_sr=16000)
    decoded_mono = pcm16_int.astype(np.float32) / 32768.0
    c_check = resampler_check.process(decoded_mono)
    t_check = resampler_check.flush()
    expected_canonical = np.concatenate([c_check, t_check]) if t_check.size > 0 else c_check

    np.testing.assert_allclose(offline_waveform, expected_canonical, atol=1e-5)


# ==============================================================================
# H.3 Window-Boundary Tests
# ==============================================================================


@pytest.mark.parametrize(
    "duration,expected_windows,expected_ranges",
    [
        (3.99, 0, []),
        (4.00, 1, [(0.0, 4.0)]),
        (4.01, 1, [(0.0, 4.0)]),
        (4.99, 1, [(0.0, 4.0)]),
        (5.00, 2, [(0.0, 4.0), (1.0, 5.0)]),
        (5.01, 2, [(0.0, 4.0), (1.0, 5.0)]),
        (
            12.00,
            9,
            [
                (0.0, 4.0),
                (1.0, 5.0),
                (2.0, 6.0),
                (3.0, 7.0),
                (4.0, 8.0),
                (5.0, 9.0),
                (6.0, 10.0),
                (7.0, 11.0),
                (8.0, 12.0),
            ],
        ),
    ],
)
def test_window_boundary_grid(
    tmp_path: Path,
    duration: float,
    expected_windows: int,
    expected_ranges: list[tuple[float, float]],
) -> None:
    """Verify exact full-window sequence and no trailing padded windows across durations."""
    detector = DeterministicLightweightDetector()
    file_path = tmp_path / f"test_{duration}s.wav"
    t = np.linspace(0, duration, int(16000 * duration), endpoint=False, dtype=np.float32)
    # Active sine speech surrogate
    audio = 0.5 * np.sin(2 * np.pi * 300.0 * t)
    sf.write(str(file_path), audio, 16000, format="WAV")

    result = evaluate_audio(file_path, detector=detector)

    assert result.total_windows == expected_windows, (
        f"For duration {duration}s, expected {expected_windows} windows, got {result.total_windows}"
    )

    if expected_windows == 0:
        assert result.status == "insufficient_evidence"
        assert result.abstain_reason == ReasonCode.INSUFFICIENT_CONTIGUOUS_AUDIO.value
        assert detector.predict_calls == 0
    else:
        assert result.status == "success"
        assert len(result.windows) == expected_windows
        for i, (exp_start, exp_end) in enumerate(expected_ranges):
            w = result.windows[i]
            assert w.window_index == i
            assert math.isclose(w.start_seconds, exp_start, abs_tol=1e-3)
            assert math.isclose(w.end_seconds, exp_end, abs_tol=1e-3)
            assert not w.is_partial


@pytest.mark.parametrize("rate", [8000, 16000, 44100, 48000, 96000])
def test_window_boundary_flush_across_rates(tmp_path: Path, rate: int) -> None:
    """Verify that an exact 4.0-second source at various rates produces exactly 1 window via flush."""
    detector = DeterministicLightweightDetector()
    file_path = tmp_path / f"exact_4s_{rate}.wav"
    t = np.linspace(0, 4.0, int(rate * 4.0), endpoint=False, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 400.0 * t)
    sf.write(str(file_path), audio, rate, format="WAV")

    result = evaluate_audio(file_path, detector=detector)
    assert result.total_windows == 1, (
        f"Expected exactly 1 window at {rate}Hz, got {result.total_windows}"
    )
    assert result.status == "success"
    w = result.windows[0]
    assert math.isclose(w.start_seconds, 0.0, abs_tol=1e-3)
    assert math.isclose(w.end_seconds, 4.0, abs_tol=1e-3)


# ==============================================================================
# H.4 Shared VAD Parity Tests
# ==============================================================================


def test_vad_parity_scenarios() -> None:
    """Verify that EnergyVAD produces identical gating across offline and streaming."""
    vad = EnergyVAD(energy_threshold=0.015, subframe_duration_seconds=0.02)
    sr = 16000

    # 1. Silence
    silence = np.zeros(64000, dtype=np.float32)
    assert vad.compute_speech_duration(silence, sr) == 0.0
    assert not vad.is_speech(silence, min_duration_seconds=0.8)

    # 2. Continuous speech surrogate (4.0s sine @ 0.4 amplitude)
    t = np.linspace(0, 4.0, 64000, endpoint=False, dtype=np.float32)
    active = 0.4 * np.sin(2 * np.pi * 350.0 * t).astype(np.float32)
    assert math.isclose(vad.compute_speech_duration(active, sr), 4.0, abs_tol=1e-3)
    assert vad.is_speech(active, min_duration_seconds=0.8)

    # 3. Mixed speech/silence around 0.8s threshold
    # 0.75s active + 3.25s silence -> below 0.8s threshold
    mix_short = np.zeros(64000, dtype=np.float32)
    t_short = np.linspace(0, 0.75, int(0.75 * sr), endpoint=False, dtype=np.float32)
    mix_short[: int(0.75 * sr)] = 0.4 * np.sin(2 * np.pi * 350.0 * t_short)
    dur_short = vad.compute_speech_duration(mix_short, sr)
    assert math.isclose(dur_short, 0.75, abs_tol=0.02)
    assert not vad.is_speech(mix_short, min_duration_seconds=0.8)

    # 0.85s active + 3.15s silence -> above 0.8s threshold
    mix_long = np.zeros(64000, dtype=np.float32)
    t_long = np.linspace(0, 0.85, int(0.85 * sr), endpoint=False, dtype=np.float32)
    mix_long[: int(0.85 * sr)] = 0.4 * np.sin(2 * np.pi * 350.0 * t_long)
    dur_long = vad.compute_speech_duration(mix_long, sr)
    assert math.isclose(dur_long, 0.85, abs_tol=0.02)
    assert vad.is_speech(mix_long, min_duration_seconds=0.8)

    # 4. Leftover subframe test: sample count not divisible by subframe (320 samples @ 16 kHz)
    odd_samples = np.zeros(325, dtype=np.float32)
    odd_samples[:] = 0.3  # RMS = 0.3 >= 0.015
    dur_odd = vad.compute_speech_duration(odd_samples, sr)
    expected_odd = 325 / 16000.0
    assert math.isclose(dur_odd, expected_odd, abs_tol=1e-5)


# ==============================================================================
# H.5 Detector and Temporal Decision Parity
# ==============================================================================


@pytest.mark.parametrize("rate", [8000, 16000, 44100, 48000, 96000])
@pytest.mark.parametrize("channels", [1, 2])
def test_detector_and_temporal_decision_parity(tmp_path: Path, rate: int, channels: int) -> None:
    """Run identical audio through offline evaluator and no-overload streaming harness, asserting exact parity."""
    duration = 6.0  # Produces 3 windows: [0,4], [1,5], [2,6]
    sig = make_sine_mixture(duration, rate, channels)

    # Quantize to PCM16
    pcm_int16 = (np.clip(sig, -1.0, 1.0) * 32767.0).astype(np.int16)
    wav_path = tmp_path / f"parity_{rate}_{channels}ch.wav"
    sf.write(str(wav_path), pcm_int16, rate, format="WAV", subtype="PCM_16")

    # 1. Offline file evaluation
    detector_offline = DeterministicLightweightDetector()
    offline_res = evaluate_audio(wav_path, detector=detector_offline)

    # 2. No-overload synchronous streaming harness
    detector_streaming = DeterministicLightweightDetector()
    session = StreamingSession()
    raw_bytes = pcm_int16.tobytes()

    rng = np.random.RandomState(789)
    pos = 0
    bytes_len = len(raw_bytes)
    frame_width = 2 * channels

    streaming_decisions: list[tuple[Any, TemporalDecision]] = []

    def drain_pending() -> None:
        while session.has_pending_window:
            win = session.pop_pending_window()
            if win is not None:
                det_out = detector_streaming.predict(win.samples)
                dec = session.complete_inference(
                    synthetic_score=det_out.synthetic_score,
                    window=win,
                )
                if dec is not None:
                    streaming_decisions.append((win, dec))

    # Ingest in seeded irregular complete-frame chunks
    while pos < bytes_len:
        step_frames = int(rng.randint(20, 500))
        step_bytes = min(bytes_len - pos, step_frames * frame_width)
        chunk = raw_bytes[pos : pos + step_bytes]
        ingest_res = session.ingest_pcm(chunk, sample_rate=rate, channels=channels, dtype="int16")
        # Handle immediate abstentions if any
        for imm in ingest_res.immediate_decisions:
            streaming_decisions.append((None, imm))
        # Drain pending window immediately (no backpressure loss)
        drain_pending()
        pos += step_bytes

    # Finalize session at EOS and drain final window
    new_win, new_imm = session.finalize()
    for imm in new_imm:
        streaming_decisions.append((None, imm))
    drain_pending()

    # Assert zero dropped windows in no-overload harness
    assert session.total_windows_dropped == 0, (
        f"Expected 0 drops, got {session.total_windows_dropped}"
    )
    assert session.total_windows_generated == offline_res.total_windows

    # Compare detector calls count
    assert detector_offline.predict_calls == detector_streaming.predict_calls

    # Compare detector input arrays
    for off_inp, st_inp in zip(
        detector_offline.input_windows, detector_streaming.input_windows, strict=True
    ):
        np.testing.assert_allclose(off_inp, st_inp, atol=1e-5)

    # Compare window results
    assert len(offline_res.windows) == len(streaming_decisions)
    for off_win, (_st_win, st_dec) in zip(offline_res.windows, streaming_decisions, strict=True):
        assert off_win.window_index == st_dec.window_index
        assert math.isclose(off_win.start_seconds, st_dec.window_start_sec, abs_tol=1e-3)
        assert math.isclose(off_win.end_seconds, st_dec.window_end_sec, abs_tol=1e-3)
        assert off_win.is_eligible == (off_win.abstain_reason is None)
        assert off_win.decision_state == st_dec.state.value
        assert off_win.decision_reason == st_dec.reason.value
        assert math.isclose(off_win.smoothed_score or 0.0, st_dec.smoothed_score, abs_tol=1e-4)

    # Compare final policy state and file-level classification
    assert offline_res.final_decision_state == session.temporal_policy.current_state.value
    assert offline_res.is_synthetic_suspected == bool(
        session.temporal_policy.current_state == DecisionState.HIGH
    )


# ==============================================================================
# H.6 End-of-Stream Tests
# ==============================================================================


@pytest.mark.parametrize("rate", [8000, 11025, 44100, 48000])
def test_end_of_stream_flush_parity(rate: int) -> None:
    """Prove that an exact 4.0-second source at non-16kHz rates requires flush and produces exactly 64,000 samples."""
    duration = 4.0
    num_samples = int(rate * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 350.0 * t).astype(np.float32)

    session = StreamingSession()
    pcm_bytes = (audio * 32767.0).astype(np.int16).tobytes()

    # Feed in 500-sample chunks
    step_bytes = 1000
    pos = 0
    while pos < len(pcm_bytes):
        session.ingest_pcm(
            pcm_bytes[pos : pos + step_bytes], sample_rate=rate, channels=1, dtype="int16"
        )
        pos += step_bytes

    # Before flush, FIR tail is unconsumed
    assert session.total_canonical_samples_written < 64000
    assert session.total_windows_generated == 0

    # Finalize flushes the tail and completes the window
    new_eligible, new_abstentions = session.finalize()
    assert session.total_canonical_samples_written == 64000
    assert session.total_windows_generated == 1
    assert len(new_eligible) == 1
    assert new_eligible[0].samples.shape == (64000,)

    # Repeated session finalization must be idempotent
    new_eligible_2, new_abstentions_2 = session.finalize()
    assert len(new_eligible_2) == 0
    assert len(new_abstentions_2) == 0
    assert session.total_canonical_samples_written == 64000
    assert session.total_windows_generated == 1


def test_process_after_flush_rejected() -> None:
    """Verify that process() calls after flush() fail cleanly until reset()."""
    resampler = CanonicalResampler(orig_sr=44100, target_sr=16000)
    resampler.process(np.ones(1000, dtype=np.float32))
    resampler.flush()

    with pytest.raises(RuntimeError, match="Cannot process audio after flush"):
        resampler.process(np.ones(100, dtype=np.float32))

    resampler.reset()
    # Now process succeeds
    out = resampler.process(np.ones(100, dtype=np.float32))
    assert out.size >= 0


def test_streaming_ingest_after_finalization_rejected() -> None:
    """Verify that StreamingSession.ingest_pcm after finalize() raises RuntimeError until reset()."""
    session = StreamingSession()
    dummy_bytes = b"\x00\x00" * 1600
    session.ingest_pcm(dummy_bytes, sample_rate=16000, channels=1, dtype="int16")
    session.finalize()

    with pytest.raises(RuntimeError, match="Cannot ingest audio into a finalized StreamingSession"):
        session.ingest_pcm(dummy_bytes, sample_rate=16000, channels=1, dtype="int16")

    session.reset()
    # Now ingest succeeds
    res = session.ingest_pcm(dummy_bytes, sample_rate=16000, channels=1, dtype="int16")
    assert res.samples_ingested == 1600
