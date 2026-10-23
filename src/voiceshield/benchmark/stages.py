"""Isolated pipeline stage latency measurement routines for VoiceShield."""

from __future__ import annotations

import io
import time
from typing import Any

import numpy as np
import soundfile as sf
import torch

from voiceshield.benchmark.schema import PROJECT_TARGET_MS, StageLatency
from voiceshield.config.schema import TemporalPolicyConfig
from voiceshield.dsp.preprocessor import load_audio
from voiceshield.models.aasist_adapter import pad_for_aasist
from voiceshield.models.base import BaseDetector
from voiceshield.streaming.policy import TemporalDecisionPolicy
from voiceshield.streaming.resampler import StatefulResampler
from voiceshield.streaming.vad import EnergyVAD


def _compute_stage_latency(
    stage_name: str,
    device_name: str,
    warmup_times: list[float],
    steady_times: list[float],
    batch_size: int = 1,
    audio_duration_seconds: float = 4.0,
    check_target: bool = False,
) -> StageLatency:
    """Compute statistics from raw warmup and steady-state millisecond latencies."""
    warmup_arr = np.array(warmup_times, dtype=np.float64)
    steady_arr = np.array(steady_times, dtype=np.float64)

    warmup_mean = float(np.mean(warmup_arr)) if len(warmup_arr) > 0 else 0.0

    p50 = float(np.percentile(steady_arr, 50))
    p95 = float(np.percentile(steady_arr, 95))
    p99 = float(np.percentile(steady_arr, 99))
    mean = float(np.mean(steady_arr))
    std = float(np.std(steady_arr))
    min_val = float(np.min(steady_arr))
    max_val = float(np.max(steady_arr))

    # Throughput (executions per second)
    throughput = (1000.0 / mean) if mean > 0 else 0.0

    # Real-Time Factor (RTF = latency_sec / audio_duration_sec)
    rtf = (p50 / 1000.0) / audio_duration_seconds if audio_duration_seconds > 0 else 0.0

    meets_target = (p50 < PROJECT_TARGET_MS) if check_target else None

    return StageLatency(
        stage_name=stage_name,
        device=device_name,
        batch_size=batch_size,
        warmup_iterations=len(warmup_times),
        warmup_latencies_ms=[round(t, 3) for t in warmup_times],
        warmup_mean_ms=round(warmup_mean, 3),
        steady_iterations=len(steady_times),
        steady_latencies_ms=[round(t, 3) for t in steady_times],
        p50_ms=round(p50, 3),
        p95_ms=round(p95, 3),
        p99_ms=round(p99, 3),
        mean_ms=round(mean, 3),
        std_ms=round(std, 3),
        min_ms=round(min_val, 3),
        max_ms=round(max_val, 3),
        throughput_hz=round(throughput, 2),
        real_time_factor=round(rtf, 5),
        project_target_ms=PROJECT_TARGET_MS,
        meets_project_target=meets_target,
    )


def sync_device(device: torch.device) -> None:
    """Synchronize device execution queue for accurate GPU/Metal timing."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def benchmark_audio_decoding_and_resampling(
    iterations: int = 50,
    warmup: int = 5,
    sample_rate_in: int = 48000,
    channels: int = 2,
    duration_seconds: float = 4.0,
    target_sr: int = 16000,
) -> StageLatency:
    """Benchmark stage 1: Audio container decoding, stereo mixing, and polyphase resampling."""
    # Create test WAV buffer in memory
    num_samples = int(sample_rate_in * duration_seconds)
    t = np.linspace(0, duration_seconds, num_samples, endpoint=False, dtype=np.float32)
    sig = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    audio_data = np.column_stack([sig, sig]) if channels == 2 else sig

    buf = io.BytesIO()
    sf.write(buf, audio_data, sample_rate_in, format="WAV", subtype="PCM_16")
    wav_bytes = buf.getvalue()

    warmup_times: list[float] = []
    for _ in range(warmup):
        t0 = time.perf_counter()
        _ = load_audio(io.BytesIO(wav_bytes), target_sr=target_sr)
        warmup_times.append((time.perf_counter() - t0) * 1000.0)

    steady_times: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = load_audio(io.BytesIO(wav_bytes), target_sr=target_sr)
        steady_times.append((time.perf_counter() - t0) * 1000.0)

    return _compute_stage_latency(
        stage_name="1_audio_decoding_and_resampling",
        device_name="cpu",
        warmup_times=warmup_times,
        steady_times=steady_times,
        batch_size=1,
        audio_duration_seconds=duration_seconds,
        check_target=False,
    )


def benchmark_feature_extraction(
    iterations: int = 50,
    warmup: int = 5,
    duration_seconds: float = 4.0,
    sample_rate: int = 16000,
    energy_threshold: float = 0.015,
) -> StageLatency:
    """Benchmark stage 2: VAD speech calculation and deterministic AASIST cyclic padding."""
    num_samples = int(sample_rate * duration_seconds)
    t = np.linspace(0, duration_seconds, num_samples, endpoint=False, dtype=np.float32)
    waveform = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    vad = EnergyVAD(energy_threshold=energy_threshold)

    def _run_features() -> torch.Tensor:
        _ = vad.compute_speech_duration(waveform, sample_rate)
        return pad_for_aasist(waveform, max_len=64600)

    warmup_times: list[float] = []
    for _ in range(warmup):
        t0 = time.perf_counter()
        _ = _run_features()
        warmup_times.append((time.perf_counter() - t0) * 1000.0)

    steady_times: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = _run_features()
        steady_times.append((time.perf_counter() - t0) * 1000.0)

    return _compute_stage_latency(
        stage_name="2_feature_extraction_and_padding",
        device_name="cpu",
        warmup_times=warmup_times,
        steady_times=steady_times,
        batch_size=1,
        audio_duration_seconds=duration_seconds,
        check_target=False,
    )


def benchmark_model_inference(
    detector: BaseDetector,
    device: torch.device,
    batch_size: int = 1,
    iterations: int = 50,
    warmup: int = 5,
    duration_seconds: float = 4.0,
) -> StageLatency:
    """Benchmark stage 3: Forward pass execution on CPU vs MPS with batch size 1."""
    # AASIST-L accepts input tensor of shape (batch_size, 64600)
    tensor_input = torch.randn(batch_size, 64600, dtype=torch.float32, device=device)

    # Resolve underlying model if available or detector.predict_batch
    raw_model: Any = getattr(detector, "model", None)
    if raw_model is not None:
        raw_model.eval()
        raw_model.to(device)

    def _run_inference() -> Any:
        sync_device(device)
        with torch.no_grad():
            if raw_model is not None:
                out = raw_model(tensor_input)
            else:
                out = detector.predict_batch(tensor_input)
        sync_device(device)
        return out

    warmup_times: list[float] = []
    for _ in range(warmup):
        t0 = time.perf_counter()
        _ = _run_inference()
        warmup_times.append((time.perf_counter() - t0) * 1000.0)

    steady_times: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = _run_inference()
        steady_times.append((time.perf_counter() - t0) * 1000.0)

    return _compute_stage_latency(
        stage_name="3_model_inference",
        device_name=device.type,
        warmup_times=warmup_times,
        steady_times=steady_times,
        batch_size=batch_size,
        audio_duration_seconds=duration_seconds,
        check_target=True,  # Primary project target: < 200 ms
    )


def benchmark_calibration_and_temporal_policy(
    iterations: int = 50,
    warmup: int = 5,
    duration_seconds: float = 4.0,
    policy_config: TemporalPolicyConfig | None = None,
    windowing_config: Any = None,
) -> StageLatency:
    """Benchmark stage 4: Calibrated likelihood computation and TemporalDecisionPolicy tracking."""
    policy = TemporalDecisionPolicy(
        policy_config=policy_config or TemporalPolicyConfig(),
        windowing_config=windowing_config,
    )
    mock_score = 0.85

    def _run_policy(idx: int) -> Any:
        return policy.evaluate(
            calibrated_score=mock_score,
            speech_seconds=3.5,
            window_index=idx,
            window_start_sec=float(idx),
            window_end_sec=float(idx + duration_seconds),
        )

    warmup_times: list[float] = []
    for i in range(warmup):
        t0 = time.perf_counter()
        _ = _run_policy(i)
        warmup_times.append((time.perf_counter() - t0) * 1000.0)

    steady_times: list[float] = []
    for i in range(iterations):
        t0 = time.perf_counter()
        _ = _run_policy(warmup + i)
        steady_times.append((time.perf_counter() - t0) * 1000.0)

    return _compute_stage_latency(
        stage_name="4_calibration_and_temporal_policy",
        device_name="cpu",
        warmup_times=warmup_times,
        steady_times=steady_times,
        batch_size=1,
        audio_duration_seconds=duration_seconds,
        check_target=False,
    )


def benchmark_end_to_end_window_processing(
    detector: BaseDetector,
    device: torch.device,
    iterations: int = 50,
    warmup: int = 5,
    duration_seconds: float = 4.0,
    sample_rate: int = 16000,
    energy_threshold: float = 0.015,
    policy_config: TemporalPolicyConfig | None = None,
    windowing_config: Any = None,
) -> StageLatency:
    """Benchmark stage 5: End-to-end processing of an eligible audio window."""
    # Raw duration_seconds stereo PCM input (realistic uncompressed mic or stream input)
    num_samples_in = int(48000 * duration_seconds)
    t = np.linspace(0, duration_seconds, num_samples_in, endpoint=False, dtype=np.float32)
    sig = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    stereo_pcm = np.column_stack([sig, sig]).astype(np.float32)

    vad = EnergyVAD(energy_threshold=energy_threshold)
    policy = TemporalDecisionPolicy(
        policy_config=policy_config or TemporalPolicyConfig(),
        windowing_config=windowing_config,
    )

    raw_model: Any = getattr(detector, "model", None)
    if raw_model is not None:
        raw_model.eval()
        raw_model.to(device)

    def _run_e2e(step: int) -> Any:
        resampler = StatefulResampler(
            orig_sr=48000, target_sr=sample_rate, channels=2, dtype="float32"
        )
        resampled = resampler.process(stereo_pcm)
        speech_sec = vad.compute_speech_duration(resampled, sample_rate)
        padded = pad_for_aasist(resampled, max_len=64600)
        batch = padded.unsqueeze(0).to(device)

        sync_device(device)
        with torch.no_grad():
            if raw_model is not None:
                logits = raw_model(batch)[1]  # Output tuple (feature, logits)
            else:
                out = detector.predict_batch(batch)
                logits = torch.tensor([out[0].raw_logits], device=device)
        sync_device(device)

        spoof_logit = float(logits[0, 0].item())
        bonafide_logit = float(logits[0, 1].item())
        calibrated_score = 1.0 / (1.0 + float(np.exp(-(spoof_logit - bonafide_logit))))
        return policy.evaluate(
            calibrated_score=calibrated_score,
            speech_seconds=speech_sec,
            window_index=step,
            window_start_sec=float(step),
            window_end_sec=float(step + duration_seconds),
        )

    warmup_times: list[float] = []
    for i in range(warmup):
        t0 = time.perf_counter()
        _ = _run_e2e(i)
        warmup_times.append((time.perf_counter() - t0) * 1000.0)

    steady_times: list[float] = []
    for i in range(iterations):
        t0 = time.perf_counter()
        _ = _run_e2e(warmup + i)
        steady_times.append((time.perf_counter() - t0) * 1000.0)

    return _compute_stage_latency(
        stage_name="5_end_to_end_window_processing",
        device_name=device.type,
        warmup_times=warmup_times,
        steady_times=steady_times,
        batch_size=1,
        audio_duration_seconds=duration_seconds,
        check_target=True,  # Primary project target: < 200 ms
    )
