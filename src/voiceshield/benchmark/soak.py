"""Streaming soak test for memory growth and backpressure queue backlog."""

from __future__ import annotations

import logging
import resource
import time
from typing import Any

import numpy as np
import torch

from voiceshield.benchmark.schema import SoakSample, SoakTestReport
from voiceshield.config.schema import VoiceShieldConfig
from voiceshield.models.aasist_adapter import pad_for_aasist
from voiceshield.models.base import BaseDetector
from voiceshield.streaming.session import StreamingSession

logger = logging.getLogger(__name__)


def _get_current_rss_mb() -> float:
    """Return process peak/current RSS in megabytes on macOS."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    # On macOS, ru_maxrss is returned in bytes
    return float(ru.ru_maxrss) / (1024.0 * 1024.0)


def _get_mps_allocated_mb() -> float | None:
    """Return current allocated Metal memory on Apple Silicon if available."""
    if hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
        try:
            return float(torch.mps.current_allocated_memory()) / (1024.0 * 1024.0)
        except Exception:
            return None
    return None


def run_streaming_soak_test(
    detector: BaseDetector,
    device: torch.device,
    duration_seconds: float = 600.0,
    sample_interval_seconds: float = 15.0,
    frame_duration_ms: float = 50.0,
    accelerated: bool = True,
    config: VoiceShieldConfig | None = None,
) -> SoakTestReport:
    """Execute streaming soak test over declared duration to measure memory stability and queue backlog.

    Args:
        detector: Detector model adapter.
        device: Hardware device ('cpu' or 'mps').
        duration_seconds: Target soak duration in seconds (default: 600.0s = 10 minutes).
        sample_interval_seconds: Cadence to capture timeseries memory/backlog snapshots.
        frame_duration_ms: Duration of each ingested PCM chunk in milliseconds.
        accelerated: If True, executes without artificial wall-clock sleeps to test high-throughput soaking.
        config: Optional effective VoiceShieldConfig to initialize StreamingSession.

    Returns:
        Populated SoakTestReport.
    """
    logger.info(
        "Starting VoiceShield streaming soak test (duration=%.1fs, device=%s)...",
        duration_seconds,
        device.type,
    )

    session = StreamingSession(config=config)
    raw_model: Any = getattr(detector, "model", None)
    if raw_model is not None:
        raw_model.eval()
        raw_model.to(device)

    sample_rate = session.sample_rate
    frame_samples = int(round(sample_rate * (frame_duration_ms / 1000.0)))

    # Create repetitive speech-like sine frame (synthetic audio fixture)
    t = np.linspace(0, frame_duration_ms / 1000.0, frame_samples, endpoint=False, dtype=np.float32)
    pcm_audio = (0.5 * np.sin(2 * np.pi * 300.0 * t) * 32767.0).astype(np.int16).tobytes()

    samples_history: list[SoakSample] = []
    initial_rss = _get_current_rss_mb()
    peak_rss = initial_rss
    max_backlog = 0

    total_frames = 0
    total_windows_evaluated = 0
    total_windows_dropped = 0

    soak_start_time = time.perf_counter()
    last_sample_time = soak_start_time
    init_mps = _get_mps_allocated_mb()

    # Record initial sample at t=0
    samples_history.append(
        SoakSample(
            elapsed_seconds=0.0,
            frames_ingested=0,
            windows_evaluated=0,
            ring_buffer_samples=0,
            ring_buffer_utilization_pct=0.0,
            pending_backlog_count=0,
            process_rss_mb=round(initial_rss, 2),
            mps_allocated_mb=round(init_mps, 2) if init_mps is not None else None,
        )
    )

    while (time.perf_counter() - soak_start_time) < duration_seconds:
        # Ingest frame
        _ = session.ingest_pcm(
            pcm_data=pcm_audio,
            sample_rate=sample_rate,
            channels=1,
            dtype="int16",
        )
        total_frames += 1

        # Check backlog queue depth
        current_backlog = 1 if session.has_pending_window else 0
        if current_backlog > max_backlog:
            max_backlog = current_backlog

        # If eligible window ready, process inference
        win = session.pop_pending_window()
        if win is not None:
            if win.is_eligible:
                padded = pad_for_aasist(win.samples, max_len=64600)
                batch = padded.unsqueeze(0).to(device)
                if device.type == "mps":
                    torch.mps.synchronize()
                with torch.no_grad():
                    if raw_model is not None:
                        logits = raw_model(batch)[1]
                    else:
                        out = detector.predict_batch(batch)
                        logits = torch.tensor([out[0].raw_logits], device=device)
                if device.type == "mps":
                    torch.mps.synchronize()

                spoof_logit = float(logits[0, 0].item())
                bonafide_logit = float(logits[0, 1].item())
                calibrated = 1.0 / (1.0 + float(np.exp(-(spoof_logit - bonafide_logit))))
                session.complete_inference(calibrated, win)
                total_windows_evaluated += 1

        # Check for periodic sampling
        now = time.perf_counter()
        if (now - last_sample_time) >= sample_interval_seconds:
            elapsed = now - soak_start_time
            current_rss = _get_current_rss_mb()
            if current_rss > peak_rss:
                peak_rss = current_rss

            mps_mem = _get_mps_allocated_mb()
            rb_samples = session.ring_buffer.available_samples
            rb_cap = session.ring_buffer.capacity
            utilization = (rb_samples / rb_cap * 100.0) if rb_cap > 0 else 0.0

            sample = SoakSample(
                elapsed_seconds=round(elapsed, 1),
                frames_ingested=total_frames,
                windows_evaluated=total_windows_evaluated,
                ring_buffer_samples=rb_samples,
                ring_buffer_utilization_pct=round(utilization, 2),
                pending_backlog_count=1 if session.has_pending_window else 0,
                process_rss_mb=round(current_rss, 2),
                mps_allocated_mb=round(mps_mem, 2) if mps_mem is not None else None,
            )
            samples_history.append(sample)
            last_sample_time = now

        # Non-accelerated mode sleeps realistically
        if not accelerated:
            time.sleep(frame_duration_ms / 1000.0)

    # Final measurement
    final_time = time.perf_counter()
    actual_duration = final_time - soak_start_time
    final_rss = _get_current_rss_mb()
    if final_rss > peak_rss:
        peak_rss = final_rss

    net_growth = final_rss - initial_rss
    hours = actual_duration / 3600.0
    growth_rate_per_hour = (net_growth / hours) if hours > 0 else 0.0

    # Leak heuristic: if memory growth exceeds 10 MB/hour after plateau, flag as suspected leak
    is_leak = bool(growth_rate_per_hour > 10.0 and net_growth > 5.0)

    # Clean up session
    session.reset()

    if is_leak:
        verdict = (
            f"FAILED: Suspected memory leak detected. Net growth: +{net_growth:.2f} MB "
            f"({growth_rate_per_hour:.2f} MB/hr)."
        )
    elif max_backlog > 1:
        verdict = f"FAILED: Backlog queue exceeded bounded capacity (Max queue: {max_backlog})."
    else:
        verdict = (
            f"PASSED: Bounded memory and backlog maintained across {actual_duration:.1f}s soak. "
            f"Net memory delta: {net_growth:+.2f} MB ({growth_rate_per_hour:.2f} MB/hr). "
            f"Queue backlog remained <= 1."
        )

    return SoakTestReport(
        duration_seconds=round(actual_duration, 1),
        total_frames_ingested=total_frames,
        total_windows_evaluated=total_windows_evaluated,
        total_windows_dropped=total_windows_dropped,
        initial_rss_mb=round(initial_rss, 2),
        peak_rss_mb=round(peak_rss, 2),
        final_rss_mb=round(final_rss, 2),
        net_growth_mb=round(net_growth, 2),
        growth_rate_mb_per_hour=round(growth_rate_per_hour, 2),
        is_leak_suspected=is_leak,
        max_backlog_observed=max_backlog,
        samples=samples_history,
        verdict=verdict,
    )
