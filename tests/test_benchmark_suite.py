"""Unit tests for the MacBook Pro M3 benchmark suite and profiling utilities."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from voiceshield.benchmark.runner import BenchmarkRunner, detect_hardware_environment
from voiceshield.benchmark.schema import BenchmarkSuiteReport, StageLatency
from voiceshield.benchmark.soak import run_streaming_soak_test
from voiceshield.benchmark.stages import (
    benchmark_audio_decoding_and_resampling,
    benchmark_calibration_and_temporal_policy,
    benchmark_end_to_end_window_processing,
    benchmark_feature_extraction,
    benchmark_model_inference,
)
from voiceshield.models.aasist_adapter import (
    AASIST_L_DEFAULT_CONFIG,
    AASISTLDetector,
)
from voiceshield.vendor.aasist import AASISTModel


def test_detect_hardware_environment() -> None:
    """Verify hardware environment detection queries macOS platform accurately."""
    env = detect_hardware_environment()
    assert env.macos_version != ""
    assert env.hardware_model != ""
    assert env.cpu_cores >= 1
    assert "mps" in env.selected_device or "cpu" in env.selected_device
    assert isinstance(env.mps_built, bool)
    assert isinstance(env.mps_available, bool)


def test_stage1_audio_decoding_benchmark() -> None:
    """Verify audio decoding and resampling stage measurement."""
    lat = benchmark_audio_decoding_and_resampling(iterations=5, warmup=2)
    assert isinstance(lat, StageLatency)
    assert lat.stage_name == "1_audio_decoding_and_resampling"
    assert lat.p50_ms > 0.0
    assert lat.throughput_hz > 0.0
    assert lat.real_time_factor > 0.0
    assert len(lat.warmup_latencies_ms) == 2
    assert len(lat.steady_latencies_ms) == 5


def test_stage2_feature_extraction_benchmark() -> None:
    """Verify VAD and AASIST cyclic padding stage measurement."""
    lat = benchmark_feature_extraction(iterations=5, warmup=2)
    assert isinstance(lat, StageLatency)
    assert lat.stage_name == "2_feature_extraction_and_padding"
    assert lat.p50_ms > 0.0
    assert lat.throughput_hz > 0.0


def test_stage3_model_inference_benchmark() -> None:
    """Verify batch size 1 model inference benchmarking on CPU and MPS."""
    mock_raw = AASISTModel(AASIST_L_DEFAULT_CONFIG)
    detector = AASISTLDetector(_instantiated_model=mock_raw)

    # Test CPU
    cpu_lat = benchmark_model_inference(
        detector=detector, device=torch.device("cpu"), batch_size=1, iterations=5, warmup=2
    )
    assert cpu_lat.stage_name == "3_model_inference"
    assert cpu_lat.device == "cpu"
    assert cpu_lat.batch_size == 1
    assert cpu_lat.p50_ms > 0.0
    assert cpu_lat.meets_project_target is not None

    # Test MPS if available
    if torch.backends.mps.is_available():
        mps_lat = benchmark_model_inference(
            detector=detector, device=torch.device("mps"), batch_size=1, iterations=5, warmup=2
        )
        assert mps_lat.device == "mps"
        assert mps_lat.p50_ms > 0.0
        assert mps_lat.meets_project_target is True


def test_stage4_calibration_and_temporal_policy_benchmark() -> None:
    """Verify calibration and temporal policy state update benchmarking."""
    lat = benchmark_calibration_and_temporal_policy(iterations=10, warmup=2)
    assert isinstance(lat, StageLatency)
    assert lat.stage_name == "4_calibration_and_temporal_policy"
    assert lat.p50_ms > 0.0
    # Policy update is extremely fast (microsecond range)
    assert lat.p50_ms < 5.0


def test_stage5_end_to_end_benchmark() -> None:
    """Verify end-to-end eligible window processing benchmark."""
    mock_raw = AASISTModel(AASIST_L_DEFAULT_CONFIG)
    detector = AASISTLDetector(_instantiated_model=mock_raw)
    dev = torch.device("cpu")

    lat = benchmark_end_to_end_window_processing(
        detector=detector, device=dev, iterations=5, warmup=2
    )
    assert lat.stage_name == "5_end_to_end_window_processing"
    assert lat.p50_ms > 0.0
    assert lat.meets_project_target is not None


def test_streaming_soak_test_execution() -> None:
    """Verify soak test runs, tracks memory, and ensures bounded backpressure queue."""
    mock_raw = AASISTModel(AASIST_L_DEFAULT_CONFIG)
    detector = AASISTLDetector(_instantiated_model=mock_raw)
    dev = torch.device("cpu")

    # Short soak test for unit testing (1.5 seconds)
    report = run_streaming_soak_test(
        detector=detector,
        device=dev,
        duration_seconds=1.5,
        sample_interval_seconds=0.5,
        accelerated=True,
    )
    assert report.duration_seconds >= 1.0
    assert report.total_frames_ingested > 0
    assert report.initial_rss_mb > 0.0
    assert report.peak_rss_mb >= report.initial_rss_mb
    assert report.max_backlog_observed <= 1
    assert not report.is_leak_suspected
    assert "PASSED" in report.verdict
    assert len(report.samples) >= 1


def test_benchmark_runner_full_suite_and_reporting(tmp_path: Path) -> None:
    """Verify BenchmarkRunner executes all stages, generates markdown and JSON, and labels mock mode."""
    runner = BenchmarkRunner(
        checkpoint_path=tmp_path / "absent_weights.pth",
        iterations=5,
        warmup=2,
        devices=["cpu"],
        soak_duration_seconds=1.0,
    )

    report = runner.run_suite()
    assert isinstance(report, BenchmarkSuiteReport)
    assert report.provenance.is_mock_weights is True
    assert report.warning_mock is not None
    assert "[MOCK DETECTOR BENCHMARK" in report.warning_mock

    # Check stage results
    assert "audio_decoding" in report.stages_cpu
    assert "model_inference" in report.stages_cpu
    assert "end_to_end" in report.stages_cpu

    # Save reports to temporary directory
    md_file, json_file = runner.save_reports(report, output_dir=tmp_path)
    assert md_file.is_file()
    assert json_file.is_file()

    md_text = md_file.read_text(encoding="utf-8")
    assert "# VoiceShield Benchmark Report: Apple MacBook Pro M3" in md_text
    assert "MOCK WEIGHTS NOTICE" in md_text
    assert "Under 200 ms inference per 4-second window" in md_text

    with open(json_file, encoding="utf-8") as f:
        json_data = json.load(f)
        assert json_data["provenance"]["is_mock_weights"] is True
        assert "audio_decoding" in json_data["stages_cpu"]
