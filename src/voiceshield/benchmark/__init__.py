"""VoiceShield hardware benchmarking and profiling suite for Apple Silicon M3."""

from __future__ import annotations

from voiceshield.benchmark.runner import BenchmarkRunner, detect_hardware_environment
from voiceshield.benchmark.schema import (
    BenchmarkProvenance,
    BenchmarkSuiteReport,
    HardwareEnvironment,
    SoakSample,
    SoakTestReport,
    StageLatency,
)
from voiceshield.benchmark.soak import run_streaming_soak_test

__all__ = [
    "BenchmarkRunner",
    "BenchmarkSuiteReport",
    "BenchmarkProvenance",
    "HardwareEnvironment",
    "StageLatency",
    "SoakSample",
    "SoakTestReport",
    "detect_hardware_environment",
    "run_streaming_soak_test",
]
