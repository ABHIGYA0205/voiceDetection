"""Pydantic schemas and data contracts for the VoiceShield benchmark suite."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

PROJECT_TARGET_MS: float = 200.0  # Defined target: < 200 ms inference per 4.0-second window
WINDOW_DURATION_SECONDS: float = 4.0


class StageLatency(BaseModel):
    """Statistical latency and throughput metrics for an isolated pipeline stage."""

    stage_name: str = Field(description="Name of the pipeline stage being benchmarked.")
    device: str = Field(description="Execution hardware device ('cpu' or 'mps').")
    batch_size: int = Field(default=1, description="Batch size evaluated (1 for real-time).")
    warmup_iterations: int = Field(description="Number of warmup iterations recorded.")
    warmup_latencies_ms: list[float] = Field(
        default_factory=list, description="Per-iteration warmup latencies in milliseconds."
    )
    warmup_mean_ms: float = Field(description="Average warmup latency in milliseconds.")
    steady_iterations: int = Field(description="Number of steady-state iterations evaluated.")
    steady_latencies_ms: list[float] = Field(
        default_factory=list, description="Raw steady-state latencies in milliseconds."
    )
    p50_ms: float = Field(description="50th percentile (median) latency in milliseconds.")
    p95_ms: float = Field(description="95th percentile latency in milliseconds.")
    p99_ms: float = Field(description="99th percentile latency in milliseconds.")
    mean_ms: float = Field(description="Arithmetic mean steady latency in milliseconds.")
    std_ms: float = Field(description="Standard deviation of steady latency in milliseconds.")
    min_ms: float = Field(description="Minimum latency observed in steady state.")
    max_ms: float = Field(description="Maximum latency observed in steady state.")
    throughput_hz: float = Field(
        description="Throughput in executions or windows per second (1000.0 / mean_ms)."
    )
    real_time_factor: float = Field(
        description="Real-Time Factor: processing_time_sec / window_duration_sec (< 1.0 is faster than real-time)."
    )
    project_target_ms: float = Field(
        default=PROJECT_TARGET_MS,
        description="Project latency target in milliseconds (< 200 ms per 4-second window).",
    )
    meets_project_target: bool | None = Field(
        default=None,
        description="Whether this stage satisfies the project target (< 200 ms).",
    )


class SoakSample(BaseModel):
    """Periodic memory and backlog health snapshot during the streaming soak test."""

    elapsed_seconds: float = Field(description="Elapsed time since soak start in seconds.")
    frames_ingested: int = Field(description="Cumulative PCM frames ingested.")
    windows_evaluated: int = Field(description="Cumulative 4.0s windows evaluated.")
    ring_buffer_samples: int = Field(description="Current samples stored in ring buffer.")
    ring_buffer_utilization_pct: float = Field(
        description="Ring buffer capacity utilization percentage."
    )
    pending_backlog_count: int = Field(
        description="Current number of pending windows awaiting inference."
    )
    process_rss_mb: float = Field(description="Current resident set size (RSS) memory in MB.")
    mps_allocated_mb: float | None = Field(
        default=None, description="Current Metal MPS allocated memory in MB if applicable."
    )


class SoakTestReport(BaseModel):
    """Results and health diagnostics from the streaming soak test."""

    duration_seconds: float = Field(description="Total duration of soak test in seconds.")
    total_frames_ingested: int = Field(description="Total frames pushed through the stream.")
    total_windows_evaluated: int = Field(description="Total 4.0s windows evaluated by model.")
    total_windows_dropped: int = Field(
        description="Total stale pending windows dropped due to backpressure."
    )
    initial_rss_mb: float = Field(description="Process RSS at soak start in MB.")
    peak_rss_mb: float = Field(description="Peak process RSS recorded in MB.")
    final_rss_mb: float = Field(description="Process RSS at soak end in MB.")
    net_growth_mb: float = Field(description="Net memory growth (final - initial) in MB.")
    growth_rate_mb_per_hour: float = Field(
        description="Extrapolated linear memory growth rate in MB/hour."
    )
    is_leak_suspected: bool = Field(
        description="Whether monotonic or unbounded memory growth was detected."
    )
    max_backlog_observed: int = Field(
        description="Maximum pending window queue depth observed during soak."
    )
    samples: list[SoakSample] = Field(
        default_factory=list, description="Periodic timeseries health samples."
    )
    verdict: str = Field(description="Human-readable soak test pass/fail assessment.")


class HardwareEnvironment(BaseModel):
    """Detailed hardware and platform configuration for the benchmark host."""

    macos_version: str = Field(description="macOS release version string.")
    hardware_model: str = Field(description="Apple hardware model identifier (e.g. Mac15,3).")
    cpu_brand: str = Field(description="CPU model and architecture (e.g. Apple M3).")
    cpu_cores: int = Field(description="Total physical and logical CPU core count.")
    python_version: str = Field(description="Active Python interpreter version.")
    torch_version: str = Field(description="PyTorch build version.")
    mps_built: bool = Field(description="Whether PyTorch binary includes MPS backend.")
    mps_available: bool = Field(description="Whether Apple Silicon MPS is hardware-available.")
    selected_device: str = Field(description="Active benchmark device resolution.")


class BenchmarkProvenance(BaseModel):
    """Reproducibility hashes and model provenance for the benchmark run."""

    timestamp_utc: str = Field(description="ISO 8601 UTC execution timestamp.")
    git_commit: str | None = Field(default=None, description="Git commit hash.")
    git_dirty: bool = Field(default=False, description="Whether git tree had uncommitted changes.")
    config_hash: str = Field(description="Deterministic SHA-256 hash of configuration.")
    checkpoint_path: str | None = Field(default=None, description="Path to checkpoint weights.")
    checkpoint_sha256: str | None = Field(
        default=None, description="SHA-256 checksum of checkpoint weights."
    )
    model_id: str = Field(description="Detector model architecture identifier.")
    is_mock_weights: bool = Field(
        description="Whether benchmark executed using mock architecture weights."
    )


class BenchmarkSuiteReport(BaseModel):
    """Consolidated benchmark report containing all stages, device comparisons, and soak test."""

    provenance: BenchmarkProvenance
    environment: HardwareEnvironment
    target_description: str = Field(
        default=(
            "Project Target: Under 200 ms inference per 4-second window (RTF < 0.050). "
            "Defined as an engineering target, not an achieved claim until measured."
        )
    )
    warning_mock: str | None = Field(
        default=None,
        description="Prominent warning banner if mock weights were evaluated.",
    )
    stages_cpu: dict[str, StageLatency] = Field(
        default_factory=dict, description="Stage latencies on CPU."
    )
    stages_mps: dict[str, StageLatency] | None = Field(
        default=None, description="Stage latencies on MPS (Apple Silicon GPU) if available."
    )
    speedup_mps_vs_cpu: dict[str, float] = Field(
        default_factory=dict, description="MPS speedup factor relative to CPU (p50_cpu / p50_mps)."
    )
    soak_test: SoakTestReport | None = Field(
        default=None, description="Streaming soak test results if executed."
    )
    profiling_summary: dict[str, Any] = Field(
        default_factory=dict, description="Percentage breakdown of time spent per pipeline stage."
    )
