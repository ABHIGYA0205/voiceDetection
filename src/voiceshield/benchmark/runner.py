"""Main benchmark runner orchestrating stage evaluations, CPU vs MPS comparisons, and report generation."""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from voiceshield.benchmark.schema import (
    BenchmarkProvenance,
    BenchmarkSuiteReport,
    HardwareEnvironment,
    SoakTestReport,
    StageLatency,
)
from voiceshield.benchmark.soak import run_streaming_soak_test
from voiceshield.benchmark.stages import (
    benchmark_audio_decoding_and_resampling,
    benchmark_end_to_end_window_processing,
    benchmark_feature_extraction,
    benchmark_model_inference,
    benchmark_score_smoothing_and_temporal_policy,
)
from voiceshield.config.loader import load_config
from voiceshield.config.overrides import apply_config_overrides
from voiceshield.config.schema import VoiceShieldConfig
from voiceshield.core.device import resolve_device_status
from voiceshield.models.aasist_adapter import (
    AASIST_L_DEFAULT_CONFIG,
    AASISTLDetector,
)
from voiceshield.reproducibility.metadata import _get_git_commit, _is_git_dirty
from voiceshield.vendor.aasist import AASISTModel

logger = logging.getLogger(__name__)


def _query_sysctl(key: str) -> str:
    """Safely query macOS sysctl string."""
    try:
        res = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, check=False, timeout=5.0
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "Unknown"


def detect_hardware_environment() -> HardwareEnvironment:
    """Inspect and collect macOS hardware and runtime platform information."""
    mps_built = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_built())
    mps_available = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    dev_status = resolve_device_status()

    hw_model = _query_sysctl("hw.model")
    if hw_model == "Unknown":
        hw_model = platform.machine()

    cpu_brand = _query_sysctl("machdep.cpu.brand_string")
    if cpu_brand == "Unknown":
        cpu_brand = platform.processor() or "Apple Silicon"

    return HardwareEnvironment(
        macos_version=platform.mac_ver()[0] or platform.platform(),
        hardware_model=hw_model,
        cpu_brand=cpu_brand,
        cpu_cores=os.cpu_count() or 1,
        python_version=sys.version.split()[0],
        torch_version=str(torch.__version__),
        mps_built=mps_built,
        mps_available=mps_available,
        selected_device=dev_status.device_name,
    )


class BenchmarkRunner:
    """Orchestrates comprehensive latency, throughput, and soak benchmarks."""

    def __init__(
        self,
        config: VoiceShieldConfig | None = None,
        config_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        model_id: str | None = None,
        mock_model: bool = False,
        iterations: int = 50,
        warmup: int = 5,
        devices: list[str] | None = None,
        soak_duration_seconds: float = 0.0,
        soak_ingestion_mode: str = "accelerated",
    ) -> None:
        base = config or load_config(config_path)
        if checkpoint_path is not None or model_id is not None:
            self.config = apply_config_overrides(
                base,
                model_id=model_id,
                checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
            )
        else:
            self.config = base
        self.mock_model = mock_model
        self.checkpoint_path = Path(self.config.model.checkpoint_path)
        self.iterations = iterations
        self.warmup = warmup
        self.devices = devices or ["cpu", "mps"]
        self.soak_duration = soak_duration_seconds
        if soak_ingestion_mode not in {"accelerated", "real_time_paced"}:
            raise ValueError("soak_ingestion_mode must be 'accelerated' or 'real_time_paced'.")
        self.soak_ingestion_mode = soak_ingestion_mode

    def _resolve_detector(self) -> AASISTLDetector:
        """Instantiate detector with real checkpoint or explicit mock model."""
        if self.mock_model:
            logger.warning(
                "Running in explicit MOCK weights mode for interface benchmarking (model_id=mock/aasist-l)."
            )
            mock_raw = AASISTModel(AASIST_L_DEFAULT_CONFIG)  # type: ignore[no-untyped-call]
            return AASISTLDetector(
                _instantiated_model=mock_raw,
                model_id="mock/aasist-l",
                threshold=self.config.temporal_policy.threshold_high,
                preferred_device=self.config.model.device_preferred,
                fallback_to_cpu=self.config.model.fallback_to_cpu,
            )

        logger.info(
            "Initializing AASIST-L detector (model_id=%s) from: %s",
            self.config.model.model_id,
            self.checkpoint_path,
        )
        return AASISTLDetector(
            checkpoint_path=self.checkpoint_path,
            model_id=self.config.model.model_id,
            threshold=self.config.temporal_policy.threshold_high,
            preferred_device=self.config.model.device_preferred,
            fallback_to_cpu=self.config.model.fallback_to_cpu,
        )

    def run_suite(self) -> BenchmarkSuiteReport:
        """Execute full benchmark suite across all stages, devices, and soak test."""
        env = detect_hardware_environment()
        detector = self._resolve_detector()
        prov = detector.provenance
        is_mock = prov.is_mock

        mock_warning: str | None = None
        if is_mock:
            mock_warning = (
                "[MOCK DETECTOR BENCHMARK - NOT FOR PITCH OR PRODUCTION CLAIMS] "
                "Explicit mock model weights were evaluated (mock/aasist-l). Latencies reflect architectural "
                "forward pass computation of the AASIST network, but must NOT be cited as verified detection claims."
            )

        provenance = BenchmarkProvenance(
            timestamp_utc=datetime.now(UTC).isoformat(),
            git_commit=_get_git_commit(),
            git_dirty=_is_git_dirty(),
            config_hash=self.config.compute_hash(),
            checkpoint_path=str(self.checkpoint_path) if not is_mock else None,
            checkpoint_sha256=prov.checkpoint_sha256,
            model_id=prov.model_id,
            is_mock_weights=is_mock,
            provenance_status=prov.provenance_status,
            checkpoint_verified=prov.checkpoint_verified,
            expected_checkpoint_sha256=prov.expected_checkpoint_sha256,
            upstream_commit=prov.upstream_commit,
            parameter_count=prov.parameter_count,
            model_provenance=prov.model_dump(),
        )

        sample_rate = self.config.audio.sample_rate
        window_seconds = self.config.windowing.window_seconds
        energy_threshold = self.config.windowing.energy_threshold
        policy_cfg = self.config.temporal_policy
        windowing_cfg = self.config.windowing

        # 1. Run CPU Stages
        stages_cpu: dict[str, StageLatency] = {}
        cpu_dev = torch.device("cpu")

        logger.info(
            "Executing CPU stage benchmarks (warmup=%d, steady=%d)...", self.warmup, self.iterations
        )
        stages_cpu["audio_decoding"] = benchmark_audio_decoding_and_resampling(
            iterations=self.iterations,
            warmup=self.warmup,
            duration_seconds=window_seconds,
            target_sr=sample_rate,
        )
        stages_cpu["feature_extraction"] = benchmark_feature_extraction(
            iterations=self.iterations,
            warmup=self.warmup,
            duration_seconds=window_seconds,
            sample_rate=sample_rate,
            energy_threshold=energy_threshold,
        )
        stages_cpu["model_inference"] = benchmark_model_inference(
            detector=detector,
            device=cpu_dev,
            batch_size=1,
            iterations=self.iterations,
            warmup=self.warmup,
            duration_seconds=window_seconds,
        )
        stages_cpu["score_smoothing_and_policy"] = benchmark_score_smoothing_and_temporal_policy(
            iterations=self.iterations,
            warmup=self.warmup,
            duration_seconds=window_seconds,
            policy_config=policy_cfg,
            windowing_config=windowing_cfg,
        )
        stages_cpu["end_to_end"] = benchmark_end_to_end_window_processing(
            detector=detector,
            device=cpu_dev,
            iterations=self.iterations,
            warmup=self.warmup,
            duration_seconds=window_seconds,
            sample_rate=sample_rate,
            energy_threshold=energy_threshold,
            policy_config=policy_cfg,
            windowing_config=windowing_cfg,
        )

        # 2. Run MPS Stages (if available and requested)
        stages_mps: dict[str, StageLatency] | None = None
        speedups: dict[str, float] = {}

        if "mps" in self.devices and env.mps_available:
            mps_dev = torch.device("mps")
            logger.info("Executing MPS stage benchmarks on Apple Silicon Metal...")
            stages_mps = {}

            # Decoding and feature extraction remain CPU/numpy bound
            stages_mps["audio_decoding"] = stages_cpu["audio_decoding"]
            stages_mps["feature_extraction"] = stages_cpu["feature_extraction"]
            stages_mps["model_inference"] = benchmark_model_inference(
                detector=detector,
                device=mps_dev,
                batch_size=1,
                iterations=self.iterations,
                warmup=self.warmup,
                duration_seconds=window_seconds,
            )
            stages_mps["score_smoothing_and_policy"] = stages_cpu["score_smoothing_and_policy"]
            stages_mps["end_to_end"] = benchmark_end_to_end_window_processing(
                detector=detector,
                device=mps_dev,
                iterations=self.iterations,
                warmup=self.warmup,
                duration_seconds=window_seconds,
                sample_rate=sample_rate,
                energy_threshold=energy_threshold,
                policy_config=policy_cfg,
                windowing_config=windowing_cfg,
            )

            # Compute speedup factors (CPU p50 / MPS p50)
            if stages_mps["model_inference"].p50_ms > 0:
                speedups["model_inference"] = round(
                    stages_cpu["model_inference"].p50_ms / stages_mps["model_inference"].p50_ms, 2
                )
            if stages_mps["end_to_end"].p50_ms > 0:
                speedups["end_to_end"] = round(
                    stages_cpu["end_to_end"].p50_ms / stages_mps["end_to_end"].p50_ms, 2
                )

        # 3. Profiling summary: percentage of end-to-end time per stage
        primary_stages = stages_mps if stages_mps is not None else stages_cpu
        primary_e2e_ms = primary_stages["end_to_end"].p50_ms
        profiling: dict[str, Any] = {
            "primary_device": "mps" if stages_mps is not None else "cpu",
            "end_to_end_p50_ms": primary_e2e_ms,
            "stage_percentages": {
                "audio_decoding_pct": round(
                    (primary_stages["audio_decoding"].p50_ms / primary_e2e_ms) * 100.0, 1
                ),
                "feature_extraction_pct": round(
                    (primary_stages["feature_extraction"].p50_ms / primary_e2e_ms) * 100.0, 1
                ),
                "model_inference_pct": round(
                    (primary_stages["model_inference"].p50_ms / primary_e2e_ms) * 100.0, 1
                ),
                "score_smoothing_and_policy_pct": round(
                    (primary_stages["score_smoothing_and_policy"].p50_ms / primary_e2e_ms) * 100.0,
                    1,
                ),
            },
            "inference_meets_target": primary_stages["model_inference"].meets_project_target,
            "end_to_end_meets_target": primary_stages["end_to_end"].meets_project_target,
        }

        # 4. Optional Streaming Soak Test
        soak_report: SoakTestReport | None = None
        if self.soak_duration > 0:
            target_dev = (
                torch.device("mps") if ("mps" in self.devices and env.mps_available) else cpu_dev
            )
            logger.info(
                "Executing streaming soak test for %.1f seconds on %s...",
                self.soak_duration,
                target_dev.type,
            )
            sample_interval = max(1.0, min(15.0, self.soak_duration / 20.0))
            soak_report = run_streaming_soak_test(
                detector=detector,
                device=target_dev,
                duration_seconds=self.soak_duration,
                sample_interval_seconds=sample_interval,
                accelerated=self.soak_ingestion_mode == "accelerated",
                config=self.config,
            )

        return BenchmarkSuiteReport(
            provenance=provenance,
            environment=env,
            warning_mock=mock_warning,
            stages_cpu=stages_cpu,
            stages_mps=stages_mps,
            speedup_mps_vs_cpu=speedups,
            soak_test=soak_report,
            profiling_summary=profiling,
        )

    def generate_markdown_report(self, report: BenchmarkSuiteReport) -> str:
        """Render markdown artifact report matching VoiceShield documentation standards."""
        lines: list[str] = []

        lines.append("# VoiceShield Benchmark Report: Apple MacBook Pro M3")
        lines.append("")
        lines.append(f"**Execution Timestamp (UTC)**: `{report.provenance.timestamp_utc}`  ")
        lines.append(
            f"**Hardware**: `{report.environment.cpu_brand}` (`{report.environment.hardware_model}`)  "
        )
        lines.append(
            f"**macOS Version**: `{report.environment.macos_version}` | **CPU Cores**: `{report.environment.cpu_cores}`  "
        )
        lines.append(
            f"**Python**: `{report.environment.python_version}` | **PyTorch**: `{report.environment.torch_version}`  "
        )
        lines.append(
            f"**Apple MPS Metal Acceleration**: `Built={report.environment.mps_built}, Available={report.environment.mps_available}`  "
        )
        lines.append("")

        lines.append("> [!IMPORTANT]")
        lines.append(f"> **Engineering Target**: {report.target_description}")
        lines.append("")

        if report.warning_mock:
            lines.append("> [!WARNING]")
            lines.append(f"> **MOCK WEIGHTS NOTICE**: {report.warning_mock}")
            lines.append("")

        lines.append("## 1. Provenance & Hash Verification")
        lines.append("")
        lines.append("| Provenance Field | Value |")
        lines.append("| :--- | :--- |")
        lines.append(f"| **Model Identifier** | `{report.provenance.model_id}` |")
        lines.append(
            f"| **Provenance Status** | `{report.provenance.provenance_status or 'N/A'}` |"
        )
        lines.append(f"| **Checkpoint Verified** | `{report.provenance.checkpoint_verified}` |")
        lines.append(f"| **Effective Canonical Config Hash** | `{report.provenance.config_hash}` |")
        lines.append(
            f"| **Checkpoint Path** | `{report.provenance.checkpoint_path or 'N/A (mock)'}` |"
        )
        lines.append(
            f"| **Checkpoint SHA-256** | `{report.provenance.checkpoint_sha256 or 'N/A (mock)'}` |"
        )
        if report.provenance.expected_checkpoint_sha256:
            lines.append(
                f"| **Expected Registry SHA-256** | `{report.provenance.expected_checkpoint_sha256}` |"
            )
        if report.provenance.upstream_commit:
            lines.append(f"| **Upstream Commit** | `{report.provenance.upstream_commit}` |")
        if report.provenance.parameter_count:
            lines.append(f"| **Parameter Count** | `{report.provenance.parameter_count}` |")
        lines.append(
            f"| **Git Commit** | `{report.provenance.git_commit or 'Uncommitted repo'}` (Dirty: `{report.provenance.git_dirty}`) |"
        )
        lines.append(
            f"| **Weights Status** | `{'MOCK WEIGHTS' if report.provenance.is_mock_weights else 'REAL OFFICIAL AASIST-L'}` |"
        )
        lines.append(
            f"| **Configured Device Preference** | `{self.config.model.device_preferred}` |"
        )
        lines.append(
            f"| **Configured Decision Threshold** | `{self.config.temporal_policy.threshold_high}` |"
        )
        lines.append(
            f"| **Evaluated Devices Matrix** | `{', '.join(self.devices)}` (suite parameter, not in config) |"
        )
        lines.append(
            f"| **Benchmark Iterations** | `{self.iterations}` (warmup: `{self.warmup}`) |"
        )
        lines.append(f"| **Streaming Soak Duration** | `{self.soak_duration}s` |")
        lines.append("")

        lines.append("## 2. Pipeline Stage Latency Breakdown (Batch Size 1)")
        lines.append("")
        lines.append(
            f"All metrics measure steady-state execution over {self.config.windowing.window_seconds:.1f}-second audio windows, with initial warm-up recorded separately."
        )
        lines.append("")
        lines.append(
            "| Pipeline Stage | Device | Warm-up Mean | Steady p50 (Median) | Steady p95 | Steady p99 | Throughput | RTF | Target (<200ms) |"
        )
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

        for _key, st in report.stages_cpu.items():
            tgt = "N/A"
            if st.meets_project_target is not None:
                tgt = "✅ MET" if st.meets_project_target else "❌ FAILED"
            lines.append(
                f"| `{st.stage_name}` | `CPU` | {st.warmup_mean_ms:.1f} ms | **{st.p50_ms:.1f} ms** | {st.p95_ms:.1f} ms | {st.p99_ms:.1f} ms | {st.throughput_hz:.1f} win/s | {st.real_time_factor:.4f} | {tgt} |"
            )

        if report.stages_mps:
            for key in ["model_inference", "end_to_end"]:
                if key in report.stages_mps:
                    st = report.stages_mps[key]
                    tgt = "✅ MET" if st.meets_project_target else "❌ FAILED"
                    lines.append(
                        f"| `{st.stage_name}` | `MPS` | {st.warmup_mean_ms:.1f} ms | **{st.p50_ms:.1f} ms** | {st.p95_ms:.1f} ms | {st.p99_ms:.1f} ms | {st.throughput_hz:.1f} win/s | {st.real_time_factor:.4f} | {tgt} |"
                    )

        lines.append("")

        if report.speedup_mps_vs_cpu:
            lines.append("## 3. Hardware Acceleration Comparison: Apple Silicon MPS vs. CPU")
            lines.append("")
            lines.append("| Workload | CPU p50 | MPS p50 | Metal Speedup |")
            lines.append("| :--- | :---: | :---: | :---: |")
            for k, sp in report.speedup_mps_vs_cpu.items():
                c_p50 = report.stages_cpu[k].p50_ms
                m_p50 = report.stages_mps[k].p50_ms if report.stages_mps else 0.0
                lines.append(f"| `{k}` | {c_p50:.1f} ms | {m_p50:.1f} ms | **{sp:.2f}x faster** |")
            lines.append("")

        lines.append("## 4. Profiling & Bottleneck Analysis")
        lines.append("")
        prof = report.profiling_summary
        lines.append(f"- **Primary Execution Device**: `{prof.get('primary_device')}`")
        lines.append(f"- **End-to-End Window Latency (p50)**: `{prof.get('end_to_end_p50_ms')} ms`")
        cpu_rtf = (
            f"{report.stages_cpu['end_to_end'].real_time_factor}"
            if "end_to_end" in report.stages_cpu
            else "N/A"
        )
        mps_rtf = (
            f"{report.stages_mps['end_to_end'].real_time_factor}"
            if (report.stages_mps and "end_to_end" in report.stages_mps)
            else "N/A"
        )
        lines.append(f"- **Real-Time Factor (RTF)**: `{cpu_rtf}` (CPU) / `{mps_rtf}` (MPS)")
        lines.append("")
        lines.append("### Stage Proportion Breakdown:")
        pcts = prof.get("stage_percentages", {})
        for s_name, s_pct in pcts.items():
            lines.append(f"- **`{s_name}`**: {s_pct}%")
        lines.append("")

        inf_met = prof.get("inference_meets_target")
        e2e_met = prof.get("end_to_end_meets_target")
        lines.append("### Project Target Assessment (< 200 ms per 4-second window):")
        lines.append(
            f"- **Model Inference**: {'✅ Target Achieved' if inf_met else '❌ Target Missed'}"
        )
        lines.append(
            f"- **End-to-End Processing**: {'✅ Target Achieved' if e2e_met else '❌ Target Missed'}"
        )
        lines.append("")

        if report.soak_test:
            soak = report.soak_test
            lines.append("## 5. Streaming Stress Soak Diagnostics")
            lines.append("")
            lines.append(
                "- **Ingestion Mode**: `accelerated` (frames are ingested without real-time pacing "
                "and may represent more audio than elapsed wall time)"
                if soak.ingestion_mode == "accelerated"
                else "- **Ingestion Mode**: `real_time_paced` (frames are paced according to represented frame duration)"
            )
            lines.append(f"- **Configured Duration**: `{soak.configured_duration_seconds:.1f}s`")
            lines.append(
                f"- **Measured Wall-Clock Duration**: `{soak.measured_duration_seconds:.1f}s` "
                f"({soak.measured_duration_seconds / 60:.1f} min)"
            )
            lines.append(f"- **Frame Duration**: `{soak.frame_duration_ms:.1f} ms`")
            lines.append(f"- **Total Frames Ingested**: `{soak.total_frames_ingested}`")
            lines.append(f"- **Total Windows Evaluated**: `{soak.total_windows_evaluated}`")
            lines.append(f"- **Initial RSS Memory**: `{soak.initial_rss_mb:.2f} MB`")
            lines.append(f"- **Peak RSS Memory**: `{soak.peak_rss_mb:.2f} MB`")
            lines.append(f"- **Final RSS Memory**: `{soak.final_rss_mb:.2f} MB`")
            lines.append(f"- **Net Memory Delta**: `{soak.net_growth_mb:+.2f} MB`")
            lines.append(
                f"- **Extrapolated Growth Rate**: `{soak.growth_rate_mb_per_hour:.2f} MB/hr`"
            )
            lines.append(
                f"- **Maximum Backlog Observed**: `{soak.max_backlog_observed}` (Queue limit: 1)"
            )
            lines.append(f"- **Verdict**: `{soak.verdict}`")
            lines.append("")

        return "\n".join(lines)

    def save_reports(
        self, report: BenchmarkSuiteReport, output_dir: str | Path = "reports/benchmarks"
    ) -> tuple[Path, Path]:
        """Save markdown and JSON benchmark reports to output directory."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        md_path = out_path / "m3_benchmark_report.md"
        json_path = out_path / "m3_benchmark_report.json"

        md_content = self.generate_markdown_report(report)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))

        logger.info("Saved benchmark reports to %s and %s", md_path, json_path)
        return md_path, json_path
