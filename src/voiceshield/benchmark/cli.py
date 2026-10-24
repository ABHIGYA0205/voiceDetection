"""CLI entrypoint for VoiceShield hardware benchmarking on MacBook Pro M3."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from voiceshield.benchmark.runner import BenchmarkRunner
from voiceshield.config.loader import ConfigError
from voiceshield.models.registry import CheckpointError, ModelRegistryError


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VoiceShield MacBook Pro M3 Benchmark Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of steady-state iterations per stage.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup iterations recorded separately.",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="cpu,mps",
        help="Comma-separated hardware devices to evaluate (e.g. 'cpu,mps').",
    )
    parser.add_argument(
        "--soak-duration",
        type=float,
        default=600.0,
        help="Streaming soak test duration in seconds (600s = 10 min; set 0 to skip).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/benchmarks",
        help="Directory where markdown and JSON reports will be written.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to explicit VoiceShield YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to AASIST-L checkpoint weights override.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        choices=["clovaai/aasist-l", "custom/aasist-l"],
        help="Optional model identity override ('clovaai/aasist-l' or 'custom/aasist-l').",
    )
    parser.add_argument(
        "--mock-model",
        action="store_true",
        default=False,
        help="Explicitly run benchmark using mock architecture weights (identity mock/aasist-l).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        device_list = [d.strip().lower() for d in args.devices.split(",") if d.strip()]

        print("=" * 80)
        print("VoiceShield Hardware Benchmark Suite: Apple MacBook Pro M3")
        print("=" * 80)
        print("Target: < 200 ms inference per 4-second window (RTF < 0.050).")
        print("Notice: Defined as an engineering target, not an achieved claim until measured.\n")

        runner = BenchmarkRunner(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            model_id=args.model_id,
            mock_model=args.mock_model,
            iterations=args.iterations,
            warmup=args.warmup,
            devices=device_list,
            soak_duration_seconds=args.soak_duration,
        )

        report = runner.run_suite()
        md_path, json_path = runner.save_reports(report, output_dir=args.output_dir)

        print("\n" + "=" * 80)
        print("Benchmark Execution Complete!")
        print("=" * 80)
        print(f"Report Markdown: {md_path}")
        print(f"Report JSON:     {json_path}")
        print("\nSummary of Results:")
        print("-" * 80)

        for dev_name, stages in [("CPU", report.stages_cpu), ("MPS", report.stages_mps)]:
            if stages:
                print(f"\n[{dev_name} Device Performance]:")
                for _name, st in stages.items():
                    target_str = ""
                    if st.meets_project_target is not None:
                        target_str = (
                            " -> [MET TARGET]" if st.meets_project_target else " -> [FAILED TARGET]"
                        )
                    print(
                        f"  - {st.stage_name:<35}: p50={st.p50_ms:>6.2f} ms | p95={st.p95_ms:>6.2f} ms | "
                        f"RTF={st.real_time_factor:>7.4f} | Throughput={st.throughput_hz:>6.1f} win/s{target_str}"
                    )

        if report.speedup_mps_vs_cpu:
            print("\n[MPS vs. CPU Speedup Factors]:")
            for k, sp in report.speedup_mps_vs_cpu.items():
                print(f"  - {k:<35}: {sp:.2f}x speedup on Apple Silicon Metal")

        if report.soak_test:
            soak = report.soak_test
            print("\n[10-Minute Streaming Soak Diagnostics]:")
            print(
                f"  - Duration: {soak.duration_seconds:.1f}s ({soak.duration_seconds / 60:.1f} min)"
            )
            print(
                f"  - Initial RSS: {soak.initial_rss_mb:.1f} MB | Peak: {soak.peak_rss_mb:.1f} MB | Final: {soak.final_rss_mb:.1f} MB"
            )
            print(
                f"  - Net Growth: {soak.net_growth_mb:+.2f} MB ({soak.growth_rate_mb_per_hour:.2f} MB/hr)"
            )
            print(f"  - Max Queue Backlog: {soak.max_backlog_observed} (Queue bound: 1)")
            print(f"  - Verdict: {soak.verdict}")

        if report.warning_mock:
            print("\n" + "!" * 80)
            print(f"WARNING: {report.warning_mock}")
            print("!" * 80)

        print()
        return 0
    except (CheckpointError, ConfigError, ModelRegistryError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
