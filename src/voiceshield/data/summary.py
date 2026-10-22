"""Summary table generators for manifest distributions across dimensions."""

from __future__ import annotations

from collections import Counter, defaultdict

from pydantic import BaseModel, Field

from voiceshield.data.schema import ManifestRecord


class DimensionSummary(BaseModel):
    """Counts and total durations for a single dimension."""

    dimension_name: str
    categories: dict[str, int] = Field(description="Sample count per category.")
    durations_seconds: dict[str, float] = Field(
        description="Total duration in seconds per category."
    )


class ManifestSummary(BaseModel):
    """Aggregate breakdown of manifest records across required evaluation dimensions."""

    total_samples: int
    total_duration_hours: float
    by_label: DimensionSummary
    by_split: DimensionSummary
    by_source: DimensionSummary
    by_generator: DimensionSummary
    by_language: DimensionSummary
    by_codec: DimensionSummary
    by_replay: DimensionSummary
    num_unique_speakers: int
    samples_per_speaker_avg: float


def _build_dim_summary(records: list[ManifestRecord], field_name: str) -> DimensionSummary:
    counts: Counter[str] = Counter()
    durations: defaultdict[str, float] = defaultdict(float)

    for rec in records:
        val = getattr(rec, field_name)
        val_str = str(val) if val is not None else "(none)"
        counts[val_str] += 1
        durations[val_str] += rec.duration

    return DimensionSummary(
        dimension_name=field_name,
        categories=dict(sorted(counts.items())),
        durations_seconds={k: round(v, 2) for k, v in sorted(durations.items())},
    )


def summarize_manifest(records: list[ManifestRecord]) -> ManifestSummary:
    """Generate comprehensive summary statistics for a list of manifest records.

    Args:
        records: List of ManifestRecord instances.

    Returns:
        ManifestSummary model with breakdowns.
    """
    total_samples = len(records)
    total_sec = sum(r.duration for r in records)
    unique_speakers = {r.speaker_id for r in records}
    num_speakers = len(unique_speakers)
    avg_per_spk = round(total_samples / num_speakers, 2) if num_speakers > 0 else 0.0

    return ManifestSummary(
        total_samples=total_samples,
        total_duration_hours=round(total_sec / 3600.0, 3),
        by_label=_build_dim_summary(records, "label"),
        by_split=_build_dim_summary(records, "split"),
        by_source=_build_dim_summary(records, "source_dataset"),
        by_generator=_build_dim_summary(records, "generator_family"),
        by_language=_build_dim_summary(records, "language"),
        by_codec=_build_dim_summary(records, "codec_channel_condition"),
        by_replay=_build_dim_summary(records, "replay_condition"),
        num_unique_speakers=num_speakers,
        samples_per_speaker_avg=avg_per_spk,
    )


def format_summary_tables(summary: ManifestSummary) -> str:
    """Render summary as clean, readable text/markdown tables."""
    lines: list[str] = [
        "===========================================================",
        f" VoiceShield Manifest Summary (Total Samples: {summary.total_samples}, {summary.total_duration_hours} hrs)",
        f" Unique Speakers: {summary.num_unique_speakers} (Avg {summary.samples_per_speaker_avg} samples/speaker)",
        "===========================================================",
    ]

    dims = [
        ("Label Distribution", summary.by_label),
        ("Split Breakdown", summary.by_split),
        ("Source Datasets", summary.by_source),
        ("Generator Families", summary.by_generator),
        ("Languages", summary.by_language),
        ("Codecs & Channel Conditions", summary.by_codec),
        ("Replay Conditions", summary.by_replay),
    ]

    for title, dim in dims:
        lines.append(f"\n### {title}")
        lines.append(f"{'Category':<30} | {'Count':<10} | {'Duration (s)':<15}")
        lines.append(f"{'-' * 30}-|-{'-' * 10}-|-{'-' * 15}")
        for cat, count in dim.categories.items():
            dur = dim.durations_seconds.get(cat, 0.0)
            lines.append(f"{cat:<30} | {count:<10} | {dur:<15.2f}")

    return "\n".join(lines)
