"""Manifest-driven dataset layer, validation, leakage detection, and partitioning."""

from voiceshield.data.io import ManifestIOError, load_manifest, save_manifest
from voiceshield.data.leakage import LeakageReport, check_leakage
from voiceshield.data.schema import (
    MANIFEST_VERSION,
    AttackType,
    LabelType,
    ManifestRecord,
    SplitType,
)
from voiceshield.data.splitter import (
    codec_holdout_split,
    generator_holdout_split,
    speaker_disjoint_split,
)
from voiceshield.data.summary import ManifestSummary, format_summary_tables, summarize_manifest

__all__ = [
    "MANIFEST_VERSION",
    "AttackType",
    "LabelType",
    "LeakageReport",
    "ManifestIOError",
    "ManifestRecord",
    "ManifestSummary",
    "SplitType",
    "check_leakage",
    "codec_holdout_split",
    "format_summary_tables",
    "generator_holdout_split",
    "load_manifest",
    "save_manifest",
    "speaker_disjoint_split",
    "summarize_manifest",
]
