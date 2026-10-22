"""Run record serialization and persistence under reports/runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from voiceshield.config.schema import VoiceShieldConfig
from voiceshield.reproducibility.metadata import RunMetadata, collect_run_metadata

DEFAULT_RUNS_DIR = Path("reports/runs")


class RunRecord(BaseModel):
    """Complete JSON-serializable experiment or inference run record."""

    run_id: str = Field(description="Unique run identifier.")
    metadata: RunMetadata = Field(description="System and provenance metadata.")
    config: dict[str, Any] = Field(description="Snapshot of configuration used in this run.")
    metrics: dict[str, Any] = Field(
        default_factory=dict, description="Evaluation or execution metrics."
    )
    notes: str | None = Field(default=None, description="Optional researcher/operator notes.")


def create_run_record(
    config: VoiceShieldConfig,
    metrics: dict[str, Any] | None = None,
    notes: str | None = None,
    manifest_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> RunRecord:
    """Create a populated RunRecord from config and optional metrics.

    Args:
        config: Loaded VoiceShield configuration.
        metrics: Optional dictionary of evaluation or runtime metrics.
        notes: Optional descriptive notes.
        manifest_path: Optional path to data manifest.
        checkpoint_path: Optional path to checkpoint.

    Returns:
        Populated RunRecord instance.
    """
    meta = collect_run_metadata(
        config=config,
        manifest_path=manifest_path,
        checkpoint_path=checkpoint_path,
    )
    return RunRecord(
        run_id=meta.run_id,
        metadata=meta,
        config=config.to_dict(),
        metrics=metrics or {},
        notes=notes,
    )


def save_run_record(
    record: RunRecord,
    output_dir: str | Path = DEFAULT_RUNS_DIR,
) -> Path:
    """Save run record as formatted JSON under the target directory.

    Args:
        record: RunRecord to persist.
        output_dir: Output directory (defaults to reports/runs).

    Returns:
        Path to the saved JSON file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_file = out_dir / f"{record.run_id}.json"

    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(record.model_dump(), f, indent=2)

    return target_file


def load_run_record(path: str | Path) -> RunRecord:
    """Load and validate a persisted RunRecord from a JSON file.

    Args:
        path: Path to run record JSON file.

    Returns:
        Validated RunRecord instance.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return RunRecord.model_validate(data)
