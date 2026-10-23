"""I/O handlers for loading and saving dataset manifests in CSV and JSONL formats."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from voiceshield.data.schema import ManifestRecord


class ManifestIOError(Exception):
    """Raised when reading, parsing, or writing manifest files fails."""


def load_manifest(path: str | Path) -> list[ManifestRecord]:
    """Load and validate manifest records from a .csv or .jsonl file.

    Args:
        path: Path to manifest file.

    Returns:
        List of validated ManifestRecord objects.

    Raises:
        ManifestIOError: If file not found, format unsupported, or records fail validation.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise ManifestIOError(f"Manifest file not found: {file_path}")

    suffix = file_path.suffix.lower()
    records: list[ManifestRecord] = []
    errors: list[str] = []

    if suffix == ".jsonl":
        with open(file_path, encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                clean_line = line.strip()
                if not clean_line:
                    continue
                try:
                    data = json.loads(clean_line)
                    records.append(ManifestRecord.model_validate(data))
                except json.JSONDecodeError as exc:
                    errors.append(f"Line {line_idx}: Invalid JSON syntax ({exc})")
                except ValidationError as exc:
                    reasons = "; ".join(f"{e['loc']}: {e['msg']}" for e in exc.errors())
                    errors.append(f"Line {line_idx}: Schema validation failed: {reasons}")
    elif suffix == ".csv":
        with open(file_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ManifestIOError(f"CSV manifest {file_path} is empty or missing headers.")

            for row_idx, row in enumerate(reader, start=2):  # row 1 is header
                parsed_row: dict[str, Any] = {}
                for k, v in row.items():
                    if v is None or v == "" or v == "null" or v == "None":
                        parsed_row[k] = None
                    else:
                        parsed_row[k] = v

                # Convert numeric fields
                try:
                    if parsed_row.get("original_sample_rate") is not None:
                        parsed_row["original_sample_rate"] = int(parsed_row["original_sample_rate"])
                    if parsed_row.get("duration") is not None:
                        parsed_row["duration"] = float(parsed_row["duration"])
                    records.append(ManifestRecord.model_validate(parsed_row))
                except (ValueError, TypeError) as exc:
                    errors.append(f"Row {row_idx}: Type conversion error: {exc}")
                except ValidationError as exc:
                    reasons = "; ".join(f"{e['loc']}: {e['msg']}" for e in exc.errors())
                    errors.append(f"Row {row_idx}: Schema validation failed: {reasons}")
    else:
        raise ManifestIOError(f"Unsupported manifest format: {suffix}. Must be .csv or .jsonl.")

    if errors:
        summary = "\n  - ".join(errors[:15])
        more = f"\n  ... and {len(errors) - 15} more errors" if len(errors) > 15 else ""
        raise ManifestIOError(
            f"Validation failed for manifest {file_path} with {len(errors)} error(s):\n  - {summary}{more}"
        )

    # Validate uniqueness of sample_id values
    seen_sids: set[str] = set()
    dup_sids: list[str] = []
    for r in records:
        if r.sample_id in seen_sids and r.sample_id not in dup_sids:
            dup_sids.append(r.sample_id)
        seen_sids.add(r.sample_id)

    if dup_sids:
        raise ManifestIOError(
            f"Duplicate sample_id found in manifest {file_path}: {dup_sids[:5]} (total {len(dup_sids)} duplicate(s))."
        )

    return records


def save_manifest(records: list[ManifestRecord], path: str | Path) -> None:
    """Save a list of ManifestRecord items to a .csv or .jsonl file.

    Args:
        records: List of ManifestRecord instances.
        path: Destination path.

    Raises:
        ManifestIOError: If destination format is unsupported.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    suffix = dest.suffix.lower()

    if suffix == ".jsonl":
        with open(dest, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(rec.model_dump_json() + "\n")
    elif suffix == ".csv":
        fieldnames = list(ManifestRecord.model_fields.keys())
        with open(dest, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in records:
                row = rec.model_dump()
                # Represent None as empty string for clean CSV
                clean_row = {k: ("" if v is None else v) for k, v in row.items()}
                writer.writerow(clean_row)
    else:
        raise ManifestIOError(f"Unsupported manifest format: {suffix}. Must be .csv or .jsonl.")
