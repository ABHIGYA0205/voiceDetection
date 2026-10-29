#!/usr/bin/env python3
"""AASIST-L ASVspoof 2019 LA Official Reproduction Runner.

Verifies prerequisites, validates directory structure, runs initial smoke tests,
and executes official ASVspoof 2019 LA Logical Access evaluation with
resumable, fail-closed, CUDA/MPS/CPU-capable execution.

Usage:
    python scripts/reproduce_asvspoof_eval.py --help
    python scripts/reproduce_asvspoof_eval.py --check-only
    python scripts/reproduce_asvspoof_eval.py --smoke-only --device cuda
    python scripts/reproduce_asvspoof_eval.py --resume --checkpoint-every 50
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import re
import shlex
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from voiceshield.core import resolve_device_status
from voiceshield.evaluation.metrics import (
    compute_eer_interpolated,
    compute_min_dcf,
)
from voiceshield.models import (
    AASISTLDetector,
    get_official_registry_entry,
    pad_for_aasist,
)
from voiceshield.reproducibility import set_seed

# Ensure deterministic workspace configuration before cuBLAS initialization
_existing_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
if _existing_cublas is not None:
    if _existing_cublas not in {":4096:8", ":16:8"}:
        sys.stderr.write(
            f"HALTING: Incompatible pre-existing CUBLAS_WORKSPACE_CONFIG={_existing_cublas!r}. "
            "Must be ':4096:8' or ':16:8'.\n"
        )
        sys.exit(1)
else:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

logger = logging.getLogger("voiceshield.reproduction")

# Official Clova AI published benchmarks for AASIST-L on ASVspoof 2019 LA Eval
OFFICIAL_AASIST_L_EER = 0.0099  # 0.99%
OFFICIAL_AASIST_L_MIN_TDCF = 0.0309

# Exact official cardinalities for both eval and dev splits
OFFICIAL_EVAL_COUNTS = {"total": 71237, "bonafide": 7355, "spoof": 63882}
OFFICIAL_DEV_COUNTS = {"total": 24844, "bonafide": 2548, "spoof": 22296}
OFFICIAL_SPLIT_COUNTS = {
    "eval": OFFICIAL_EVAL_COUNTS,
    "dev": OFFICIAL_DEV_COUNTS,
}

# Default expected paths
DEFAULT_CHECKPOINT = Path("models/weights/AASIST-L.pth")
DEFAULT_DATASET_ROOT = Path("data/ASVspoof2019/LA")


@dataclass(frozen=True)
class ProgressState:
    """Structured, immutable representation of validated evaluation progress."""

    status: str
    contract: dict[str, Any]
    contract_hash: str
    scores: dict[str, float]
    artifact_hashes: dict[str, str]
    completed_count: int


def atomic_write_text(path: Path, content: str) -> None:
    """Atomically write text content to disk using a unique temporary file and fsync."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        try:
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def get_git_provenance() -> tuple[str, bool]:
    """Capture the actual commit and complete working-tree dirty state."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True
    return commit, dirty


def classify_eer_comparison(
    actual_eer: float | None,
    *,
    split: str,
    official_cardinality: bool,
) -> tuple[str, float | None]:
    """Compare EER only for the canonical official evaluation protocol."""
    if actual_eer is None:
        return "NOT COMPUTED", None
    if split != "eval" or not official_cardinality:
        return "NOT COMPARABLE", None
    difference = actual_eer - OFFICIAL_AASIST_L_EER
    status = "MATCHED (within tolerance)" if abs(difference) < 0.002 else "DISCREPANCY"
    return status, difference


def format_invoked_command(executable: str, argv: list[str]) -> str:
    """Format the full command invocation safely for report output."""
    cmd_parts = [executable] + list(argv)
    return " ".join(shlex.quote(p) for p in cmd_parts)


def format_optional_float(value: float | None, digits: int = 4) -> str:
    """Format optional float for Markdown reports without crashing."""
    if value is None:
        return "NOT COMPUTED"
    return f"{value:.{digits}f}"


def smoke_report_replacements(diagnostics: dict[str, Any], *, skipped: bool) -> dict[str, str]:
    """Build report template substitutions for the smoke test."""
    if skipped:
        return {
            "{{ smoke_status }}": "SKIPPED",
            "{{ smoke_samples_count }}": "0",
            "{{ smoke_mean_bonafide_score }}": "NOT COMPUTED",
            "{{ smoke_mean_spoof_score }}": "NOT COMPUTED",
            "{{ smoke_direction_aligned }}": "NOT COMPUTED",
            "{{ smoke_samples }}": "NOT COMPUTED (smoke verification skipped)",
            "{{ smoke_bonafide_count }}": "NOT COMPUTED",
            "{{ smoke_spoof_count }}": "NOT COMPUTED",
            "{{ smoke_bonafide_mean }}": "NOT COMPUTED",
            "{{ smoke_spoof_mean }}": "NOT COMPUTED",
            "{{ smoke_direction_status }}": "SKIPPED",
            "{{ smoke_elapsed }}": "NOT COMPUTED",
        }
    samples = diagnostics.get("samples_evaluated", "NOT COMPUTED")
    return {
        "{{ smoke_status }}": "PASSED" if diagnostics.get("direction_aligned") else "FAILED",
        "{{ smoke_samples_count }}": str(samples),
        "{{ smoke_mean_bonafide_score }}": format_optional_float(
            diagnostics.get("mean_bonafide_score")
        ),
        "{{ smoke_mean_spoof_score }}": format_optional_float(
            diagnostics.get("mean_spoof_score")
        ),
        "{{ smoke_direction_aligned }}": str(diagnostics.get("direction_aligned", "NOT COMPUTED")),
        "{{ smoke_samples }}": str(samples),
        "{{ smoke_bonafide_count }}": str(diagnostics.get("bonafide_samples", "NOT COMPUTED")),
        "{{ smoke_spoof_count }}": str(diagnostics.get("spoof_samples", "NOT COMPUTED")),
        "{{ smoke_bonafide_mean }}": format_optional_float(diagnostics.get("mean_bonafide_score")),
        "{{ smoke_spoof_mean }}": format_optional_float(diagnostics.get("mean_spoof_score")),
        "{{ smoke_direction_status }}": "ALIGNED (bonafide > spoof)"
        if diagnostics.get("direction_aligned")
        else "DISCREPANCY",
        "{{ smoke_elapsed }}": f"{diagnostics.get('elapsed_seconds', 0.0):.3f}s"
        if "elapsed_seconds" in diagnostics
        else "NOT COMPUTED",
    }


def render_report_template(template: str, replacements: dict[str, str]) -> str:
    """Render report template and verify no unexpanded placeholders remain."""
    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace(key, value)
    unresolved = sorted(set(re.findall(r"\{\{[^{}]+\}\}", rendered)))
    if unresolved:
        raise ValueError(f"Unresolved report template placeholders: {unresolved}")
    return rendered


def validate_protocol_cardinality(
    trials: list[tuple[str, str, str, str]],
    split: str = "eval",
    *,
    strict_official: bool = False,
) -> dict[str, int]:
    """Validate protocol trial counts against official ASVspoof 2019 partitions."""
    sample_ids = [entry[1] for entry in trials]
    duplicates = sorted(sid for sid, count in Counter(sample_ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate protocol IDs: {duplicates[:5]}")

    counts = Counter(entry[3] for entry in trials)
    unknown = sorted(set(counts) - {"bonafide", "spoof"})
    if unknown:
        raise ValueError(f"Unsupported protocol labels: {unknown}")

    bonafide = counts.get("bonafide", 0)
    spoof = counts.get("spoof", 0)
    total = len(trials)
    observed = {"total": total, "bonafide": bonafide, "spoof": spoof}

    expected = OFFICIAL_EVAL_COUNTS if split == "eval" else OFFICIAL_DEV_COUNTS
    if strict_official:
        if split not in {"eval", "dev"}:
            raise ValueError(
                f"Strict official cardinality verification requested for unsupported split: {split}"
            )
        if observed != expected:
            raise ValueError(
                f"Protocol trial counts for '{split}' do not match official ASVspoof 2019 LA partition "
                f"(cardinality mismatch)! Expected: {expected}, Observed: {observed}"
            )
    return observed


def validate_protocol_audio_coverage(
    trials: list[tuple[str, str, str, str]],
    audio_dir: Path,
    split: str = "eval",
    *,
    strict_official: bool = False,
) -> dict[str, Any]:
    """Validate CM protocol IDs against FLAC stems without scoring extras."""
    counts = validate_protocol_cardinality(trials, split=split, strict_official=strict_official)
    protocol_ids = {entry[1] for entry in trials}
    audio_paths = [
        path for path in audio_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".flac"
    ]
    stem_counts = Counter(path.stem for path in audio_paths)
    duplicates = sorted(stem for stem, count in stem_counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate audio FLAC stems: {duplicates[:5]}")
    audio_ids = set(stem_counts)
    missing = sorted(protocol_ids - audio_ids)
    extra = sorted(audio_ids - protocol_ids)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} protocol audio files are missing; first IDs: {missing[:5]}"
        )
    return {
        **counts,
        "protocol_ids": len(protocol_ids),
        "audio_files": len(audio_ids),
        "missing_files": len(missing),
        "extra_files": len(extra),
    }


def select_stratified_smoke_subset(
    trials: list[tuple[str, str, str, str]], smoke_count: int
) -> list[tuple[str, str, str, str]]:
    """Select a deterministic smoke subset containing both protocol classes."""
    if smoke_count < 2:
        raise ValueError("Smoke count must be at least 2 for stratified selection.")
    by_label = {
        "bonafide": [trial for trial in trials if trial[3] == "bonafide"],
        "spoof": [trial for trial in trials if trial[3] == "spoof"],
    }
    if not by_label["bonafide"] or not by_label["spoof"]:
        raise ValueError("Stratified smoke selection requires both bonafide and spoof classes.")
    if smoke_count > len(trials):
        raise ValueError(
            f"Requested stratified smoke subset of {smoke_count} exceeds {len(trials)} trials."
        )
    allocations = {
        "bonafide": max(1, smoke_count // 2),
        "spoof": max(1, smoke_count - smoke_count // 2),
    }
    if allocations["bonafide"] + allocations["spoof"] != smoke_count:
        raise ValueError(
            f"Invalid smoke allocation for requested count {smoke_count}: {allocations}"
        )
    for label, requested in allocations.items():
        available = len(by_label[label])
        if available < requested:
            raise ValueError(
                f"Insufficient {label} trials for stratified smoke selection: "
                f"requested {requested}, available {available}."
            )
    selected = (
        by_label["bonafide"][: allocations["bonafide"]] + by_label["spoof"][: allocations["spoof"]]
    )
    if len(selected) != smoke_count:
        raise ValueError(
            f"Stratified smoke selection returned {len(selected)} records; expected {smoke_count}."
        )
    return selected


def verify_directory_structure(
    checkpoint_path: Path,
    dataset_root: Path,
    split: str = "eval",
) -> tuple[bool, dict[str, Any]]:
    """Inspect expected files and build status checklist."""
    protocol_path = (
        dataset_root / "ASVspoof2019_LA_cm_protocols" / f"ASVspoof2019.LA.cm.{split}.trl.txt"
    )
    audio_dir = dataset_root / f"ASVspoof2019_LA_{split}" / "flac"

    checks = {
        "checkpoint": {
            "path": checkpoint_path,
            "exists": checkpoint_path.is_file(),
            "description": "Official AASIST-L weights (.pth)",
        },
        "dataset_root": {
            "path": dataset_root,
            "exists": dataset_root.is_dir(),
            "description": "ASVspoof 2019 LA root directory",
        },
        "protocol_file": {
            "path": protocol_path,
            "exists": protocol_path.is_file(),
            "description": f"Official CM {split} protocol file (ASVspoof2019.LA.cm.{split}.trl.txt)",
        },
        "audio_flac_dir": {
            "path": audio_dir,
            "exists": audio_dir.is_dir(),
            "description": f"FLAC audio directory for {split} split",
        },
    }

    all_present = all(c["exists"] for c in checks.values())
    return all_present, checks


def print_resource_checklist(checks: dict[str, Any]) -> None:
    """Print clean, actionable checklist for missing resources."""
    sys.stderr.write("\n" + "=" * 90 + "\n")
    sys.stderr.write(" VoiceShield: AASIST-L & ASVspoof 2019 Reproduction Resource Checklist\n")
    sys.stderr.write("=" * 90 + "\n\n")

    for _name, c in checks.items():
        status = "[PRESENT] " if c["exists"] else "[MISSING] "
        sys.stderr.write(f"  {status:<10} {c['description']}\n")
        sys.stderr.write(f"             Path: {c['path']}\n\n")

    sys.stderr.write("-" * 90 + "\n")
    sys.stderr.write(" ACTION REQUIRED TO RUN OFFICIAL REPRODUCTION:\n")
    sys.stderr.write("-" * 90 + "\n\n")

    reg = get_official_registry_entry("clovaai/aasist-l")
    if not checks["checkpoint"]["exists"]:
        sys.stderr.write(" 1. Download official AASIST-L checkpoint:\n")
        sys.stderr.write("    mkdir -p models/weights\n")
        sys.stderr.write(
            f"    curl -L -o {checks['checkpoint']['path']} \\\n      {reg.checkpoint_url}\n\n"
        )

    if not checks["protocol_file"]["exists"] or not checks["audio_flac_dir"]["exists"]:
        sys.stderr.write(" 2. Acquire and extract official ASVspoof 2019 LA dataset:\n")
        sys.stderr.write("    Official Archive: ASVspoof 2019 Logical Access (LA) partition\n")
        sys.stderr.write("    Expected layout:\n")
        sys.stderr.write("      data/ASVspoof2019/LA/\n")
        sys.stderr.write("        ├── ASVspoof2019_LA_cm_protocols/\n")
        sys.stderr.write("        │   ├── ASVspoof2019.LA.cm.dev.trl.txt\n")
        sys.stderr.write("        │   └── ASVspoof2019.LA.cm.eval.trl.txt\n")
        sys.stderr.write("        ├── ASVspoof2019_LA_dev/\n")
        sys.stderr.write("        │   └── flac/ (LA_D_*.flac)\n")
        sys.stderr.write("        └── ASVspoof2019_LA_eval/\n")
        sys.stderr.write("            └── flac/ (LA_E_*.flac)\n\n")
        sys.stderr.write("    Dataset Download References:\n")
        sys.stderr.write(
            "    - University of Edinburgh / DataShare: https://datashare.ed.ac.uk/handle/10283/3336\n"
        )
        sys.stderr.write("    - Academic Torrent: ASVspoof 2019 LA.zip\n\n")

    sys.stderr.write("=" * 90 + "\n\n")


def load_asvspoof_protocol(protocol_path: Path) -> list[tuple[str, str, str, str]]:
    """Parse ASVspoof 2019 protocol file.

    Format:
        SPEAKER_ID AUDIO_FILE_NAME SYSTEM_ID KEY_OR_TRASH LABEL
        e.g. LA_0070 LA_E_2833633 - - bonafide
        e.g. LA_0070 LA_E_8879621 A07 - spoof

    Returns:
        List of tuples: (speaker_id, sample_id, attack_system, label_str)
    """
    entries: list[tuple[str, str, str, str]] = []
    with open(protocol_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                speaker_id = parts[0]
                sample_id = parts[1]
                attack_system = parts[2]
                label_str = parts[4].lower()
                entries.append((speaker_id, sample_id, attack_system, label_str))
    return entries


def run_smoke_test(
    detector: AASISTLDetector,
    trials: list[tuple[str, str, str, str]],
    audio_dir: Path,
    smoke_count: int = 50,
) -> dict[str, Any]:
    """Execute a stratified smoke test to verify pipeline and label direction."""
    subset = select_stratified_smoke_subset(trials, smoke_count)
    bonafide_scores: list[float] = []
    spoof_scores: list[float] = []

    start_time = time.perf_counter()
    for _, sample_id, _, label in subset:
        flac_path = audio_dir / f"{sample_id}.flac"
        if not flac_path.is_file():
            raise FileNotFoundError(f"Smoke test audio not found: {flac_path}")

        samples, sr = sf.read(str(flac_path), dtype="float32")
        if sr != 16000:
            raise ValueError(f"Audio {sample_id} sample rate is {sr} Hz, expected 16000 Hz.")
        if samples.ndim != 1:
            raise ValueError(f"Audio {sample_id} is not mono (shape: {samples.shape}).")
        if not np.isfinite(samples).all():
            raise ValueError(f"Audio {sample_id} contains non-finite samples.")

        out = detector.predict(samples)
        if not math.isfinite(out.bonafide_score):
            raise RuntimeError(f"Detector returned non-finite score for smoke sample {sample_id}.")

        if label == "bonafide":
            bonafide_scores.append(out.bonafide_score)
        else:
            spoof_scores.append(out.bonafide_score)

    elapsed = time.perf_counter() - start_time
    if not bonafide_scores or not spoof_scores:
        raise ValueError("Smoke subset must produce scores for both protocol classes.")
    mean_bf = float(np.mean(bonafide_scores))
    mean_sp = float(np.mean(spoof_scores))

    # Direction check: bonafide score should be higher for authentic speech
    direction_aligned = bool(mean_bf > mean_sp)

    return {
        "samples_evaluated": len(subset),
        "bonafide_samples": len(bonafide_scores),
        "spoof_samples": len(spoof_scores),
        "elapsed_seconds": round(elapsed, 3),
        "mean_bonafide_score": round(mean_bf, 4),
        "mean_spoof_score": round(mean_sp, 4),
        "direction_aligned": direction_aligned,
    }


def compute_contract_hash(contract: dict[str, Any]) -> str:
    """Canonical helper computing a tamper-evident SHA-256 hash across contract fields."""
    cleaned = {k: v for k, v in contract.items() if k != "contract_hash"}
    return hashlib.sha256(
        json.dumps(cleaned, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def build_resume_contract(
    *,
    seed: int,
    git_commit: str,
    git_dirty: bool,
    resolved_device: str,
    gpu_identity: str | None,
    gpu_capability: Any,
    torch_version: str,
    cuda_version: str | None,
    cudnn_version: str | None,
    determinism: dict[str, Any],
    protocol_sha256: str,
    checkpoint_sha256: str,
    model_id: str,
    config_hash: str,
    upstream_commit: str,
    preprocessing_contract: str,
    selected_split: str,
    batch_size: int,
    score_direction: str,
    python_version: str | None = None,
    platform_name: str | None = None,
    numpy_version: str | None = None,
    soundfile_version: str | None = None,
) -> dict[str, Any]:
    """Construct a complete, tamper-evident resume contract and compute its hash."""
    contract = {
        "batch_size": batch_size,
        "checkpoint_sha256": checkpoint_sha256,
        "clean_git_state": not git_dirty,
        "config_hash": config_hash,
        "cuda_version": cuda_version,
        "cudnn_version": cudnn_version,
        "determinism": determinism,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "gpu_capability": list(gpu_capability)
        if isinstance(gpu_capability, (list, tuple))
        else gpu_capability,
        "gpu_identity": gpu_identity,
        "model_id": model_id,
        "numpy_version": numpy_version or np.__version__,
        "platform": platform_name or platform.platform(),
        "preprocessing_contract": preprocessing_contract,
        "protocol_sha256": protocol_sha256,
        "python_version": python_version or platform.python_version(),
        "resolved_device": resolved_device,
        "score_direction": score_direction,
        "seed": seed,
        "selected_split": selected_split,
        "soundfile_version": soundfile_version or sf.__version__,
        "torch_version": torch_version,
        "upstream_commit": upstream_commit,
    }
    contract["contract_hash"] = compute_contract_hash(contract)
    return contract


def _validate_raw_checkpoint_data(data: Any, path: Path) -> dict[str, Any]:
    """Strictly validate schema, types, and score formats of raw checkpoint JSON data."""
    if not isinstance(data, dict):
        raise ValueError(
            f"Progress checkpoint root must be a JSON object at {path}, got {type(data).__name__}"
        )
    if data.get("schema_version") != "1.0":
        raise ValueError(
            f"Unsupported progress schema version at {path}: {data.get('schema_version')!r}"
        )
    status = data.get("status")
    if status not in {"INCOMPLETE", "COMPLETED"}:
        raise ValueError(f"Invalid progress checkpoint status at {path}: {status!r}")

    completed_count = data.get("completed_count")
    if not isinstance(completed_count, int) or completed_count < 0:
        raise ValueError(f"Invalid completed_count at {path}: {completed_count!r}")

    raw_scores = data.get("scores")
    if not isinstance(raw_scores, list):
        raise ValueError(f"Progress checkpoint 'scores' must be a list at {path}")
    if len(raw_scores) != completed_count:
        raise ValueError(
            f"Progress completed_count ({completed_count}) != scores length ({len(raw_scores)}) at {path}"
        )

    stored_contract = data.get("contract")
    if not isinstance(stored_contract, dict):
        raise ValueError(f"Missing valid contract object at {path}")

    stored_hash = data.get("contract_hash")
    if not isinstance(stored_hash, str) or len(stored_hash) != 64:
        raise ValueError(f"Missing or invalid top-level contract_hash at {path}: {stored_hash!r}")

    # Recompute and verify contract hash
    recomputed = compute_contract_hash(stored_contract)
    if stored_hash != recomputed:
        raise ValueError(
            f"Tampered contract detected in checkpoint at {path}: claimed hash {stored_hash}, "
            f"recomputed {recomputed}"
        )

    # Verify score entries
    seen_ids: set[str] = set()
    for item in raw_scores:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"Malformed score entry in {path}: {item!r}")
        sid, val = item
        if not isinstance(sid, str) or not sid:
            raise ValueError(f"Malformed sample ID in {path}: {sid!r}")
        if sid in seen_ids:
            raise ValueError(f"Duplicate sample ID in {path}: {sid}")
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise ValueError(f"Malformed non-numeric score for sample {sid} in {path}: {val!r}")
        val_float = float(val)
        if not math.isfinite(val_float):
            raise ValueError(f"Non-finite score for sample {sid} in {path}: {val_float}")
        seen_ids.add(sid)

    artifact_hashes = data.get("artifact_hashes", {})
    if not isinstance(artifact_hashes, dict):
        raise ValueError(f"artifact_hashes must be a dictionary in {path}")
    for name, hval in artifact_hashes.items():
        if (
            not isinstance(hval, str)
            or len(hval) != 64
            or not all(c in "0123456789abcdefABCDEF" for c in hval)
        ):
            raise ValueError(f"Invalid SHA-256 hash for artifact {name} in {path}: {hval!r}")

    return data


def save_progress_checkpoint(
    progress_path: Path,
    contract: dict[str, Any],
    scores: list[tuple[str, float]],
    *,
    status: str = "INCOMPLETE",
    artifact_hashes: dict[str, str] | None = None,
    backup_existing: bool = True,
) -> None:
    """Atomically persist evaluation progress to disk and maintain a crash-safe backup."""
    if status not in {"INCOMPLETE", "COMPLETED"}:
        raise ValueError(f"Unsupported status: {status!r}")

    computed_hash = compute_contract_hash(contract)
    contract_copy = dict(contract)
    contract_copy["contract_hash"] = computed_hash

    # Validate scores before serializing
    seen_ids: set[str] = set()
    for item in scores:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"Malformed score entry: {item!r}")
        sid, val = item
        if not isinstance(sid, str) or not sid:
            raise ValueError(f"Malformed sample ID: {sid!r}")
        if sid in seen_ids:
            raise ValueError(f"Duplicate sample ID: {sid}")
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise ValueError(f"Malformed non-numeric score for sample {sid}: {val!r}")
        if not math.isfinite(float(val)):
            raise ValueError(f"Non-finite score for sample {sid}: {val}")
        seen_ids.add(sid)

    art_hashes = artifact_hashes or {}
    for aname, ahash in art_hashes.items():
        if (
            not isinstance(ahash, str)
            or len(ahash) != 64
            or not all(c in "0123456789abcdefABCDEF" for c in ahash)
        ):
            raise ValueError(f"Invalid SHA-256 hash for {aname}: {ahash!r}")
    if status == "COMPLETED" and (
        "report_sha256" not in art_hashes or "raw_scores_sha256" not in art_hashes
    ):
        raise ValueError("COMPLETED progress must include report_sha256 and raw_scores_sha256")

    bak_path = progress_path.with_name(f"{progress_path.name}.bak")

    # If primary exists and is valid, back it up. If backup rotation fails, fail closed!
    if backup_existing and progress_path.is_file():
        try:
            content = progress_path.read_text(encoding="utf-8")
            parsed = json.loads(content)
            _validate_raw_checkpoint_data(parsed, progress_path)
            # Atomically write backup
            atomic_write_text(bak_path, content)
        except Exception as exc:
            raise RuntimeError(
                f"Fail-closed: backup rotation failed for {progress_path} -> {bak_path}: {exc}. "
                "Preserving existing primary untouched."
            ) from exc

    payload = {
        "schema_version": "1.0",
        "status": status,
        "completed_count": len(scores),
        "contract": contract_copy,
        "contract_hash": computed_hash,
        "scores": [[sid, float(score)] for sid, score in scores],
        "artifact_hashes": art_hashes,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    encoded = json.dumps(payload, indent=2, allow_nan=False)
    atomic_write_text(progress_path, encoded)


def load_and_validate_progress(
    progress_path: Path,
    expected_contract: dict[str, Any],
    valid_protocol_ids: set[str],
) -> ProgressState:
    """Load and validate an existing progress checkpoint artifact.

    Recovers from .bak if the primary file is truncated or corrupt without destroying the backup.
    Returns a structured ProgressState.
    """
    bak_path = progress_path.with_name(f"{progress_path.name}.bak")
    data: dict[str, Any] | None = None
    recovered_from_bak = False
    raw_content: str | None = None
    primary_error: Exception | None = None

    if progress_path.is_file():
        try:
            raw_content = progress_path.read_text(encoding="utf-8")
            parsed = json.loads(raw_content)
            data = _validate_raw_checkpoint_data(parsed, progress_path)
        except Exception as exc:
            primary_error = exc
            logger.warning(
                "Primary progress file %s is corrupt or invalid (%s). Attempting recovery from backup...",
                progress_path,
                exc,
            )
            data = None

    if data is None and bak_path.is_file():
        try:
            raw_content = bak_path.read_text(encoding="utf-8")
            parsed = json.loads(raw_content)
            data = _validate_raw_checkpoint_data(parsed, bak_path)
            recovered_from_bak = True
            logger.info("Successfully recovered progress state from backup %s", bak_path)
        except Exception as exc:
            logger.warning("Backup progress file %s is also corrupt or invalid: %s", bak_path, exc)
            data = None

    if data is None:
        if primary_error is not None:
            raise primary_error
        if progress_path.is_file() or bak_path.is_file():
            raise RuntimeError(
                f"Progress checkpoint at {progress_path} is corrupt or truncated, and no valid backup exists."
            )
        raise FileNotFoundError(f"No progress checkpoint found at {progress_path}")

    # Validate contract match
    stored_contract = data["contract"]
    stored_hash = data["contract_hash"]
    expected_hash = expected_contract.get("contract_hash") or compute_contract_hash(
        expected_contract
    )

    clean_stored = {k: v for k, v in stored_contract.items() if k != "contract_hash"}
    clean_expected = {k: v for k, v in expected_contract.items() if k != "contract_hash"}

    if stored_hash != expected_hash or clean_stored != clean_expected:
        all_keys = sorted(set(clean_expected.keys()) | set(clean_stored.keys()))
        mismatches: list[str] = []
        for k in all_keys:
            ev = clean_expected.get(k)
            sv = clean_stored.get(k)
            if ev != sv:
                mismatches.append(f"'{k}': expected {ev!r}, got {sv!r}")
        mismatch_str = (
            "; ".join(mismatches)
            if mismatches
            else f"hash mismatch ({stored_hash} != {expected_hash})"
        )
        raise ValueError(f"Resume rejected due to contract mismatch: {mismatch_str}")

    # Validate sample IDs against official protocol
    validated_scores: dict[str, float] = {}
    for sid, val in data["scores"]:
        if sid not in valid_protocol_ids:
            raise ValueError(f"Unknown sample ID in progress checkpoint: {sid}")
        validated_scores[sid] = float(val)

    if recovered_from_bak and raw_content is not None:
        # Restore the primary progress file atomically preserving the backup untouched
        atomic_write_text(progress_path, raw_content)

    return ProgressState(
        status=data["status"],
        contract=stored_contract,
        contract_hash=stored_hash,
        scores=validated_scores,
        artifact_hashes=data.get("artifact_hashes", {}),
        completed_count=data["completed_count"],
    )


def evaluate_protocol_dataset(
    detector: AASISTLDetector,
    trials: list[tuple[str, str, str, str]],
    audio_dir: Path,
    batch_size: int = 24,
    checkpoint_every: int = 50,
    on_batch_complete: Callable[[list[tuple[str, float]], bool], None] | None = None,
) -> tuple[
    np.ndarray[Any, np.dtype[np.int_]],
    np.ndarray[Any, np.dtype[np.float64]],
    dict[str, float],
]:
    """Evaluate full ASVspoof protocol trials using batched inference.

    Validates 16 kHz sample rate, mono channel, finite audio, and finite model outputs.
    Supports atomic incremental persistence through on_batch_complete callback.
    Never treats a partially written batch as complete.
    """
    all_labels: list[int] = []
    all_scores: list[float] = []
    scores_dict: dict[str, float] = {}

    total = len(trials)
    device = detector.device
    batches_since_flush = 0

    for start_idx in range(0, total, batch_size):
        batch_trials = trials[start_idx : start_idx + batch_size]
        batch_tensors: list[torch.Tensor] = []
        batch_labels: list[int] = []
        batch_ids: list[str] = []

        for _, sample_id, _, label in batch_trials:
            flac_path = audio_dir / f"{sample_id}.flac"
            if not flac_path.is_file():
                raise FileNotFoundError(f"Evaluation audio file not found: {flac_path}")

            audio_samples, sr = sf.read(str(flac_path), dtype="float32")
            if sr != 16000:
                raise ValueError(
                    f"Audio file {flac_path.name} sample rate is {sr} Hz; expected 16000 Hz."
                )
            if audio_samples.ndim != 1:
                raise ValueError(
                    f"Audio file {flac_path.name} is not mono (shape: {audio_samples.shape})."
                )
            if not np.isfinite(audio_samples).all():
                raise ValueError(
                    f"Audio file {flac_path.name} contains non-finite (NaN or Inf) audio samples."
                )

            # Enforce exact 64,600-sample framing
            padded_tensor = pad_for_aasist(audio_samples, max_len=64600)
            if not torch.isfinite(padded_tensor).all():
                raise ValueError(f"Padded audio tensor for {sample_id} contains non-finite values.")

            batch_tensors.append(padded_tensor)
            batch_labels.append(1 if label == "spoof" else 0)
            batch_ids.append(sample_id)

        stacked = torch.stack(batch_tensors, dim=0).to(device)
        with torch.no_grad():
            _, outputs = detector.model(stacked)
            if not torch.isfinite(outputs).all():
                raise RuntimeError(
                    f"Detector produced non-finite (NaN or Inf) output logits for batch at index {start_idx}."
                )
            bf_scores = outputs[:, 1].detach().cpu().numpy()

        batch_results: list[tuple[str, float]] = []
        for sid, score, y in zip(batch_ids, bf_scores, batch_labels, strict=True):
            score_float = float(score)
            if not math.isfinite(score_float):
                raise RuntimeError(
                    f"Detector produced non-finite score for sample {sid}: {score_float}"
                )
            scores_dict[sid] = score_float
            all_scores.append(score_float)
            all_labels.append(y)
            batch_results.append((sid, score_float))

        batches_since_flush += 1
        is_last_batch = (start_idx + batch_size) >= total
        should_flush = is_last_batch or (batches_since_flush >= checkpoint_every)

        if on_batch_complete is not None:
            on_batch_complete(batch_results, should_flush)
            if should_flush:
                batches_since_flush = 0

        if (start_idx // batch_size) % 50 == 0 or is_last_batch:
            logger.info("Processed %d / %d trials...", min(start_idx + batch_size, total), total)

    return np.array(all_labels, dtype=int), np.array(all_scores, dtype=np.float64), scores_dict


def finalize_evaluation(
    trials: list[tuple[str, str, str, str]],
    all_scores: dict[str, float],
    split: str,
    output_dir: Path,
    contract: dict[str, Any],
    progress_path: Path,
    report_template_content: str,
    report_replacements: dict[str, str],
) -> tuple[Path, Path, dict[str, str]]:
    """Complete official publication: validate completeness, format raw scores, render report, and publish atomically."""
    expected_counts = OFFICIAL_SPLIT_COUNTS.get(split)
    if not expected_counts:
        raise ValueError(f"Unsupported split for finalization: {split}")

    expected_total = expected_counts["total"]
    if len(all_scores) != expected_total:
        raise ValueError(
            f"Cannot finalize incomplete {split} evaluation: expected {expected_total} scores, "
            f"got {len(all_scores)}."
        )

    # 1. Format raw scores in canonical protocol order with round-trip-safe precision (.9g)
    raw_scores_lines: list[str] = []
    for _, sample_id, _, _ in trials:
        if sample_id not in all_scores:
            raise ValueError(f"Missing score for protocol sample: {sample_id}")
        score_val = all_scores[sample_id]
        raw_scores_lines.append(f"{sample_id} {score_val:.9g}\n")

    raw_scores_text = "".join(raw_scores_lines)
    raw_scores_sha256 = hashlib.sha256(raw_scores_text.encode("utf-8")).hexdigest()

    raw_scores_file = output_dir / "raw_scores.txt"
    report_file = output_dir / "report.md"

    # 2. Render report with raw_scores_sha256 and resume_contract_hash included
    final_replacements = dict(report_replacements)
    final_replacements["{{ raw_scores_sha256 }}"] = raw_scores_sha256
    final_replacements["{{ resume_contract_hash }}"] = contract.get("contract_hash", "")
    final_replacements["{{ raw_scores_path }}"] = str(raw_scores_file)
    final_replacements["{{ report_path }}"] = str(report_file)

    rendered_report = render_report_template(report_template_content, final_replacements)
    report_sha256 = hashlib.sha256(rendered_report.encode("utf-8")).hexdigest()

    # 3. Publish raw_scores.txt first
    atomic_write_text(raw_scores_file, raw_scores_text)

    # 4. Publish report.md second
    atomic_write_text(report_file, rendered_report)

    artifact_hashes = {
        "raw_scores_sha256": raw_scores_sha256,
        "report_sha256": report_sha256,
    }

    # 5. Mark progress COMPLETED last
    canonical_score_list = [(entry[1], all_scores[entry[1]]) for entry in trials]
    save_progress_checkpoint(
        progress_path=progress_path,
        contract=contract,
        scores=canonical_score_list,
        status="COMPLETED",
        artifact_hashes=artifact_hashes,
    )

    return raw_scores_file, report_file, artifact_hashes


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser for reproduction runner."""
    parser = argparse.ArgumentParser(
        prog="reproduce_asvspoof_eval",
        description="Verify prerequisites and reproduce official AASIST-L on ASVspoof 2019 LA.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to official AASIST-L.pth checkpoint file.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Path to ASVspoof 2019 LA root directory.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="eval",
        choices=["eval", "dev"],
        help="Evaluation protocol split to benchmark ('eval' or 'dev').",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["mps", "cpu", "cuda"],
        help="Preferred compute device ('cuda', 'mps', or 'cpu').",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=24,
        help="Inference batch size matching official AASIST config (default: 24).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted evaluation from an atomic progress checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Number of batches between incremental progress checkpoint flushes (default: 50).",
    )
    parser.add_argument(
        "--smoke-count",
        type=int,
        default=50,
        help="Number of deterministic stratified samples for smoke diagnostics (default: 50).",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip initial 50-sample smoke verification.",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Execute prerequisites and smoke diagnostics only, write diagnostic artifact, and exit before full evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed for evaluation (default: 42, must be in [0, 2**32 - 1]).",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate checkpoint, protocol cardinality, and protocol-to-FLAC coverage without inference.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/reproduction/asvspoof2019_la"),
        help="Destination directory for benchmark report and scores.",
    )
    return parser


def validate_cli_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Validate parsed CLI arguments, rejecting invalid numeric values and incompatible modes."""
    if not (0 <= args.seed <= 2**32 - 1):
        parser.error(
            f"--seed must be an integer between 0 and 2**32 - 1 ({2**32 - 1}), got {args.seed}"
        )
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be at least 1")
    if args.smoke_only and args.skip_smoke:
        parser.error("--smoke-only cannot be combined with --skip-smoke")
    if args.smoke_only and args.resume:
        parser.error("--smoke-only cannot be combined with --resume")
    if args.smoke_only and args.check_only:
        parser.error("--smoke-only cannot be combined with --check-only")
    if args.check_only and args.resume:
        parser.error("--check-only cannot be combined with --resume")
    if not args.skip_smoke and args.smoke_count < 2:
        parser.error("--smoke-count must be at least 2 unless --skip-smoke is used")


def main(argv: list[str] | None = None) -> int:
    """Main reproduction entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_cli_arguments(parser, args)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Early collision check for non-resume runs before any heavy operations or inference
    progress_path = args.output_dir / f"{args.split}_progress.json"
    bak_path = progress_path.with_name(f"{args.split}_progress.json.bak")
    report_file = args.output_dir / "report.md"
    raw_scores_file = args.output_dir / "raw_scores.txt"

    if not args.resume and not args.check_only and args.output_dir.is_dir():
        stale_candidates = (
            [
                progress_path,
                bak_path,
                report_file,
                raw_scores_file,
            ]
            + list(args.output_dir.glob("*.tmp"))
            + list(args.output_dir.glob(".*.tmp"))
        )
        for candidate in stale_candidates:
            if candidate.exists():
                sys.stderr.write(
                    f"HALTING: Stale collision detected in {args.output_dir}. "
                    f"Artifact already exists: {candidate.name}. "
                    "Use --resume to continue or specify a clean --output-dir.\n"
                )
                return 1

    # 1. Deterministic seeding and strict full-run provenance
    is_full_run = not (args.check_only or args.smoke_only)
    try:
        provenance_deterministic = set_seed(
            args.seed,
            deterministic=True,
            fail_closed=is_full_run,
        )
    except (ValueError, RuntimeError) as exc:
        sys.stderr.write(f"HALTING: Deterministic setup failed: {exc}\n")
        return 1

    # Check Git provenance
    git_commit, git_dirty = get_git_provenance()

    # 2. Enforce official model registry integrity on checkpoint BEFORE dataset access
    reg = get_official_registry_entry("clovaai/aasist-l")

    if not args.checkpoint.is_file():
        sys.stderr.write(
            f"HALTING: Official AASIST-L checkpoint not found at: {args.checkpoint}\n"
            "Please acquire the official checkpoint following the instructions below.\n"
        )
        _, checks = verify_directory_structure(args.checkpoint, args.dataset_root, args.split)
        print_resource_checklist(checks)
        return 1

    checkpoint_sha = compute_file_sha256(args.checkpoint)
    if checkpoint_sha != reg.checkpoint_sha256:
        sys.stderr.write(
            f"HALTING: Checkpoint SHA-256 integrity verification failed for {args.checkpoint}!\n"
            f"  Expected (registry): {reg.checkpoint_sha256}\n"
            f"  Actual:              {checkpoint_sha}\n"
            "Official reproduction requires byte verification against the immutable registry.\n"
        )
        return 1

    # 3. Verify directory structure & prerequisites
    all_present, checks = verify_directory_structure(
        checkpoint_path=args.checkpoint,
        dataset_root=args.dataset_root,
        split=args.split,
    )

    if not all_present:
        print_resource_checklist(checks)
        sys.stderr.write(
            "HALTING: Cannot proceed with reproduction. Required dataset files are missing.\n"
            "Please acquire the official dataset following the instructions above.\n\n"
        )
        return 1

    protocol_file = checks["protocol_file"]["path"]
    audio_dir = checks["audio_flac_dir"]["path"]

    protocol_sha = compute_file_sha256(protocol_file)
    trials = load_asvspoof_protocol(protocol_file)
    strict_official = args.split in {"eval", "dev"}
    try:
        coverage = validate_protocol_audio_coverage(
            trials, audio_dir, split=args.split, strict_official=strict_official
        )
    except (FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"HALTING: Protocol/audio validation failed: {exc}\n")
        return 1
    logger.info("Protocol/audio coverage for %s: %s", args.split, coverage)
    if args.check_only:
        print(f"Protocol/audio validation passed: {coverage}")
        return 0

    logger.info("AASIST-L Checkpoint: %s (SHA-256: %s)", args.checkpoint, checkpoint_sha[:16])
    logger.info("Protocol File: %s (SHA-256: %s)", protocol_file, protocol_sha[:16])

    # 4. Resolve device and load detector with explicit official identity
    try:
        dev_status = resolve_device_status(
            preferred=args.device,
            fallback_to_cpu=False if args.device == "cuda" else True,
        )
        detector = AASISTLDetector(
            checkpoint_path=args.checkpoint,
            model_id="clovaai/aasist-l",
            preferred_device=dev_status.device_name,
            fallback_to_cpu=False,
        )
    except (RuntimeError, ValueError) as exc:
        sys.stderr.write(f"HALTING: Device or detector initialization failed: {exc}\n")
        return 1

    logger.info("AASIST-L initialized on device: %s", detector.device)
    logger.info("Loaded %d trials from protocol %s", len(trials), protocol_file.name)

    config_hash = hashlib.sha256(
        json.dumps(detector.config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    repro_cmd = format_invoked_command(sys.executable, sys.argv)
    # 5. Full scoring clean Git working tree validation
    if is_full_run and (git_commit == "unknown" or git_dirty):
        sys.stderr.write(
            f"HALTING: Official full evaluation requires a clean Git working tree. "
            f"Detected commit={git_commit!r}, dirty={git_dirty}. Refusing evaluation.\n"
        )
        return 1

    # 6. Full split evaluation contract and atomic resumability verification
    resume_contract = build_resume_contract(
        seed=args.seed,
        git_commit=git_commit,
        git_dirty=git_dirty,
        resolved_device=detector.device.type,
        gpu_identity=dev_status.cuda_device_name if dev_status.is_cuda else None,
        gpu_capability=provenance_deterministic.get("cuda_capability"),
        torch_version=torch.__version__,
        cuda_version=provenance_deterministic.get("cuda_version"),
        cudnn_version=provenance_deterministic.get("cudnn_version"),
        determinism={
            "cublas_workspace_config": provenance_deterministic.get("cublas_workspace_config"),
            "cuda_allow_tf32": provenance_deterministic.get("cuda_allow_tf32"),
            "cudnn_allow_tf32": provenance_deterministic.get("cudnn_allow_tf32"),
            "cudnn_benchmark": provenance_deterministic.get("cudnn_benchmark"),
            "cudnn_deterministic": provenance_deterministic.get("cudnn_deterministic"),
            "deterministic_algorithms": provenance_deterministic.get(
                "deterministic_algorithms", False
            ),
        },
        protocol_sha256=protocol_sha,
        checkpoint_sha256=checkpoint_sha,
        model_id=detector.provenance.model_id,
        config_hash=config_hash,
        upstream_commit=reg.upstream_commit,
        preprocessing_contract="pad_for_aasist: 64600 samples, cyclic pad/truncate",
        selected_split=args.split,
        batch_size=args.batch_size,
        score_direction="raw_bonafide_score=logit[1]; metric_spoof_score=-logit[1]",
    )

    valid_protocol_ids = {entry[1] for entry in trials}
    already_scored: dict[str, float] = {}

    if args.resume:
        logger.info("Resuming evaluation from progress checkpoint %s...", progress_path)
        try:
            progress_state = load_and_validate_progress(
                progress_path=progress_path,
                expected_contract=resume_contract,
                valid_protocol_ids=valid_protocol_ids,
            )
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            sys.stderr.write(f"HALTING: Resume failed: {exc}\n")
            return 1

        # Check if already completed
        if progress_state.status == "COMPLETED":
            expected_total = OFFICIAL_SPLIT_COUNTS[args.split]["total"]
            if len(progress_state.scores) != expected_total:
                sys.stderr.write(
                    f"HALTING: Progress marked COMPLETED but trial count ({len(progress_state.scores)}) "
                    f"does not match official {args.split} total ({expected_total}).\n"
                )
                return 1
            if not report_file.is_file() or not raw_scores_file.is_file():
                sys.stderr.write(
                    f"HALTING: Progress marked COMPLETED but required artifacts are missing: "
                    f"report={report_file.is_file()}, raw_scores={raw_scores_file.is_file()}.\n"
                )
                return 1
            rep_hash = compute_file_sha256(report_file)
            scores_hash = compute_file_sha256(raw_scores_file)
            exp_rep = progress_state.artifact_hashes.get("report_sha256")
            exp_scores = progress_state.artifact_hashes.get("raw_scores_sha256")
            if rep_hash != exp_rep or scores_hash != exp_scores:
                sys.stderr.write(
                    f"HALTING: COMPLETED artifact hash verification failed!\n"
                    f"  Report SHA: Expected {exp_rep}, Actual {rep_hash}\n"
                    f"  Scores SHA: Expected {exp_scores}, Actual {scores_hash}\n"
                )
                return 1
            logger.info(
                "Resume verified existing COMPLETED evaluation artifacts in %s. Exiting cleanly.",
                args.output_dir,
            )
            print(f"Evaluation already complete and verified: {report_file}")
            return 0

        # When INCOMPLETE, reject pre-existing final report or raw scores as stale collisions
        if report_file.exists() or raw_scores_file.exists():
            sys.stderr.write(
                f"HALTING: Stale final evidence found in {args.output_dir} alongside INCOMPLETE progress. "
                "Remove stale report.md / raw_scores.txt before resuming.\n"
            )
            return 1

        already_scored = dict(progress_state.scores)
        logger.info("Resumed %d already-scored trials from checkpoint.", len(already_scored))

    # 6. Smoke test subset (if not already completed)
    smoke_diagnostics: dict[str, Any] = {}
    if not args.skip_smoke:
        logger.info("Running initial %d-sample smoke subset...", args.smoke_count)
        if detector.device.type != resume_contract["resolved_device"]:
            sys.stderr.write(
                f"HALTING: Device inconsistency before smoke inference. "
                f"Expected {resume_contract['resolved_device']!r}, got {detector.device.type!r}.\n"
            )
            return 1
        assert detector.device.type == resume_contract["resolved_device"]

        try:
            smoke_diagnostics = run_smoke_test(
                detector=detector,
                trials=trials,
                audio_dir=audio_dir,
                smoke_count=args.smoke_count,
            )
        except Exception as exc:
            sys.stderr.write(f"HALTING: Smoke validation failed: {exc}\n")
            return 1

        if detector.device.type != resume_contract["resolved_device"]:
            sys.stderr.write(
                f"HALTING: Scoring device changed during smoke inference! "
                f"Expected {resume_contract['resolved_device']!r}, got {detector.device.type!r}. "
                "Halting before full scoring to prevent mixed-device evaluation.\n"
            )
            return 1
        assert detector.device.type == resume_contract["resolved_device"]

        logger.info(
            "Smoke complete. Mean Bonafide: %.4f | Mean Spoof: %.4f | Direction Aligned: %s",
            smoke_diagnostics["mean_bonafide_score"],
            smoke_diagnostics["mean_spoof_score"],
            smoke_diagnostics["direction_aligned"],
        )
        if not smoke_diagnostics["direction_aligned"]:
            sys.stderr.write("ERROR: Label direction test failed in smoke subset! Halting.\n")
            return 1

        if args.smoke_only:
            diagnostic_path = args.output_dir / "smoke_diagnostic.json"
            diagnostic_payload = {
                "checkpoint_sha256": checkpoint_sha,
                "config_hash": config_hash,
                "cuda_version": provenance_deterministic.get("cuda_version"),
                "cudnn_version": provenance_deterministic.get("cudnn_version"),
                "determinism": provenance_deterministic,
                "git_commit": git_commit,
                "git_dirty": git_dirty,
                "gpu_capability": provenance_deterministic.get("cuda_capability"),
                "model_id": detector.provenance.model_id,
                "official_comparison": "NOT COMPARABLE",
                "official_reference_eer": None,
                "protocol_sha256": protocol_sha,
                "protocol_split": args.split,
                "reproduction_command": repro_cmd,
                "requested_device": str(args.device or os.getenv("VOICESHIELD_DEVICE", "mps")),
                "resolved_device": detector.device.type,
                "seed": args.seed,
                "smoke_diagnostics": smoke_diagnostics,
                "status": "NON_COMPARABLE_DIAGNOSTIC_ONLY",
                "timestamp": datetime.now(UTC).isoformat(),
                "torch_version": torch.__version__,
            }
            atomic_write_text(
                diagnostic_path, json.dumps(diagnostic_payload, indent=2, allow_nan=False)
            )
            logger.info("Saved smoke diagnostic artifact to %s", diagnostic_path)
            print(f"Smoke-only diagnostic verification completed successfully: {diagnostic_path}")
            return 0
    else:
        logger.warning("Smoke verification explicitly skipped; smoke metrics will be NOT COMPUTED.")

    if detector.device.type != resume_contract["resolved_device"]:
        sys.stderr.write(
            f"HALTING: Scoring device inconsistency before full scoring! "
            f"Expected {resume_contract['resolved_device']!r}, got {detector.device.type!r}.\n"
        )
        return 1
    assert detector.device.type == resume_contract["resolved_device"]

    remaining_trials = [t for t in trials if t[1] not in already_scored]
    logger.info(
        "Evaluation progress: %d total trials | %d resumed | %d remaining to compute",
        len(trials),
        len(already_scored),
        len(remaining_trials),
    )

    all_scores: dict[str, float] = dict(already_scored)

    def handle_batch_complete(batch_scores: list[tuple[str, float]], flush: bool) -> None:
        for sid, score in batch_scores:
            all_scores[sid] = score
        if flush:
            save_progress_checkpoint(
                progress_path=progress_path,
                contract=resume_contract,
                scores=list(all_scores.items()),
                status="INCOMPLETE",
            )

    if remaining_trials:
        logger.info(
            "Beginning evaluation on %d remaining trials (batch_size=%d, checkpoint_every=%d)...",
            len(remaining_trials),
            args.batch_size,
            args.checkpoint_every,
        )
        try:
            evaluate_protocol_dataset(
                detector=detector,
                trials=remaining_trials,
                audio_dir=audio_dir,
                batch_size=args.batch_size,
                checkpoint_every=args.checkpoint_every,
                on_batch_complete=handle_batch_complete,
            )
        except KeyboardInterrupt:
            sys.stderr.write(
                "HALTING: Evaluation interrupted by user (SIGINT). Persisting completed batches...\n"
            )
            try:
                save_progress_checkpoint(
                    progress_path=progress_path,
                    contract=resume_contract,
                    scores=list(all_scores.items()),
                    status="INCOMPLETE",
                )
            except Exception as persist_exc:
                sys.stderr.write(
                    "HALTING: Best-effort checkpoint persistence failed after SIGINT: "
                    f"{persist_exc}\n"
                )
            return 130
        except Exception as exc:
            sys.stderr.write(
                f"HALTING: Evaluation interrupted due to error ({exc}). Persisting completed batches...\n"
            )
            try:
                save_progress_checkpoint(
                    progress_path=progress_path,
                    contract=resume_contract,
                    scores=list(all_scores.items()),
                    status="INCOMPLETE",
                )
            except Exception as persist_exc:
                sys.stderr.write(
                    f"HALTING: Best-effort checkpoint persistence failed after original error ({exc}): "
                    f"{persist_exc}\n"
                )
            return 1

    # 7. Strict completeness validation
    missing_ids = valid_protocol_ids - set(all_scores.keys())
    if missing_ids:
        sys.stderr.write(
            f"HALTING: Incomplete evaluation. {len(missing_ids)} official trials are missing scores. "
            f"First missing IDs: {sorted(missing_ids)[:5]}. Output marked incomplete.\n"
        )
        return 1

    extra_ids = set(all_scores.keys()) - valid_protocol_ids
    if extra_ids:
        sys.stderr.write(
            f"HALTING: Illegal non-protocol samples were scored ({len(extra_ids)} extras). "
            f"First extra IDs: {sorted(extra_ids)[:5]}.\n"
        )
        return 1

    expected_split_counts = OFFICIAL_SPLIT_COUNTS.get(args.split)
    if strict_official and expected_split_counts:
        if len(all_scores) != expected_split_counts["total"]:
            sys.stderr.write(
                f"HALTING: Strict official {args.split} requires exactly {expected_split_counts['total']} scores, "
                f"got {len(all_scores)}. Refusing final metrics and report.\n"
            )
            return 1

    # Assemble canonical scores for metric calculation
    canonical_scores_dict: dict[str, float] = {}
    y_true_list: list[int] = []
    scores_bonafide_list: list[float] = []

    for _, sample_id, _, label in trials:
        score = all_scores[sample_id]
        canonical_scores_dict[sample_id] = score
        scores_bonafide_list.append(score)
        y_true_list.append(1 if label == "spoof" else 0)

    y_true = np.array(y_true_list, dtype=int)
    scores_bonafide = np.array(scores_bonafide_list, dtype=np.float64)

    # 8. Compute final benchmark metrics
    scores_spoof = -scores_bonafide

    eer_val, eer_thresh = compute_eer_interpolated(y_true, scores_spoof)
    diagnostic_min_dcf, diagnostic_min_dcf_thresh = compute_min_dcf(
        y_true, scores_spoof, c_miss=1.0, c_fa=10.0, p_target=0.05
    )

    actual_eer = float(eer_val) if eer_val is not None else None
    actual_diagnostic_min_dcf = (
        float(diagnostic_min_dcf) if diagnostic_min_dcf is not None else None
    )

    official_cardinality = args.split == "eval" and all(
        coverage[key] == expected for key, expected in OFFICIAL_EVAL_COUNTS.items()
    )
    status_eer, diff_eer = classify_eer_comparison(
        actual_eer,
        split=args.split,
        official_cardinality=official_cardinality,
    )

    # 9. Render report from template
    template_path = Path("reports/templates/asvspoof_reproduction_report_template.md")
    template_content = template_path.read_text(encoding="utf-8")
    timestamp_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    report_replacements = {
        "{{ actual_eer_pct }}": f"{actual_eer:.2%}" if actual_eer is not None else "NOT COMPUTED",
        "{{ status_eer }}": status_eer,
        "{{ diff_eer }}": f"{diff_eer:+.4f}" if diff_eer is not None else "N/A",
        "{{ actual_mindcf }}": f"{actual_diagnostic_min_dcf:.4f}"
        if actual_diagnostic_min_dcf is not None
        else "NOT COMPUTED",
        "{{ status_mindcf }}": "NOT COMPUTED",
        "{{ diff_mindcf }}": "N/A",
        "{{ evaluated_protocol }}": f"ASVspoof 2019 LA ({args.split})",
        "{{ execution_timestamp }}": timestamp_str,
        "{{ compute_device }}": str(detector.device),
        "{{ requested_device }}": str(args.device or os.getenv("VOICESHIELD_DEVICE", "mps")),
        "{{ cuda_available }}": str(dev_status.cuda_available),
        "{{ cuda_device_name }}": str(dev_status.cuda_device_name or "N/A"),
        "{{ mps_available }}": str(dev_status.mps_available),
        "{{ seed }}": str(args.seed),
        "{{ deterministic_algorithms }}": str(
            provenance_deterministic.get("deterministic_algorithms", False)
        ),
        "{{ os_version }}": platform.platform(),
        "{{ python_version }}": platform.python_version(),
        "{{ torch_version }}": torch.__version__,
        "{{ git_commit }}": git_commit,
        "{{ git_dirty }}": str(git_dirty),
        "{{ model_id }}": detector.provenance.model_id,
        "{{ provenance_status }}": detector.provenance.provenance_status,
        "{{ expected_checkpoint_sha256 }}": reg.checkpoint_sha256,
        "{{ checkpoint_registry_url }}": reg.checkpoint_url,
        "{{ checkpoint_sha256 }}": checkpoint_sha,
        "{{ upstream_commit }}": reg.upstream_commit,
        "{{ parameter_count }}": str(reg.parameter_count),
        "{{ protocol_sha256 }}": protocol_sha,
        "{{ config_sha256 }}": f"reproduction_parameters:{config_hash}",
        "{{ batch_size }}": str(args.batch_size),
        "{{ selected_split }}": args.split,
        "{{ smoke_memory_mb }}": "NOT COMPUTED",
        "{{ total_eval_trials }}": str(len(trials)),
        "{{ bonafide_eval_trials }}": str(int(np.sum(y_true == 0))),
        "{{ spoof_eval_trials }}": str(int(np.sum(y_true == 1))),
        "{{ protocol_id_count }}": str(coverage["protocol_ids"]),
        "{{ audio_file_count }}": str(coverage["audio_files"]),
        "{{ missing_audio_count }}": str(coverage["missing_files"]),
        "{{ extra_audio_count }}": str(coverage["extra_files"]),
        "{{ eer_threshold }}": format_optional_float(
            float(eer_thresh) if eer_thresh is not None else None
        ),
        "{{ mindcf_threshold }}": format_optional_float(
            float(diagnostic_min_dcf_thresh) if diagnostic_min_dcf_thresh is not None else None
        ),
        "{{ roc_auc }}": "N/A",
        "{{ pr_auc }}": "N/A",
        "{{ tp }}": "N/A",
        "{{ fp }}": "N/A",
        "{{ tn }}": "N/A",
        "{{ fn }}": "N/A",
        "{{ fpr_at_90 }}": "N/A",
        "{{ fpr_at_95 }}": "N/A",
        "{{ fpr_at_99 }}": "N/A",
        "{{ reproduction_command }}": repro_cmd,
    }
    report_replacements.update(
        smoke_report_replacements(smoke_diagnostics, skipped=bool(args.skip_smoke))
    )

    # 10. Atomic finalization
    try:
        raw_scores_file, report_file, artifact_hashes = finalize_evaluation(
            trials=trials,
            all_scores=all_scores,
            split=args.split,
            output_dir=args.output_dir,
            contract=resume_contract,
            progress_path=progress_path,
            report_template_content=template_content,
            report_replacements=report_replacements,
        )
    except Exception as exc:
        sys.stderr.write(f"HALTING: Finalization failed: {exc}\n")
        return 1

    logger.info(
        "Saved reproduction report to: %s (SHA-256: %s)",
        report_file,
        artifact_hashes["report_sha256"][:16],
    )
    logger.info(
        "Saved raw scores to: %s (SHA-256: %s)",
        raw_scores_file,
        artifact_hashes["raw_scores_sha256"][:16],
    )
    if status_eer in {"MATCHED (within tolerance)", "DISCREPANCY"}:
        logger.info(
            "Benchmark EER: %s (Official eval reference: %.2f%%; status: %s)",
            f"{actual_eer * 100:.2f}%" if actual_eer is not None else "NOT COMPUTED",
            OFFICIAL_AASIST_L_EER * 100,
            status_eer,
        )
    else:
        logger.info(
            "Benchmark EER: %s (official eval comparison: %s)",
            f"{actual_eer * 100:.2f}%" if actual_eer is not None else "NOT COMPUTED",
            status_eer,
        )
    logger.info(
        "VoiceShield diagnostic normalized minDCF: %s (not comparable with upstream min t-DCF %.4f)",
        actual_diagnostic_min_dcf if actual_diagnostic_min_dcf is not None else "NOT COMPUTED",
        OFFICIAL_AASIST_L_MIN_TDCF,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
