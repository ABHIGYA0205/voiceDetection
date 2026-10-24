"""Reproducible verification and benchmark evaluation for official AASIST-L on ASVspoof.

Enforces strict reproduction constraints:
- Halts immediately if checkpoint or dataset files are missing.
- Verifies directory structure and provides an actionable setup checklist.
- When present:
  1. Runs a deterministic stratified smoke subset first.
  2. Verifies label mapping and score direction against the official protocol.
  3. Verifies preprocessing (16 kHz, 64,600 samples cyclic pad) and score format.
  4. Runs the approved evaluation split.
  5. Computes linear-interpolated EER plus a separate, non-comparable diagnostic minDCF.
  6. Compares EER with the official eval reference only for the canonical eval protocol;
     min t-DCF is not computed.
  7. Captures complete provenance, hashes, command, and generated report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import re
import shlex
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from voiceshield.evaluation.metrics import (
    compute_eer_interpolated,
    compute_min_dcf,
)
from voiceshield.models import (
    AASISTLDetector,
    get_official_registry_entry,
    pad_for_aasist,
)

logger = logging.getLogger("voiceshield.reproduction")

# Official Clova AI published benchmarks for AASIST-L on ASVspoof 2019 LA Eval
OFFICIAL_AASIST_L_EER = 0.0099  # 0.99%
OFFICIAL_AASIST_L_MIN_TDCF = 0.0309
OFFICIAL_EVAL_COUNTS = {"total": 71237, "bonafide": 7355, "spoof": 63882}

# Default expected paths
DEFAULT_CHECKPOINT = Path("models/weights/AASIST-L.pth")
DEFAULT_DATASET_ROOT = Path("data/ASVspoof2019/LA")


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
    """Render a shell-safe command that preserves arguments containing spaces."""
    return shlex.join([executable, *argv])


def format_optional_float(value: float | None, *, precision: int = 4) -> str:
    """Render a finite metric value without treating zero as unavailable."""
    return f"{value:.{precision}f}" if value is not None else "NOT COMPUTED"


def render_report_template(template: str, replacements: dict[str, str]) -> str:
    """Render a report and fail closed if any template placeholder remains."""
    rendered = template
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    unresolved = sorted(set(re.findall(r"\{\{[^{}]+\}\}", rendered)))
    if unresolved:
        raise ValueError(f"Unresolved report template placeholders: {unresolved}")
    return rendered


def smoke_report_replacements(
    smoke_diagnostics: dict[str, Any], *, skipped: bool
) -> dict[str, str]:
    """Build explicit report fields for completed or intentionally skipped smoke tests."""
    if skipped:
        return {
            "{{ smoke_samples_count }}": "0",
            "{{ smoke_bonafide_count }}": "0",
            "{{ smoke_spoof_count }}": "0",
            "{{ smoke_status }}": "SKIPPED",
            "{{ smoke_mean_bonafide_score }}": "NOT COMPUTED",
            "{{ smoke_mean_spoof_score }}": "NOT COMPUTED",
            "{{ smoke_direction_aligned }}": "NOT COMPUTED",
        }
    return {
        "{{ smoke_samples_count }}": str(smoke_diagnostics["samples_evaluated"]),
        "{{ smoke_bonafide_count }}": str(smoke_diagnostics["bonafide_samples"]),
        "{{ smoke_spoof_count }}": str(smoke_diagnostics["spoof_samples"]),
        "{{ smoke_status }}": "PASSED",
        "{{ smoke_mean_bonafide_score }}": str(smoke_diagnostics["mean_bonafide_score"]),
        "{{ smoke_mean_spoof_score }}": str(smoke_diagnostics["mean_spoof_score"]),
        "{{ smoke_direction_aligned }}": str(smoke_diagnostics["direction_aligned"]),
    }


def validate_protocol_cardinality(
    trials: list[tuple[str, str, str, str]],
    *,
    strict_official: bool = False,
) -> dict[str, int]:
    """Validate class counts and reject duplicate protocol sample IDs in linear time."""
    sample_id_counts = Counter(entry[1] for entry in trials)
    duplicates = sorted(sample_id for sample_id, count in sample_id_counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate protocol sample IDs: {duplicates[:5]}")
    counts = {
        "total": len(trials),
        "bonafide": sum(label == "bonafide" for *_, label in trials),
        "spoof": sum(label == "spoof" for *_, label in trials),
    }
    if counts["bonafide"] + counts["spoof"] != counts["total"]:
        raise ValueError(f"Unsupported protocol labels or count mismatch: {counts}")
    if strict_official and counts != OFFICIAL_EVAL_COUNTS:
        raise ValueError(
            "Official ASVspoof 2019 LA eval cardinality mismatch: "
            f"expected {OFFICIAL_EVAL_COUNTS}, got {counts}."
        )
    return counts


def validate_protocol_audio_coverage(
    trials: list[tuple[str, str, str, str]],
    audio_dir: Path,
    *,
    strict_official: bool = False,
) -> dict[str, Any]:
    """Validate CM protocol IDs against FLAC stems without scoring extras."""
    counts = validate_protocol_cardinality(trials, strict_official=strict_official)
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

        out = detector.predict(samples)
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


def evaluate_protocol_dataset(
    detector: AASISTLDetector,
    trials: list[tuple[str, str, str, str]],
    audio_dir: Path,
    batch_size: int = 24,
) -> tuple[
    np.ndarray[Any, np.dtype[np.int_]],
    np.ndarray[Any, np.dtype[np.float64]],
    dict[str, float],
]:
    """Evaluate full ASVspoof protocol trials using batched inference."""
    all_labels: list[int] = []
    all_scores: list[float] = []
    scores_dict: dict[str, float] = {}

    total = len(trials)
    device = detector.device

    for start_idx in range(0, total, batch_size):
        batch_trials = trials[start_idx : start_idx + batch_size]
        batch_tensors: list[torch.Tensor] = []
        batch_labels: list[int] = []
        batch_ids: list[str] = []

        for _, sample_id, _, label in batch_trials:
            flac_path = audio_dir / f"{sample_id}.flac"
            audio_samples, _ = sf.read(str(flac_path), dtype="float32")
            # Enforce exact 64,600-sample framing
            padded_tensor = pad_for_aasist(audio_samples, max_len=64600)
            batch_tensors.append(padded_tensor)
            batch_labels.append(1 if label == "spoof" else 0)
            batch_ids.append(sample_id)

        stacked = torch.stack(batch_tensors, dim=0).to(device)
        with torch.no_grad():
            _, outputs = detector.model(stacked)
            # outputs shape: (B, 2) -> index 0: spoof, index 1: bonafide
            # ASVspoof standard bona fide score: output[:, 1]
            bf_scores = outputs[:, 1].detach().cpu().numpy()

        for sid, score, y in zip(batch_ids, bf_scores, batch_labels, strict=False):
            scores_dict[sid] = float(score)
            all_scores.append(float(score))
            all_labels.append(y)

        if (start_idx // batch_size) % 50 == 0 or (start_idx + batch_size) >= total:
            logger.info("Processed %d / %d trials...", min(start_idx + batch_size, total), total)

    return np.array(all_labels, dtype=int), np.array(all_scores, dtype=np.float64), scores_dict


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
        choices=["mps", "cpu"],
        help="Preferred compute device ('mps' or 'cpu').",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=24,
        help="Inference batch size matching official AASIST config (default: 24).",
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


def main(argv: list[str] | None = None) -> int:
    """Main reproduction entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if not args.skip_smoke and args.smoke_count < 2:
        parser.error("--smoke-count must be at least 2 unless --skip-smoke is used")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # 1. Enforce official model registry integrity on checkpoint BEFORE dataset access
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

    # 2. Verify directory structure & prerequisites
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
    strict_official = args.split == "eval"
    try:
        coverage = validate_protocol_audio_coverage(
            trials, audio_dir, strict_official=strict_official
        )
    except (FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"HALTING: Protocol/audio validation failed: {exc}\n")
        return 1
    logger.info("Protocol/audio coverage: %s", coverage)
    if args.check_only:
        print(f"Protocol/audio validation passed: {coverage}")
        return 0

    logger.info("AASIST-L Checkpoint: %s (SHA-256: %s)", args.checkpoint, checkpoint_sha[:16])
    logger.info("Protocol File: %s (SHA-256: %s)", protocol_file, protocol_sha[:16])

    # 3. Load detector with explicit official identity
    detector = AASISTLDetector(
        checkpoint_path=args.checkpoint,
        model_id="clovaai/aasist-l",
        preferred_device=args.device,
    )
    logger.info("AASIST-L initialized on device: %s", detector.device)

    logger.info("Loaded %d trials from protocol %s", len(trials), protocol_file.name)

    # 5. Smoke test subset
    smoke_diagnostics: dict[str, Any] = {}
    if not args.skip_smoke:
        logger.info("Running initial %d-sample smoke subset...", args.smoke_count)
        try:
            smoke_diagnostics = run_smoke_test(
                detector=detector,
                trials=trials,
                audio_dir=audio_dir,
                smoke_count=args.smoke_count,
            )
        except (FileNotFoundError, ValueError) as exc:
            sys.stderr.write(f"HALTING: Smoke validation failed: {exc}\n")
            return 1
        logger.info(
            "Smoke complete. Mean Bonafide: %.4f | Mean Spoof: %.4f | Direction Aligned: %s",
            smoke_diagnostics["mean_bonafide_score"],
            smoke_diagnostics["mean_spoof_score"],
            smoke_diagnostics["direction_aligned"],
        )
        if not smoke_diagnostics["direction_aligned"]:
            sys.stderr.write("ERROR: Label direction test failed in smoke subset! Halting.\n")
            return 1
    else:
        logger.warning("Smoke verification explicitly skipped; smoke metrics will be NOT COMPUTED.")

    # 6. Full split evaluation
    logger.info(
        "Beginning full evaluation on %d trials (batch_size=%d)...", len(trials), args.batch_size
    )
    y_true, scores_bonafide, scores_dict = evaluate_protocol_dataset(
        detector=detector,
        trials=trials,
        audio_dir=audio_dir,
        batch_size=args.batch_size,
    )

    # ASVspoof raw scores are bona fide logits (higher = bona fide).
    # Negation supplies the higher-is-spoof convention required by the local EER
    # and separate diagnostic normalized-minDCF helpers.
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
    git_commit, git_dirty = get_git_provenance()
    reproduction_parameters = {
        "batch_size": args.batch_size,
        "checkpoint_registry_id": reg.model_id,
        "checkpoint_sha256": checkpoint_sha,
        "evaluation_framing": "pad_for_aasist: 64600 samples, cyclic pad/truncate",
        "model_config": detector.config,
        "protocol_sha256": protocol_sha,
        "resolved_device": str(detector.device),
        "score_direction": "raw_bonafide_score=logit[1]; metric_spoof_score=-logit[1]",
        "selected_split": args.split,
        "skip_smoke": bool(args.skip_smoke),
        "smoke_count": args.smoke_count,
        "upstream_commit": reg.upstream_commit,
    }
    reproduction_hash = hashlib.sha256(
        json.dumps(reproduction_parameters, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    # 7. Render report from template
    template_path = Path("reports/templates/asvspoof_reproduction_report_template.md")
    template_content = template_path.read_text(encoding="utf-8")

    timestamp_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    repro_cmd = format_invoked_command(sys.executable, sys.argv)

    replacements = {
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
        "{{ mps_available }}": str(torch.backends.mps.is_available()),
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
        "{{ config_sha256 }}": f"reproduction_parameters:{reproduction_hash}",
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
        "{{ raw_scores_path }}": str(args.output_dir / "raw_scores.txt"),
        "{{ report_path }}": str(args.output_dir / "report.md"),
    }

    replacements.update(smoke_report_replacements(smoke_diagnostics, skipped=bool(args.skip_smoke)))
    try:
        rendered_report = render_report_template(template_content, replacements)
    except ValueError as exc:
        sys.stderr.write(f"HALTING: Report rendering failed: {exc}\n")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_file = args.output_dir / "report.md"
    report_file.write_text(rendered_report, encoding="utf-8")

    # Save raw scores in Clova AI format: <sample_id> <score>
    raw_scores_file = args.output_dir / "raw_scores.txt"
    with open(raw_scores_file, "w", encoding="utf-8") as f:
        for sid, score in scores_dict.items():
            f.write(f"{sid} {score:.6f}\n")

    logger.info("Saved reproduction report to: %s", report_file)
    logger.info("Saved raw scores to: %s", raw_scores_file)
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
