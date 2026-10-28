"""Tests for resumable official ASVspoof evaluation, fail-closed guards, and device handling.

Invokes production helpers and evaluator paths directly, verifying:
- Resume contract hash & comprehensive metadata validation
- Exact official cardinalities for both eval (71,237) and dev (24,844: 2,548 bonafide, 22,296 spoof)
- Atomic progress checkpointing and crash-safe backup restoration (preserving valid backup)
- Batch audio validation (16 kHz, mono, finite samples) and finite model outputs
- Atomic final artifact publication with artifact hash logging and stale collision rejection
- Production smoke payload containing no 0.99% or 0.0099 reference
- Strict parser rejection of invalid arguments and incompatible modes
- Stale collision rejection for non-resume runs
- Tampered contract detection
- Interruption checkpointing and missing-only resumed inference
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
import torch

import scripts.reproduce_asvspoof_eval as repro
from scripts.reproduce_asvspoof_eval import (
    OFFICIAL_DEV_COUNTS,
    OFFICIAL_EVAL_COUNTS,
    OFFICIAL_SPLIT_COUNTS,
    build_parser,
    build_resume_contract,
    compute_file_sha256,
    evaluate_protocol_dataset,
    finalize_evaluation,
    load_and_validate_progress,
    main,
    save_progress_checkpoint,
    validate_cli_arguments,
    validate_protocol_audio_coverage,
    validate_protocol_cardinality,
)
from voiceshield.reproducibility import set_seed


def _make_dummy_trials(count: int = 4) -> list[tuple[str, str, str, str]]:
    trials: list[tuple[str, str, str, str]] = []
    for i in range(count):
        label = "bonafide" if i % 2 == 0 else "spoof"
        attack = "-" if label == "bonafide" else f"A{i:02d}"
        trials.append((f"spk_{i:04d}", f"LA_E_{1000000 + i}", attack, label))
    return trials


def _make_dummy_contract(**overrides: Any) -> dict[str, Any]:
    return build_resume_contract(
        seed=overrides.get("seed", 42),
        git_commit=overrides.get("git_commit", "2d37e2bde103dedb10a8e4a9d7a687fb1b91ef4f"),
        git_dirty=overrides.get("git_dirty", False),
        resolved_device=overrides.get("resolved_device", "cpu"),
        gpu_identity=overrides.get("gpu_identity", None),
        gpu_capability=overrides.get("gpu_capability", None),
        torch_version=overrides.get("torch_version", "2.6.0"),
        cuda_version=overrides.get("cuda_version", None),
        cudnn_version=overrides.get("cudnn_version", None),
        determinism=overrides.get(
            "determinism",
            {
                "cublas_workspace_config": ":4096:8",
                "cuda_allow_tf32": False,
                "cudnn_allow_tf32": False,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "deterministic_algorithms": True,
            },
        ),
        protocol_sha256=overrides.get(
            "protocol_sha256",
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        ),
        checkpoint_sha256=overrides.get(
            "checkpoint_sha256",
            "814331d088032bb4c3fa61cc014789eadeed464209dd094ab3a2dd6ffbdce27a",
        ),
        model_id=overrides.get("model_id", "clovaai/aasist-l"),
        config_hash=overrides.get(
            "config_hash", "reproduction_parameters:test_config_hash_1234567890abcdef"
        ),
        upstream_commit=overrides.get("upstream_commit", "a04c986"),
        preprocessing_contract=overrides.get(
            "preprocessing_contract",
            "pad_for_aasist: 64600 samples, cyclic pad/truncate",
        ),
        selected_split=overrides.get("selected_split", "eval"),
        batch_size=overrides.get("batch_size", 24),
        score_direction=overrides.get(
            "score_direction",
            "raw_bonafide_score=logit[1]; metric_spoof_score=-logit[1]",
        ),
    )


# 1. Exact official cardinalities for both eval and dev
def test_official_cardinalities_for_eval_and_dev() -> None:
    assert OFFICIAL_EVAL_COUNTS == {"total": 71237, "bonafide": 7355, "spoof": 63882}
    assert OFFICIAL_DEV_COUNTS == {"total": 24844, "bonafide": 2548, "spoof": 22296}
    assert OFFICIAL_SPLIT_COUNTS["eval"] == OFFICIAL_EVAL_COUNTS
    assert OFFICIAL_SPLIT_COUNTS["dev"] == OFFICIAL_DEV_COUNTS

    # Eval mismatch
    with pytest.raises(ValueError, match="do not match official ASVspoof 2019 LA partition"):
        validate_protocol_cardinality(_make_dummy_trials(4), split="eval", strict_official=True)

    # Dev mismatch
    with pytest.raises(ValueError, match="do not match official ASVspoof 2019 LA partition"):
        validate_protocol_cardinality(_make_dummy_trials(4), split="dev", strict_official=True)


# 2. Strict parser rejection tests for invalid arguments and incompatible modes
def test_parser_rejection_invalid_numeric_arguments() -> None:
    parser = build_parser()

    # batch-size < 1
    args = parser.parse_args(["--batch-size", "0"])
    with pytest.raises(SystemExit):
        validate_cli_arguments(parser, args)

    # checkpoint-every < 1
    args = parser.parse_args(["--checkpoint-every", "0"])
    with pytest.raises(SystemExit):
        validate_cli_arguments(parser, args)

    # smoke-count < 2
    args = parser.parse_args(["--smoke-count", "1"])
    with pytest.raises(SystemExit):
        validate_cli_arguments(parser, args)


def test_parser_rejection_incompatible_modes() -> None:
    parser = build_parser()

    # smoke-only with skip-smoke
    args = parser.parse_args(["--smoke-only", "--skip-smoke"])
    with pytest.raises(SystemExit):
        validate_cli_arguments(parser, args)

    # smoke-only with resume
    args = parser.parse_args(["--smoke-only", "--resume"])
    with pytest.raises(SystemExit):
        validate_cli_arguments(parser, args)

    # smoke-only with check-only
    args = parser.parse_args(["--smoke-only", "--check-only"])
    with pytest.raises(SystemExit):
        validate_cli_arguments(parser, args)

    # check-only with resume
    args = parser.parse_args(["--check-only", "--resume"])
    with pytest.raises(SystemExit):
        validate_cli_arguments(parser, args)

    # invalid choices
    with pytest.raises(SystemExit):
        parser.parse_args(["--split", "test"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--device", "tpu"])


# 3. Resume contract object and hash validation
def test_resume_contract_hash_and_mismatch_refusal(tmp_path: Path) -> None:
    trials = _make_dummy_trials(2)
    valid_ids = {t[1] for t in trials}
    expected_contract = _make_dummy_contract()
    assert "contract_hash" in expected_contract

    # Change seed -> should reject
    corrupt_contract = _make_dummy_contract(seed=999)
    progress_file = tmp_path / "eval_progress.json"
    save_progress_checkpoint(
        progress_file,
        corrupt_contract,
        [(trials[0][1], 1.0)],
        status="INCOMPLETE",
    )

    with pytest.raises(ValueError, match="Resume rejected due to contract mismatch"):
        load_and_validate_progress(progress_file, expected_contract, valid_ids)


# 4. Progress schema, root type, status, completed_count, and finite values validation
def test_progress_schema_validation(tmp_path: Path) -> None:
    contract = _make_dummy_contract()
    progress_file = tmp_path / "eval_progress.json"

    # Non-dict root
    progress_file.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object"):
        load_and_validate_progress(progress_file, contract, {"ID1"})

    # Bad schema version
    bad_schema = {
        "schema_version": "2.0",
        "status": "INCOMPLETE",
        "completed_count": 0,
        "scores": [],
        "contract": contract,
        "contract_hash": contract["contract_hash"],
    }
    progress_file.write_text(json.dumps(bad_schema), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported progress schema version"):
        load_and_validate_progress(progress_file, contract, {"ID1"})

    # Bad status
    bad_status = {
        "schema_version": "1.0",
        "status": "UNKNOWN",
        "completed_count": 0,
        "scores": [],
        "contract": contract,
        "contract_hash": contract["contract_hash"],
    }
    progress_file.write_text(json.dumps(bad_status), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid progress checkpoint status"):
        load_and_validate_progress(progress_file, contract, {"ID1"})

    # completed_count mismatch
    bad_count = {
        "schema_version": "1.0",
        "status": "INCOMPLETE",
        "completed_count": 5,
        "scores": [["ID1", 1.0]],
        "contract": contract,
        "contract_hash": contract["contract_hash"],
    }
    progress_file.write_text(json.dumps(bad_count), encoding="utf-8")
    with pytest.raises(ValueError, match="!= scores length"):
        load_and_validate_progress(progress_file, contract, {"ID1"})

    # Non-finite score
    bad_score = {
        "schema_version": "1.0",
        "status": "INCOMPLETE",
        "completed_count": 1,
        "scores": [["ID1", "NaN"]],
        "contract": contract,
        "contract_hash": contract["contract_hash"],
    }
    progress_file.write_text(json.dumps(bad_score), encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed non-numeric score"):
        load_and_validate_progress(progress_file, contract, {"ID1"})


# 5. Backup restoration preserves the valid backup instead of rotating corrupt primary over it
def test_backup_restoration_preserves_valid_backup(tmp_path: Path) -> None:
    trials = _make_dummy_trials(2)
    valid_ids = {t[1] for t in trials}
    contract = _make_dummy_contract()
    progress_file = tmp_path / "eval_progress.json"
    bak_file = tmp_path / "eval_progress.json.bak"

    # 1. Save initial batch 1
    save_progress_checkpoint(progress_file, contract, [(trials[0][1], 1.25)], status="INCOMPLETE")
    # 2. Save batch 2 -> copies batch 1 to bak_file
    save_progress_checkpoint(
        progress_file, contract, [(trials[0][1], 1.25), (trials[1][1], -0.5)], status="INCOMPLETE"
    )

    assert bak_file.is_file()
    valid_backup_content = bak_file.read_text(encoding="utf-8")
    assert '"completed_count": 1' in valid_backup_content

    # 3. Corrupt primary file
    progress_file.write_text('{"schema_version": "1.0", TRUNCATED', encoding="utf-8")

    # 4. Attempt to save or recover: loading recovers from backup
    recovered = load_and_validate_progress(progress_file, contract, valid_ids)
    assert len(recovered.scores) == 1
    assert trials[0][1] in recovered.scores

    # 5. Crucial: backup file MUST still contain valid backup, NOT the corrupt truncated primary!
    assert bak_file.is_file()
    current_bak = bak_file.read_text(encoding="utf-8")
    assert "TRUNCATED" not in current_bak
    parsed_bak = json.loads(current_bak)
    assert parsed_bak["completed_count"] == 1


# 6. Audio batch validation (16 kHz, mono, finite audio, finite model outputs)
def test_batch_validation_sample_rate_and_channels(tmp_path: Path) -> None:
    trials = _make_dummy_trials(2)
    audio_dir = tmp_path / "flac"
    audio_dir.mkdir()

    # Create wrong sample rate audio (8000 Hz)
    bad_sr_path = audio_dir / f"{trials[0][1]}.flac"
    sf.write(str(bad_sr_path), np.zeros(8000, dtype=np.float32), 8000)

    good_path = audio_dir / f"{trials[1][1]}.flac"
    sf.write(str(good_path), np.zeros(16000, dtype=np.float32), 16000)

    detector = MagicMock()
    detector.device = torch.device("cpu")

    with pytest.raises(ValueError, match="sample rate is 8000 Hz; expected 16000 Hz"):
        evaluate_protocol_dataset(detector, trials, audio_dir, batch_size=2)

    # Create stereo audio
    stereo_path = audio_dir / f"{trials[0][1]}.flac"
    sf.write(str(stereo_path), np.zeros((16000, 2), dtype=np.float32), 16000)
    with pytest.raises(ValueError, match="is not mono"):
        evaluate_protocol_dataset(detector, trials, audio_dir, batch_size=2)


def test_batch_validation_non_finite_audio_and_model_outputs(tmp_path: Path) -> None:
    trials = _make_dummy_trials(1)
    audio_dir = tmp_path / "flac"
    audio_dir.mkdir()

    valid_path = audio_dir / f"{trials[0][1]}.flac"
    sf.write(str(valid_path), np.zeros(16000, dtype=np.float32), 16000)

    detector = MagicMock()
    detector.device = torch.device("cpu")
    detector.model.return_value = (None, torch.tensor([[0.0, 1.0]]))

    # 1. Non-finite audio samples
    nan_audio = np.zeros(16000, dtype=np.float32)
    nan_audio[100] = np.nan
    with patch("soundfile.read", return_value=(nan_audio, 16000)):
        with pytest.raises(ValueError, match="contains non-finite"):
            evaluate_protocol_dataset(detector, trials, audio_dir, batch_size=1)

    # 2. Valid audio, but model outputs non-finite logits
    detector.model.return_value = (None, torch.tensor([[float("nan"), 1.0]]))
    with pytest.raises(RuntimeError, match="Detector produced non-finite"):
        evaluate_protocol_dataset(detector, trials, audio_dir, batch_size=1)


# 7. Production smoke-only output contains NO 0.99% or 0.0099 references
def test_production_smoke_only_contains_no_reference_eer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(repro, "OFFICIAL_EVAL_COUNTS", {"total": 2, "bonafide": 1, "spoof": 1})
    output_dir = tmp_path / "smoke_output"
    checkpoint_file = tmp_path / "AASIST-L.pth"
    checkpoint_file.write_bytes(b"dummy_weights")
    dataset_root = tmp_path / "data" / "ASVspoof2019" / "LA"
    proto_dir = dataset_root / "ASVspoof2019_LA_cm_protocols"
    proto_dir.mkdir(parents=True)
    proto_file = proto_dir / "ASVspoof2019.LA.cm.eval.trl.txt"
    proto_file.write_text("LA_0070 LA_E_1001 - - bonafide\nLA_0070 LA_E_1002 A07 - spoof\n")
    flac_dir = dataset_root / "ASVspoof2019_LA_eval" / "flac"
    flac_dir.mkdir(parents=True)
    sf.write(str(flac_dir / "LA_E_1001.flac"), np.zeros(16000, dtype=np.float32), 16000)
    sf.write(str(flac_dir / "LA_E_1002.flac"), np.zeros(16000, dtype=np.float32), 16000)

    reg_mock = MagicMock()
    reg_mock.checkpoint_sha256 = repro.compute_file_sha256(checkpoint_file)
    reg_mock.model_id = "clovaai/aasist-l"
    reg_mock.upstream_commit = "a04c986"

    mock_detector = MagicMock()
    mock_detector.device = torch.device("cpu")
    mock_detector.config = {"architecture": "AASIST-L", "nb_samp": 64600}
    mock_detector.provenance = MagicMock(
        model_id="clovaai/aasist-l",
        provenance_status="OFFICIAL_UNMODIFIED",
    )
    mock_detector.predict.side_effect = [
        MagicMock(bonafide_score=1.5),
        MagicMock(bonafide_score=-1.5),
    ]

    with (
        patch(
            "scripts.reproduce_asvspoof_eval.get_git_provenance",
            return_value=("2d37e2bde103dedb10a8e4a9d7a687fb1b91ef4f", False),
        ),
        patch("scripts.reproduce_asvspoof_eval.get_official_registry_entry", return_value=reg_mock),
        patch("scripts.reproduce_asvspoof_eval.AASISTLDetector", return_value=mock_detector),
        patch(
            "sys.argv",
            [
                "reproduce_asvspoof_eval.py",
                "--smoke-only",
                "--smoke-count",
                "2",
                "--checkpoint",
                str(checkpoint_file),
                "--dataset-root",
                str(dataset_root),
                "--output-dir",
                str(output_dir),
            ],
        ),
    ):
        ret = main()
        assert ret == 0

    diagnostic_file = output_dir / "smoke_diagnostic.json"
    assert diagnostic_file.is_file()
    raw_content = diagnostic_file.read_text(encoding="utf-8")

    # Strictly verify that 0.99 or 0.0099 is NOT present anywhere in the production smoke artifact
    assert "0.99" not in raw_content
    assert "0.0099" not in raw_content
    assert "NOT COMPARABLE" in raw_content


# 8. Extra FLAC files in eval are permitted but never scored
def test_extra_flac_files_permitted_but_not_scored(tmp_path: Path) -> None:
    trials = _make_dummy_trials(2)
    audio_dir = tmp_path / "flac"
    audio_dir.mkdir()
    for _, sid, _, _ in trials:
        sf.write(str(audio_dir / f"{sid}.flac"), np.zeros(16000, dtype=np.float32), 16000)

    # Add extra FLAC file not in protocol
    sf.write(str(audio_dir / "LA_E_EXTRA999.flac"), np.zeros(16000, dtype=np.float32), 16000)

    coverage = validate_protocol_audio_coverage(
        trials, audio_dir, split="eval", strict_official=False
    )
    assert coverage["extra_files"] == 1
    assert coverage["missing_files"] == 0
    assert coverage["protocol_ids"] == 2


# 9. Early stale progress, backup, report, raw-scores, or temporary artifact collision rejected
def test_early_stale_collision_rejected_non_resume(tmp_path: Path) -> None:
    for stale_name in [
        "eval_progress.json",
        "eval_progress.json.bak",
        "report.md",
        "raw_scores.txt",
        "stale_artifact.tmp",
        ".hidden_stale.tmp",
    ]:
        out_dir = tmp_path / f"stale_test_{stale_name.replace('.', '_')}"
        out_dir.mkdir(parents=True)
        (out_dir / stale_name).write_text("dummy", encoding="utf-8")
        with patch("sys.argv", ["reproduce_asvspoof_eval.py", "--output-dir", str(out_dir)]):
            ret = main()
            assert ret == 1, f"Expected non-resume run to fail closed on stale {stale_name}"


# 10. Tampered contract with unchanged claimed hash is strictly rejected
def test_tampered_contract_field_rejected_even_if_top_hash_unchanged(tmp_path: Path) -> None:
    trials = _make_dummy_trials(2)
    valid_ids = {t[1] for t in trials}
    contract = _make_dummy_contract()
    progress_file = tmp_path / "eval_progress.json"
    save_progress_checkpoint(progress_file, contract, [(trials[0][1], 1.0)], status="INCOMPLETE")

    # Mutate contract field in stored JSON without updating top-level contract_hash
    data = json.loads(progress_file.read_text(encoding="utf-8"))
    data["contract"]["seed"] = 99999
    progress_file.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="Tampered contract detected in checkpoint"):
        load_and_validate_progress(progress_file, contract, valid_ids)


# 11. save_progress_checkpoint strictly rejects invalid status, duplicate IDs, and non-finite scores
def test_save_progress_checkpoint_validation(tmp_path: Path) -> None:
    contract = _make_dummy_contract()
    p = tmp_path / "eval_progress.json"

    # 1. Unsupported status
    with pytest.raises(ValueError, match="Unsupported status"):
        save_progress_checkpoint(p, contract, [("ID1", 1.0)], status="UNKNOWN")

    # 2. Duplicate sample ID
    with pytest.raises(ValueError, match="Duplicate sample ID"):
        save_progress_checkpoint(p, contract, [("ID1", 1.0), ("ID1", 2.0)])

    # 3. Non-finite score (NaN)
    with pytest.raises(ValueError, match="Non-finite score"):
        save_progress_checkpoint(p, contract, [("ID1", float("nan"))])

    # 4. Non-finite score (Inf)
    with pytest.raises(ValueError, match="Non-finite score"):
        save_progress_checkpoint(p, contract, [("ID1", float("inf"))])

    # 5. Non-numeric score (string)
    with pytest.raises(ValueError, match="Malformed non-numeric score"):
        save_progress_checkpoint(p, contract, cast(Any, [("ID1", "1.0")]))

    # 6. Boolean score
    with pytest.raises(ValueError, match="Malformed non-numeric score"):
        save_progress_checkpoint(p, contract, cast(Any, [("ID1", True)]))

    # 7. Empty sample ID
    with pytest.raises(ValueError, match="Malformed sample ID"):
        save_progress_checkpoint(p, contract, [("", 1.0)])

    # 8. Malformed artifact hash
    with pytest.raises(ValueError, match="Invalid SHA-256 hash"):
        save_progress_checkpoint(
            p, contract, [("ID1", 1.0)], artifact_hashes={"report_sha256": "too_short"}
        )

    # 9. COMPLETED status missing required artifact hashes
    with pytest.raises(
        ValueError, match="COMPLETED progress must include report_sha256 and raw_scores_sha256"
    ):
        save_progress_checkpoint(p, contract, [("ID1", 1.0)], status="COMPLETED")


# 12. Backup write failure fails closed, leaving primary progress intact
def test_save_progress_checkpoint_fails_closed_on_backup_failure(tmp_path: Path) -> None:
    contract = _make_dummy_contract()
    p = tmp_path / "eval_progress.json"
    save_progress_checkpoint(p, contract, [("ID1", 1.0)], status="INCOMPLETE")
    original_content = p.read_text(encoding="utf-8")

    real_atomic_write = repro.atomic_write_text

    def fake_atomic_write(path: Path, content: str) -> None:
        if path.name.endswith(".bak"):
            raise OSError("Simulated backup disk full")
        real_atomic_write(path, content)

    with patch("scripts.reproduce_asvspoof_eval.atomic_write_text", side_effect=fake_atomic_write):
        with pytest.raises(RuntimeError, match="Fail-closed: backup rotation failed"):
            save_progress_checkpoint(p, contract, [("ID1", 1.0), ("ID2", 2.0)])

    # Primary progress file MUST remain untouched
    assert p.is_file()
    assert p.read_text(encoding="utf-8") == original_content


# 13. COMPLETED progress verification on --resume
def test_resume_completed_artifact_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(repro, "OFFICIAL_EVAL_COUNTS", {"total": 2, "bonafide": 1, "spoof": 1})
    monkeypatch.setattr(
        repro,
        "OFFICIAL_SPLIT_COUNTS",
        {
            "eval": {"total": 2, "bonafide": 1, "spoof": 1},
            "dev": {"total": 2, "bonafide": 1, "spoof": 1},
        },
    )

    out_dir = tmp_path / "completed_out"
    out_dir.mkdir()
    dataset_root = tmp_path / "data" / "ASVspoof2019" / "LA"
    proto_dir = dataset_root / "ASVspoof2019_LA_cm_protocols"
    proto_dir.mkdir(parents=True)
    (proto_dir / "ASVspoof2019.LA.cm.eval.trl.txt").write_text(
        "LA_0070 LA_E_1001 - - bonafide\nLA_0070 LA_E_1002 A07 - spoof\n"
    )
    flac_dir = dataset_root / "ASVspoof2019_LA_eval" / "flac"
    flac_dir.mkdir(parents=True)
    sf.write(str(flac_dir / "LA_E_1001.flac"), np.zeros(16000, dtype=np.float32), 16000)
    sf.write(str(flac_dir / "LA_E_1002.flac"), np.zeros(16000, dtype=np.float32), 16000)

    ckpt = tmp_path / "AASIST-L.pth"
    ckpt.write_bytes(b"dummy_weights")
    reg_mock = MagicMock()
    reg_mock.checkpoint_sha256 = compute_file_sha256(ckpt)
    reg_mock.model_id = "clovaai/aasist-l"
    reg_mock.upstream_commit = "a04c986"

    report_file = out_dir / "report.md"
    raw_scores_file = out_dir / "raw_scores.txt"
    report_file.write_text("# Report", encoding="utf-8")
    raw_scores_file.write_text("LA_E_1001 1.0\nLA_E_1002 -1.0\n", encoding="utf-8")

    rep_sha = compute_file_sha256(report_file)
    raw_sha = compute_file_sha256(raw_scores_file)

    mock_det = MagicMock()
    mock_det.device = torch.device("cpu")
    mock_det.config = {"architecture": "AASIST-L", "nb_samp": 64600}
    mock_det.provenance = MagicMock(
        model_id="clovaai/aasist-l",
        provenance_status="OFFICIAL_UNMODIFIED",
    )
    cfg_hash = hashlib.sha256(
        json.dumps(mock_det.config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    contract = build_resume_contract(
        seed=42,
        git_commit="2d37e2bde103dedb10a8e4a9d7a687fb1b91ef4f",
        git_dirty=False,
        resolved_device="cpu",
        gpu_identity=None,
        gpu_capability=None,
        torch_version=torch.__version__,
        cuda_version=None,
        cudnn_version=None,
        determinism={
            "cublas_workspace_config": ":4096:8",
            "cuda_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "deterministic_algorithms": True,
        },
        protocol_sha256=compute_file_sha256(proto_dir / "ASVspoof2019.LA.cm.eval.trl.txt"),
        checkpoint_sha256=reg_mock.checkpoint_sha256,
        model_id="clovaai/aasist-l",
        config_hash=cfg_hash,
        upstream_commit="a04c986",
        preprocessing_contract="pad_for_aasist: 64600 samples, cyclic pad/truncate",
        selected_split="eval",
        batch_size=24,
        score_direction="raw_bonafide_score=logit[1]; metric_spoof_score=-logit[1]",
    )

    progress_file = out_dir / "eval_progress.json"
    save_progress_checkpoint(
        progress_file,
        contract,
        [("LA_E_1001", 1.0), ("LA_E_1002", -1.0)],
        status="COMPLETED",
        artifact_hashes={"report_sha256": rep_sha, "raw_scores_sha256": raw_sha},
    )

    # 1. Successful resume of COMPLETED: exits 0 without recomputing
    with (
        patch("scripts.reproduce_asvspoof_eval.get_official_registry_entry", return_value=reg_mock),
        patch("scripts.reproduce_asvspoof_eval.AASISTLDetector", return_value=mock_det),
        patch(
            "scripts.reproduce_asvspoof_eval.get_git_provenance",
            return_value=("2d37e2bde103dedb10a8e4a9d7a687fb1b91ef4f", False),
        ),
        patch(
            "sys.argv",
            [
                "reproduce_asvspoof_eval.py",
                "--resume",
                "--checkpoint",
                str(ckpt),
                "--dataset-root",
                str(dataset_root),
                "--output-dir",
                str(out_dir),
            ],
        ),
    ):
        ret = main()
        assert ret == 0

    # 2. Reject missing report file
    report_file.unlink()
    with (
        patch("scripts.reproduce_asvspoof_eval.get_official_registry_entry", return_value=reg_mock),
        patch("scripts.reproduce_asvspoof_eval.AASISTLDetector", return_value=mock_det),
        patch(
            "scripts.reproduce_asvspoof_eval.get_git_provenance",
            return_value=("2d37e2bde103dedb10a8e4a9d7a687fb1b91ef4f", False),
        ),
        patch(
            "sys.argv",
            [
                "reproduce_asvspoof_eval.py",
                "--resume",
                "--checkpoint",
                str(ckpt),
                "--dataset-root",
                str(dataset_root),
                "--output-dir",
                str(out_dir),
            ],
        ),
    ):
        assert main() == 1

    # 3. Reject mismatched raw scores file
    report_file.write_text("# Report", encoding="utf-8")
    raw_scores_file.write_text("corrupted_scores", encoding="utf-8")
    with (
        patch("scripts.reproduce_asvspoof_eval.get_official_registry_entry", return_value=reg_mock),
        patch("scripts.reproduce_asvspoof_eval.AASISTLDetector", return_value=mock_det),
        patch(
            "scripts.reproduce_asvspoof_eval.get_git_provenance",
            return_value=("2d37e2bde103dedb10a8e4a9d7a687fb1b91ef4f", False),
        ),
        patch(
            "sys.argv",
            [
                "reproduce_asvspoof_eval.py",
                "--resume",
                "--checkpoint",
                str(ckpt),
                "--dataset-root",
                str(dataset_root),
                "--output-dir",
                str(out_dir),
            ],
        ),
    ):
        assert main() == 1


# 14. finalize_evaluation checks completeness and canonical publication order for eval and dev
def test_finalize_evaluation_eval_and_dev(tmp_path: Path) -> None:
    contract = _make_dummy_contract()
    template_path = Path("reports/templates/asvspoof_reproduction_report_template.md")
    template_content = template_path.read_text(encoding="utf-8")

    # 1. Incomplete eval refuses publication
    with pytest.raises(ValueError, match="Cannot finalize incomplete eval evaluation"):
        finalize_evaluation(
            trials=_make_dummy_trials(4),
            all_scores={"LA_E_1000000": 1.0},
            split="eval",
            output_dir=tmp_path / "eval_fail",
            contract=contract,
            progress_path=tmp_path / "eval_fail" / "eval_progress.json",
            report_template_content=template_content,
            report_replacements={},
        )

    # 2. Incomplete dev refuses publication
    with pytest.raises(ValueError, match="Cannot finalize incomplete dev evaluation"):
        finalize_evaluation(
            trials=_make_dummy_trials(4),
            all_scores={"LA_E_1000000": 1.0},
            split="dev",
            output_dir=tmp_path / "dev_fail",
            contract=contract,
            progress_path=tmp_path / "dev_fail" / "dev_progress.json",
            report_template_content=template_content,
            report_replacements={},
        )

    # 3. Full mock split finalization
    trials = [
        ("spk_0001", "LA_D_1001", "-", "bonafide"),
        ("spk_0002", "LA_D_1002", "A01", "spoof"),
    ]
    scores = {"LA_D_1001": 1.23456789123, "LA_D_1002": -0.98765432198}
    dev_out = tmp_path / "dev_success"
    dev_out.mkdir()
    progress_path = dev_out / "dev_progress.json"

    replacements = {
        "{{ actual_eer_pct }}": "0.00%",
        "{{ status_eer }}": "NOT COMPARABLE",
        "{{ diff_eer }}": "N/A",
        "{{ actual_mindcf }}": "NOT COMPUTED",
        "{{ status_mindcf }}": "NOT COMPUTED",
        "{{ diff_mindcf }}": "N/A",
        "{{ evaluated_protocol }}": "ASVspoof 2019 LA (dev)",
        "{{ execution_timestamp }}": "2026-10-29 00:00:00 UTC",
        "{{ compute_device }}": "cpu",
        "{{ requested_device }}": "cpu",
        "{{ cuda_available }}": "False",
        "{{ cuda_device_name }}": "N/A",
        "{{ mps_available }}": "False",
        "{{ seed }}": "42",
        "{{ deterministic_algorithms }}": "True",
        "{{ os_version }}": "Linux",
        "{{ python_version }}": "3.11.0",
        "{{ torch_version }}": "2.6.0",
        "{{ git_commit }}": "2d37e2bde103dedb10a8e4a9d7a687fb1b91ef4f",
        "{{ git_dirty }}": "False",
        "{{ model_id }}": "clovaai/aasist-l",
        "{{ provenance_status }}": "OFFICIAL_UNMODIFIED",
        "{{ expected_checkpoint_sha256 }}": "814331d088032bb4c3fa61cc014789eadeed464209dd094ab3a2dd6ffbdce27a",
        "{{ checkpoint_registry_url }}": "https://example.com/ckpt",
        "{{ checkpoint_sha256 }}": "814331d088032bb4c3fa61cc014789eadeed464209dd094ab3a2dd6ffbdce27a",
        "{{ upstream_commit }}": "a04c986",
        "{{ parameter_count }}": "85306",
        "{{ protocol_sha256 }}": "dummy_proto_sha",
        "{{ config_sha256 }}": "dummy_config_sha",
        "{{ batch_size }}": "24",
        "{{ selected_split }}": "dev",
        "{{ smoke_memory_mb }}": "NOT COMPUTED",
        "{{ total_eval_trials }}": "2",
        "{{ bonafide_eval_trials }}": "1",
        "{{ spoof_eval_trials }}": "1",
        "{{ protocol_id_count }}": "2",
        "{{ audio_file_count }}": "2",
        "{{ missing_audio_count }}": "0",
        "{{ extra_audio_count }}": "0",
        "{{ eer_threshold }}": "NOT COMPUTED",
        "{{ mindcf_threshold }}": "NOT COMPUTED",
        "{{ roc_auc }}": "N/A",
        "{{ pr_auc }}": "N/A",
        "{{ tp }}": "N/A",
        "{{ fp }}": "N/A",
        "{{ tn }}": "N/A",
        "{{ fn }}": "N/A",
        "{{ fpr_at_90 }}": "N/A",
        "{{ fpr_at_95 }}": "N/A",
        "{{ fpr_at_99 }}": "N/A",
        "{{ reproduction_command }}": "python repro.py",
        "{{ smoke_status }}": "SKIPPED",
        "{{ smoke_samples_count }}": "0",
        "{{ smoke_bonafide_count }}": "0",
        "{{ smoke_spoof_count }}": "0",
        "{{ smoke_mean_bonafide_score }}": "NOT COMPUTED",
        "{{ smoke_mean_spoof_score }}": "NOT COMPUTED",
        "{{ smoke_direction_aligned }}": "NOT COMPUTED",
    }

    with patch.dict(repro.OFFICIAL_SPLIT_COUNTS, {"dev": {"total": 2, "bonafide": 1, "spoof": 1}}):
        raw_file, rep_file, hashes = finalize_evaluation(
            trials=trials,
            all_scores=scores,
            split="dev",
            output_dir=dev_out,
            contract=contract,
            progress_path=progress_path,
            report_template_content=template_content,
            report_replacements=replacements,
        )

    assert raw_file.is_file()
    assert rep_file.is_file()
    assert progress_path.is_file()

    # Round-trip-safe precision .9g check
    raw_text = raw_file.read_text(encoding="utf-8")
    assert "LA_D_1001 1.23456789\n" in raw_text
    assert "LA_D_1002 -0.987654322\n" in raw_text

    # Hashes match files
    assert compute_file_sha256(raw_file) == hashes["raw_scores_sha256"]
    assert compute_file_sha256(rep_file) == hashes["report_sha256"]

    # Verify report does not have nested evaluation-set rendering
    rep_text = rep_file.read_text(encoding="utf-8")
    assert "ASVspoof 2019 LA (ASVspoof 2019 LA" not in rep_text
    assert "| Evaluation set | `ASVspoof 2019 LA (dev)` |" in rep_text

    # Progress marked COMPLETED with both hashes
    prog_data = json.loads(progress_path.read_text(encoding="utf-8"))
    assert prog_data["status"] == "COMPLETED"
    assert prog_data["artifact_hashes"] == hashes


# 15. Interruption flushes completed batches; resumed inference scores missing IDs only
def test_interruption_flush_and_resumed_inference_no_duplicates(tmp_path: Path) -> None:
    trials = [
        ("spk_0001", "LA_E_1001", "-", "bonafide"),
        ("spk_0002", "LA_E_1002", "-", "bonafide"),
        ("spk_0003", "LA_E_1003", "A01", "spoof"),
        ("spk_0004", "LA_E_1004", "A02", "spoof"),
    ]
    flac_dir = tmp_path / "flac"
    flac_dir.mkdir()
    for _, sid, _, _ in trials:
        sf.write(str(flac_dir / f"{sid}.flac"), np.zeros(16000, dtype=np.float32), 16000)

    detector = MagicMock()
    detector.device = torch.device("cpu")

    # Batch 1 (size=2): LA_E_1001, LA_E_1002 succeeds
    # Batch 2 (size=2): LA_E_1003, LA_E_1004 raises KeyboardInterrupt
    call_count = 0

    def mock_forward(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            raise KeyboardInterrupt("Simulated user interruption in batch 2")
        return (None, torch.tensor([[0.0, 1.0], [0.0, 1.0]]))

    detector.model.side_effect = mock_forward

    contract = _make_dummy_contract()
    progress_path = tmp_path / "eval_progress.json"
    all_scores: dict[str, float] = {}

    def on_batch_complete(batch_scores: list[tuple[str, float]], flush: bool) -> None:
        for sid, sval in batch_scores:
            all_scores[sid] = sval
        if flush:
            save_progress_checkpoint(
                progress_path=progress_path,
                contract=contract,
                scores=list(all_scores.items()),
                status="INCOMPLETE",
            )

    # Interruption handling logic matching main()
    exit_code = 0
    try:
        evaluate_protocol_dataset(
            detector=detector,
            trials=trials,
            audio_dir=flac_dir,
            batch_size=2,
            checkpoint_every=5,  # Checkpoint every larger than 1
            on_batch_complete=on_batch_complete,
        )
    except KeyboardInterrupt:
        save_progress_checkpoint(
            progress_path=progress_path,
            contract=contract,
            scores=list(all_scores.items()),
            status="INCOMPLETE",
        )
        exit_code = 130

    assert exit_code == 130
    assert progress_path.is_file()

    # Verify batch 1 was persisted and batch 2 items were NEVER added
    saved_prog = json.loads(progress_path.read_text(encoding="utf-8"))
    assert saved_prog["status"] == "INCOMPLETE"
    assert saved_prog["completed_count"] == 2
    saved_ids = {s[0] for s in saved_prog["scores"]}
    assert saved_ids == {"LA_E_1001", "LA_E_1002"}
    assert "LA_E_1003" not in saved_ids
    assert "LA_E_1004" not in saved_ids

    # Resume that checkpoint through production filtering
    valid_ids = {t[1] for t in trials}
    loaded = load_and_validate_progress(progress_path, contract, valid_ids)
    already_scored = dict(loaded.scores)
    remaining = [t for t in trials if t[1] not in already_scored]

    # Prove inference receives ONLY missing IDs
    assert len(remaining) == 2
    assert [t[1] for t in remaining] == ["LA_E_1003", "LA_E_1004"]


# 16. Invalid seed bounds rejected by both set_seed and CLI validator
def test_invalid_seed_bounds_rejected() -> None:
    # set_seed checks
    with pytest.raises(ValueError, match="Seed must be an integer between 0 and 2\\*\\*32 - 1"):
        set_seed(-1)
    with pytest.raises(ValueError, match="Seed must be an integer between 0 and 2\\*\\*32 - 1"):
        set_seed(2**32)
    with pytest.raises(ValueError, match="Seed must be an integer between 0 and 2\\*\\*32 - 1"):
        set_seed(cast(Any, True))

    # Valid boundaries
    res_0 = set_seed(0, deterministic=False)
    assert res_0["seed"] == 0
    res_max = set_seed(2**32 - 1, deterministic=False)
    assert res_max["seed"] == 2**32 - 1

    # Parser CLI checks
    parser = build_parser()
    args_neg = parser.parse_args(["--seed", "-1"])
    with pytest.raises(SystemExit):
        validate_cli_arguments(parser, args_neg)

    args_over = parser.parse_args(["--seed", str(2**32)])
    with pytest.raises(SystemExit):
        validate_cli_arguments(parser, args_over)


def _setup_dummy_eval_fixtures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_count: int = 2,
) -> tuple[Path, Path, MagicMock]:
    """Helper to set up minimal valid filesystem fixtures for runner testing."""
    monkeypatch.setattr(
        repro,
        "OFFICIAL_EVAL_COUNTS",
        {"total": sample_count, "bonafide": 1, "spoof": sample_count - 1},
    )
    checkpoint_file = tmp_path / "AASIST-L.pth"
    checkpoint_file.write_bytes(b"dummy_weights_12345")
    dataset_root = tmp_path / "data" / "ASVspoof2019" / "LA"
    proto_dir = dataset_root / "ASVspoof2019_LA_cm_protocols"
    proto_dir.mkdir(parents=True, exist_ok=True)
    proto_file = proto_dir / "ASVspoof2019.LA.cm.eval.trl.txt"
    flac_dir = dataset_root / "ASVspoof2019_LA_eval" / "flac"
    flac_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    for i in range(sample_count):
        label = "bonafide" if i == 0 else "spoof"
        attack = "-" if i == 0 else "A07"
        utt_id = f"LA_E_{1001 + i}"
        lines.append(f"LA_0070 {utt_id} {attack} - {label}\n")
        sf.write(str(flac_dir / f"{utt_id}.flac"), np.zeros(16000, dtype=np.float32), 16000)
    proto_file.write_text("".join(lines), encoding="utf-8")

    reg_mock = MagicMock()
    reg_mock.checkpoint_sha256 = repro.compute_file_sha256(checkpoint_file)
    reg_mock.model_id = "clovaai/aasist-l"
    reg_mock.upstream_commit = "a04c986"

    return checkpoint_file, dataset_root, reg_mock


# 17. Production runner locks scoring device and halts on simulated MPS smoke forward failure
def test_production_runner_halts_on_mps_smoke_forward_failure_without_mixed_device_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint_file, dataset_root, reg_mock = _setup_dummy_eval_fixtures(tmp_path, monkeypatch)
    output_dir = tmp_path / "output_mps_failure"

    from voiceshield.core.device import DeviceStatus

    mps_status = DeviceStatus(
        device_name="mps",
        is_mps=True,
        mps_built=True,
        mps_available=True,
        explanation="MPS device simulated",
    )

    detector_init_kwargs: dict[str, Any] = {}

    class MockMPSDetector:
        def __init__(self, **kwargs: Any) -> None:
            detector_init_kwargs.update(kwargs)
            self.device = torch.device("mps")
            self.config = {"architecture": "AASIST-L", "nb_samp": 64600}
            self.provenance = MagicMock(
                model_id="clovaai/aasist-l",
                provenance_status="OFFICIAL_UNMODIFIED",
            )

        def predict(self, audio: Any) -> Any:
            # Simulate an MPS forward failure during smoke
            raise RuntimeError("MPS forward execution failed: out of memory on Metal device")

    with (
        patch(
            "scripts.reproduce_asvspoof_eval.get_git_provenance",
            return_value=("2d37e2bde103dedb10a8e4a9d7a687fb1b91ef4f", False),
        ),
        patch("scripts.reproduce_asvspoof_eval.resolve_device_status", return_value=mps_status),
        patch("scripts.reproduce_asvspoof_eval.get_official_registry_entry", return_value=reg_mock),
        patch("scripts.reproduce_asvspoof_eval.AASISTLDetector", MockMPSDetector),
        patch(
            "sys.argv",
            [
                "reproduce_asvspoof_eval.py",
                "--device",
                "mps",
                "--smoke-count",
                "2",
                "--checkpoint",
                str(checkpoint_file),
                "--dataset-root",
                str(dataset_root),
                "--output-dir",
                str(output_dir),
            ],
        ),
    ):
        ret = main()

    # 1. Runner halts with failure
    assert ret == 1

    # 2. AASISTLDetector was initialized with preferred_device=dev_status.device_name and fallback_to_cpu=False
    assert detector_init_kwargs.get("preferred_device") == "mps"
    assert detector_init_kwargs.get("fallback_to_cpu") is False

    # 3. Halt message outputted to stderr
    captured = capsys.readouterr()
    assert "Smoke validation failed: MPS forward execution failed" in captured.err

    # 4. Proves runner halts rather than producing or resuming a mixed-device score set
    assert not (output_dir / "eval_progress.json").exists()
    assert not (output_dir / "report.md").exists()
    assert not (output_dir / "raw_scores.txt").exists()


# 18. Scoring device transition during smoke is caught and halts runner before full scoring
def test_production_runner_halts_on_device_transition_during_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint_file, dataset_root, reg_mock = _setup_dummy_eval_fixtures(tmp_path, monkeypatch)
    output_dir = tmp_path / "output_device_transition"

    from voiceshield.core.device import DeviceStatus

    mps_status = DeviceStatus(
        device_name="mps",
        is_mps=True,
        mps_built=True,
        mps_available=True,
        explanation="MPS device simulated",
    )

    class TransitioningDetector:
        def __init__(self, **kwargs: Any) -> None:
            self.device = torch.device("mps")
            self.config = {"architecture": "AASIST-L", "nb_samp": 64600}
            self.provenance = MagicMock(
                model_id="clovaai/aasist-l",
                provenance_status="OFFICIAL_UNMODIFIED",
            )

        def predict(self, audio: Any) -> Any:
            # Silently transitions device to CPU during smoke inference
            self.device = torch.device("cpu")
            return MagicMock(bonafide_score=1.0)

    with (
        patch(
            "scripts.reproduce_asvspoof_eval.get_git_provenance",
            return_value=("2d37e2bde103dedb10a8e4a9d7a687fb1b91ef4f", False),
        ),
        patch("scripts.reproduce_asvspoof_eval.resolve_device_status", return_value=mps_status),
        patch("scripts.reproduce_asvspoof_eval.get_official_registry_entry", return_value=reg_mock),
        patch("scripts.reproduce_asvspoof_eval.AASISTLDetector", TransitioningDetector),
        patch(
            "sys.argv",
            [
                "reproduce_asvspoof_eval.py",
                "--device",
                "mps",
                "--smoke-count",
                "2",
                "--checkpoint",
                str(checkpoint_file),
                "--dataset-root",
                str(dataset_root),
                "--output-dir",
                str(output_dir),
            ],
        ),
    ):
        ret = main()

    assert ret == 1
    captured = capsys.readouterr()
    assert "Scoring device changed during smoke inference" in captured.err
    assert not (output_dir / "eval_progress.json").exists()


# 19. Best-effort checkpoint persistence failure does not mask KeyboardInterrupt
def test_interruption_persistence_failure_does_not_mask_keyboard_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint_file, dataset_root, reg_mock = _setup_dummy_eval_fixtures(tmp_path, monkeypatch)
    output_dir = tmp_path / "output_interrupt"

    mock_detector = MagicMock()
    mock_detector.device = torch.device("cpu")
    mock_detector.config = {"architecture": "AASIST-L", "nb_samp": 64600}
    mock_detector.provenance = MagicMock(
        model_id="clovaai/aasist-l",
        provenance_status="OFFICIAL_UNMODIFIED",
    )
    mock_detector.predict.return_value = MagicMock(bonafide_score=1.0)

    with (
        patch(
            "scripts.reproduce_asvspoof_eval.get_git_provenance",
            return_value=("2d37e2bde103dedb10a8e4a9d7a687fb1b91ef4f", False),
        ),
        patch("scripts.reproduce_asvspoof_eval.get_official_registry_entry", return_value=reg_mock),
        patch("scripts.reproduce_asvspoof_eval.AASISTLDetector", return_value=mock_detector),
        patch(
            "scripts.reproduce_asvspoof_eval.evaluate_protocol_dataset",
            side_effect=KeyboardInterrupt,
        ),
        patch(
            "scripts.reproduce_asvspoof_eval.save_progress_checkpoint",
            side_effect=OSError("Disk full error"),
        ),
        patch(
            "sys.argv",
            [
                "reproduce_asvspoof_eval.py",
                "--skip-smoke",
                "--checkpoint",
                str(checkpoint_file),
                "--dataset-root",
                str(dataset_root),
                "--output-dir",
                str(output_dir),
            ],
        ),
    ):
        ret = main()

    # Must return 130 for KeyboardInterrupt
    assert ret == 130
    captured = capsys.readouterr()
    # Must report both the original interruption (SIGINT) and the persistence failure
    assert "SIGINT" in captured.err
    assert "Disk full error" in captured.err


# 20. Best-effort checkpoint persistence failure does not mask ordinary exception
def test_interruption_persistence_failure_does_not_mask_ordinary_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint_file, dataset_root, reg_mock = _setup_dummy_eval_fixtures(tmp_path, monkeypatch)
    output_dir = tmp_path / "output_error"

    mock_detector = MagicMock()
    mock_detector.device = torch.device("cpu")
    mock_detector.config = {"architecture": "AASIST-L", "nb_samp": 64600}
    mock_detector.provenance = MagicMock(
        model_id="clovaai/aasist-l",
        provenance_status="OFFICIAL_UNMODIFIED",
    )
    mock_detector.predict.return_value = MagicMock(bonafide_score=1.0)

    with (
        patch(
            "scripts.reproduce_asvspoof_eval.get_git_provenance",
            return_value=("2d37e2bde103dedb10a8e4a9d7a687fb1b91ef4f", False),
        ),
        patch("scripts.reproduce_asvspoof_eval.get_official_registry_entry", return_value=reg_mock),
        patch("scripts.reproduce_asvspoof_eval.AASISTLDetector", return_value=mock_detector),
        patch(
            "scripts.reproduce_asvspoof_eval.evaluate_protocol_dataset",
            side_effect=RuntimeError("Audio corrupted"),
        ),
        patch(
            "scripts.reproduce_asvspoof_eval.save_progress_checkpoint",
            side_effect=PermissionError("Read-only filesystem"),
        ),
        patch(
            "sys.argv",
            [
                "reproduce_asvspoof_eval.py",
                "--skip-smoke",
                "--checkpoint",
                str(checkpoint_file),
                "--dataset-root",
                str(dataset_root),
                "--output-dir",
                str(output_dir),
            ],
        ),
    ):
        ret = main()

    # Must return 1 for ordinary error
    assert ret == 1
    captured = capsys.readouterr()
    # Must report both the original error and the persistence failure
    assert "Audio corrupted" in captured.err
    assert "Read-only filesystem" in captured.err


# 21. Check-only is not blocked by stale output artifacts because it does not write or score
def test_check_only_not_blocked_by_stale_output_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint_file, dataset_root, reg_mock = _setup_dummy_eval_fixtures(tmp_path, monkeypatch)
    stale_out = tmp_path / "stale_output_dir"
    stale_out.mkdir(parents=True)
    # Populate with stale candidates
    (stale_out / "eval_progress.json").write_text("{}", encoding="utf-8")
    (stale_out / "eval_progress.json.bak").write_text("{}", encoding="utf-8")
    (stale_out / "report.md").write_text("stale report", encoding="utf-8")
    (stale_out / "raw_scores.txt").write_text("stale scores", encoding="utf-8")
    (stale_out / "scratch.tmp").write_text("stale tmp", encoding="utf-8")

    with (
        patch("scripts.reproduce_asvspoof_eval.get_official_registry_entry", return_value=reg_mock),
        patch(
            "sys.argv",
            [
                "reproduce_asvspoof_eval.py",
                "--check-only",
                "--checkpoint",
                str(checkpoint_file),
                "--dataset-root",
                str(dataset_root),
                "--output-dir",
                str(stale_out),
            ],
        ),
    ):
        ret = main()

    # Check-only must succeed without being blocked by stale output directory contents
    assert ret == 0
    captured = capsys.readouterr()
    assert "Protocol/audio validation passed" in captured.out
    # Stale files must remain untouched
    assert (stale_out / "report.md").read_text(encoding="utf-8") == "stale report"


# 22. Smoke-only collision behavior: blocks on full-run artifacts, permits smoke_diagnostic.json update
def test_smoke_only_collision_behavior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint_file, dataset_root, reg_mock = _setup_dummy_eval_fixtures(tmp_path, monkeypatch)

    mock_detector = MagicMock()
    mock_detector.device = torch.device("cpu")
    mock_detector.config = {"architecture": "AASIST-L", "nb_samp": 64600}
    mock_detector.provenance = MagicMock(
        model_id="clovaai/aasist-l",
        provenance_status="OFFICIAL_UNMODIFIED",
    )
    mock_detector.predict.side_effect = [
        MagicMock(bonafide_score=1.5),
        MagicMock(bonafide_score=-1.5),
    ]

    # Case A: stale eval_progress.json in output_dir -> halts with code 1
    stale_eval_dir = tmp_path / "stale_eval_dir"
    stale_eval_dir.mkdir(parents=True)
    (stale_eval_dir / "eval_progress.json").write_text("{}", encoding="utf-8")

    with (
        patch(
            "scripts.reproduce_asvspoof_eval.get_git_provenance",
            return_value=("2d37e2bde103dedb10a8e4a9d7a687fb1b91ef4f", False),
        ),
        patch("scripts.reproduce_asvspoof_eval.get_official_registry_entry", return_value=reg_mock),
        patch("scripts.reproduce_asvspoof_eval.AASISTLDetector", return_value=mock_detector),
        patch(
            "sys.argv",
            [
                "reproduce_asvspoof_eval.py",
                "--smoke-only",
                "--smoke-count",
                "2",
                "--checkpoint",
                str(checkpoint_file),
                "--dataset-root",
                str(dataset_root),
                "--output-dir",
                str(stale_eval_dir),
            ],
        ),
    ):
        ret_a = main()
    assert ret_a == 1
    captured_a = capsys.readouterr()
    assert "Stale collision detected" in captured_a.err

    # Case B: pre-existing smoke_diagnostic.json -> succeeds and updates atomically
    smoke_only_dir = tmp_path / "existing_smoke_dir"
    smoke_only_dir.mkdir(parents=True)
    diag_file = smoke_only_dir / "smoke_diagnostic.json"
    diag_file.write_text('{"status": "OLD_SMOKE"}', encoding="utf-8")

    with (
        patch(
            "scripts.reproduce_asvspoof_eval.get_git_provenance",
            return_value=("2d37e2bde103dedb10a8e4a9d7a687fb1b91ef4f", False),
        ),
        patch("scripts.reproduce_asvspoof_eval.get_official_registry_entry", return_value=reg_mock),
        patch("scripts.reproduce_asvspoof_eval.AASISTLDetector", return_value=mock_detector),
        patch(
            "sys.argv",
            [
                "reproduce_asvspoof_eval.py",
                "--smoke-only",
                "--smoke-count",
                "2",
                "--checkpoint",
                str(checkpoint_file),
                "--dataset-root",
                str(dataset_root),
                "--output-dir",
                str(smoke_only_dir),
            ],
        ),
    ):
        ret_b = main()
    assert ret_b == 0
    updated_diag = json.loads(diag_file.read_text(encoding="utf-8"))
    assert updated_diag["status"] == "NON_COMPARABLE_DIAGNOSTIC_ONLY"


# 23. Report template evaluation set rendering is not nested
def test_report_template_evaluation_set_rendering_not_nested() -> None:
    template_path = Path("reports/templates/asvspoof_reproduction_report_template.md")
    template_text = template_path.read_text(encoding="utf-8")
    rendered = template_text.replace("{{ evaluated_protocol }}", "ASVspoof 2019 LA (eval)")
    assert "ASVspoof 2019 LA (ASVspoof 2019 LA (eval))" not in rendered
    assert "| Evaluation set | `ASVspoof 2019 LA (eval)` |" in rendered
