"""Unit tests for random seeding, run metadata collection, and run record persistence."""

from pathlib import Path

import numpy as np
import torch

from voiceshield.config import VoiceShieldConfig
from voiceshield.reproducibility import (
    collect_run_metadata,
    create_run_record,
    load_run_record,
    save_run_record,
    set_seed,
)


def test_seed_reproducibility() -> None:
    """Verify that set_seed produces deterministic outputs in NumPy and PyTorch."""
    set_seed(42)
    np_rand_1 = np.random.rand(5)
    torch_rand_1 = torch.randn(5)

    # Re-seed with the same value
    set_seed(42)
    np_rand_2 = np.random.rand(5)
    torch_rand_2 = torch.randn(5)

    np.testing.assert_allclose(np_rand_1, np_rand_2)
    assert torch.allclose(torch_rand_1, torch_rand_2)

    # Seed with different value produces different output
    set_seed(99)
    np_rand_3 = np.random.rand(5)
    assert not np.allclose(np_rand_1, np_rand_3)


def test_collect_run_metadata(tmp_path: Path) -> None:
    """Verify runtime metadata collection captures environment and hashes."""
    # Create mock manifest file
    mock_manifest = tmp_path / "manifest.jsonl"
    mock_manifest.write_text('{"speaker_id": "spk01"}\n', encoding="utf-8")

    config = VoiceShieldConfig()
    metadata = collect_run_metadata(config=config, manifest_path=mock_manifest)

    assert metadata.config_hash == config.compute_hash()
    assert metadata.manifest_hash is not None
    assert len(metadata.manifest_hash) == 64  # SHA-256 length
    assert metadata.platform != ""
    assert metadata.python_version != ""
    assert metadata.torch_version != ""
    assert metadata.selected_device in ("mps", "cpu")
    assert metadata.timestamp_utc != ""


def test_run_record_persistence_roundtrip(tmp_path: Path) -> None:
    """Verify creating, saving, and reloading a RunRecord preserves all fields."""
    config = VoiceShieldConfig()
    record = create_run_record(
        config=config,
        metrics={"eer": 0.042, "accuracy": 0.958},
        notes="Phase 1 baseline reproducibility test",
    )

    out_file = save_run_record(record, output_dir=tmp_path)
    assert out_file.exists()
    assert out_file.name == f"{record.run_id}.json"

    loaded = load_run_record(out_file)
    assert loaded.run_id == record.run_id
    assert loaded.metadata.config_hash == record.metadata.config_hash
    assert loaded.metrics["eer"] == 0.042
    assert loaded.notes == "Phase 1 baseline reproducibility test"
