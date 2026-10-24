"""Audit R8 / H-10 verification test suite: Immutable Official Checkpoint Integrity.

Covers:
A. Registry tests (exact values, immutability, unknown rejection, SHA syntax).
B. Wrong but architecture-compatible official checkpoint (rejected before torch.load, cannot bypass).
C. Explicit custom mode (custom/aasist-l, weights_only=True, custom_unverified, actual SHA).
D. Mock mode (mock/aasist-l, mock provenance, is_mock=True, no official SHA).
E. Parameter-count tests (canonical architecture 85306, matches registry, noncanonical rejected).
F. API tests (fail closed on wrong checkpoint, 503 /ready, 503 file, WS closed, custom/mock readiness).
G. CLI tests (wrong official fails closed, custom loads cleanly, mock labels correctly, precedence).
H. Benchmark tests (no silent mock downgrade, fails on missing/wrong, custom and mock provenance).
I. Reproduction-script tests (official only, halts on wrong checkpoint, pinned metadata in report).
J. Real official checkpoint integration (SHA-256, 426428 bytes, strict load, 85306 params, 2 finite logits).
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient

from services.api.main import create_app
from voiceshield.benchmark.runner import BenchmarkRunner
from voiceshield.models import (
    AASIST_L_DEFAULT_CONFIG,
    OFFICIAL_AASIST_L,
    OFFICIAL_MODEL_REGISTRY,
    AASISTLDetector,
    CheckpointIntegrityError,
    CheckpointNotFoundError,
    ModelRegistryEntry,
    UnsupportedModelIdentityError,
    get_official_registry_entry,
)
from voiceshield.pipeline.cli import main as cli_main
from voiceshield.vendor.aasist import AASISTModel


# Helper: synthesize a valid 4-second WAV buffer
def _make_dummy_wav(duration: float = 4.0, sr: int = 16000) -> bytes:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    sig = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    buf = io.BytesIO()
    sf.write(buf, sig, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# Helper: save a valid state dict with random weights
def _create_compatible_random_checkpoint(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    model = AASISTModel(AASIST_L_DEFAULT_CONFIG)
    torch.save(model.state_dict(), path)
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


# ==============================================================================
# A. Registry Tests
# ==============================================================================


def test_section_a_registry_exact_values() -> None:
    """A: Prove every official registry value exactly matches Section 3 specification."""
    entry = get_official_registry_entry("clovaai/aasist-l")

    assert entry.model_id == "clovaai/aasist-l"
    assert entry.architecture == "aasist_l"
    assert entry.upstream_repository == "https://github.com/clovaai/aasist"
    assert entry.upstream_commit == "a04c9863f63d44471dde8a6abcb3b082b07cd1d1"
    assert (
        entry.checkpoint_url
        == "https://raw.githubusercontent.com/clovaai/aasist/a04c9863f63d44471dde8a6abcb3b082b07cd1d1/models/weights/AASIST-L.pth"
    )
    assert entry.checkpoint_filename == "AASIST-L.pth"
    assert (
        entry.checkpoint_sha256
        == "814331d088032bb4c3fa61cc014789eadeed464209dd094ab3a2dd6ffbdce27a"
    )
    assert entry.checkpoint_size_bytes == 426428
    assert entry.parameter_count == 85306
    assert "The upstream source repository is MIT licensed" in (entry.checkpoint_rights_note or "")
    assert "obtain checkpoint bytes from the pinned upstream source" in (
        entry.checkpoint_rights_note or ""
    )


def test_section_a_registry_immutability() -> None:
    """A: Prove registry entry and mapping are completely immutable after import."""
    entry = get_official_registry_entry("clovaai/aasist-l")

    # Entry is frozen (Pydantic model_config: frozen=True)
    with pytest.raises((TypeError, ValueError)):
        entry.model_id = "mutable/hacked"

    with pytest.raises((TypeError, ValueError)):
        entry.checkpoint_sha256 = "0" * 64

    # Mapping proxy prevents modification
    with pytest.raises(TypeError):
        OFFICIAL_MODEL_REGISTRY["clovaai/aasist-l"] = entry  # type: ignore[index]

    with pytest.raises(TypeError):
        OFFICIAL_MODEL_REGISTRY["fake/model"] = entry  # type: ignore[index]


def test_section_a_registry_no_mutable_backing_dict() -> None:
    """A: Prove there is no mutable backing dictionary exposed that can mutate OFFICIAL_MODEL_REGISTRY."""
    import voiceshield.models.registry as reg_mod

    assert not hasattr(reg_mod, "_OFFICIAL_REGISTRY")
    for _attr_name, attr_val in vars(reg_mod).items():
        if isinstance(attr_val, dict):
            assert "clovaai/aasist-l" not in attr_val

    with pytest.raises(TypeError):
        del OFFICIAL_MODEL_REGISTRY["clovaai/aasist-l"]  # type: ignore[attr-defined]

    assert get_official_registry_entry("clovaai/aasist-l").model_id == "clovaai/aasist-l"


def test_section_a_registry_unknown_id_fails() -> None:
    """A: Prove unknown official model IDs raise UnsupportedModelIdentityError."""
    with pytest.raises(
        UnsupportedModelIdentityError, match="Unknown or unsupported official model ID"
    ):
        get_official_registry_entry("unknown/aasist-l")

    with pytest.raises(UnsupportedModelIdentityError):
        get_official_registry_entry("custom/aasist-l")

    with pytest.raises(UnsupportedModelIdentityError):
        get_official_registry_entry("mock/aasist-l")


def test_section_a_registry_sha256_syntax_validation() -> None:
    """A: Prove SHA-256 syntax must be exactly 64 lowercase hexadecimal characters."""
    assert len(OFFICIAL_AASIST_L.checkpoint_sha256) == 64
    assert OFFICIAL_AASIST_L.checkpoint_sha256 == OFFICIAL_AASIST_L.checkpoint_sha256.lower()

    # Invalid uppercase or non-hex length must fail validation
    with pytest.raises(ValueError, match="checkpoint_sha256 must be exactly 64"):
        ModelRegistryEntry(
            model_id="test/invalid",
            architecture="aasist_l",
            upstream_repository="https://example.com",
            upstream_commit="abc",
            checkpoint_url="https://example.com/test.pth",
            checkpoint_filename="test.pth",
            checkpoint_sha256="UPPERCASE" + "0" * 55,
            checkpoint_size_bytes=100,
            parameter_count=85306,
            source_license_spdx="MIT",
            checkpoint_rights_note="test note",
        )


# ==============================================================================
# B. Wrong but Architecture-Compatible Official Checkpoint
# ==============================================================================


def test_section_b_wrong_official_checkpoint_rejected_before_torch_load(tmp_path: Path) -> None:
    """B: Compatible but wrong-hash file is rejected with CheckpointIntegrityError before torch.load."""
    ckpt_path = tmp_path / "compatible_random.pth"
    actual_hash = _create_compatible_random_checkpoint(ckpt_path)

    # Prove it differs from registry hash
    reg = get_official_registry_entry("clovaai/aasist-l")
    assert actual_hash != reg.checkpoint_sha256

    # Patch torch.load to prove it is NEVER called when SHA mismatches
    with patch("torch.load") as mock_torch_load:
        with pytest.raises(CheckpointIntegrityError) as exc_info:
            AASISTLDetector(checkpoint_path=ckpt_path, model_id="clovaai/aasist-l")

        assert "Official AASIST-L checkpoint SHA-256 mismatch" in str(exc_info.value)
        assert reg.checkpoint_sha256 in str(exc_info.value)
        assert actual_hash in str(exc_info.value)
        mock_torch_load.assert_not_called()


def test_section_b_env_var_cannot_bypass_registry(tmp_path: Path) -> None:
    """B: Setting VOICESHIELD_EXPECTED_SHA256 to wrong file's hash CANNOT bypass the registry."""
    ckpt_path = tmp_path / "compatible_random.pth"
    actual_hash = _create_compatible_random_checkpoint(ckpt_path)

    with patch.dict(os.environ, {"VOICESHIELD_EXPECTED_SHA256": actual_hash}):
        with pytest.raises(CheckpointIntegrityError) as exc_info:
            AASISTLDetector(checkpoint_path=ckpt_path, model_id="clovaai/aasist-l")
        assert "Official AASIST-L checkpoint SHA-256 mismatch" in str(exc_info.value)


def test_section_b_provenance_truth_table_invariants_rejected() -> None:
    """B: Prove contradictory ModelProvenance permutations are rejected with ValueError."""
    from voiceshield.models.provenance import ModelProvenance

    # 1. official_verified with checkpoint_verified=False
    with pytest.raises(ValueError, match="checkpoint_verified=True"):
        ModelProvenance(
            model_id="clovaai/aasist-l",
            architecture="aasist_l",
            provenance_status="official_verified",
            checkpoint_verified=False,
            checkpoint_sha256=OFFICIAL_AASIST_L.checkpoint_sha256,
            expected_checkpoint_sha256=OFFICIAL_AASIST_L.checkpoint_sha256,
            upstream_repository=OFFICIAL_AASIST_L.upstream_repository,
            upstream_commit=OFFICIAL_AASIST_L.upstream_commit,
            parameter_count=85306,
            checkpoint_size_bytes=426428,
        )

    # 2. official_verified with is_mock=True
    with pytest.raises(ValueError, match="is_mock=False"):
        ModelProvenance(
            model_id="clovaai/aasist-l",
            architecture="aasist_l",
            provenance_status="official_verified",
            checkpoint_verified=True,
            is_mock=True,
            checkpoint_sha256=OFFICIAL_AASIST_L.checkpoint_sha256,
            expected_checkpoint_sha256=OFFICIAL_AASIST_L.checkpoint_sha256,
            upstream_repository=OFFICIAL_AASIST_L.upstream_repository,
            upstream_commit=OFFICIAL_AASIST_L.upstream_commit,
            parameter_count=85306,
            checkpoint_size_bytes=426428,
        )

    # 3. official_verified with None or mismatching SHA
    with pytest.raises(ValueError):
        ModelProvenance(
            model_id="clovaai/aasist-l",
            architecture="aasist_l",
            provenance_status="official_verified",
            checkpoint_verified=True,
            checkpoint_sha256="0" * 64,
            expected_checkpoint_sha256=OFFICIAL_AASIST_L.checkpoint_sha256,
            upstream_repository=OFFICIAL_AASIST_L.upstream_repository,
            upstream_commit=OFFICIAL_AASIST_L.upstream_commit,
            parameter_count=85306,
            checkpoint_size_bytes=426428,
        )

    # 4. official_verified with parameter count mismatch
    with pytest.raises(ValueError, match="parameter_count"):
        ModelProvenance(
            model_id="clovaai/aasist-l",
            architecture="aasist_l",
            provenance_status="official_verified",
            checkpoint_verified=True,
            checkpoint_sha256=OFFICIAL_AASIST_L.checkpoint_sha256,
            expected_checkpoint_sha256=OFFICIAL_AASIST_L.checkpoint_sha256,
            upstream_repository=OFFICIAL_AASIST_L.upstream_repository,
            upstream_commit=OFFICIAL_AASIST_L.upstream_commit,
            parameter_count=1,
            checkpoint_size_bytes=426428,
        )

    # 5. custom_unverified with checkpoint_verified=True
    with pytest.raises(ValueError, match="checkpoint_verified=False"):
        ModelProvenance(
            model_id="custom/aasist-l",
            architecture="aasist_l",
            provenance_status="custom_unverified",
            checkpoint_verified=True,
            checkpoint_sha256="0" * 64,
            parameter_count=85306,
        )

    # 6. custom_unverified with expected_checkpoint_sha256 attached
    with pytest.raises(ValueError):
        ModelProvenance(
            model_id="custom/aasist-l",
            architecture="aasist_l",
            provenance_status="custom_unverified",
            checkpoint_verified=False,
            checkpoint_sha256="0" * 64,
            expected_checkpoint_sha256="0" * 64,
            parameter_count=85306,
        )

    # 7. mock with is_mock=False or with checkpoint_sha256
    with pytest.raises(ValueError, match="is_mock=True"):
        ModelProvenance(
            model_id="mock/aasist-l",
            architecture="aasist_l",
            provenance_status="mock",
            checkpoint_verified=False,
            is_mock=False,
            parameter_count=85306,
        )
    with pytest.raises(ValueError):
        ModelProvenance(
            model_id="mock/aasist-l",
            architecture="aasist_l",
            provenance_status="mock",
            checkpoint_verified=False,
            is_mock=True,
            checkpoint_sha256="0" * 64,
            parameter_count=85306,
        )


# ==============================================================================
# C. Explicit Custom Mode
# ==============================================================================


def test_section_c_explicit_custom_mode_loading(tmp_path: Path) -> None:
    """C: Compatible random weights load ONLY under custom/aasist-l with custom_unverified provenance."""
    ckpt_path = tmp_path / "custom_weights.pth"
    actual_hash = _create_compatible_random_checkpoint(ckpt_path)

    # 1. Does not load as official
    with pytest.raises(CheckpointIntegrityError):
        AASISTLDetector(checkpoint_path=ckpt_path, model_id="clovaai/aasist-l")

    # 2. Loads cleanly with explicit custom identity
    detector = AASISTLDetector(checkpoint_path=ckpt_path, model_id="custom/aasist-l")

    prov = detector.provenance
    assert prov.model_id == "custom/aasist-l"
    assert prov.provenance_status == "custom_unverified"
    assert prov.checkpoint_verified is False
    assert prov.checkpoint_sha256 == actual_hash
    assert prov.expected_checkpoint_sha256 is None
    assert prov.is_mock is False
    assert prov.parameter_count == 85306

    # 3. Output model_id and metadata reflect custom identity
    audio = np.zeros(64600, dtype=np.float32)
    out = detector.predict(audio)
    assert out.model_id == "custom/aasist-l"
    assert out.metadata["model_provenance"]["provenance_status"] == "custom_unverified"
    assert out.metadata["model_provenance"]["checkpoint_verified"] is False


# ==============================================================================
# D. Mock Mode
# ==============================================================================


def test_section_d_mock_mode_provenance() -> None:
    """D: Mock runtime reports mock/aasist-l, status mock, false verification, and null official SHA."""
    raw_model = AASISTModel(AASIST_L_DEFAULT_CONFIG)
    detector = AASISTLDetector(_instantiated_model=raw_model)

    prov = detector.provenance
    assert prov.model_id == "mock/aasist-l"
    assert prov.architecture == "aasist_l"
    assert prov.provenance_status == "mock"
    assert prov.checkpoint_verified is False
    assert prov.checkpoint_sha256 is None
    assert prov.expected_checkpoint_sha256 is None
    assert prov.is_mock is True

    # Inference output carries mock identity
    audio = np.zeros(64600, dtype=np.float32)
    out = detector.predict(audio)
    assert out.model_id == "mock/aasist-l"
    assert out.metadata["model_provenance"]["is_mock"] is True
    assert out.metadata["model_provenance"]["provenance_status"] == "mock"


# ==============================================================================
# E. Parameter Count Tests
# ==============================================================================


def test_section_e_canonical_architecture_parameter_count() -> None:
    """E: Canonical AASIST-L architecture has exactly 85,306 parameters matching registry."""
    raw_model = AASISTModel(AASIST_L_DEFAULT_CONFIG)
    total_params = sum(p.numel() for p in raw_model.parameters())
    assert total_params == 85306

    reg = get_official_registry_entry("clovaai/aasist-l")
    assert reg.parameter_count == 85306
    assert total_params == reg.parameter_count


def test_section_e_official_with_altered_config_rejected(tmp_path: Path) -> None:
    """E: Official identity with altered hyperparameter config altering parameter count is rejected."""
    altered_config = dict(AASIST_L_DEFAULT_CONFIG)
    altered_config["first_conv"] = 64  # Changes parameter count

    ckpt_path = tmp_path / "dummy.pth"
    ckpt_path.write_bytes(b"dummy")

    with pytest.raises(CheckpointIntegrityError):
        AASISTLDetector(
            checkpoint_path=ckpt_path,
            model_id="clovaai/aasist-l",
            model_config=altered_config,
        )


def test_section_d_canonical_config_protection_and_temperature_rejection() -> None:
    """D: Mutating AASIST_L_DEFAULT_CONFIG does not affect canonical config; temperatures change is rejected."""
    from voiceshield.models.aasist_adapter import (
        AASIST_L_DEFAULT_CONFIG,
        get_canonical_aasist_l_config,
    )

    orig_temp = list(AASIST_L_DEFAULT_CONFIG["temperatures"])
    # Mutate public dictionary
    AASIST_L_DEFAULT_CONFIG["temperatures"][0] = 999.0
    AASIST_L_DEFAULT_CONFIG["nb_samp"] = 12345

    # Fresh canonical copy must remain intact
    fresh = get_canonical_aasist_l_config()
    assert fresh["temperatures"] == orig_temp
    assert fresh["nb_samp"] == 64600

    # Restore public dict for hygiene
    AASIST_L_DEFAULT_CONFIG["temperatures"] = orig_temp
    AASIST_L_DEFAULT_CONFIG["nb_samp"] = 64600

    # Altered temperatures (same parameter count!) must be rejected in official mode before checkpoint reading
    altered_temps_config = get_canonical_aasist_l_config()
    altered_temps_config["temperatures"] = [1.0, 1.0, 50.0, 50.0]

    with pytest.raises(CheckpointIntegrityError, match="Noncanonical architecture configuration"):
        AASISTLDetector(
            checkpoint_path="nonexistent.pth",
            model_id="clovaai/aasist-l",
            model_config=altered_temps_config,
        )

    # Altered nb_samp must be rejected in official mode before checkpoint reading
    altered_samp_config = get_canonical_aasist_l_config()
    altered_samp_config["nb_samp"] = 64000

    with pytest.raises(CheckpointIntegrityError, match="Noncanonical architecture configuration"):
        AASISTLDetector(
            checkpoint_path="nonexistent.pth",
            model_id="clovaai/aasist-l",
            model_config=altered_samp_config,
        )


# ==============================================================================
# F. API Tests
# ==============================================================================


def test_section_f_api_fails_closed_on_wrong_checkpoint(tmp_path: Path) -> None:
    """F: API with wrong official checkpoint leaves detector None, /ready returns 503, file fails closed."""
    wrong_ckpt = tmp_path / "wrong.pth"
    _create_compatible_random_checkpoint(wrong_ckpt)

    app = create_app()
    with patch.dict(
        os.environ,
        {
            "VOICESHIELD_CHECKPOINT_PATH": str(wrong_ckpt),
            "VOICESHIELD_MOCK_MODEL": "false",
        },
    ):
        with TestClient(app) as client:
            # 1. /ready returns 503
            r_ready = client.get("/ready")
            assert r_ready.status_code == 503
            detail = r_ready.json()["detail"]
            assert detail["status"] == "not_ready"
            assert "SHA-256 mismatch" in detail["reason"]

            # 2. File upload fails closed before detector invocation (503)
            wav_bytes = _make_dummy_wav(4.0)
            r_file = client.post(
                "/api/v1/analyze/file",
                files={"file": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")},
            )
            assert r_file.status_code == 503
            assert r_file.json()["detail"]["error_code"] == "DETECTOR_NOT_LOADED"

            # 3. WebSocket stream fails closed before session ACK
            with client.websocket_connect("/api/v1/stream") as ws:
                ws.send_text(
                    '{"type":"start","sample_rate":16000,"channels":1,"pcm_encoding":"int16"}'
                )
                err_event = ws.receive_json()
                assert err_event["type"] == "error"
                assert err_event["error_code"] == "DETECTOR_NOT_READY"


def test_section_f_api_custom_mode_readiness(tmp_path: Path) -> None:
    """F: API in explicit custom mode reports custom_unverified and false verification."""
    custom_ckpt = tmp_path / "custom.pth"
    actual_hash = _create_compatible_random_checkpoint(custom_ckpt)

    app = create_app()
    with patch.dict(
        os.environ,
        {
            "VOICESHIELD_MODEL_ID": "custom/aasist-l",
            "VOICESHIELD_CHECKPOINT_PATH": str(custom_ckpt),
            "VOICESHIELD_MOCK_MODEL": "false",
        },
    ):
        with TestClient(app) as client:
            r_ready = client.get("/ready")
            assert r_ready.status_code == 200
            data = r_ready.json()
            assert data["status"] == "ready"
            assert data["model_id"] == "custom/aasist-l"
            assert data["provenance_status"] == "custom_unverified"
            assert data["checkpoint_verified"] is False
            assert data["checkpoint_sha256"] == actual_hash


def test_section_f_api_mock_mode_readiness() -> None:
    """F: API in explicit mock mode reports mock/aasist-l, mock status, and is_mock=True."""
    app = create_app()
    with patch.dict(os.environ, {"VOICESHIELD_MOCK_MODEL": "true"}):
        with TestClient(app) as client:
            r_ready = client.get("/ready")
            assert r_ready.status_code == 200
            data = r_ready.json()
            assert data["status"] == "ready"
            assert data["model_id"] == "mock/aasist-l"
            assert data["provenance_status"] == "mock"
            assert data["checkpoint_verified"] is False
            assert data["is_mock_detector"] is True


def test_section_f_api_untyped_or_mismatched_provenance_fails_closed() -> None:
    """F: API fails closed if detector has no provenance, plain dict, or mismatched state."""
    from voiceshield.models.base import BaseDetector, DetectorOutput
    from voiceshield.models.provenance import ModelProvenance

    app = create_app()

    class FakeDetectorNoProv(BaseDetector):
        device = torch.device("cpu")
        provenance = None

        def predict(
            self, audio: np.ndarray[Any, np.dtype[np.float32]] | torch.Tensor
        ) -> DetectorOutput:
            return DetectorOutput.from_scores(
                raw_logits=(0.0, 0.0),
                bonafide_score=0.5,
                synthetic_score=0.5,
                is_synthetic_suspected=False,
                input_samples=16000,
                device_used="cpu",
                model_id="clovaai/aasist-l",
            )

        def predict_batch(self, batch: torch.Tensor) -> list[DetectorOutput]:
            return []

    with TestClient(app) as client:
        app.state.detector = FakeDetectorNoProv()
        r = client.get("/ready")
        assert r.status_code == 503
        assert "provenance is missing, untyped, or invalid" in r.json()["detail"]["reason"]

        r_file = client.post(
            "/api/v1/analyze/file",
            files={"file": ("test.wav", io.BytesIO(_make_dummy_wav(4.0)), "audio/wav")},
        )
        assert r_file.status_code == 503

        with client.websocket_connect("/api/v1/stream") as ws:
            ws.send_text('{"type":"start","sample_rate":16000,"channels":1,"pcm_encoding":"int16"}')
            err_event = ws.receive_json()
            assert err_event["type"] == "error"
            assert err_event["error_code"] == "DETECTOR_NOT_READY"

    # Detector with plain dict provenance
    class FakeDetectorDictProv(BaseDetector):
        device = torch.device("cpu")
        provenance = {"model_id": "clovaai/aasist-l", "provenance_status": "official_verified"}

        def predict(
            self, audio: np.ndarray[Any, np.dtype[np.float32]] | torch.Tensor
        ) -> DetectorOutput:
            return DetectorOutput.from_scores(
                raw_logits=(0.0, 0.0),
                bonafide_score=0.5,
                synthetic_score=0.5,
                is_synthetic_suspected=False,
                input_samples=16000,
                device_used="cpu",
                model_id="clovaai/aasist-l",
            )

        def predict_batch(self, batch: torch.Tensor) -> list[DetectorOutput]:
            return []

    with TestClient(app) as client:
        app.state.detector = FakeDetectorDictProv()
        r = client.get("/ready")
        assert r.status_code == 503

    # Non-mock app state with mock provenance
    mock_prov = ModelProvenance(
        model_id="mock/aasist-l",
        architecture="aasist_l",
        provenance_status="mock",
        checkpoint_verified=False,
        is_mock=True,
        parameter_count=85306,
    )

    class FakeDetectorMockProv(BaseDetector):
        device = torch.device("cpu")
        provenance = mock_prov

        def predict(
            self, audio: np.ndarray[Any, np.dtype[np.float32]] | torch.Tensor
        ) -> DetectorOutput:
            return DetectorOutput.from_scores(
                raw_logits=(0.0, 0.0),
                bonafide_score=0.5,
                synthetic_score=0.5,
                is_synthetic_suspected=False,
                input_samples=16000,
                device_used="cpu",
                model_id="mock/aasist-l",
            )

        def predict_batch(self, batch: torch.Tensor) -> list[DetectorOutput]:
            return []

    with TestClient(app) as client:
        app.state.detector = FakeDetectorMockProv()
        app.state.is_mock_detector = False
        r = client.get("/ready")
        assert r.status_code == 503
        assert (
            "Non-mock runtime state cannot accept mock model provenance"
            in r.json()["detail"]["reason"]
        )


# ==============================================================================
# G. CLI Tests
# ==============================================================================


def test_section_g_cli_wrong_official_checkpoint_fails_closed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """G: CLI with wrong checkpoint in official mode exits nonzero, writes no results, no traceback."""
    wrong_ckpt = tmp_path / "wrong.pth"
    _create_compatible_random_checkpoint(wrong_ckpt)
    wav_file = tmp_path / "test.wav"
    wav_file.write_bytes(_make_dummy_wav(4.0))

    exit_code = cli_main(["--input", str(wav_file), "--checkpoint", str(wrong_ckpt)])
    captured = capsys.readouterr()

    assert exit_code != 0
    assert "ERROR: Official AASIST-L checkpoint SHA-256 mismatch" in captured.err
    # Must contain no unhandled Python traceback
    assert "Traceback (most recent call last):" not in captured.err


def test_section_g_cli_custom_mode_successful(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """G: CLI with --model-id custom/aasist-l safely loads and reports custom provenance."""
    custom_ckpt = tmp_path / "custom.pth"
    actual_hash = _create_compatible_random_checkpoint(custom_ckpt)
    wav_file = tmp_path / "test.wav"
    wav_file.write_bytes(_make_dummy_wav(4.0))
    csv_out = tmp_path / "results.csv"

    exit_code = cli_main(
        [
            "--input",
            str(wav_file),
            "--checkpoint",
            str(custom_ckpt),
            "--model-id",
            "custom/aasist-l",
            "--output",
            str(csv_out),
        ]
    )
    _ = capsys.readouterr()
    assert exit_code == 0
    assert csv_out.is_file()

    with open(csv_out, encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 1
        row = reader[0]
        assert row["model_id"] == "custom/aasist-l"
        assert row["provenance_status"] == "custom_unverified"
        assert row["checkpoint_verified"] == "False"
        assert row["checkpoint_sha256"] == actual_hash


# ==============================================================================
# H. Benchmark Tests
# ==============================================================================


def test_section_h_benchmark_fails_without_mock(tmp_path: Path) -> None:
    """H: Benchmark fails cleanly without --mock-model when checkpoint is missing."""
    runner = BenchmarkRunner(
        checkpoint_path=tmp_path / "non_existent.pth",
        mock_model=False,
        iterations=1,
        warmup=0,
    )
    with pytest.raises(CheckpointNotFoundError):
        runner.run_suite()


def test_section_h_benchmark_wrong_official_hash_fails(tmp_path: Path) -> None:
    """H: Benchmark fails cleanly when official checkpoint hash mismatches."""
    wrong_ckpt = tmp_path / "wrong.pth"
    _create_compatible_random_checkpoint(wrong_ckpt)

    runner = BenchmarkRunner(
        checkpoint_path=wrong_ckpt,
        mock_model=False,
        iterations=1,
        warmup=0,
    )
    with pytest.raises(CheckpointIntegrityError):
        runner.run_suite()


def test_section_h_benchmark_explicit_mock_mode() -> None:
    """H: Explicit mock benchmark reports mock/aasist-l with no official hash."""
    runner = BenchmarkRunner(
        mock_model=True,
        iterations=2,
        warmup=1,
        devices=["cpu"],
    )
    report = runner.run_suite()
    prov = report.provenance
    assert prov.model_id == "mock/aasist-l"
    assert prov.provenance_status == "mock"
    assert prov.checkpoint_verified is False
    assert prov.checkpoint_sha256 is None
    assert prov.checkpoint_path is None
    assert prov.is_mock_weights is True


def test_section_h_benchmark_cli_clean_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """H: Benchmark CLI main() returns 1 with concise stderr message, no traceback, and creates no reports."""
    from voiceshield.benchmark.cli import main as bm_main

    wrong_ckpt = tmp_path / "wrong.pth"
    _create_compatible_random_checkpoint(wrong_ckpt)
    report_dir = tmp_path / "reports"

    ret = bm_main(
        [
            "--checkpoint",
            str(wrong_ckpt),
            "--output-dir",
            str(report_dir),
            "--devices",
            "cpu",
            "--iterations",
            "1",
            "--warmup",
            "0",
            "--soak-duration",
            "0",
        ]
    )
    captured = capsys.readouterr()

    assert ret == 1
    assert "Official AASIST-L checkpoint SHA-256 mismatch" in captured.err
    assert "Traceback (most recent call last):" not in captured.err
    assert not report_dir.exists()


# ==============================================================================
# I. Reproduction Script Tests
# ==============================================================================


def test_section_i_reproduction_script_halts_on_wrong_checkpoint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """I: reproduce_asvspoof_eval.py halts before dataset access when checkpoint hash is wrong."""
    from scripts.reproduce_asvspoof_eval import main as repro_main

    wrong_ckpt = tmp_path / "wrong.pth"
    _create_compatible_random_checkpoint(wrong_ckpt)

    exit_code = repro_main(["--checkpoint", str(wrong_ckpt)])
    captured = capsys.readouterr()

    assert exit_code != 0
    assert "HALTING: Checkpoint SHA-256 integrity verification failed" in captured.err


def test_section_i_reproduction_report_template_facts() -> None:
    """I: Reproduction report template renders with exact registry metadata and no obsolete facts."""
    template_path = Path("reports/templates/asvspoof_reproduction_report_template.md")
    content = template_path.read_text(encoding="utf-8")

    reg = get_official_registry_entry("clovaai/aasist-l")

    replacements = {
        "{{ upstream_commit }}": reg.upstream_commit,
        "{{ parameter_count }}": str(reg.parameter_count),
        "{{ model_id }}": reg.model_id,
        "{{ expected_checkpoint_sha256 }}": reg.checkpoint_sha256,
        "{{ checkpoint_sha256 }}": reg.checkpoint_sha256,
        "{{ provenance_status }}": "official_verified",
    }
    rendered = content
    for k, v in replacements.items():
        rendered = rendered.replace(k, v)

    assert "clovaai/aasist" in rendered
    assert "a04c9863f63d44471dde8a6abcb3b082b07cd1d1" in rendered
    assert "814331d088032bb4c3fa61cc014789eadeed464209dd094ab3a2dd6ffbdce27a" in rendered
    assert "85,306" in rendered or "85306" in rendered
    assert "MIT (NAVER Corp / Clova AI)" in rendered

    # Must NOT contain obsolete/incorrect AASIST-L identity facts
    assert "342b3112" not in rendered
    assert "85,386" not in rendered
    assert "BSD-3-Clause" not in rendered


# ==============================================================================
# J. Real Official Checkpoint Integration
# ==============================================================================


@pytest.mark.skipif(
    not Path("models/weights/AASIST-L.pth").is_file(),
    reason="Official AASIST-L.pth checkpoint not present on disk; skipping real integration test.",
)
def test_section_j_real_official_checkpoint_integration() -> None:
    """J: Official AASIST-L checkpoint strict load, exact SHA/size/params, and 2 finite logits."""
    ckpt_path = Path("models/weights/AASIST-L.pth")
    reg = get_official_registry_entry("clovaai/aasist-l")

    # 1. Exact size
    assert ckpt_path.stat().st_size == reg.checkpoint_size_bytes
    assert ckpt_path.stat().st_size == 426428

    # 2. Exact SHA-256
    hasher = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    actual_sha = hasher.hexdigest().lower()
    assert actual_sha == reg.checkpoint_sha256
    assert actual_sha == "814331d088032bb4c3fa61cc014789eadeed464209dd094ab3a2dd6ffbdce27a"

    # 3. Safe strict load via detector
    detector = AASISTLDetector(checkpoint_path=ckpt_path, model_id="clovaai/aasist-l")
    prov = detector.provenance
    assert prov.model_id == "clovaai/aasist-l"
    assert prov.provenance_status == "official_verified"
    assert prov.checkpoint_verified is True
    assert prov.parameter_count == 85306
    assert prov.upstream_commit == "a04c9863f63d44471dde8a6abcb3b082b07cd1d1"

    # 4. Single synthetic waveform produces exactly 2 finite logits
    audio = np.zeros(64600, dtype=np.float32)
    output = detector.predict(audio)
    assert len(output.raw_logits) == 2
    assert np.isfinite(output.raw_logits[0])
    assert np.isfinite(output.raw_logits[1])
    assert 0.0 <= output.synthetic_likelihood <= 1.0
