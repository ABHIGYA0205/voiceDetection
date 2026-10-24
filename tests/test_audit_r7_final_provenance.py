"""Deterministic regression tests for R7/H-09 final bounded configuration corrections.

Covers:
Correction A: Deep configuration immutability (immutable target_tpr_levels tuple, no container mutation, YAML list loading, hash determinism).
Correction B: Truthful effective_threshold and config_hash on all early FileInferenceResult error returns.
Correction C: Truthful benchmark configuration (--config support, --checkpoint precedence, stages and soak plumbing, invocation controls separation).
Correction D: Real package-resource loading (zip archive on sys.path, no Path(str(traversable)) conversion, malformed resource fails visibly).
Correction E: Fail-closed readiness check (/ready requires detector, valid VoiceShieldConfig, non-empty matching hash; HTTP 503 on integrity failure).
Correction F: Guaranteed CPU fallback source (batch_cpu is guaranteed to be on CPU for all inputs; fallback updates device; no-fallback re-raises).
"""

from __future__ import annotations

import io
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient
from pydantic import ValidationError

from services.api.main import create_app
from voiceshield.benchmark.runner import BenchmarkRunner
from voiceshield.benchmark.schema import StageLatency
from voiceshield.config.loader import ConfigError, dump_config, load_config
from voiceshield.config.overrides import apply_config_overrides
from voiceshield.config.schema import VoiceShieldConfig
from voiceshield.models.aasist_adapter import (
    AASIST_L_DEFAULT_CONFIG,
    AASISTLDetector,
)
from voiceshield.pipeline.file_evaluator import evaluate_audio
from voiceshield.vendor.aasist import AASISTModel

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------


def _synthesize_test_wav(duration_seconds: float = 4.0, sample_rate: int = 16000) -> bytes:
    """Generate in-memory WAV audio bytes with a simple sine mixture."""
    num_samples = int(sample_rate * duration_seconds)
    t = np.linspace(0, duration_seconds, num_samples, endpoint=False, dtype=np.float32)
    sig = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, sig, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ------------------------------------------------------------------------------
# Correction A: Deep Configuration Immutability
# ------------------------------------------------------------------------------


def test_correction_a_deep_configuration_immutability() -> None:
    """Correction A: Target TPR levels is an immutable tuple and containers cannot be mutated."""
    c = VoiceShieldConfig()
    old_hash = c.compute_hash()

    assert isinstance(c.evaluation.target_tpr_levels, tuple)
    assert not hasattr(c.evaluation.target_tpr_levels, "append")

    # Item assignment must fail
    with pytest.raises(TypeError, match="'tuple' object does not support item assignment"):
        c.evaluation.target_tpr_levels[0] = 0.5  # type: ignore[index]

    # Reassignment must fail via frozen model
    with pytest.raises(ValidationError):
        c.evaluation.target_tpr_levels = (0.5,)

    with pytest.raises(ValidationError):
        c.windowing.hop_seconds = 2.0

    # Failed mutation attempts must leave compute_hash() untouched
    assert c.compute_hash() == old_hash


def test_correction_a_yaml_loading_accepts_list_and_produces_sorted_tuple(
    tmp_path: Path,
) -> None:
    """Correction A: YAML list syntax validates into a sorted immutable tuple with deterministic hashing."""
    yaml_file = tmp_path / "custom_tpr.yaml"
    yaml_file.write_text(
        "evaluation:\n  target_tpr_levels:\n    - 0.99\n    - 0.90\n    - 0.95\n",
        encoding="utf-8",
    )
    loaded = load_config(yaml_file)
    assert isinstance(loaded.evaluation.target_tpr_levels, tuple)
    assert loaded.evaluation.target_tpr_levels == (0.90, 0.95, 0.99)

    # Dump and reload preserves hash and values
    dump_path = tmp_path / "dumped.yaml"
    dump_config(loaded, dump_path)
    reloaded = load_config(dump_path)
    assert reloaded.evaluation.target_tpr_levels == (0.90, 0.95, 0.99)
    assert reloaded.compute_hash() == loaded.compute_hash()


# ------------------------------------------------------------------------------
# Correction B: Truthful Threshold on Every File Result
# ------------------------------------------------------------------------------


def test_correction_b_malformed_audio_reports_effective_threshold(tmp_path: Path) -> None:
    """Correction B: Audio decode/preprocessing error explicitly reports effective_threshold."""
    custom_cfg = apply_config_overrides(VoiceShieldConfig(), threshold=0.82)
    corrupt_file = tmp_path / "corrupt.wav"
    corrupt_file.write_bytes(
        b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00corrupted_data_not_valid_pcm"
    )

    mock_detector = MagicMock(spec=AASISTLDetector)
    res = evaluate_audio(corrupt_file, detector=mock_detector, config=custom_cfg)

    mock_detector.predict.assert_not_called()
    assert res.status == "error"
    assert res.effective_threshold == 0.82
    assert res.metadata["effective_threshold"] == 0.82
    assert res.config_hash == custom_cfg.compute_hash()
    assert res.metadata["config_hash"] == custom_cfg.compute_hash()


def test_correction_b_unsupported_audio_reports_effective_threshold(tmp_path: Path) -> None:
    """Correction B: Unsupported audio format explicitly reports effective_threshold."""
    custom_cfg = apply_config_overrides(VoiceShieldConfig(), threshold=0.82)
    text_file = tmp_path / "test.txt"
    text_file.write_text("not audio", encoding="utf-8")

    mock_detector = MagicMock(spec=AASISTLDetector)
    res = evaluate_audio(text_file, detector=mock_detector, config=custom_cfg)

    mock_detector.predict.assert_not_called()
    assert res.status == "error"
    assert res.effective_threshold == 0.82
    assert res.metadata["effective_threshold"] == 0.82
    assert res.config_hash == custom_cfg.compute_hash()
    assert res.metadata["config_hash"] == custom_cfg.compute_hash()


def test_correction_b_duration_limit_reports_effective_threshold(tmp_path: Path) -> None:
    """Correction B: Duration ceiling error explicitly reports effective_threshold."""
    custom_cfg = apply_config_overrides(VoiceShieldConfig(), threshold=0.82)
    long_wav = tmp_path / "long.wav"
    long_wav.write_bytes(_synthesize_test_wav(duration_seconds=5.0))

    mock_detector = MagicMock(spec=AASISTLDetector)
    res = evaluate_audio(
        long_wav,
        detector=mock_detector,
        config=custom_cfg,
        max_duration_seconds=2.0,
    )

    mock_detector.predict.assert_not_called()
    assert res.status == "error"
    assert res.effective_threshold == 0.82
    assert res.metadata["effective_threshold"] == 0.82
    assert res.config_hash == custom_cfg.compute_hash()
    assert res.metadata["config_hash"] == custom_cfg.compute_hash()


# ------------------------------------------------------------------------------
# Correction C: Truthful Benchmark Configuration
# ------------------------------------------------------------------------------


def test_correction_c_checkpoint_precedence_and_hash(tmp_path: Path) -> None:
    """Correction C: Benchmark CLI preserves custom YAML checkpoint and overrides via --checkpoint."""
    custom_cfg = apply_config_overrides(
        VoiceShieldConfig(), checkpoint_path=str(tmp_path / "yaml_ckpt.pth")
    )
    custom_yaml = tmp_path / "custom_bench.yaml"
    dump_config(custom_cfg, custom_yaml)

    # 1. Omitted checkpoint preserves YAML checkpoint
    runner1 = BenchmarkRunner(config_path=custom_yaml, checkpoint_path=None)
    assert str(runner1.checkpoint_path) == str(tmp_path / "yaml_ckpt.pth")
    assert runner1.config.compute_hash() == custom_cfg.compute_hash()

    # 2. Explicit checkpoint overrides YAML checkpoint and alters effective hash
    cli_override_path = tmp_path / "cli_ckpt.pth"
    runner2 = BenchmarkRunner(config_path=custom_yaml, checkpoint_path=cli_override_path)
    assert str(runner2.checkpoint_path) == str(cli_override_path)
    assert runner2.config.compute_hash() != custom_cfg.compute_hash()
    expected_override_cfg = apply_config_overrides(
        custom_cfg, checkpoint_path=str(cli_override_path)
    )
    assert runner2.config.compute_hash() == expected_override_cfg.compute_hash()


def test_correction_c_custom_config_values_reach_stages_and_soak(tmp_path: Path) -> None:
    """Correction C: Custom window, VAD threshold, and policy values reach stages and soak test."""
    custom_cfg = VoiceShieldConfig.model_validate(
        {
            "windowing": {
                "window_seconds": 2.0,
                "hop_seconds": 0.5,
                "min_speech_seconds": 0.5,
                "energy_threshold": 0.042,
            },
            "temporal_policy": {
                "threshold_high": 0.88,
                "threshold_low": 0.44,
            },
            "model": {
                "device_preferred": "cpu",
                "fallback_to_cpu": True,
            },
        }
    )

    runner = BenchmarkRunner(
        config=custom_cfg,
        mock_model=True,
        devices=["cpu"],
        iterations=2,
        warmup=1,
        soak_duration_seconds=1.0,
    )

    with (
        patch("voiceshield.benchmark.runner.benchmark_audio_decoding_and_resampling") as mock_s1,
        patch("voiceshield.benchmark.runner.benchmark_feature_extraction") as mock_s2,
        patch("voiceshield.benchmark.runner.benchmark_model_inference") as mock_s3,
        patch("voiceshield.benchmark.runner.benchmark_calibration_and_temporal_policy") as mock_s4,
        patch("voiceshield.benchmark.runner.benchmark_end_to_end_window_processing") as mock_s5,
        patch("voiceshield.benchmark.runner.run_streaming_soak_test") as mock_soak,
    ):
        dummy_st = StageLatency(
            stage_name="stage",
            device="cpu",
            batch_size=1,
            warmup_iterations=1,
            warmup_latencies_ms=[10.0],
            warmup_mean_ms=10.0,
            steady_iterations=1,
            steady_latencies_ms=[10.0],
            p50_ms=10.0,
            p95_ms=10.0,
            p99_ms=10.0,
            mean_ms=10.0,
            std_ms=0.0,
            min_ms=10.0,
            max_ms=10.0,
            throughput_hz=100.0,
            real_time_factor=0.005,
            meets_project_target=True,
        )
        mock_s1.return_value = dummy_st
        mock_s2.return_value = dummy_st
        mock_s3.return_value = dummy_st
        mock_s4.return_value = dummy_st
        mock_s5.return_value = dummy_st
        mock_soak.return_value = None

        runner.run_suite()

        # Check stage 1: duration_seconds and target_sr
        assert mock_s1.call_args.kwargs["duration_seconds"] == 2.0
        assert mock_s1.call_args.kwargs["target_sr"] == 16000

        # Check stage 2: energy_threshold and duration_seconds
        assert mock_s2.call_args.kwargs["duration_seconds"] == 2.0
        assert mock_s2.call_args.kwargs["energy_threshold"] == 0.042

        # Check stage 3: duration_seconds
        assert mock_s3.call_args.kwargs["duration_seconds"] == 2.0

        # Check stage 4: policy_config and duration_seconds
        assert mock_s4.call_args.kwargs["duration_seconds"] == 2.0
        assert mock_s4.call_args.kwargs["policy_config"].threshold_high == 0.88

        # Check stage 5: duration_seconds, energy_threshold, policy_config
        assert mock_s5.call_args.kwargs["duration_seconds"] == 2.0
        assert mock_s5.call_args.kwargs["energy_threshold"] == 0.042
        assert mock_s5.call_args.kwargs["policy_config"].threshold_high == 0.88

        # Check soak: device is CPU (since MPS not requested) and config is passed
        assert mock_soak.call_args.kwargs["device"].type == "cpu"
        assert mock_soak.call_args.kwargs["config"] == custom_cfg


def test_correction_c_device_matrix_separated_in_markdown_report() -> None:
    """Correction C: Evaluated device matrix and invocation parameters are separated from config in report."""
    runner = BenchmarkRunner(
        devices=["cpu", "mps"], iterations=10, warmup=2, soak_duration_seconds=5.0
    )
    mock_report = MagicMock()
    mock_report.provenance.config_hash = runner.config.compute_hash()
    mock_report.provenance.model_id = "AASIST-L"
    mock_report.provenance.checkpoint_path = "models/weights/AASIST-L.pth"
    mock_report.provenance.checkpoint_sha256 = "abc123"
    mock_report.provenance.git_commit = "deadbeef"
    mock_report.provenance.git_dirty = False
    mock_report.provenance.is_mock_weights = False
    mock_report.environment.hardware_model = "MacBookPro"
    mock_report.environment.cpu_brand = "Apple M3"
    mock_report.environment.cpu_cores = 8
    mock_report.environment.macos_version = "15.0"
    mock_report.environment.torch_version = "2.5.0"
    mock_report.environment.mps_built = True
    mock_report.environment.mps_available = True
    mock_report.target_description = "Target"
    mock_report.warning_mock = None
    mock_report.stages_cpu = {}
    mock_report.stages_mps = None
    mock_report.speedup_mps_vs_cpu = {}
    mock_report.soak_test = None
    mock_report.profiling_summary = {
        "primary_device": "cpu",
        "end_to_end_p50_ms": 50.0,
        "stage_percentages": {
            "audio_decoding_pct": 10.0,
            "feature_extraction_pct": 10.0,
            "model_inference_pct": 70.0,
            "calibration_and_policy_pct": 10.0,
        },
        "inference_meets_target": True,
        "end_to_end_meets_target": True,
    }

    md = runner.generate_markdown_report(mock_report)
    assert "| **Effective Canonical Config Hash** |" in md
    assert "| **Configured Device Preference** | `mps` |" in md
    assert "| **Evaluated Devices Matrix** | `cpu, mps` (suite parameter, not in config) |" in md
    assert "| **Benchmark Iterations** | `10` (warmup: `2`) |" in md
    assert "| **Streaming Soak Duration** | `5.0s` |" in md


# ------------------------------------------------------------------------------
# Correction D: Real Package-Resource Loading
# ------------------------------------------------------------------------------


def test_correction_d_true_zip_sys_path_loading(tmp_path: Path) -> None:
    """Correction D: load_config(None) successfully loads from a true zip archive on sys.path outside repo."""
    zip_pkg_path = tmp_path / "app_package.zip"
    repo_yaml_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    canonical_yaml_bytes = repo_yaml_path.read_bytes()

    # Create zip archive with voiceshield/config/default.yaml and empty __init__.py files
    with zipfile.ZipFile(zip_pkg_path, "w") as zf:
        zf.writestr("voiceshield/__init__.py", "")
        zf.writestr("voiceshield/config/__init__.py", "")
        zf.writestr("voiceshield/config/default.yaml", canonical_yaml_bytes)
        # Include necessary loader and schema modules from source tree
        loader_py = (
            Path(__file__).resolve().parents[1] / "src" / "voiceshield" / "config" / "loader.py"
        ).read_bytes()
        schema_py = (
            Path(__file__).resolve().parents[1] / "src" / "voiceshield" / "config" / "schema.py"
        ).read_bytes()
        zf.writestr("voiceshield/config/loader.py", loader_py)
        zf.writestr("voiceshield/config/schema.py", schema_py)

    unrelated_cwd = tmp_path / "isolated_cwd"
    unrelated_cwd.mkdir()

    verify_script = unrelated_cwd / "test_zip_load.py"
    verify_script.write_text(
        "import sys\n"
        f"sys.path.insert(0, r'{zip_pkg_path}')\n"
        "from voiceshield.config.loader import load_config\n"
        "from voiceshield.config.schema import VoiceShieldConfig\n"
        "cfg = load_config(None)\n"
        "expected = VoiceShieldConfig()\n"
        "assert cfg.compute_hash() == expected.compute_hash()\n"
        "print('ZIP_IMPORT_LOAD_SUCCESS')\n",
        encoding="utf-8",
    )

    res = subprocess.run(
        [sys.executable, str(verify_script)],
        cwd=str(unrelated_cwd),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "ZIP_IMPORT_LOAD_SUCCESS" in res.stdout


def test_correction_d_packaged_resource_absent_allows_repository_fallback() -> None:
    """Correction D: When packaged resource is absent, repository fallback is permitted."""
    fake_absent = MagicMock()
    fake_absent.is_file.return_value = False
    fake_module = MagicMock()
    fake_module.joinpath.return_value = fake_absent

    with patch("importlib.resources.files", return_value=fake_module):
        cfg = load_config(None)
        assert (
            cfg.compute_hash() == "774483ef6871c5a0c83f81098cf50cc2e691d39305c9e7c262d478bfb5088596"
        )


def test_correction_d_packaged_resource_valid_wins_over_repository() -> None:
    """Correction D: When packaged resource exists and is valid, packaged resource takes precedence."""
    import yaml

    base_cfg = VoiceShieldConfig()
    custom_cfg = apply_config_overrides(
        base_cfg, threshold=0.77, window_seconds=3.0, checkpoint_path="models/weights/custom.pth"
    )
    valid_custom_yaml = yaml.safe_dump(custom_cfg.model_dump())

    class ValidResource:
        def is_file(self) -> bool:
            return True

        def read_text(self, encoding: str = "utf-8") -> str:
            return valid_custom_yaml

    fake_module = MagicMock()
    fake_module.joinpath.return_value = ValidResource()

    with patch("importlib.resources.files", return_value=fake_module):
        cfg = load_config(None)
        assert cfg.model.checkpoint_path == "models/weights/custom.pth"
        assert cfg.windowing.window_seconds == 3.0
        assert cfg.temporal_policy.threshold_high == 0.77
        assert cfg.compute_hash() == custom_cfg.compute_hash()


def test_correction_d_malformed_packaged_resource_fails_visibly() -> None:
    """Correction D: Malformed higher-precedence packaged resource raises ConfigError and does not silently fall through."""

    class FakeTraversable:
        def is_file(self) -> bool:
            return True

        def read_text(self, encoding: str = "utf-8") -> str:
            return "audio:\n  sample_rate: invalid_unclosed_yaml: ["

    fake_module = MagicMock()
    fake_module.joinpath.return_value = FakeTraversable()

    with patch("importlib.resources.files", return_value=fake_module):
        with pytest.raises(ConfigError, match="YAML syntax error"):
            load_config(None)


def test_correction_d_packaged_resource_unicode_decode_error_fails_closed() -> None:
    """Correction D: Packaged resource with UnicodeDecodeError raises ConfigError and does not fall through."""

    class BrokenResource:
        def is_file(self) -> bool:
            return True

        def read_text(self, encoding: str = "utf-8") -> str:
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    fake_module = MagicMock()
    fake_module.joinpath.return_value = BrokenResource()

    with patch("importlib.resources.files", return_value=fake_module):
        with pytest.raises(ConfigError, match="Invalid UTF-8 in packaged canonical configuration"):
            load_config(None)


def test_correction_d_packaged_resource_os_error_fails_closed() -> None:
    """Correction D: Packaged resource with OSError raises ConfigError and does not fall through."""

    class BrokenResource:
        def is_file(self) -> bool:
            return True

        def read_text(self, encoding: str = "utf-8") -> str:
            raise OSError("I/O error reading package archive entry")

    fake_module = MagicMock()
    fake_module.joinpath.return_value = BrokenResource()

    with patch("importlib.resources.files", return_value=fake_module):
        with pytest.raises(ConfigError, match="Failed to read packaged canonical configuration"):
            load_config(None)


# ------------------------------------------------------------------------------
# Correction E: Fail-Closed Readiness
# ------------------------------------------------------------------------------


def test_correction_e_readiness_fail_closed_matrix() -> None:
    """Correction E: /ready returns HTTP 503 if detector is missing or config integrity check fails."""
    mock_raw = AASISTModel(AASIST_L_DEFAULT_CONFIG)
    det = AASISTLDetector(_instantiated_model=mock_raw, preferred_device="cpu")
    canonical_cfg = VoiceShieldConfig()
    canonical_hash = canonical_cfg.compute_hash()

    with patch.dict("os.environ", {"VOICESHIELD_MOCK_MODEL": "true"}):
        test_app = create_app()

        with TestClient(test_app) as client:
            # 1. Detector missing -> 503 with model load error
            test_app.state.detector = None
            r1 = client.get("/ready")
            assert r1.status_code == 503
            assert "Detector model is not initialized" in r1.json()["detail"]["reason"]

            # 2. Detector exists but config missing -> 503 with integrity error
            test_app.state.detector = det
            test_app.state.effective_config = None
            test_app.state.effective_config_hash = canonical_hash
            r2 = client.get("/ready")
            assert r2.status_code == 503
            assert (
                r2.json()["detail"]["reason"]
                == "Effective configuration unavailable or integrity check failed."
            )

            # 3. Detector exists but stored hash is missing or empty -> 503
            test_app.state.effective_config = canonical_cfg
            test_app.state.effective_config_hash = ""
            r3 = client.get("/ready")
            assert r3.status_code == 503
            assert (
                r3.json()["detail"]["reason"]
                == "Effective configuration unavailable or integrity check failed."
            )

            # 4. Detector exists but stored hash mismatches -> 503
            test_app.state.effective_config_hash = "mismatched_corrupt_hash"
            r4 = client.get("/ready")
            assert r4.status_code == 503
            assert (
                r4.json()["detail"]["reason"]
                == "Effective configuration unavailable or integrity check failed."
            )

            # 5. Detector exists but config hashing raises -> 503 without traceback leak
            mock_broken_cfg = MagicMock(spec=VoiceShieldConfig)
            mock_broken_cfg.compute_hash.side_effect = RuntimeError("Internal hashing failure")
            test_app.state.effective_config = mock_broken_cfg
            test_app.state.effective_config_hash = canonical_hash
            r5 = client.get("/ready")
            assert r5.status_code == 503
            assert (
                r5.json()["detail"]["reason"]
                == "Effective configuration unavailable or integrity check failed."
            )

            # 6. Detector and valid matching config -> 200 ready
            test_app.state.effective_config = canonical_cfg
            test_app.state.effective_config_hash = canonical_hash
            r6 = client.get("/ready")
            assert r6.status_code == 200
            assert r6.json()["status"] == "ready"
            assert r6.json()["device"] == "cpu"
            assert r6.json()["config_hash"] == canonical_hash


# ------------------------------------------------------------------------------
# Correction F: Guaranteed CPU Fallback Source
# ------------------------------------------------------------------------------


def test_correction_f_cpu_fallback_guarantee() -> None:
    """Correction F: Retained batch_cpu is guaranteed to be on CPU for all inputs, and fallback works truthfully."""
    mock_raw = AASISTModel(AASIST_L_DEFAULT_CONFIG)

    # Create an exact-length tensor
    test_tensor = torch.randn(1, 64600)

    # 1. CPU execution succeeds
    det_cpu = AASISTLDetector(_instantiated_model=mock_raw, preferred_device="cpu")
    out = det_cpu.predict_batch(test_tensor)
    assert len(out) == 1
    assert out[0].device_used == "cpu"
    assert out[0].metadata["fallback_used"] is False

    # 2. Transfer failure with fallback_to_cpu=True moves model to CPU and executes
    det_fb = AASISTLDetector(
        _instantiated_model=mock_raw, preferred_device="cpu", fallback_to_cpu=True
    )
    det_fb.device = torch.device("mps")

    orig_to = torch.Tensor.to
    transfer_failed = False

    def custom_to(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if len(args) > 0 and isinstance(args[0], torch.device) and args[0].type == "mps":
            if transfer_failed:
                raise RuntimeError("MPS Device Out of Memory")
            return self
        return orig_to(self, *args, **kwargs)

    transfer_failed = True
    with patch.object(torch.Tensor, "to", custom_to):
        out_fb = det_fb.predict_batch(test_tensor)
        assert len(out_fb) == 1
        assert out_fb[0].device_used == "cpu"
        assert det_fb.device.type == "cpu"
        assert out_fb[0].metadata["fallback_used"] is True

    # 3. Transfer failure with fallback_to_cpu=False re-raises without changing device
    det_no_fb = AASISTLDetector(
        _instantiated_model=mock_raw, preferred_device="cpu", fallback_to_cpu=False
    )
    det_no_fb.device = torch.device("mps")

    with patch.object(torch.Tensor, "to", custom_to):
        with pytest.raises(RuntimeError, match="MPS Device Out of Memory"):
            det_no_fb.predict_batch(test_tensor)
        assert det_no_fb.device.type == "mps"

    # 4. Forward failure with fallback_to_cpu=True
    det_fwd_fb = AASISTLDetector(
        _instantiated_model=mock_raw, preferred_device="cpu", fallback_to_cpu=True
    )
    det_fwd_fb.device = torch.device("mps")
    transfer_failed = False
    attempts = 0
    orig_forward = mock_raw.forward

    def fail_first_forward(x: torch.Tensor) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("MPS forward kernel fault")
        return orig_forward(x)

    with (
        patch.object(torch.Tensor, "to", custom_to),
        patch.object(mock_raw, "forward", fail_first_forward),
    ):
        out_fwd = det_fwd_fb.predict_batch(test_tensor)
        assert len(out_fwd) == 1
        assert out_fwd[0].device_used == "cpu"
        assert det_fwd_fb.device.type == "cpu"
        assert out_fwd[0].metadata["fallback_used"] is True
        assert attempts == 2

    # 5. Forward failure with fallback_to_cpu=False re-raises without changing device
    det_fwd_no_fb = AASISTLDetector(
        _instantiated_model=mock_raw, preferred_device="cpu", fallback_to_cpu=False
    )
    det_fwd_no_fb.device = torch.device("mps")

    with (
        patch.object(torch.Tensor, "to", custom_to),
        patch.object(mock_raw, "forward", side_effect=RuntimeError("MPS forward kernel fault")),
    ):
        with pytest.raises(RuntimeError, match="MPS forward kernel fault"):
            det_fwd_no_fb.predict_batch(test_tensor)
        assert det_fwd_no_fb.device.type == "mps"
