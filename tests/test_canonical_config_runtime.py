"""Deterministic regression tests for R7 canonical configuration runtime wiring (audit finding H-09).

Covers:
A. Strict config (schema validation, aasist_l, legacy field rejection, NaN/Inf rejection).
B. Load once at startup and failure before accepting work on missing/invalid config.
C. Streaming session injection (effective config drives session params, hash in ACK, no route fallback).
D. File evaluation injection (effective config used on omission, request override precedence, hash consistency).
E. CLI override precedence (canonical -> custom YAML -> CLI flags).
F. Detector configuration injection (device, threshold, fallback_to_cpu=False consistency).
G. Readiness consistency (/ready, session_started, and file result match effective config hash).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient
from pydantic import ValidationError

from services.api.main import app, create_app
from voiceshield.config.loader import (
    ConfigError,
    dump_config,
    load_config,
)
from voiceshield.config.overrides import apply_config_overrides
from voiceshield.config.schema import (
    AudioConfig,
    TemporalPolicyConfig,
    VoiceShieldConfig,
    WindowingConfig,
)
from voiceshield.models.aasist_adapter import (
    AASIST_L_DEFAULT_CONFIG,
    AASISTLDetector,
)
from voiceshield.pipeline.cli import main as cli_main
from voiceshield.pipeline.file_evaluator import evaluate_audio
from voiceshield.vendor.aasist import AASISTModel


# Helper: synthesize canonical test WAV audio
def _synthesize_test_wav(duration_seconds: float = 4.5, sample_rate: int = 16000) -> bytes:
    """Generate a clean synthetic sine wave in 16 kHz 16-bit mono PCM."""
    t = np.linspace(
        0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False, dtype=np.float32
    )
    # 440 Hz tone with audible amplitude
    samples = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ------------------------------------------------------------------------------
# 9.A: Strict Config Tests
# ------------------------------------------------------------------------------


def test_default_yaml_architecture_is_aasist_l() -> None:
    """Verify default YAML specifies architecture 'aasist_l'."""
    cfg = load_config(Path("configs/default.yaml"))
    assert cfg.model.architecture == "aasist_l"


def test_default_yaml_hash_equals_voiceshield_config_default_hash() -> None:
    """Verify VoiceShieldConfig() and configs/default.yaml produce identical deterministic hash."""
    from_file = load_config(Path("configs/default.yaml"))
    from_python = VoiceShieldConfig()
    assert from_file.compute_hash() == from_python.compute_hash()


def test_unknown_root_key_rejected(tmp_path: Path) -> None:
    """Verify unknown root configuration key is rejected with exact field path."""
    bad_yaml = tmp_path / "unknown_root.yaml"
    bad_yaml.write_text("unknown_root_key: true\n", encoding="utf-8")
    with pytest.raises(ConfigError) as exc_info:
        load_config(bad_yaml)
    assert "[unknown_root_key]" in str(exc_info.value)


def test_unknown_nested_key_rejected(tmp_path: Path) -> None:
    """Verify unknown nested configuration key is rejected with exact field path."""
    bad_yaml = tmp_path / "unknown_nested.yaml"
    bad_yaml.write_text("model:\n  unknown_field: 123\n", encoding="utf-8")
    with pytest.raises(ConfigError) as exc_info:
        load_config(bad_yaml)
    assert "[model.unknown_field]" in str(exc_info.value)


@pytest.mark.parametrize(
    "legacy_yaml,expected_field_path",
    [
        ("training:\n  batch_size: 32\n", "[training]"),
        ("calibration:\n  method: temperature_scaling\n", "[calibration]"),
        ("api:\n  host: 127.0.0.1\n", "[api]"),
        ("logging:\n  level: INFO\n", "[logging]"),
        ("model:\n  num_classes: 2\n", "[model.num_classes]"),
        ("model:\n  embedding_dim: 128\n", "[model.embedding_dim]"),
        (
            "temporal_policy:\n  consecutive_hops_required: 2\n",
            "[temporal_policy.consecutive_hops_required]",
        ),
        ("temporal_policy:\n  decay_factor: 0.85\n", "[temporal_policy.decay_factor]"),
        ("temporal_policy:\n  alert_threshold: 0.65\n", "[temporal_policy.alert_threshold]"),
        (
            "temporal_policy:\n  explanation_rules:\n    - REQUIRE_MINIMUM_SPEECH_DURATION\n",
            "[temporal_policy.explanation_rules]",
        ),
    ],
)
def test_removed_legacy_fields_rejected_with_exact_field_path(
    tmp_path: Path, legacy_yaml: str, expected_field_path: str
) -> None:
    """Verify each removed legacy field fails with ConfigError and its exact field path."""
    test_file = tmp_path / "legacy.yaml"
    test_file.write_text(legacy_yaml, encoding="utf-8")
    with pytest.raises(ConfigError) as exc_info:
        load_config(test_file)
    assert expected_field_path in str(exc_info.value)


def test_unsupported_architecture_rejected(tmp_path: Path) -> None:
    """Verify unsupported architecture values such as 'lcnn' are clearly rejected."""
    bad_yaml = tmp_path / "bad_arch.yaml"
    bad_yaml.write_text("model:\n  architecture: lcnn\n", encoding="utf-8")
    with pytest.raises(ConfigError) as exc_info:
        load_config(bad_yaml)
    assert "[model.architecture]" in str(exc_info.value)


@pytest.mark.parametrize("invalid_val", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_numeric_values_rejected(invalid_val: float) -> None:
    """Verify NaN and Inf are rejected across all numeric configuration fields."""
    with pytest.raises(ValidationError):
        WindowingConfig(window_seconds=invalid_val)
    with pytest.raises(ValidationError):
        TemporalPolicyConfig(threshold_high=invalid_val)
    with pytest.raises(ValidationError):
        TemporalPolicyConfig(timing_gap_threshold_seconds=invalid_val)
    with pytest.raises(ValidationError):
        AudioConfig(normalize_range=(-1.0, invalid_val))


# ------------------------------------------------------------------------------
# 9.B: Load Once and Startup Failure Tests
# ------------------------------------------------------------------------------


def test_config_loaded_once_at_startup_not_per_request(tmp_path: Path) -> None:
    """Verify that configuration is loaded once during app lifespan and not re-read per request."""
    custom_cfg = apply_config_overrides(VoiceShieldConfig(), threshold=0.72)
    cfg_file = tmp_path / "test_app_config.yaml"
    dump_config(custom_cfg, cfg_file)

    test_app = create_app(config_path=cfg_file)

    with patch("services.api.main.load_config", wraps=load_config) as mock_load:
        with TestClient(test_app) as client:
            # Lifespan should have loaded the configuration once
            assert mock_load.call_count == 1
            assert test_app.state.effective_config.temporal_policy.threshold_high == 0.72

            # Sending health checks must not re-read the configuration file
            r1 = client.get("/health")
            assert r1.status_code == 200
            assert mock_load.call_count == 1

            r2 = client.get("/health")
            assert r2.status_code == 200
            assert mock_load.call_count == 1


def test_missing_selected_config_path_fails_startup(tmp_path: Path) -> None:
    """Verify startup fails clearly when an explicitly selected config path is missing."""
    missing_path = tmp_path / "does_not_exist.yaml"
    test_app = create_app(config_path=missing_path)

    with pytest.raises(ConfigError, match="Configuration file not found"):
        with TestClient(test_app):
            pass


def test_invalid_selected_config_path_fails_startup(tmp_path: Path) -> None:
    """Verify startup fails clearly when an explicitly selected config path is malformed."""
    malformed_path = tmp_path / "malformed.yaml"
    malformed_path.write_text("model:\n  architecture: [unclosed\n", encoding="utf-8")
    test_app = create_app(config_path=malformed_path)

    with pytest.raises(ConfigError, match="YAML syntax error"):
        with TestClient(test_app):
            pass


# ------------------------------------------------------------------------------
# 9.C: Streaming Configuration Injection Tests
# ------------------------------------------------------------------------------


def test_streaming_session_injected_with_effective_config(tmp_path: Path) -> None:
    """Verify that validated effective configuration is injected into StreamingSession and reported in session_started."""
    custom_cfg = VoiceShieldConfig.model_validate(
        {
            "windowing": {
                "window_seconds": 4.0,
                "hop_seconds": 0.5,
                "min_speech_seconds": 0.5,
                "energy_threshold": 0.02,
            },
            "temporal_policy": {
                "threshold_high": 0.80,
                "threshold_low": 0.40,
                "smoothing_method": "median",
                "median_window_size": 7,
                "elevation_n": 4,
                "elevation_m": 6,
                "hysteresis_consecutive_drops": 2,
                "timing_gap_threshold_seconds": 0.40,
            },
        }
    )

    cfg_file = tmp_path / "streaming_config.yaml"
    dump_config(custom_cfg, cfg_file)
    expected_hash = custom_cfg.compute_hash()

    test_app = create_app(config_path=cfg_file)

    with patch.dict("os.environ", {"VOICESHIELD_MOCK_MODEL": "true"}):
        with TestClient(test_app) as client:
            with client.websocket_connect("/api/v1/stream") as ws:
                ws.send_json(
                    {
                        "type": "start",
                        "protocol_version": "1.0",
                        "sample_rate": 16000,
                        "channels": 1,
                        "pcm_encoding": "int16",
                    }
                )
                ack = ws.receive_json()
                assert ack["type"] == "session_started"
                assert ack["config_hash"] == expected_hash
                assert ack["window_seconds"] == 4.0
                assert ack["hop_seconds"] == 0.5
                assert ack["min_speech_seconds"] == 0.5

                # Inspect session in app.state
                session_id = ack["session_id"]
                session = test_app.state.active_streaming_sessions[session_id]
                assert session.config.windowing.hop_seconds == 0.5
                assert session.config.windowing.min_speech_seconds == 0.5
                assert session.config.temporal_policy.threshold_high == 0.80
                assert session.config.temporal_policy.smoothing_method == "median"
                assert session.temporal_policy.policy_cfg.median_window_size == 7
                assert session.temporal_policy.policy_cfg.elevation_n == 4
                assert session.temporal_policy.policy_cfg.elevation_m == 6
                assert session.temporal_policy.policy_cfg.timing_gap_threshold_seconds == 0.40
                assert session.vad.energy_threshold == 0.02


def test_streaming_fails_closed_without_app_state_config() -> None:
    """Verify that WebSocket route fails closed without fallback when app.state config is missing."""
    with patch.dict("os.environ", {"VOICESHIELD_MOCK_MODEL": "true"}):
        with TestClient(app) as client:
            # Deliberately remove effective_config
            saved_cfg = getattr(app.state, "effective_config", None)
            app.state.effective_config = None
            try:
                with client.websocket_connect("/api/v1/stream") as ws:
                    ws.send_json(
                        {
                            "type": "start",
                            "protocol_version": "1.0",
                            "sample_rate": 16000,
                            "channels": 1,
                            "pcm_encoding": "int16",
                        }
                    )
                    resp = ws.receive_json()
                    assert resp["type"] == "error"
                    assert resp["error_code"] == "CONFIG_UNAVAILABLE"
            finally:
                app.state.effective_config = saved_cfg


# ------------------------------------------------------------------------------
# 9.D: File Evaluation Injection and Override Semantics Tests
# ------------------------------------------------------------------------------


def test_file_eval_uses_yaml_defaults_when_request_params_omitted(tmp_path: Path) -> None:
    """Verify custom YAML hop_seconds and threshold take effect when REST query parameters are omitted."""
    custom_cfg = apply_config_overrides(
        VoiceShieldConfig(),
        hop_seconds=0.5,
        threshold=0.82,
    )
    cfg_file = tmp_path / "file_eval_config.yaml"
    dump_config(custom_cfg, cfg_file)
    expected_hash = custom_cfg.compute_hash()

    test_app = create_app(config_path=cfg_file)
    wav_bytes = _synthesize_test_wav(duration_seconds=5.0)

    with patch.dict("os.environ", {"VOICESHIELD_MOCK_MODEL": "true"}):
        with TestClient(test_app) as client:
            resp = client.post(
                "/api/v1/analyze/file",
                files={"file": ("test.wav", wav_bytes, "audio/wav")},
            )
            assert resp.status_code == 200, resp.text
            data = resp.json()

            # YAML values must have taken effect
            assert data["effective_threshold"] == 0.82
            assert data["config_hash"] == expected_hash
            assert data["metadata"]["config_hash"] == expected_hash
            assert data["metadata"]["effective_threshold"] == 0.82
            # 5.0s audio with 4.0s window and 0.5s hop produces 3 windows: [0, 4.0], [0.5, 4.5], [1.0, 5.0]
            assert data["total_windows"] == 3


def test_file_eval_explicit_override_wins_and_computes_new_effective_hash(tmp_path: Path) -> None:
    """Verify explicit REST request override produces a different, validated effective hash matching metadata."""
    custom_cfg = VoiceShieldConfig()
    base_hash = custom_cfg.compute_hash()
    cfg_file = tmp_path / "base_config.yaml"
    dump_config(custom_cfg, cfg_file)

    test_app = create_app(config_path=cfg_file)
    wav_bytes = _synthesize_test_wav(duration_seconds=5.0)

    with patch.dict("os.environ", {"VOICESHIELD_MOCK_MODEL": "true"}):
        with TestClient(test_app) as client:
            resp = client.post(
                "/api/v1/analyze/file?threshold=0.88&hop_seconds=0.5",
                files={"file": ("test.wav", wav_bytes, "audio/wav")},
            )
            assert resp.status_code == 200, resp.text
            data = resp.json()

            assert data["effective_threshold"] == 0.88
            override_hash = data["config_hash"]
            assert override_hash != base_hash
            assert data["metadata"]["config_hash"] == override_hash
            assert data["metadata"]["effective_threshold"] == 0.88
            assert data["total_windows"] == 3


def test_file_eval_direct_function_override_and_hash_parity() -> None:
    """Verify evaluate_audio uses base config on None and explicit overrides update the hash."""
    base_cfg = VoiceShieldConfig()
    base_hash = base_cfg.compute_hash()

    mock_raw = AASISTModel(AASIST_L_DEFAULT_CONFIG)
    detector = AASISTLDetector(_instantiated_model=mock_raw, preferred_device="cpu")
    wav_bytes = _synthesize_test_wav(duration_seconds=4.5)

    # 1. Without overrides
    res_default = evaluate_audio(source=wav_bytes, detector=detector, config=base_cfg)
    assert res_default.config_hash == base_hash
    assert res_default.effective_threshold == base_cfg.temporal_policy.threshold_high
    assert res_default.total_windows == 1  # 4.5s with 4.0s window and 1.0s hop = 1 window

    # 2. With hop_seconds override
    res_hop = evaluate_audio(source=wav_bytes, detector=detector, config=base_cfg, hop_seconds=0.5)
    assert res_hop.config_hash != base_hash
    assert res_hop.total_windows == 2  # [0, 4.0], [0.5, 4.5]


# ------------------------------------------------------------------------------
# 9.E: CLI Precedence Tests
# ------------------------------------------------------------------------------


def test_cli_omitted_flags_use_yaml_defaults(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify CLI loads canonical config and preserves effective hash when options are omitted."""
    custom_cfg = apply_config_overrides(VoiceShieldConfig(), threshold=0.77)
    cfg_file = tmp_path / "cli_cfg.yaml"
    dump_config(custom_cfg, cfg_file)
    expected_hash = custom_cfg.compute_hash()

    wav_file = tmp_path / "sample.wav"
    wav_file.write_bytes(_synthesize_test_wav(duration_seconds=4.5))

    ret = cli_main(
        [
            "--input",
            str(wav_file),
            "--config",
            str(cfg_file),
            "--mock-model",
        ]
    )
    assert ret == 0

    captured = capsys.readouterr().out
    assert f"Effective Config Hash: {expected_hash}" in captured


def test_cli_explicit_override_wins_and_changes_hash(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify explicit CLI option overrides YAML value and computes a new effective hash."""
    custom_cfg = VoiceShieldConfig()
    cfg_file = tmp_path / "cli_cfg.yaml"
    dump_config(custom_cfg, cfg_file)
    base_hash = custom_cfg.compute_hash()

    wav_file = tmp_path / "sample.wav"
    wav_file.write_bytes(_synthesize_test_wav(duration_seconds=4.5))

    ret = cli_main(
        [
            "--input",
            str(wav_file),
            "--config",
            str(cfg_file),
            "--threshold",
            "0.89",
            "--mock-model",
        ]
    )
    assert ret == 0

    captured = capsys.readouterr().out
    # Hash must change from base
    assert f"Effective Config Hash: {base_hash}" not in captured
    assert "Effective Config Hash: " in captured


# ------------------------------------------------------------------------------
# 9.F: Detector Injection and Fallback Tests
# ------------------------------------------------------------------------------


def test_detector_fallback_to_cpu_false_raises_when_mps_unavailable() -> None:
    """Verify AASISTLDetector with fallback_to_cpu=False raises RuntimeError when MPS is unavailable."""
    with patch("torch.backends.mps.is_available", return_value=False):
        with pytest.raises(
            RuntimeError, match="MPS device was requested, but MPS is not available"
        ):
            AASISTLDetector(
                _instantiated_model=AASISTModel(AASIST_L_DEFAULT_CONFIG),
                preferred_device="mps",
                fallback_to_cpu=False,
            )


def test_detector_fallback_to_cpu_false_does_not_silently_fallback_on_inference() -> None:
    """Verify AASISTLDetector with fallback_to_cpu=False re-raises exceptions rather than falling back to CPU."""
    mock_model = MagicMock()
    mock_model.eval = MagicMock()
    mock_model.to = MagicMock()
    mock_model.parameters.return_value = [torch.nn.Parameter(torch.zeros(1))]
    # Simulate a device error during forward pass
    mock_model.side_effect = RuntimeError("Hardware execution error")

    detector = AASISTLDetector(
        _instantiated_model=mock_model,
        preferred_device="cpu",
        fallback_to_cpu=False,
    )
    # Mock device type to non-cpu to verify behavior
    detector.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    dummy_audio = np.zeros(64600, dtype=np.float32)
    with pytest.raises(RuntimeError, match="Hardware execution error"):
        detector.predict(dummy_audio)


# ------------------------------------------------------------------------------
# 9.G: Readiness Consistency Tests
# ------------------------------------------------------------------------------


def test_readiness_hash_matches_stream_and_file_result(tmp_path: Path) -> None:
    """Verify /ready, session_started, and no-override file result expose the identical config hash."""
    custom_cfg = apply_config_overrides(VoiceShieldConfig(), threshold=0.74)
    cfg_file = tmp_path / "unified_config.yaml"
    dump_config(custom_cfg, cfg_file)
    expected_hash = custom_cfg.compute_hash()

    test_app = create_app(config_path=cfg_file)
    wav_bytes = _synthesize_test_wav(duration_seconds=4.5)

    with patch.dict("os.environ", {"VOICESHIELD_MOCK_MODEL": "true"}):
        with TestClient(test_app) as client:
            # 1. /ready endpoint
            ready_resp = client.get("/ready")
            assert ready_resp.status_code == 200
            ready_hash = ready_resp.json()["config_hash"]

            # 2. WebSocket session_started
            with client.websocket_connect("/api/v1/stream") as ws:
                ws.send_json(
                    {
                        "type": "start",
                        "protocol_version": "1.0",
                        "sample_rate": 16000,
                        "channels": 1,
                        "pcm_encoding": "int16",
                    }
                )
                ack = ws.receive_json()
                ws_hash = ack["config_hash"]

            # 3. File inference without request overrides
            file_resp = client.post(
                "/api/v1/analyze/file",
                files={"file": ("test.wav", wav_bytes, "audio/wav")},
            )
            assert file_resp.status_code == 200
            file_hash = file_resp.json()["config_hash"]

            # All three must match exactly
            assert ready_hash == expected_hash
            assert ws_hash == expected_hash
            assert file_hash == expected_hash


def test_ready_503_exposes_config_hash_when_checkpoint_missing(tmp_path: Path) -> None:
    """Verify GET /ready 503 detail exposes config_hash when config is loaded but detector is unavailable."""
    custom_cfg = VoiceShieldConfig()
    cfg_file = tmp_path / "ready_503.yaml"
    dump_config(custom_cfg, cfg_file)

    # In this test, VOICESHIELD_CHECKPOINT_PATH overrides model.checkpoint_path.
    # The effective hash reported in 503 detail must reflect that post-override configuration.
    override_cfg = apply_config_overrides(custom_cfg, checkpoint_path=str(tmp_path / "missing.pth"))
    expected_hash = override_cfg.compute_hash()

    test_app = create_app(config_path=cfg_file)

    # Ensure mock model is disabled so detector fails due to missing checkpoint
    with patch.dict(
        "os.environ",
        {
            "VOICESHIELD_CHECKPOINT_PATH": str(tmp_path / "missing.pth"),
            "VOICESHIELD_MOCK_MODEL": "false",
        },
    ):
        with TestClient(test_app) as client:
            ready_resp = client.get("/ready")
            assert ready_resp.status_code == 503
            detail = ready_resp.json()["detail"]
            assert detail["status"] == "not_ready"
            assert detail["config_hash"] == expected_hash


# ------------------------------------------------------------------------------
# 12. Corrective Pass Regressions (12.A - 12.K)
# ------------------------------------------------------------------------------


def test_12a_unrelated_cwd_shadow_cannot_alter_load_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """12.A: A temp CWD containing configs/default.yaml must NOT shadow load_config(None)."""
    fake_cfg_dir = tmp_path / "configs"
    fake_cfg_dir.mkdir(parents=True)
    rogue_yaml = fake_cfg_dir / "default.yaml"
    rogue_yaml.write_text(
        "audio:\n  sample_rate: 16000\n  channels: 1\n  dtype: float32\n  normalize_range: [-1.0, 1.0]\n"
        "windowing:\n  window_seconds: 4.0\n  hop_seconds: 1.0\n  min_speech_seconds: 0.8\n  energy_threshold: 0.015\n"
        "model:\n  architecture: aasist_l\n  checkpoint_path: rogue.pth\n  device_preferred: cpu\n  fallback_to_cpu: true\n"
        "temporal_policy:\n  threshold_high: 0.999\n  threshold_low: 0.001\n"
        "evaluation:\n  bootstrap_iterations: 100\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    loaded = load_config(None)
    canonical = VoiceShieldConfig()

    assert loaded.temporal_policy.threshold_high == canonical.temporal_policy.threshold_high
    assert loaded.compute_hash() == canonical.compute_hash()
    assert loaded.model.checkpoint_path == canonical.model.checkpoint_path


def test_12b_wheel_contains_canonical_yaml_and_loads_outside_repo(tmp_path: Path) -> None:
    """12.B: Wheel artifact contains canonical YAML, and load_config(None) works from unrelated CWD with matching hash."""
    import subprocess
    import sys
    import zipfile

    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path), "--offline"],
        check=True,
    )
    wheels = list(tmp_path.glob("*.whl"))
    assert len(wheels) == 1
    wheel_path = wheels[0]

    with zipfile.ZipFile(wheel_path, "r") as zf:
        namelist = zf.namelist()
        assert "voiceshield/config/default.yaml" in namelist
        pkg_yaml_bytes = zf.read("voiceshield/config/default.yaml")

    repo_yaml_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    repo_yaml_bytes = repo_yaml_path.read_bytes()
    assert pkg_yaml_bytes == repo_yaml_bytes

    # Extract wheel and run standalone script from an unrelated working directory
    extract_dir = tmp_path / "extracted"
    with zipfile.ZipFile(wheel_path, "r") as zf:
        zf.extractall(extract_dir)

    run_script = extract_dir / "verify_load.py"
    run_script.write_text(
        "import sys\n"
        f"sys.path.insert(0, r'{extract_dir}')\n"
        "from voiceshield.config.loader import load_config\n"
        "from voiceshield.config.schema import VoiceShieldConfig\n"
        "cfg = load_config(None)\n"
        "expected = VoiceShieldConfig()\n"
        "assert cfg.compute_hash() == expected.compute_hash(), f'{cfg.compute_hash()} != {expected.compute_hash()}'\n"
        "print('LOAD_SUCCESS')\n",
        encoding="utf-8",
    )
    unrelated_cwd = tmp_path / "unrelated_cwd"
    unrelated_cwd.mkdir()
    res = subprocess.run(
        [sys.executable, str(run_script)],
        cwd=str(unrelated_cwd),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "LOAD_SUCCESS" in res.stdout


def test_12c_effective_config_deep_immutability() -> None:
    """12.C: Root and nested assignments fail, and compute_hash rejects non-standard JSON numbers."""
    cfg = VoiceShieldConfig()
    original_hash = cfg.compute_hash()

    with pytest.raises(ValidationError):
        cfg.windowing.hop_seconds = 2.0

    with pytest.raises(ValidationError):
        cfg.temporal_policy.threshold_high = 0.9

    with pytest.raises(ValidationError):
        cfg.model.checkpoint_path = "changed.pth"

    with pytest.raises(ValidationError):
        cfg.audio.sample_rate = 8000

    with pytest.raises(ValidationError):
        cfg.windowing = WindowingConfig()

    # Hash remains untouched
    assert cfg.compute_hash() == original_hash

    # compute_hash rejects NaN/Infinity if bypass occurred
    object.__setattr__(cfg.windowing, "hop_seconds", float("nan"))
    with pytest.raises(ValueError, match="Out of range float values"):
        cfg.compute_hash()


def test_12d_file_api_missing_or_corrupted_config_fails_closed_503(tmp_path: Path) -> None:
    """12.D: Missing or invalid config returns structured 503 CONFIG_UNAVAILABLE and reservation is released."""
    test_app = create_app()
    wav_bytes = _synthesize_test_wav(duration_seconds=4.5)

    with patch.dict("os.environ", {"VOICESHIELD_MOCK_MODEL": "true"}):
        with TestClient(test_app) as client:
            # 1. Missing config on app.state
            test_app.state.effective_config = None
            resp = client.post(
                "/api/v1/analyze/file",
                files={"file": ("test.wav", wav_bytes, "audio/wav")},
            )
            assert resp.status_code == 503
            assert resp.json()["detail"]["error_code"] == "CONFIG_UNAVAILABLE"
            assert test_app.state.file_upload_admission.reserved_count == 0

            # 2. Corrupted hash mismatch on app.state
            test_app.state.effective_config = VoiceShieldConfig()
            test_app.state.effective_config_hash = "mismatched_corrupt_hash"
            resp2 = client.post(
                "/api/v1/analyze/file",
                files={"file": ("test.wav", wav_bytes, "audio/wav")},
            )
            assert resp2.status_code == 503
            assert resp2.json()["detail"]["error_code"] == "CONFIG_UNAVAILABLE"
            assert test_app.state.file_upload_admission.reserved_count == 0


def test_12e_invalid_cross_field_rest_override_returns_422(tmp_path: Path) -> None:
    """12.E: REST query override exceeding custom window returns HTTP 422 with INVALID_CONFIG_OVERRIDE and zero leaks."""
    custom_cfg = VoiceShieldConfig.model_validate(
        {
            "windowing": {
                "window_seconds": 1.0,
                "hop_seconds": 0.5,
                "min_speech_seconds": 0.5,
            }
        }
    )
    cfg_file = tmp_path / "small_window.yaml"
    dump_config(custom_cfg, cfg_file)

    test_app = create_app(config_path=cfg_file)
    wav_bytes = _synthesize_test_wav(duration_seconds=3.0)

    with patch.dict("os.environ", {"VOICESHIELD_MOCK_MODEL": "true"}):
        with TestClient(test_app) as client:
            resp = client.post(
                "/api/v1/analyze/file?hop_seconds=2.0",
                files={"file": ("test.wav", wav_bytes, "audio/wav")},
            )
            assert resp.status_code == 422
            data = resp.json()
            assert data["detail"]["error_code"] == "INVALID_CONFIG_OVERRIDE"
            assert (
                "Hop duration (2.0s) cannot be greater than window duration (1.0s)"
                in data["detail"]["message"]
            )
            assert test_app.state.file_upload_admission.reserved_count == 0


def test_12f_invalid_cli_override_clean_exit_without_detector(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """12.F: Invalid CLI override exits with code 2 without traceback and does not construct detector."""
    custom_cfg = VoiceShieldConfig.model_validate(
        {
            "windowing": {
                "window_seconds": 1.0,
                "hop_seconds": 0.5,
                "min_speech_seconds": 0.5,
            }
        }
    )
    cfg_file = tmp_path / "small_window.yaml"
    dump_config(custom_cfg, cfg_file)
    wav_file = tmp_path / "test.wav"
    wav_file.write_bytes(_synthesize_test_wav(duration_seconds=3.0))

    with patch("voiceshield.pipeline.cli.AASISTLDetector") as mock_det:
        ret = cli_main(
            [
                "--input",
                str(wav_file),
                "--config",
                str(cfg_file),
                "--hop-seconds",
                "2.0",
            ]
        )
        assert ret == 2
        mock_det.assert_not_called()

    captured = capsys.readouterr()
    assert "CONFIG ERROR: Invalid configuration override" in captured.err
    assert "Traceback" not in captured.err


def test_12g_tensor_transfer_and_forward_fallback_matrix() -> None:
    """12.G: Device transfer and model forward fallback matrix truthfully moves or retains device state."""
    mock_raw = AASISTModel(AASIST_L_DEFAULT_CONFIG)
    test_tensor = torch.randn(1, 64600)
    orig_to = torch.Tensor.to

    transfer_fails = False

    def custom_to(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if len(args) > 0 and isinstance(args[0], torch.device) and args[0].type == "mps":
            if transfer_fails:
                raise RuntimeError("MPS Out of Memory")
            # Return self so tensor can be processed even without MPS hardware support
            return self
        return orig_to(self, *args, **kwargs)

    # Case 1: Device transfer failure + fallback_to_cpu=True -> moves model to CPU and reports cpu
    transfer_fails = True
    det_transfer_fb = AASISTLDetector(
        _instantiated_model=mock_raw, preferred_device="cpu", fallback_to_cpu=True
    )
    det_transfer_fb.device = torch.device("mps")

    with patch.object(torch.Tensor, "to", custom_to):
        out1 = det_transfer_fb.predict_batch(test_tensor)
        assert len(out1) == 1
        assert out1[0].device_used == "cpu"
        assert det_transfer_fb.device.type == "cpu"
        assert out1[0].metadata["fallback_used"] is True

    # Case 2: Device transfer failure + fallback_to_cpu=False -> re-raises and does NOT move to CPU
    det_transfer_no_fb = AASISTLDetector(
        _instantiated_model=mock_raw, preferred_device="cpu", fallback_to_cpu=False
    )
    det_transfer_no_fb.device = torch.device("mps")

    with patch.object(torch.Tensor, "to", custom_to):
        with pytest.raises(RuntimeError, match="MPS Out of Memory"):
            det_transfer_no_fb.predict_batch(test_tensor)
        assert det_transfer_no_fb.device.type == "mps"

    # Case 3: Model forward failure + fallback_to_cpu=True -> moves model to CPU and reports cpu
    transfer_fails = False
    det_forward_fb = AASISTLDetector(
        _instantiated_model=mock_raw, preferred_device="cpu", fallback_to_cpu=True
    )
    det_forward_fb.device = torch.device("mps")
    attempts = 0
    orig_forward = mock_raw.forward

    def fail_first_forward(x: torch.Tensor) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("MPS forward execution kernel failure")
        return orig_forward(x)

    with (
        patch.object(torch.Tensor, "to", custom_to),
        patch.object(mock_raw, "forward", fail_first_forward),
    ):
        out3 = det_forward_fb.predict_batch(test_tensor)
        assert len(out3) == 1
        assert out3[0].device_used == "cpu"
        assert det_forward_fb.device.type == "cpu"
        assert out3[0].metadata["fallback_used"] is True
        assert attempts == 2

    # Case 4: Model forward failure + fallback_to_cpu=False -> re-raises and does NOT move to CPU
    det_forward_no_fb = AASISTLDetector(
        _instantiated_model=mock_raw, preferred_device="cpu", fallback_to_cpu=False
    )
    det_forward_no_fb.device = torch.device("mps")

    with (
        patch.object(torch.Tensor, "to", custom_to),
        patch.object(mock_raw, "forward", side_effect=RuntimeError("MPS forward failed")),
    ):
        with pytest.raises(RuntimeError, match="MPS forward failed"):
            det_forward_no_fb.predict_batch(test_tensor)
        assert det_forward_no_fb.device.type == "mps"


def test_12h_readiness_reports_truthful_actual_device() -> None:
    """12.H: /ready success reports the actual detector device after placement or fallback."""
    orig_to = torch.Tensor.to

    def custom_to(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if len(args) > 0 and isinstance(args[0], torch.device) and args[0].type == "mps":
            raise RuntimeError("MPS failed")
        return orig_to(self, *args, **kwargs)

    # 1. Config requests CPU on MPS-capable host -> reports cpu
    with patch("voiceshield.core.device.torch.backends.mps.is_available", return_value=True):
        with patch.dict(
            "os.environ", {"VOICESHIELD_MOCK_MODEL": "true", "VOICESHIELD_DEVICE": "cpu"}
        ):
            test_app = create_app()
            with TestClient(test_app) as client:
                r = client.get("/ready")
                assert r.status_code == 200
                assert r.json()["device"] == "cpu"

                h = client.get("/health")
                assert h.status_code == 200
                assert h.json()["device"]["name"] == "cpu"
                assert h.json()["device"]["configured_preference"] == "cpu"

    # 2. MPS detector falls back to CPU -> /ready reports cpu
    with patch.dict("os.environ", {"VOICESHIELD_MOCK_MODEL": "true"}):
        test_app2 = create_app()
        with TestClient(test_app2) as client:
            det = test_app2.state.detector
            det.device = torch.device("mps")
            # Trigger fallback
            with patch.object(torch.Tensor, "to", custom_to):
                det.predict(np.zeros(64600, dtype=np.float32))

            assert det.device.type == "cpu"
            r2 = client.get("/ready")
            assert r2.status_code == 200
            assert r2.json()["device"] == "cpu"


def test_12i_custom_window_insufficient_audio_explanation() -> None:
    """12.I: Insufficient audio explanation dynamically reports window_samples and window_seconds from config."""
    custom_cfg = VoiceShieldConfig.model_validate(
        {
            "windowing": {
                "window_seconds": 2.0,
                "hop_seconds": 1.0,
                "min_speech_seconds": 0.5,
            }
        }
    )
    mock_raw = AASISTModel(AASIST_L_DEFAULT_CONFIG)
    detector = AASISTLDetector(_instantiated_model=mock_raw, preferred_device="cpu")
    short_audio = _synthesize_test_wav(duration_seconds=1.5)

    res = evaluate_audio(source=short_audio, detector=detector, config=custom_cfg)
    assert res.status == "insufficient_evidence"
    assert "fewer than 32,000 canonical contiguous samples (2.0s)" in (
        res.temporal_rule_explanation or ""
    )
    assert "64,000" not in (res.temporal_rule_explanation or "")
    assert "4.0s" not in (res.temporal_rule_explanation or "")


def test_12j_voiceshield_eval_cli_canonical_and_explicit_config(tmp_path: Path) -> None:
    """12.J: voiceshield-eval uses canonical config (500 bootstrap iterations) on omission, and explicit config overrides."""
    from voiceshield.evaluation.cli import main as eval_main

    manifest_file = tmp_path / "manifest.csv"
    manifest_file.write_text(
        "sample_id,label,speaker_id\n"
        "s1,spoof,spk1\n"
        "s2,bonafide,spk2\n"
        "s3,spoof,spk1\n"
        "s4,bonafide,spk2\n",
        encoding="utf-8",
    )
    scores_file = tmp_path / "scores.jsonl"
    scores_file.write_text(
        '{"sample_id": "s1", "score": 0.9}\n'
        '{"sample_id": "s2", "score": 0.1}\n'
        '{"sample_id": "s3", "score": 0.8}\n'
        '{"sample_id": "s4", "score": 0.2}\n',
        encoding="utf-8",
    )
    out_dir = tmp_path / "eval_out"

    with (
        patch("voiceshield.evaluation.cli.evaluate_deepfake_detector") as mock_eval,
        patch("voiceshield.evaluation.cli.save_evaluation_report"),
    ):
        dummy_report = MagicMock()
        dummy_report.overall.curves.eer = 0.05
        dummy_report.overall.curves.roc_auc = 0.95
        dummy_report.overall.dcf.min_dcf = 0.05
        dummy_report.overall.accuracy.accuracy = 0.95
        dummy_report.metadata = {"config_hash": VoiceShieldConfig().compute_hash()}
        mock_eval.return_value = dummy_report

        # 1. Omitted --config
        ret = eval_main(
            [
                "--manifest",
                str(manifest_file),
                "--scores",
                str(scores_file),
                "--output-dir",
                str(out_dir),
            ]
        )
        assert ret == 0
        mock_eval.assert_called_once()
        used_cfg = mock_eval.call_args.kwargs["config"]
        assert used_cfg is not None
        assert used_cfg.evaluation.bootstrap_iterations == 500
        assert used_cfg.compute_hash() == VoiceShieldConfig().compute_hash()

        # 2. Explicit --config wins
        custom_cfg = VoiceShieldConfig.model_validate({"evaluation": {"bootstrap_iterations": 200}})
        custom_cfg_file = tmp_path / "custom_eval.yaml"
        dump_config(custom_cfg, custom_cfg_file)

        mock_eval.reset_mock()
        ret2 = eval_main(
            [
                "--manifest",
                str(manifest_file),
                "--scores",
                str(scores_file),
                "--output-dir",
                str(out_dir),
                "--config",
                str(custom_cfg_file),
            ]
        )
        assert ret2 == 0
        used_cfg2 = mock_eval.call_args.kwargs["config"]
        assert used_cfg2.evaluation.bootstrap_iterations == 200
        assert used_cfg2.compute_hash() == custom_cfg.compute_hash()


def test_12k_no_config_disk_reads_during_requests_or_websocket(tmp_path: Path) -> None:
    """12.K: Verify no config YAML disk reads occur during normal file requests or WebSocket sessions after lifespan."""
    wav_bytes = _synthesize_test_wav(duration_seconds=4.5)
    test_app = create_app()

    with patch.dict("os.environ", {"VOICESHIELD_MOCK_MODEL": "true"}):
        with TestClient(test_app) as client:
            with patch("voiceshield.config.loader.open", wraps=open) as mock_open:
                # 1. Multiple file requests
                for _ in range(3):
                    r = client.post(
                        "/api/v1/analyze/file",
                        files={"file": ("test.wav", wav_bytes, "audio/wav")},
                    )
                    assert r.status_code == 200

                # 2. WebSocket session
                with client.websocket_connect("/api/v1/stream") as ws:
                    ws.send_json(
                        {
                            "type": "start",
                            "protocol_version": "1.0",
                            "sample_rate": 16000,
                            "channels": 1,
                            "pcm_encoding": "int16",
                        }
                    )
                    ack = ws.receive_json()
                    assert ack["type"] == "session_started"

                # Prove that open was never called for any config YAML file
                for call in mock_open.call_args_list:
                    opened_path = str(call[0][0])
                    assert not opened_path.endswith((".yaml", ".yml")), (
                        f"Config file was re-read from disk during request: {opened_path}"
                    )
