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
    custom_cfg = VoiceShieldConfig()
    custom_cfg.temporal_policy.threshold_high = 0.72
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
    custom_cfg = VoiceShieldConfig()
    custom_cfg.windowing.window_seconds = 4.0
    custom_cfg.windowing.hop_seconds = 0.5
    custom_cfg.windowing.min_speech_seconds = 0.5
    custom_cfg.windowing.energy_threshold = 0.02
    custom_cfg.temporal_policy.threshold_high = 0.80
    custom_cfg.temporal_policy.threshold_low = 0.40
    custom_cfg.temporal_policy.smoothing_method = "median"
    custom_cfg.temporal_policy.median_window_size = 7
    custom_cfg.temporal_policy.elevation_n = 4
    custom_cfg.temporal_policy.elevation_m = 6
    custom_cfg.temporal_policy.hysteresis_consecutive_drops = 2
    custom_cfg.temporal_policy.timing_gap_threshold_seconds = 0.40

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
    custom_cfg = VoiceShieldConfig()
    custom_cfg.windowing.hop_seconds = 0.5
    custom_cfg.temporal_policy.threshold_high = 0.82
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
    custom_cfg = VoiceShieldConfig()
    custom_cfg.temporal_policy.threshold_high = 0.77
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
    custom_cfg = VoiceShieldConfig()
    custom_cfg.temporal_policy.threshold_high = 0.74
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
    override_cfg = custom_cfg.model_copy(deep=True)
    override_cfg.model.checkpoint_path = str(tmp_path / "missing.pth")
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
