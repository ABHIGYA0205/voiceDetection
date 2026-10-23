"""Unit tests for configuration validation, YAML loading, and deterministic hashing."""

from pathlib import Path

import pytest
import yaml

from voiceshield.config import (
    AudioConfig,
    ConfigError,
    VoiceShieldConfig,
    WindowingConfig,
    compute_config_hash,
    dump_config,
    load_config,
)


def test_default_config_valid() -> None:
    """Verify that canonical configs/default.yaml loads and satisfies strict contract."""
    config = load_config(Path("configs/default.yaml"))

    assert config.audio.sample_rate == 16000
    assert config.audio.channels == 1
    assert config.audio.dtype == "float32"
    assert config.audio.normalize_range == (-1.0, 1.0)

    assert config.windowing.window_seconds == 4.0
    assert config.windowing.hop_seconds == 1.0
    assert config.windowing.min_speech_seconds == 0.8
    assert config.windowing.window_samples(16000) == 64000
    assert config.windowing.hop_samples(16000) == 16000


def test_invalid_sample_rate_raises() -> None:
    """Verify that an audio sample rate other than 16000 Hz raises validation error."""
    with pytest.raises(ValueError, match="strictly requires 16000 Hz"):
        AudioConfig(sample_rate=44100)

    with pytest.raises(ValueError, match="strictly requires 16000 Hz"):
        AudioConfig(sample_rate=8000)


def test_invalid_channels_raises() -> None:
    """Verify that multi-channel audio configuration raises validation error."""
    with pytest.raises(ValueError, match="strictly requires single-channel"):
        AudioConfig(channels=2)


def test_invalid_hop_window_relationship_raises() -> None:
    """Verify that a hop duration larger than window duration raises validation error."""
    with pytest.raises(ValueError, match="Hop duration .* cannot be greater than window duration"):
        WindowingConfig(window_seconds=4.0, hop_seconds=5.0)


def test_invalid_min_speech_relationship_raises() -> None:
    """Verify that min_speech_seconds exceeding window_seconds raises validation error."""
    with pytest.raises(
        ValueError, match="Minimum speech duration .* cannot exceed window duration"
    ):
        WindowingConfig(window_seconds=4.0, min_speech_seconds=4.5)


def test_deterministic_config_hashing() -> None:
    """Verify that config hashing is deterministic and sensitive to changes."""
    cfg1 = VoiceShieldConfig()
    cfg2 = VoiceShieldConfig()

    hash1 = cfg1.compute_hash()
    hash2 = cfg2.compute_hash()
    assert hash1 == hash2

    # Constructing a configuration with a modified field must alter the hash
    from voiceshield.config.overrides import apply_config_overrides

    cfg_altered = apply_config_overrides(cfg2, threshold=0.85)
    assert cfg_altered.compute_hash() != hash1


def test_missing_or_invalid_yaml_raises_config_error(tmp_path: Path) -> None:
    """Verify that missing files, malformed YAML, or schema violations raise ConfigError."""
    # 1. Non-existent file
    with pytest.raises(ConfigError, match="Configuration file not found"):
        load_config(tmp_path / "non_existent.yaml")

    # 2. Syntax error
    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("audio:\n  sample_rate: [unclosed list", encoding="utf-8")
    with pytest.raises(ConfigError, match="YAML syntax error"):
        load_config(malformed)

    # 3. Schema validation error with explicit field path
    invalid_schema = tmp_path / "invalid_schema.yaml"
    data = {
        "audio": {"sample_rate": 48000},  # Invalid sample rate
    }
    with open(invalid_schema, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)

    with pytest.raises(
        ConfigError, match=r"Validation failed for configuration[\s\S]*audio\.sample_rate"
    ):
        load_config(invalid_schema)


def test_dump_and_reload_roundtrip(tmp_path: Path) -> None:
    """Verify that dumping a config and reloading preserves values and hash."""
    from voiceshield.config.overrides import apply_config_overrides

    base = VoiceShieldConfig()
    orig = apply_config_overrides(base, device_preferred="cpu", threshold=0.8)

    dump_path = tmp_path / "custom_config.yaml"
    dump_config(orig, dump_path)

    reloaded = load_config(dump_path)
    assert reloaded.model.device_preferred == "cpu"
    assert reloaded.temporal_policy.threshold_high == 0.8
    assert reloaded.compute_hash() == orig.compute_hash()
    assert compute_config_hash(dump_path) == orig.compute_hash()
