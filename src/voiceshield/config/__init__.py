"""Configuration package for VoiceShield."""

from voiceshield.config.loader import (
    ConfigError,
    compute_config_hash,
    dump_config,
    load_config,
)
from voiceshield.config.overrides import apply_config_overrides
from voiceshield.config.schema import (
    AudioConfig,
    DCFConfig,
    EvaluationConfig,
    ModelConfig,
    TemporalPolicyConfig,
    VoiceShieldConfig,
    WindowingConfig,
)

__all__ = [
    "AudioConfig",
    "ConfigError",
    "DCFConfig",
    "EvaluationConfig",
    "ModelConfig",
    "TemporalPolicyConfig",
    "VoiceShieldConfig",
    "WindowingConfig",
    "apply_config_overrides",
    "compute_config_hash",
    "dump_config",
    "load_config",
]
