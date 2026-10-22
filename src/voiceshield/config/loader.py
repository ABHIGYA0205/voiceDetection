"""YAML loader and serializer for VoiceShield configurations with explicit validation errors."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import ValidationError

from voiceshield.config.schema import VoiceShieldConfig

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


class ConfigError(Exception):
    """Raised when configuration file loading or validation fails."""


def load_config(path: str | Path | None = None) -> VoiceShieldConfig:
    """Load and validate a VoiceShield configuration from a YAML file.

    Args:
        path: Path to the YAML file. If None, tries configs/default.yaml or canonical defaults.

    Returns:
        Validated VoiceShieldConfig instance.

    Raises:
        ConfigError: If the file does not exist, YAML syntax is invalid, or schema validation fails.
    """
    target_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH

    if not target_path.exists():
        if path is None:
            # Fall back to canonical defaults if default file not found
            logger.info(
                "Default config file %s not found; using built-in canonical defaults.", target_path
            )
            return VoiceShieldConfig()
        raise ConfigError(f"Configuration file not found: {target_path}")

    try:
        with open(target_path, encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML syntax error while reading {target_path}: {exc}") from exc

    if raw_data is None:
        return VoiceShieldConfig()

    if not isinstance(raw_data, dict):
        raise ConfigError(
            f"Configuration in {target_path} must be a YAML mapping (dictionary), got {type(raw_data).__name__}."
        )

    try:
        return VoiceShieldConfig.model_validate(raw_data)
    except ValidationError as exc:
        formatted_errors: list[str] = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"])
            msg = err["msg"]
            formatted_errors.append(f"  - [{loc}]: {msg}")
        joined = "\n".join(formatted_errors)
        raise ConfigError(f"Validation failed for configuration {target_path}:\n{joined}") from exc


def dump_config(config: VoiceShieldConfig, path: str | Path) -> None:
    """Serialize VoiceShield configuration to a YAML file.

    Args:
        config: Validated VoiceShieldConfig instance.
        path: Destination file path.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        yaml.safe_dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)


def compute_config_hash(config_or_path: VoiceShieldConfig | str | Path) -> str:
    """Compute the deterministic SHA-256 hash of a config instance or YAML file path."""
    if isinstance(config_or_path, (str, Path)):
        cfg = load_config(config_or_path)
    else:
        cfg = config_or_path
    return cfg.compute_hash()
