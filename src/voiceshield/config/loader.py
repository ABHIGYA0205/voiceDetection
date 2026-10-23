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


def find_default_config_path() -> Path:
    """Resolve canonical default configuration artifact without inspecting caller CWD.

    Precedence:
    1. Repository source tree artifact (<repo_root>/configs/default.yaml).
    2. Packaged resource artifact (voiceshield/config/default.yaml).

    Never inspects the current working directory for implicit default configuration.
    """
    # 1. Source-tree canonical location: <repo_root>/src/voiceshield/config/loader.py -> <repo_root>/configs/default.yaml
    repo_path = Path(__file__).resolve().parents[3] / "configs" / "default.yaml"
    if repo_path.is_file():
        return repo_path

    # 2. Packaged wheel/installed resource location: voiceshield/config/default.yaml
    pkg_path = Path(__file__).resolve().parent / "default.yaml"
    if pkg_path.is_file():
        return pkg_path

    # 3. Fallback via importlib.resources for zip/archive imports
    try:
        import importlib.resources as pkg_resources

        traversable = pkg_resources.files("voiceshield.config").joinpath("default.yaml")
        candidate = Path(str(traversable))
        if candidate.is_file():
            return candidate
    except Exception:
        pass

    raise ConfigError(
        f"Canonical default configuration file not found in repository ({repo_path}) "
        f"or package resources ({pkg_path})."
    )


def load_config(path: str | Path | None = None) -> VoiceShieldConfig:
    """Load and validate a VoiceShield configuration from a YAML file.

    Args:
        path: Path to the YAML file. If None, resolves the canonical default config file.

    Returns:
        Validated VoiceShieldConfig instance.

    Raises:
        ConfigError: If the file does not exist, YAML syntax is invalid, or schema validation fails.
    """
    if path is not None:
        target_path = Path(path)
        if not target_path.is_file():
            raise ConfigError(f"Configuration file not found: {target_path}")
    else:
        target_path = find_default_config_path()

    try:
        with open(target_path, encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML syntax error while reading {target_path}: {exc}") from exc

    if raw_data is None:
        raise ConfigError(f"Configuration file {target_path} is empty.")

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
