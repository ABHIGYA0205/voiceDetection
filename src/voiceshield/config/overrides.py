"""Centralized, typed configuration override helpers for VoiceShield."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from voiceshield.config.loader import ConfigError
from voiceshield.config.schema import VoiceShieldConfig


def apply_config_overrides(
    base_config: VoiceShieldConfig,
    checkpoint_path: str | Path | None = None,
    device_preferred: str | None = None,
    hop_seconds: float | None = None,
    window_seconds: float | None = None,
    min_speech_seconds: float | None = None,
    threshold: float | None = None,
) -> VoiceShieldConfig:
    """Construct a new validated, immutable VoiceShieldConfig with selective overrides.

    Preserves the base config unchanged without in-place mutation.
    Overridden fields are validated atomically through VoiceShieldConfig schema validation.

    Invariant preservation rule for threshold:
    When threshold is provided, threshold_high is set to float(threshold), and
    threshold_low is set to min(base_config.temporal_policy.threshold_low, new_threshold_high),
    preserving the invariant threshold_low <= threshold_high.

    Args:
        base_config: Source immutable VoiceShieldConfig.
        checkpoint_path: Optional override for model.checkpoint_path.
        device_preferred: Optional override for model.device_preferred ('mps' or 'cpu').
        hop_seconds: Optional override for windowing.hop_seconds.
        window_seconds: Optional override for windowing.window_seconds.
        min_speech_seconds: Optional override for windowing.min_speech_seconds.
        threshold: Optional override for temporal_policy.threshold_high.

    Returns:
        A new frozen VoiceShieldConfig instance.

    Raises:
        ConfigError: If any override value violates field or cross-field validation rules.
    """
    model_updates: dict[str, Any] = {}
    if checkpoint_path is not None:
        chk_str = str(checkpoint_path).strip()
        if not chk_str:
            raise ConfigError("Invalid configuration override: checkpoint_path cannot be empty.")
        model_updates["checkpoint_path"] = chk_str

    if device_preferred is not None:
        dev_str = device_preferred.strip().lower()
        if not dev_str:
            raise ConfigError("Invalid configuration override: device_preferred cannot be empty.")
        model_updates["device_preferred"] = dev_str

    window_updates: dict[str, Any] = {}
    if hop_seconds is not None:
        window_updates["hop_seconds"] = float(hop_seconds)
    if window_seconds is not None:
        window_updates["window_seconds"] = float(window_seconds)
    if min_speech_seconds is not None:
        window_updates["min_speech_seconds"] = float(min_speech_seconds)

    policy_updates: dict[str, Any] = {}
    if threshold is not None:
        new_th = float(threshold)
        policy_updates["threshold_high"] = new_th
        policy_updates["threshold_low"] = min(base_config.temporal_policy.threshold_low, new_th)

    if not model_updates and not window_updates and not policy_updates:
        return base_config

    root_dict = base_config.model_dump()
    if model_updates:
        root_dict["model"].update(model_updates)
    if window_updates:
        root_dict["windowing"].update(window_updates)
    if policy_updates:
        root_dict["temporal_policy"].update(policy_updates)

    try:
        return VoiceShieldConfig.model_validate(root_dict)
    except (ValidationError, ValueError) as exc:
        context_parts: list[str] = []
        if checkpoint_path is not None:
            context_parts.append(f"checkpoint_path={checkpoint_path!r}")
        if device_preferred is not None:
            context_parts.append(f"device_preferred={device_preferred!r}")
        if hop_seconds is not None:
            context_parts.append(f"hop_seconds={hop_seconds}")
        if window_seconds is not None:
            context_parts.append(f"window_seconds={window_seconds}")
        if min_speech_seconds is not None:
            context_parts.append(f"min_speech_seconds={min_speech_seconds}")
        if threshold is not None:
            context_parts.append(f"threshold={threshold}")
        ctx_desc = ", ".join(context_parts) if context_parts else "unspecified"

        if isinstance(exc, ValidationError):
            formatted_errors: list[str] = []
            for err in exc.errors():
                loc = ".".join(str(p) for p in err["loc"])
                msg = err["msg"]
                formatted_errors.append(f"Field '{loc}': {msg}")
            joined = "; ".join(formatted_errors)
            raise ConfigError(f"Invalid configuration override ({ctx_desc}): {joined}") from exc
        raise ConfigError(f"Invalid configuration override ({ctx_desc}): {exc}") from exc
