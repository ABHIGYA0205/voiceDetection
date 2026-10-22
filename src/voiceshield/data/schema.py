"""Manifest record schema and strict validation rules for dataset provenance."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

MANIFEST_VERSION = "1.0.0"

LabelType = Literal["bonafide", "spoof"]
AttackType = Literal["none", "TTS", "VC", "replay"]
SplitType = Literal["train", "dev", "test"]


class ManifestRecord(BaseModel):
    """A verified, versioned single audio sample entry in a VoiceShield manifest."""

    manifest_version: str = Field(default=MANIFEST_VERSION, description="Manifest schema version.")
    sample_id: str = Field(description="Globally unique sample identifier.")
    path: str = Field(description="Relative or absolute path to audio file.")
    label: LabelType = Field(
        description="Ground truth acoustic classification ('bonafide' or 'spoof')."
    )
    speaker_id: str = Field(description="Unique speaker identifier.")
    source_utterance_id: str | None = Field(
        default=None,
        description="Source transcript/utterance ID to prevent prompt leakage across splits.",
    )
    source_dataset: str = Field(
        description="Origin dataset name (e.g. 'ASVspoof2019', 'InHouseControlled')."
    )
    generator_family: str | None = Field(
        default=None,
        description="Synthesis or voice conversion engine family (e.g. 'ElevenLabs', 'VITS', 'Diff-TTS').",
    )
    generator_version: str | None = Field(
        default=None,
        description="Specific model version or release checkpoint.",
    )
    attack_type: AttackType = Field(
        description="Category of synthesis attack ('none', 'TTS', 'VC', 'replay').",
    )
    language: str = Field(default="en", description="Spoken language ISO code.")
    accent_locale: str | None = Field(default=None, description="Accent or locale if known.")
    original_sample_rate: int = Field(
        gt=0, description="Native sampling rate before resampling in Hz."
    )
    codec_channel_condition: str = Field(
        default="pcm_clean",
        description="Acoustic codec / channel condition (e.g. 'pcm_clean', 'g711_alaw', 'opus_16k').",
    )
    replay_condition: str = Field(
        default="none",
        description="Replay channel description ('none', 'simulated_room', 'physical_recording').",
    )
    duration: float = Field(gt=0, description="Audio duration in seconds.")
    consent_licence_ref: str = Field(
        description="Dataset licence or contributor consent identifier.",
    )
    split: SplitType | None = Field(
        default=None,
        description="Assigned data split ('train', 'dev', 'test').",
    )
    sha256: str = Field(
        min_length=64,
        max_length=64,
        description="64-character lowercase hexadecimal SHA-256 checksum of raw audio file.",
    )

    @model_validator(mode="after")
    def validate_label_consistency(self) -> ManifestRecord:
        """Enforce consistency between label, attack_type, and generator_family."""
        if self.label == "bonafide":
            if self.attack_type != "none":
                raise ValueError(
                    f"Sample '{self.sample_id}' has label 'bonafide' but attack_type is '{self.attack_type}'. "
                    "Bona fide speech must have attack_type='none'."
                )
            if self.generator_family is not None:
                raise ValueError(
                    f"Sample '{self.sample_id}' has label 'bonafide' but specifies generator_family '{self.generator_family}'. "
                    "Bona fide speech must not specify a generator family."
                )
        elif self.label == "spoof":
            if self.attack_type == "none":
                raise ValueError(
                    f"Sample '{self.sample_id}' has label 'spoof' but attack_type is 'none'. "
                    "Spoof speech must specify attack_type ('TTS', 'VC', 'replay')."
                )
            if self.attack_type in ("TTS", "VC") and not self.generator_family:
                raise ValueError(
                    f"Sample '{self.sample_id}' has attack_type '{self.attack_type}' but missing generator_family."
                )
        return self
