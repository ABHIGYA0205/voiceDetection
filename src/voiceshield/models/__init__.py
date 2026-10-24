"""Acoustic detector architectures, base contracts, and vendor adapters."""

from voiceshield.models.aasist_adapter import (
    AASIST_L_DEFAULT_CONFIG,
    AASISTLDetector,
    pad_for_aasist,
)
from voiceshield.models.base import BaseDetector, DetectorOutput
from voiceshield.models.provenance import ModelProvenance, ProvenanceStatus
from voiceshield.models.registry import (
    OFFICIAL_AASIST_L,
    OFFICIAL_MODEL_REGISTRY,
    CheckpointError,
    CheckpointIntegrityError,
    CheckpointLoadError,
    CheckpointNotFoundError,
    ModelRegistryEntry,
    ModelRegistryError,
    UnsupportedModelIdentityError,
    get_official_model_ids,
    get_official_registry_entry,
)

__all__ = [
    "AASISTLDetector",
    "AASIST_L_DEFAULT_CONFIG",
    "BaseDetector",
    "CheckpointError",
    "CheckpointIntegrityError",
    "CheckpointLoadError",
    "CheckpointNotFoundError",
    "DetectorOutput",
    "ModelProvenance",
    "ModelRegistryEntry",
    "ModelRegistryError",
    "OFFICIAL_AASIST_L",
    "OFFICIAL_MODEL_REGISTRY",
    "ProvenanceStatus",
    "UnsupportedModelIdentityError",
    "get_official_model_ids",
    "get_official_registry_entry",
    "pad_for_aasist",
]
