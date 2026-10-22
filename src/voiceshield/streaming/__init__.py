"""Streaming audio core for real-time VoiceShield synthetic speech detection."""

from __future__ import annotations

from voiceshield.streaming.policy import (
    DecisionState,
    ReasonCode,
    TemporalDecision,
    TemporalDecisionPolicy,
)
from voiceshield.streaming.resampler import StatefulResampler
from voiceshield.streaming.ring_buffer import AudioRingBuffer
from voiceshield.streaming.session import (
    AudioQualityFlags,
    IngestResult,
    StreamingSession,
    WindowForInference,
)
from voiceshield.streaming.vad import BaseVAD, EnergyVAD

__all__ = [
    "AudioQualityFlags",
    "AudioRingBuffer",
    "BaseVAD",
    "DecisionState",
    "EnergyVAD",
    "IngestResult",
    "ReasonCode",
    "StatefulResampler",
    "StreamingSession",
    "TemporalDecision",
    "TemporalDecisionPolicy",
    "WindowForInference",
]
