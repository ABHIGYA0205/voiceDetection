"""Framework-independent streaming audio session management and windowing."""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from voiceshield.config.schema import VoiceShieldConfig
from voiceshield.dsp.exceptions import InvalidAudioValueError, UnsupportedAudioError
from voiceshield.streaming.policy import (
    ReasonCode,
    TemporalDecision,
    TemporalDecisionPolicy,
)
from voiceshield.streaming.resampler import StatefulResampler
from voiceshield.streaming.ring_buffer import AudioRingBuffer
from voiceshield.streaming.vad import BaseVAD, EnergyVAD


class AudioQualityFlags(BaseModel):
    """Real-time audio signal quality diagnostics and anomaly flags."""

    is_clipped: bool = Field(default=False, description="Whether digital clipping was detected.")
    is_silent: bool = Field(
        default=False, description="Whether the audio frame is effectively silent."
    )
    has_timing_gap: bool = Field(
        default=False, description="Whether a timestamp gap exceeded threshold."
    )
    unsupported_input: bool = Field(
        default=False, description="Whether input frame had invalid format or non-finite values."
    )
    clipping_count: int = Field(default=0, description="Total clipped frames detected in stream.")
    gap_count: int = Field(default=0, description="Total timing discontinuities detected.")


class WindowForInference(BaseModel):
    """Eligible or abstained 4-second audio window prepared for inference."""

    model_config = {"arbitrary_types_allowed": True}

    window_index: int = Field(description="Sequential window index.")
    samples: np.ndarray[Any, np.dtype[np.float32]] = Field(
        description="1D canonical 16 kHz float32 audio samples."
    )
    start_time: float = Field(description="Stream start timestamp in seconds.")
    end_time: float = Field(description="Stream end timestamp in seconds.")
    speech_seconds: float = Field(description="Active speech duration in seconds.")
    is_eligible: bool = Field(
        description="Whether window passed speech duration gating and is eligible for model inference."
    )
    abstain_reason: ReasonCode | None = Field(
        default=None, description="Reason code if window was abstained from model inference."
    )
    quality_flags: AudioQualityFlags = Field(
        default_factory=AudioQualityFlags, description="Quality flags associated with this window."
    )


class IngestResult(BaseModel):
    """Result of ingesting an audio PCM frame into the streaming session."""

    model_config = {"arbitrary_types_allowed": True}

    session_id: str
    samples_ingested: int
    resampled_samples_produced: int
    current_stream_time: float
    window_ready: bool
    pending_window_available: bool
    immediate_decision: TemporalDecision | None = None
    quality_flags: AudioQualityFlags


class StreamingSession:
    """Isolated, framework-independent streaming audio session.

    Handles stateful resampling to 16 kHz mono, ring-buffered windowing, VAD gating,
    bounded backpressure queuing (at most 1 in-flight inference and 1 pending window),
    and temporal decision tracking.
    """

    def __init__(
        self,
        config: VoiceShieldConfig | None = None,
        vad: BaseVAD | None = None,
        session_id: str | None = None,
    ) -> None:
        """Initialize StreamingSession.

        Args:
            config: VoiceShield configuration object (defaults to canonical default config).
            vad: Replaceable Voice Activity Detection interface (defaults to EnergyVAD).
            session_id: Optional unique session identifier string.
        """
        self.config: VoiceShieldConfig = config or VoiceShieldConfig()
        self.session_id: str = session_id or str(uuid.uuid4())

        self.sample_rate: int = self.config.audio.sample_rate
        self.window_samples: int = self.config.windowing.window_samples(self.sample_rate)
        self.hop_samples: int = self.config.windowing.hop_samples(self.sample_rate)
        self.min_speech_seconds: float = self.config.windowing.min_speech_seconds

        # Replaceable VAD
        self.vad: BaseVAD = vad or EnergyVAD(
            energy_threshold=self.config.windowing.energy_threshold
        )

        # Ring buffer with 10.0-second bounded capacity
        capacity_samples = max(self.window_samples * 2, self.sample_rate * 10)
        self.ring_buffer = AudioRingBuffer(capacity_samples=capacity_samples)

        # Stateful resampler (lazily initialized upon first declared input format)
        self._resampler: StatefulResampler | None = None
        self._declared_sr: int | None = None
        self._declared_channels: int | None = None

        # Temporal Decision Policy
        self.temporal_policy = TemporalDecisionPolicy(
            policy_config=self.config.temporal_policy,
            windowing_config=self.config.windowing,
        )

        # Windowing and backpressure tracking
        self.samples_since_last_hop: int = 0
        self.window_index: int = 0
        self._stream_start_timestamp: float | None = None
        self._last_frame_timestamp: float | None = None
        self._expected_next_timestamp: float | None = None

        # Concurrency & bounded backpressure: at most 1 in-flight and 1 pending window
        self._in_flight_inference: bool = False
        self._pending_window: WindowForInference | None = None

        # Cumulative quality flags
        self.quality_flags = AudioQualityFlags()

    @property
    def in_flight_inference(self) -> bool:
        """Return whether model inference is currently in flight for this session."""
        return self._in_flight_inference

    @property
    def has_pending_window(self) -> bool:
        """Return whether an eligible window is waiting for inference."""
        return self._pending_window is not None

    def reset(self) -> None:
        """Reset the session state cleanly on disconnect or stream restart."""
        self.ring_buffer.reset()
        if self._resampler is not None:
            self._resampler.reset()
        self.temporal_policy.reset()

        self.samples_since_last_hop = 0
        self.window_index = 0
        self._stream_start_timestamp = None
        self._last_frame_timestamp = None
        self._expected_next_timestamp = None

        self._in_flight_inference = False
        self._pending_window = None

        self.quality_flags = AudioQualityFlags()

    def ingest_pcm(
        self,
        pcm_data: bytes | np.ndarray[Any, Any],
        sample_rate: int,
        channels: int = 1,
        timestamp: float | None = None,
        dtype: str = "int16",
    ) -> IngestResult:
        """Ingest timestamped PCM audio frame.

        Executes fast, synchronous DSP only (resampling, ring buffer write, VAD gating).
        Does NOT block on model inference.

        Args:
            pcm_data: Raw PCM audio bytes or NumPy array.
            sample_rate: Declared sampling rate in Hz of incoming audio.
            channels: Declared channel count (1=mono, 2=stereo).
            timestamp: External timestamp in seconds of this frame's first sample.
            dtype: PCM format if bytes ('int16', 'int32', or 'float32').

        Returns:
            IngestResult summarizing ingestion status, window availability, and quality flags.
        """
        frame_flags = AudioQualityFlags(
            clipping_count=self.quality_flags.clipping_count,
            gap_count=self.quality_flags.gap_count,
        )

        # 1. Validation & unsupported input detection
        if sample_rate <= 0 or channels <= 0:
            frame_flags.unsupported_input = True
            self.quality_flags.unsupported_input = True
            raise UnsupportedAudioError(
                f"Invalid audio parameters: sample_rate={sample_rate}, channels={channels}."
            )

        # 2. Timing gap detection
        gap_threshold = self.config.temporal_policy.timing_gap_threshold_seconds
        if timestamp is not None:
            if self._stream_start_timestamp is None:
                self._stream_start_timestamp = timestamp
            elif self._expected_next_timestamp is not None:
                delta = timestamp - self._expected_next_timestamp
                if delta > gap_threshold:
                    frame_flags.has_timing_gap = True
                    self.quality_flags.has_timing_gap = True
                    self.quality_flags.gap_count += 1
                    frame_flags.gap_count = self.quality_flags.gap_count
            self._last_frame_timestamp = timestamp

        # 3. Resampler configuration
        if (
            self._resampler is None
            or self._declared_sr != sample_rate
            or self._declared_channels != channels
        ):
            self._resampler = StatefulResampler(
                orig_sr=sample_rate,
                target_sr=self.sample_rate,
                channels=channels,
                dtype=dtype,
            )
            self._declared_sr = sample_rate
            self._declared_channels = channels

        # 4. Stateful resampling to canonical 16 kHz mono float32
        try:
            resampled = self._resampler.process(pcm_data, channels=channels, dtype=dtype)
        except (InvalidAudioValueError, UnsupportedAudioError):
            frame_flags.unsupported_input = True
            self.quality_flags.unsupported_input = True
            raise

        if resampled.size == 0:
            current_time = (
                self._stream_start_timestamp or 0.0
            ) + self.ring_buffer.total_samples_written / self.sample_rate
            return IngestResult(
                session_id=self.session_id,
                samples_ingested=0,
                resampled_samples_produced=0,
                current_stream_time=current_time,
                window_ready=False,
                pending_window_available=self._pending_window is not None,
                immediate_decision=None,
                quality_flags=frame_flags,
            )

        # 5. Quality checks on resampled frame (clipping and silence)
        max_abs = float(np.max(np.abs(resampled)))
        if max_abs >= 0.999:
            frame_flags.is_clipped = True
            self.quality_flags.is_clipped = True
            self.quality_flags.clipping_count += 1
            frame_flags.clipping_count = self.quality_flags.clipping_count

        rms = float(np.sqrt(np.mean(resampled.astype(np.float64) ** 2)))
        if rms < 1e-4:
            frame_flags.is_silent = True
            self.quality_flags.is_silent = True

        # Update expected timestamp for next frame
        num_raw_samples = (
            len(pcm_data) // (2 if dtype == "int16" else 4) // channels
            if isinstance(pcm_data, bytes)
            else (pcm_data.shape[0] if pcm_data.ndim == 1 else pcm_data.shape[-1])
        )
        frame_duration = num_raw_samples / sample_rate
        if timestamp is not None:
            self._expected_next_timestamp = timestamp + frame_duration

        # 6. Push to ring buffer
        self.ring_buffer.write(resampled)
        self.samples_since_last_hop += resampled.size

        current_stream_time = (
            self._stream_start_timestamp or 0.0
        ) + self.ring_buffer.total_samples_written / self.sample_rate

        # 7. Check if an eligible 4-second inference window is ready (every 1.0s hop)
        window_ready = False
        immediate_decision: TemporalDecision | None = None

        if (
            self.ring_buffer.available_samples >= self.window_samples
            and self.samples_since_last_hop >= self.hop_samples
        ):
            window_ready = True
            self.samples_since_last_hop = 0  # Advance hop step

            # Extract chronological 4-second window
            window_samples = self.ring_buffer.read_tail(self.window_samples)

            # Calculate window timestamps
            t_base = self._stream_start_timestamp or 0.0
            window_end = t_base + self.ring_buffer.total_samples_written / self.sample_rate
            window_start = window_end - (self.window_samples / self.sample_rate)

            # Compute speech duration with pluggable VAD
            speech_sec = self.vad.compute_speech_duration(window_samples, self.sample_rate)

            curr_window_index = self.window_index
            self.window_index += 1

            if speech_sec < self.min_speech_seconds:
                # Abstain from inference: active speech is below threshold
                window = WindowForInference(
                    window_index=curr_window_index,
                    samples=window_samples,
                    start_time=window_start,
                    end_time=window_end,
                    speech_seconds=speech_sec,
                    is_eligible=False,
                    abstain_reason=ReasonCode.INSUFFICIENT_SPEECH_DURATION,
                    quality_flags=frame_flags,
                )
                immediate_decision = self.temporal_policy.evaluate_insufficient_speech(
                    window_index=curr_window_index,
                    window_start_sec=window_start,
                    window_end_sec=window_end,
                    speech_seconds=speech_sec,
                )
            else:
                # Window is eligible for model inference
                window = WindowForInference(
                    window_index=curr_window_index,
                    samples=window_samples,
                    start_time=window_start,
                    end_time=window_end,
                    speech_seconds=speech_sec,
                    is_eligible=True,
                    abstain_reason=None,
                    quality_flags=frame_flags,
                )

                # Backpressure guard: at most one pending window.
                # If a previous pending window exists (whether or not inference is in flight),
                # replace it with this fresher window to prevent unbounded queueing.
                self._pending_window = window

        return IngestResult(
            session_id=self.session_id,
            samples_ingested=num_raw_samples,
            resampled_samples_produced=resampled.size,
            current_stream_time=current_stream_time,
            window_ready=window_ready,
            pending_window_available=self._pending_window is not None,
            immediate_decision=immediate_decision,
            quality_flags=frame_flags,
        )

    def pop_pending_window(self) -> WindowForInference | None:
        """Retrieve the oldest pending window eligible for inference and mark inference in-flight.

        Returns None if an inference is already in-flight or if no pending window is ready.
        """
        if self._in_flight_inference or self._pending_window is None:
            return None

        window = self._pending_window
        self._pending_window = None
        self._in_flight_inference = True
        return window

    def complete_inference(
        self,
        synthetic_score: float | None = None,
        window: WindowForInference | None = None,
        calibrated_score: float | None = None,
        score_is_calibrated: bool = False,
    ) -> TemporalDecision:
        """Complete an in-flight inference and update the temporal decision policy.

        Args:
            synthetic_score: Uncalibrated acoustic anomaly score [0.0, 1.0].
            window: The WindowForInference that was processed.
            calibrated_score: Optional calibrated probability (None if uncalibrated).
            score_is_calibrated: Whether score is empirically calibrated.

        Returns:
            Updated TemporalDecision from policy.
        """
        self._in_flight_inference = False
        assert window is not None

        decision = self.temporal_policy.evaluate(
            calibrated_score=calibrated_score,
            synthetic_score=synthetic_score,
            score_is_calibrated=score_is_calibrated,
            speech_seconds=window.speech_seconds,
            window_index=window.window_index,
            window_start_sec=window.start_time,
            window_end_sec=window.end_time,
        )
        return decision
