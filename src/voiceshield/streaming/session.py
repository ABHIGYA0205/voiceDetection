"""Framework-independent streaming audio session management and windowing."""

from __future__ import annotations

import math
import uuid
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from voiceshield.config.schema import VoiceShieldConfig
from voiceshield.dsp.exceptions import (
    AudioDecodingError,
    CorruptAudioError,
    EmptyAudioError,
    InvalidAudioValueError,
    UnsupportedAudioError,
)
from voiceshield.dsp.scheduler import CanonicalWindowScheduler
from voiceshield.streaming.policy import (
    ReasonCode,
    TemporalDecision,
    TemporalDecisionPolicy,
)
from voiceshield.streaming.resampler import (
    MAX_SUPPORTED_SAMPLE_RATE,
    MIN_SUPPORTED_SAMPLE_RATE,
    StatefulResampler,
    validate_and_decode_pcm,
)
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
    continuity_segment_id: int = Field(
        default=0, description="Monotonically increasing continuity segment identifier."
    )
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
    windows_generated_this_ingest: int = 0
    generated_window_indices: list[int] = Field(default_factory=list)
    generated_window_end_times: list[float] = Field(default_factory=list)
    immediate_decisions: list[TemporalDecision] = Field(default_factory=list)
    immediate_decision: TemporalDecision | None = None
    windows_dropped_this_ingest: int = 0
    total_windows_generated: int = 0
    total_windows_dropped: int = 0
    quality_flags: AudioQualityFlags
    continuity_reset: bool = False
    continuity_segment_id: int = 0
    timestamp_delta: float = 0.0


class StreamingSession:
    """Isolated, framework-independent streaming audio session.

    Handles stateful resampling to 16 kHz mono, shared canonical ring-buffered windowing,
    shared EnergyVAD gating, bounded backpressure queuing (at most 1 in-flight inference
    and 1 pending window), continuity segment resets, and temporal decision tracking.
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

        # Canonical sample-grid window scheduler with 10.0-second bounded capacity
        capacity_samples = max(self.window_samples * 2, self.sample_rate * 10)
        self.scheduler = CanonicalWindowScheduler(
            sample_rate=self.sample_rate,
            window_samples=self.window_samples,
            hop_samples=self.hop_samples,
            capacity_samples=capacity_samples,
        )

        # Stateful resampler (lazily initialized upon first declared input format)
        self._resampler: StatefulResampler | None = None
        self._declared_sr: int | None = None
        self._declared_channels: int | None = None
        self._declared_dtype: str | None = None

        # Temporal Decision Policy
        self.temporal_policy = TemporalDecisionPolicy(
            policy_config=self.config.temporal_policy,
            windowing_config=self.config.windowing,
        )

        self.total_windows_dropped: int = 0

        self._stream_start_timestamp: float | None = None
        self._segment_start_timestamp: float | None = None
        self._last_frame_timestamp: float | None = None
        self._expected_next_timestamp: float | None = None

        # Concurrency & bounded backpressure: at most 1 in-flight and 1 pending window
        self._in_flight_inference: bool = False
        self._in_flight_window: WindowForInference | None = None
        self._in_flight_window_stale: bool = False
        self._pending_window: WindowForInference | None = None

        # Cumulative quality flags
        self.quality_flags = AudioQualityFlags()
        self._is_finalized: bool = False

    @property
    def is_finalized(self) -> bool:
        """Return whether the session has been finalized at end-of-stream."""
        return self._is_finalized

    @property
    def ring_buffer(self) -> AudioRingBuffer:
        return self.scheduler.ring_buffer

    @property
    def segment_samples_written(self) -> int:
        return self.scheduler.segment_samples_written

    @segment_samples_written.setter
    def segment_samples_written(self, val: int) -> None:
        self.scheduler.segment_samples_written = val

    @property
    def total_canonical_samples_written(self) -> int:
        return self.scheduler.total_samples_written

    @property
    def next_window_end_sample(self) -> int:
        return self.scheduler.next_window_end_sample

    @next_window_end_sample.setter
    def next_window_end_sample(self, val: int) -> None:
        self.scheduler.next_window_end_sample = val

    @property
    def window_index(self) -> int:
        return self.scheduler.window_index

    @window_index.setter
    def window_index(self, val: int) -> None:
        self.scheduler.window_index = val

    @property
    def continuity_segment_id(self) -> int:
        return self.scheduler.continuity_segment_id

    @continuity_segment_id.setter
    def continuity_segment_id(self, val: int) -> None:
        self.scheduler.continuity_segment_id = val

    @property
    def total_windows_generated(self) -> int:
        return self.scheduler.total_windows_generated

    @total_windows_generated.setter
    def total_windows_generated(self, val: int) -> None:
        self.scheduler.total_windows_generated = val

    @property
    def samples_since_last_hop(self) -> int:
        return self.scheduler.samples_since_last_hop

    @property
    def in_flight_inference(self) -> bool:
        """Return whether model inference is currently in flight for this session."""
        return self._in_flight_inference

    @property
    def in_flight_window(self) -> WindowForInference | None:
        """Return the WindowForInference currently being processed by model inference."""
        return self._in_flight_window

    @property
    def has_pending_window(self) -> bool:
        """Return whether an eligible window is waiting for inference."""
        return self._pending_window is not None

    def reset(self) -> None:
        """Reset the session state cleanly on disconnect or stream restart."""
        self.scheduler.reset()
        if self._resampler is not None:
            self._resampler.reset()
        self.temporal_policy.reset()
        self.vad.reset()

        self.total_windows_dropped = 0

        self._stream_start_timestamp = None
        self._segment_start_timestamp = None
        self._last_frame_timestamp = None
        self._expected_next_timestamp = None

        self._declared_sr = None
        self._declared_channels = None
        self._declared_dtype = None

        self._in_flight_inference = False
        self._in_flight_window = None
        self._in_flight_window_stale = False
        self._pending_window = None

        self.quality_flags = AudioQualityFlags()
        self._is_finalized = False

    def finalize(self) -> tuple[list[WindowForInference], list[TemporalDecision]]:
        """Idempotently finalize the streaming session at end-of-stream.

        Flushes the canonical resampler, passes flushed canonical samples into the scheduler,
        generates any newly completed full windows and immediate abstentions, and marks the session finalized.

        Returns:
            tuple[list[WindowForInference], list[TemporalDecision]]: Newly completed eligible windows
            and immediate abstentions resulting from resampler flush.
        """
        if self._is_finalized:
            return [], []

        self._is_finalized = True

        new_eligible: list[WindowForInference] = []
        new_abstentions: list[TemporalDecision] = []

        if self._resampler is None:
            return new_eligible, new_abstentions

        flushed = self._resampler.flush()
        if flushed.size == 0:
            return new_eligible, new_abstentions

        due_windows = self.scheduler.push_samples(flushed)
        for lw in due_windows:
            speech_sec = self.vad.compute_speech_duration(lw.samples, self.sample_rate)
            if speech_sec < self.min_speech_seconds:
                imm = self.temporal_policy.evaluate_insufficient_speech(
                    window_index=lw.window_index,
                    window_start_sec=lw.start_time,
                    window_end_sec=lw.end_time,
                    speech_seconds=speech_sec,
                )
                new_abstentions.append(imm)
            else:
                win = WindowForInference(
                    window_index=lw.window_index,
                    continuity_segment_id=lw.continuity_segment_id,
                    samples=lw.samples,
                    start_time=lw.start_time,
                    end_time=lw.end_time,
                    speech_seconds=speech_sec,
                    is_eligible=True,
                    abstain_reason=None,
                    quality_flags=self.quality_flags,
                )
                new_eligible.append(win)
                if self._pending_window is not None:
                    self.total_windows_dropped += 1
                self._pending_window = win

        return new_eligible, new_abstentions

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
        if self._is_finalized:
            raise RuntimeError(
                "Cannot ingest audio into a finalized StreamingSession. Call reset() first."
            )

        # 1. Non-mutating validation & unsupported input detection
        if sample_rate < MIN_SUPPORTED_SAMPLE_RATE or sample_rate > MAX_SUPPORTED_SAMPLE_RATE:
            self.quality_flags.unsupported_input = True
            raise UnsupportedAudioError(
                f"Unsupported sampling rate: {sample_rate} Hz. "
                f"Permitted range is {MIN_SUPPORTED_SAMPLE_RATE:,} Hz to {MAX_SUPPORTED_SAMPLE_RATE:,} Hz."
            )

        if channels not in (1, 2):
            self.quality_flags.unsupported_input = True
            raise UnsupportedAudioError(
                f"Unsupported channel count: {channels}. Permitted channels are 1 or 2."
            )

        norm_dtype = str(dtype).lower()
        if norm_dtype not in ("int16", "int32", "float32", "float"):
            self.quality_flags.unsupported_input = True
            raise UnsupportedAudioError(f"Unsupported PCM encoding: {dtype!r}.")

        # Enforce format immutability once accepted
        if self._declared_sr is not None:
            if (
                sample_rate != self._declared_sr
                or channels != self._declared_channels
                or norm_dtype != self._declared_dtype
            ):
                self.quality_flags.unsupported_input = True
                raise UnsupportedAudioError(
                    f"Mid-stream audio format change rejected. Session locked to "
                    f"{self._declared_sr} Hz, {self._declared_channels} ch, {self._declared_dtype}; "
                    f"received {sample_rate} Hz, {channels} ch, {norm_dtype}."
                )

        # Strictly validate and decode input without mutating accepted session state
        try:
            mono_tensor, num_raw_samples = validate_and_decode_pcm(
                pcm_data, channels=channels, dtype=norm_dtype
            )
        except (
            InvalidAudioValueError,
            UnsupportedAudioError,
            EmptyAudioError,
            CorruptAudioError,
            AudioDecodingError,
        ):
            self.quality_flags.unsupported_input = True
            raise
        except Exception as exc:
            self.quality_flags.unsupported_input = True
            raise AudioDecodingError(f"Failed to inspect audio frame: {exc}") from exc

        if num_raw_samples == 0:
            self.quality_flags.unsupported_input = True
            raise EmptyAudioError("Incoming audio frame contains zero samples.")

        # Input is valid: lock format on first accepted frame
        if self._declared_sr is None:
            self._declared_sr = sample_rate
            self._declared_channels = channels
            self._declared_dtype = norm_dtype

        # 2. Timing gap & discontinuity detection
        gap_threshold = self.config.temporal_policy.timing_gap_threshold_seconds
        is_discontinuity = False
        timestamp_delta = 0.0

        if timestamp is not None:
            if not math.isfinite(timestamp):
                is_discontinuity = True
                timestamp_delta = float("nan")
            elif self._stream_start_timestamp is None:
                self._stream_start_timestamp = timestamp
                self._segment_start_timestamp = timestamp
                self.scheduler.segment_start_timestamp = timestamp
            elif self._expected_next_timestamp is not None:
                delta = timestamp - self._expected_next_timestamp
                timestamp_delta = delta
                if (
                    self._last_frame_timestamp is not None
                    and timestamp < self._last_frame_timestamp - 1e-6
                ):
                    # Out-of-order timestamp
                    is_discontinuity = True
                elif delta > gap_threshold:
                    # Positive timing gap
                    is_discontinuity = True
                elif delta < -gap_threshold:
                    # Materially overlapping timestamp
                    is_discontinuity = True

        windows_dropped_this_ingest = 0
        frame_flags = AudioQualityFlags(
            clipping_count=self.quality_flags.clipping_count,
            gap_count=self.quality_flags.gap_count,
        )

        if is_discontinuity:
            frame_flags.has_timing_gap = True
            self.quality_flags.has_timing_gap = True
            self.quality_flags.gap_count += 1
            frame_flags.gap_count = self.quality_flags.gap_count

            # Continuity segment reset: resets scheduler, resampler, policy, vad
            new_segment_id = self.continuity_segment_id + 1
            if timestamp is not None and math.isfinite(timestamp):
                self._segment_start_timestamp = timestamp
            else:
                self._segment_start_timestamp = self._expected_next_timestamp or 0.0

            self.scheduler.reset_continuity(
                segment_id=new_segment_id,
                segment_start_timestamp=self._segment_start_timestamp,
            )

            if self._resampler is not None:
                self._resampler.reset()
            self.temporal_policy.reset()
            self.vad.reset()
            self._is_finalized = False

            # Discard and count stale in-flight window
            if self._in_flight_window is not None and not self._in_flight_window_stale:
                self.total_windows_dropped += 1
                windows_dropped_this_ingest += 1
                self._in_flight_window_stale = True

            # Discard and count stale pending window
            if self._pending_window is not None:
                self.total_windows_dropped += 1
                windows_dropped_this_ingest += 1
                self._pending_window = None

        # Update expected timestamp for next frame
        frame_duration = num_raw_samples / sample_rate
        if timestamp is not None and math.isfinite(timestamp):
            self._last_frame_timestamp = timestamp
            self._expected_next_timestamp = timestamp + frame_duration
        elif self._expected_next_timestamp is not None:
            self._expected_next_timestamp += frame_duration

        # 3. Resampler configuration
        if self._resampler is None:
            self._resampler = StatefulResampler(
                orig_sr=sample_rate,
                target_sr=self.sample_rate,
                channels=channels,
                dtype=norm_dtype,
            )

        # 4. Stateful resampling to canonical 16 kHz mono float32
        resampled = self._resampler.process_tensor(mono_tensor)

        if resampled.size == 0:
            current_time = (
                self._stream_start_timestamp or 0.0
            ) + self.total_canonical_samples_written / self.sample_rate
            return IngestResult(
                session_id=self.session_id,
                samples_ingested=num_raw_samples,
                resampled_samples_produced=0,
                current_stream_time=current_time,
                window_ready=False,
                pending_window_available=self._pending_window is not None,
                windows_generated_this_ingest=0,
                generated_window_indices=[],
                generated_window_end_times=[],
                immediate_decisions=[],
                immediate_decision=None,
                windows_dropped_this_ingest=windows_dropped_this_ingest,
                total_windows_generated=self.total_windows_generated,
                total_windows_dropped=self.total_windows_dropped,
                quality_flags=frame_flags,
                continuity_reset=is_discontinuity,
                continuity_segment_id=self.continuity_segment_id,
                timestamp_delta=timestamp_delta,
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

        # 6. Sample-accurate window scheduling across frame boundaries via shared CanonicalWindowScheduler
        windows_generated_this_ingest = 0
        generated_window_indices: list[int] = []
        generated_window_end_times: list[float] = []
        immediate_decisions: list[TemporalDecision] = []

        due_windows = self.scheduler.push_samples(resampled)
        for lw in due_windows:
            windows_generated_this_ingest += 1
            generated_window_indices.append(lw.window_index)
            generated_window_end_times.append(lw.end_time)

            speech_sec = self.vad.compute_speech_duration(lw.samples, self.sample_rate)

            if speech_sec < self.min_speech_seconds:
                # Abstain from inference
                window = WindowForInference(
                    window_index=lw.window_index,
                    continuity_segment_id=lw.continuity_segment_id,
                    samples=lw.samples,
                    start_time=lw.start_time,
                    end_time=lw.end_time,
                    speech_seconds=speech_sec,
                    is_eligible=False,
                    abstain_reason=ReasonCode.INSUFFICIENT_SPEECH_DURATION,
                    quality_flags=frame_flags,
                )
                decision = self.temporal_policy.evaluate_insufficient_speech(
                    window_index=lw.window_index,
                    window_start_sec=lw.start_time,
                    window_end_sec=lw.end_time,
                    speech_seconds=speech_sec,
                )
                immediate_decisions.append(decision)
            else:
                # Window is eligible for model inference
                window = WindowForInference(
                    window_index=lw.window_index,
                    continuity_segment_id=lw.continuity_segment_id,
                    samples=lw.samples,
                    start_time=lw.start_time,
                    end_time=lw.end_time,
                    speech_seconds=speech_sec,
                    is_eligible=True,
                    abstain_reason=None,
                    quality_flags=frame_flags,
                )
                # Bounded backpressure: at most 1 pending window.
                # If a pending window exists, replace it and increment dropped count.
                if self._pending_window is not None:
                    self.total_windows_dropped += 1
                    windows_dropped_this_ingest += 1
                self._pending_window = window

        current_stream_time = (
            self._stream_start_timestamp or 0.0
        ) + self.total_canonical_samples_written / self.sample_rate

        return IngestResult(
            session_id=self.session_id,
            samples_ingested=num_raw_samples,
            resampled_samples_produced=resampled.size,
            current_stream_time=current_stream_time,
            window_ready=windows_generated_this_ingest > 0,
            pending_window_available=self._pending_window is not None,
            windows_generated_this_ingest=windows_generated_this_ingest,
            generated_window_indices=generated_window_indices,
            generated_window_end_times=generated_window_end_times,
            immediate_decisions=immediate_decisions,
            immediate_decision=immediate_decisions[0] if immediate_decisions else None,
            windows_dropped_this_ingest=windows_dropped_this_ingest,
            total_windows_generated=self.total_windows_generated,
            total_windows_dropped=self.total_windows_dropped,
            quality_flags=frame_flags,
            continuity_reset=is_discontinuity,
            continuity_segment_id=self.continuity_segment_id,
            timestamp_delta=timestamp_delta,
        )

    def pop_pending_window(self) -> WindowForInference | None:
        """Retrieve the oldest pending window eligible for inference and mark inference in-flight.

        Returns None if an inference is already in-flight or if no pending window is ready.
        """
        if self._in_flight_inference or self._pending_window is None:
            return None

        window = self._pending_window
        self._pending_window = None
        self._in_flight_window = window
        self._in_flight_window_stale = False
        self._in_flight_inference = True
        return window

    def complete_inference(
        self,
        synthetic_score: float | None = None,
        window: WindowForInference | None = None,
        calibrated_score: float | None = None,
        score_is_calibrated: bool = False,
    ) -> TemporalDecision | None:
        """Complete an in-flight inference and update the temporal decision policy.

        Suppresses stale results if the window belongs to an older continuity segment.

        Args:
            synthetic_score: Uncalibrated acoustic anomaly score [0.0, 1.0].
            window: The WindowForInference that was processed.
            calibrated_score: Optional calibrated probability (None if uncalibrated).
            score_is_calibrated: Whether score is empirically calibrated.

        Returns:
            Updated TemporalDecision from policy, or None if suppressed as stale.
        """
        self._in_flight_inference = False
        self._in_flight_window = None
        was_stale = self._in_flight_window_stale
        self._in_flight_window_stale = False

        assert window is not None

        # Stale-result suppression across continuity segments
        if was_stale or window.continuity_segment_id != self.continuity_segment_id:
            return None

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
