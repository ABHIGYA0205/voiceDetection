"""Shared canonical window scheduler for offline file evaluation and real-time streaming."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from voiceshield.dsp.ring_buffer import AudioRingBuffer


@dataclass
class LogicalWindow:
    """A full 64,000-sample canonical window produced by CanonicalWindowScheduler."""

    window_index: int
    continuity_segment_id: int
    samples: np.ndarray[Any, np.dtype[np.float32]]
    start_sample: int
    end_sample: int
    start_time: float
    end_time: float


class CanonicalWindowScheduler:
    """Sample-grid window scheduler shared between offline file analysis and real-time streaming.

    Generates full 4.0-second (64,000 samples at 16 kHz) windows on a 1.0-second (16,000 samples)
    hop grid:
      - window 0: samples [0, 64,000), time [0.0, 4.0]
      - window 1: samples [16,000, 80,000), time [1.0, 5.0]
      - window k: samples [k*16,000, k*16,000 + 64,000), time [k*1.0, k*1.0 + 4.0]

    Retains bounded memory using AudioRingBuffer (O(1) memory bound).
    Emits all due logical windows crossed by each input chunk in chronological order.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        window_samples: int = 64000,
        hop_samples: int = 16000,
        capacity_samples: int = 160000,
    ) -> None:
        """Initialize CanonicalWindowScheduler.

        Args:
            sample_rate: Canonical sampling rate in Hz (default: 16,000 Hz).
            window_samples: Number of samples per full window (default: 64,000).
            hop_samples: Number of samples between consecutive window starts (default: 16,000).
            capacity_samples: Ring buffer capacity in samples (default: 160,000 = 10s @ 16 kHz).
        """
        self.sample_rate = int(sample_rate)
        self.window_samples = int(window_samples)
        self.hop_samples = int(hop_samples)
        self.capacity_samples = max(int(capacity_samples), self.window_samples * 2)

        self.ring_buffer = AudioRingBuffer(capacity_samples=self.capacity_samples)

        self.segment_samples_written: int = 0
        self.total_samples_written: int = 0
        self.next_window_end_sample: int = self.window_samples
        self.window_index: int = 0
        self.continuity_segment_id: int = 0
        self.segment_start_timestamp: float = 0.0
        self.total_windows_generated: int = 0

    @property
    def samples_since_last_hop(self) -> int:
        """Return the number of unconsumed canonical samples since the last hop boundary."""
        return self.segment_samples_written % self.hop_samples

    def reset(self, segment_id: int = 0, segment_start_timestamp: float = 0.0) -> None:
        """Reset the scheduler state completely for a fresh stream or file."""
        self.ring_buffer.reset()
        self.segment_samples_written = 0
        self.total_samples_written = 0
        self.next_window_end_sample = self.window_samples
        self.window_index = 0
        self.continuity_segment_id = int(segment_id)
        self.segment_start_timestamp = float(segment_start_timestamp)
        self.total_windows_generated = 0

    def reset_continuity(self, segment_id: int, segment_start_timestamp: float = 0.0) -> None:
        """Reset segment state on timing discontinuity while preserving cumulative window indices."""
        self.ring_buffer.reset()
        self.segment_samples_written = 0
        self.next_window_end_sample = self.window_samples
        self.continuity_segment_id = int(segment_id)
        self.segment_start_timestamp = float(segment_start_timestamp)

    def push_samples(self, samples: np.ndarray[Any, np.dtype[np.float32]]) -> list[LogicalWindow]:
        """Push canonical 16 kHz mono samples and return all newly completed logical full windows.

        Args:
            samples: 1D float32 audio samples array.

        Returns:
            List of LogicalWindow instances completed by this input chunk in chronological order.
        """
        windows: list[LogicalWindow] = []
        if samples.size == 0:
            return windows

        offset = 0
        num_samples = samples.size

        while offset < num_samples:
            remaining = num_samples - offset
            needed = self.next_window_end_sample - self.segment_samples_written
            if remaining >= needed:
                slice_to_write = samples[offset : offset + needed]
                self.ring_buffer.write(slice_to_write)
                self.segment_samples_written += needed
                self.total_samples_written += needed
                offset += needed

                # Boundary reached: extract window ending at this exact canonical sample
                window_samples = self.ring_buffer.read_tail(self.window_samples)
                window_end = self.segment_start_timestamp + (
                    self.next_window_end_sample / self.sample_rate
                )
                window_start = window_end - (self.window_samples / self.sample_rate)

                lw = LogicalWindow(
                    window_index=self.window_index,
                    continuity_segment_id=self.continuity_segment_id,
                    samples=window_samples,
                    start_sample=self.next_window_end_sample - self.window_samples,
                    end_sample=self.next_window_end_sample,
                    start_time=round(window_start, 6),
                    end_time=round(window_end, 6),
                )
                windows.append(lw)
                self.window_index += 1
                self.total_windows_generated += 1
                self.next_window_end_sample += self.hop_samples
            else:
                slice_to_write = samples[offset:]
                self.ring_buffer.write(slice_to_write)
                self.segment_samples_written += slice_to_write.size
                self.total_samples_written += slice_to_write.size
                offset = num_samples
                break

        return windows
