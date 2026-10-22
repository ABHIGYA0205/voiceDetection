"""Bounded circular audio ring buffer for streaming inference."""

from __future__ import annotations

from typing import Any

import numpy as np


class AudioRingBuffer:
    """Fixed-capacity circular audio ring buffer providing bounded memory O(1) operations.

    Stores mono float32 audio samples and permits extraction of chronological tail windows
    without reallocating memory.
    """

    def __init__(self, capacity_samples: int = 160000) -> None:
        """Initialize AudioRingBuffer.

        Args:
            capacity_samples: Maximum capacity in samples (default: 160,000 samples = 10s @ 16 kHz).
        """
        if capacity_samples <= 0:
            raise ValueError(f"capacity_samples must be positive, got {capacity_samples}.")

        self.capacity: int = int(capacity_samples)
        self._buffer: np.ndarray[Any, np.dtype[np.float32]] = np.zeros(
            self.capacity, dtype=np.float32
        )
        self._write_pos: int = 0
        self.total_samples_written: int = 0

    @property
    def available_samples(self) -> int:
        """Return the number of stored samples available to read (at most capacity)."""
        return min(self.total_samples_written, self.capacity)

    @property
    def is_full(self) -> bool:
        """Return whether the circular buffer has reached its full capacity."""
        return self.total_samples_written >= self.capacity

    def write(self, samples: np.ndarray[Any, np.dtype[np.float32]]) -> None:
        """Append 1D float32 audio samples into the circular buffer.

        If input exceeds capacity, only the most recent `capacity` samples are retained.
        """
        if samples.ndim != 1:
            samples = samples.ravel()
        if samples.size == 0:
            return

        n = samples.size
        if n >= self.capacity:
            # When writing more than the entire buffer, keep only the latest capacity samples
            self._buffer[:] = samples[-self.capacity :]
            self._write_pos = 0
            self.total_samples_written += n
            return

        # Write in one or two circular segments
        end_pos = self._write_pos + n
        if end_pos <= self.capacity:
            self._buffer[self._write_pos : end_pos] = samples
        else:
            first_len = self.capacity - self._write_pos
            self._buffer[self._write_pos :] = samples[:first_len]
            self._buffer[: end_pos - self.capacity] = samples[first_len:]

        self._write_pos = end_pos % self.capacity
        self.total_samples_written += n

    def read_tail(self, num_samples: int) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Read the most recent `num_samples` in contiguous chronological order.

        Args:
            num_samples: Number of samples to read.

        Returns:
            Contiguous 1D float32 array containing the requested samples.

        Raises:
            ValueError: If num_samples exceeds available samples or capacity.
        """
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}.")
        if num_samples > self.available_samples:
            raise ValueError(
                f"Requested {num_samples} samples, but only {self.available_samples} are available."
            )

        # The tail starts `num_samples` before `_write_pos`
        start_pos = (self._write_pos - num_samples) % self.capacity
        end_pos = self._write_pos

        if start_pos < end_pos:
            return self._buffer[start_pos:end_pos].copy()
        else:
            return np.concatenate((self._buffer[start_pos:], self._buffer[:end_pos]))

    def reset(self) -> None:
        """Zero out the buffer and reset all write pointers and counters."""
        self._buffer.fill(0.0)
        self._write_pos = 0
        self.total_samples_written = 0
