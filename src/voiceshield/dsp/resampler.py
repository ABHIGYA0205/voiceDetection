"""Deterministic, high-quality audio resampling via bandlimited sinc interpolation.

Provides a single canonical stateful resampler for both offline file evaluation and real-time streaming,
maintaining bounded FIR history, chunk-boundary invariance, and exact output sample length.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from voiceshield.dsp.exceptions import InvalidAudioValueError, UnsupportedAudioError

MIN_SUPPORTED_SAMPLE_RATE: int = 8000
MAX_SUPPORTED_SAMPLE_RATE: int = 192000
CANONICAL_LOWPASS_FILTER_WIDTH: int = 6
CANONICAL_ROLLOFF: float = 0.99
CANONICAL_RESAMPLING_METHOD: str = "sinc_interp_hann"


def _get_sinc_kernel(orig_sr: int, target_sr: int, gcd: int) -> tuple[torch.Tensor, int]:
    """Encapsulate dependency on torchaudio sinc resampling kernel generation.

    Fails with a clear, descriptive UnsupportedAudioError if the underlying torchaudio
    API is unavailable or incompatible.
    """
    try:
        import torchaudio.functional.functional as F_func

        if not hasattr(F_func, "_get_sinc_resample_kernel"):
            raise AttributeError(
                "_get_sinc_resample_kernel is missing from torchaudio.functional.functional"
            )
        kernel, width = F_func._get_sinc_resample_kernel(
            orig_freq=orig_sr,
            new_freq=target_sr,
            gcd=gcd,
            lowpass_filter_width=CANONICAL_LOWPASS_FILTER_WIDTH,
            rolloff=CANONICAL_ROLLOFF,
            resampling_method=CANONICAL_RESAMPLING_METHOD,
            beta=None,
        )
        return kernel, width
    except Exception as exc:
        raise UnsupportedAudioError(
            f"Failed to initialize bandlimited sinc resampling kernel for {orig_sr} -> {target_sr} Hz: {exc}"
        ) from exc


class CanonicalResampler:
    """Canonical stateful resampler providing exact output length and chunk-boundary invariance.

    Consumes 1D mono float32 audio and incrementally resamples to target_sr (default 16,000 Hz).
    Maintains bounded FIR filter history (O(1) memory with respect to stream length).
    The lifecycle requires:
        process(chunk) -> canonical samples
        flush() -> final tail samples completing the exact canonical duration
        reset() -> resets state for clean reuse
    """

    def __init__(self, orig_sr: int, target_sr: int = 16000) -> None:
        """Initialize CanonicalResampler.

        Args:
            orig_sr: Input sampling rate in Hz (8,000 to 192,000 Hz).
            target_sr: Target sampling rate in Hz (default: 16,000 Hz).

        Raises:
            UnsupportedAudioError: If sampling rates are outside the supported bounds.
        """
        if orig_sr < MIN_SUPPORTED_SAMPLE_RATE or orig_sr > MAX_SUPPORTED_SAMPLE_RATE:
            raise UnsupportedAudioError(
                f"Unsupported source sampling rate: {orig_sr} Hz. "
                f"Permitted range is {MIN_SUPPORTED_SAMPLE_RATE:,} Hz to {MAX_SUPPORTED_SAMPLE_RATE:,} Hz."
            )
        if target_sr <= 0:
            raise UnsupportedAudioError(
                f"Target sampling rate must be positive, got target_sr={target_sr}."
            )

        self.orig_sr = int(orig_sr)
        self.target_sr = int(target_sr)
        self.gcd = math.gcd(self.orig_sr, self.target_sr)
        self.in_ratio = self.orig_sr // self.gcd
        self.out_ratio = self.target_sr // self.gcd

        self.total_source_frames: int = 0
        self.total_canonical_samples: int = 0
        self._is_flushed: bool = False
        self.kernel: torch.Tensor | None = None

        if self.orig_sr != self.target_sr:
            self.kernel, self.width = _get_sinc_kernel(self.orig_sr, self.target_sr, self.gcd)
            self.k_len: int = self.kernel.shape[-1]
            self.buffer: torch.Tensor = torch.zeros(self.width, dtype=torch.float32)
        else:
            self.kernel = None
            self.width = 0
            self.k_len = 0
            self.buffer = torch.empty(0, dtype=torch.float32)

    @property
    def is_flushed(self) -> bool:
        """Return whether the resampler has been finalized via flush()."""
        return self._is_flushed

    def reset(self) -> None:
        """Reset internal filter history and sample counters for clean stream reuse."""
        self.total_source_frames = 0
        self.total_canonical_samples = 0
        self._is_flushed = False
        if self.orig_sr != self.target_sr:
            self.buffer = torch.zeros(self.width, dtype=torch.float32)
        else:
            self.buffer = torch.empty(0, dtype=torch.float32)

    def process(
        self, audio: np.ndarray[Any, np.dtype[np.float32]] | torch.Tensor
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Incrementally resample an audio chunk to target_sr (16 kHz mono float32).

        Args:
            audio: 1D float32 mono NumPy array or Torch tensor.

        Returns:
            1D float32 NumPy array of resampled samples.

        Raises:
            RuntimeError: If called after flush() before reset().
            UnsupportedAudioError: If input dimensionality is not 1D.
            InvalidAudioValueError: If input contains NaN or Infinite values.
        """
        if self._is_flushed:
            raise RuntimeError("Cannot process audio after flush() without reset().")

        if isinstance(audio, np.ndarray):
            if audio.ndim != 1:
                raise UnsupportedAudioError(
                    f"CanonicalResampler requires 1D mono audio, got shape {audio.shape}."
                )
            if not np.all(np.isfinite(audio)):
                raise InvalidAudioValueError("Audio array contains NaN or Infinite values.")
            num_frames = audio.shape[0]
            if num_frames == 0:
                return np.empty(0, dtype=np.float32)
            x = torch.from_numpy(
                np.ascontiguousarray(audio, dtype=np.float32)
                if audio.dtype != np.float32
                else audio
            )
        elif isinstance(audio, torch.Tensor):
            if audio.ndim != 1:
                raise UnsupportedAudioError(
                    f"CanonicalResampler requires 1D mono audio, got shape {tuple(audio.shape)}."
                )
            if not bool(torch.all(torch.isfinite(audio))):
                raise InvalidAudioValueError("Audio tensor contains NaN or Infinite values.")
            num_frames = audio.shape[0]
            if num_frames == 0:
                return np.empty(0, dtype=np.float32)
            x = audio.to(torch.float32) if audio.dtype != torch.float32 else audio
        else:
            raise UnsupportedAudioError(
                f"Unsupported audio type: {type(audio).__name__}. Expected np.ndarray or torch.Tensor."
            )

        self.total_source_frames += num_frames

        # Identity fast path for 16 kHz: preserve exact samples with zero phase/filtering delay
        if self.orig_sr == self.target_sr:
            out_arr = x.detach().cpu().numpy().astype(np.float32)
            self.total_canonical_samples += num_frames
            return np.ascontiguousarray(out_arr, dtype=np.float32)

        assert self.kernel is not None
        full = torch.cat([self.buffer, x])
        if full.shape[0] < self.k_len:
            self.buffer = full
            return np.empty(0, dtype=np.float32)

        num_strides = (full.shape[0] - self.k_len) // self.in_ratio + 1
        usable_in_len = (num_strides - 1) * self.in_ratio + self.k_len
        to_process = full[:usable_in_len]

        out = torch.nn.functional.conv1d(
            to_process[None, None, :], self.kernel, stride=self.in_ratio
        )
        out_1d = out.transpose(1, 2).reshape(-1)

        consumed = num_strides * self.in_ratio
        self.buffer = full[consumed:]

        res_np = out_1d.detach().cpu().numpy().astype(np.float32)
        self.total_canonical_samples += res_np.shape[0]
        return np.ascontiguousarray(res_np, dtype=np.float32)

    def flush(self) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Flush the resampler at end-of-stream, returning the exact tail to meet expected length.

        Output length across all process() calls and flush() strictly equals:
            ceil(total_source_frames * target_sr / orig_sr)

        Flush is idempotent: a second flush returns an empty array.
        After flush(), process() calls are rejected until reset().
        """
        if self._is_flushed:
            return np.empty(0, dtype=np.float32)

        self._is_flushed = True
        expected_total = math.ceil(self.total_source_frames * self.target_sr / self.orig_sr)
        remaining_needed = expected_total - self.total_canonical_samples

        if self.orig_sr == self.target_sr or remaining_needed <= 0:
            self.buffer = torch.empty(0, dtype=torch.float32)
            return np.empty(0, dtype=np.float32)

        assert self.kernel is not None
        # Right pad with zeros to complete remaining convolution strides
        right_pad = torch.zeros(self.width + self.in_ratio, dtype=torch.float32)
        full_flush = torch.cat([self.buffer, right_pad])

        if full_flush.shape[0] >= self.k_len and remaining_needed > 0:
            num_strides = (full_flush.shape[0] - self.k_len) // self.in_ratio + 1
            usable_in_len = (num_strides - 1) * self.in_ratio + self.k_len
            to_process = full_flush[:usable_in_len]
            out = torch.nn.functional.conv1d(
                to_process[None, None, :], self.kernel, stride=self.in_ratio
            )
            flush_out = out.transpose(1, 2).reshape(-1)[:remaining_needed]
            tail = flush_out.detach().cpu().numpy().astype(np.float32)
        else:
            tail = np.empty(0, dtype=np.float32)

        self.buffer = torch.empty(0, dtype=torch.float32)
        self.total_canonical_samples += tail.shape[0]
        return np.ascontiguousarray(tail, dtype=np.float32)


def resample_audio(
    audio: np.ndarray[Any, np.dtype[np.float32]],
    orig_sr: int,
    target_sr: int = 16000,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Resample 1D mono audio to target_sr using the canonical stateful sinc resampler.

    Provided for backward compatibility. Internally uses CanonicalResampler with process() and flush().

    Args:
        audio: 1D float32 mono audio array.
        orig_sr: Source sampling rate in Hz.
        target_sr: Target sampling rate in Hz (default: 16,000 Hz).

    Returns:
        1D float32 audio array resampled to target_sr.

    Raises:
        UnsupportedAudioError: If sampling rate or audio dimension is invalid.
        InvalidAudioValueError: If input audio contains non-finite values.
    """
    resampler = CanonicalResampler(orig_sr=orig_sr, target_sr=target_sr)
    chunk = resampler.process(audio)
    tail = resampler.flush()

    if tail.size > 0:
        return np.ascontiguousarray(np.concatenate([chunk, tail]), dtype=np.float32)
    return np.ascontiguousarray(chunk, dtype=np.float32)
