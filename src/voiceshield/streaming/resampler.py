"""Stateful audio resampler and channel downmixer providing chunk-boundary invariance."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torchaudio.functional.functional as F

from voiceshield.dsp.exceptions import (
    CorruptAudioError,
    EmptyAudioError,
    InvalidAudioValueError,
    UnsupportedAudioError,
)

MIN_SUPPORTED_SAMPLE_RATE = 8000
MAX_SUPPORTED_SAMPLE_RATE = 192000
SUPPORTED_PCM_DTYPES = ("int16", "int32", "float32", "float")


def compute_raw_frame_count(
    pcm_data: bytes | np.ndarray[Any, Any] | torch.Tensor,
    channels: int,
    dtype: str = "int16",
) -> int:
    """Compute number of audio frames (time samples) consistently across formats and layouts."""
    if isinstance(pcm_data, bytes):
        bytes_per_sample = 2 if dtype.lower() == "int16" else 4
        frame_width = bytes_per_sample * channels
        if frame_width == 0:
            raise UnsupportedAudioError(f"Invalid frame width with channels={channels}.")
        return len(pcm_data) // frame_width
    elif isinstance(pcm_data, np.ndarray):
        if pcm_data.ndim == 1:
            return int(pcm_data.shape[0]) if channels == 1 else int(pcm_data.shape[0] // channels)
        elif pcm_data.ndim == 2:
            if pcm_data.shape[1] == channels and pcm_data.shape[0] != channels:
                return int(pcm_data.shape[0])
            elif pcm_data.shape[0] == channels and pcm_data.shape[1] != channels:
                return int(pcm_data.shape[1])
            elif pcm_data.shape[0] == channels and pcm_data.shape[1] == channels:
                return int(pcm_data.shape[0])
            else:
                raise UnsupportedAudioError(
                    f"Audio array shape {pcm_data.shape} is incompatible with channels={channels}."
                )
        else:
            raise UnsupportedAudioError(f"Audio array has unsupported ndim={pcm_data.ndim}.")
    elif isinstance(pcm_data, torch.Tensor):
        if pcm_data.ndim == 1:
            return pcm_data.shape[0] if channels == 1 else pcm_data.shape[0] // channels
        elif pcm_data.ndim == 2:
            if pcm_data.shape[1] == channels and pcm_data.shape[0] != channels:
                return int(pcm_data.shape[0])
            elif pcm_data.shape[0] == channels and pcm_data.shape[1] != channels:
                return int(pcm_data.shape[1])
            elif pcm_data.shape[0] == channels and pcm_data.shape[1] == channels:
                return int(pcm_data.shape[0])
            else:
                raise UnsupportedAudioError(
                    f"Audio tensor shape {pcm_data.shape} is incompatible with channels={channels}."
                )
        else:
            raise UnsupportedAudioError(f"Audio tensor has unsupported ndim={pcm_data.ndim}.")
    else:
        raise UnsupportedAudioError(f"Unsupported audio data type: {type(pcm_data)!r}.")


class StatefulResampler:
    """Stateful resampler that ingests arbitrary PCM audio frames and resamples to 16 kHz mono.

    Maintains filter history across chunk boundaries, guaranteeing chunk-boundary invariance:
    streaming audio in small arbitrary chunks produces the identical output sequence as
    processing the entire audio in a single call (within float32 precision limits).
    """

    def __init__(
        self,
        orig_sr: int,
        target_sr: int = 16000,
        channels: int = 1,
        dtype: str = "float32",
    ) -> None:
        """Initialize StatefulResampler.

        Args:
            orig_sr: Declared source sampling rate in Hz. Permitted: 8,000 to 192,000 Hz.
            target_sr: Target canonical sampling rate in Hz (default: 16,000 Hz).
            channels: Declared number of audio channels (1 for mono, 2 for stereo).
            dtype: Declared input PCM sample format ('int16', 'int32', or 'float32').

        Raises:
            UnsupportedAudioError: If sample rate, channel count, or dtype is not supported.
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
        if channels not in (1, 2):
            raise UnsupportedAudioError(
                f"Unsupported channel count: {channels}. Permitted channels are 1 or 2."
            )

        dtype_norm = str(dtype).lower()
        if dtype_norm not in SUPPORTED_PCM_DTYPES:
            raise UnsupportedAudioError(f"Unsupported PCM encoding: {dtype!r}.")

        self.orig_sr = int(orig_sr)
        self.target_sr = int(target_sr)
        self.channels = int(channels)
        self.dtype = dtype_norm

        # Rational resampling factors
        self.gcd = math.gcd(self.orig_sr, self.target_sr)
        self.in_ratio = self.orig_sr // self.gcd
        self.out_ratio = self.target_sr // self.gcd

        if self.orig_sr != self.target_sr:
            # Precompute bandlimited sinc resampling kernel
            self.kernel, self.width = F._get_sinc_resample_kernel(
                self.orig_sr, self.target_sr, self.gcd
            )
            # Buffer holds unconsumed input history needed for FIR convolution continuity
            self.buffer = torch.zeros(self.width, dtype=torch.float32)
        else:
            self.kernel = None
            self.width = 0
            self.buffer = torch.empty(0, dtype=torch.float32)

    def reset(self) -> None:
        """Reset internal filter history and buffer state."""
        if self.orig_sr != self.target_sr:
            self.buffer = torch.zeros(self.width, dtype=torch.float32)
        else:
            self.buffer = torch.empty(0, dtype=torch.float32)

    def _convert_pcm_to_mono_float32(
        self,
        pcm_data: bytes | np.ndarray[Any, Any] | torch.Tensor,
        channels: int | None = None,
        dtype: str | None = None,
    ) -> torch.Tensor:
        """Convert raw bytes, NumPy array, or Torch tensor into a 1D float32 mono torch.Tensor.

        Validates byte alignment against complete frame width (bytes_per_sample * channels),
        decodes little-endian PCM, verifies sample finiteness and range [-1.0, 1.0], and
        downmixes channels.
        """
        num_channels = channels or self.channels
        if num_channels not in (1, 2):
            raise UnsupportedAudioError(
                f"Unsupported channel count: {num_channels}. Permitted channels are 1 or 2."
            )
        sample_dtype = (dtype or self.dtype).lower()
        if sample_dtype not in SUPPORTED_PCM_DTYPES:
            raise UnsupportedAudioError(f"Unsupported PCM encoding: {sample_dtype!r}.")

        if isinstance(pcm_data, bytes):
            if len(pcm_data) == 0:
                raise EmptyAudioError("Incoming audio frame is empty (0 bytes).")

            bytes_per_sample = 2 if sample_dtype == "int16" else 4
            frame_width = bytes_per_sample * num_channels

            if sample_dtype == "int16" and len(pcm_data) % 2 != 0:
                raise CorruptAudioError(f"Odd-length int16 buffer rejected: {len(pcm_data)} bytes.")
            if sample_dtype in ("int32", "float32", "float") and len(pcm_data) % 4 != 0:
                raise CorruptAudioError(
                    f"{sample_dtype} buffer length ({len(pcm_data)} bytes) is not divisible by 4."
                )
            if len(pcm_data) % frame_width != 0:
                raise CorruptAudioError(
                    f"PCM byte buffer length ({len(pcm_data)} bytes) is not divisible by "
                    f"complete frame width ({frame_width} bytes for {num_channels} ch)."
                )

            try:
                if sample_dtype == "int16":
                    arr = np.frombuffer(pcm_data, dtype="<i2").astype(np.float32) / 32768.0
                elif sample_dtype == "int32":
                    arr = np.frombuffer(pcm_data, dtype="<i4").astype(np.float32) / 2147483648.0
                elif sample_dtype in ("float32", "float"):
                    arr = np.frombuffer(pcm_data, dtype="<f4").astype(np.float32).copy()
                else:
                    raise UnsupportedAudioError(f"Unsupported PCM byte format: {sample_dtype!r}.")
            except (UnsupportedAudioError, CorruptAudioError):
                raise
            except Exception as exc:
                raise CorruptAudioError(f"Failed to decode PCM byte buffer: {exc}") from exc

            if not np.all(np.isfinite(arr)):
                raise InvalidAudioValueError(
                    "Decoded PCM audio contains non-finite (NaN or Inf) values."
                )
            if np.any(arr < -1.0) or np.any(arr > 1.0):
                raise InvalidAudioValueError(
                    "Decoded float audio samples out of valid range [-1.0, 1.0]."
                )

            if num_channels > 1:
                arr = arr.reshape(-1, num_channels).mean(axis=1)

            tensor = torch.from_numpy(arr)

        elif isinstance(pcm_data, np.ndarray):
            if pcm_data.size == 0:
                raise EmptyAudioError("Audio array contains zero samples.")

            if not np.all(np.isfinite(pcm_data)):
                raise InvalidAudioValueError("Audio array contains non-finite (NaN or Inf) values.")

            if pcm_data.dtype == np.int16:
                arr = pcm_data.astype(np.float32) / 32768.0
            elif pcm_data.dtype == np.int32:
                arr = pcm_data.astype(np.float32) / 2147483648.0
            elif pcm_data.dtype in (np.float32, np.float64):
                arr = pcm_data.astype(np.float32)
            else:
                raise UnsupportedAudioError(f"Audio array has unsupported dtype: {pcm_data.dtype}.")

            if np.any(arr < -1.0) or np.any(arr > 1.0):
                raise InvalidAudioValueError("Audio samples out of valid range [-1.0, 1.0].")

            if arr.ndim == 1:
                if num_channels > 1:
                    if arr.size % num_channels != 0:
                        raise CorruptAudioError(
                            f"1D array size ({arr.size}) is not divisible by {num_channels} channels."
                        )
                    arr = arr.reshape(-1, num_channels).mean(axis=1)
            elif arr.ndim == 2:
                if arr.shape[1] == num_channels and arr.shape[0] != num_channels:
                    arr = arr.mean(axis=1)
                elif arr.shape[0] == num_channels and arr.shape[1] != num_channels:
                    arr = arr.mean(axis=0)
                elif arr.shape[0] == num_channels and arr.shape[1] == num_channels:
                    arr = arr.mean(axis=1)
                else:
                    raise UnsupportedAudioError(
                        f"Audio array shape {arr.shape} is incompatible with channels={num_channels}."
                    )
            else:
                raise UnsupportedAudioError(f"Audio array has unsupported shape: {arr.shape}.")

            tensor = torch.from_numpy(arr.astype(np.float32))

        elif isinstance(pcm_data, torch.Tensor):
            if pcm_data.numel() == 0:
                raise EmptyAudioError("Audio tensor contains zero samples.")

            t = pcm_data.detach().cpu()
            if not torch.all(torch.isfinite(t)):
                raise InvalidAudioValueError(
                    "Audio tensor contains non-finite (NaN or Inf) values."
                )

            if t.dtype == torch.int16:
                t = t.to(torch.float32) / 32768.0
            elif t.dtype == torch.int32:
                t = t.to(torch.float32) / 2147483648.0
            elif t.dtype in (torch.float32, torch.float64):
                t = t.to(torch.float32)
            else:
                raise UnsupportedAudioError(f"Audio tensor has unsupported dtype: {t.dtype}.")

            if bool(torch.any(t < -1.0) or torch.any(t > 1.0)):
                raise InvalidAudioValueError("Audio tensor samples out of valid range [-1.0, 1.0].")

            if t.ndim == 1:
                if num_channels > 1:
                    if t.numel() % num_channels != 0:
                        raise CorruptAudioError(
                            f"1D tensor size ({t.numel()}) is not divisible by {num_channels} channels."
                        )
                    t = t.reshape(-1, num_channels).mean(dim=1)
            elif t.ndim == 2:
                if t.shape[1] == num_channels and t.shape[0] != num_channels:
                    t = t.mean(dim=1)
                elif t.shape[0] == num_channels and t.shape[1] != num_channels:
                    t = t.mean(dim=0)
                elif t.shape[0] == num_channels and t.shape[1] == num_channels:
                    t = t.mean(dim=1)
                else:
                    raise UnsupportedAudioError(
                        f"Audio tensor shape {t.shape} is incompatible with channels={num_channels}."
                    )
            else:
                raise UnsupportedAudioError(f"Audio tensor has unsupported shape: {t.shape}.")

            tensor = t
        else:
            raise UnsupportedAudioError(f"Unsupported pcm_data type: {type(pcm_data)!r}.")

        return tensor

    def process(
        self,
        pcm_data: bytes | np.ndarray[Any, Any] | torch.Tensor,
        channels: int | None = None,
        dtype: str | None = None,
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Process incoming PCM audio chunk and return resampled 16 kHz mono float32 array.

        Args:
            pcm_data: Incoming raw bytes, NumPy array, or Torch tensor.
            channels: Optional channels override.
            dtype: Optional sample format override.

        Returns:
            1D float32 NumPy array at 16,000 Hz.
        """
        x = self._convert_pcm_to_mono_float32(pcm_data, channels=channels, dtype=dtype)
        if x.numel() == 0:
            return np.empty(0, dtype=np.float32)

        if self.orig_sr == self.target_sr:
            return x.numpy()

        assert self.kernel is not None
        full = torch.cat([self.buffer, x])
        k_len = self.kernel.shape[-1]
        if full.shape[0] < k_len:
            self.buffer = full
            return np.empty(0, dtype=np.float32)

        num_strides = (full.shape[0] - k_len) // self.in_ratio + 1
        usable_in_len = (num_strides - 1) * self.in_ratio + k_len
        to_process = full[:usable_in_len]

        out = torch.nn.functional.conv1d(
            to_process[None, None, :], self.kernel, stride=self.in_ratio
        )
        out_1d = out.transpose(1, 2).reshape(-1)

        consumed = num_strides * self.in_ratio
        self.buffer = full[consumed:]

        return out_1d.numpy()
