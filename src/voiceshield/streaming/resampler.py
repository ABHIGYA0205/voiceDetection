"""Stateful audio resampler and channel downmixer providing chunk-boundary invariance."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torchaudio.functional.functional as F

from voiceshield.dsp.exceptions import InvalidAudioValueError, UnsupportedAudioError


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
            orig_sr: Declared source sampling rate in Hz.
            target_sr: Target canonical sampling rate in Hz (default: 16,000 Hz).
            channels: Declared number of audio channels (e.g., 1 for mono, 2 for stereo).
            dtype: Declared input PCM sample format ('int16', 'int32', or 'float32').

        Raises:
            UnsupportedAudioError: If sample rate or channel count is non-positive.
        """
        if orig_sr <= 0 or target_sr <= 0:
            raise UnsupportedAudioError(
                f"Sampling rates must be positive, got orig_sr={orig_sr}, target_sr={target_sr}."
            )
        if channels <= 0:
            raise UnsupportedAudioError(f"Channel count must be positive, got channels={channels}.")

        self.orig_sr = int(orig_sr)
        self.target_sr = int(target_sr)
        self.channels = int(channels)
        self.dtype = str(dtype).lower()

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
        """Convert raw bytes or array into a 1D float32 mono torch.Tensor normalized to [-1.0, 1.0]."""
        num_channels = channels or self.channels
        sample_dtype = (dtype or self.dtype).lower()

        if isinstance(pcm_data, bytes):
            if sample_dtype == "int16":
                arr = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
            elif sample_dtype == "int32":
                arr = np.frombuffer(pcm_data, dtype=np.int32).astype(np.float32) / 2147483648.0
            elif sample_dtype in ("float32", "float"):
                arr = np.frombuffer(pcm_data, dtype=np.float32).copy()
            else:
                raise UnsupportedAudioError(f"Unsupported PCM byte format: {sample_dtype!r}.")

            if num_channels > 1:
                # De-interleave interleaved PCM bytes
                if arr.size % num_channels != 0:
                    raise UnsupportedAudioError(
                        f"PCM byte buffer length ({len(pcm_data)}) is not divisible by {num_channels} channels."
                    )
                arr = arr.reshape(-1, num_channels).mean(axis=1)

            tensor = torch.from_numpy(arr)
        elif isinstance(pcm_data, np.ndarray):
            arr = pcm_data
            if not np.all(np.isfinite(arr)):
                raise InvalidAudioValueError("Audio array contains non-finite (NaN or Inf) values.")

            if arr.dtype == np.int16:
                arr = arr.astype(np.float32) / 32768.0
            elif arr.dtype == np.int32:
                arr = arr.astype(np.float32) / 2147483648.0
            elif arr.dtype == np.float64:
                arr = arr.astype(np.float32)

            if arr.ndim == 2:
                # Downmix: shape can be (samples, channels) or (channels, samples)
                if arr.shape[1] == num_channels:
                    arr = arr.mean(axis=1)
                elif arr.shape[0] == num_channels:
                    arr = arr.mean(axis=0)
                else:
                    arr = arr.mean(axis=0)
            elif arr.ndim > 2:
                raise UnsupportedAudioError(f"Audio array has unsupported shape: {arr.shape}.")

            tensor = torch.from_numpy(arr.astype(np.float32))
        elif isinstance(pcm_data, torch.Tensor):
            t = pcm_data.detach().cpu()
            if not torch.all(torch.isfinite(t)):
                raise InvalidAudioValueError(
                    "Audio tensor contains non-finite (NaN or Inf) values."
                )

            if t.dtype == torch.int16:
                t = t.to(torch.float32) / 32768.0
            elif t.dtype == torch.int32:
                t = t.to(torch.float32) / 2147483648.0
            elif t.dtype == torch.float64:
                t = t.to(torch.float32)

            if t.ndim == 2:
                if t.shape[1] == num_channels:
                    t = t.mean(dim=1)
                elif t.shape[0] == num_channels:
                    t = t.mean(dim=0)
                else:
                    t = t.mean(dim=0)
            elif t.ndim > 2:
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
