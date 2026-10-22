"""Unit and integration tests for the AASIST-L detector adapter."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

from voiceshield.models import (
    AASIST_L_DEFAULT_CONFIG,
    AASISTLDetector,
    CheckpointNotFoundError,
    DetectorOutput,
    pad_for_aasist,
)
from voiceshield.vendor.aasist import AASISTModel


def test_pad_for_aasist_behavior() -> None:
    """Verify cyclic padding for short inputs and truncation for long inputs."""
    target_len = 64600

    # 1. Short audio (16,000 samples)
    short = np.arange(16000, dtype=np.float32)
    padded = pad_for_aasist(short, max_len=target_len)
    assert isinstance(padded, torch.Tensor)
    assert padded.shape == (target_len,)
    assert padded.dtype == torch.float32
    # Verify cyclic tiling
    np.testing.assert_array_equal(padded[:16000].numpy(), short)
    np.testing.assert_array_equal(padded[16000:32000].numpy(), short)

    # 2. Exact length (64,600 samples)
    exact = np.full(target_len, 0.42, dtype=np.float32)
    padded_exact = pad_for_aasist(exact, max_len=target_len)
    assert padded_exact.shape == (target_len,)
    np.testing.assert_array_equal(padded_exact.numpy(), exact)

    # 3. Long audio (70,000 samples) -> head cropped to 64,600
    long_audio = np.arange(70000, dtype=np.float32)
    cropped = pad_for_aasist(long_audio, max_len=target_len)
    assert cropped.shape == (target_len,)
    np.testing.assert_array_equal(cropped.numpy(), long_audio[:target_len])

    # 4. Empty audio raises ValueError
    with pytest.raises(ValueError, match="Cannot pad an empty audio waveform"):
        pad_for_aasist(np.array([], dtype=np.float32), max_len=target_len)


def test_aasist_shape_and_forward_pass_mock_weights() -> None:
    """Verify input/output shapes and BaseDetector control flow using an instantiated model."""
    raw_model = AASISTModel(AASIST_L_DEFAULT_CONFIG)
    detector = AASISTLDetector(_instantiated_model=raw_model)

    # Test single-window prediction with short input (16,000 samples @ 16 kHz)
    audio = 0.5 * np.sin(2 * np.pi * 440.0 * np.linspace(0, 1.0, 16000, dtype=np.float32))
    output = detector.predict(audio)

    assert isinstance(output, DetectorOutput)
    assert len(output.raw_logits) == 2
    assert 0.0 <= output.synthetic_likelihood <= 1.0
    assert output.input_samples == 64600
    assert output.model_id == "clovaai/aasist-l"
    assert output.metadata["label_mapping"] == {"0": "spoof", "1": "bonafide"}


def test_label_direction_locked() -> None:
    """Strictly test label direction: index 0 is spoof, index 1 is bonafide."""
    dummy_model = MagicMock(spec=nn.Module)
    dummy_model.eval.return_value = dummy_model
    dummy_model.to.return_value = dummy_model

    detector = AASISTLDetector(_instantiated_model=dummy_model)

    # Case A: High spoof logit (index 0 = 5.0, index 1 = -2.0)
    # Output must indicate high synthetic likelihood (> 0.99)
    dummy_model.return_value = (None, torch.tensor([[5.0, -2.0]]))
    out_spoof = detector.predict(np.zeros(64600, dtype=np.float32))

    assert out_spoof.raw_logits == (5.0, -2.0)
    assert out_spoof.bonafide_score == -2.0
    assert out_spoof.synthetic_likelihood > 0.99
    assert out_spoof.is_synthetic_suspected is True

    # Case B: High bonafide logit (index 0 = -3.0, index 1 = 4.0)
    # Output must indicate low synthetic likelihood (< 0.01)
    dummy_model.return_value = (None, torch.tensor([[-3.0, 4.0]]))
    out_bonafide = detector.predict(np.zeros(64600, dtype=np.float32))

    assert out_bonafide.raw_logits == (-3.0, 4.0)
    assert out_bonafide.bonafide_score == 4.0
    assert out_bonafide.synthetic_likelihood < 0.01
    assert out_bonafide.is_synthetic_suspected is False


def test_missing_checkpoint_raises_with_instructions(tmp_path: Path) -> None:
    """Verify that absent checkpoint raises CheckpointNotFoundError with setup commands."""
    non_existent = tmp_path / "models" / "weights" / "AASIST-L.pth"

    with pytest.raises(CheckpointNotFoundError) as exc_info:
        AASISTLDetector(checkpoint_path=non_existent)

    err_msg = str(exc_info.value)
    assert "AASIST-L checkpoint not found at:" in err_msg
    assert "curl -L -o" in err_msg
    assert "SpeechAntiSpoofingBenchmarks/AASIST-L" in err_msg
    assert "VoiceShield forbids fabricating or inventing model weights." in err_msg


def test_batch_prediction_multi_samples() -> None:
    """Verify batch inference handles variable length items by padding to 64,600."""
    raw_model = AASISTModel(AASIST_L_DEFAULT_CONFIG)
    detector = AASISTLDetector(_instantiated_model=raw_model)

    # 3 audio waveforms of exact 64600 length in batch tensor
    batch_tensor = torch.randn(3, 64600, dtype=torch.float32)
    outputs = detector.predict_batch(batch_tensor)

    assert len(outputs) == 3
    for out in outputs:
        assert isinstance(out, DetectorOutput)
        assert len(out.raw_logits) == 2
        assert 0.0 <= out.synthetic_likelihood <= 1.0


@pytest.mark.skipif(
    not Path("models/weights/AASIST-L.pth").is_file(),
    reason="Official AASIST-L.pth checkpoint not downloaded; skipping integration test.",
)
def test_real_checkpoint_integration() -> None:
    """Integration test executed only when the real official checkpoint is present on disk."""
    detector = AASISTLDetector(checkpoint_path="models/weights/AASIST-L.pth")
    test_audio = np.zeros(64600, dtype=np.float32)
    output = detector.predict(test_audio)
    assert isinstance(output, DetectorOutput)
    assert len(output.raw_logits) == 2
