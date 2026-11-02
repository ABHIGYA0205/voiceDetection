"""Focused tests for Stage A2 input-reliability integration in file inference and temporal policy.

Verifies:
1. Clean reliable audio preserves existing file behavior and allows origin verdict.
2. A clipped active-speech window is LIMITED, detector.predict is never called, and all score/logit fields are null.
3. A marginal-SNR active-speech window is LIMITED, detector.predict is never called, and all score/logit fields are null.
4. Silent/insufficient audio remains INSUFFICIENT, detector.predict is never called, and all score/logit fields are null.
5. A limited-only file returns limited_reliability + uncertain + no synthetic suspicion.
6. A mixed reliable-LOW plus LIMITED file returns limited_reliability/uncertain, never LOW.
7. A mixed reliable-HIGH plus LIMITED file returns limited_reliability/uncertain, never HIGH.
8. Reliable windows plus ordinary speech-insufficient pause windows preserve existing aggregation and do not become limited_reliability.
9. Reliability counters sum exactly to total_windows across all scenarios.
10. eligible_windows equals reliable_window_count in all cases.
11. The temporal abstention helper rejects RELIABLE assessments.
12. The temporal abstention helper accepts no model score argument.
13. The temporal abstention helper does not mutate score histories.
14. API JSON response includes per-window input_reliability and all file-level counters.
15. Existing error and short-file contracts remain backward compatible.
"""

from __future__ import annotations

import inspect
import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from services.api.main import app
from voiceshield.dsp.quality import AudioQualityMetrics
from voiceshield.dsp.reliability import (
    InputReliabilityAssessment,
    ReliabilityReason,
    ReliabilityState,
    assess_input_reliability,
)
from voiceshield.models.base import BaseDetector, DetectorOutput
from voiceshield.pipeline.file_evaluator import (
    FileInferenceResult,
    WindowResult,
    _validate_window_safety_contracts,
    evaluate_audio,
)
from voiceshield.streaming.policy import (
    DecisionState,
    ReasonCode,
    TemporalDecisionPolicy,
)


class CountingSpyDetector(BaseDetector):
    """Spy detector tracking predict call count, arguments, and programmed scores."""

    is_mock = True

    def __init__(self, scores: list[float] | None = None, default_score: float = 0.1) -> None:
        self.scores = scores or []
        self.default_score = default_score
        self.predict_calls = 0
        self.input_windows: list[np.ndarray[Any, Any]] = []
        self.model_id = "counting-spy-detector"
        self.device = "cpu"

    def predict(self, audio: Any) -> DetectorOutput:
        self.predict_calls += 1
        arr = np.asarray(audio, dtype=np.float32)
        self.input_windows.append(arr)
        idx = self.predict_calls - 1
        score = self.scores[idx] if idx < len(self.scores) else self.default_score
        return DetectorOutput.from_scores(
            raw_logits=(float(score), float(1.0 - score)),
            bonafide_score=float(1.0 - score),
            synthetic_score=float(score),
            is_synthetic_suspected=bool(score >= 0.65),
            input_samples=len(arr),
            device_used="cpu",
            model_id=self.model_id,
        )

    def predict_batch(self, batch: Any) -> list[DetectorOutput]:
        return [self.predict(chunk) for chunk in batch]


CountingDetector = CountingSpyDetector


def make_clean_wav(path: Path, duration_s: float = 4.0, sr: int = 16000) -> Path:
    """Create a clean, unclipped, high-SNR sine wave file."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32)
    samples = 0.5 * np.sin(2 * np.pi * 300.0 * t)
    sf.write(str(path), samples, sr, format="WAV", subtype="PCM_16")
    return path


def make_clipped_wav(path: Path, duration_s: float = 4.0, sr: int = 16000) -> Path:
    """Create a heavily clipped, active speech audio file."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32)
    samples = np.clip(5.0 * np.sin(2 * np.pi * 300.0 * t), -1.0, 1.0).astype(np.float32)
    sf.write(str(path), samples, sr, format="WAV", subtype="PCM_16")
    return path


def make_marginal_snr_wav(path: Path, duration_s: float = 4.0, sr: int = 16000) -> Path:
    """Create an active speech audio file with marginal SNR (0 <= SNR < 12 dB)."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32)
    samples = (0.05 + 0.03 * np.sin(2 * np.pi * 300.0 * t)).astype(np.float32)
    sf.write(str(path), samples, sr, format="WAV", subtype="PCM_16")
    return path


def make_silence_wav(path: Path, duration_s: float = 4.0, sr: int = 16000) -> Path:
    """Create a completely silent audio file."""
    samples = np.zeros(int(sr * duration_s), dtype=np.float32)
    sf.write(str(path), samples, sr, format="WAV", subtype="PCM_16")
    return path


def make_short_speech_wav(
    path: Path,
    speech_duration_s: float = 0.2,
    total_duration_s: float = 4.0,
    sr: int = 16000,
) -> Path:
    """Create an audio file where active speech duration is below threshold."""
    samples = np.zeros(int(sr * total_duration_s), dtype=np.float32)
    n_pulse = int(speech_duration_s * sr)
    t = np.linspace(0, speech_duration_s, n_pulse, endpoint=False, dtype=np.float32)
    samples[:n_pulse] = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    sf.write(str(path), samples, sr, format="WAV", subtype="PCM_16")
    return path


# ---------------------------------------------------------------------------
# Requirement 1: Clean reliable audio preserves existing behavior
# ---------------------------------------------------------------------------


def test_clean_reliable_audio_preserves_behavior(tmp_path: Path) -> None:
    wav_path = make_clean_wav(tmp_path / "clean.wav", duration_s=4.0)
    detector = CountingDetector(scores=[0.10])

    result = evaluate_audio(wav_path, detector=detector, window_seconds=4.0, hop_seconds=4.0)

    assert result.status == "success"
    assert result.quality_origin_verdict_allowed is True
    assert result.total_windows == 1
    assert result.reliable_window_count == 1
    assert result.limited_reliability_window_count == 0
    assert result.insufficient_reliability_window_count == 0
    assert result.eligible_windows == 1
    assert detector.predict_calls == 1

    win = result.windows[0]
    assert win.is_eligible is True
    assert win.input_reliability.state == ReliabilityState.RELIABLE
    assert win.input_reliability.origin_verdict_allowed is True
    assert win.input_reliability.reasons == ()
    assert win.synthetic_score == 0.10
    assert win.raw_spoof_logit is not None
    assert win.raw_bonafide_logit is not None
    assert win.raw_bonafide_score is not None
    assert win.abstain_reason is None


# ---------------------------------------------------------------------------
# Requirement 2: Clipped window is LIMITED and never calls detector
# ---------------------------------------------------------------------------


def test_clipped_window_is_limited_and_detector_not_called(tmp_path: Path) -> None:
    wav_path = make_clipped_wav(tmp_path / "clipped.wav", duration_s=4.0)
    detector = CountingDetector(scores=[0.99])

    result = evaluate_audio(wav_path, detector=detector, window_seconds=4.0, hop_seconds=4.0)

    assert detector.predict_calls == 0
    assert result.status == "limited_reliability"
    assert result.final_decision_state == DecisionState.UNCERTAIN.value
    assert result.is_synthetic_suspected is False
    assert result.quality_origin_verdict_allowed is False
    assert result.reliable_window_count == 0
    assert result.limited_reliability_window_count == 1
    assert result.insufficient_reliability_window_count == 0
    assert result.total_windows == 1
    assert result.eligible_windows == 0

    win = result.windows[0]
    assert win.is_eligible is False
    assert win.input_reliability.state == ReliabilityState.LIMITED
    assert win.input_reliability.origin_verdict_allowed is False
    assert ReliabilityReason.SEVERE_CLIPPING in win.input_reliability.reasons
    assert win.synthetic_score is None
    assert win.uncalibrated_synthetic_score is None
    assert win.synthetic_likelihood is None
    assert win.calibrated_synthetic_likelihood is None
    assert win.raw_spoof_logit is None
    assert win.raw_bonafide_logit is None
    assert win.raw_bonafide_score is None
    assert win.score_is_calibrated is False
    assert win.is_synthetic_suspected is False
    assert win.abstain_reason == ReasonCode.SIGNAL_QUALITY_DEGRADED.value
    assert win.decision_state == DecisionState.UNCERTAIN.value
    assert win.decision_reason == ReasonCode.SIGNAL_QUALITY_DEGRADED.value


# ---------------------------------------------------------------------------
# Requirement 3: Marginal SNR window is LIMITED and does not call detector
# ---------------------------------------------------------------------------


def test_marginal_snr_window_is_limited_and_detector_not_called(tmp_path: Path) -> None:
    wav_path = make_marginal_snr_wav(tmp_path / "marginal.wav", duration_s=4.0)
    detector = CountingDetector(scores=[0.99])

    result = evaluate_audio(wav_path, detector=detector, window_seconds=4.0, hop_seconds=4.0)

    assert detector.predict_calls == 0
    assert result.status == "limited_reliability"
    assert result.final_decision_state == DecisionState.UNCERTAIN.value
    assert result.is_synthetic_suspected is False
    assert result.quality_origin_verdict_allowed is False
    assert result.reliable_window_count == 0
    assert result.limited_reliability_window_count == 1
    assert result.eligible_windows == 0

    win = result.windows[0]
    assert win.is_eligible is False
    assert win.input_reliability.state == ReliabilityState.LIMITED
    assert win.input_reliability.origin_verdict_allowed is False
    assert ReliabilityReason.MARGINAL_SNR in win.input_reliability.reasons
    assert win.synthetic_score is None
    assert win.raw_spoof_logit is None
    assert win.raw_bonafide_logit is None
    assert win.raw_bonafide_score is None
    assert win.abstain_reason == ReasonCode.SIGNAL_QUALITY_DEGRADED.value


# ---------------------------------------------------------------------------
# Requirement 4: Silent / insufficient audio remains INSUFFICIENT
# ---------------------------------------------------------------------------


def test_silence_remains_insufficient_and_detector_not_called(tmp_path: Path) -> None:
    wav_path = make_silence_wav(tmp_path / "silence.wav", duration_s=4.0)
    detector = CountingDetector(scores=[0.99])

    result = evaluate_audio(wav_path, detector=detector, window_seconds=4.0, hop_seconds=4.0)

    # 1. Detector predict_calls is 0
    assert detector.predict_calls == 0

    # 2. Silence remains file-level insufficient_evidence
    assert result.status == "insufficient_evidence"
    assert result.abstain_reason == ReasonCode.INSUFFICIENT_SPEECH_DURATION.value
    assert result.final_decision_state == DecisionState.INSUFFICIENT_EVIDENCE.value
    assert result.is_synthetic_suspected is False
    assert result.quality_origin_verdict_allowed is False
    assert result.reliable_window_count == 0
    assert result.limited_reliability_window_count == 0
    assert result.insufficient_reliability_window_count == 1
    assert result.total_windows == 1
    assert result.eligible_windows == 0

    # 3. Silence plus insufficient speech uses SIGNAL_QUALITY_DEGRADED at window level
    win = result.windows[0]
    assert win.is_eligible is False
    assert win.input_reliability.state == ReliabilityState.INSUFFICIENT
    assert win.input_reliability.origin_verdict_allowed is False
    assert win.input_reliability.reasons == (
        ReliabilityReason.SILENT_AUDIO,
        ReliabilityReason.INSUFFICIENT_SPEECH,
    )
    assert win.decision is not None
    assert win.decision.reason == ReasonCode.SIGNAL_QUALITY_DEGRADED
    assert win.decision_reason == ReasonCode.SIGNAL_QUALITY_DEGRADED.value
    assert "silent_audio" in str(win.decision_reason_details)
    assert "insufficient_speech" in str(win.decision_reason_details)
    assert win.decision_state == DecisionState.INSUFFICIENT_EVIDENCE.value
    assert win.abstain_reason == ReasonCode.INSUFFICIENT_SPEECH_DURATION.value
    assert win.synthetic_score is None
    assert win.raw_spoof_logit is None
    assert win.raw_bonafide_logit is None
    assert win.raw_bonafide_score is None
    assert win.history_scores == []


def test_short_speech_remains_insufficient_and_detector_not_called(tmp_path: Path) -> None:
    wav_path = make_short_speech_wav(
        tmp_path / "short_speech.wav", speech_duration_s=0.5, total_duration_s=4.0
    )
    detector = CountingDetector(scores=[0.99])

    result = evaluate_audio(
        wav_path,
        detector=detector,
        window_seconds=4.0,
        hop_seconds=4.0,
        min_speech_seconds=0.8,
    )

    # 1. Detector predict_calls is 0
    assert detector.predict_calls == 0

    # 2. Pure insufficient speech uses INSUFFICIENT_SPEECH_DURATION
    assert result.status == "insufficient_evidence"
    assert result.abstain_reason == ReasonCode.INSUFFICIENT_SPEECH_DURATION.value
    assert result.final_decision_state == DecisionState.INSUFFICIENT_EVIDENCE.value
    assert result.reliable_window_count == 0
    assert result.insufficient_reliability_window_count == 1
    assert result.quality_origin_verdict_allowed is False

    win = result.windows[0]
    assert win.is_eligible is False
    assert win.input_reliability.state == ReliabilityState.INSUFFICIENT
    assert win.input_reliability.reasons == (ReliabilityReason.INSUFFICIENT_SPEECH,)
    assert win.decision is not None
    assert win.decision.reason == ReasonCode.INSUFFICIENT_SPEECH_DURATION
    assert win.decision_reason == ReasonCode.INSUFFICIENT_SPEECH_DURATION.value
    assert win.abstain_reason == ReasonCode.INSUFFICIENT_SPEECH_DURATION.value
    assert win.decision_state == DecisionState.INSUFFICIENT_EVIDENCE.value
    assert win.synthetic_score is None
    assert win.history_scores == []


def test_reason_codes_policy_history_and_delegation_invariants() -> None:
    """Prove pure speech vs silence reason codes, history non-mutation, and detector abstention."""
    policy = TemporalDecisionPolicy()
    policy.evaluate(synthetic_score=0.40, speech_seconds=3.0)
    policy.evaluate(synthetic_score=0.50, speech_seconds=3.0)
    raw_before = list(policy.raw_history)
    smoothed_before = list(policy.smoothed_history)
    assert len(raw_before) == 2

    # 1. Pure insufficient speech delegates to evaluate_insufficient_speech
    speech_insufficient = InputReliabilityAssessment(
        state=ReliabilityState.INSUFFICIENT,
        reasons=(ReliabilityReason.INSUFFICIENT_SPEECH,),
        explanation="pure short speech",
        origin_verdict_allowed=False,
    )
    dec_speech = policy.evaluate_reliability_abstention(
        speech_insufficient,
        window_index=2,
        window_start_sec=8.0,
        window_end_sec=12.0,
        speech_seconds=0.2,
    )
    assert dec_speech.reason == ReasonCode.INSUFFICIENT_SPEECH_DURATION
    assert dec_speech.state == DecisionState.INSUFFICIENT_EVIDENCE
    assert policy.raw_history == raw_before
    assert policy.smoothed_history == smoothed_before

    # 2. Silence + insufficient speech uses SIGNAL_QUALITY_DEGRADED at window level
    silence_assessment = InputReliabilityAssessment(
        state=ReliabilityState.INSUFFICIENT,
        reasons=(ReliabilityReason.SILENT_AUDIO, ReliabilityReason.INSUFFICIENT_SPEECH),
        explanation="silent audio and short speech",
        origin_verdict_allowed=False,
    )
    dec_silence = policy.evaluate_reliability_abstention(
        silence_assessment,
        window_index=3,
        window_start_sec=12.0,
        window_end_sec=16.0,
        speech_seconds=0.0,
    )
    assert dec_silence.reason == ReasonCode.SIGNAL_QUALITY_DEGRADED
    assert dec_silence.state == DecisionState.INSUFFICIENT_EVIDENCE
    assert "silent_audio" in dec_silence.reason_details
    assert "insufficient_speech" in dec_silence.reason_details
    assert policy.raw_history == raw_before
    assert policy.smoothed_history == smoothed_before


# ---------------------------------------------------------------------------
# Requirement 5: Limited-only file returns limited_reliability + uncertain + null aggregates
# ---------------------------------------------------------------------------


def test_limited_only_file_returns_limited_reliability_uncertain(tmp_path: Path) -> None:
    wav_path = make_clipped_wav(tmp_path / "clipped_only.wav", duration_s=4.0)
    detector = CountingDetector(scores=[0.99])

    result = evaluate_audio(wav_path, detector=detector, window_seconds=4.0, hop_seconds=4.0)

    assert result.status == "limited_reliability"
    assert result.final_decision_state == "uncertain"
    assert result.is_synthetic_suspected is False
    assert result.quality_origin_verdict_allowed is False
    assert result.mean_synthetic_score is None
    assert result.max_synthetic_score is None
    assert result.mean_synthetic_likelihood is None
    assert result.max_synthetic_likelihood is None
    assert result.raw_high_window_count == 0
    assert result.uncertain_window_count == 0
    assert result.high_window_count == 0
    assert detector.predict_calls == 0
    assert "degraded input quality" in result.temporal_rule_explanation.lower()
    assert "reliability signal" in result.temporal_rule_explanation.lower()


# ---------------------------------------------------------------------------
# Requirement 6: Mixed reliable-LOW plus LIMITED file returns limited_reliability/uncertain, never LOW
# ---------------------------------------------------------------------------


def test_mixed_reliable_low_plus_limited_returns_uncertain(tmp_path: Path) -> None:
    sr = 16000
    t = np.linspace(0, 4.0, int(sr * 4.0), endpoint=False, dtype=np.float32)
    s_clean = (0.5 * np.sin(2 * np.pi * 300.0 * t)).astype(np.float32)
    s_clipped = np.clip(5.0 * np.sin(2 * np.pi * 300.0 * t), -1.0, 1.0).astype(np.float32)
    samples = np.concatenate([s_clean, s_clipped])

    wav_path = tmp_path / "mixed_low_limited.wav"
    sf.write(str(wav_path), samples, sr, format="WAV", subtype="PCM_16")

    detector = CountingDetector(scores=[0.10, 0.99])
    result = evaluate_audio(wav_path, detector=detector, window_seconds=4.0, hop_seconds=4.0)

    # Exactly 1 predict call for the clean window; clipped window was skipped
    assert detector.predict_calls == 1
    assert result.total_windows == 2
    assert result.reliable_window_count == 1
    assert result.limited_reliability_window_count == 1
    assert result.eligible_windows == 1

    # Invariant: Never LOW when any LIMITED window is present
    assert result.status == "limited_reliability"
    assert result.final_decision_state == "uncertain"
    assert result.final_decision_state != "low"
    assert result.is_synthetic_suspected is False
    assert result.quality_origin_verdict_allowed is False

    # Reliable window scores preserved in detailed evidence
    assert result.mean_synthetic_score == 0.10
    assert result.max_synthetic_score == 0.10
    assert result.windows[0].synthetic_score == 0.10
    assert result.windows[1].synthetic_score is None


# ---------------------------------------------------------------------------
# Requirement 7: Mixed reliable-HIGH plus LIMITED file returns limited_reliability/uncertain, never HIGH
# ---------------------------------------------------------------------------


def test_mixed_reliable_high_plus_limited_returns_uncertain(tmp_path: Path) -> None:
    sr = 16000
    t = np.linspace(0, 4.0, int(sr * 4.0), endpoint=False, dtype=np.float32)
    s_clean = (0.5 * np.sin(2 * np.pi * 300.0 * t)).astype(np.float32)
    s_clipped = np.clip(5.0 * np.sin(2 * np.pi * 300.0 * t), -1.0, 1.0).astype(np.float32)
    # 3 consecutive high-score clean windows satisfying 3-of-5 elevation rule, followed by a clipped window
    samples = np.concatenate([s_clean, s_clean, s_clean, s_clipped])

    wav_path = tmp_path / "mixed_high_limited.wav"
    sf.write(str(wav_path), samples, sr, format="WAV", subtype="PCM_16")

    detector = CountingDetector(scores=[0.90, 0.90, 0.90, 0.99])
    result = evaluate_audio(
        wav_path, detector=detector, window_seconds=4.0, hop_seconds=4.0, threshold=0.65
    )

    # 1. LIMITED window is not passed to detector (only the 3 clean windows are passed)
    assert detector.predict_calls == 3
    assert result.total_windows == 4
    assert result.reliable_window_count == 3
    assert result.limited_reliability_window_count == 1
    assert result.eligible_windows == 3
    assert result.windows[3].input_reliability.state == ReliabilityState.LIMITED
    assert result.windows[3].synthetic_score is None
    assert result.windows[3].is_eligible is False

    # 2. At least one window has decision_state == HIGH
    assert result.windows[0].decision_state == DecisionState.UNCERTAIN.value
    assert result.windows[1].decision_state == DecisionState.UNCERTAIN.value
    assert result.windows[2].decision_state == DecisionState.HIGH.value
    assert any(w.decision_state == DecisionState.HIGH.value for w in result.windows)

    # 3. high_window_count > 0 and raw_high_window_count reflects high scores
    assert result.high_window_count > 0
    assert result.high_window_count == 1
    assert result.raw_high_window_count == 3

    # 4. Final file status is limited_reliability
    assert result.status == "limited_reliability"

    # 5. Final decision state is uncertain (never HIGH)
    assert result.final_decision_state == DecisionState.UNCERTAIN.value
    assert result.final_decision_state != DecisionState.HIGH.value

    # 6. is_synthetic_suspected is False
    assert result.is_synthetic_suspected is False

    # 7. quality_origin_verdict_allowed is False
    assert result.quality_origin_verdict_allowed is False

    # Scores from reliable windows are preserved in detailed metrics
    assert result.mean_synthetic_score == 0.90
    assert result.max_synthetic_score == 0.90


# ---------------------------------------------------------------------------
# Requirement 8: Reliable windows + ordinary pause windows preserve aggregation
# ---------------------------------------------------------------------------


def test_reliable_windows_plus_pause_preserves_aggregation(tmp_path: Path) -> None:
    sr = 16000
    t = np.linspace(0, 4.0, int(sr * 4.0), endpoint=False, dtype=np.float32)
    s_clean = (0.5 * np.sin(2 * np.pi * 300.0 * t)).astype(np.float32)
    s_silence = np.zeros(int(sr * 4.0), dtype=np.float32)
    # [clean, silence, clean]
    samples = np.concatenate([s_clean, s_silence, s_clean])

    wav_path = tmp_path / "pause_speech.wav"
    sf.write(str(wav_path), samples, sr, format="WAV", subtype="PCM_16")

    detector = CountingDetector(scores=[0.10, 0.15])
    result = evaluate_audio(wav_path, detector=detector, window_seconds=4.0, hop_seconds=4.0)

    # Only 2 calls for the 2 clean windows; silence window abstained
    assert detector.predict_calls == 2
    assert result.total_windows == 3
    assert result.reliable_window_count == 2
    assert result.insufficient_reliability_window_count == 1
    assert result.limited_reliability_window_count == 0
    assert result.eligible_windows == 2

    # Status must be success, origin verdict allowed, and aggregation preserved
    assert result.status == "success"
    assert result.quality_origin_verdict_allowed is True
    assert result.final_decision_state == "low"
    assert result.is_synthetic_suspected is False


# ---------------------------------------------------------------------------
# Requirement 9 & 10: Counter sums and eligible_windows equality
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("clean_count", "clipped_count", "silent_count"),
    [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (2, 0, 1),
        (1, 1, 1),
    ],
)
def test_reliability_counters_and_eligible_equality(
    tmp_path: Path,
    clean_count: int,
    clipped_count: int,
    silent_count: int,
) -> None:
    sr = 16000
    t = np.linspace(0, 4.0, int(sr * 4.0), endpoint=False, dtype=np.float32)
    blocks: list[np.ndarray[Any, Any]] = []
    for _ in range(clean_count):
        blocks.append((0.5 * np.sin(2 * np.pi * 300.0 * t)).astype(np.float32))
    for _ in range(clipped_count):
        blocks.append(np.clip(5.0 * np.sin(2 * np.pi * 300.0 * t), -1.0, 1.0).astype(np.float32))
    for _ in range(silent_count):
        blocks.append(np.zeros(int(sr * 4.0), dtype=np.float32))

    samples = np.concatenate(blocks)
    wav_path = tmp_path / f"combo_{clean_count}_{clipped_count}_{silent_count}.wav"
    sf.write(str(wav_path), samples, sr, format="WAV", subtype="PCM_16")

    detector = CountingDetector(scores=[0.2] * clean_count)
    result = evaluate_audio(wav_path, detector=detector, window_seconds=4.0, hop_seconds=4.0)

    # Invariant 1: Sum equals total_windows
    assert (
        result.reliable_window_count
        + result.limited_reliability_window_count
        + result.insufficient_reliability_window_count
        == result.total_windows
    )
    # Invariant 2: eligible_windows equals reliable_window_count
    assert result.eligible_windows == result.reliable_window_count
    assert detector.predict_calls == result.reliable_window_count


# ---------------------------------------------------------------------------
# Requirement 11, 12, 13: Temporal abstention helper invariants
# ---------------------------------------------------------------------------


def test_helper_rejects_reliable_assessment() -> None:
    policy = TemporalDecisionPolicy()
    reliable = InputReliabilityAssessment(
        state=ReliabilityState.RELIABLE,
        reasons=(),
        explanation="clean signal",
        origin_verdict_allowed=True,
    )
    with pytest.raises(ValueError, match="Cannot evaluate reliability abstention for a RELIABLE"):
        policy.evaluate_reliability_abstention(reliable)


def test_helper_signature_accepts_no_model_score() -> None:
    sig = inspect.signature(TemporalDecisionPolicy.evaluate_reliability_abstention)
    param_names = set(sig.parameters.keys())
    # Assert parameters do not include any model score variable
    for forbidden in ["score", "synthetic_score", "calibrated_score", "raw_score", "model_score"]:
        assert forbidden not in param_names, f"Forbidden score parameter found: {forbidden}"


def test_helper_does_not_mutate_score_histories() -> None:
    policy = TemporalDecisionPolicy()
    # Populate some initial score history
    policy.evaluate(synthetic_score=0.45, speech_seconds=3.0)
    policy.evaluate(synthetic_score=0.55, speech_seconds=3.0)

    raw_before = list(policy.raw_history)
    smoothed_before = list(policy.smoothed_history)
    assert len(raw_before) == 2

    limited = InputReliabilityAssessment(
        state=ReliabilityState.LIMITED,
        reasons=(ReliabilityReason.SEVERE_CLIPPING,),
        explanation="digital clipping",
        origin_verdict_allowed=False,
    )
    dec = policy.evaluate_reliability_abstention(
        assessment=limited,
        window_index=2,
        window_start_sec=8.0,
        window_end_sec=12.0,
        speech_seconds=4.0,
    )

    # Invariants: histories are NOT mutated
    assert policy.raw_history == raw_before
    assert policy.smoothed_history == smoothed_before
    assert dec.state == DecisionState.UNCERTAIN
    assert dec.synthetic_score is None
    assert dec.calibrated_score is None
    assert dec.score_is_calibrated is False
    assert dec.reason == ReasonCode.SIGNAL_QUALITY_DEGRADED
    assert "limited" in dec.reason_details
    assert "severe_clipping" in dec.reason_details


def test_helper_insufficient_speech_backward_compatibility() -> None:
    policy = TemporalDecisionPolicy()
    pure_insufficient = InputReliabilityAssessment(
        state=ReliabilityState.INSUFFICIENT,
        reasons=(ReliabilityReason.INSUFFICIENT_SPEECH,),
        explanation="speech too short",
        origin_verdict_allowed=False,
    )
    dec = policy.evaluate_reliability_abstention(
        assessment=pure_insufficient,
        window_index=0,
        window_start_sec=0.0,
        window_end_sec=4.0,
        speech_seconds=0.2,
    )
    assert dec.state == DecisionState.INSUFFICIENT_EVIDENCE
    assert dec.reason == ReasonCode.INSUFFICIENT_SPEECH_DURATION
    assert dec.synthetic_score is None


# ---------------------------------------------------------------------------
# Requirement 14: FastAPI JSON includes per-window input_reliability and counters
# ---------------------------------------------------------------------------


def test_fastapi_json_includes_reliability_fields(tmp_path: Path) -> None:
    wav_path = make_clean_wav(tmp_path / "api_test.wav", duration_s=4.0)
    with wav_path.open("rb") as f:
        file_bytes = f.read()

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/analyze/file",
            files={"file": ("api_test.wav", io.BytesIO(file_bytes), "audio/wav")},
        )

        assert response.status_code == 200
        data = response.json()

        # File-level counters
        assert "reliable_window_count" in data
        assert "limited_reliability_window_count" in data
        assert "insufficient_reliability_window_count" in data
        assert "quality_origin_verdict_allowed" in data
        assert data["reliable_window_count"] == 1
        assert data["limited_reliability_window_count"] == 0
        assert data["insufficient_reliability_window_count"] == 0
        assert data["quality_origin_verdict_allowed"] is True

        # Per-window assessment
        windows = data.get("windows", [])
        assert len(windows) == 1
        win = windows[0]
        assert "input_reliability" in win
        rel = win["input_reliability"]
        assert rel["state"] == "reliable"
        assert rel["reasons"] == []
        assert rel["origin_verdict_allowed"] is True
        assert isinstance(rel["explanation"], str)


# ---------------------------------------------------------------------------
# Requirement 15: Existing error and short file contracts remain backward compatible
# ---------------------------------------------------------------------------


def test_short_file_under_window_samples_backward_compatible(tmp_path: Path) -> None:
    # 2.0s audio contains fewer samples than required 4.0s window
    wav_path = make_clean_wav(tmp_path / "short.wav", duration_s=2.0)
    detector = CountingDetector()

    result = evaluate_audio(wav_path, detector=detector, window_seconds=4.0)

    assert result.status == "insufficient_evidence"
    assert result.abstain_reason == ReasonCode.INSUFFICIENT_CONTIGUOUS_AUDIO.value
    assert result.total_windows == 0
    assert result.eligible_windows == 0
    assert result.reliable_window_count == 0
    assert result.limited_reliability_window_count == 0
    assert result.insufficient_reliability_window_count == 0
    assert result.quality_origin_verdict_allowed is False
    assert detector.predict_calls == 0


def test_complex_multi_window_file_invariants(tmp_path: Path) -> None:
    """Complex scenario: 5 windows [clean, silence, clipped, marginal, clean]."""
    sr = 16000
    t = np.linspace(0, 4.0, int(sr * 4.0), endpoint=False, dtype=np.float32)
    s_clean1 = (0.5 * np.sin(2 * np.pi * 300.0 * t)).astype(np.float32)
    s_silence = np.zeros(int(sr * 4.0), dtype=np.float32)
    s_clipped = np.clip(5.0 * np.sin(2 * np.pi * 300.0 * t), -1.0, 1.0).astype(np.float32)
    s_marginal = (0.05 + 0.03 * np.sin(2 * np.pi * 300.0 * t)).astype(np.float32)
    s_clean2 = (0.5 * np.sin(2 * np.pi * 500.0 * t)).astype(np.float32)

    samples = np.concatenate([s_clean1, s_silence, s_clipped, s_marginal, s_clean2])
    wav_path = tmp_path / "complex_5win.wav"
    sf.write(str(wav_path), samples, sr, format="WAV", subtype="PCM_16")

    detector = CountingDetector(scores=[0.10, 0.20])
    result = evaluate_audio(wav_path, detector=detector, window_seconds=4.0, hop_seconds=4.0)

    # Invariants
    assert result.total_windows == 5
    assert result.reliable_window_count == 2
    assert result.insufficient_reliability_window_count == 1
    assert result.limited_reliability_window_count == 2
    assert (
        result.reliable_window_count
        + result.limited_reliability_window_count
        + result.insufficient_reliability_window_count
        == result.total_windows
    )
    assert result.eligible_windows == result.reliable_window_count
    # Detector invoked ONLY for the 2 clean windows
    assert detector.predict_calls == 2

    # File-level result must abstain due to presence of limited windows
    assert result.status == "limited_reliability"
    assert result.final_decision_state == "uncertain"
    assert result.is_synthetic_suspected is False
    assert result.quality_origin_verdict_allowed is False
    assert result.abstain_reason == ReasonCode.SIGNAL_QUALITY_DEGRADED.value

    # Score fields populated from the 2 eligible windows
    assert result.mean_synthetic_score == 0.15
    assert result.max_synthetic_score == 0.20


def test_helper_type_validation() -> None:
    policy = TemporalDecisionPolicy()
    with pytest.raises(TypeError, match="assessment must be an InputReliabilityAssessment"):
        policy.evaluate_reliability_abstention("invalid_assessment")  # type: ignore[arg-type]


def test_helper_marginal_snr_emits_uncertain() -> None:
    policy = TemporalDecisionPolicy()
    limited = InputReliabilityAssessment(
        state=ReliabilityState.LIMITED,
        reasons=(ReliabilityReason.MARGINAL_SNR,),
        explanation="marginal snr",
        origin_verdict_allowed=False,
    )
    dec = policy.evaluate_reliability_abstention(limited)
    assert dec.state == DecisionState.UNCERTAIN
    assert dec.reason == ReasonCode.SIGNAL_QUALITY_DEGRADED
    assert dec.synthetic_score is None


def test_helper_empty_audio_emits_insufficient_evidence() -> None:
    policy = TemporalDecisionPolicy()
    insufficient = InputReliabilityAssessment(
        state=ReliabilityState.INSUFFICIENT,
        reasons=(ReliabilityReason.EMPTY_AUDIO,),
        explanation="empty buffer",
        origin_verdict_allowed=False,
    )
    dec = policy.evaluate_reliability_abstention(insufficient)
    assert dec.state == DecisionState.INSUFFICIENT_EVIDENCE
    assert dec.reason == ReasonCode.SIGNAL_QUALITY_DEGRADED
    assert dec.synthetic_score is None


def test_helper_window_count_increments() -> None:
    policy = TemporalDecisionPolicy()
    assert policy.window_count == 0
    insufficient = InputReliabilityAssessment(
        state=ReliabilityState.INSUFFICIENT,
        reasons=(ReliabilityReason.INSUFFICIENT_SPEECH,),
        explanation="insufficient speech",
        origin_verdict_allowed=False,
    )
    policy.evaluate_reliability_abstention(insufficient)
    assert policy.window_count == 1

    limited = InputReliabilityAssessment(
        state=ReliabilityState.LIMITED,
        reasons=(ReliabilityReason.LOW_ENERGY,),
        explanation="low energy",
        origin_verdict_allowed=False,
    )
    policy.evaluate_reliability_abstention(limited)
    assert policy.window_count == 2


def test_models_json_serialization_roundtrip() -> None:
    quality = AudioQualityMetrics(
        peak_level=0.5,
        rms=0.1,
        clipping_fraction=0.0,
        is_clipped=False,
        is_silent=False,
        snr_db_estimate=25.0,
        quality_status="clean",
    )
    rel = assess_input_reliability(
        quality, active_speech_seconds=3.0, required_min_speech_seconds=2.0
    )
    win = WindowResult(
        window_index=0,
        start_seconds=0.0,
        end_seconds=4.0,
        raw_spoof_logit=0.1,
        raw_bonafide_logit=0.9,
        raw_bonafide_score=0.9,
        synthetic_score=0.1,
        active_speech_seconds=3.0,
        is_synthetic_suspected=False,
        quality_status=quality,
        input_reliability=rel,
    )
    dumped = win.model_dump_json()
    assert "input_reliability" in dumped
    assert '"state":"reliable"' in dumped

    file_res = FileInferenceResult(
        file_path="test.wav",
        reliable_window_count=1,
        limited_reliability_window_count=0,
        insufficient_reliability_window_count=0,
        quality_origin_verdict_allowed=True,
        windows=[win],
    )
    file_dumped = file_res.model_dump_json()
    assert '"reliable_window_count":1' in file_dumped
    assert '"quality_origin_verdict_allowed":true' in file_dumped


# ---------------------------------------------------------------------------
# Requirement 2: Fail-closed runtime safety contract validation tests
# ---------------------------------------------------------------------------


def test_validate_window_safety_contracts_counter_mismatch() -> None:
    with pytest.raises(RuntimeError, match="Reliability window counters invariant violated"):
        _validate_window_safety_contracts(
            windows=[],
            num_windows=2,
            reliable_window_count=1,
            limited_reliability_window_count=0,
            insufficient_reliability_window_count=0,
            num_eligible=1,
        )


def test_validate_window_safety_contracts_eligible_mismatch() -> None:
    with pytest.raises(RuntimeError, match="Eligible windows count invariant violated"):
        _validate_window_safety_contracts(
            windows=[],
            num_windows=1,
            reliable_window_count=1,
            limited_reliability_window_count=0,
            insufficient_reliability_window_count=0,
            num_eligible=0,
        )


def test_validate_window_safety_contracts_eligible_violations() -> None:
    quality = AudioQualityMetrics(
        peak_level=0.5,
        rms=0.1,
        clipping_fraction=0.0,
        is_clipped=False,
        is_silent=False,
        snr_db_estimate=25.0,
        quality_status="clean",
    )
    limited = InputReliabilityAssessment(
        state=ReliabilityState.LIMITED,
        reasons=(ReliabilityReason.SEVERE_CLIPPING,),
        explanation="clipped",
        origin_verdict_allowed=False,
    )
    # Eligible window with non-RELIABLE state
    bad_rel_win = WindowResult(
        window_index=0,
        start_seconds=0.0,
        end_seconds=4.0,
        synthetic_score=0.8,
        active_speech_seconds=3.0,
        is_eligible=True,
        is_synthetic_suspected=True,
        abstain_reason=None,
        quality_status=quality,
        input_reliability=limited,
        raw_spoof_logit=0.8,
        raw_bonafide_logit=0.2,
        raw_bonafide_score=0.2,
    )
    with pytest.raises(RuntimeError, match="expected 'reliable'"):
        _validate_window_safety_contracts(
            windows=[bad_rel_win],
            num_windows=1,
            reliable_window_count=1,
            limited_reliability_window_count=0,
            insufficient_reliability_window_count=0,
            num_eligible=1,
        )


def test_validate_window_safety_contracts_non_reliable_violations() -> None:
    quality = AudioQualityMetrics(
        peak_level=0.5,
        rms=0.1,
        clipping_fraction=0.0,
        is_clipped=False,
        is_silent=False,
        snr_db_estimate=25.0,
        quality_status="clean",
    )
    limited = InputReliabilityAssessment(
        state=ReliabilityState.LIMITED,
        reasons=(ReliabilityReason.SEVERE_CLIPPING,),
        explanation="clipped",
        origin_verdict_allowed=False,
    )

    # Window with non-None score when abstained
    bad_win = WindowResult(
        window_index=0,
        start_seconds=0.0,
        end_seconds=4.0,
        synthetic_score=0.8,
        active_speech_seconds=3.0,
        is_eligible=False,
        is_synthetic_suspected=False,
        abstain_reason=ReasonCode.SIGNAL_QUALITY_DEGRADED.value,
        quality_status=quality,
        input_reliability=limited,
    )
    with pytest.raises(RuntimeError, match="has synthetic_score="):
        _validate_window_safety_contracts(
            windows=[bad_win],
            num_windows=1,
            reliable_window_count=0,
            limited_reliability_window_count=1,
            insufficient_reliability_window_count=0,
            num_eligible=0,
        )

    # Window with invalid abstain_reason
    bad_abstain_win = WindowResult(
        window_index=0,
        start_seconds=0.0,
        end_seconds=4.0,
        synthetic_score=None,
        active_speech_seconds=3.0,
        is_eligible=False,
        is_synthetic_suspected=False,
        abstain_reason="invalid_reason",
        quality_status=quality,
        input_reliability=limited,
    )
    with pytest.raises(RuntimeError, match="has invalid abstain_reason"):
        _validate_window_safety_contracts(
            windows=[bad_abstain_win],
            num_windows=1,
            reliable_window_count=0,
            limited_reliability_window_count=1,
            insufficient_reliability_window_count=0,
            num_eligible=0,
        )
