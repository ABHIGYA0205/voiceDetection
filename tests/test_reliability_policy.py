"""Focused tests for the standalone input-reliability assessment policy.

Verifies:
- ReliabilityState enum members and string values.
- ReliabilityReason enum members and string values.
- Deep immutability and invariant enforcement of InputReliabilityAssessment.
- Model serialization to valid JSON.
- Rejection of duplicate or out-of-order reasons and contradictory state combinations.
- Rejection of empty or whitespace-only explanations.
- Deterministic classification for reliable, limited, and insufficient states using REASON_ORDER.
- Exact boundary condition where active_speech_seconds == required_min_speech_seconds.
- Fail-closed validation for unknown quality_status values.
- Fail-closed validation for invalid physical metric ranges (peak < 0, rms < 0, clipping not in [0, 1], non-finite).
- Acceptance and classification of negative finite SNR.
- Correct fail-closed handling and error message for invalid metrics type.
- Fail-closed validation for NaN, Inf, negative, and invalid duration parameters.
- Absence of detector/model score parameters in the assessment function.
"""

from __future__ import annotations

import inspect
import json
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from voiceshield.dsp import (
    AudioQualityMetrics,
    InputReliabilityAssessment,
    ReliabilityReason,
    ReliabilityState,
    assess_input_reliability,
    compute_quality_metrics,
)


def _make_clean_metrics() -> AudioQualityMetrics:
    """Helper creating nominal clean audio quality metrics."""
    return AudioQualityMetrics(
        peak_level=0.5,
        rms=0.1,
        clipping_fraction=0.0,
        is_clipped=False,
        is_silent=False,
        snr_db_estimate=25.0,
        quality_status="clean",
    )


# 1. Enum specifications
def test_reliability_state_enum() -> None:
    """Verify ReliabilityState contains exactly reliable, limited, insufficient."""
    assert len(ReliabilityState) == 3
    assert ReliabilityState.RELIABLE.value == "reliable"
    assert ReliabilityState.LIMITED.value == "limited"
    assert ReliabilityState.INSUFFICIENT.value == "insufficient"
    assert {s.value for s in ReliabilityState} == {"reliable", "limited", "insufficient"}


def test_reliability_reason_enum() -> None:
    """Verify ReliabilityReason covers all required reason codes."""
    required = {
        "empty_audio",
        "silent_audio",
        "insufficient_speech",
        "severe_clipping",
        "low_energy",
        "marginal_snr",
    }
    actual = {r.value for r in ReliabilityReason}
    assert required.issubset(actual)


# 2. Immutability, serialization, and invariants of assessment model
def test_assessment_model_immutability() -> None:
    """Verify that InputReliabilityAssessment is frozen and reasons is deeply immutable."""
    assessment = InputReliabilityAssessment(
        state=ReliabilityState.RELIABLE,
        reasons=(),
        explanation="Clean audio input.",
        origin_verdict_allowed=True,
    )
    assert isinstance(assessment.reasons, tuple)
    assert not hasattr(assessment.reasons, "append")

    with pytest.raises(ValidationError):
        assessment.state = ReliabilityState.LIMITED

    with pytest.raises(ValidationError):
        assessment.origin_verdict_allowed = False

    with pytest.raises(ValidationError):
        assessment.explanation = "Modified"

    with pytest.raises(ValidationError):
        assessment.reasons = (ReliabilityReason.LOW_ENERGY,)


def test_assessment_model_serialization() -> None:
    """Verify InputReliabilityAssessment serializes to standard Pydantic JSON/dict format."""
    assessment = InputReliabilityAssessment(
        state=ReliabilityState.LIMITED,
        reasons=(ReliabilityReason.LOW_ENERGY, ReliabilityReason.MARGINAL_SNR),
        explanation="Low energy and marginal SNR.",
        origin_verdict_allowed=False,
    )
    dumped = assessment.model_dump()
    assert dumped["state"] == "limited"
    assert dumped["reasons"] == (ReliabilityReason.LOW_ENERGY, ReliabilityReason.MARGINAL_SNR)
    assert dumped["explanation"] == "Low energy and marginal SNR."
    assert dumped["origin_verdict_allowed"] is False

    json_str = assessment.model_dump_json()
    parsed = json.loads(json_str)
    assert parsed["state"] == "limited"
    assert parsed["reasons"] == ["low_energy", "marginal_snr"]
    assert parsed["origin_verdict_allowed"] is False


def test_assessment_model_forbids_extra_fields() -> None:
    """Verify assessment model forbids unexpected extra fields."""
    with pytest.raises(ValidationError):
        InputReliabilityAssessment(
            state=ReliabilityState.RELIABLE,
            reasons=(),
            explanation="OK",
            origin_verdict_allowed=True,
            extra_field="disallowed",  # type: ignore[call-arg]
        )


def test_assessment_model_rejects_empty_or_whitespace_explanation() -> None:
    """Verify empty or whitespace-only explanations raise ValidationError."""
    with pytest.raises(ValidationError, match="non-whitespace text"):
        InputReliabilityAssessment(
            state=ReliabilityState.RELIABLE,
            reasons=(),
            explanation="",
            origin_verdict_allowed=True,
        )

    with pytest.raises(ValidationError, match="non-whitespace text"):
        InputReliabilityAssessment(
            state=ReliabilityState.RELIABLE,
            reasons=(),
            explanation="   \t\n  ",
            origin_verdict_allowed=True,
        )


def test_assessment_model_rejects_duplicate_reasons() -> None:
    """Verify duplicate reason entries raise ValidationError."""
    with pytest.raises(ValidationError, match="duplicate entries"):
        InputReliabilityAssessment(
            state=ReliabilityState.LIMITED,
            reasons=(ReliabilityReason.LOW_ENERGY, ReliabilityReason.LOW_ENERGY),
            explanation="Degraded",
            origin_verdict_allowed=False,
        )


def test_assessment_model_rejects_out_of_order_reasons() -> None:
    """Verify reasons not strictly conforming to REASON_ORDER raise ValidationError."""
    # MARGINAL_SNR precedes SEVERE_CLIPPING in this tuple, violating REASON_ORDER
    with pytest.raises(ValidationError, match="REASON_ORDER"):
        InputReliabilityAssessment(
            state=ReliabilityState.LIMITED,
            reasons=(ReliabilityReason.MARGINAL_SNR, ReliabilityReason.SEVERE_CLIPPING),
            explanation="Degraded",
            origin_verdict_allowed=False,
        )


@pytest.mark.parametrize(
    ("state", "reasons", "verdict_allowed", "error_match"),
    [
        # RELIABLE with non-empty reasons
        (
            ReliabilityState.RELIABLE,
            (ReliabilityReason.LOW_ENERGY,),
            True,
            "zero reasons",
        ),
        # RELIABLE with origin_verdict_allowed=False
        (
            ReliabilityState.RELIABLE,
            (),
            False,
            "origin_verdict_allowed must be True",
        ),
        # LIMITED with zero reasons
        (
            ReliabilityState.LIMITED,
            (),
            False,
            "at least one LIMITED_REASONS",
        ),
        # LIMITED with insufficient reasons
        (
            ReliabilityState.LIMITED,
            (ReliabilityReason.EMPTY_AUDIO,),
            False,
            "cannot contain any INSUFFICIENT_REASONS",
        ),
        # LIMITED with origin_verdict_allowed=True
        (
            ReliabilityState.LIMITED,
            (ReliabilityReason.LOW_ENERGY,),
            True,
            "origin_verdict_allowed must be False",
        ),
        # INSUFFICIENT with zero reasons
        (
            ReliabilityState.INSUFFICIENT,
            (),
            False,
            "at least one INSUFFICIENT_REASONS",
        ),
        # INSUFFICIENT with only limited reasons
        (
            ReliabilityState.INSUFFICIENT,
            (ReliabilityReason.LOW_ENERGY,),
            False,
            "at least one INSUFFICIENT_REASONS",
        ),
        # INSUFFICIENT with origin_verdict_allowed=True
        (
            ReliabilityState.INSUFFICIENT,
            (ReliabilityReason.EMPTY_AUDIO,),
            True,
            "origin_verdict_allowed must be False",
        ),
    ],
)
def test_assessment_model_rejects_contradictory_combinations(
    state: ReliabilityState,
    reasons: tuple[ReliabilityReason, ...],
    verdict_allowed: bool,
    error_match: str,
) -> None:
    """Verify contradictory manually constructed assessment objects raise ValidationError."""
    with pytest.raises(ValidationError, match=error_match):
        InputReliabilityAssessment(
            state=state,
            reasons=reasons,
            explanation="Manual test",
            origin_verdict_allowed=verdict_allowed,
        )


# 3. Reliable assessment
def test_reliable_audio_assessment() -> None:
    """Verify clean audio with sufficient speech yields reliable state and allowed origin verdict."""
    metrics = _make_clean_metrics()
    assessment = assess_input_reliability(
        metrics=metrics,
        active_speech_seconds=2.5,
        required_min_speech_seconds=1.0,
    )
    assert assessment.state == ReliabilityState.RELIABLE
    assert assessment.reasons == ()
    assert isinstance(assessment.reasons, tuple)
    assert assessment.origin_verdict_allowed is True
    assert "reliable" in assessment.explanation.lower()


# 4. Individual insufficient reasons
def test_insufficient_reason_empty_audio() -> None:
    """Verify empty audio yields insufficient state and empty_audio reason."""
    metrics = AudioQualityMetrics(
        peak_level=0.0,
        rms=0.0,
        clipping_fraction=0.0,
        is_clipped=False,
        is_silent=True,
        snr_db_estimate=0.0,
        quality_status="empty",
    )
    assessment = assess_input_reliability(
        metrics=metrics,
        active_speech_seconds=1.0,
        required_min_speech_seconds=1.0,
    )
    assert assessment.state == ReliabilityState.INSUFFICIENT
    assert assessment.reasons == (ReliabilityReason.EMPTY_AUDIO,)
    assert isinstance(assessment.reasons, tuple)
    assert assessment.origin_verdict_allowed is False


def test_insufficient_reason_silent_audio() -> None:
    """Verify silent audio yields insufficient state and silent_audio reason."""
    metrics = AudioQualityMetrics(
        peak_level=0.00005,
        rms=0.00001,
        clipping_fraction=0.0,
        is_clipped=False,
        is_silent=True,
        snr_db_estimate=0.0,
        quality_status="silent",
    )
    assessment = assess_input_reliability(
        metrics=metrics,
        active_speech_seconds=1.0,
        required_min_speech_seconds=1.0,
    )
    assert assessment.state == ReliabilityState.INSUFFICIENT
    assert assessment.reasons == (ReliabilityReason.SILENT_AUDIO,)
    assert assessment.origin_verdict_allowed is False


def test_insufficient_reason_insufficient_speech() -> None:
    """Verify active speech below required threshold yields insufficient state."""
    metrics = _make_clean_metrics()
    assessment = assess_input_reliability(
        metrics=metrics,
        active_speech_seconds=0.75,
        required_min_speech_seconds=1.0,
    )
    assert assessment.state == ReliabilityState.INSUFFICIENT
    assert assessment.reasons == (ReliabilityReason.INSUFFICIENT_SPEECH,)
    assert assessment.origin_verdict_allowed is False
    assert "active speech duration" in assessment.explanation


# 5. Individual limited reasons
def test_limited_reason_severe_clipping() -> None:
    """Verify clipped audio yields limited state and severe_clipping reason."""
    metrics = AudioQualityMetrics(
        peak_level=1.0,
        rms=0.2,
        clipping_fraction=0.02,
        is_clipped=True,
        is_silent=False,
        snr_db_estimate=22.0,
        quality_status="clipped",
    )
    assessment = assess_input_reliability(
        metrics=metrics,
        active_speech_seconds=2.0,
        required_min_speech_seconds=1.0,
    )
    assert assessment.state == ReliabilityState.LIMITED
    assert assessment.reasons == (ReliabilityReason.SEVERE_CLIPPING,)
    assert assessment.origin_verdict_allowed is False


def test_limited_reason_low_energy() -> None:
    """Verify low-energy audio yields limited state and low_energy reason."""
    metrics = AudioQualityMetrics(
        peak_level=0.02,
        rms=0.005,
        clipping_fraction=0.0,
        is_clipped=False,
        is_silent=False,
        snr_db_estimate=18.0,
        quality_status="low_energy",
    )
    assessment = assess_input_reliability(
        metrics=metrics,
        active_speech_seconds=2.0,
        required_min_speech_seconds=1.0,
    )
    assert assessment.state == ReliabilityState.LIMITED
    assert assessment.reasons == (ReliabilityReason.LOW_ENERGY,)
    assert assessment.origin_verdict_allowed is False


def test_limited_reason_marginal_snr() -> None:
    """Verify degraded SNR audio yields limited state and marginal_snr reason."""
    metrics = AudioQualityMetrics(
        peak_level=0.3,
        rms=0.05,
        clipping_fraction=0.0,
        is_clipped=False,
        is_silent=False,
        snr_db_estimate=8.5,
        quality_status="marginal_snr",
    )
    assessment = assess_input_reliability(
        metrics=metrics,
        active_speech_seconds=2.0,
        required_min_speech_seconds=1.0,
    )
    assert assessment.state == ReliabilityState.LIMITED
    assert assessment.reasons == (ReliabilityReason.MARGINAL_SNR,)
    assert assessment.origin_verdict_allowed is False


# 6. Boundary condition test: active_speech_seconds == required_min_speech_seconds
def test_boundary_active_speech_equals_minimum() -> None:
    """Verify exact boundary where active_speech_seconds == required_min is not insufficient."""
    metrics = _make_clean_metrics()
    exact_min = 1.0

    # Exactly equal: Must NOT trigger insufficient_speech
    assessment_equal = assess_input_reliability(
        metrics=metrics,
        active_speech_seconds=exact_min,
        required_min_speech_seconds=exact_min,
    )
    assert assessment_equal.state == ReliabilityState.RELIABLE
    assert assessment_equal.origin_verdict_allowed is True
    assert ReliabilityReason.INSUFFICIENT_SPEECH not in assessment_equal.reasons

    # Epsilon below: MUST trigger insufficient_speech
    assessment_below = assess_input_reliability(
        metrics=metrics,
        active_speech_seconds=exact_min - 1e-5,
        required_min_speech_seconds=exact_min,
    )
    assert assessment_below.state == ReliabilityState.INSUFFICIENT
    assert assessment_below.origin_verdict_allowed is False
    assert assessment_below.reasons == (ReliabilityReason.INSUFFICIENT_SPEECH,)

    # Epsilon above: Must NOT trigger insufficient_speech
    assessment_above = assess_input_reliability(
        metrics=metrics,
        active_speech_seconds=exact_min + 1e-5,
        required_min_speech_seconds=exact_min,
    )
    assert assessment_above.state == ReliabilityState.RELIABLE
    assert assessment_above.origin_verdict_allowed is True


# 7. Multiple simultaneous reasons and deterministic ordering using REASON_ORDER
def test_multiple_limited_reasons_deterministic_order() -> None:
    """Verify multiple limited reasons are captured in deterministic REASON_ORDER."""
    metrics = AudioQualityMetrics(
        peak_level=1.0,
        rms=0.005,  # < 0.01 -> low energy
        clipping_fraction=0.01,  # > 0.0005 -> clipped
        is_clipped=True,
        is_silent=False,
        snr_db_estimate=6.0,  # < 12.0 -> marginal SNR
        quality_status="clipped",
    )
    assessment = assess_input_reliability(
        metrics=metrics,
        active_speech_seconds=2.0,
        required_min_speech_seconds=1.0,
    )
    assert assessment.state == ReliabilityState.LIMITED
    assert assessment.origin_verdict_allowed is False
    expected_order = (
        ReliabilityReason.SEVERE_CLIPPING,
        ReliabilityReason.LOW_ENERGY,
        ReliabilityReason.MARGINAL_SNR,
    )
    assert assessment.reasons == expected_order


def test_mixed_insufficient_and_limited_reasons_deterministic_order() -> None:
    """Verify insufficient reasons take precedence in state while preserving REASON_ORDER."""
    metrics = AudioQualityMetrics(
        peak_level=1.0,
        rms=0.005,
        clipping_fraction=0.01,
        is_clipped=True,
        is_silent=False,
        snr_db_estimate=5.0,
        quality_status="clipped",
    )
    assessment = assess_input_reliability(
        metrics=metrics,
        active_speech_seconds=0.4,
        required_min_speech_seconds=1.0,
    )
    assert assessment.state == ReliabilityState.INSUFFICIENT
    assert assessment.origin_verdict_allowed is False
    expected_order = (
        ReliabilityReason.INSUFFICIENT_SPEECH,
        ReliabilityReason.SEVERE_CLIPPING,
        ReliabilityReason.LOW_ENERGY,
        ReliabilityReason.MARGINAL_SNR,
    )
    assert assessment.reasons == expected_order


def test_empty_audio_with_insufficient_speech() -> None:
    """Verify empty audio with zero speech reports both reasons deterministically."""
    metrics = AudioQualityMetrics(
        peak_level=0.0,
        rms=0.0,
        clipping_fraction=0.0,
        is_clipped=False,
        is_silent=True,
        snr_db_estimate=0.0,
        quality_status="empty",
    )
    assessment = assess_input_reliability(
        metrics=metrics,
        active_speech_seconds=0.0,
        required_min_speech_seconds=1.0,
    )
    assert assessment.state == ReliabilityState.INSUFFICIENT
    assert assessment.origin_verdict_allowed is False
    assert assessment.reasons == (
        ReliabilityReason.EMPTY_AUDIO,
        ReliabilityReason.INSUFFICIENT_SPEECH,
    )


# 8. Origin verdict allowed invariant
def test_origin_verdict_allowed_invariant() -> None:
    """Verify origin_verdict_allowed is strictly True for reliable and False otherwise."""
    clean = _make_clean_metrics()
    assert assess_input_reliability(clean, 2.0, 1.0).origin_verdict_allowed is True

    limited_metrics = AudioQualityMetrics(
        peak_level=0.2,
        rms=0.05,
        clipping_fraction=0.0,
        is_clipped=False,
        is_silent=False,
        snr_db_estimate=5.0,
        quality_status="marginal_snr",
    )
    assert assess_input_reliability(limited_metrics, 2.0, 1.0).origin_verdict_allowed is False
    assert assess_input_reliability(clean, 0.5, 1.0).origin_verdict_allowed is False


# 9. Fail-closed quality_status validation
@pytest.mark.parametrize(
    "unknown_status",
    [
        "unknown",
        "degraded",
        "noisy",
        "uncalibrated",
        "clean_audio",
        "",
        "None",
    ],
)
def test_unknown_quality_status_fails_closed(unknown_status: str) -> None:
    """Verify any quality_status not emitted by compute_quality_metrics raises ValueError."""
    metrics = AudioQualityMetrics(
        peak_level=0.5,
        rms=0.1,
        clipping_fraction=0.0,
        is_clipped=False,
        is_silent=False,
        snr_db_estimate=25.0,
        quality_status=unknown_status,
    )
    with pytest.raises(ValueError, match="Unknown AudioQualityMetrics.quality_status"):
        assess_input_reliability(metrics, 2.0, 1.0)


# 10. Fail-closed physical metric range validation
def test_negative_peak_level_fails_closed() -> None:
    """Verify negative peak_level raises ValueError."""
    metrics = AudioQualityMetrics(
        peak_level=-0.01,
        rms=0.1,
        clipping_fraction=0.0,
        is_clipped=False,
        is_silent=False,
        snr_db_estimate=20.0,
        quality_status="clean",
    )
    with pytest.raises(ValueError, match="peak_level must be finite and >= 0"):
        assess_input_reliability(metrics, 2.0, 1.0)


def test_negative_rms_fails_closed() -> None:
    """Verify negative rms raises ValueError."""
    metrics = AudioQualityMetrics(
        peak_level=0.5,
        rms=-0.001,
        clipping_fraction=0.0,
        is_clipped=False,
        is_silent=False,
        snr_db_estimate=20.0,
        quality_status="clean",
    )
    with pytest.raises(ValueError, match="rms must be finite and >= 0"):
        assess_input_reliability(metrics, 2.0, 1.0)


@pytest.mark.parametrize("invalid_clipping", [-0.01, 1.05, 2.0])
def test_clipping_fraction_out_of_range_fails_closed(invalid_clipping: float) -> None:
    """Verify clipping_fraction outside [0, 1] raises ValueError."""
    metrics = AudioQualityMetrics(
        peak_level=0.5,
        rms=0.1,
        clipping_fraction=invalid_clipping,
        is_clipped=False,
        is_silent=False,
        snr_db_estimate=20.0,
        quality_status="clean",
    )
    with pytest.raises(ValueError, match=r"clipping_fraction must be finite and in range \[0, 1\]"):
        assess_input_reliability(metrics, 2.0, 1.0)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("peak_level", float("nan")),
        ("peak_level", float("inf")),
        ("rms", float("nan")),
        ("rms", float("inf")),
        ("clipping_fraction", float("nan")),
        ("clipping_fraction", float("inf")),
        ("snr_db_estimate", float("nan")),
        ("snr_db_estimate", float("inf")),
        ("snr_db_estimate", float("-inf")),
    ],
)
def test_non_finite_metric_fields_fail_closed(field: str, bad_value: float) -> None:
    """Verify non-finite float fields in AudioQualityMetrics raise ValueError."""
    kwargs: dict[str, Any] = {
        "peak_level": 0.5,
        "rms": 0.1,
        "clipping_fraction": 0.0,
        "is_clipped": False,
        "is_silent": False,
        "snr_db_estimate": 20.0,
        "quality_status": "clean",
    }
    kwargs[field] = bad_value
    metrics = AudioQualityMetrics(**kwargs)
    with pytest.raises(ValueError, match=r"must be (?:a )?finite"):
        assess_input_reliability(metrics, 2.0, 1.0)


# 11. Negative finite SNR acceptance
def test_negative_finite_snr_accepted_as_marginal_snr() -> None:
    """Verify negative finite SNR is accepted and classified as marginal_snr."""
    metrics = AudioQualityMetrics(
        peak_level=0.5,
        rms=0.1,
        clipping_fraction=0.0,
        is_clipped=False,
        is_silent=False,
        snr_db_estimate=-5.0,  # Negative finite SNR
        quality_status="marginal_snr",
    )
    assessment = assess_input_reliability(metrics, 2.0, 1.0)
    assert assessment.state == ReliabilityState.LIMITED
    assert assessment.reasons == (ReliabilityReason.MARGINAL_SNR,)
    assert assessment.origin_verdict_allowed is False


# 12. Fail-closed duration parameter validation
@pytest.mark.parametrize(
    "invalid_speech",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        -0.01,
        -10.0,
    ],
)
def test_invalid_active_speech_seconds_fail_closed(invalid_speech: float) -> None:
    """Verify NaN, Inf, and negative active speech seconds raise ValueError."""
    metrics = _make_clean_metrics()
    with pytest.raises(ValueError):
        assess_input_reliability(metrics, invalid_speech, 1.0)


@pytest.mark.parametrize(
    "invalid_speech_type",
    [
        "1.0",
        None,
        [1.0],
        True,
        False,
    ],
)
def test_invalid_active_speech_type_fail_closed(invalid_speech_type: Any) -> None:
    """Verify non-numeric active speech seconds types raise TypeError."""
    metrics = _make_clean_metrics()
    with pytest.raises(TypeError):
        assess_input_reliability(metrics, invalid_speech_type, 1.0)


@pytest.mark.parametrize(
    "invalid_min",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        0.0,
        -0.01,
        -1.0,
    ],
)
def test_invalid_required_min_speech_seconds_fail_closed(invalid_min: float) -> None:
    """Verify NaN, Inf, zero, and negative minimum speech seconds raise ValueError."""
    metrics = _make_clean_metrics()
    with pytest.raises(ValueError):
        assess_input_reliability(metrics, 1.5, invalid_min)


@pytest.mark.parametrize(
    "invalid_min_type",
    [
        "1.0",
        None,
        [1.0],
        True,
        False,
    ],
)
def test_invalid_required_min_speech_type_fail_closed(invalid_min_type: Any) -> None:
    """Verify non-numeric required minimum speech seconds types raise TypeError."""
    metrics = _make_clean_metrics()
    with pytest.raises(TypeError):
        assess_input_reliability(metrics, 1.5, invalid_min_type)


# 13. Correct invalid-metrics-type test
def test_invalid_metrics_type_fail_closed() -> None:
    """Verify non-AudioQualityMetrics instance passed positionally raises TypeError with explicit message."""
    with pytest.raises(
        TypeError, match=r"metrics must be an AudioQualityMetrics instance, got str"
    ):
        assess_input_reliability("not_a_metric", 1.5, 1.0)  # type: ignore[arg-type]

    with pytest.raises(
        TypeError, match=r"metrics must be an AudioQualityMetrics instance, got NoneType"
    ):
        assess_input_reliability(None, 1.5, 1.0)  # type: ignore[arg-type]


# 14. Policy signature inspection: no detector/model score parameters
def test_policy_has_no_detector_score_parameter() -> None:
    """Verify assess_input_reliability signature does not accept or inspect model scores."""
    sig = inspect.signature(assess_input_reliability)
    param_names = list(sig.parameters.keys())

    # Only quality metrics and speech duration bounds allowed
    assert param_names == ["metrics", "active_speech_seconds", "required_min_speech_seconds"]

    forbidden_keywords = {
        "score",
        "model",
        "detector",
        "synthetic",
        "spoof",
        "bonafide",
        "human",
        "logit",
        "prob",
        "prediction",
        "verdict",
    }
    for param in param_names:
        for kw in forbidden_keywords:
            assert kw not in param.lower(), f"Forbidden keyword '{kw}' found in parameter '{param}'"


# 15. Integration with physical compute_quality_metrics
def test_integration_with_compute_quality_metrics() -> None:
    """Verify end-to-end integration between compute_quality_metrics and assess_input_reliability."""
    # Empty audio
    empty_samples = np.array([], dtype=np.float32)
    empty_metrics = compute_quality_metrics(empty_samples)
    assessment_empty = assess_input_reliability(empty_metrics, 0.0, 1.0)
    assert assessment_empty.state == ReliabilityState.INSUFFICIENT
    assert ReliabilityReason.EMPTY_AUDIO in assessment_empty.reasons

    # Silent audio
    silent_samples = np.zeros(16000, dtype=np.float32)
    silent_metrics = compute_quality_metrics(silent_samples)
    assessment_silent = assess_input_reliability(silent_metrics, 0.0, 1.0)
    assert assessment_silent.state == ReliabilityState.INSUFFICIENT
    assert ReliabilityReason.SILENT_AUDIO in assessment_silent.reasons

    # Clean sine wave
    t = np.linspace(0, 1.0, 16000, endpoint=False, dtype=np.float32)
    clean_samples = 0.5 * np.sin(2 * np.pi * 440 * t)
    clean_metrics = compute_quality_metrics(clean_samples)
    assessment_clean = assess_input_reliability(clean_metrics, 1.5, 1.0)
    assert assessment_clean.state == ReliabilityState.RELIABLE
    assert assessment_clean.origin_verdict_allowed is True
    assert assessment_clean.reasons == ()

    # Clipped audio
    clipped_samples = np.clip(clean_samples * 10.0, -1.0, 1.0)
    clipped_metrics = compute_quality_metrics(clipped_samples)
    assessment_clipped = assess_input_reliability(clipped_metrics, 1.5, 1.0)
    assert assessment_clipped.state == ReliabilityState.LIMITED
    assert assessment_clipped.origin_verdict_allowed is False
    assert ReliabilityReason.SEVERE_CLIPPING in assessment_clipped.reasons
