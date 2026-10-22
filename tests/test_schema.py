"""Unit tests for the output contract schema and advisory disclaimers."""

from voiceshield.core.schema import DetectionResult, QualityStatus, UncertaintyStatus


def test_detection_result_schema_valid() -> None:
    """Verify that a valid DetectionResult initializes with all contract fields."""
    result = DetectionResult(
        synthetic_likelihood=0.82,
        is_synthetic_suspected=True,
        window_start_seconds=1.0,
        window_end_seconds=5.0,
        quality_status=QualityStatus(
            is_clipping=False,
            snr_db_estimate=28.5,
            is_silent=False,
        ),
        uncertainty_status=UncertaintyStatus(
            is_uncertain=False,
            reasons=[],
        ),
        temporal_rule_explanation="High confidence sustained over 2 consecutive hops.",
    )

    dumped = result.model_dump()
    assert dumped["synthetic_likelihood"] == 0.82
    assert dumped["is_synthetic_suspected"] is True
    assert "disclaimers" in dumped
    assert "identity_proof" in dumped["disclaimers"]
    assert "fraud_probability" in dumped["disclaimers"]
    assert "advisory_action" in dumped["disclaimers"]
    assert "capture_boundary" in dumped["disclaimers"]
