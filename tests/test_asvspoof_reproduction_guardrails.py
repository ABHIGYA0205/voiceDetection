from pathlib import Path

import pytest

import scripts.reproduce_asvspoof_eval as repro
from scripts.reproduce_asvspoof_eval import (
    OFFICIAL_EVAL_COUNTS,
    build_parser,
    classify_eer_comparison,
    format_invoked_command,
    format_optional_float,
    render_report_template,
    select_stratified_smoke_subset,
    smoke_report_replacements,
    validate_protocol_audio_coverage,
    validate_protocol_cardinality,
)


def _trials() -> list[tuple[str, str, str, str]]:
    return [
        ("spk1", "bf1", "-", "bonafide"),
        ("spk2", "sp1", "A01", "spoof"),
        ("spk3", "bf2", "-", "bonafide"),
        ("spk4", "sp2", "A02", "spoof"),
    ]


def test_protocol_cardinality_rejects_duplicate_ids() -> None:
    trials = _trials() + [("spk5", "bf1", "-", "bonafide")]
    with pytest.raises(ValueError, match="Duplicate protocol"):
        validate_protocol_cardinality(trials)


def test_protocol_cardinality_rejects_unknown_labels() -> None:
    trials = _trials() + [("spk5", "bad1", "-", "unknown")]
    with pytest.raises(ValueError, match="Unsupported protocol labels"):
        validate_protocol_cardinality(trials)


def test_protocol_audio_coverage_reports_missing_and_extra(tmp_path: Path) -> None:
    audio_dir = tmp_path / "flac"
    audio_dir.mkdir()
    (audio_dir / "bf1.flac").touch()
    (audio_dir / "sp1.flac").touch()
    (audio_dir / "enrollment.flac").touch()

    with pytest.raises(FileNotFoundError, match="missing"):
        validate_protocol_audio_coverage(_trials(), audio_dir)

    (audio_dir / "bf2.flac").touch()
    (audio_dir / "sp2.flac").touch()
    result = validate_protocol_audio_coverage(_trials(), audio_dir)
    assert result["protocol_ids"] == 4
    assert result["audio_files"] == 5
    assert result["missing_files"] == 0
    assert result["extra_files"] == 1


def test_protocol_audio_coverage_rejects_duplicate_stems(tmp_path: Path) -> None:
    audio_dir = tmp_path / "flac"
    audio_dir.mkdir()
    for name in ("bf1.flac", "sp1.flac", "bf2.flac", "sp2.flac"):
        (audio_dir / name).touch()
    nested = audio_dir / "nested"
    nested.mkdir()
    (nested / "bf1.flac").touch()
    with pytest.raises(ValueError, match="Duplicate audio"):
        validate_protocol_audio_coverage(_trials(), audio_dir)


def test_strict_official_cardinality_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    assert validate_protocol_cardinality(_trials())["total"] == 4
    with pytest.raises(ValueError, match="cardinality mismatch"):
        validate_protocol_cardinality(_trials(), strict_official=True)

    miniature_counts = {"total": 4, "bonafide": 2, "spoof": 2}
    monkeypatch.setattr(repro, "OFFICIAL_EVAL_COUNTS", miniature_counts)
    assert validate_protocol_cardinality(_trials(), strict_official=True) == miniature_counts
    assert OFFICIAL_EVAL_COUNTS == {"total": 71237, "bonafide": 7355, "spoof": 63882}


def test_stratified_smoke_selection_is_deterministic_and_two_class() -> None:
    selected = select_stratified_smoke_subset(_trials(), 3)
    assert selected == [_trials()[0], _trials()[1], _trials()[3]]
    assert {trial[3] for trial in selected} == {"bonafide", "spoof"}
    assert len(selected) == 3


@pytest.mark.parametrize("smoke_count", [0, 1])
def test_stratified_smoke_selection_rejects_invalid_count(smoke_count: int) -> None:
    with pytest.raises(ValueError, match="at least 2"):
        select_stratified_smoke_subset(_trials(), smoke_count)


def test_stratified_smoke_selection_rejects_single_class() -> None:
    trials = [("spk1", "bf1", "-", "bonafide")]
    with pytest.raises(ValueError, match="both"):
        select_stratified_smoke_subset(trials, 2)


def test_stratified_smoke_selection_rejects_class_allocation_underflow() -> None:
    trials = [("spk1", "bf1", "-", "bonafide")]
    trials.extend((f"spk{i}", f"sp{i}", "A01", "spoof") for i in range(10))
    with pytest.raises(ValueError, match=r"Insufficient bonafide.*requested 2, available 1"):
        select_stratified_smoke_subset(trials, 4)


def test_production_parser_has_no_cardinality_bypass() -> None:
    option_strings = {
        option for action in build_parser()._actions for option in action.option_strings
    }
    assert "--allow-miniature-fixture" not in option_strings


def test_eer_comparison_is_eval_only() -> None:
    matched_status, matched_diff = classify_eer_comparison(
        0.0099, split="eval", official_cardinality=True
    )
    assert matched_status == "MATCHED (within tolerance)"
    assert matched_diff == pytest.approx(0.0)

    dev_status, dev_diff = classify_eer_comparison(0.0099, split="dev", official_cardinality=False)
    assert dev_status == "NOT COMPARABLE"
    assert dev_diff is None

    unavailable_status, unavailable_diff = classify_eer_comparison(
        None, split="eval", official_cardinality=True
    )
    assert unavailable_status == "NOT COMPUTED"
    assert unavailable_diff is None


def test_skipped_smoke_is_not_reported_as_passed() -> None:
    fields = smoke_report_replacements({}, skipped=True)
    assert fields["{{ smoke_status }}"] == "SKIPPED"
    assert fields["{{ smoke_samples_count }}"] == "0"
    assert fields["{{ smoke_mean_bonafide_score }}"] == "NOT COMPUTED"
    assert fields["{{ smoke_direction_aligned }}"] == "NOT COMPUTED"


def test_command_provenance_quotes_paths_with_spaces() -> None:
    rendered = format_invoked_command(
        "/Users/Abhi/deepfake voice/.venv/bin/python",
        ["scripts/reproduce_asvspoof_eval.py", "--dataset-root", "/Users/Abhi/data set/LA"],
    )
    assert "'/Users/Abhi/deepfake voice/.venv/bin/python'" in rendered
    assert "'/Users/Abhi/data set/LA'" in rendered


def test_zero_threshold_is_rendered() -> None:
    assert format_optional_float(0.0) == "0.0000"
    assert format_optional_float(None) == "NOT COMPUTED"


def test_report_renderer_rejects_unresolved_placeholders() -> None:
    assert render_report_template("value={{ value }}", {"{{ value }}": "ok"}) == "value=ok"
    with pytest.raises(ValueError, match="Unresolved report template"):
        render_report_template("value={{ missing }}", {})


def test_report_template_and_script_are_fail_closed() -> None:
    template = Path("reports/templates/asvspoof_reproduction_report_template.md").read_text(
        encoding="utf-8"
    )
    script = Path("scripts/reproduce_asvspoof_eval.py").read_text(encoding="utf-8")
    active_text = template + "\n" + script

    for stale in (
        "0.0346",
        "71,189",
        "71189",
        "calibrated probability",
        "uncommitted_local",
        '"{{ git_dirty }}": "True"',
        "Apple Silicon M3",
    ):
        assert stale not in active_text

    assert "Official min t-DCF: NOT COMPUTED" in template
    assert "uncalibrated bona fide logit" in template
    assert "Internal metric score: `-logit[1]`" in template
    assert "Neither score is calibrated or probabilistic" in template
    assert "Checkpoint rights" in template
    assert "checkpoint bytes are not redistributed" in template
    assert "--allow-miniature-fixture" not in script


def test_active_reproduction_docs_match_fail_closed_runner() -> None:
    evaluation_card = Path("model_cards/aasist_l_evaluation_card.md").read_text(encoding="utf-8")
    reproduction_doc = Path("docs/aasist_reproduction.md").read_text(encoding="utf-8")
    implementation_plan = Path("docs/implementation_plan.md").read_text(encoding="utf-8")

    for stale in (
        "71,189",
        "63,834",
        "0.0346",
        "~98.8%",
        "~0.999",
        "--smoke --max-samples",
        "dataset setup",
    ):
        assert stale not in evaluation_card

    assert "71,237" in evaluation_card
    assert "63,882" in evaluation_card
    assert "0.0309" in evaluation_card
    assert "Official min t-DCF" in evaluation_card
    assert "NOT COMPUTED" in evaluation_card
    assert "--check-only" in evaluation_card
    assert "uncalibrated score, not a probability" in reproduction_doc
    assert "does not compute official min t-DCF" in reproduction_doc
    assert "Checkpoint rights" in reproduction_doc
    assert "Calibrated continuous probability" not in implementation_plan
    assert "No probability calibrator loaded" in implementation_plan
    assert "not the v0.1.1 as-built contract" in implementation_plan
