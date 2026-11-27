"""Replay labeled product audio through the file and streaming decision paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import soundfile as sf

from voiceshield.config.loader import ConfigError, load_config
from voiceshield.config.schema import VoiceShieldConfig
from voiceshield.data.io import ManifestIOError, load_manifest
from voiceshield.data.leakage import check_leakage
from voiceshield.data.schema import ManifestRecord
from voiceshield.dsp.exceptions import AudioPreprocessingError
from voiceshield.evaluation.evaluator import get_duration_bucket
from voiceshield.models.aasist_adapter import AASISTLDetector
from voiceshield.models.base import BaseDetector
from voiceshield.pipeline.file_evaluator import evaluate_audio
from voiceshield.streaming.policy import DecisionState, TemporalDecision
from voiceshield.streaming.session import StreamingSession, WindowForInference


@dataclass(frozen=True)
class Observation:
    """Policy outcome on one trial. Timings use the audio timeline."""

    covered: bool
    alert_times: tuple[float, ...]


@dataclass(frozen=True)
class Trial:
    record: ManifestRecord
    seconds: float
    file: Observation
    stream: Observation
    file_candidates: tuple[tuple[float, float], ...]
    file_partial: bool
    stream_candidates: tuple[tuple[float, float], ...]


def first_high_times(decisions: list[TemporalDecision]) -> tuple[float, ...]:
    """Count a transition into sustained HIGH as one alert episode."""
    previous_high = False
    alerts = []
    for decision in decisions:
        high = decision.state == DecisionState.HIGH
        if high and not previous_high:
            alerts.append(decision.window_end_sec)
        previous_high = high
    return tuple(alerts)


def stream_decisions(
    path: Path, detector: BaseDetector, config: VoiceShieldConfig
) -> list[TemporalDecision]:
    """Run the production streaming core with 20 ms PCM frames and synchronous inference."""
    session = StreamingSession(config=config)
    decisions: list[TemporalDecision] = []

    def score(window: WindowForInference) -> None:
        decision = session.complete_inference(
            synthetic_score=detector.predict(window.samples).synthetic_score, window=window
        )
        if decision is not None:
            decisions.append(decision)

    with sf.SoundFile(path) as audio:
        rate = audio.samplerate
        offset = 0
        for frame in audio.blocks(blocksize=max(1, rate // 50), dtype="float32", always_2d=True):
            ingested = session.ingest_pcm(
                frame.tobytes(),
                sample_rate=rate,
                channels=audio.channels,
                timestamp=offset / rate,
                dtype="float32",
            )
            offset += len(frame)
            decisions.extend(ingested.immediate_decisions)
            if window := session.pop_pending_window():
                score(window)
    windows, abstentions = session.finalize()
    decisions.extend(event.decision for event in abstentions)
    for window in windows:
        score(window)
    return sorted(decisions, key=lambda decision: decision.window_index)


def replay_trial(
    record: ManifestRecord,
    path: Path,
    detector: BaseDetector,
    config: VoiceShieldConfig,
    short_config: VoiceShieldConfig,
) -> Trial:
    """Evaluate the configured modes and collect scores for candidate alert rules."""
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != record.sha256:
        raise ValueError(f"Audio SHA-256 mismatch for {record.sample_id}.")

    file = evaluate_audio(path, detector, config)
    if file.status == "error":
        raise ValueError(f"Upload failed for {record.sample_id}: {file.error_message}")
    seconds = sf.info(path).duration
    if seconds <= 0:
        raise ValueError(f"Missing audio duration for {record.sample_id}.")

    alerted = file.final_decision_state == DecisionState.HIGH
    file_observation = Observation(
        covered=file.final_decision_state in (DecisionState.LOW, DecisionState.HIGH),
        alert_times=(seconds,) if alerted else (),
    )
    candidate_file = file
    if not file.windows and seconds < config.windowing.window_seconds:
        candidate_file = evaluate_audio(path, detector, short_config)

    file_candidates: tuple[tuple[float, float], ...] = ()
    if candidate_file.status == "success":
        top = sorted(
            (w.synthetic_score for w in candidate_file.windows if w.synthetic_score is not None),
            reverse=True,
        )[:2]
        if top:
            file_candidates = ((sum(top) / len(top), seconds),)

    stream = stream_decisions(path, detector, config)
    stream_alerts = first_high_times(stream)
    stream_observation = Observation(
        covered=bool(stream_alerts) or any(d.state == DecisionState.LOW for d in stream),
        alert_times=stream_alerts,
    )

    recent: list[float] = []
    previous_index = -1
    candidate_stream: list[tuple[float, float]] = []
    for decision in stream:
        if decision.window_index != previous_index + 1:
            recent.clear()
        previous_index = decision.window_index
        if decision.synthetic_score is None:
            recent.clear()
        else:
            recent = [*recent[-2:], decision.synthetic_score]
        # A missing or ineligible window breaks the candidate's run.
        if len(recent) >= 2:
            candidate_stream.append((sorted(recent)[-2], decision.window_end_sec))
        else:
            candidate_stream.append((-1.0, decision.window_end_sec))
    return Trial(
        record,
        seconds,
        file_observation,
        stream_observation,
        file_candidates,
        any(window.is_partial for window in candidate_file.windows),
        tuple(candidate_stream),
    )


def select_threshold(scores: list[float], target_fpr: float) -> float:
    """Choose the lowest supported threshold under the dev bona fide clip FPR cap."""
    descending = sorted(scores, reverse=True)
    allowed = math.floor(len(descending) * target_fpr)
    if allowed == len(descending):
        return 0.0
    return max(0.0, math.nextafter(descending[allowed], math.inf))


def candidate_observation(trace: tuple[tuple[float, float], ...], threshold: float) -> Observation:
    alerts = []
    previous_high = False
    for score, end in trace:
        high = score >= threshold
        if high and not previous_high:
            alerts.append(end)
        previous_high = high
    return Observation(any(score >= 0.0 for score, _ in trace), tuple(alerts))


def summarize(rows: list[tuple[Trial, Observation]]) -> dict[str, Any]:
    """Use all labeled clips as denominators, including cases with abstentions."""
    spoof = [(trial, result) for trial, result in rows if trial.record.label == "spoof"]
    genuine = [(trial, result) for trial, result in rows if trial.record.label == "bonafide"]
    first_alerts = [result.alert_times[0] for _, result in spoof if result.alert_times]
    false_alerts = sum(len(result.alert_times) for _, result in genuine)
    genuine_seconds = sum(trial.seconds for trial, _ in genuine)
    abstentions = sum(not result.covered for _, result in rows)
    return {
        "trials": len(rows),
        "spoof_trials": len(spoof),
        "bonafide_trials": len(genuine),
        "spoof_recall": len(first_alerts) / len(spoof) if spoof else None,
        "bonafide_clip_fpr": (
            sum(bool(result.alert_times) for _, result in genuine) / len(genuine)
            if genuine
            else None
        ),
        "abstention_rate": abstentions / len(rows) if rows else None,
        "abstentions": abstentions,
        "false_alerts": false_alerts,
        "bonafide_audio_hours": genuine_seconds / 3600,
        "false_alerts_per_bonafide_hour": (
            3600 * false_alerts / genuine_seconds if genuine_seconds else None
        ),
        "median_seconds_to_first_spoof_alert": median(first_alerts) if first_alerts else None,
    }


def compare(trials: list[Trial], target_fpr: float) -> dict[str, Any]:
    """Fit candidate thresholds on dev and report each policy on unseen test data."""
    dev_genuine = [t for t in trials if t.record.split == "dev" and t.record.label == "bonafide"]
    test = [t for t in trials if t.record.split == "test"]
    candidate_traces: dict[str, Callable[[Trial], tuple[tuple[float, float], ...]]] = {
        "file_top_2": lambda trial: () if trial.file_partial else trial.file_candidates,
        "file_top_2_with_short_files": lambda trial: trial.file_candidates,
        "stream_unsmoothed_2_of_3": lambda trial: trial.stream_candidates,
    }
    thresholds = {
        name: select_threshold(
            [max((score for score, _ in trace(trial)), default=-1.0) for trial in dev_genuine],
            target_fpr,
        )
        for name, trace in candidate_traces.items()
    }
    policies = {
        "file_current": [trial.file for trial in test],
        "stream_current": [trial.stream for trial in test],
        **{
            name: [candidate_observation(trace(trial), thresholds[name]) for trial in test]
            for name, trace in candidate_traces.items()
        },
    }
    # Keep source labels and durations in the report as aggregates only.
    results: dict[str, Any] = {}
    for name, observations in policies.items():
        rows = list(zip(test, observations, strict=True))
        slices: dict[str, dict[str, list[tuple[Trial, Observation]]]] = {
            dimension: defaultdict(list)
            for dimension in ("generator", "codec", "language", "duration")
        }
        for trial, result in rows:
            record = trial.record
            slices["generator"][record.generator_family or record.label].append((trial, result))
            slices["codec"][record.codec_channel_condition].append((trial, result))
            slices["language"][record.language].append((trial, result))
            slices["duration"][get_duration_bucket(trial.seconds)].append((trial, result))
        results[name] = {
            "test": summarize(rows),
            "test_slices": {
                dimension: {value: summarize(group) for value, group in groups.items()}
                for dimension, groups in slices.items()
            },
        }
    return {
        "dev_target_bonafide_clip_fpr": target_fpr,
        "dev_bonafide_trials": len(dev_genuine),
        "candidate_thresholds": thresholds,
        "candidate_dev_bonafide_clip_fpr": {
            name: sum(
                bool(candidate_observation(trace(trial), thresholds[name]).alert_times)
                for trial in dev_genuine
            )
            / len(dev_genuine)
            for name, trace in candidate_traces.items()
        },
        "policies": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="Labeled CSV or JSONL.")
    parser.add_argument("--config", type=Path, help="Optional VoiceShield YAML config.")
    parser.add_argument(
        "--target-fpr",
        type=float,
        default=0.01,
        help="Maximum candidate false-positive clip rate on development audio (default: 0.01).",
    )
    args = parser.parse_args(argv)
    try:
        if not 0.0 <= args.target_fpr <= 1.0:
            raise ValueError("--target-fpr must be between 0 and 1.")
        manifest_bytes = args.manifest.read_bytes()
        records = load_manifest(args.manifest)
        for split in ("dev", "test"):
            if {record.label for record in records if record.split == split} != {
                "bonafide",
                "spoof",
            }:
                raise ValueError(f"The {split} split must contain bona fide and spoof trials.")
        if any(
            record.split is None or not record.consent_licence_ref.strip() for record in records
        ):
            raise ValueError("Each trial needs a split and a consent or license reference.")
        leakage = check_leakage(records)
        if leakage.has_leakage:
            raise ValueError("Manifest leakage detected. " + " ".join(leakage.summary_messages[:5]))
        dev_generators = {r.generator_family for r in records if r.split != "test"}
        held_out = {r.generator_family for r in records if r.split == "test"} - dev_generators
        config = load_config(args.config)
        short_config = VoiceShieldConfig.model_validate(
            {
                **config.model_dump(),
                "windowing": {**config.windowing.model_dump(), "allow_short_files": True},
            }
        )
        detector = AASISTLDetector(
            checkpoint_path=config.model.checkpoint_path,
            model_id=config.model.model_id,
            preferred_device=config.model.device_preferred,
            fallback_to_cpu=config.model.fallback_to_cpu,
        )
        trials = [
            replay_trial(record, args.manifest.parent / record.path, detector, config, short_config)
            for record in records
            if record.split in ("dev", "test")
        ]
        report = {
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "config_hash": config.compute_hash(),
            "short_file_config_hash": short_config.compute_hash(),
            "model": detector.provenance.model_dump(mode="json"),
            "held_out_generators": sorted(g for g in held_out if g),
            "stream_replay": "20 ms PCM frames through StreamingSession with synchronous inference",
            **compare(trials, args.target_fpr),
        }
    except (
        AudioPreprocessingError,
        ConfigError,
        ManifestIOError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"Field evaluation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
