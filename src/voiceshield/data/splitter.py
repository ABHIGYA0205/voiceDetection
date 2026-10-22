"""Deterministic dataset splitters enforcing speaker-disjointness, generator holdout, and codec holdout."""

from __future__ import annotations

import random
from typing import Literal

from voiceshield.data.schema import ManifestRecord

SplitMode = Literal["speaker_disjoint", "generator_holdout", "codec_holdout"]


def speaker_disjoint_split(
    records: list[ManifestRecord],
    train_ratio: float = 0.70,
    dev_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> list[ManifestRecord]:
    """Partition records such that no speaker (or source utterance) spans multiple splits.

    Args:
        records: List of manifest records.
        train_ratio: Fraction of speakers allocated to training.
        dev_ratio: Fraction of speakers allocated to dev/validation.
        test_ratio: Fraction of speakers allocated to test.
        seed: Random seed for deterministic partition.

    Returns:
        New list of ManifestRecord instances with populated 'split' fields.
    """
    if not (0.99 <= train_ratio + dev_ratio + test_ratio <= 1.01):
        raise ValueError("Split ratios must sum to 1.0.")

    # Group speakers and link utterances to speakers to prevent cross-speaker prompt leakage
    speakers = sorted({rec.speaker_id for rec in records})
    rng = random.Random(seed)
    rng.shuffle(speakers)

    num_speakers = len(speakers)
    n_train = max(1, int(round(num_speakers * train_ratio)))
    n_dev = max(1, int(round(num_speakers * dev_ratio)))

    train_speakers = set(speakers[:n_train])
    dev_speakers = set(speakers[n_train : n_train + n_dev])
    test_speakers = set(speakers[n_train + n_dev :])

    # If rounding left test empty, ensure test has at least 1 speaker if total >= 3
    if not test_speakers and num_speakers >= 3:
        test_speakers.add(dev_speakers.pop())

    speaker_to_split: dict[str, Literal["train", "dev", "test"]] = {}
    for spk in train_speakers:
        speaker_to_split[spk] = "train"
    for spk in dev_speakers:
        speaker_to_split[spk] = "dev"
    for spk in test_speakers:
        speaker_to_split[spk] = "test"

    split_records: list[ManifestRecord] = []
    for rec in records:
        target_split = speaker_to_split[rec.speaker_id]
        rec_dict = rec.model_dump()
        rec_dict["split"] = target_split
        split_records.append(ManifestRecord.model_validate(rec_dict))

    return split_records


def generator_holdout_split(
    records: list[ManifestRecord],
    holdout_generators: list[str],
    train_ratio: float = 0.70,
    dev_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> list[ManifestRecord]:
    """Partition records holding out designated generator families strictly for dev and test.

    All samples with generator_family in holdout_generators are partitioned 50/50 between
    dev and test; zero holdout samples are placed in train.
    Remaining samples are partitioned using standard speaker-disjoint splitting.
    """
    holdout_set = set(holdout_generators)
    holdout_samples: list[ManifestRecord] = []
    standard_samples: list[ManifestRecord] = []

    for rec in records:
        if rec.generator_family in holdout_set:
            holdout_samples.append(rec)
        else:
            standard_samples.append(rec)

    # Partition standard samples speaker-disjointly
    partitioned_standard = speaker_disjoint_split(
        standard_samples,
        train_ratio=train_ratio,
        dev_ratio=dev_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    # Partition holdout samples 50/50 between dev and test (by speaker)
    rng = random.Random(seed)
    holdout_speakers = sorted({rec.speaker_id for rec in holdout_samples})
    rng.shuffle(holdout_speakers)
    mid = len(holdout_speakers) // 2
    dev_holdout_spks = set(holdout_speakers[:mid]) if mid > 0 else set(holdout_speakers)

    partitioned_holdout: list[ManifestRecord] = []
    for rec in holdout_samples:
        rec_dict = rec.model_dump()
        target_split: Literal["dev", "test"] = (
            "dev" if rec.speaker_id in dev_holdout_spks else "test"
        )
        rec_dict["split"] = target_split
        partitioned_holdout.append(ManifestRecord.model_validate(rec_dict))

    return partitioned_standard + partitioned_holdout


def codec_holdout_split(
    records: list[ManifestRecord],
    holdout_codecs: list[str],
    train_ratio: float = 0.70,
    dev_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> list[ManifestRecord]:
    """Partition records holding out designated codecs strictly for dev and test."""
    holdout_set = set(holdout_codecs)
    holdout_samples: list[ManifestRecord] = []
    standard_samples: list[ManifestRecord] = []

    for rec in records:
        if rec.codec_channel_condition in holdout_set:
            holdout_samples.append(rec)
        else:
            standard_samples.append(rec)

    partitioned_standard = speaker_disjoint_split(
        standard_samples,
        train_ratio=train_ratio,
        dev_ratio=dev_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    rng = random.Random(seed)
    holdout_speakers = sorted({rec.speaker_id for rec in holdout_samples})
    rng.shuffle(holdout_speakers)
    mid = len(holdout_speakers) // 2
    dev_holdout_spks = set(holdout_speakers[:mid]) if mid > 0 else set(holdout_speakers)

    partitioned_holdout: list[ManifestRecord] = []
    for rec in holdout_samples:
        rec_dict = rec.model_dump()
        target_split: Literal["dev", "test"] = (
            "dev" if rec.speaker_id in dev_holdout_spks else "test"
        )
        rec_dict["split"] = target_split
        partitioned_holdout.append(ManifestRecord.model_validate(rec_dict))

    return partitioned_standard + partitioned_holdout
