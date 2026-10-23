"""Deterministic dataset splitters enforcing connected-component isolation, speaker-disjointness, generator holdout, and codec holdout."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Callable
from typing import Literal

from voiceshield.data.schema import ManifestRecord, SplitType

SplitMode = Literal["speaker_disjoint", "generator_holdout", "codec_holdout"]


class DisjointSet:
    """Disjoint Set Union (DSU) data structure with deterministic union-by-key."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        if item not in self.parent:
            self.parent[item] = item
            return item
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a: str, b: str) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            if root_a < root_b:
                self.parent[root_b] = root_a
            else:
                self.parent[root_a] = root_b


def build_connected_components(records: list[ManifestRecord]) -> list[list[ManifestRecord]]:
    """Partition records into connected components based on speaker_id, source_utterance_id, and sha256.

    At minimum, two records belong to the same component if they share any non-empty value of:
    - speaker_id
    - source_utterance_id
    - sha256

    Connectivity is transitive. Returns components sorted deterministically by canonical key.
    """
    if not records:
        return []

    # 1. Enforce unique sample_id values
    sample_ids: list[str] = [r.sample_id for r in records]
    if len(sample_ids) != len(set(sample_ids)):
        seen: set[str] = set()
        dups: list[str] = []
        for sid in sample_ids:
            if sid in seen and sid not in dups:
                dups.append(sid)
            seen.add(sid)
        raise ValueError(f"Duplicate sample_id found in manifest records: {dups}")

    # 2. Sort records deterministically by sample_id to guarantee identical DSU construction
    sorted_records = sorted(records, key=lambda r: r.sample_id)

    dsu = DisjointSet()
    for rec in sorted_records:
        r_node = f"sample:{rec.sample_id}"
        dsu.find(r_node)
        if rec.speaker_id and str(rec.speaker_id).strip():
            dsu.union(r_node, f"spk:{rec.speaker_id.strip()}")
        if rec.source_utterance_id and str(rec.source_utterance_id).strip():
            dsu.union(r_node, f"utt:{rec.source_utterance_id.strip()}")
        if rec.sha256 and str(rec.sha256).strip():
            dsu.union(r_node, f"sha:{rec.sha256.strip().lower()}")

    # 3. Group records by component root
    groups: dict[str, list[ManifestRecord]] = defaultdict(list)
    for rec in sorted_records:
        root = dsu.find(f"sample:{rec.sample_id}")
        groups[root].append(rec)

    # 4. Sort each component internally by sample_id, and sort all components by canonical key
    components: list[list[ManifestRecord]] = []
    for comp_records in groups.values():
        comp_records.sort(key=lambda r: r.sample_id)
        components.append(comp_records)

    # Sort components by minimum sample_id in component
    components.sort(key=lambda comp: comp[0].sample_id)
    return components


def validate_ratios(train_ratio: float, dev_ratio: float, test_ratio: float) -> None:
    """Validate that split ratios are finite, non-negative, and sum to 1.0."""
    for name, val in [
        ("train_ratio", train_ratio),
        ("dev_ratio", dev_ratio),
        ("test_ratio", test_ratio),
    ]:
        if (
            not isinstance(val, (int, float))
            or isinstance(val, bool)
            or not math.isfinite(val)
            or val < 0.0
        ):
            raise ValueError(f"Split ratio '{name}' must be finite and non-negative, got {val!r}.")

    total = train_ratio + dev_ratio + test_ratio
    if not (0.999 <= total <= 1.001):
        raise ValueError(f"Split ratios must sum to 1.0 (within numeric tolerance), got {total}.")


def _allocate_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    """Distribute total items across splits with positive ratios using largest remainder method."""
    pos_splits = [s for s, r in ratios.items() if r > 0.0]
    if not pos_splits or total <= 0:
        return dict.fromkeys(ratios, 0)

    sum_pos = sum(ratios[s] for s in pos_splits)
    alloc = dict.fromkeys(ratios, 0)
    fractional: list[tuple[float, str]] = []

    for s in pos_splits:
        quota = total * (ratios[s] / sum_pos) if sum_pos > 0 else 0.0
        base = int(quota)
        alloc[s] = base
        fractional.append((quota - base, s))

    leftover = total - sum(alloc.values())
    fractional.sort(key=lambda item: (-item[0], item[1]))
    for i in range(leftover):
        alloc[fractional[i][1]] += 1

    return alloc


def speaker_disjoint_split(
    records: list[ManifestRecord],
    train_ratio: float = 0.70,
    dev_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> list[ManifestRecord]:
    """Partition records such that no connected component spans multiple splits.

    Components are defined transitively by shared speaker_id, source_utterance_id, or sha256.

    Args:
        records: List of manifest records.
        train_ratio: Fraction allocated to train.
        dev_ratio: Fraction allocated to dev/validation.
        test_ratio: Fraction allocated to test.
        seed: Random seed for deterministic partition.

    Returns:
        New list of ManifestRecord instances with populated 'split' fields in original order.
    """
    validate_ratios(train_ratio, dev_ratio, test_ratio)
    if not records:
        return []

    components = build_connected_components(records)
    num_components = len(components)

    target_ratios = {"train": train_ratio, "dev": dev_ratio, "test": test_ratio}

    # Deterministic shuffling of components based on seed
    rng = random.Random(seed)
    shuffled_indices = list(range(num_components))
    rng.shuffle(shuffled_indices)
    shuffled_components = [components[i] for i in shuffled_indices]

    allocations = _allocate_counts(num_components, target_ratios)

    sample_to_split: dict[str, SplitType] = {}
    split_names: tuple[SplitType, ...] = ("train", "dev", "test")
    curr = 0
    for split_name in split_names:
        count = allocations.get(split_name, 0)
        for comp in shuffled_components[curr : curr + count]:
            for r in comp:
                sample_to_split[r.sample_id] = split_name
        curr += count

    return [
        ManifestRecord.model_validate({**r.model_dump(), "split": sample_to_split[r.sample_id]})
        for r in records
    ]


def _partition_holdout(
    records: list[ManifestRecord],
    holdout_predicate: Callable[[ManifestRecord], bool],
    train_ratio: float = 0.70,
    dev_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    holdout_desc: str = "holdout",
) -> list[ManifestRecord]:
    """Partition records holding out designated components strictly for dev and test."""
    validate_ratios(train_ratio, dev_ratio, test_ratio)
    if not records:
        return []

    components = build_connected_components(records)

    # Classify whole components as holdout vs standard
    holdout_comps: list[list[ManifestRecord]] = []
    standard_comps: list[list[ManifestRecord]] = []

    for comp in components:
        if any(holdout_predicate(r) for r in comp):
            holdout_comps.append(comp)
        else:
            standard_comps.append(comp)

    # Validate holdout feasibility
    if holdout_comps:
        if dev_ratio <= 0.0 and test_ratio <= 0.0:
            raise ValueError(
                f"Cannot hold out {holdout_desc} components when dev and test ratios are both zero."
            )

    if train_ratio > 0.0 and len(standard_comps) == 0:
        raise ValueError(
            f"Impossible split: all records belong to {holdout_desc} components, leaving train empty."
        )

    rng = random.Random(seed)

    # 1. Assign holdout components strictly to dev and test
    sample_to_split: dict[str, SplitType] = {}

    if holdout_comps:
        h_indices = list(range(len(holdout_comps)))
        rng.shuffle(h_indices)
        shuffled_h_comps = [holdout_comps[i] for i in h_indices]

        h_ratios = {"dev": dev_ratio, "test": test_ratio}
        h_alloc = _allocate_counts(len(holdout_comps), h_ratios)

        holdout_splits: tuple[SplitType, ...] = ("dev", "test")
        curr = 0
        for split_name in holdout_splits:
            count = h_alloc.get(split_name, 0)
            for comp in shuffled_h_comps[curr : curr + count]:
                for r in comp:
                    sample_to_split[r.sample_id] = split_name
            curr += count

    # 2. Assign standard components to train, dev, and test
    if standard_comps:
        s_indices = list(range(len(standard_comps)))
        rng.shuffle(s_indices)
        shuffled_s_comps = [standard_comps[i] for i in s_indices]

        # Allocate standard components targeting overall ratios
        target_ratios = {"train": train_ratio, "dev": dev_ratio, "test": test_ratio}
        overall_alloc = _allocate_counts(len(components), target_ratios)
        needed_train = min(len(standard_comps), overall_alloc.get("train", 0))
        if train_ratio > 0.0 and needed_train < 1:
            needed_train = 1

        for comp in shuffled_s_comps[:needed_train]:
            for r in comp:
                sample_to_split[r.sample_id] = "train"

        remaining_standard = shuffled_s_comps[needed_train:]
        if remaining_standard:
            eval_ratios = {"dev": dev_ratio, "test": test_ratio}
            eval_alloc = _allocate_counts(len(remaining_standard), eval_ratios)
            remaining_splits: tuple[SplitType, ...] = ("dev", "test")
            for split_name in remaining_splits:
                count = eval_alloc.get(split_name, 0)
                for comp in remaining_standard[curr : curr + count]:
                    for r in comp:
                        sample_to_split[r.sample_id] = split_name
                curr += count

    return [
        ManifestRecord.model_validate({**r.model_dump(), "split": sample_to_split[r.sample_id]})
        for r in records
    ]


def generator_holdout_split(
    records: list[ManifestRecord],
    holdout_generators: list[str],
    train_ratio: float = 0.70,
    dev_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> list[ManifestRecord]:
    """Partition records holding out designated generator families strictly for dev and test.

    All connected components containing at least one sample with generator_family in
    holdout_generators are partitioned strictly between dev and test; zero holdout
    component samples are placed in train.
    """
    holdout_set = set(holdout_generators)
    return _partition_holdout(
        records=records,
        holdout_predicate=lambda r: bool(r.generator_family and r.generator_family in holdout_set),
        train_ratio=train_ratio,
        dev_ratio=dev_ratio,
        test_ratio=test_ratio,
        seed=seed,
        holdout_desc=f"generator ({holdout_generators})",
    )


def codec_holdout_split(
    records: list[ManifestRecord],
    holdout_codecs: list[str],
    train_ratio: float = 0.70,
    dev_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> list[ManifestRecord]:
    """Partition records holding out designated codecs strictly for dev and test.

    All connected components containing at least one sample with codec_channel_condition in
    holdout_codecs are partitioned strictly between dev and test; zero holdout
    component samples are placed in train.
    """
    holdout_set = set(holdout_codecs)
    return _partition_holdout(
        records=records,
        holdout_predicate=lambda r: bool(
            r.codec_channel_condition and r.codec_channel_condition in holdout_set
        ),
        train_ratio=train_ratio,
        dev_ratio=dev_ratio,
        test_ratio=test_ratio,
        seed=seed,
        holdout_desc=f"codec ({holdout_codecs})",
    )
