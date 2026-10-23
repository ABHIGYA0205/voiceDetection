"""Deterministic dataset splitters enforcing connected-component isolation, speaker-disjointness, generator holdout, and codec holdout."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Callable
from typing import Literal, TypeVar

from voiceshield.data.schema import ManifestRecord, SplitType

SplitMode = Literal["speaker_disjoint", "generator_holdout", "codec_holdout"]
T = TypeVar("T", bound=str)


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


def _allocate_counts(total: int, ratios: dict[T, float]) -> dict[T, int]:
    """Distribute total items across splits with positive ratios using largest remainder method."""
    pos_splits: list[T] = [s for s, r in ratios.items() if r > 0.0]
    if not pos_splits or total <= 0:
        return dict.fromkeys(ratios, 0)

    if total < len(pos_splits):
        raise ValueError(
            f"Cannot allocate {total} atomic components across {len(pos_splits)} positive-ratio splits ({pos_splits}) without leaving a split empty."
        )

    alloc: dict[T, int] = dict.fromkeys(ratios, 0)
    for s in pos_splits:
        alloc[s] = 1

    remaining = total - len(pos_splits)
    if remaining > 0:
        sum_pos = sum(ratios[s] for s in pos_splits)
        fractional: list[tuple[float, T]] = []

        for s in pos_splits:
            quota = remaining * (ratios[s] / sum_pos) if sum_pos > 0 else 0.0
            base = int(quota)
            alloc[s] += base
            fractional.append((quota - base, s))

        allocated_remaining = sum(alloc[s] - 1 for s in pos_splits)
        leftover = remaining - allocated_remaining
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

    target_ratios: dict[SplitType, float] = {
        "train": train_ratio,
        "dev": dev_ratio,
        "test": test_ratio,
    }
    pos_splits = [s for s, r in target_ratios.items() if r > 0.0]

    if num_components < len(pos_splits):
        raise ValueError(
            f"Too few atomic components ({num_components}) for {len(pos_splits)} positive-ratio splits: {pos_splits}."
        )

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

    # Explicit internal postconditions:
    assigned_sample_ids = set(sample_to_split.keys())
    input_sample_ids = {r.sample_id for r in records}
    if assigned_sample_ids != input_sample_ids:
        missing = input_sample_ids - assigned_sample_ids
        extra = assigned_sample_ids - input_sample_ids
        raise ValueError(f"Split assignment postcondition failed: missing={missing}, extra={extra}")

    for s in pos_splits:
        assigned_in_split = [sid for sid, split in sample_to_split.items() if split == s]
        if not assigned_in_split:
            raise ValueError(
                f"Split assignment postcondition failed: split '{s}' has positive ratio ({target_ratios[s]}) but no samples assigned."
            )

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

    target_ratios: dict[SplitType, float] = {
        "train": train_ratio,
        "dev": dev_ratio,
        "test": test_ratio,
    }
    all_splits: tuple[SplitType, ...] = ("train", "dev", "test")
    pos_splits: list[SplitType] = [s for s in all_splits if target_ratios[s] > 0.0]
    if len(components) < len(pos_splits):
        raise ValueError(
            f"Too few atomic components ({len(components)}) for {len(pos_splits)} positive-ratio splits: {pos_splits}."
        )

    # Classify whole components as holdout vs standard
    holdout_comps: list[list[ManifestRecord]] = []
    standard_comps: list[list[ManifestRecord]] = []

    for comp in components:
        if any(holdout_predicate(r) for r in comp):
            holdout_comps.append(comp)
        else:
            standard_comps.append(comp)

    eval_splits: tuple[SplitType, ...] = ("dev", "test")
    eval_pos_splits: list[SplitType] = [s for s in eval_splits if target_ratios[s] > 0.0]

    # Validate holdout feasibility
    if holdout_comps:
        if not eval_pos_splits:
            raise ValueError(
                f"Cannot hold out {holdout_desc} components when dev and test ratios are both zero."
            )

    if target_ratios["train"] > 0.0 and len(standard_comps) == 0:
        raise ValueError(
            f"Impossible split: all records belong to {holdout_desc} components, leaving train empty."
        )

    rng = random.Random(seed)
    sample_to_split: dict[str, SplitType] = {}
    assigned_counts: dict[SplitType, int] = {"train": 0, "dev": 0, "test": 0}

    # 1. Assign holdout components strictly to dev and test
    h_alloc: dict[SplitType, int] = {}
    if holdout_comps:
        h_indices = list(range(len(holdout_comps)))
        rng.shuffle(h_indices)
        shuffled_h_comps = [holdout_comps[i] for i in h_indices]

        if len(eval_pos_splits) == 1:
            h_alloc = {eval_pos_splits[0]: len(holdout_comps)}
        else:
            # Both dev and test have positive ratios
            if len(holdout_comps) >= 2:
                h_alloc = _allocate_counts(
                    len(holdout_comps),
                    {s: target_ratios[s] for s in eval_pos_splits},
                )
            else:
                # 1 holdout component: assign to split with higher ratio (tie-break dev)
                if target_ratios["dev"] >= target_ratios["test"]:
                    h_alloc = {"dev": 1, "test": 0}
                else:
                    h_alloc = {"dev": 0, "test": 1}

        h_curr = 0
        holdout_splits: tuple[SplitType, ...] = ("dev", "test")
        for split_name in holdout_splits:
            count = h_alloc.get(split_name, 0)
            assigned_counts[split_name] = count
            for comp in shuffled_h_comps[h_curr : h_curr + count]:
                for r in comp:
                    sample_to_split[r.sample_id] = split_name
            h_curr += count

    # 2. Assign standard components targeting overall ratios across remaining needs
    if standard_comps:
        s_indices = list(range(len(standard_comps)))
        rng.shuffle(s_indices)
        shuffled_s_comps = [standard_comps[i] for i in s_indices]

        needed_pos: list[SplitType] = [s for s in pos_splits if assigned_counts[s] == 0]
        if len(standard_comps) < len(needed_pos):
            raise ValueError(
                f"Too few standard components ({len(standard_comps)}) to satisfy positive-ratio splits {needed_pos} without leakage."
            )

        standard_alloc: dict[SplitType, int] = dict.fromkeys(pos_splits, 0)
        for s in needed_pos:
            standard_alloc[s] = 1

        leftover = len(standard_comps) - len(needed_pos)
        if leftover > 0:
            ideal: dict[SplitType, float] = {s: len(components) * target_ratios[s] for s in pos_splits}
            current: dict[SplitType, int] = {s: assigned_counts[s] + standard_alloc[s] for s in pos_splits}
            split_order: tuple[SplitType, ...] = ("train", "dev", "test")
            for _ in range(leftover):
                best_split = max(
                    pos_splits,
                    key=lambda s: (ideal[s] - current[s], -split_order.index(s)),
                )
                standard_alloc[best_split] += 1
                current[best_split] += 1

        s_curr = 0
        standard_splits: tuple[SplitType, ...] = ("train", "dev", "test")
        for split_name in standard_splits:
            count = standard_alloc.get(split_name, 0)
            for comp in shuffled_s_comps[s_curr : s_curr + count]:
                for r in comp:
                    sample_to_split[r.sample_id] = split_name
            s_curr += count

    # Explicit internal postconditions:
    assigned_sample_ids = set(sample_to_split.keys())
    input_sample_ids = {r.sample_id for r in records}
    if assigned_sample_ids != input_sample_ids:
        missing = input_sample_ids - assigned_sample_ids
        extra = assigned_sample_ids - input_sample_ids
        raise ValueError(f"Split assignment postcondition failed: missing={missing}, extra={extra}")

    for s in pos_splits:
        assigned_in_split = [sid for sid, split in sample_to_split.items() if split == s]
        if not assigned_in_split:
            raise ValueError(
                f"Split assignment postcondition failed: split '{s}' has positive ratio ({target_ratios[s]}) but no samples assigned."
            )

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
    if not holdout_generators:
        raise ValueError(
            "holdout_generators list must not be empty. Use speaker_disjoint_split if no holdout is desired."
        )
    manifest_generators = {r.generator_family for r in records if r.generator_family}
    missing = [g for g in holdout_generators if g not in manifest_generators]
    if missing:
        raise ValueError(
            f"Requested holdout generator(s) not found in manifest: {missing}. Available generators: {sorted(manifest_generators)}"
        )

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
    if not holdout_codecs:
        raise ValueError(
            "holdout_codecs list must not be empty. Use speaker_disjoint_split if no holdout is desired."
        )
    manifest_codecs = {r.codec_channel_condition for r in records if r.codec_channel_condition}
    missing = [c for c in holdout_codecs if c not in manifest_codecs]
    if missing:
        raise ValueError(
            f"Requested holdout codec(s) not found in manifest: {missing}. Available codecs: {sorted(manifest_codecs)}"
        )

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
