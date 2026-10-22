"""Data leakage verification across speaker, utterance, generator, and channel splits."""

from __future__ import annotations

from collections import defaultdict

from pydantic import BaseModel, Field

from voiceshield.data.schema import ManifestRecord


class LeakageReport(BaseModel):
    """Structured report documenting partition disjointness and duplicate checks."""

    has_leakage: bool = Field(description="Whether any prohibited data leakage was detected.")
    speaker_leakage: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Speakers present in multiple splits (e.g. 'train_vs_test': ['spk1']).",
    )
    source_utterance_leakage: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Source utterance IDs present across splits.",
    )
    generator_leakage: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Generator families present in train that were designated as holdouts.",
    )
    codec_leakage: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Codecs present in train that were designated as holdouts.",
    )
    duplicate_checksums: list[str] = Field(
        default_factory=list,
        description="SHA-256 checksums appearing on multiple distinct sample IDs.",
    )
    summary_messages: list[str] = Field(
        default_factory=list,
        description="Human-readable diagnostics of all identified leakages.",
    )


def check_leakage(
    records: list[ManifestRecord],
    holdout_generators: list[str] | None = None,
    holdout_codecs: list[str] | None = None,
) -> LeakageReport:
    """Analyze a manifest for cross-split leakage and duplicate checksums.

    Args:
        records: List of manifest records with assigned splits.
        holdout_generators: Optional list of generator families that must NOT appear in train.
        holdout_codecs: Optional list of codecs that must NOT appear in train.

    Returns:
        Populated LeakageReport model.
    """
    speakers_by_split: dict[str, set[str]] = defaultdict(set)
    utterances_by_split: dict[str, set[str]] = defaultdict(set)
    generators_by_split: dict[str, set[str]] = defaultdict(set)
    codecs_by_split: dict[str, set[str]] = defaultdict(set)
    samples_by_checksum: dict[str, list[str]] = defaultdict(list)

    for rec in records:
        samples_by_checksum[rec.sha256].append(rec.sample_id)
        if rec.split is not None:
            speakers_by_split[rec.split].add(rec.speaker_id)
            if rec.source_utterance_id:
                utterances_by_split[rec.split].add(rec.source_utterance_id)
            if rec.generator_family:
                generators_by_split[rec.split].add(rec.generator_family)
            if rec.codec_channel_condition:
                codecs_by_split[rec.split].add(rec.codec_channel_condition)

    speaker_leakage: dict[str, list[str]] = {}
    utterance_leakage: dict[str, list[str]] = {}
    generator_leakage: dict[str, list[str]] = {}
    codec_leakage: dict[str, list[str]] = {}
    duplicate_checksums: list[str] = []
    messages: list[str] = []

    # 1. Pairwise split overlaps for speakers and source utterances
    split_pairs = [("train", "dev"), ("train", "test"), ("dev", "test")]
    for s1, s2 in split_pairs:
        pair_key = f"{s1}_vs_{s2}"
        # Speaker overlap
        common_speakers = sorted(speakers_by_split[s1].intersection(speakers_by_split[s2]))
        if common_speakers:
            speaker_leakage[pair_key] = common_speakers
            messages.append(
                f"PROHIBITED SPEAKER LEAKAGE between {s1} and {s2}: {len(common_speakers)} shared speaker(s) {common_speakers[:5]}."
            )

        # Source utterance overlap
        common_utterances = sorted(utterances_by_split[s1].intersection(utterances_by_split[s2]))
        if common_utterances:
            utterance_leakage[pair_key] = common_utterances
            messages.append(
                f"PROHIBITED UTTERANCE LEAKAGE between {s1} and {s2}: {len(common_utterances)} shared utterance(s) {common_utterances[:5]}."
            )

    # 2. Generator holdout check (holdouts must not appear in train)
    if holdout_generators:
        train_gens = generators_by_split.get("train", set())
        leaked_gens = sorted(train_gens.intersection(set(holdout_generators)))
        if leaked_gens:
            generator_leakage["train_holdout_violation"] = leaked_gens
            messages.append(
                f"GENERATOR HOLDOUT LEAKAGE: Generator(s) {leaked_gens} designated as holdout appeared in training split."
            )

    # 3. Codec holdout check (holdouts must not appear in train)
    if holdout_codecs:
        train_codecs = codecs_by_split.get("train", set())
        leaked_codecs = sorted(train_codecs.intersection(set(holdout_codecs)))
        if leaked_codecs:
            codec_leakage["train_holdout_violation"] = leaked_codecs
            messages.append(
                f"CODEC HOLDOUT LEAKAGE: Codec(s) {leaked_codecs} designated as holdout appeared in training split."
            )

    # 4. Duplicate checksums
    for sha, sample_ids in samples_by_checksum.items():
        if len(sample_ids) > 1:
            duplicate_checksums.append(sha)
            messages.append(f"DUPLICATE CHECKSUM {sha[:12]}... shared by sample IDs: {sample_ids}.")

    has_leakage = bool(
        speaker_leakage
        or utterance_leakage
        or generator_leakage
        or codec_leakage
        or duplicate_checksums
    )

    return LeakageReport(
        has_leakage=has_leakage,
        speaker_leakage=speaker_leakage,
        source_utterance_leakage=utterance_leakage,
        generator_leakage=generator_leakage,
        codec_leakage=codec_leakage,
        duplicate_checksums=duplicate_checksums,
        summary_messages=messages,
    )
