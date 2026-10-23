# VoiceShield Data Protocol & Leakage Prevention Rules

## 1. Overview
In voice anti-spoofing and deepfake detection, naive data splitting is the primary cause of inflated, non-reproducible benchmark performance. Machine learning detectors easily overfit to incidental acoustic features—such as specific speaker vocal tract geometry, identical transcript phoneme transitions, room acoustic impulse responses, or particular vocoder artifacts.

This protocol establishes strict rules for dataset partitioning, augmentation assignment, and leakage prevention.

---

## 2. Core Rule 1: Partitioning Before Temporal Chunking

> [!CRITICAL]
> **Data splitting must happen at the root utterance and speaker level BEFORE any temporal windowing or chunking is performed.**

### Why Chunking Before Splitting Destroys Model Validity:
VoiceShield operates on $4.0\text{-second}$ windows with $1.0\text{-second}$ sliding hops. If a $10\text{-second}$ recording is first divided into overlapping $4.0\text{-second}$ chunks and those chunks are shuffled into training and test sets:
- The training split and test split will contain identical phonetic transitions separated by only fractions of a second.
- The model memorizes short-term speaker characteristics and channel noise rather than learning generalizable synthetic speech cues.

**Enforced Order**:
1. Ingest raw recording and assign unique `speaker_id` and `source_utterance_id`.
2. Partition the manifest into `train`, `dev`, and `test` splits based strictly on speaker and utterance disjointness.
3. Apply sliding window chunking ($4.0\text{s}$ window, $1.0\text{s}$ hop) **independently within each assigned split**.

---

## 3. Core Rule 2: Partitioning Before Derived Augmentations

> [!CRITICAL]
> **Data splitting must happen BEFORE derived channel, codec, or environmental augmentations are assigned.**

### The Multi-Condition Leakage Trap:
If a clean bona fide utterance is copied and transformed into multiple synthetic channel conditions:
- Condition A: Clean PCM
- Condition B: G.711 $\mu$-law telephone codec
- Condition C: Opus $16\text{ kbps}$ VoIP stream
- Condition D: Room impulse response (RIR) reverberation

If Condition A is placed in `train` and Condition B is placed in `test`, the model is tested on the exact same underlying speech content and speaker identity. The test set fails to evaluate zero-shot detection.

**Enforced Order**:
1. Assign `split` (`train`, `dev`, `test`) to the parent utterance.
2. All augmented descendants of a parent utterance automatically inherit the parent's split.
3. Augmentations must be applied symmetrically across both bona fide and synthetic classes.

---

## 4. Leakage Vectors & Atomic Connected-Component Splitting

All split strategies (`speaker_disjoint_split`, `generator_holdout_split`, and `codec_holdout_split`) operate on **atomic connected components** built over the entire manifest:

- **Connectivity Criteria**: Two records belong to the same component if they share any non-empty value of `speaker_id`, `source_utterance_id`, or exact `sha256` audio checksum.
- **Transitivity**: Connectivity is strictly transitive (if A shares speaker with B, and B shares source utterance with C, then A, B, and C form one atomic component).
- **Holdout Isolation**: In generator or codec holdout splits, if any record in a component uses a designated holdout generator or codec, the **entire component** is forbidden from `train` and assigned atomically to `dev` or `test`. Non-holdout records transitively linked to holdout records remain outside `train`.
- **Ratio Allocation**: Split ratios currently apply to atomic component counts rather than exact record counts because components cannot be divided. Ratios are distributed via the Hare-Niemeyer largest-remainder method without splitting components.
- **Checksum Links & Duplicates**: Declared SHA-256 equality is used for component grouping and duplicate detection. The toolkit does not yet verify audio files against declared checksums and does not perform acoustic near-duplicate detection.

| Leakage Dimension | Definition | Mitigation in VoiceShield Data Layer |
| :--- | :--- | :--- |
| **Speaker Leakage** | Same human speaker appears in both `train` and `dev`/`test`. | `speaker_disjoint_split` groups all transitively connected records into atomic components and assigns whole components to a single split. |
| **Source Utterance Leakage** | Voice-converted (VC) speech or synthesis prompt uses the same transcript/recording as bona fide audio in another split. | `build_connected_components` transitively links records sharing `source_utterance_id`; `check_leakage` validates zero cross-split intersection. |
| **Generator Leakage** | The model is evaluated on the exact same vocoder or synthesis engine seen in training, masking failure on novel architectures. | `generator_holdout_split` isolates entire connected components containing holdout generators strictly to `dev` or `test`. |
| **Codec Leakage** | Telephony codec artifacts are present in training, preventing true zero-shot channel evaluation. | `codec_holdout_split` isolates entire connected components containing holdout codecs strictly to `dev` or `test`. |
| **Duplicate Audio** | Identical recordings labeled under different filenames or IDs. | Declared `sha256` checksum matches form connected components and are checked for dataset integrity. |

---

## 5. Manifest Schema (Version 1.0.0)

Every sample is tracked in a standardized `.csv` or `.jsonl` manifest:

```json
{
  "manifest_version": "1.0.0",
  "sample_id": "asv_spoof_001249",
  "path": "data/audio/asv_spoof_001249.flac",
  "label": "spoof",
  "speaker_id": "spk_401",
  "source_utterance_id": "utt_prompt_09",
  "source_dataset": "ASVspoof2019",
  "generator_family": "ElevenLabs",
  "generator_version": "v2_turbo",
  "attack_type": "TTS",
  "language": "en",
  "accent_locale": "en-US",
  "original_sample_rate": 24000,
  "codec_channel_condition": "pcm_clean",
  "replay_condition": "none",
  "duration": 4.12,
  "consent_licence_ref": "CC-BY-4.0",
  "split": "train",
  "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
}
```

---

## 6. Evaluation Protocol Rules

1. **Frozen Checkpoints**: Models evaluated on the `test` split must have weights frozen before evaluation.
2. **Threshold Isolation**: Threshold selection (e.g. EER threshold) must be conducted strictly on the `dev` split (or `train` split only if no dev split exists). The `test` split is NEVER used for threshold optimization. Manifests with pure test records or unassigned records require an explicit deployment threshold.
3. **Complete Score Coverage**: Evaluator requires 100% complete score coverage across all manifest sample IDs by default. Unknown score IDs are strictly rejected. Partial evaluation requires explicit opt-in (`--allow-partial-scores`) and is prominently reported with missing ID lists and coverage fractions.
4. **Fail-Closed Calibration**: Raw/softmax scores bounded in $[0, 1]$ are still uncalibrated. Calibration metrics (Brier score, ECE) and reliability diagrams are unavailable through the high-level evaluator until an identified calibrator artifact loader is implemented.
5. **Single-Class Guardrails**: If evaluation subsets contain only a single class, metrics dependent on two classes are marked undefined (`None`) rather than fabricating zero values.
