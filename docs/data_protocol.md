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

## 4. Leakage Vectors & Automated Prevention

| Leakage Dimension | Definition | Mitigation in VoiceShield Data Layer |
| :--- | :--- | :--- |
| **Speaker Leakage** | Same human speaker appears in both `train` and `dev`/`test`. | `speaker_disjoint_split` groups all records by `speaker_id` and places the entire speaker cluster into a single split. |
| **Source Utterance Leakage** | Voice-converted (VC) speech or synthesis prompt uses the same transcript/recording as bona fide audio in another split. | `check_leakage` tracks `source_utterance_id` and verifies zero intersection between splits. |
| **Generator Leakage** | The model is evaluated on the exact same vocoder or synthesis engine seen in training, masking failure on novel architectures. | `generator_holdout_split` isolates designated generator families exclusively for `dev` and `test`. |
| **Codec Leakage** | Telephony codec artifacts are present in training, preventing true zero-shot channel evaluation. | `codec_holdout_split` isolates designated transmission codecs for evaluation splits. |
| **Duplicate Audio** | Identical recordings labeled under different filenames or IDs. | `sha256` hashing is computed on all raw files and checked for duplicates across the entire manifest. |

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
2. **No Post-Test Tuning**: Threshold selection, score calibration (temperature scaling), and early stopping must be conducted strictly on the `dev` split. The final `test` split is evaluated exactly once for final reporting.
