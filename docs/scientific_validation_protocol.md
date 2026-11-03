# VoiceShield Scientific Validation Protocol

## 1. Executive Summary & Purpose

This document specifies the scientific validation protocol for VoiceShield synthetic speech detection and acoustic anomaly evaluation. The protocol defines the rigorous empirical standards, fail-closed governance gates, leakage prevention criteria, partition strategies, slicing requirements, and reporting conventions that govern future evaluation of acoustic deepfake detection systems.

In accordance with VoiceShield core governance principles, **this protocol defines validation processes and standards only; it makes no empirical performance claims, universal accuracy claims, or legal guarantees.**

---

## 2. Governance, Rights & Acquisition Gate

Before any external dataset is acquired, it must be registered and receive an explicit project rights decision as described below. Acquisition may proceed only when that decision permits it. After the exact snapshot has been hashed, converted to the canonical manifest, and marked `verified`, every evaluation, calibration, training, product, or redistribution use must pass the purpose-specific **Rights-Aware Corpus Registry and Use Gate** (`src/voiceshield/data/corpus_registry.py`). Canonical manifest construction and integrity verification are lifecycle steps required to reach `verified`; they are not downstream uses authorized by the use gate.

### Lifecycle Sequence:
1. **Registration & Rights Review Before Acquisition**:
   * Every candidate corpus must be registered in `configs/corpus_registry.yaml` with documented provenance, source URLs, and declared license metadata.
   * Registration and explicit rights review happen strictly before acquisition.
   * A project rights reviewer must review license terms, restrictions (such as non-commercial clauses, redistribution limits, or attribution requirements), and record an explicit `rights_decision` (`approved`, `restricted`, `pending`, or `blocked`) along with reviewer identity and review date.
   * `pending` and `blocked` corpora cannot be acquired or verified. Candidate entries remain in `data_state: planned` with conservative unreviewed metadata (`attribution_required: null`, `raw_redistribution_status: unknown`, `allowed_purposes: []`).
2. **Acquisition & Immediate Archive Hashing**:
   * Once explicit rights review permits acquisition (`approved` or `restricted`), the raw corpus archive is acquired and immediately hashed using SHA-256 (`archive_sha256`) before unpacking or processing. Cryptographic verification cannot happen before the archive bytes are acquired.
   * The snapshot version reference (`version_ref`) must be permanently pinned.
3. **Canonical Manifest Conversion & Hashing**:
   * The raw audio files are indexed and converted into the canonical JSONL manifest format, and the manifest is cryptographically hashed with SHA-256 (`manifest_sha256`).
4. **Verification State Transition**:
   * The snapshot is marked `data_state: verified` only after complete integrity metadata (`version_ref`, `archive_sha256`, `manifest_sha256`, and positive `sample_count`) is recorded in the registry.
5. **Purpose-Specific Authorization Gate**:
   * Evaluation, calibration, training, product use, and redistribution require successful purpose-specific authorization of a verified snapshot via `authorize_corpus_usage`.
6. **Disclaimer Regarding Legal Advice**:
   * License metadata, SPDX identifiers, and project rights decisions recorded in the registry reflect internal operational compliance assessments and **do not constitute legal advice or formal legal conclusions**.

---

## 3. Data Ingestion, Canonical Manifests & Leakage Prevention

External datasets must be transformed into the canonical, immutable `ManifestRecord` format (`src/voiceshield/data/schema.py`) before any evaluation can proceed.

1. **Manifest Format**:
   * Audio files are indexed with metadata: `sample_id`, `path`, `label` (`bonafide` or `spoof`), `speaker_id`, `source_utterance_id`, `source_dataset`, `generator_family`, `generator_version`, `attack_type`, `language`, `accent_locale`, `original_sample_rate`, and `codec_channel_condition`.
2. **Pre-Evaluation Leakage Audit**:
   * Evaluation partitions must undergo automated leakage verification (`src/voiceshield/data/leakage.py`) before inference.
   * Checks strictly verify zero overlap across train, development, and test splits for:
     * **Speaker identities** (`speaker_id` intersection must be empty).
     * **Source utterances/prompts** (`source_utterance_id` intersection must be empty to prevent lexical or prompt memorization).
     * **Audio content checksums** (SHA-256 audio payload hashes must have zero overlap across splits).

---

## 4. Experimental Partition Design

To ensure scientific generalization and prevent over-optimistic performance estimates, evaluation designs must enforce strict holdout boundaries:

1. **Speaker-Disjoint Design**:
   * No speaker present in training or development sets may ever appear in the evaluation/test set.
2. **Generator-Disjoint (Unseen Attack) Design**:
   * Evaluation must include generator families and acoustic synthesis architectures that are completely absent from training splits to evaluate out-of-distribution generalization.
3. **Untouched Final Test Set**:
   * The test set is sequestered and held out until final reporting.
   * Operating thresholds, decision boundaries, and model parameters must be tuned **exclusively on development data**. Zero threshold tuning or hyperparameter adjustment is permitted on the final evaluation set.

---

## 5. Evaluation Slices & Stress Conditions

Reporting aggregate metrics alone masks critical localized vulnerabilities and systematic biases. All evaluation reports must produce sliced breakdowns across the following axes:

1. **Required Slices**:
   * **Source Dataset**: Performance disaggregated by origin corpus.
   * **Generator Family**: Performance broken down by acoustic engine (e.g., autoregressive TTS, diffusion, voice conversion, neural vocoders).
   * **Language & Locale**: Disaggregated by spoken language and regional accent.
     * *Safeguard on Sensitive Demographics*: Sensitive demographic attributes must not be inferred from audio and may only be used when explicitly supplied under appropriate consent, documented rights review, and an approved protocol.
   * **Codec & Channel**: Uncompressed PCM, telephony codecs, VoIP codecs, and cellular transmission conditions.
   * **Replay Condition**: Acoustic transmission path (digital direct, physical microphone acoustic replay, loudspeaker playback).
   * **Duration Bucket**: Sample length buckets (e.g., $<2.0\text{ s}$, $2.0\text{--}4.0\text{ s}$, $>4.0\text{ s}$).
   * **Clean vs. Degraded Input**: Explicit comparison of nominal laboratory audio versus corrupted inputs.
2. **Planned Degradation Axes**:
   * **Background Noise & SNR Bands**: Additive noise (ambient babble, office, traffic) across calibrated SNR bands (e.g., $30\text{ dB}$, $20\text{ dB}$, $10\text{ dB}$, $0\text{ dB}$).
   * **Clipping Distortion**: Amplitude saturation and clipping fractions ($1\%$, $5\%$, $10\%$).
   * **Reverberation**: Simulated room impulse responses ($RT_{60}$ from $100\text{ ms}$ to $800\text{ ms}$).
   * **Resampling Artifacts**: Downsampling and upsampling between $8\text{ kHz}$, $16\text{ kHz}$, $24\text{ kHz}$, and $48\text{ kHz}$.
   * **Channel Codecs**: Lossy compression transcoding via G.711 ($\mu$-law/A-law), AMR-WB, Opus ($6\text{--}32\text{ kbps}$), and simulated packet loss.

---

## 6. Error Analysis & Quantitative Metrics

Every evaluation report must include detailed false-positive and false-negative analyses governed by strict metric conventions.

### Explicit Directional Convention & Binary Ground Truth:
All evaluation tools and reports must strictly adhere to the frozen repository metric convention:
* Ground truth label $y = 1$ denotes **spoof / synthetic speech**.
* Ground truth label $y = 0$ denotes **bona fide human-origin speech under the corpus ground truth**.
* A higher normalized detector score denotes **more spoof-like acoustic evidence**. Unless an explicit frozen calibrator is used and validated, the score is not a confidence value or posterior probability.
* **True Positive (TP)**: Spoof sample correctly classified as spoof ($y = 1, \hat{y} = 1$).
* **False Positive (FP) / False Positive Rate (FPR)**: Bona fide sample incorrectly classified as spoof ($y = 0, \hat{y} = 1$). This corresponds to **false alarm** or **bona fide rejection**.
* **True Negative (TN)**: Bona fide sample correctly classified as bona fide ($y = 0, \hat{y} = 0$).
* **False Negative (FN) / False Negative Rate (FNR)**: Spoof sample missed and classified as bona fide ($y = 1, \hat{y} = 0$). This corresponds to **attack acceptance** or **false acceptance** of synthetic speech.
* **Biometrics Terminology Note**: Because VoiceShield treats spoof as the positive class ($y = 1$), false acceptance of an attack corresponds to $\text{FNR}$ (missed spoof), while false rejection / false alarm on bona fide speech corresponds to $\text{FPR}$. Evaluation tools must not conflate or swap $\text{FAR}/\text{FPR}$ or $\text{FRR}/\text{FNR}$ without explicitly declaring this mapping.

### Required Error Analyses:
* **False-positive analysis**: Count and inspect bona fide samples classified as spoof, broken down by every applicable required slice and degradation axis. Report recurring acoustic conditions, durations, channels, languages/locales, source corpora, and input-reliability states without making identity or intent claims.
* **False-negative analysis**: Count and inspect spoof samples classified as bona fide, broken down by generator family/version, attack type, post-processing, replay/channel condition, duration, language/locale, source corpus, and input-reliability state.
* For both directions, report absolute numerators and denominators alongside rates, document the uncertainty method, and include representative failure categories selected by a declared procedure rather than cherry-picking examples.

### Quantitative Metrics:
1. **Frozen Operating-Point Metrics**:
   * At every declared threshold, report the full confusion matrix ($\text{TP}$, $\text{FP}$, $\text{TN}$, $\text{FN}$), $\text{FPR}$, $\text{FNR}$, sensitivity/recall ($\text{TP}/(\text{TP}+\text{FN})$), and specificity ($\text{TN}/(\text{TN}+\text{FP})$), both in aggregate and for required slices with sufficient support.
   * If a report also uses FAR or FRR terminology, it must state the mapping to FNR or FPR under the spoof-positive convention above.
2. **Equal Error Rate (EER)**:
   * The operating threshold where $\text{FPR} = \text{FNR}$ (bona fide rejection equals attack acceptance), calculated using monotonic linear interpolation under the explicitly declared convention above.
3. **Minimum Detection Cost Function ($\text{minDCF}$)**:
   * At threshold $\theta$, the unnormalized detection cost under the spoof-positive convention is:
     $$\text{DCF}(\theta) = C_{\text{miss}} \cdot P_{\text{target}} \cdot \text{FNR}(\theta) + C_{\text{fa}} \cdot (1 - P_{\text{target}}) \cdot \text{FPR}(\theta)$$
     where $P_{\text{target}} = P(y=1)$ is the declared spoof target prior. The unnormalized $\text{minDCF}$ is $\min_{\theta}\text{DCF}(\theta)$ over thresholds evaluated on the frozen score set.
   * If normalized $\text{minDCF}$ is reported, divide the unnormalized minimum by $\min(C_{\text{miss}}P_{\text{target}}, C_{\text{fa}}(1-P_{\text{target}}))$ and label the result explicitly. Reports must not call the unnormalized cost normalized.
   * The parameters $C_{\text{miss}}$, $C_{\text{fa}}$, and $P_{\text{target}}$ must be defined in and loaded from the frozen evaluation configuration.
   * All three parameter values and the normalization choice must be explicitly recorded in the evaluation report. Arbitrary constants must never be labeled or assumed as universal standards.
4. **Threshold-Independent Metrics**:
   * Area under the Receiver Operating Characteristic curve (ROC-AUC) and Precision-Recall curve (PR-AUC) under the $y = 1$ (spoof) target convention.
5. **Calibration Metrics**:
   * Expected Calibration Error (ECE) and Brier score reported **only if an explicit, frozen calibrator artifact exists**. Raw heuristic scores must not be represented as calibrated posterior probabilities.
6. **Abstention & Selective Risk**:
   * Coverage and error rate under fail-closed input reliability gating (`InputReliabilityAssessment`), quantifying accuracy on evaluated subsets versus abstention rates on degraded windows.

---

## 7. Provenance, Reproducibility & Integrity Standards

To ensure complete scientific auditability, all published evaluation reports must record immutable cryptographic hashes:

1. **Required Immutable Revisions & Cryptographic Artifact Hashes**:
   * Exact VoiceShield source commit SHA; if the tree is not clean, the evaluation is non-release evidence and must also record the SHA-256 of the applied patch.
   * Detector model checkpoint SHA-256 (`checkpoint_sha256`).
   * Evaluation configuration SHA-256 (`config_sha256`).
   * Corpus registry file SHA-256 (`registry_sha256`).
   * Canonical train, development, and test manifest SHA-256 values, each recorded separately.
   * Pre-evaluation leakage-audit output SHA-256 (`leakage_audit_sha256`).
   * Raw prediction and score file SHA-256 (`scores_sha256`).
2. **Environment & Seed Pinning**:
   * Exact software runtime environment (`torch`, `torchaudio`, `numpy`, `python`, OS kernel, cuBLAS deterministic configuration `CUBLAS_WORKSPACE_CONFIG=:4096:8`).
   * Deterministic seed configuration across random number generators (`torch.manual_seed`, `np.random.seed`).

---

## 8. Anti-Misrepresentation Boundaries

VoiceShield explicitly prohibits deceptive or unscientific claims in evaluation reports, product interfaces, and documentation:

* **No Universal Accuracy Claims**: Models must never claim "100% accuracy", "zero false alarms", or "infallible detection".
* **No Identity Verification Claims**: Acoustic synthetic speech detection identifies statistical anomalousness in acoustic features; it **does not verify or authenticate caller identity**.
* **No Fraud Probability Claims**: An anomaly score is an acoustic diagnostic, **not a legal or actuarial probability of financial fraud or malicious intent**.
* **Advisory Only**: Detection outputs are advisory acoustic indicators intended for risk-aware triage and human-in-the-loop verification, never autonomous punitive decisions.
