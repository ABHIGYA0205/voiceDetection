# VoiceShield Evaluation Card: AASIST-L Baseline Detector

**Card Identifier**: `VoiceShield-EvalCard-AASIST-L-v0.1.1`<br>
**Associated Model Card**: [`aasist_l_model_card.md`](./aasist_l_model_card.md)
**Evaluation Standard**: ASVspoof 2019 Logical Access (LA) Protocol & VoiceShield M3 Benchmark Suite  
**Status**: **PARTIAL — ENGINEERING VALIDATED; SCIENTIFIC BENCHMARK PENDING**
**Card Version**: `1.0.1` (Truthful Baseline)<br>
**Date**: October 2026  

---

## 1. Executive Summary & Integrity Notice

> [!IMPORTANT]
> **Truthful Evaluation Policy**  
> In strict compliance with scientific integrity standards, VoiceShield forbids inventing evaluation metrics, claiming unverified benchmark numbers, or reporting results on cherry-picked audio files.  
> - **Upstream Clova AI Reference**: Published ICASSP 2022 metrics are recorded for architectural baseline reference.  
> - **Official checkpoint**: Acquisition and integrity verification of the official AASIST-L checkpoint are complete.
> - **Engineering validation**: The real-checkpoint API, transport, CPU/MPS benchmark, browser, and stress-soak validation are complete.
> - **Scientific evaluation status**: Official eval prerequisites were validated locally without inference: 71,237 protocol trials (7,355 bona fide and 63,882 spoof), 71,933 FLAC stems, zero missing files, and 696 permitted extras. A full scoring attempt was interrupted because of host memory pressure and is not evidence. Local EER, official min t-DCF, ROC-AUC, balanced accuracy, calibration, FPR/FNR, confidence intervals, and external/telephony performance remain not computed.
> - **Hardware profiling**: Real-checkpoint latency measurements are linked to the generated report artifact.

---

## 2. Evaluation Split Specification & Leakage Controls

### 2.1 ASVspoof 2019 Logical Access Protocol

| Split Identifier | Total Trials | Bona Fide Utterances | Spoofed Utterances | Speakers | Generators |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Train** | 25,380 | 2,580 | 22,800 | 20 (8 M, 12 F) | Known In-Set: A01–A06 |
| **Development** | 24,844 | 2,548 | 22,296 | 20 (8 M, 12 F) | Known In-Set: A01–A06 |
| **Evaluation (Target)** | **71,237** | **7,355** | **63,882** | **48 (21 M, 27 F)** | **Unseen attacks: A07–A19** |

### 2.2 Runtime integrity controls and boundaries

The reproduction runner enforces the selected official protocol file rather than
creating new dataset splits. In canonical eval mode it requires exact class
cardinality, unique protocol IDs, complete protocol-to-FLAC coverage, unique
FLAC stems, supported labels, and scoring of protocol IDs only. Extra FLACs in
the official archive are counted but are not scored.

### 2.3 Resumable execution and CUDA evaluation foundation (v0.2.0 Phase 1)

Phase 1 of v0.2.0 adds crash-safe, resumable evaluation (`--resume`, `--checkpoint-every`)
and fail-closed CUDA support (`--device cuda`) to enable full scoring on interruption-prone
environments such as Google Colab.

- **Split Cardinalities**: Supported official partitions are evaluation (exactly 71,237 trials: 7,355 bona fide, 63,882 spoof) and development (exactly 24,844 trials: 2,548 bona fide, 22,296 spoof). Upstream ICASSP 2022 EER reference comparison (0.99%) is strictly **eval-only**; development scoring is supported for diagnostic verification but is never compared with 0.99%.
- **Integrity Boundary**: Resumability and CUDA device selection do not validate model accuracy or alter benchmark formulas.
- **Completeness Enforcement**: Release evidence (`raw_scores.txt`, `report.md`) is generated strictly when all trials for the selected split are fully scored.
- **min t-DCF Status**: Official min t-DCF remains NOT COMPUTED and unimplemented.
- **Predeclared Tolerance**: Numerical tolerance between CPU and CUDA inference is predeclared at $\text{atol}=10^{-4}, \text{rtol}=10^{-4}$; live equivalence remains **UNVERIFIED** until physical CUDA tests pass on real hardware.
- **Hardware Prerequisites**: CUDA execution requires an external research-environment PyTorch build. Default project dependencies remain CPU-only.

Train/dev/eval speaker separation and attack holdout are properties of the
official ASVspoof protocol. This runner does not independently reconstruct
source utterances, verify source-text disjointness, or perform acoustic
near-duplicate detection; no stronger leakage claim is made.

---

## 3. Reported Metric Provenance & Status

### 3.1 Benchmark Comparison Table

| Metric | Upstream Published (Clova AI / ICASSP 2022) | VoiceShield Local Evaluation | Status | Report Artefact & Evidence |
| :--- | :---: | :---: | :---: | :--- |
| **Equal Error Rate (EER)** | **0.99%** (0.0099) | **NOT COMPUTED** | Pending full scoring | Upstream value only; no local comparison is claimed. |
| **Official min t-DCF** | **0.0309** | **NOT COMPUTED** | Not implemented by this runner | Requires official ASV scores, keys, priors, costs, and t-DCF implementation. |
| **VoiceShield diagnostic normalized minDCF** | Not comparable | **NOT COMPUTED** | Pending full scoring | Not official min t-DCF and never compared with 0.0309. |
| **Balanced Accuracy** | Not reported in pinned AASIST-L README | **NOT COMPUTED** | Pending full scoring | No upstream estimate is imputed. |
| **ROC-AUC** | Not reported in pinned AASIST-L README | **NOT COMPUTED** | Pending full scoring | No upstream estimate is imputed. |
| **Brier Score** | Not reported upstream | **NOT COMPUTED** | Uncalibrated | Requires an identified fitted calibrator. |
| **Expected Calibration Error (ECE)** | Not reported upstream | **NOT COMPUTED** | Uncalibrated | Requires an identified fitted calibrator. |

---

## 4. Fine-Grained Subgroup Performance Breakdowns

### 4.1 By Speech Generator Family

| Generator ID | Generator Family / Synthesis Type | Upstream Baseline EER | VoiceShield Local Status |
| :--- | :--- | :---: | :--- |
| **A07–A19 (ASVspoof 2019 LA eval)** | Historical logical-access attacks | Only aggregate AASIST-L EER is referenced | **NOT COMPUTED** |
| **Modern neural TTS / voice cloning** | Outside ASVspoof 2019 coverage | Not reported | **NOT COMPUTED**; one observed file cannot establish reliability |

### 4.2 By Language & Accent

| Language | Dataset Provenance | Local Status | Evaluation Notes |
| :--- | :--- | :--- | :--- |
| **English (UK / US)** | ASVspoof 2019 LA | **Not yet evaluated** | Standard benchmark language. |
| **Hindi / Indic Languages** | In-the-Wild / Synthetic Indic | **Not yet evaluated** | Critical for SIH; requires regional evaluation partition. |
| **Spanish / Romance Languages** | Multilingual datasets | **Not yet evaluated** | Unseen linguistic phonemes. |
| **Mandarin / Tonal Languages** | Multilingual datasets | **Not yet evaluated** | Tonal pitch contours not in training data. |

### 4.3 By Channel & Codec Degradation

| Codec / Channel Condition | Bitrate / Bandwidth | Local Status | Expected Degradation Risk |
| :--- | :--- | :--- | :--- |
| **PCM Linear (Uncompressed)** | $16\text{ kHz}$ 16-bit ($256\text{ kbps}$) | **Not yet evaluated** | Target clean training condition. |
| **G.711 ($\mu$-law / A-law)** | $8\text{ kHz}$ ($64\text{ kbps}$) | **Not yet evaluated** | High risk: bandlimited to $3.4\text{ kHz}$, phase distorted. |
| **AMR-WB (Adaptive Multi-Rate)** | $12.65\text{ kbps}$ ($16\text{ kHz}$) | **Not yet evaluated** | Moderate risk: cellular speech coding quantization. |
| **Opus (VoIP Standard)** | $16\text{ kbps}$ VBR | **Not yet evaluated** | Variable: frame-loss concealment artifacts. |

### 4.4 By Acoustic Replay Condition

| Replay Environment | Acoustic Setting | Local Status | Notes |
| :--- | :--- | :--- | :--- |
| **Logical Access (Direct Stream)** | Digital stream / direct line | **Not yet evaluated** | In-scope for AASIST-L architecture. |
| **Physical Replay (Loudspeaker)** | Room impulse response / re-recording | **Not yet evaluated** | Out-of-scope: PA artifacts differ from LA. |

---

## 5. Hardware Latency & Real-Time Factor Audit

Measurements conducted on the target host hardware and linked to verified benchmark reports:

- **Hardware Platform**: Apple MacBook Pro M3 (`Mac15,3`, 8 CPU cores, macOS 27.0.1, PyTorch 2.14.1)
- **Primary Execution Mode**: Apple MPS Metal Acceleration
- **Batch Size**: 1 (Primary real-time conversational streaming case)
- **Report Artefacts**:
  - Markdown Report: [`../reports/benchmarks/m3_benchmark_report.md`](../reports/benchmarks/m3_benchmark_report.md)
  - Raw JSON Telemetry: [`../reports/benchmarks/m3_benchmark_report.json`](../reports/benchmarks/m3_benchmark_report.json)
  - Config SHA-256: `774483ef6871c5a0c83f81098cf50cc2e691d39305c9e7c262d478bfb5088596`

### 5.1 Pipeline Stage Latencies (Batch Size 1, 4.0-second Window)

| Pipeline Stage | Device | Warm-up Mean | Steady p50 (Median) | Steady p95 | Steady p99 | Throughput | RTF | Target (<200ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage 1: Decoding/Resampling** | CPU | 18.117 ms | **6.257 ms** | 8.716 ms | 14.713 ms | 139.94 win/s | 0.00156 | N/A |
| **Stage 2: Feature Extraction/Padding** | CPU | 0.097 ms | **0.052 ms** | 0.054 ms | 0.057 ms | 19,972.38 win/s | 0.00001 | N/A |
| **Stage 3: Model Inference** | CPU | 164.333 ms | **112.999 ms** | 129.814 ms | 181.354 ms | 8.45 win/s | 0.02825 | ✅ MET |
| **Stage 3: Model Inference** | **MPS** | 70.577 ms | **30.187 ms** | **30.761 ms** | 34.043 ms | 32.91 win/s | 0.00755 | **✅ MET** |
| **Stage 4: Score Smoothing & Temporal Policy** | CPU | 0.019 ms | **0.004 ms** | 0.004 ms | 0.005 ms | 263,941.37 win/s | 0.0 | N/A |
| **Stage 5: End-to-End Processing** | CPU | 112.668 ms | **111.981 ms** | 117.614 ms | 125.402 ms | 8.83 win/s | 0.02800 | ✅ MET |
| **Stage 5: End-to-End Processing** | **MPS** | 34.418 ms | **33.754 ms** | **34.546 ms** | 37.079 ms | 29.44 win/s | 0.00844 | **✅ MET** |

### 5.2 Accelerated 600-Second Streaming Stress Soak

- **Duration**: `600.0s` (10.0 minutes)
- **Frame Duration**: `50.0 ms`
- **Total Ingested Frames**: `367,780`
- **Total Windows Evaluated**: `18,386`
- **Windows Dropped**: `0`
- **Initial RSS**: `986.81 MB` | **Peak RSS**: `986.81 MB` | **Final RSS**: `986.81 MB`
- **Net Memory Growth**: **`+0.00 MB`** (`0.00 MB/hr`)
- **Maximum Queue Backlog**: `1` (Queue bound: 1)
- **Ingestion**: Accelerated, not real-time-paced.
- **Verdict**: `PASSED: Bounded memory and backlog maintained across 600.0s soak.`

---

## 6. Actionable Checklist to Complete Local Evaluation

To complete scientific detection-performance evaluation:

1. **Acquire and verify the official checkpoint** from the pinned upstream source:
   ```bash
   mkdir -p models/weights
   curl -L -o models/weights/AASIST-L.pth \
     https://raw.githubusercontent.com/clovaai/aasist/a04c9863f63d44471dde8a6abcb3b082b07cd1d1/models/weights/AASIST-L.pth
   shasum -a 256 models/weights/AASIST-L.pth
   # Expected SHA-256: 814331d088032bb4c3fa61cc014789eadeed464209dd094ab3a2dd6ffbdce27a
   # Expected Size:    426428 bytes
   # Parameter Count:  85306
   ```
2. **Keep the official ASVspoof 2019 LA dataset outside Git.** Download it from
   [Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/3336) and pass
   its extracted LA directory explicitly.
3. **Run the fail-closed prerequisite check before inference**:
   ```bash
   .venv/bin/python scripts/reproduce_asvspoof_eval.py \
     --dataset-root /path/to/ASVspoof2019/LA \
     --split eval \
     --check-only
   ```
4. **Run all 71,237 eval trials only on a suitably provisioned host**. The
   interrupted 8 GB M3 batch-24 attempt is invalid and must not be resumed or
   reported as evidence:
   ```bash
   .venv/bin/python scripts/reproduce_asvspoof_eval.py \
     --dataset-root /path/to/ASVspoof2019/LA \
     --split eval \
     --batch-size 24 \
     --output-dir /path/outside-git/asvspoof2019-la
   ```
5. **Publish only complete generated artifacts.** Record the exact hashes and
   reproduced EER after a successful full run. Official min t-DCF remains NOT
   COMPUTED because this runner does not integrate its required ASV inputs.
