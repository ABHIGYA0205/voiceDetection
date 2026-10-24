# VoiceShield Evaluation Card: AASIST-L Baseline Detector

**Card Identifier**: `VoiceShield-EvalCard-AASIST-L-v0.1.0`  
**Associated Model Card**: [`aasist_l_model_card.md`](./aasist_l_model_card.md)
**Evaluation Standard**: ASVspoof 2019 Logical Access (LA) Protocol & VoiceShield M3 Benchmark Suite  
**Status**: **PARTIAL — ENGINEERING VALIDATED; SCIENTIFIC BENCHMARK PENDING**
**Card Version**: `1.0.0` (Truthful Baseline)  
**Date**: October 2026  

---

## 1. Executive Summary & Integrity Notice

> [!IMPORTANT]
> **Truthful Evaluation Policy**  
> In strict compliance with scientific integrity standards, VoiceShield forbids inventing evaluation metrics, claiming unverified benchmark numbers, or reporting results on cherry-picked audio files.  
> - **Upstream Clova AI Reference**: Published ICASSP 2022 metrics are recorded for architectural baseline reference.  
> - **Official checkpoint**: Acquisition and integrity verification of the official AASIST-L checkpoint are complete.
> - **Engineering validation**: The real-checkpoint API, transport, CPU/MPS benchmark, browser, and stress-soak validation are complete.
> - **Scientific evaluation data**: The ASVspoof 2019 LA evaluation data is absent and has not been evaluated locally. Local EER, min-tDCF, ROC-AUC, balanced accuracy, calibration, FPR/FNR, confidence intervals, and external/telephony performance remain not evaluated.
> - **Hardware profiling**: Real-checkpoint latency measurements are linked to the generated report artifact.

---

## 2. Evaluation Split Specification & Leakage Controls

### 2.1 ASVspoof 2019 Logical Access Protocol

| Split Identifier | Total Trials | Bona Fide Utterances | Spoofed Utterances | Speakers | Generators |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Train** | 25,380 | 2,580 | 22,800 | 20 (8 M, 12 F) | Known In-Set: A01–A06 |
| **Development** | 24,844 | 2,548 | 22,296 | 20 (8 M, 12 F) | Known In-Set: A01–A06 |
| **Evaluation (Target)** | **71,189** | **7,355** | **63,834** | **48 (21 M, 27 F)** | **Unseen Zero-Shot: A07–A19** |

### 2.2 Leakage Controls Enforced

```mermaid
flowchart TD
    Dataset[Audio Data Corpus] --> Splitter[Speaker-Disjoint Splitter]
    Splitter --> Train[Train Split: 20 Speakers]
    Splitter --> Dev[Dev Split: 20 Speakers]
    Splitter --> Eval[Eval Split: 48 Speakers]

    Eval -.->|Strict Isolation: 0 Overlapping Speakers| Train
    Eval -.->|Strict Isolation: 0 Overlapping Speakers| Dev
    Eval -.->|Unseen Holdout: A07-A19 Excluded from Train| Train
```

1. **Speaker Disjointness**: Zero speaker overlap across Train, Dev, and Eval partitions ($S_{\text{train}} \cap S_{\text{dev}} \cap S_{\text{eval}} = \emptyset$).
2. **Generator Holdout**: Evaluation partition contains 13 unseen synthetic speech algorithms (A07–A19), spanning waveform concatenation, neural vocoding, transfer learning, and modern neural acoustic models.
3. **Source Utterance Deduplication**: Source text prompts are partitioned to prevent phonetic leakage across partitions.

---

## 3. Reported Metric Provenance & Status

### 3.1 Benchmark Comparison Table

| Metric | Upstream Published (Clova AI / ICASSP 2022) | VoiceShield Local Evaluation | Status | Report Artefact & Evidence |
| :--- | :---: | :---: | :---: | :--- |
| **Equal Error Rate (EER)** | **0.99%** (0.0099) | **Not yet evaluated** | Unverified locally | Requires local ASVspoof 2019 LA partition. |
| **min-tDCF** ($C_{\text{miss}}=1, C_{\text{fa}}=10, P=0.05$) | **0.0346** | **Not yet evaluated** | Unverified locally | Requires local ASVspoof 2019 LA partition. |
| **Balanced Accuracy** | Upstream: ~98.8% | **Not yet evaluated** | Unverified locally | Local run pending dataset acquisition. |
| **ROC-AUC** | Upstream: ~0.999 | **Not yet evaluated** | Unverified locally | Local run pending dataset acquisition. |
| **Brier Score** | Not reported upstream | **Not yet evaluated** | Uncalibrated | Pending dev-set probability calibration. |
| **Expected Calibration Error (ECE)** | Not reported upstream | **Not yet evaluated** | Uncalibrated | Pending dev-set probability calibration. |

---

## 4. Fine-Grained Subgroup Performance Breakdowns

### 4.1 By Speech Generator Family

| Generator ID | Generator Family / Synthesis Type | Upstream Baseline EER | VoiceShield Local Status |
| :--- | :--- | :---: | :--- |
| **A01** | Neural TTS (WaveNet vocoder) | ~0.00% | **Not yet evaluated** |
| **A02** | Neural TTS (World vocoder) | ~0.00% | **Not yet evaluated** |
| **A03** | Neural TTS (Linear vocoder) | ~0.00% | **Not yet evaluated** |
| **A04** | Waveform Concatenation | ~0.02% | **Not yet evaluated** |
| **A05** | Voice Conversion (VCC2018) | ~0.01% | **Not yet evaluated** |
| **A06** | Voice Conversion (VCC2020) | ~0.01% | **Not yet evaluated** |
| **A07–A19 (Unseen)** | Unknown Zero-Shot Generative Models | 0.99% (Aggregate) | **Not yet evaluated** |
| **Modern Diffusion (F5-TTS, CosyVoice)** | Flow-Matching / Diffusion Transformers | Not in ASVspoof 2019 | **Not yet evaluated** |

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

1. **Acquire Official Checkpoint**:
   ```bash
   mkdir -p models/weights
   curl -L -o models/weights/AASIST-L.pth \
     https://raw.githubusercontent.com/clovaai/aasist/a04c9863f63d44471dde8a6abcb3b082b07cd1d1/models/weights/AASIST-L.pth
   shasum -a 256 models/weights/AASIST-L.pth
   # Expected SHA-256: 814331d088032bb4c3fa61cc014789eadeed464209dd094ab3a2dd6ffbdce27a
   # Expected Size:    426428 bytes
   # Parameter Count:  85306
   ```
2. **Download ASVspoof 2019 LA Evaluation Dataset**:
   Download the official dataset archive from [Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/3336) and extract to `data/ASVspoof2019/LA/`.
3. **Execute Smoke Verification**:
   ```bash
   .venv/bin/python scripts/reproduce_asvspoof_eval.py --smoke --max-samples 100
   ```
4. **Run Full 71,189-Trial Evaluation**:
   ```bash
   .venv/bin/python scripts/reproduce_asvspoof_eval.py --split eval --output-dir reports/runs/asvspoof_eval_official
   ```
5. **Publish Scientific Evaluation Results**:
   Update Section 3 with the exact generated SHA-256 hashes, EER, and min-tDCF from `reports/runs/asvspoof_eval_official/asvspoof_reproduction_report.md`.
