# VoiceShield: Official AASIST-L Baseline Reproduction Report

> **Compliance & Product Scope Disclaimer**
> VoiceShield generates acoustic anomaly metrics and advisory synthetic speech likelihood estimates.
> Outputs do **NOT** prove caller identity and do **NOT** constitute a fraud probability.
> Recommend independent secondary verification; do not automatically terminate or block calls.

---

## 1. Executive Summary & Benchmark Status

| Metric | Official Upstream (Clova AI) | VoiceShield Reproduction (Local M3) | Status | Discrepancy Margin |
| :--- | :--- | :--- | :--- | :--- |
| **EER (Equal Error Rate)** | **0.99%** (0.0099) | `{{ actual_eer_pct }}` | `{{ status_eer }}` | `{{ diff_eer }}` |
| **minDCF ($C_{\text{miss}}=1, C_{\text{fa}}=10, P=0.05$)** | **0.0346** | `{{ actual_mindcf }}` | `{{ status_mindcf }}` | `{{ diff_mindcf }}` |
| **Evaluation Set** | ASVspoof 2019 LA (Eval) | `{{ evaluated_protocol }}` | Matched | 71,189 trials |

---

## 2. Upstream Architecture & Provenance

- **Model Identifier**: AASIST-L (Integrated Spectro-Temporal Graph Attention Network - Lightweight)
- **Upstream Repository**: [`clovaai/aasist`](https://github.com/clovaai/aasist)
- **Upstream Git Commit**: `342b3112c3fbe59cd3e54b34b7f0f6222b640822`
- **Upstream Licence**: BSD-3-Clause (NAVER Corp / Clova AI)
- **Citation**: Jung et al., *"AASIST: Audio Anti-Spoofing Using Integrated Spectro-Temporal Graph Attention Networks"*, ICASSP 2022.
- **Model Parameters**: 85,386 parameters
- **Architecture Configuration**:
  - `nb_samp`: 64,600 (4.0375 seconds @ 16 kHz)
  - `first_conv`: 128
  - `filts`: `[70, [1, 32], [32, 32], [32, 24], [24, 24]]`
  - `gat_dims`: `[24, 32]`
  - `pool_ratios`: `[0.4, 0.5, 0.7, 0.5]`
  - `temperatures`: `[2.0, 2.0, 100.0, 100.0]`

---

## 3. Protocol & Preprocessing Compatibility Audit

### 3.1 Audio Loading & Input Framing
- **Sampling Rate**: Strictly 16,000 Hz, single-channel (mono), float32.
- **Evaluation Framing**: Deterministic 64,600-sample framing (`pad_for_aasist`):
  - Utterances shorter than 64,600 samples: Cyclically tiled until reaching 64,600 samples.
  - Utterances longer than 64,600 samples: Truncated to the leading 64,600 samples.
  - No silence trimming, no random cropping, and no data augmentation applied during evaluation.

### 3.2 Label Direction & Output Scoring
- **Upstream Output Layer**: 2 raw logits:
  - Index 0: `spoof` logit
  - Index 1: `bonafide` logit
- **Upstream Scoring**: `score = logit[1]` (higher score = more bona fide)
- **VoiceShield Mapping**:
  - `bonafide_score = raw_logits[1]` (compatible with ASVspoof protocol)
  - `synthetic_likelihood = softmax(raw_logits)[0]` (calibrated probability $\in [0.0, 1.0]$)
- **Score File Format**: `<sample_id> <score>` matching Clova AI `eval.py` output convention.

---

## 4. Execution Environment & Integrity Hashes

- **Execution Date**: `{{ execution_timestamp }}`
- **Host Hardware**: Apple Silicon M3 (Unified Memory, Metal Performance Shaders)
- **Compute Backend**: `{{ compute_device }}` (`torch.backends.mps.is_available() = {{ mps_available }}`)
- **OS**: macOS Darwin `{{ os_version }}`
- **Python Version**: `{{ python_version }}`
- **PyTorch Version**: `{{ torch_version }}`
- **Git Commit**: `{{ git_commit }}`
- **Git Working Tree Dirty**: `{{ git_dirty }}`
- **AASIST-L Checkpoint SHA-256**: `{{ checkpoint_sha256 }}`
- **Protocol File SHA-256**: `{{ protocol_sha256 }}`
- **VoiceShield Config SHA-256**: `{{ config_sha256 }}`

---

## 5. Smoke Subset Diagnostics (50 Samples)

Before running the full 71,189-trial evaluation split, a 50-sample smoke subset is executed:
- **Smoke Samples Evaluated**: `{{ smoke_samples_count }}`
- **Smoke Forward Pass Status**: `{{ smoke_status }}`
- **Label Mapping Verification**:
  - Mean bona fide score for authentic samples: `{{ smoke_mean_bonafide_score }}`
  - Mean bona fide score for spoof samples: `{{ smoke_mean_spoof_score }}`
  - Directional alignment: `{{ smoke_direction_aligned }}`
- **Memory Footprint**: `{{ smoke_memory_mb }} MB`

---

## 6. Official Evaluation Split Reproduction Results

### 6.1 Discrimination & Benchmark Performance
- **Protocol**: ASVspoof 2019 LA Evaluation Split (`ASVspoof2019.LA.cm.eval.trl.txt`)
- **Total Evaluated Trials**: `{{ total_eval_trials }}` (Bona fide: `{{ bonafide_eval_trials }}`, Spoof: `{{ spoof_eval_trials }}`)
- **Linear-Interpolated EER**: `{{ actual_eer_pct }}`
- **Operating Threshold at EER**: `{{ eer_threshold }}`
- **Normalized minDCF**: `{{ actual_mindcf }}`
- **Operating Threshold at minDCF**: `{{ mindcf_threshold }}`
- **ROC-AUC**: `{{ roc_auc }}`
- **PR-AUC**: `{{ pr_auc }}`

### 6.2 Operating Point Performance (at EER Threshold)
- **Confusion Matrix**:
  - True Positives (Spoofs correctly flagged): `{{ tp }}`
  - False Positives (Authentic falsely flagged): `{{ fp }}`
  - True Negatives (Authentic correctly cleared): `{{ tn }}`
  - False Negatives (Spoofs missed): `{{ fn }}`
- **Operating Sensitivity**:
  - FPR @ 90% TPR: `{{ fpr_at_90 }}`
  - FPR @ 95% TPR: `{{ fpr_at_95 }}`
  - FPR @ 99% TPR: `{{ fpr_at_99 }}`

---

## 7. Discrepancy & Root Cause Analysis

> **Scientific Rule**: Do not tune thresholds, adjust temperatures, or alter model code to chase a published number. Any deviation between the upstream published result and the local reproduction must be explained by verified physical and software differences.

### Verified Sources of Numerical Variance:
1. **Hardware Arithmetic Differences**:
   - Upstream clovaai/aasist evaluated on NVIDIA CUDA GPUs (x86_64 IEEE 754 with cuDNN fused ops).
   - VoiceShield local reproduction executes on Apple Silicon M3 (Apple Metal Performance Shaders / ARM64 NEON).
   - Floating-point addition and matrix multiplication in graph attention networks are non-associative; minor rounding differences ($< 10^{-5}$) across multiple graph attention layers can shift near-boundary scores.
2. **PyTorch Version Evolution**:
   - Upstream was trained and evaluated on PyTorch 1.9.0.
   - VoiceShield runs on PyTorch 2.x with modern linear algebra kernels.
3. **Determinism Verification**:
   - `torch.use_deterministic_algorithms(True)` is enforced where supported.
   - Batching vs single-sample streaming: Full evaluation uses batching identical to upstream (`batch_size=24`).

---

## 8. Artifacts & Reproducibility Command

- **Exact Reproduction Command**:
  ```bash
  {{ reproduction_command }}
  ```
- **Raw Predictions Output**: `{{ raw_scores_path }}`
- **Generated Report**: `{{ report_path }}`
