# VoiceShield Benchmark Report: Apple MacBook Pro M3

**Execution Timestamp (UTC)**: `2026-10-24T19:18:20.096907+00:00`  
**Hardware**: `Apple M3` (`Mac15,3`)  
**macOS Version**: `27.0.1` | **CPU Cores**: `8`  
**Python**: `3.11.15` | **PyTorch**: `2.14.1`  
**Apple MPS Metal Acceleration**: `Built=True, Available=True`  

> [!IMPORTANT]
> **Engineering Target**: Project Target: Under 200 ms inference per 4-second window (RTF < 0.050). Defined as an engineering target, not an achieved claim until measured.

## 1. Provenance & Hash Verification

| Provenance Field | Value |
| :--- | :--- |
| **Model Identifier** | `clovaai/aasist-l` |
| **Provenance Status** | `official_verified` |
| **Checkpoint Verified** | `True` |
| **Effective Canonical Config Hash** | `774483ef6871c5a0c83f81098cf50cc2e691d39305c9e7c262d478bfb5088596` |
| **Checkpoint Path** | `models/weights/AASIST-L.pth` |
| **Checkpoint SHA-256** | `814331d088032bb4c3fa61cc014789eadeed464209dd094ab3a2dd6ffbdce27a` |
| **Expected Registry SHA-256** | `814331d088032bb4c3fa61cc014789eadeed464209dd094ab3a2dd6ffbdce27a` |
| **Upstream Commit** | `a04c9863f63d44471dde8a6abcb3b082b07cd1d1` |
| **Parameter Count** | `85306` |
| **Git Commit** | `392ca34eaf76eecf4afd8d23ca5049f39eec08f5` (Dirty: `False`) |
| **Weights Status** | `REAL OFFICIAL AASIST-L` |
| **Configured Device Preference** | `mps` |
| **Configured Decision Threshold** | `0.65` |
| **Evaluated Devices Matrix** | `cpu, mps` (suite parameter, not in config) |
| **Benchmark Iterations** | `100` (warmup: `10`) |
| **Streaming Soak Duration** | `600.0s` |

## 2. Pipeline Stage Latency Breakdown (Batch Size 1)

All metrics measure steady-state execution over 4.0-second audio windows, with initial warm-up recorded separately.

| Pipeline Stage | Device | Warm-up Mean | Steady p50 (Median) | Steady p95 | Steady p99 | Throughput | RTF | Target (<200ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1_audio_decoding_and_resampling` | `CPU` | 18.1 ms | **6.3 ms** | 8.7 ms | 14.7 ms | 139.9 win/s | 0.0016 | N/A |
| `2_feature_extraction_and_padding` | `CPU` | 0.1 ms | **0.1 ms** | 0.1 ms | 0.1 ms | 19972.4 win/s | 0.0000 | N/A |
| `3_model_inference` | `CPU` | 164.3 ms | **113.0 ms** | 129.8 ms | 181.4 ms | 8.4 win/s | 0.0283 | ✅ MET |
| `4_score_smoothing_and_temporal_policy` | `CPU` | 0.0 ms | **0.0 ms** | 0.0 ms | 0.0 ms | 263941.4 win/s | 0.0000 | N/A |
| `5_end_to_end_window_processing` | `CPU` | 112.7 ms | **112.0 ms** | 117.6 ms | 125.4 ms | 8.8 win/s | 0.0280 | ✅ MET |
| `3_model_inference` | `MPS` | 70.6 ms | **30.2 ms** | 30.8 ms | 34.0 ms | 32.9 win/s | 0.0076 | ✅ MET |
| `5_end_to_end_window_processing` | `MPS` | 34.4 ms | **33.8 ms** | 34.5 ms | 37.1 ms | 29.4 win/s | 0.0084 | ✅ MET |

## 3. Hardware Acceleration Comparison: Apple Silicon MPS vs. CPU

| Workload | CPU p50 | MPS p50 | Metal Speedup |
| :--- | :---: | :---: | :---: |
| `model_inference` | 113.0 ms | 30.2 ms | **3.74x faster** |
| `end_to_end` | 112.0 ms | 33.8 ms | **3.32x faster** |

## 4. Profiling & Bottleneck Analysis

- **Primary Execution Device**: `mps`
- **End-to-End Window Latency (p50)**: `33.754 ms`
- **Real-Time Factor (RTF)**: `0.028` (CPU) / `0.00844` (MPS)

### Stage Proportion Breakdown:
- **`audio_decoding_pct`**: 18.5%
- **`feature_extraction_pct`**: 0.2%
- **`model_inference_pct`**: 89.4%
- **`score_smoothing_and_policy_pct`**: 0.0%

### Project Target Assessment (< 200 ms per 4-second window):
- **Model Inference**: ✅ Target Achieved
- **End-to-End Processing**: ✅ Target Achieved

## 5. Streaming Stress Soak Diagnostics

- **Ingestion Mode**: `accelerated` (frames are ingested without real-time pacing and may represent more audio than elapsed wall time)
- **Configured Duration**: `600.0s`
- **Measured Wall-Clock Duration**: `600.0s` (10.0 min)
- **Frame Duration**: `50.0 ms`
- **Total Frames Ingested**: `367780`
- **Total Windows Evaluated**: `18386`
- **Initial RSS Memory**: `986.81 MB`
- **Peak RSS Memory**: `986.81 MB`
- **Final RSS Memory**: `986.81 MB`
- **Net Memory Delta**: `+0.00 MB`
- **Extrapolated Growth Rate**: `0.00 MB/hr`
- **Maximum Backlog Observed**: `1` (Queue limit: 1)
- **Verdict**: `PASSED: Bounded memory and backlog maintained across 600.0s soak. Net memory delta: +0.00 MB (0.00 MB/hr). Queue backlog remained <= 1.`
