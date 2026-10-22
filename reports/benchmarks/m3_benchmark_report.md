# VoiceShield Benchmark Report: Apple MacBook Pro M3

**Execution Timestamp (UTC)**: `2026-10-22T22:15:46.465181+00:00`  
**Hardware**: `Apple M3` (`Mac15,3`)  
**macOS Version**: `27.0.1` | **CPU Cores**: `8`  
**Python**: `3.11.15` | **PyTorch**: `2.14.1`  
**Apple MPS Metal Acceleration**: `Built=True, Available=True`  

> [!IMPORTANT]
> **Engineering Target**: Project Target: Under 200 ms inference per 4-second window (RTF < 0.050). Defined as an engineering target, not an achieved claim until measured.

> [!WARNING]
> **MOCK WEIGHTS NOTICE**: [MOCK DETECTOR BENCHMARK - NOT FOR PITCH OR PRODUCTION CLAIMS] Official AASIST-L.pth checkpoint was not present. Latencies reflect architectural forward pass computation of the AASIST network, but must NOT be cited as verified detection claims.

## 1. Provenance & Hash Verification

| Provenance Field | Value |
| :--- | :--- |
| **Model Architecture** | `AASIST-L` |
| **Config Hash (SHA-256)** | `67427d513bead1064a39eddd4ab2a729e4eb41b13c763a84c9fec55ff59d2da9` |
| **Checkpoint Path** | `MOCK_WEIGHTS (NONE)` |
| **Checkpoint SHA-256** | `MOCK_WEIGHTS (NONE)` |
| **Git Commit** | `Uncommitted repo` (Dirty: `True`) |
| **Weights Status** | `MOCK WEIGHTS` |

## 2. Pipeline Stage Latency Breakdown (Batch Size 1)

All metrics measure steady-state execution over 4.0-second audio windows, with initial warm-up recorded separately.

| Pipeline Stage | Device | Warm-up Mean | Steady p50 (Median) | Steady p95 | Steady p99 | Throughput | RTF | Target (<200ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1_audio_decoding_and_resampling` | `CPU` | 6.0 ms | **5.7 ms** | 8.6 ms | 38.4 ms | 140.6 win/s | 0.0014 | N/A |
| `2_feature_extraction_and_padding` | `CPU` | 0.1 ms | **0.0 ms** | 0.1 ms | 0.1 ms | 20547.9 win/s | 0.0000 | N/A |
| `3_model_inference` | `CPU` | 139.3 ms | **130.1 ms** | 132.0 ms | 147.3 ms | 7.6 win/s | 0.0325 | ✅ MET |
| `4_calibration_and_temporal_policy` | `CPU` | 0.0 ms | **0.0 ms** | 0.0 ms | 0.0 ms | 257295.6 win/s | 0.0000 | N/A |
| `5_end_to_end_window_processing` | `CPU` | 131.8 ms | **132.3 ms** | 135.1 ms | 178.5 ms | 7.5 win/s | 0.0331 | ✅ MET |
| `3_model_inference` | `MPS` | 893.6 ms | **30.4 ms** | 31.3 ms | 31.8 ms | 32.8 win/s | 0.0076 | ✅ MET |
| `5_end_to_end_window_processing` | `MPS` | 34.0 ms | **34.7 ms** | 39.4 ms | 39.9 ms | 28.1 win/s | 0.0087 | ✅ MET |

## 3. Hardware Acceleration Comparison: Apple Silicon MPS vs. CPU

| Workload | CPU p50 | MPS p50 | Metal Speedup |
| :--- | :---: | :---: | :---: |
| `model_inference` | 130.1 ms | 30.4 ms | **4.29x faster** |
| `end_to_end` | 132.3 ms | 34.7 ms | **3.81x faster** |

## 4. Profiling & Bottleneck Analysis

- **Primary Execution Device**: `mps`
- **End-to-End Window Latency (p50)**: `34.735 ms`
- **Real-Time Factor (RTF)**: `0.03306` (CPU) / `0.00868` (MPS)

### Stage Proportion Breakdown:
- **`audio_decoding_pct`**: 16.5%
- **`feature_extraction_pct`**: 0.1%
- **`model_inference_pct`**: 87.4%
- **`calibration_and_policy_pct`**: 0.0%

### Project Target Assessment (< 200 ms per 4-second window):
- **Model Inference**: ✅ Target Achieved
- **End-to-End Processing**: ✅ Target Achieved

## 5. 10-Minute Streaming Soak Test Diagnostics

- **Duration**: `600.0s` (10.0 min)
- **Total Frames Ingested**: `333480`
- **Total Windows Evaluated**: `16671`
- **Initial RSS Memory**: `1212.67 MB`
- **Peak RSS Memory**: `1212.67 MB`
- **Final RSS Memory**: `1212.67 MB`
- **Net Memory Delta**: `+0.00 MB`
- **Extrapolated Growth Rate**: `0.00 MB/hr`
- **Maximum Backlog Observed**: `1` (Queue limit: 1)
- **Verdict**: `PASSED: Bounded memory and backlog maintained across 600.0s soak. Net memory delta: +0.00 MB (0.00 MB/hr). Queue backlog remained <= 1.`
