# VoiceShield

VoiceShield is an acoustic machine learning system designed to detect synthetic, cloned, or voice-converted speech from controlled audio streams (microphones, file uploads, WebRTC, SIP/VoIP).

> **Important Advisory Disclaimer**
> VoiceShield provides acoustic anomaly metrics and advisory warnings. Its outputs do **not** constitute identity proof and do **not** represent a fraud probability. VoiceShield warns and prompts independent verification; it does not automatically block calls. Browser clients access permitted local microphones only and cannot intercept arbitrary remote cellular calls.

---

## 1. System Requirements & Apple Silicon Prerequisites

- **Machine**: Apple Silicon Mac (M1/M2/M3/M4) running macOS 13+.
- **Package Manager**: `uv` (recommended for deterministic Python dependency management).
- **Python**: Python 3.11 (`brew install python@3.11`).
- **Native Tools**: `ffmpeg`, `libsndfile` (`brew install ffmpeg libsndfile`).
- **Node.js**: Node 20+ with `npm` (for the `apps/web` React dashboard).

---

## 2. Quickstart

### 2.1 Setup Virtual Environment and Dependencies
```bash
# Clone and enter directory
cd "deepfake voice"

# Run setup (creates .venv with Python 3.11 and installs all dependency groups)
make setup
```

Alternatively using `uv` directly:
```bash
uv venv --python /opt/homebrew/bin/python3.11 .venv
source .venv/bin/activate
uv sync --all-groups
```

### 2.2 Environment Configuration
Copy the template environment configuration:
```bash
cp .env.example .env
```

---

## 3. Running Checks & Tests

```bash
# Run linting
make lint

# Run static typing
make typecheck

# Run unit tests
make test-python

# Run full checks (lint + typecheck + python tests + frontend build)
make check-all
```

---

## 4. Running the Development Services

### 4.1 FastAPI Backend
```bash
make api
# Server starts at http://127.0.0.1:8000
# OpenAPI Docs: http://127.0.0.1:8000/docs
```

### 4.2 React + TypeScript + Vite Dashboard
```bash
make web
# Dashboard runs at http://localhost:5173
```

---

## 5. Hardware Acceleration (Apple Silicon MPS)

VoiceShield automatically queries `torch.backends.mps.is_available()`. If supported, inference executes on the Apple Silicon Neural Engine / GPU via MPS. If an unsupported operator is encountered or if `VOICESHIELD_DEVICE=cpu` is set in `.env`, the pipeline gracefully falls back to CPU execution without crashing or altering detector behavior.

---

## 6. Configuration & Reproducibility Flow

VoiceShield enforces strict configuration validation and complete provenance tracking:

### 6.1 Canonical Configuration
The canonical settings reside in `configs/default.yaml` and are parsed into strictly typed Pydantic models:
- **Audio contract**: Strictly 16,000 Hz, mono (1 channel), float32 normalized to $[-1.0, 1.0]$.
- **Windowing**: 4.0-second window (64,000 samples) with 1.0-second streaming hop (16,000 samples).
- **Speech gating**: $\ge 0.8\text{s}$ active speech duration required per window before inference.

```python
from voiceshield.config import load_config

# Load validated configuration with human-readable error reporting
config = load_config("configs/default.yaml")

print(f"Sample Rate: {config.audio.sample_rate} Hz")
print(f"Window Samples: {config.windowing.window_samples()} samples")
print(f"Config Hash: {config.compute_hash()}")
```

### 6.2 Deterministic Seeding
```python
from voiceshield.reproducibility import set_seed

# Seeds Python, NumPy, PyTorch (CPU & MPS) with deterministic algorithm flags
set_seed(42, deterministic=True)
```

### 6.3 Provenance & Run Records
All experiments and evaluations automatically generate a structured JSON run record saved to `reports/runs/<run_id>.json`:
```python
from voiceshield.reproducibility import create_run_record, save_run_record

record = create_run_record(
    config=config,
    metrics={"eer": 0.045, "min_dcf": 0.12},
    notes="Baseline evaluation on held-out validation split",
)
save_run_record(record)
```
The run record captures:
- Exact Git commit and working-tree dirty status;
- Deterministic SHA-256 config hash;
- Dataset manifest hash and model weights checkpoint hash;
- macOS kernel/platform, Python version, PyTorch version, and active acceleration device (`mps` / `cpu`).

---

## 7. Offline File Inference CLI

VoiceShield includes an offline file evaluation CLI (`voiceshield-infer`) for single files or entire audio directories:

```bash
# Evaluate a single file using mock model (or provide --checkpoint path/to/AASIST-L.pth)
uv run voiceshield-infer --input sample.wav --mock-model --hop-seconds 1.0

# Batch evaluate a directory and export structured CSV/JSONL records
uv run voiceshield-infer --input path/to/recordings/ --mock-model --output reports/eval.jsonl
```

Key features:
- **Canonical audio preprocessor**: Loads WAV and FLAC with sample rate normalization (16 kHz mono float32).
- **Deterministic temporal windowing**: 4.0-second sliding windows with configurable hop step (default: 1.0s).
- **Graceful degradation**: Reports unsupported files cleanly and continues past individual corrupt files without crashing.
- **Detailed provenance**: Captures per-window raw spoof/bonafide logits, uncalibrated synthetic acoustic anomaly scores (effective threshold and provenance source), active speech duration, quality diagnostics, model/checkpoint hashes, and compliance disclaimers.

---

## 8. Advisory REST API

The FastAPI service exposes an ephemeral file upload analysis route guarded by concurrency control optimized for Apple Silicon M3:

- **Health check**: `GET /health` returns system status, Apple Silicon MPS hardware diagnostics, and advisory disclaimers.
- **File analysis**: `POST /api/v1/analyze/file` accepts `multipart/form-data` audio uploads.
  - **Constraints**: 25 MB max file size, 300.0s max audio duration, WAV/FLAC formats only.
  - **Zero audio retention**: Reads audio directly into ephemeral memory buffers; no audio is written to disk.
  - **M3 concurrency guard**: Serialized via an `asyncio.Semaphore(1)` to prevent unified memory pressure or GPU contention.
  - **Lifespan preloading**: Loads detector model once at service startup rather than per request.

---

## 9. Scientific Evaluation Toolkit

VoiceShield provides an evaluation toolkit for benchmark analysis:

```bash
# Evaluate model predictions against a canonical manifest
uv run voiceshield-eval \
  --manifest data/manifests/eval.jsonl \
  --scores reports/runs/predictions.jsonl \
  --direction higher_is_spoof \
  --threshold 0.65 \
  --output-dir reports/evaluation/
```

Key capabilities:
- **Discrimination metrics**: Linear-interpolated Equal Error Rate (EER), ROC-AUC, PR-AUC, minDCF with configurable costs ($C_{\text{miss}}, C_{\text{fa}}$) and prior $P_{\text{target}}$, operating FPR at target TPR levels (90%, 95%, 99%).
- **Operating point metrics**: Confusion matrix (TP, FP, TN, FN), Precision, Recall, Specificity, F1 Score, and Balanced Accuracy.
- **Probabilistic calibration**: Brier score and Expected Calibration Error (ECE) evaluated on calibrated probabilities.
- **Clustered bootstrap inference**: 95% confidence intervals clustered by speaker ID or source utterance to prevent overconfident standard errors from overlapping windows.
- **Multi-slice reporting**: Breakdown by generator family, source dataset, language, codec/channel condition, replay condition, and duration bucket.
- **Cross-condition evaluation**: Matched, unseen-generator, unseen-channel, and external-test generalization tables.
- **Latency profiling**: Median (p50), 95th percentile (p95), mean latency, and Real-Time Factor (RTF).
- **Publication-quality artifacts**: Auto-generates Markdown summary tables, machine-readable `metrics.csv`, and figures (`roc_curve.png`, `score_distributions.png`, `calibration_curve.png`).
- **Scientific guardrails**: Refuses to tune decision thresholds on splits marked `test`; clearly labels single-class metrics as undefined without fabricating zero values.



