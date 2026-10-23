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

## 8. Advisory REST API & Streaming Service

The FastAPI service exposes real-time streaming WebSocket and ephemeral file upload analysis routes guarded by atomic admission controls and resource ceilings:

- **Health check**: `GET /health` returns system status, Apple Silicon MPS hardware diagnostics, and advisory disclaimers. `GET /ready` verifies model readiness.
- **Atomic WebSocket Streaming**: `GET /api/v1/stream` establishes real-time detection sessions.
  - **Atomic Session Admission (Audit M-03)**: Reserved atomically via `AtomicAdmissionController` *before* `websocket.accept()` or handshake wait. Handshaking connections count against the ceiling (`VOICESHIELD_MAX_CONCURRENT_SESSIONS`, default: 10). Excess connections receive `CONCURRENCY_LIMIT_REACHED` (code 1008).
  - **Metadata Ceilings**: Client handshake metadata can only *tighten* (lower) server ceilings; metadata cannot raise server limits (default 128 KB frames, 1-hour duration, 30.0s idle timeout).
- **File Upload Analysis**: `POST /api/v1/analyze/file` accepts `multipart/form-data` audio uploads.
  - **Pre-Parsing Admission (Audit M-03)**: Governed by ASGI middleware *before* body consumption and multipart parsing (`VOICESHIELD_MAX_CONCURRENT_FILE_UPLOADS`, default: 2). Saturated requests promptly receive HTTP 429 (`FILE_UPLOAD_CONCURRENCY_LIMIT_REACHED`).
  - **Two-Layer Request Ceiling**: Total request body capped at 26 MB (25 MB audio + 1 MB multipart overhead). Enforced via `Content-Length` precheck and streaming receive chunk counter before endpoint routing. Audio files exceeding 25 MB or 300.0s duration are rejected.
  - **Ephemeral Audio Lifecycle**: Uploads under 1 MB are processed in RAM; uploads over 1 MB use Starlette's `SpooledTemporaryFile` which automatically cleans up the OS temporary file upon request completion. No uploaded audio is persistently stored.
  - **M3 concurrency guard**: Physical model evaluation across all routes is strictly serialized through a single process-wide `InferenceWorker` to prevent unified memory thrashing or GPU contention.
- **Deployment Hardening**: Application resource limits provide defense-in-depth but do not replace TLS, authentication, host hardening, or reverse proxies. Production deployments should use a reverse proxy (e.g. Nginx) with matching body limits (`client_max_body_size 26m`), connection rate limits, and proxy timeouts ($\le 30$s).

---

## 9. Scientific Evaluation Toolkit

VoiceShield provides an evaluation toolkit with scientific guardrails for benchmark analysis:

```bash
# Evaluate model predictions against a canonical manifest
uv run voiceshield-eval \
  --manifest data/manifests/eval.jsonl \
  --scores reports/runs/predictions.jsonl \
  --direction higher_is_spoof \
  --threshold 0.65 \
  --output-dir reports/evaluation/

# Allow explicit partial-score evaluation when some scores are missing
uv run voiceshield-eval \
  --manifest data/manifests/eval.jsonl \
  --scores reports/runs/partial_predictions.jsonl \
  --allow-partial-scores \
  --output-dir reports/evaluation/
```

Key capabilities and scientific guardrails:
- **Connected-Component Partitioning**: Dataset splits are atomic connected components across declared `speaker_id`, `source_utterance_id`, and declared exact `sha256` checksum links. Holdout constraints operate on whole components. Split ratios currently apply to atomic component counts rather than exact record counts because components cannot be divided. Declared SHA-256 equality is used for component grouping and duplicate detection; the code does not yet verify audio files against declared checksums and does not perform acoustic near-duplicate detection.
- **Strict Score Ingestion**: Preserves exact `0.0` detection scores without Boolean fallback; strictly rejects duplicate IDs, empty values, Boolean types, non-numeric values, and NaN/Inf scores.
- **Complete Score Coverage by Default**: Default evaluation requires 100% complete score coverage across all unique manifest sample IDs. Unknown score IDs are refused. Partial evaluation requires explicit opt-in (`--allow-partial-scores`) and is prominently flagged in reports.
- **Threshold-Selection Isolation**: Decision thresholds are selected on `dev` split only (or on `train` split only when no dev split is present); test splits and unassigned mixtures are never used for threshold optimization.
- **Fail-Closed Calibration**: Raw/softmax scores are uncalibrated and do not produce Brier scores, Expected Calibration Error (ECE), or reliability diagrams without an identified, fitted calibrator artifact. Calibration metrics and reliability plots are unavailable through the high-level evaluator until an identified calibrator artifact loader is implemented.
- **Discrimination Metrics**: Linear-interpolated Equal Error Rate (EER), ROC-AUC, PR-AUC, minDCF with configurable costs ($C_{\text{miss}}, C_{\text{fa}}$) and prior $P_{\text{target}}$, operating FPR at target TPR levels (90%, 95%, 99%).
- **Operating Point Metrics**: Confusion matrix (TP, FP, TN, FN), Precision, Recall, Specificity, F1 Score, and Balanced Accuracy.
- **Clustered Bootstrap Inference**: 95% confidence intervals clustered by speaker ID or source utterance to prevent overconfident standard errors from repeated speakers.
- **Multi-Slice Reporting**: Breakdown by generator family, source dataset, language, codec/channel condition, replay condition, and duration bucket.
- **Cross-Condition Evaluation**: Matched, unseen-generator, unseen-channel, and external-test generalization tables.
- **Latency Profiling**: Median (p50), 95th percentile (p95), mean latency, and Real-Time Factor (RTF).
- **Publication-Quality Artifacts**: Auto-generates Markdown summary tables, machine-readable `metrics.csv`, and figures (`roc_curve.png`, `score_distributions.png`).



