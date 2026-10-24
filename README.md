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
- **Node.js**: Node 22.12+ with `npm` (for the `apps/web` React dashboard; `.nvmrc` selects Node 22).

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
uv sync --frozen --all-groups
cd apps/web && npm ci
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

The complete non-mutating local gate is `make check-all`. It runs frozen Python
sync checks, Ruff lint/format checks, strict MyPy, the full Python coverage gate,
`npm ci`, the production build, Vitest, both mocked and live Playwright suites,
dependency vulnerability audits, resolved Python/npm license checks, and package
metadata checks.

### 3.1 License and supply-chain boundaries

Original VoiceShield code is Apache-2.0. Vendored AASIST source is MIT and
retains its upstream notices. The AASIST-L checkpoint and referenced datasets
are separate terms; Apache-2.0 does not grant checkpoint redistribution rights.
Python and npm dependencies retain their own licenses. CI validates these
boundaries and does not claim legal clearance, detector accuracy, scientific
validity, production security, or Safari compatibility.

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

## 6. Configuration & Runtime Wiring (Audit H-09)

VoiceShield enforces strict canonical configuration wiring and complete algorithmic provenance:

### 6.1 Canonical Algorithmic Configuration
`configs/default.yaml` is the canonical algorithmic configuration governing model detector construction, real-time streaming sessions, and offline file evaluation:
- **Model Identity**: Strictly `clovaai/aasist-l` (canonical default) or `custom/aasist-l`. Mock runtime identity is `mock/aasist-l`.
- **Architecture**: Strictly `aasist_l` (`clovaai/aasist` baseline, 85,306 parameters). Other architectures are rejected.
- **Official Checkpoint Integrity**: Official mode (`clovaai/aasist-l`) unconditionally verifies SHA-256 (`814331d088032bb4c3fa61cc014789eadeed464209dd094ab3a2dd6ffbdce27a`), file size (426,428 bytes), and parameter count (85,306) against the immutable model registry prior to `torch.load` and fails closed on mismatch.
- **Custom Mode**: Custom checkpoints require explicit `model_id: "custom/aasist-l"` and report provenance status `custom_unverified`.
- **Audio contract**: Strictly 16,000 Hz, mono (1 channel), float32 normalized to $[-1.0, 1.0]$.
- **Windowing**: 4.0-second window (64,000 samples) with 1.0-second streaming hop (16,000 samples).
- **Speech gating**: $\ge 0.8\text{s}$ active speech duration required per window before inference.
- **Temporal Policy**: High threshold ($0.65$), low threshold ($0.35$), EWMA smoothing ($\alpha = 0.3$), 3-of-5 elevation, 3-drop hysteresis cooldown, and 0.25s timing gap threshold.
- **Scores remain uncalibrated**: Softmax probabilities are raw acoustic anomaly heuristics; score calibration is not implemented.
- **Removed dead schemas**: Unused training and calibration configuration sections were removed from the schema because those implementations do not exist. Dead model fields (`num_classes`, `embedding_dim`) and dead temporal policy fields (`consecutive_hops_required`, `decay_factor`, `alert_threshold`, `explanation_rules`) were removed.
- **Deployment settings outside hash**: Operational settings (such as bind host/port, CORS origins, and admission limits) do not alter inference output and are configured via environment variables outside the algorithmic configuration hash.

### 6.2 Strict Schema Validation
All configuration models enforce `extra="forbid"`. Any unknown root or nested key fails startup or CLI invocation with a clear `ConfigError` detailing the exact field path. Non-finite numeric values (`NaN`, `±Inf`) are strictly rejected.

```python
from voiceshield.config import VoiceShieldConfig, load_config

# Load canonical configuration (fails fast on missing or invalid schema)
config = load_config("configs/default.yaml")

print(f"Model ID: {config.model.model_id}")
print(f"Architecture: {config.model.architecture}")
print(f"Config Hash: {config.compute_hash()}")

# Parity guarantee: VoiceShieldConfig() and configs/default.yaml yield identical defaults and SHA-256 hash
assert config.compute_hash() == VoiceShieldConfig().compute_hash()
```

### 6.3 Configuration Path Selection & Precedence
VoiceShield establishes one validated effective configuration for each API process or CLI invocation.

**Precedence Order**:
1. **Canonical Default Config**: Loaded from repository/package `configs/default.yaml` when no path is specified.
2. **Explicit Custom Config Path**:
   - API: `VOICESHIELD_CONFIG_PATH=/path/to/custom.yaml`
   - CLI: `--config /path/to/custom.yaml`
3. **Explicit Operational Overrides**:
   - Environment variables: `VOICESHIELD_MODEL_ID`, `VOICESHIELD_CHECKPOINT_PATH`, `VOICESHIELD_DEVICE`.
   - CLI flags: `--model-id`, `--checkpoint`, `--device`, `--hop-seconds`, `--threshold`.
   - REST API query parameters: `hop_seconds`, `threshold` (omission preserves server configuration).

**Deterministic Hash Computation**:
Whenever an explicit override is provided, VoiceShield validates and instantiates a new effective configuration, and computes the SHA-256 hash *after* the overrides are applied. The pre-override hash is never reported.

### 6.4 Deterministic Seeding
```python
from voiceshield.reproducibility import set_seed

# Seeds Python, NumPy, PyTorch (CPU & MPS) with deterministic algorithm flags
set_seed(42, deterministic=True)
```

### 6.5 Provenance & Run Records
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
- Deterministic SHA-256 effective config hash;
- Dataset manifest hash and model weights checkpoint hash;
- macOS kernel/platform, Python version, PyTorch version, and active acceleration device (`mps` / `cpu`).

---

## 7. Offline File Inference CLI

VoiceShield includes an offline file evaluation CLI (`voiceshield-infer`) for single files or entire audio directories:

```bash
# Evaluate a single file using mock model (uses canonical config defaults when flags are omitted)
uv run voiceshield-infer --input sample.wav --mock-model

# Override decision threshold and hop size explicitly (updates effective hash)
uv run voiceshield-infer --input sample.wav --mock-model --threshold 0.80 --hop-seconds 0.5

# Batch evaluate a directory and export structured CSV/JSONL records
uv run voiceshield-infer --input path/to/recordings/ --mock-model --output reports/eval.jsonl
```

Key features:
- **Canonical audio preprocessor**: Loads WAV and FLAC with sample rate normalization (16 kHz mono float32).
- **Deterministic temporal windowing**: 4.0-second sliding windows with configurable hop step (default: 1.0s).
- **Effective Configuration Provenance**: Outputs effective config hash in CLI summary and structured results (`config_hash` field).
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


