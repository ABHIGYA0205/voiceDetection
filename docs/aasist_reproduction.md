# AASIST-L Baseline Integration & Reproduction Checklist

## 1. Upstream Provenance & Licensing

- **Upstream Repository**: [`https://github.com/clovaai/aasist`](https://github.com/clovaai/aasist)
- **Pinned Commit**: `a04c9863f63d44471dde8a6abcb3b082b07cd1d1` (Merge PR #4 from clovaai/main)
- **Upstream Authors**: Jee-weon Jung, Hee-Soo Heo, Hye-jin Shim, Ha-Jin Yu (NAVER Clova AI / University of Seoul)
- **Publication**: *"AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks"*, ICASSP 2022.
- **Source-code licence**: **MIT License** (Copyright (c) 2021-present NAVER Corp.).
  - Vendored source under `src/voiceshield/vendor/aasist/` retains the upstream copyright and licence.
- **Checkpoint rights**: Separate from the source-code licence and not independently established by VoiceShield. Checkpoint bytes are obtained from the pinned upstream source and are not redistributed in VoiceShield release artifacts.

---

## 2. Architecture Specification (AASIST-L)

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Model Variant** | `AASIST-L` | Lightweight variant (~85K parameters) |
| **Sampling Rate** | $16,000\text{ Hz}$ | Strictly 16 kHz mono float32 |
| **Input Frame Length (`nb_samp`)** | $64,600\text{ samples}$ | Equivalent to $4.0375\text{ seconds}$ at 16 kHz |
| **First SincConv Kernel** | 128 taps | SincNet parameterized Mel-bandpass filterbank |
| **Residual Filters (`filts`)** | `[70, [1, 32], [32, 32], [32, 24], [24, 24]]` | Narrow residual encoder blocks |
| **GAT Dimensions (`gat_dims`)** | `[24, 32]` | Spectro-temporal graph attention dimensions |
| **Graph Pooling Ratios (`pool_ratios`)** | `[0.4, 0.5, 0.7, 0.5]` | Hierarchical top-$k$ node pooling fractions |
| **GAT Temperatures** | `[2.0, 2.0, 100.0, 100.0]` | Attention softmax temperature scalers |

---

## 3. Label Mapping & Scoring Direction

> [!IMPORTANT]
> The upstream `clovaai/aasist` model outputs two logits `(logits[0], logits[1])`. The label direction is strictly locked as follows:

- **Index `0`**: **`spoof`** (synthetic speech, voice conversion, or replay).
- **Index `1`**: **`bonafide`** (authentic human speech).

### Scoring Formulas:
- **Upstream ASVspoof Benchmark Score**:
  $$s_{\text{ASVspoof}} = \text{logit}_1 \quad (\text{higher indicates authentic / bona fide})$$
- **VoiceShield uncalibrated synthetic score**:
  $$s_{\text{spoof}} = \frac{e^{\text{logit}_0}}{e^{\text{logit}_0} + e^{\text{logit}_1}} = \sigma(\text{logit}_0 - \text{logit}_1) \in [0.0, 1.0]$$
  This bounded softmax component is an uncalibrated score, not a probability.
  A sample crosses the raw operating threshold when $s_{\text{spoof}} \ge 0.65$;
  aggregate HIGH still requires the configured sustained temporal policy.

---

## 4. Deterministic Evaluation Preprocessing (`pad_for_aasist`)

Upstream `clovaai/aasist` data processing operates on raw waveforms with deterministic cyclic repetition:
- **Short Audio ($N < 64,600$)**: The waveform is cyclically tiled ($N_{\text{repeats}} = \lfloor 64600 / N \rfloor + 1$) and truncated to exactly $64,600$ samples.
- **Long Audio ($N \ge 64,600$)**: The first $64,600$ samples are sliced ($x[:64600]$).
- Implemented in `src/voiceshield/models/aasist_adapter.py::pad_for_aasist`.

---

## 5. Checkpoint Acquisition & Immutable Verification

VoiceShield policy strictly forbids fabricating weights or checking binary models into Git. Official mode (`clovaai/aasist-l`) unconditionally verifies checkpoint bytes against the central immutable registry before deserialization.

To acquire the official pre-trained checkpoint from the pinned upstream release:

```bash
# 1. Create target weights directory
mkdir -p models/weights

# 2. Download official weights from pinned upstream commit
curl -L -o models/weights/AASIST-L.pth \
  https://raw.githubusercontent.com/clovaai/aasist/a04c9863f63d44471dde8a6abcb3b082b07cd1d1/models/weights/AASIST-L.pth

# 3. Verify SHA-256 checksum (enforced automatically at runtime)
shasum -a 256 models/weights/AASIST-L.pth
# Expected SHA-256: 814331d088032bb4c3fa61cc014789eadeed464209dd094ab3a2dd6ffbdce27a
# Expected Size:    426428 bytes
# Parameters:       85306
```

- **Official Verification**: Official mode (`clovaai/aasist-l`) verifies the SHA-256 hash automatically before `torch.load` and fails closed on any mismatch.
- **Custom Checkpoints**: Non-official weights require setting model identity to `custom/aasist-l` and remain reported as `custom_unverified` with `checkpoint_verified = false`.
- **Mock Mode**: Testing and UI demo mock mode (`mock/aasist-l`) is never official.

---

## 6. Hardware Acceleration (CUDA & Apple Silicon MPS)

The runner and detector support validated device selection across `cuda`, `mps`, and `cpu`:

- **CUDA Device (`--device cuda`)**:
  - Activated only when `torch.cuda.is_available()` evaluates to `True`.
  - An explicit `--device cuda` request fails closed with `RuntimeError` if CUDA is unavailable; it **never silently falls back to CPU**.
  - A CUDA-capable PyTorch build is an explicit research-environment prerequisite. The default locked Linux dependency path (`uv.lock`) remains CPU-only.
  - Acceptance numerical tolerance between CPU and CUDA inference is predeclared at $\text{atol}=10^{-4}, \text{rtol}=10^{-4}$. Because exact bitwise equality across distinct processor architectures is not expected due to differing floating-point reduction order, cuDNN kernel implementations, and FMA instructions, this predeclared tolerance defines numerical parity. Crucially, live CUDA numerical equivalence remains **UNVERIFIED** until the physical CUDA equivalence test passes on real hardware, and this tolerance must not be described as empirically validated yet.
- **Apple Silicon Metal Performance Shaders (`mps`)**:
  - Initializes on Apple Silicon when available.
  - If MPS is requested with fallback enabled (`fallback_to_cpu=True`), operators that raise a runtime error on MPS log a warning and switch execution to CPU.
- **CPU Device (`--device cpu`)**:
  - Standard deterministic CPU execution baseline.

---

## 7. Resumable Evaluation & Crash-Safe Checkpointing

The reproduction runner supports both official ASVspoof 2019 LA evaluation and development partitions with exact partition cardinalities:
- **`eval` Split**: Exactly **71,237** trials (7,355 bona fide, 63,882 spoof). Upstream ICASSP 2022 EER comparison (0.99%) is **eval-only**.
- **`dev` Split**: Exactly **24,844** trials (2,548 bona fide, 22,296 spoof). Scoring dev is supported for diagnostic verification, but its metrics are strictly marked `NOT COMPARABLE` with upstream published eval EER.

Full evaluation can be interrupted on transient or preemptible infrastructure (e.g., Google Colab). The runner provides atomic, crash-safe checkpointing:

- **Split-Specific Progress Names**:
  - `eval_progress.json` and `eval_progress.json.bak` for the evaluation split.
  - `dev_progress.json` and `dev_progress.json.bak` for the development split.
- **Clean Git Tree Requirement**: Full scoring runs require a clean working tree (`git_commit != 'unknown'` and `git_dirty is False`) and halt with a clear error if uncommitted changes are present (`git status --porcelain` must be empty). `--check-only` and `--smoke-only` may execute with dirty working trees, recording dirty state in their provenance artifacts.
- **`--resume`**: Resumes evaluation from an existing split progress checkpoint (`eval_progress.json` or `dev_progress.json`).
  - Re-evaluates only missing official protocol IDs; eliminates duplicate inference on already-scored samples.
  - Validates tamper-evident resume contract match (checkpoint SHA-256, protocol SHA-256, Git commit, model config hash, model identity, upstream commit, python/platform versions, preprocessing contract, score direction, selected split, and seed).
  - Validates all stored scores: rejects duplicate IDs, unknown IDs, malformed scores, and non-finite scores (NaN, Inf).
  - Automatically recovers from `.bak` backup if the primary progress file was truncated mid-write, while failing closed and preserving the valid primary if backup rotation fails.
  - **Completed Artifact Verification**: When `--resume` loads a `COMPLETED` progress state, it requires exact split completeness, requires `report.md` and `raw_scores.txt` to exist, recomputes and verifies their SHA-256 hashes against stored `artifact_hashes`, and exits successfully without recomputing inference or overwriting evidence.
  - If progress is `INCOMPLETE`, pre-existing `report.md` or `raw_scores.txt` files are rejected as stale final collisions.
  - If `--resume` is supplied when no progress file exists, it fails closed with `FileNotFoundError`.
- **Collision Rules & Output Directory Safeguards**:
  - **`--check-only`**: Performs only prerequisite, cardinality, and audio coverage checks without writing or scoring anything; never blocked by existing artifacts in `--output-dir`.
  - **`--smoke-only`**: Writes only `smoke_diagnostic.json`. Rejects stale evaluation output collisions (`{split}_progress.json`, `{split}_progress.json.bak`, `report.md`, `raw_scores.txt`, and temporary `*.tmp` files) in `--output-dir` to prevent colliding with or contaminating existing full evaluation outputs. Existing `smoke_diagnostic.json` files are updated atomically.
  - **Initial Full Run (non-resume)**: For initial non-resume runs, any existing progress, backup, report, raw-scores, or temporary files in `--output-dir` are rejected before inference begins with a controlled exit code 1. Existing `smoke_diagnostic.json` files from prior `--smoke-only` runs are permitted, allowing smoke diagnostics and full evaluation to share an output directory.
- **`--checkpoint-every N`**: Flushes progress to disk every $N$ batches (default: 50 batches). Never treats a partially written batch as complete. On SIGINT (`KeyboardInterrupt`) or error, best-effort persistence flushes completed batches without masking the original interruption or exception, returning code 130 or 1.
- **`--smoke-only`**: Performs prerequisite checks and deterministic stratified smoke diagnostics, writes a clearly marked non-comparable diagnostic artifact (`smoke_diagnostic.json`), and exits with code 0 before full evaluation. The smoke diagnostic artifact records full hardware and determinism provenance and never cites 0.99% or 0.0099.
- **Publication Sequence**: Final publication formats raw scores in canonical protocol order with round-trip-safe float32 `.9g` precision, computes raw score SHA-256, embeds the raw-score SHA-256 and resume-contract hash into `report.md`, publishes `raw_scores.txt` first via atomic write, publishes `report.md` second, and updates progress to `COMPLETED` last with both artifact hashes.

---

## 8. Safe Google Colab Execution Workflow

To reproduce on interruption-prone Google Colab GPU instances without redistributing restricted checkpoints or embedding secrets:

```bash
# 1. Clone repository and switch to local research branch
git clone https://github.com/ABHIGYA0205/voiceDetection.git
cd voiceDetection
git switch research/v0.2.0

# 2. Inspect and validate hardware & software environment before installation
python -c "
import platform, torch
print('Python:', platform.python_version())
print('Torch:', torch.__version__)
try:
    import torchaudio
    print('torchaudio:', torchaudio.__version__)
except ImportError:
    print('torchaudio: not installed')
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA Version:', torch.version.cuda)
    print('cuDNN Version:', torch.backends.cudnn.version() if hasattr(torch.backends, 'cudnn') else 'N/A')
    print('Device Name:', torch.cuda.get_device_name(0))
    print('Device Capability:', torch.cuda.get_device_capability(0))
"

# 3. Install dependencies safely without overwriting Colab's pre-installed CUDA PyTorch
# CRITICAL: Do NOT run 'uv sync' or an unconstrained 'pip install -e .', as doing so will
# replace Colab's pre-installed CUDA PyTorch build with the CPU-only package from the lockfile.
pip install --no-deps -e .
pip install soundfile scipy pydantic

# 4. Mount Google Drive to persist checkpoints and scores across disconnections
# from google.colab import drive
# drive.mount('/content/drive')
mkdir -p /content/drive/MyDrive/voiceshield_repro

# 5. Acquire official AASIST-L checkpoint safely from pinned upstream release
mkdir -p models/weights
curl -L -o models/weights/AASIST-L.pth \
  https://raw.githubusercontent.com/clovaai/aasist/a04c9863f63d44471dde8a6abcb3b082b07cd1d1/models/weights/AASIST-L.pth

# 6. Extract official ASVspoof 2019 LA partition under an external directory outside Git
# CRITICAL: Do NOT extract under the repository working tree (e.g., data/), because full evaluation
# requires a clean Git working tree. Keep dataset files strictly external:
mkdir -p /content/ASVspoof2019/LA
# (User extracts their authorized ASVspoof2019 LA archive into /content/ASVspoof2019/LA)
# Expected structure:
#   /content/ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt
#   /content/ASVspoof2019/LA/ASVspoof2019_LA_eval/flac/*.flac

# 7. Run fast prerequisite check pointing to the external dataset root
python scripts/reproduce_asvspoof_eval.py \
  --checkpoint models/weights/AASIST-L.pth \
  --dataset-root /content/ASVspoof2019/LA \
  --check-only

# 8. Run explicit live CUDA equivalence test forcing official checkpoint validation
VOICESHIELD_OFFICIAL_CHECKPOINT=models/weights/AASIST-L.pth \
  pytest tests/test_cuda_equivalence.py -v

# 9. Verify clean Git working tree before full evaluation (must output nothing)
git status --porcelain

# 10. Run smoke-only diagnostic verification
python scripts/reproduce_asvspoof_eval.py \
  --device cuda \
  --checkpoint models/weights/AASIST-L.pth \
  --dataset-root /content/ASVspoof2019/LA \
  --smoke-only \
  --output-dir /content/drive/MyDrive/voiceshield_repro

# 11. Run initial full evaluation (omit --resume on the first run)
python scripts/reproduce_asvspoof_eval.py \
  --device cuda \
  --checkpoint models/weights/AASIST-L.pth \
  --dataset-root /content/ASVspoof2019/LA \
  --checkpoint-every 25 \
  --output-dir /content/drive/MyDrive/voiceshield_repro

# 12. Reconnect / resume command (if interrupted or disconnected, add --resume with the exact same arguments)
python scripts/reproduce_asvspoof_eval.py \
  --device cuda \
  --checkpoint models/weights/AASIST-L.pth \
  --dataset-root /content/ASVspoof2019/LA \
  --checkpoint-every 25 \
  --output-dir /content/drive/MyDrive/voiceshield_repro \
  --resume
```

---

## 9. Benchmark Integrity Notice

VoiceShield does not claim a reproduced EER until the complete official ASVspoof 2019 LA evaluation protocol is scored with frozen weights. Adding CUDA support and crash-safe resumability improves engineering usability on interruption-prone research platforms but does not validate or change model accuracy.

The runner reports a separate diagnostic normalized minDCF, but it does not compute official min t-DCF; that metric requires the official ASV scores, keys, priors, costs, and t-DCF implementation. Official min t-DCF remains unimplemented in this runner.
