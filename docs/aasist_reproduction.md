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

## 6. Hardware Acceleration & Apple Silicon Fallback

The adapter initializes on Apple Silicon `mps` when available:
- If MPS executes cleanly, GPU acceleration is active.
- If any operator (e.g. dynamic graph gathering or scatter on specific PyTorch builds) raises a runtime error on MPS, the adapter catches the exception, logs a warning, switches execution to CPU, and marks `metadata["fallback_used"] = True`.

---

## 7. Benchmark Integrity Notice

VoiceShield does not claim a reproduced EER until the complete official ASVspoof 2019 LA evaluation protocol is scored with frozen weights. The runner reports a separate diagnostic normalized minDCF, but it does not compute official min t-DCF; that metric requires the official ASV scores, keys, priors, costs, and t-DCF implementation.
