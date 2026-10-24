# VoiceShield Claims & Pitch Deck Verification Checklist

**Purpose**: Enforce scientific integrity and prevent exaggerated, misleading, or unverified claims across project documentation, `README.md`, pitch decks, and hackathon presentations (e.g. Smart India Hackathon / SIH).

**Governing Standards**:
- Model Card: [`aasist_l_model_card.md`](./aasist_l_model_card.md)
- Evaluation Card: [`aasist_l_evaluation_card.md`](./aasist_l_evaluation_card.md)
- Security Specification: [`docs/security_privacy.md`](../docs/security_privacy.md)

---

## 1. Strict Prohibitions (Hard Red Lines)

Any README, slide deck, or demonstration containing any of the following statements is **unsupported, deceptive, and strictly rejected**:

| Prohibited Claim | Why It Is Deceptive | Permitted Alternative |
| :--- | :--- | :--- |
| ❌ *"100% deepfake prevention"* or *"Zero false positives"* | Mathematically impossible in stochastic audio detection; generative speech models evolve continuously. | *"Probabilistic acoustic anomaly detector providing advisory risk signals."* |
| ❌ *"Proves identity of caller"* or *"Authenticates human speaker"* | VoiceShield evaluates synthetic speech acoustic artifacts, **not** biometric speaker identity. A scammer can use their real human voice. | *"Detects artificial voice synthesis artifacts; does NOT verify caller identity."* |
| ❌ *"Proven 99.01% accuracy / 0.99% EER on our platform"* (before running local ASVspoof) | Local evaluation split has not yet been executed on-disk; citing upstream papers as local achievements is fraudulent. | *"AASIST-L upstream paper reports 0.99% EER on ASVspoof 2019 LA; local evaluation split is pending dataset setup."* |
| ❌ *"Prevents financial fraud / bank account takeover"* | An audio score alone cannot stop banking fraud. Fraudsters use human coercion, social engineering, and SIM swaps. | *"Provides defense-in-depth advisory signals prompting independent human verification."* |
| ❌ *"Real-time 30ms latency on real models"* without qualification | A latency measurement is not detection-performance evidence and does not establish real-time accuracy. | *"On Apple M3 using the verified official AASIST-L checkpoint, MPS end-to-end p95 was 34.546 ms over 100 steady-state iterations."* |
| ❌ *"Compliant with EU AI Act / GDPR / BIPA"* | Software cannot grant legal compliance; deployer policies and legal review govern statutory compliance. | *"Engineered with volatile-only audio memory and zero disk retention to facilitate compliance reviews."* |

---

## 2. Permitted & Empirically Grounded Claims

The following claims are **verified by code, unit tests, and saved report artefacts** and may be cited:

1. **Lightweight Edge Architecture**:
   - AASIST-L has **85,306 parameters (~341 KB FP32)**, making it small enough to run locally on consumer hardware without cloud GPU clusters.
2. **Apple Silicon M3 Hardware Acceleration**:
   - On Apple M3 using the verified official AASIST-L checkpoint, MPS model inference p50 was **30.187 ms** and p95 was **30.761 ms** (RTF = 0.00755).
   - On the same run, MPS end-to-end p50 was **33.754 ms** and p95 was **34.546 ms** (RTF = 0.00844).
   - MPS end-to-end speedup was **3.32x** versus CPU.
   - These are engineering latency measurements, not detection-accuracy evidence.
   - *Provenance Artefact*: [`reports/benchmarks/m3_benchmark_report.md`](../reports/benchmarks/m3_benchmark_report.md).
3. **Streaming Soak & Memory Stability**:
   - A 600-second accelerated streaming stress soak completed with bounded reported memory and backlog (`+0.00 MB`, maximum queue backlog of 1 window).
4. **Zero Audio Persistence**:
   - Audio is buffered exclusively in volatile RAM (`io.BytesIO`, `AudioRingBuffer`). **Zero audio frames or spoken transcripts are written to disk or recorded in logs.**
5. **Deterministic Preprocessing**:
   - Canonical 16,000 Hz float32 mono contract, deterministic channel averaging, Kaiser-windowed polyphase resampling, and deterministic cyclic framing matching Clova AI.
6. **Defense-in-Depth Temporal Policy**:
   - Median and EWMA smoothing, hysteresis elevation with configurable $N$-of-$M$ thresholds, and explicit abstention when active speech duration is $< 0.8\text{ s}$.

---

## 3. Pre-Pitch / README Verification Checklist

Before publishing documentation, creating release tags, or presenting the Smart India Hackathon (SIH) deck, verify that each box is checked:

### Scope & Framing
- [ ] The solution is described as an **advisory risk sensor**, not an autonomous decision-maker.
- [ ] Slides/docs explicitly state that VoiceShield **does not prove caller identity**.
- [ ] No slide promises "100% fraud prevention" or "elimination of deepfakes".
- [ ] Call-to-action advises: *"Pause and verify through a known independent channel."*

### Performance & Metric Truthfulness
- [ ] If ASVspoof 2019 LA evaluation has **not** been run locally, accuracy/EER is clearly labeled as **"Upstream reference (Jung et al. 2022); local evaluation not yet evaluated"**.
- [ ] Latency numbers specify **Apple MacBook Pro M3 hardware**, **batch size 1**, and cite `reports/benchmarks/m3_benchmark_report.md`.
- [x] The release benchmark identifies the verified official checkpoint and distinguishes engineering latency from detection-performance evidence.
- [ ] Unseen generator, language, and telephony codec limitations are explicitly listed as areas for future adaptation.

### Privacy & Architecture Integrity
- [ ] Zero disk storage and volatile RAM processing are documented accurately.
- [ ] Safe deserialization (`weights_only=True`) and SHA-256 weight integrity checks are cited.
- [ ] Legal compliance is **not** claimed; the requirement for independent legal review is preserved.
- [ ] AudioWorklet microphone capture with explicit user gesture is highlighted.
