# VoiceShield Model & Evaluation Cards

This directory contains truthful, evidence-backed model cards, evaluation cards, and pitch deck audit checklists for detectors deployed in VoiceShield.

In strict compliance with scientific integrity standards:
- **Every reported metric is linked to a saved report artefact and cryptographic hashes.**
- **If local evaluation evidence is absent, metrics are truthfully marked "Not yet evaluated" rather than guessed or fabricated.**
- **Pitch decks and documentation are governed by an actionable claims verification checklist.**

---

## Registered Documentation & Cards

| Document | Scope | Status |
| :--- | :--- | :---: |
| [**`aasist_l_model_card.md`**](./aasist_l_model_card.md) | AASIST-L architecture, provenance, preprocessing contract, threat model, calibration, and known technical limitations. | **Active (card v1.0.0; product v0.1.1)** |
| [**`aasist_l_evaluation_card.md`**](./aasist_l_evaluation_card.md) | Evaluation splits, leakage controls, subgroup breakdowns (generators, languages, codecs), and M3 hardware latency profiling. | **Active (card v1.0.0; product v0.1.1)** |
| [**`claims_verification_checklist.md`**](./claims_verification_checklist.md) | Pre-pitch and README verification gate to prevent overclaiming, deceptive accuracy metrics, or false identity claims. | **Active (card v1.0.0; product v0.1.1)** |

---

## Evaluation Artefacts Referenced

- **M3 Hardware Latency Benchmark Report**: [`../reports/benchmarks/m3_benchmark_report.md`](../reports/benchmarks/m3_benchmark_report.md)
- **M3 Hardware JSON Telemetry**: [`../reports/benchmarks/m3_benchmark_report.json`](../reports/benchmarks/m3_benchmark_report.json)
- **Official Reproduction Checklist & Script**: [`../docs/aasist_reproduction.md`](../docs/aasist_reproduction.md) & [`../scripts/reproduce_asvspoof_eval.py`](../scripts/reproduce_asvspoof_eval.py)
