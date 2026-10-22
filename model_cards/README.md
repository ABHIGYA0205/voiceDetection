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
| [**`aasist_l_model_card.md`**](file:///Users/Abhi/Downloads/projects/deepfake%20voice/model_cards/aasist_l_model_card.md) | AASIST-L architecture, provenance, preprocessing contract, threat model, calibration, and known technical limitations. | **Active (v1.0.0)** |
| [**`aasist_l_evaluation_card.md`**](file:///Users/Abhi/Downloads/projects/deepfake%20voice/model_cards/aasist_l_evaluation_card.md) | Evaluation splits, leakage controls, subgroup breakdowns (generators, languages, codecs), and M3 hardware latency profiling. | **Active (v1.0.0)** |
| [**`claims_verification_checklist.md`**](file:///Users/Abhi/Downloads/projects/deepfake%20voice/model_cards/claims_verification_checklist.md) | Pre-pitch and README verification gate to prevent overclaiming, deceptive accuracy metrics, or false identity claims. | **Active (v1.0.0)** |

---

## Evaluation Artefacts Referenced

- **M3 Hardware Latency Benchmark Report**: [`reports/benchmarks/m3_benchmark_report.md`](file:///Users/Abhi/Downloads/projects/deepfake%20voice/reports/benchmarks/m3_benchmark_report.md)
- **M3 Hardware JSON Telemetry**: [`reports/benchmarks/m3_benchmark_report.json`](file:///Users/Abhi/Downloads/projects/deepfake%20voice/reports/benchmarks/m3_benchmark_report.json)
- **Official Reproduction Checklist & Script**: [`docs/aasist_reproduction.md`](file:///Users/Abhi/Downloads/projects/deepfake%20voice/docs/aasist_reproduction.md) & [`scripts/reproduce_asvspoof_eval.py`](file:///Users/Abhi/Downloads/projects/deepfake%20voice/scripts/reproduce_asvspoof_eval.py)
