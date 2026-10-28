# VoiceShield: AASIST-L ASVspoof 2019 LA Reproduction Report

> VoiceShield reports an **uncalibrated synthetic acoustic anomaly score** for
> advisory use. It does not prove identity, estimate fraud probability, or
> authenticate a caller.

## 1. Benchmark status

| Metric | Pinned upstream reference | VoiceShield reproduction | Status |
|---|---:|---:|---|
| EER | 0.99% (0.0099) | `{{ actual_eer_pct }}` | `{{ status_eer }}` |
| Official min t-DCF | 0.0309 | NOT COMPUTED | NOT COMPUTED |
| Evaluation set | `{{ evaluated_protocol }}` | `{{ total_eval_trials }}` trials | Counts derived from protocol |

The pinned AASIST repository reports AASIST-L EER 0.99%, min t-DCF 0.0309,
and 85,306 parameters:
`https://github.com/clovaai/aasist/tree/{{ upstream_commit }}`.
Official min t-DCF requires the official ASV scores, keys, priors, costs, and
t-DCF implementation; those inputs are not integrated into this runner.

The local normalized binary detection-cost metric, when present below, is a
**VoiceShield diagnostic only**. It is not min t-DCF and is not comparable with
the upstream 0.0309 reference.

## 2. Protocol and model provenance

- Model identifier: `{{ model_id }}`
- Upstream repository: `https://github.com/clovaai/aasist`
- Upstream commit: `{{ upstream_commit }}`
- Upstream source-code licence: MIT (NAVER Corp / Clova AI)
- Checkpoint rights: separate from the source-code licence and not independently established by VoiceShield
- Checkpoint distribution: obtained from the pinned upstream source; checkpoint bytes are not redistributed in generated artifacts
- Parameter count: `{{ parameter_count }}`
- Protocol: `{{ evaluated_protocol }}`
- Protocol SHA-256: `{{ protocol_sha256 }}`
- Evaluation framing: cyclic pad/truncate to 64,600 samples at 16 kHz
- Raw prediction artifact: uncalibrated bona fide logit, `logit[1]`
- Raw-score direction: higher `logit[1]` means more bona fide
- Internal metric score: `-logit[1]`, so higher means more spoof-like
- Neither score is calibrated or probabilistic
- Score calibrator loaded: no
- Brier score: NOT COMPUTED
- Expected calibration error (ECE): NOT COMPUTED
- Reliability claims: NOT COMPUTED

## 3. Execution provenance

- Execution timestamp: `{{ execution_timestamp }}`
- Git commit: `{{ git_commit }}`
- Git working tree dirty: `{{ git_dirty }}`
- Checkpoint registry identity: `{{ model_id }}`
- Checkpoint registry URL: `{{ checkpoint_registry_url }}`
- Registry checkpoint SHA-256: `{{ expected_checkpoint_sha256 }}`
- Loaded checkpoint SHA-256: `{{ checkpoint_sha256 }}`
- Device: `{{ compute_device }}`
- Requested device: `{{ requested_device }}`
- CUDA available: `{{ cuda_available }}`
- CUDA device name: `{{ cuda_device_name }}`
- MPS available: `{{ mps_available }}`
- Deterministic seed: `{{ seed }}`
- Deterministic algorithms: `{{ deterministic_algorithms }}`
- Operating system/platform: `{{ os_version }}`
- Python: `{{ python_version }}`
- PyTorch: `{{ torch_version }}`
- Batch size: `{{ batch_size }}`
- Selected split: `{{ selected_split }}`
- Smoke size: `{{ smoke_samples_count }}`
- Smoke bona fide count: `{{ smoke_bonafide_count }}`
- Smoke spoof count: `{{ smoke_spoof_count }}`
- Reproduction-parameter hash: `{{ config_sha256 }}`
- Exact invoked command: `{{ reproduction_command }}`

## 4. Protocol cardinality and smoke diagnostics

- Total protocol trials: `{{ total_eval_trials }}`
- Bona fide trials: `{{ bonafide_eval_trials }}`
- Spoof trials: `{{ spoof_eval_trials }}`
- Protocol IDs: `{{ protocol_id_count }}`
- FLAC files found: `{{ audio_file_count }}`
- Missing protocol audio files: `{{ missing_audio_count }}`
- Extra unreferenced FLAC files: `{{ extra_audio_count }}`
- Smoke forward-pass status: `{{ smoke_status }}`
- Mean bona fide score: `{{ smoke_mean_bonafide_score }}`
- Mean spoof score: `{{ smoke_mean_spoof_score }}`
- Empirical score-direction check: `{{ smoke_direction_aligned }}`

Extra audio files are informational and are not evaluated or deleted.

## 5. Evaluation results

- EER: `{{ actual_eer_pct }}`
- EER threshold: `{{ eer_threshold }}`
- EER discrepancy from pinned reference: `{{ diff_eer }}`
- Official min t-DCF: NOT COMPUTED
- Local min t-DCF: NOT COMPUTED
- VoiceShield diagnostic normalized minDCF (non-comparable): `{{ actual_mindcf }}`
- Diagnostic threshold: `{{ mindcf_threshold }}`
- ROC-AUC: NOT COMPUTED
- PR-AUC: NOT COMPUTED
- Confusion matrix: NOT COMPUTED
- Operating-point metrics: NOT COMPUTED

## 6. Limitations

This run tests ASVspoof 2019 LA generalization only. It does not validate
ElevenLabs audio, telephony, score calibration, fraud detection, or
authentication. No detection-performance claim should be inferred from this
reproduction alone.

## 7. Artifacts

- Raw predictions: `{{ raw_scores_path }}` (SHA-256: `{{ raw_scores_sha256 }}`)
- Generated report: `{{ report_path }}`
- Resume contract hash: `{{ resume_contract_hash }}`
