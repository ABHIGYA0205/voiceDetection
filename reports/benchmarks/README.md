# Benchmark evidence

## Canonical v0.1.0 report

The top-level [Markdown report](./m3_benchmark_report.md) and
[JSON report](./m3_benchmark_report.json) are the canonical VoiceShield v0.1.0
real-checkpoint benchmark artifacts.

- Markdown SHA-256:
  `5fc5432b220e4c5b9ed2a980948a6fa8c7de32808c1a8d8e88e8873049531438`
- JSON SHA-256:
  `73b548187230662ef487b4829cf36b3e0e6f7ee494d9937956353a7fdbd74405`
- Benchmark source commit: `392ca34eaf76eecf4afd8d23ca5049f39eec08f5`
- Official checkpoint SHA-256:
  `814331d088032bb4c3fa61cc014789eadeed464209dd094ab3a2dd6ffbdce27a`
- Iterations: 10 warmups and 100 steady-state iterations per measured stage.
- Stress soak: 600 seconds of accelerated ingestion.

Accelerated ingestion feeds frames without real-time pacing and may represent
more audio than elapsed wall-clock time. It is a stress-test methodology, not a
real-time-paced streaming claim.

The performance evidence does not establish accuracy, calibration, robustness
to unseen generators, telephony validity, or production readiness. Pipeline
stage timings are measured independently. Any stage-to-end-to-end ratios are
diagnostic comparisons and are not additive percentages.

## Archived R11 evidence

[The archived JSON report](./archive/m3_real_checkpoint_60m_r11_raw.json) is raw
historical R11 acceptance evidence, preserved unchanged.

- SHA-256:
  `ec723f7643f0fb2880c793c0328993f00e748f750704d60e4c648d401b229b18`
- Generated at commit
  `0f486fc31e2a3e271293476f06b4c5e7cd893fdd`, before the reporting terminology
  correction.
- Ingestion was accelerated.

This archive is not the canonical v0.1.0 report.
