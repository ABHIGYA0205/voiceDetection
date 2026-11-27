# AASIST-L on ASVspoof 2019 Logical Access

VoiceShield completed the official evaluation split on November 2, 2026 IST using
the upstream AASIST-L checkpoint at VoiceShield commit
[`f67383a`](https://github.com/ABHIGYA0205/voiceDetection/commit/f67383a99c6566c3d2766d313b94e08260d6fd79).
The saved output covers 71,237 unique trials (7,355 bona fide and 63,882 spoof).
The recorded EER is **0.99%**, matching the
[pinned upstream result](https://github.com/clovaai/aasist/blob/a04c9863f63d44471dde8a6abcb3b082b07cd1d1/README.md).

| Metric | Completed run |
| --- | --- |
| Equal error rate | **0.99%** |
| Diagnostic normalized binary minDCF | 0.2615 |
| Official min t-DCF | Uncomputed |

The runner used the raw bona fide logit from one 64,600-sample crop or cyclically
padded utterance, with a sign flip for spoof-positive metrics. The app uses
overlapping windows, a different score, quality checks, and a temporal policy.
This run measures the official 2019 LA protocol. Modern generators, call audio,
and app alert accuracy require separate evaluation.

| Provenance | Value |
| --- | --- |
| Device | Tesla T4 (CUDA) |
| Batch size and deterministic seed | 24 and 42 |
| Official protocol SHA-256 | `84fce576efff092269a0a6ad247d2e9f2f7b4d4e25a4545efb2518da92708036` |
| Weights SHA-256 | `814331d088032bb4c3fa61cc014789eadeed464209dd094ab3a2dd6ffbdce27a` |
| Completed raw scores SHA-256 | `5315929dc9fc9c13447dd661c83cd1d6ae05d32202df022a565a02b0f393d4d7` |
| Original run report SHA-256 | `74890e2e0ef6102e133d902997c1c4e05d87c9cc8b6204b0102b551fb9385f5a` |

The evidence review verified all archive file hashes and matched each one of the
71,237 saved score rows to the completed resume state. See the
[reproduction guide](../docs/aasist_reproduction.md) to regenerate the evidence
with the official corpus and checkpoint.
