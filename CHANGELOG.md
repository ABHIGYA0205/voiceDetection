# Changelog

All notable changes to VoiceShield are documented here.

## [0.1.1] - 2026-10-25

### Fixed

- Preserve meaningful raw threshold crossings and uncertain windows in offline
  file verdicts instead of presenting a final LOW window as uniformly low
  history; HIGH still requires sustained temporal-policy evidence.
- Fail closed when an ASVspoof 2019 LA reproduction has incorrect official
  cardinality, incomplete protocol/audio coverage, invalid smoke allocation,
  a non-canonical upstream EER comparison, or unresolved report provenance.
- Correct stale ASVspoof counts, upstream metric references, score semantics,
  and reproduction commands in active documentation.

### Validation

- Official eval prerequisite validation passed for 71,237 protocol trials
  (7,355 bona fide and 63,882 spoof), 71,933 FLAC stems, zero missing files,
  and 696 permitted extra files.
- The complete local gate passed, followed by successful Linux CI on the exact
  release-preparation parent commit.

### Limitations

No reproduced EER, official min t-DCF, calibrated probability, or broad modern
TTS reliability claim is included. The interrupted local full-evaluation
attempt is not scientific evidence. See the
[v0.1.1 release notes](docs/releases/v0.1.1.md).

## [0.1.0] - 2026-10-25

### Added

- Advisory local microphone streaming and WAV/FLAC file analysis.
- Official AASIST-L checkpoint integrity and provenance controls.
- Canonical preprocessing and temporal policy with abstention and quality
  diagnostics.
- CPU/MPS engineering benchmark evidence on Apple M3 hardware.
- Browser/API integration with privacy, resource, and security controls.
- Reproducibility, packaging, license, vulnerability, and CI gates.

### Limitations

This release does not provide scientific detection-performance validation,
calibrated probabilities, identity authentication, fraud proof, cellular-call
interception, or production deployment certification. See the
[v0.1.0 release notes](docs/releases/v0.1.0.md).
