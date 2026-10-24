# Third-party notices

## VoiceShield source

Original VoiceShield source code is licensed under the Apache License 2.0.
See [LICENSE](./LICENSE).

## Vendored AASIST source

This repository vendors source from [clovaai/aasist](https://github.com/clovaai/aasist)
at commit `a04c9863f63d44471dde8a6abcb3b082b07cd1d1`. The vendored source is
licensed under the upstream MIT terms. The complete upstream notice is preserved
at `src/voiceshield/vendor/aasist/NOTICE`, alongside the upstream `LICENSE`.

## AASIST-L checkpoint

The official AASIST-L checkpoint is a separate model-weight provenance boundary.
It is obtained, when needed, only from
https://raw.githubusercontent.com/clovaai/aasist/a04c9863f63d44471dde8a6abcb3b082b07cd1d1/models/weights/AASIST-L.pth
at upstream commit `a04c9863f63d44471dde8a6abcb3b082b07cd1d1`. Its verified
SHA-256 is
`814331d088032bb4c3fa61cc014789eadeed464209dd094ab3a2dd6ffbdce27a`.
The verified checkpoint size is `426,428 bytes`.
It is not committed to this repository. The Apache-2.0 license for VoiceShield
source code does not grant permission to redistribute that checkpoint.

## Datasets and user audio

ASVspoof and other referenced datasets are not distributed by this repository and
must be obtained and used lawfully under their own terms. User-provided audio is
not covered by the VoiceShield source-code license.

## Python and npm dependencies

Resolved Python dependencies are inventoried in `uv.lock`; resolved npm
dependencies are inventoried in `apps/web/package-lock.json`. Each dependency
retains its own applicable license. The repository's resolved-license gate checks
these inventories and rejects unknown or disallowed executable dependency terms.
