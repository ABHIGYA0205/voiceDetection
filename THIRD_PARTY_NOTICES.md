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
It is obtained, when needed, from the pinned upstream URL recorded by the model
registry and is not committed to this repository. The Apache-2.0 license for
VoiceShield source code does not grant permission to redistribute that checkpoint.

## Datasets and user audio

ASVspoof and other referenced datasets are not distributed by this repository and
must be obtained and used lawfully under their own terms. User-provided audio is
not covered by the VoiceShield source-code license.

## Python and npm dependencies

Resolved Python dependencies are inventoried in `uv.lock`; resolved npm
dependencies are inventoried in `apps/web/package-lock.json`. Each dependency
retains its own applicable license. The repository's resolved-license gate checks
these inventories and rejects unknown or disallowed executable dependency terms.
