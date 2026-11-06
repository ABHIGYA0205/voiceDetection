# Stage C3 Corpus-Rights Recommendations

## Status and scope

This document records a recommendations-only review for the single purpose
`research_evaluation`. It is based on VoiceShield commit
`76124adfde57b4624667ee632cd0908a01a1991f`, the accepted C2 evidence file, and
the frozen C1 corpus registry.

This package is not legal advice and does not make a project rights decision.
It does not authorize a registry change, account creation, acceptance of terms,
data acquisition, evaluation, calibration, training, product use, or
redistribution. A named human reviewer must resolve the documented
preconditions and make a separate purpose-specific decision before the registry
can change.

The user selected the following Stage C3 boundaries:

- research evaluation only;
- recommendations before any actual decision;
- no dataset acquisition;
- no account creation or acceptance of terms;
- no change to the C1 registry.

## Integrity anchors

| Item | Frozen value |
| --- | --- |
| Base source commit | `76124adfde57b4624667ee632cd0908a01a1991f` |
| C2 evidence SHA-256 | `5017f75d282b71ec6b05e589af08b02e541cbc3c27953363a03601fb78993197` |
| C1 registry SHA-256 | `686ad25c1f0e390f67d59d6b8da817d0f7009c82d7a08d8f0d9a75aeaa87ce85` |
| Recommendation date | 2026-11-06 |
| Scoped purpose | Research evaluation only |

The validator fails closed if either linked file changes byte-for-byte, if the
corpus sets diverge, if an evidence source reference is unknown, or if any
registry entry ceases to be pending and planned with an empty allowed-purpose
list.

## Recommendation summary

| Priority | Corpus | Current recommendation | Primary blocker |
| ---: | --- | --- | --- |
| 1 | ASVspoof 5 | Defer pending conditions | ODC-By-1.0 does not independently clear individual audio Contents; component-level audio rights, privacy, and consent remain unresolved. |
| 2 | MLAAD v9 | Defer pending conditions | NonCommercial scope and separate M-AILABS component terms have not been resolved for the exact evaluation workflow. |
| 3 | Common Voice 18 | Defer pending conditions | Exact MDC access contract and Data Consumer License are not frozen; anti-re-identification controls are not yet specified and tested. |
| 4 | IndicVoices | Defer pending conditions | Complete gated conditions are unavailable in the frozen public evidence, and distributed metadata requires a default-deny privacy and minimization control. |

“Defer” is not a rejection of a corpus. It means the evidence currently
supports continued review but does not support an authorization.

## 1. ASVspoof 5

### Why it is first

ASVspoof 5 is directly accessible and is the strongest candidate among the four
for a modern spoof-detection benchmark. It should be reviewed first because it
can provide direct attack-evaluation evidence without the account and gated
access steps required by the other candidates.

### Why evaluation is still deferred

The declared ODC-By-1.0 terms cover the defined Database, Database Rights, and
contract rights. Section 2.4 does not independently grant rights in individual
Contents other than Database Rights or contract rights. The available evidence
does not establish complete rights, consent, and privacy clearance for every
underlying audio component.

A later human review must establish the exact component terms, evaluate the
Database-versus-Contents boundary, and determine the notices required by
Sections 4.2 and 4.3. No corpus files should be downloaded while those issues
remain unresolved.

### Minimum boundary if later approved

A later decision should initially be restricted to research evaluation. It
should not authorize calibration, training, product use, or raw redistribution.
Only required audio, anonymous trial keys, labels, and minimal evaluation
metadata should enter the research environment.

## 2. MLAAD v9

### Potential value

MLAAD offers multilingual synthetic speech and modern generator coverage. Its
CC-BY-NC-4.0 declaration may support genuinely non-commercial research
evaluation.

### Why evaluation is still deferred

The benchmark requires separate bona fide reference audio from M-AILABS.
M-AILABS has component-specific provenance, and its repository documents a
Ukrainian machine-learning-only exception. The project has not completed a
language-subset terms matrix or established that the exact VoiceShield activity
qualifies as NonCommercial under Section 1(i) of CC-BY-NC-4.0.

The analysis must also consider whether institutional affiliation,
sponsorship, hosted services, or the planned product track changes that
NonCommercial characterization. Research and product artifacts must remain
segregated.

### Minimum boundary if later approved

A later decision should limit MLAAD to a segregated non-commercial
research-evaluation environment. It should exclude product evaluation,
calibration, training, product artifacts, and raw redistribution. MLAAD
synthetic data and each M-AILABS bona fide component must retain separate
provenance, terms, and notices.

## 3. Common Voice 18

### Potential value

Common Voice could provide broad bona fide language and accent coverage for
false-positive evaluation.

### Why evaluation is still deferred

The CC0 dedication operates only to the extent of rights held by the affirmer
and does not itself prove privacy, consent, or third-party clearance. Access is
mediated through Mozilla Data Collective. The applicable contractual layer
includes an anti-re-identification covenant and redistribution constraints, but
the exact Data Consumer License for the intended snapshot has not been
immutably captured.

Before any decision, the project needs the exact access-time contract, a
qualified reconciliation of the CC0 and contractual layers, and testable
controls that prohibit contributor identification, matching, tracing, or
source re-identification.

### Minimum boundary if later approved

A later decision should restrict Common Voice to bona fide research evaluation,
prohibit raw mirroring and redistribution, and exclude optional demographics.
Only audio, anonymous tokens required for leakage control, language or locale,
and essential integrity metadata should be retained.

## 4. IndicVoices

### Potential value

IndicVoices is especially relevant to Indian-language and regional robustness.
The public materials declare CC-BY-4.0 and describe a large multilingual speech
collection.

### Why evaluation is still deferred

Repository access is gated, and the complete conditions shown during the
approval workflow are not part of the frozen evidence. The public dataset card
also exposes transcripts, demographic and geographic attributes,
verification-related fields, and unsanitized transcript fields. Public evidence
does not establish that every distributed field is free of incidental personal
data.

The project therefore needs a public, non-acceptance copy of the complete gated
conditions, qualified human review, and an enforceable field allowlist before
it can consider access.

### Minimum boundary if later approved

A later decision should permit only research evaluation and should enforce a
default-deny allowlist containing audio, anonymous speaker tokens, language
labels, and essential integrity fields. Transcripts, demographic and geographic
fields, verification reports, names, phone numbers, consent material, and
incidental personal data must be excluded. Demographic, geographic, identity,
and speaker-recognition analysis must remain prohibited.

## Required sequence before any registry change

For each corpus, the project must complete the following sequence independently:

1. Resolve the corpus-specific preconditions in the machine-readable packet.
2. Capture and hash every mutable source relied upon by the reviewer.
3. Preserve the exact terms and version references reviewed.
4. Obtain a named human reviewer’s explicit decision for research evaluation.
5. Record only the allowed purpose and mandatory restrictions supported by that decision.
6. Run registry and evidence validation before acquisition.
7. Acquire the exact approved snapshot only after the decision.
8. Immediately hash the acquired archive before unpacking or processing.

No decision for research evaluation automatically permits calibration,
training, product use, or redistribution. Each later purpose requires a
separate review.

## Validation

Run the deterministic validator from the repository root:

```bash
PYTHONPATH=src python scripts/validate_corpus_rights_recommendations.py
```

A valid result reports:

- four checked recommendations;
- all four dispositions as `defer_pending_conditions`;
- `research_evaluation` as the only scope;
- all authorization flags as `false`;
- exact evidence, registry, and recommendation hashes;
- no registry mutation.
