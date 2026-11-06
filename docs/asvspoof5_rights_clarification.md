# Stage C4-A ASVspoof 5 Rights Clarification

## Status and boundary

This is a non-authorizing public-source clarification for the single purpose
`research_evaluation`. It is not legal advice or a project rights decision. It
does not authorize contacting maintainers, creating accounts, accepting terms,
downloading data, evaluating, training, calibrating, using data in a product, or
redistributing data.

ASVspoof 5 remains `planned` with a `pending` rights decision and no allowed
purposes. The recommended disposition remains `defer_pending_conditions`.

Stage C4-A did not download audio, corpus archives, the protocol archive, model
weights, or any other dataset payload. It did not create an account or accept
terms. It captured only public legal text, release metadata, README files,
research papers, evaluation plans, source-corpus pages, and model-license
metadata.

## Integrity anchors

| Item | Frozen value |
| --- | --- |
| Accepted C3 commit | `4d4af273acb16e28569c0573f68fa65272e013ec` |
| C2 evidence SHA-256 | `5017f75d282b71ec6b05e589af08b02e541cbc3c27953363a03601fb78993197` |
| C3 recommendations SHA-256 | `d80ed9ab61d4e97ff5bca9201179e281e725c875f03399704b1986cd6a1cebb3` |
| Frozen registry SHA-256 | `686ad25c1f0e390f67d59d6b8da817d0f7009c82d7a08d8f0d9a75aeaa87ce85` |
| Capture manifest SHA-256 | `382b2ae9646a287e84110f6dcf92c6e17358a05bb003572e7255cd473a0da845` |
| Captured public sources | 19 artifacts, 9,477,682 bytes |
| Clarification date | 2026-11-07 |

The committed capture manifest records each exact retrieval URL, source
identity, revision, UTC retrieval timestamp, byte size, and SHA-256 value. The
source documents themselves are retained outside the repository so the project
does not republish third-party legal and research documents.

## What public evidence clarified

### 1. Database-level terms are confirmed

The Zenodo record identifies ASVspoof 5 v0.1 as an ODC-By dataset. The official
`LICENSE.txt` is the ODC-By 1.0 legal text. This confirms the database-level
licence and captures the separate notice paths for conveying the database and
publicly using a Produced Work.

This does not settle individual audio content. ODC-By Section 2.4 expressly
separates Database rights from independent rights in individual Contents,
including copyright, data-protection, privacy, and personality rights.

### 2. Bona fide content is materially clearer

The official `README.txt` expressly states that the bona fide data is
distributed under CC BY 4.0. The design paper says the bona fide recordings are
sourced directly from MLS English. OpenSLR identifies MLS as CC BY 4.0 and as
derived from LibriVox audiobooks.

LibriVox states that volunteer recordings are donated to the public domain and
may be reused. It also warns that its public-domain determination is certain in
the United States but may differ in other countries.

This resolves the earlier absence of an explicit public content-licence
statement for bona fide ASVspoof 5 audio. It does not by itself resolve licensor
authority, exact attribution, international public-domain treatment, or
privacy, personality, voice, and data-protection questions.

### 3. Spoof content remains the main rights blocker

The design paper explains that external contributors generated TTS and voice
conversion attacks and that adversarial filters were applied to some evaluation
attacks. The README credits those contributors but gives a separate CC BY 4.0
content statement only for bona fide data.

No captured public source supplies a content-level licence, contributor
assignment, or equivalent permission covering every spoof and adversarial audio
file. The ODC-By database licence does not independently fill that gap.

### 4. Some attack-model terms may affect outputs

The paper identifies A28 as a pre-trained Coqui YourTTS system and A29 as a
pre-trained Coqui XTTS system. It also says A17, A28, and A29 were pre-trained
on external datasets and documents optional Common Voice 11 use for speaker
encoders.

Pinned public Coqui metadata labels YourTTS `CC BY-NC-ND 4.0` and XTTS v1.1 and
v2 `CPML`. The pinned XTTS v2 CPML text expressly limits both the model and its
outputs to non-commercial purposes and requires downstream recipients of an
output to receive the terms or their URL.

That evidence is a warning, not a conclusion about the released ASVspoof files.
The public ASVspoof paper does not identify the exact checkpoints, versions,
terms in force, or any separate licences used by contributors. Those facts must
come from maintainers.

### 5. Common Voice is an indirect component

Historical Mozilla terms in force by the Common Voice 11 release required
contributors to dedicate contributions to CC0 and assure that contributions
were their own and did not infringe third-party rights. The corresponding
privacy notice said voice recordings and optional demographics could be made
public under CC0.

The ASVspoof documentation describes Common Voice 11 as optional data for
training speaker encoders. It does not identify which attack IDs used it or
whether an access-time covenant or notice was intended to flow to generated
outputs.

### 6. Challenge rules and current public use are not the same question

The Phase 2 v0.6 plan contains mandatory rules for 2024 challenge participants,
including data-condition restrictions, no train/development pooling,
independent test scoring, submission duties, and ethics language. The current
Zenodo release is directly accessible and recommends the plan when comparable
results are desired.

The public documents do not clearly distinguish which rules remain binding
conditions of independent post-challenge use, which applied only to ranked 2024
participation, and which are now methodological recommendations. Maintainer
clarification is needed.

## C3 precondition status

| C3 precondition | C4-A status | Result |
| --- | --- | --- |
| Authoritative component-rights documentation | Partially satisfied | Bona fide CC BY 4.0 and source provenance are now documented; spoof content rights and model-output terms remain open. |
| Qualified ODC-By review | Unresolved | Exact legal text and workflow questions are captured, but no qualified human review occurred. |
| Immutable capture without corpus acquisition | Satisfied for this public packet | Nineteen public artifacts were captured and hashed; later maintainer evidence must be added before a decision. |
| Compatible terms across selected components and no controlling participation agreement | Unresolved | Partition uniformity, spoof terms, external model terms, and post-challenge rule status are not confirmed. |

“Satisfied for this public packet” is not permission to acquire or evaluate.
It means only that the current public documentation set has an immutable,
reviewable record.

## Questions reserved for qualified review

A qualified reviewer must determine:

- how ODC-By contract rights and its Produced Work notice apply to offline
  evaluation, aggregate reports, and detector artifacts;
- the authority and international effect of the ASVspoof and MLS CC BY grants in
  light of LibriVox jurisdictional cautions;
- rights and contributor grants for TTS, VC, and adversarial output files;
- whether exact attack-model licences governed the released outputs and whether
  downstream evaluation can comply;
- privacy, personality, voice, data-protection, and possible biometric
  obligations for the exact minimized workflow and jurisdictions;
- whether the proposed activity is non-commercial if CPML-governed outputs are
  confirmed, including grants, sponsorship, institutional work, hosted
  services, indirect payment, and future product separation; and
- the legal or methodological status of the historic challenge rules for
  current direct use.

## Maintainer-question draft — not sent

**Proposed recipient:** `info@asvspoof.org`

**Proposed subject:** ASVspoof 5 v0.1 component-rights clarification for a
research-evaluation review

> Hello ASVspoof organisers,
>
> We are conducting a rights review for a narrow, offline Track 1 research
> evaluation. We have not downloaded the corpus, created an account, accepted
> terms, or used the data. We found the v0.1 Zenodo README and licence, the
> ASVspoof 5 design paper, and the Phase 2 v0.6 evaluation plan. Before seeking
> internal approval, could you please clarify the following or point us to
> authoritative public documentation?
>
> 1. What exact content-level licence or permission applies to every TTS, VC,
>    and adversarial spoof audio file, distinct from ODC-By database rights?
> 2. Did each spoof-data contributor grant the organising committee sufficient
>    rights to publish, permit research evaluation of, and where applicable
>    redistribute the generated outputs? Is a public summary of those terms
>    available?
> 3. Which exact versions and checkpoints generated A17, A28, and A29? What
>    model and output terms applied, and were separate licences obtained?
> 4. Which attack IDs used Common Voice 11 or other external datasets, and do
>    any access covenants, attribution duties, or output restrictions flow to
>    the released spoof files?
> 5. Do the same database-level and content-level terms cover all train,
>    development, evaluation, enrolment, codec-processed, and adversarial
>    components? If not, could you provide a component-by-component terms
>    table?
> 6. Does the README statement that bona fide data is CC BY 4.0 cover every
>    bona fide and enrolment file? What exact creator identification,
>    copyright notice, source URI, and attribution wording should users retain?
> 7. For independent research using the current direct Zenodo release, which
>    Phase 2 rules remain binding conditions, which applied only to ranked 2024
>    participation, and which are recommendations for comparability?
> 8. Assuming no audio redistribution, is offline Track 1 evaluation and
>    publication of aggregate metrics permitted? What ODC-By Produced Work
>    notice and scholarly citation do you require?
> 9. What consent, notice, or other rights basis supports using
>    LibriVox-derived recordings as target voices for cloning and distributing
>    those cloned outputs? Are there jurisdictional, withdrawal, identity, or
>    biometric-processing restrictions downstream users should follow?
>
> If possible, a public clarification or an update to the Zenodo README would
> help other researchers perform the same review. Thank you.

No message should be sent until the user reviews and explicitly authorizes the
outbound contact.

## Proposed minimum boundary if a later human review approves

Any later approval should initially be narrower than the complete corpus:

- segregated, non-product Track 1 research evaluation of an existing detector;
- no Track 2 speaker verification;
- no training, fine-tuning, calibration, threshold freezing, product
  evaluation, product use, hosted-service use, or raw redistribution;
- only required audio, trial keys, labels, attack and codec tags needed for
  stratified error analysis, and integrity metadata;
- hashed or remapped speaker IDs only for leakage prevention, with gender and
  unnecessary metadata excluded;
- no identity inference, speaker search, contributor matching, voiceprint
  enrolment, demographic analysis, target-speaker analysis, or reusable speaker
  embeddings;
- controlled raw-data storage, aggregate-only reporting, and no publication of
  clips, transcripts, protocols, or linkable row-level scores; and
- exact notices, deletion controls, incident response, and a fresh rights
  review before any purpose, artifact, partition, jurisdiction, or distribution
  boundary changes.

These are proposed safeguards, not an approval.

## Validation

From the repository root:

```bash
PYTHONPATH=src python scripts/validate_asvspoof5_rights_clarification.py
```

To verify retained source captures as well:

```bash
PYTHONPATH=src python scripts/validate_asvspoof5_rights_clarification.py \
  --capture-directory /path/to/asvspoof5-clarification-captures
```

A valid result must report the packet as non-authorizing, the disposition as
`defer_pending_conditions`, the registry as `planned/pending`, the C3
disposition as deferred, and all action flags as false.
