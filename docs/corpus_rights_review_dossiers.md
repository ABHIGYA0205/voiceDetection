# VoiceShield Corpus Rights-Review Dossiers

> [!WARNING]
> **DISCLAIMER & LEGAL BOUNDARIES**:
> This document and the associated machine-readable dossiers in `configs/corpus_rights_evidence.yaml` contain factual evidence captures, technical provenance records, and compliance risk assessments compiled for research governance. **This document does NOT constitute legal advice or formal legal conclusions.**
>
> SPDX identifiers and source license labels **do not authorize dataset use**. In accordance with VoiceShield Stage C2 governance policy:
> 1. **No corpus has been accessed, downloaded, or acquired.**
> 2. **No dataset terms of use or platform covenants have been accepted.**
> 3. **No accounts have been registered.**
> 4. **No training or detector inference has been performed.**
> 5. **All candidate datasets remain in `data_state: planned` and `rights_decision: pending` with empty `allowed_purposes`.**

- **Evidence As-Of Date**: 2026-11-03
- **Schema Version**: 1.0.0
- **Associated Configuration**: `configs/corpus_rights_evidence.yaml`
- **Governing Code**: `src/voiceshield/data/corpus_rights_evidence.py`

---

## 1. Executive Summary & Governance Model

The VoiceShield framework enforces a fail-closed data governance gate. Before any external acoustic speech dataset may be downloaded, processed, or introduced into experimental pipelines, a formal rights-evidence dossier must be compiled.

Stage C2 establishes the **Rights-Evidence Dossier Layer**, which:
1. **Separates Factual Source Signals from Project Decisions**: Factual disclosures made by upstream creators (e.g., license declarations, non-commercial restrictions, platform terms) are documented as evidence signals (`EvidenceSignal`), strictly distinct from internal project rights decisions (`RightsDecision`).
2. **Segregates Research and Product Tracks**: Clear boundaries are established between non-commercial academic research exploration and commercial product integration.
3. **Documents Privacy, Consent, and Minimization Safeguards**: Identifies personal data risks, voice data considerations, and mandatory data minimization protocols prior to ingestion.
4. **Highlights Unresolved Questions for Human Review**: Every dossier identifies concrete legal, contractual, and technical questions that must be resolved by a qualified human reviewer before any dataset snapshot can be approved.

---

## 2. Research-Track vs. Product-Track Governance Matrix

The following matrix summarizes the factual source evidence signals across all four candidate corpora currently tracked in `configs/corpus_registry.yaml`.

| Corpus ID | Declared SPDX | Access Mode | Research Evaluation Signal | Research Training Signal | Product Evaluation Signal | Product Training Signal | Raw Redistribution Signal | Track Status & Key Restriction |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`asvspoof5`** | `ODC-By-1.0` | `public_direct` | `potentially_compatible_not_reviewed` | `potentially_compatible_not_reviewed` | `potentially_compatible_not_reviewed` | `potentially_compatible_not_reviewed` | `unclear` | **PENDING**: ODC-By §2.2 covers rights in the Database; §2.4 does not grant independent rights in individual Contents other than Database Rights or contract; underlying audio copyright, privacy, and consent remain unreviewed. |
| **`mlaad_v9`** | `CC-BY-NC-4.0` | `account_required` | `potentially_compatible_not_reviewed` | `potentially_compatible_not_reviewed` | **`source_declares_prohibited`** | **`source_declares_prohibited`** | `source_declares_restricted` | **EXCLUDED FROM PRODUCT USE**: Excluded from product work under current evidence by CC BY-NC §1(i) and §2(a)(1); research use pending. |
| **`common_voice_18`** | `CC0-1.0` | `account_required` | `potentially_compatible_not_reviewed` | `potentially_compatible_not_reviewed` | `potentially_compatible_not_reviewed` | `potentially_compatible_not_reviewed` | `source_declares_restricted` | **PENDING**: CC0 operates only for affirmer rights; Common Voice Terms ask not to mirror; MDC Consumer Terms §2(b)(viii) restricts re-identification. |
| **`indicvoices`** | `CC-BY-4.0` | `gated_approval` | `potentially_compatible_not_reviewed` | `potentially_compatible_not_reviewed` | `potentially_compatible_not_reviewed` | `potentially_compatible_not_reviewed` | `unclear` | **PENDING**: Gated approval on Hugging Face; published metadata requires default-deny data minimization to audio and language only. |

*Note: All four corpora have `rights_decision: pending` and `data_state: planned` in the authoritative registry. No evidence signal in the table above confers permission or authorization to download or evaluate audio.*

---

## 3. Individual Corpus Rights-Evidence Dossiers

### 3.1 ASVspoof 5 (Release v0.1)

#### 1. Factual Source Disclosures
* **Official Sources**:
  * Zenodo Record 14498691: `https://zenodo.org/records/14498691` (Title: *"ASVspoof 5: Design, Collection and Validation of Resources for Spoofing, Deepfake, and Adversarial Attack Detection Using Crowdsourced Speech"*, Release `v0.1`, published 2024-12-18, Pinned Record `14498691`)
  * Open Data Commons Attribution License (ODC-By) v1.0: `https://opendatacommons.org/licenses/by/1-0/`
* **Declared License**: `ODC-By-1.0` (Open Data Commons Attribution License v1.0).
* **Access Mode**: `public_direct`. The official Zenodo record presents direct file-download links without mandatory account registration. No platform or dataset terms have been accepted by the project.
* **Precise Database-versus-Contents Distinction**: Under ODC-By-1.0 Section 2.2, the license covers copyright and neighbouring rights in the defined Database, Database Rights, and contract. Crucially, Section 2.4 states that the license does not cover rights in individual Contents other than Database Rights or rights arising in contract, leaving underlying speech audio copyright, privacy, and consent unaddressed by the license.

#### 2. Track Compatibility Signals
* **Research Evaluation & Training**: `potentially_compatible_not_reviewed`. ODC-By grants rights to Use the Database, including extraction under Section 3.1, but Section 2.4 does not grant independent rights in individual speech Contents other than Database Rights or rights arising in contract; content-level rights have not been verified.
* **Product Evaluation & Training**: `potentially_compatible_not_reviewed`. Section 3.1 expressly includes commercial Use of the Database, subject to the license conditions. Section 4.2 governs notices for public conveyance and Section 4.3 separately governs the notice for Publicly Used Produced Works. Commercial readiness cannot be approved without content-level clearance.
* **Raw Redistribution**: `unclear`. ODC-By allows public conveyance of the Database subject to Section 4.2, but Section 2.4 does not grant independent rights in individual Contents other than Database Rights or contract; content-level rights remain unresolved.
* **Attribution**: `source_declares_restricted`. Section 4.2 requires specified license and rights notices when Publicly Conveying the Database, a Derivative Database, or a Collective Database; Section 4.3 separately requires a database-origin notice for a Publicly Used Produced Work. Both are distinct from scholarly citation requests.

#### 3. Privacy, Obligations & Data Minimization
* **Data Subject Considerations**: Audio clips capture human voice recordings across authentic speakers and simulated/spoofed attack attempts. These recordings raise data-protection, privacy, and consent considerations without asserting definitive biometric classifications. ODC-By-1.0 explicitly disclaims patent, trademark, and privacy/publicity grants.
* **Data Minimization**: VoiceShield pipelines must ingest only audio waveforms, anonymous speaker identifier hashes, and trial evaluation keys. Challenge participant demographic and extraneous metadata fields must not be ingested.

#### 4. Unresolved Questions for Legal & Review Gate
1. *Database vs. Contents Rights*: Under ODC-By-1.0 Section 2.2, rights cover the defined Database, Database Rights, and contract, while Section 2.4 does not cover rights in individual Contents other than Database Rights or rights arising in contract. Do dataset creators hold explicit copyright clearance or licenses for underlying speech audio Contents?
2. *Attribution vs. Citation Requirements*: What exact Section 4.2 notices apply to public conveyance, what Section 4.3 database-origin notice applies to Publicly Used Produced Works, and what separate scholarly citation is merely requested by the publication record?
3. *Produced Work Status*: Does a trained deepfake detector model checkpoint or an evaluation score file constitute a "Produced Work" under ODC-By Section 4.3?
4. *Component Uniformity*: Do all challenge subsets (training, development, and evaluation partitions) share identical legal terms, or do certain subsets carry separate constraints?

---

### 3.2 MLAAD v9 (Multi-Language Audio Anti-Spoofing Dataset)

#### 1. Factual Source Disclosures
* **Official Sources**:
  * Project Website: `https://deepfake-total.com/mlaad`
  * Hugging Face Dataset Card: `https://huggingface.co/datasets/mueller91/MLAAD` (Pinned commit: `30c3dec763fa4803f11c2df1557d3b0828395366`)
  * Creative Commons CC BY-NC 4.0: `https://creativecommons.org/licenses/by-nc/4.0/`
* **Official Release Label**: Version 9 (`v9`) is recorded strictly as an official source release label, not an acquired snapshot version reference.
* **Declared License**: `CC-BY-NC-4.0` (Creative Commons Attribution-NonCommercial 4.0 International).
* **Access Mode**: `account_required`. Access is authenticated through Hugging Face user login.
* **Explicit Non-Commercial Scope & Limitation**: Section 1(i) of CC BY-NC 4.0 defines NonCommercial as "not primarily intended for or directed towards commercial advantage or monetary compensation." Section 2(a)(1) limits the licence grant strictly to NonCommercial purposes.
* **External Bona Fide Dependency**: MLAAD provides synthetic speech samples paired against bona fide reference audio from the **M-AILABS** speech corpus. M-AILABS is an external bona fide dependency requiring separate component-level provenance and rights review before any pairing.

#### 2. Strict Research / Product Track Segregation
* **Product Evaluation**: **`source_declares_prohibited`**
* **Product Training**: **`source_declares_prohibited`**
  > [!CAUTION]
  > **PRODUCT TRACK EXCLUSION**:
  > MLAAD v9 is excluded from product evaluation and product training under current evidence pending clarification of commercial and redistribution terms. Section 1(i) defines NonCommercial and Section 2(a)(1) limits the licence grant strictly to NonCommercial purposes. No code or model destined for commercial deployment may consume MLAAD v9 data unless a separate commercial license is negotiated and executed with copyright holders.
* **Research Evaluation & Training**: `potentially_compatible_not_reviewed`. Non-commercial academic research benchmarking may be permitted under Section 2(a)(1) if rights review approves it, subject to complete isolation from commercial weights.
* **Raw Redistribution**: `source_declares_restricted`. Sharing must remain NonCommercial under Section 2(a)(1), and Section 3(a) requires retention of supplied creator identification and specified notices, a license reference, and indication of modifications.
* **Attribution**: `source_declares_restricted`. Section 3(a) requires retention of creator identification and other attribution information supplied with the Licensed Material, specified notices, a license reference, and indication of modifications when Shared; this is distinct from academic paper citation.

#### 3. Privacy, Obligations & Data Minimization
* **Data Subject Considerations**: Synthetic speech generated from neural text-to-speech (TTS) and voice conversion (VC) architectures mimicking human vocal characteristics. Reference bona fide voices derive from M-AILABS, an external bona fide dependency requiring separate component-level provenance and rights review.
* **Data Minimization**: Ingest only synthetic audio clips, language identifiers, and generator architecture tags. Exclude extraneous text transcripts and metadata.

#### 4. Unresolved Questions for Legal & Review Gate
1. *M-AILABS Pairing & Licensing*: What are the separate component-level provenance records, license terms, and rights review requirements for the M-AILABS corpus before it can be paired with MLAAD in scientific evaluations?
2. *Commercial License Availability*: Is a separate commercial license or waiver obtainable from Fraunhofer AISEC for product benchmarking or commercial detector validation?
3. *Scope of NonCommercial*: Does public academic benchmarking conducted by commercial or corporate research affiliates qualify as non-commercial research under CC-BY-NC-4.0 Section 1(i) and Section 2(a)(1)?
4. *Platform Obligations*: Do Hugging Face repository hosting policies impose downstream indemnity or redistribution restrictions?

---

### 3.3 Mozilla Common Voice 18 (Release 18.0)

#### 1. Factual Source Disclosures
* **Official Sources**:
  * Release Announcement (June 2024): `https://www.mozillafoundation.org/en/blog/common-voice-18-dataset-release/` (Release `18.0`)
  * Common Voice Terms of Use (Effective 2025-10-31): `https://commonvoice.mozilla.org/en/terms`
  * Mozilla Data Collective (MDC) Consumer Terms (Last updated 2026-10-28): `https://mozilladatacollective.com/terms/consumers`
  * Creative Commons CC0 1.0 Universal Legal Code: `https://creativecommons.org/publicdomain/zero/1.0/legalcode.en`
* **Declared License**: `CC0-1.0` (Creative Commons CC0 1.0 Universal Public Domain Dedication).
* **Access Mode**: `account_required`. Direct downloads require registering an authenticated user account through the Mozilla Data Collective (MDC) portal.
* **Limited Effect of CC0 & Contractual Terms Distinction**:
  * **CC0 Limited Effect**: CC0 operates only for rights held by the affirmer and only to the extent legally waivable or licensable. It does not itself establish contributor consent, privacy clearance, contractual authorization, or clearance of third-party rights.
  * **Common Voice Terms of Use**: These terms only ask users **not to post, distribute, or mirror datasets**. They do not contain a re-identification covenant.
  * **MDC Consumer Terms**: MDC platform terms govern download portal access. Mirroring and redistribution are restricted except as permitted under the applicable Data Consumer License. Crucially, Section 2(b)(viii) separately prohibits attempting to determine, trace, match, or re-identify the individual contributor or source of data contained in a Dataset.

#### 2. Track Compatibility Signals
* **Research Evaluation & Training**: `potentially_compatible_not_reviewed`. CC0 operates only for affirmer rights, while contractual covenants accepted upon download via MDC impose binding covenants that require legal review.
* **Product Evaluation & Training**: `potentially_compatible_not_reviewed`. CC0 does not restrict commercial evaluation, but CC0 does not clear third-party rights or consent, and MDC Consumer Terms accepted at download time require contractual analysis for product use.
* **Raw Redistribution**: `source_declares_restricted`. Common Voice Terms ask users not to post, distribute, or mirror datasets, and MDC Consumer Terms restrict mirroring and redistribution except as permitted under the applicable Data Consumer License.
* **Attribution**: `potentially_compatible_not_reviewed`. Legally waived under CC0 to the extent of affirmer rights, but moral rights and academic attribution conventions encourage citation.

#### 3. Privacy, Obligations & Data Minimization
* **Data Subject Considerations**: Voluntary crowd-sourced human speech recordings across diverse global languages. MDC Consumer Terms Section 2(b)(viii) contractually prohibits attempting to determine, trace, match, or re-identify the individual contributor or source of data contained in a Dataset. CC0 dedication does not itself establish privacy or consent clearance.
* **Data Minimization**: Ingest strictly audio clips, anonymized client tokens, and verified language/accent tags. Never ingest optional demographic survey responses or perform voice re-identification.

#### 4. Unresolved Questions for Legal & Review Gate
1. *MDC Terms vs. CC0 Reconciliation*: How do MDC Consumer Terms (restricting mirroring except under applicable Data Consumer Licenses) interact with the declared CC0-1.0 dedication, which operates only for rights held by the affirmer?
2. *Binding Re-Identification Covenant*: How will VoiceShield pipelines strictly enforce the binding anti-re-identification restriction under MDC Consumer Terms Section 2(b)(viii) during acoustic modeling and speaker holdout evaluation?
3. *Version-Specific Terms Temporal Alignment*: Terms in force at future access time (e.g., MDC Consumer Terms updated 2026-10-28 and CV Terms effective 2025-10-31) must be reconciled with terms active at June 2024 release.
4. *Commercial Product Deployment*: Does downstream commercial product distribution risk violating platform access covenants despite CC0 dataset copyright status?

---

### 3.4 AI4Bharat IndicVoices

#### 1. Factual Source Disclosures
* **Official Sources**:
  * Hugging Face Dataset Card: `https://huggingface.co/datasets/ai4bharat/IndicVoices` (Pinned commit: `c96f9088f138cf89d419da7e8e643e1f05c00a87`)
  * Academic Publication (arXiv:2403.01926): `https://arxiv.org/abs/2403.01926` (Version `2403.01926v1`, submitted 2024-03-04)
  * Creative Commons CC BY 4.0: `https://creativecommons.org/licenses/by/4.0/`
* **Declared License**: `CC-BY-4.0` (Creative Commons Attribution 4.0 International).
* **Access Mode**: `gated_approval`. Access is gated on Hugging Face, requires authentication, and publicly asks the user to share Hugging Face user information and review and accept repository conditions. Unseen conditions remain unresolved.
* **Research Paper Methodology vs. Distributed Schema**: The paper states that registration captured names, phone numbers, and an uploaded signed consent form, while participant instructions state that registration information would not be shared with third parties. It separately states that verification videos could not be downloaded or shared and were deleted after verification. The published dataset card schema contains audio, transcripts, demographic/geographic fields, a `verification_report` field, and unsanitized transcript fields; no assumption is made that those distributed fields are free of incidental personal data.
* **Commercial Permission Claims**: The research paper explicitly asserts CC-BY-4.0 commercial use permissions.

#### 2. Track Compatibility Signals
* **Research Evaluation & Training**: `potentially_compatible_not_reviewed`. Permitted under CC-BY-4.0 subject to attribution, but gated repository access conditions and privacy compliance have not been cleared.
* **Product Evaluation & Training**: `potentially_compatible_not_reviewed`. CC-BY-4.0 permits commercial exploitation as cited in the paper, but gated repository conditions must be reviewed.
* **Raw Redistribution**: `unclear`. CC-BY-4.0 allows redistribution, but Hugging Face gated repository controls may imply distribution limits.
* **Attribution**: `source_declares_restricted`. Section 3(a) requires retention of creator identification and other attribution information supplied with the Licensed Material, specified notices, a license reference, and indication of modifications when Shared; this is distinct from scholarly citation of the arXiv paper.

#### 3. Privacy, Obligations & Data Minimization
* **Data Subject Considerations**: Human speech from diverse linguistic and socio-economic demographics in India. While informed consent is documented by authors and verification videos were deleted after verification, the presence of demographic and geographic fields in the published card creates privacy considerations.
* **Default-Deny Data Minimization Safeguards**:
  * **Audio & Language Only**: Ingest strictly raw audio waveforms, anonymous speaker identifier hashes, language codes, and dialect tags.
  * **Default-Deny Filtering**: Demographic attributes (gender, age group) and geographic origin (district, state) are denied and excluded by default.
  * **Protocol Requirement for Demographic/Geographic Use**: Any future demographic or geographic use would require a separate, purpose-specific, human-approved protocol with minimization, access controls, retention limits, and documented review. **No such exception is approved in Stage C2.**
  * **Strict Data Sanitization**: Prohibit copying unsanitized text transcripts, direct identifiers, or extraneous metadata into VoiceShield manifests or repositories.

#### 4. Unresolved Questions for Legal & Review Gate
1. *Gated Access Conditions*: What specific legal terms, restrictions, or covenants are incorporated into the Hugging Face gated access conditions that users must review and accept?
2. *Demographic and Geographic Protocol Governance*: Because demographic and geographic fields are denied by default, what specific human-approved governance protocol, minimization controls, and retention limits would be required if demographic analysis were ever proposed in the future?
3. *Attribution vs. Citation Requirements*: What exact legal copyright notice is required under CC-BY-4.0 versus academic citation of the arXiv 2403.01926 paper?
4. *Commercial Rights Under Gated Access*: Does the AI4Bharat consortium intend to permit commercial product integration under CC-BY-4.0 despite gating access behind repository conditions?

---

## 4. Governance Decision Boundaries

1. **No Legal Conclusions**: The evidence signals recorded in these dossiers reflect technical observations of upstream disclosures. They do not represent legal advice.
2. **Pending State Preserved**: All four datasets remain in `rights_decision: pending` and `data_state: planned`.
3. **Requirement for Qualified Human Review**: Before any candidate dataset transitions from `planned` to `acquired` or `verified`:
   * An authorized project rights reviewer must review the complete dossier.
   * All unresolved questions must be resolved in writing.
   * An explicit `rights_decision` (`approved` or `restricted`) must be recorded in `configs/corpus_registry.yaml` with the reviewer's identity and review date.
   * The raw archive must be acquired and immediately hashed (`archive_sha256`) before unpacking.
   * The canonical manifest must be hashed (`manifest_sha256`) and the exact sample count recorded.
4. **Product Pipeline Exclusion**: MLAAD v9 is excluded from product pipelines under current evidence unless a separate commercial license is obtained and verified.
