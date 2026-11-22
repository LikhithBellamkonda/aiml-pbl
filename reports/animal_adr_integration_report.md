# Animal ADR Evidence Integration Report

Generated: 2026-11-22T14:15:47.838923+00:00

Animal spontaneous adverse-event evidence (FDA CVM ADAE with VeDDRA
terminology) is integrated as an independent evidence branch. It is
kept separate from animal experimental toxicity, human ADR evidence and
human pregnancy outcomes. No incidence, causality or pregnancy label is
derived from these reports.

## Acquisition

| Source | Status | Portal |
| --- | --- | --- |
| fda_adae | acquired | https://open.fda.gov/data/downloads/ |
| veddra | acquired | https://www.ema.europa.eu/en/veterinary-regulatory-overview/post-authorisation-veterinary-medicines/pharmacovigilance-veterinary-medicines/eudravigilance-veterinary |
| ema_evvet | manual_access_required | https://www.adrreports.eu/vet/en/index.html |
| uk_vmd | manual_access_required | https://www.data.gov.uk/dataset/ab3eea9d-1dd6-4c61-8b38-7e36e5518d99/suspected-adverse-event-data-submitted-to-the-vmd |

EMA EVVet and UK VMD do not expose a scriptable structured bulk export
in this environment; both are recorded as `manual_access_required` with
explicit local-file ingestion interfaces (`ingest_evvet`, `ingest_vmd`).

## Data

- Raw FDA ADAE reports: 18673
- Cleaned reaction records: 123749
- Unique reports: 18673
- Unique species: 21
- Unique reaction terms: 1807
- Reaction categories: 27
- Reporting period: ['2026-04-01', '2026-06-26']
- Duplicate rows removed: 5490
- Missing reaction descriptions (retained, flagged): 7

## Identifier mapping

- Distinct product/ingredient pairs: 2017
- Directly matched products: 160
- Combination products: 611 (all ingredients matched: 10, partial: 130)
- Ambiguous products: 357
- Unmatched products: 1360
- Unique standardised drugs: 96

## Overlap with existing project evidence

| Measure | Value |
| --- | --- |
| animal_adr_drugs | 96 |
| animal_adr_unmatched_products | 1360 |
| animal_adr_ambiguous_products | 357 |
| animal_adr_combination_products | 611 |
| general_adr_drugs | 685 |
| animal_adr_shared_with_general_adr | 67 |
| pregnancy_drugs | 67 |
| animal_adr_shared_with_pregnancy | 17 |
| toxicity_drugs | 50 |
| animal_adr_shared_with_toxicity | 6 |
| faers_drugs | 75 |
| animal_adr_shared_with_faers | 18 |
| animal_adr_shared_with_all_human_families | 3 |

## Features and model branch

- Drugs with animal ADR features: 95
- Reports contributing to mapped drugs: 793
- Encoder: `src/models/animal_adr_encoder.py` (MLP + species/reaction
  embeddings, zero output when evidence is unavailable).
- Fusion: sixth source in `MaskAwareCrossAttentionGating`, gated by the
  availability mask and evidence quality; 5-source default retained.

## Training and evaluation

- Animal ADR branch: **not trained** (no supervised target exists for
  it; see the evaluation report).
- MEPRank retraining: **not performed**.
- Mask semantics check (A == D mean probability): **PASSED** - masking the missing animal source reproduces the five-source model exactly.
- Reason: No validated human pregnancy-outcome labels (data/pregnancy_labels.csv) are available, so AUROC, AUPRC, Precision@K, Recall@K, Brier and ECE cannot be computed. Animal ADR reports are never used as human outcome labels.
- Ablation A/B/C/D: executed forward passes only; metrics are
  recorded as blocked, not as zero or as an improvement.

## Pipeline status

| Stage | Status |
| --- | --- |
| ingest | ok |
| process | ok |
| animal_adr | ok |
| features | ok |
| feasibility | ok |

## Dashboard

A dedicated **Animal Adverse Drug Reaction Evidence** section reports
availability, sources, species, reaction categories, report counts,
serious reports, completeness, mapping confidence, reporting dates and
limitations, plus an evidence-category distinction table. Drugs without
records show: "Animal ADR evidence unavailable. This does not indicate
that the drug is safe."

## Remaining limitations

- Spontaneous reporting is voluntary and has no exposure denominator;
  counts are not incidence and imply no causality.
- 96 standardised drugs match the project's human drug universe; the
  remaining veterinary products are retained as unmatched records.
- No validated human pregnancy-outcome labels exist, so the animal
  branch's contribution to a pregnancy endpoint is unmeasured.
- EMA EVVet and UK VMD require manual structured exports.
