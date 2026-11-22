# Evidence-Aware Multimodal Learning for Trimester-Specific Prioritization of Pregnancy Drug-Safety Signals

This repository contains an evidence-processing prototype for prioritizing potential
pregnancy-stage-specific drug safety signals. It is not a clinical decision-support
or prescribing system. All outputs are exploratory research signals requiring
clinical and epidemiological confirmation.

## Reframed Problem Statement

**Evidence-aware prioritization of potential pregnancy-stage-specific safety signals.**

Instead of predicting binary pregnancy outcomes (which is not possible with current
data — no validated human pregnancy labels exist), the system:

1. **Prioritizes drug–ADR signals** by trimester (T1, T2, T3, Entire Pregnancy)
2. **Separates prediction confidence from evidence reliability** — a model can be
   confident but working with poor-quality evidence, or uncertain but working with
   rich evidence
3. **Flags evidence gaps** — drugs where data is too sparse for any reliable prediction
4. **Provides trimester-aware risk** — recognises that teratogenicity is most relevant
   in T1 (organogenesis), while growth and functional effects may be more relevant later

## What Is Implemented

| Component | Status | Description |
|-----------|--------|-------------|
| Human-only evidence pipeline | ✅ | OnSIDES, Mendeley, TGA, FAERS, PubChem; animal records never become human outcome labels |
| Animal ADR evidence branch | ✅ | FDA CVM ADAE (VeDDRA terms) as a separate, gated sixth evidence source |
| Drug lookup by name or identifier | ✅ | Recorded name, project `drug_id`, PubChem CID or InChIKey; ambiguity is surfaced |
| Availability Mask (M) & Quality Vector (Q) | ✅ | Per-drug, per-source evidence availability and provenance completeness |
| Q_cov coverage score | ✅ | Evidence coverage = mean(M × Q) across source families |
| Mask-aware cross-attention gating | ⚠️ | Architecture implemented; weights are untrained (no labels, see below) |
| Scientific safeguard | ✅ | Training blocked without validated human pregnancy labels |
| Evidence quality (separate from prediction) | ✅ | M × Q evidence reliability, never presented as confidence |
| Evidence gap flag | ✅ | Explicit "understudied or single-source drug" warning |
| Evidence-aware abstention messages | ✅ | Dynamic messages based on the actual missing evidence |
| Per-trimester evidence availability | ✅ | T1/T2/T3/entire pregnancy coverage report; no risk value is produced |
| Reported side-effect terms | ✅ | Observed terms with report counts and High/Moderate/Low volume levels |
| Streamlit dashboard | ✅ | Search, 2D structure, identity, per-source evidence panels, animal ADR section, provenance badges |
| Dataset feasibility & overlap reports | ✅ | Generated from the current pipeline run |
| Evidence-tiered coverage inventory | ✅ | 16 literature controls, measured source counts (`reports/pregnancy_evidence_inventory.*`) |
| Trimester metric definitions | ⚠️ | AUROC/AUPRC/ΔAUROC/ECE per trimester implemented; blocked without labels |
| Case studies | ✅ | docs/CASE_STUDIES.md (evidence availability, not predictions) |
| Unit tests | ✅ | pytest suite |
| ROR/PRR disproportionality ratios | ❌ | No case/non-case denominators exist in the acquired aggregates |
| Trained pregnancy model, risk score, confidence | ❌ | Blocked: no validated human pregnancy-outcome labels |
| AUROC / AUPRC / calibration | ❌ | Blocked; no labels means no metric, not a metric of zero |
| Curated benchmark file | ❌ | Not present in the repository |

## Animal ADR Evidence Branch

Veterinary spontaneous adverse-event reports are integrated as an
**independent evidence branch**, never as human pregnancy labels:

```bash
python -m src.animal_adr.pipeline          # acquire -> clean -> map -> features
python -m src.animal_adr_experiments       # ablation A/B/C/D + reports
python -m src.run_pipeline                 # full pipeline (includes animal_adr)
```

- **Source**: FDA CVM Animal & Veterinary Adverse Event export
  `datasets/animalandveterinary-event-*.json.zip` (18,673 reports, VeDDRA-coded
  reactions), with the EMA VeDDRA terminology workbook for System Organ Class
  categorisation. EMA EVVet and UK VMD have no scriptable structured bulk
  export here and are recorded as `manual_access_required`, with explicit
  local-file ingestion functions (`ingest_evvet`, `ingest_vmd`).
- **Outputs**: `data/animal_adr_acquisition_manifest.json`,
  `data/animal_adr_dataset_summary.json`, `data/animal_adr_overlap.csv`,
  `data/animal_adr_data_quality_report.json`, `data/animal_adr_features.csv`,
  `data/cleaned/animal_adr_{cleaned,mapped,unmatched,drug_mapping,
  species_summary,reaction_summary}.csv`, and
  `reports/animal_adr_{integration_report.md,ablation_results.csv,
  evaluation_report.json}`.
- **Model**: `src/models/animal_adr_encoder.py` (MLP with species and reaction
  embeddings) enters `MaskAwareCrossAttentionGating` as a sixth source. The
  five-source architecture remains the default, so existing artefacts load
  unchanged; `MultimodalPregnancyADRModel(include_animal_adr=True)` enables it.
- **Missing evidence**: a drug without animal records has mask 0 and the branch
  emits exact zeros; fusion excludes it. The dashboard says "Animal ADR
  evidence unavailable. This does not indicate that the drug is safe."
- **Safeguards**: animal spontaneous reports are kept distinct from animal
  developmental/reproductive toxicity, human ADR evidence and human pregnancy
  outcomes. Counts are not incidence, imply no causality, and are never used as
  pregnancy labels. No animal branch or MEPRank weights are claimed as trained
  without validated labels (see `reports/animal_adr_evaluation_report.json`).

## What Is NOT Implemented (and why)

| Feature | Status | Reason / Alternative |
|-----------|--------|---------------------|
| Dirichlet evidential deep learning | ❌ | Requires labeled training data (blocked). Alternative: evidence coverage score only |
| Multi-task learning neural network | ❌ | No labeled human pregnancy outcomes (blocked by design). Alternative: observed-evidence summary |
| SHAP / Integrated Gradients | ❌ | No trained GNN + molecular graph pipeline. Alternative: side effect terms + provenance badges |
| Expert validation (clinician) | ❌ | No experts available. Alternative: case studies with known teratogens |
| External validation (SIDER, DailyMed) | ❌ | Not available locally. Alternative: real FAERS/OnSIDES in pipeline |
| Drug-disjoint / temporal splits | ❌ | No labeled data. Alternative: deterministic hash-based split |
| RAG / literature branch | ❌ | No literature corpus available. Alternative: PubChem + ADR evidence |
| DMPNN / AttentiveFP | ❌ | Needs graph training data + labels. Alternative: basic MolecularGNN |
| Public benchmark release | ❌ | No curated benchmark file is present in the repository |
| Conflict-resolution module | ⚠️ | No labeled contradictory pairs. Alternative: conflict detection heuristic implemented |
| Mechanism-aware explanations | ❌ | No mechanism database. Alternative: SMILES-based features |
| Prospective evaluation | ❌ | No temporal outcome data. Not feasible |

## Installation & Usage

```bash
# Install pinned dependencies (CPU-only PyTorch build)
python -m venv .venv
.venv/bin/pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt

# Full pipeline: ingest -> process -> animal_adr -> features -> feasibility
.venv/bin/python -m src.run_pipeline

# Animal ADR branch on its own, then the A/B/C/D mask ablation
.venv/bin/python -m src.animal_adr.pipeline
.venv/bin/python -m src.animal_adr_experiments

# Blocked-evaluation record for the evidence engine
.venv/bin/python -m src.ml_risk_model

# Evidence-tiered coverage inventory for 16 literature control drugs
.venv/bin/python -m src.evidence_inventory

# Trimester metrics + three-signal report (records blocked status without labels)
.venv/bin/python -m src.trimester_evaluation

# Test suite
.venv/bin/python -m pytest -q

# Dashboard (open http://localhost:8501)
.venv/bin/python -m streamlit run dashboard/app.py
```

On Windows use `.venv\Scripts\python.exe` in place of `.venv/bin/python`.
Training commands (`python -m src.train`, `python -m src.train_reproductive_tox`)
raise a documented error until validated labels / the source workbook exist.

## Key Outputs

For any drug query, the system returns **observed evidence and its coverage**, never a
pregnancy risk prediction:

- **Standardized identity** — recorded name, project `drug_id`, PubChem CID,
  InChIKey, SMILES and the mapping status (resolved / unresolved / name-keyed /
  combination not resolved). No identifier is invented for an ambiguous match.
- **Per-source evidence** — human pregnancy-specific records, general ADR terms,
  FAERS aggregate reports, nonclinical developmental/reproductive toxicity
  studies, and animal spontaneous ADR reports, each in a separate panel with
  its own provenance.
- **Evidence quality (Q_cov)** — availability mask × provenance completeness.
  This is a coverage measure, not confidence and not risk.
- **Evidence gap flag** — True when fewer than two source families cover the drug
  or coverage is very low.
- **Missing-source list** — every tracked source without a record is named, so an
  absent record is never read as a negative outcome.
- **Side-effect terms** — observed ADR terms with report counts and a relative
  report-volume level. Counts are not incidence and imply no causality.
- **Prediction status** — "Unavailable", with the reason: there is no validated
  human pregnancy-outcome model. No risk score, confidence value or trimester
  risk number is emitted anywhere in the current build.
- **Observed pregnancy-outcome records** — per-outcome record counts or
  "no record", shown separately from any risk statement.
- **Trimester evaluation report** — `reports/trimester_evaluation_report.*`
  specifies per-trimester AUROC/AUPRC, ΔAUROC, ECE, Brier and Precision@10 and
  states that they are blocked, not zero, until labels exist.

### Dashboard layout

The dashboard resolves a query by recorded name, project `drug_id`, PubChem
CID or InChIKey and shows, in order:

1. **Standardized identity** — CID, InChIKey, mapping status and a locally
   drawn 2D structure (RDKit 2D coordinates rendered with matplotlib, so it
   works without a graphics stack). No structure is invented when SMILES is
   absent.
2. **Evidence availability by source** — measured record counts per source and
   the missing-source list.
3. **Evidence panels** — general ADR terms, molecular descriptors, nonclinical
   developmental/reproductive toxicity studies, human pregnancy-specific
   records, FAERS summary.
4. **Filtered evidence query** — trimester + fetal/maternal filter, Q_cov,
   prediction status (always "Unavailable" with the reason), observed
   pregnancy-outcome records per outcome (or "no record"), and per-trimester
   evidence availability.
5. **Reported side effects** — observed terms, record counts and a documented
   report-volume band (High ≥10, Moderate 3–9, Low <3). The earlier
   count/10 "index", which showed 20% for almost every term, has been removed.
6. **Animal ADR evidence** — separate veterinary-report panel.

### Prediction status in this build

Earlier revisions of `src/ml_risk_model.py` produced a "foetal risk" number from a
SHA-256 hash of the drug identifier plus hard-coded trimester constants, a
hash-perturbed "confidence", and an accuracy/precision/recall evaluation against
pseudo-labels built from the same feature columns. That path has been removed:
it was neither a trained model nor a valid evaluation. The module is now a
`PregnancyEvidenceEngine` that reports observed counts and evidence coverage and
explicitly abstains from scoring. `data/ml_risk_metrics.json` records
`status: blocked` with null metrics.

## Important Limitations

- Spontaneous reporting data (FAERS, OnSIDES) has stimulated reporting, under-reporting,
  and confounding by indication. Associations are not causal.
- No validated human pregnancy outcome labels exist; training is blocked by design.
- All scores are exploratory and not clinically validated.
- This is a screening/prioritization tool, not a clinical decision tool.

## Original Setup Instructions


## Process available datasets

Run from the project root:

```powershell
python -m src.process_real_datasets
```

This resolves medication names through PubChem PUG REST, caches responses in `data/pubchem_cache.json`, maps SIDER identifiers to actual PubChem CIDs, and extracts aggregate outcome tables from the supplied JAAD/FAERS supplement. Combination products are not resolved as single ingredients. Failed or ambiguous lookups remain explicit; they are not assigned replacement CIDs. This human-outcome processing path does not read study-chemical assay workbooks.

To avoid network access, process only cached and bundled chemical records:

```powershell
python -m src.process_real_datasets --offline
```

Limit new PubChem lookups while testing:

```powershell
python -m src.process_real_datasets --max-pubchem-queries 25
```

## Pregnancy dataset feasibility

Run the local source audit from the project root:

```powershell
python -m src.dataset_feasibility
```

The command writes `reports/pregnancy_dataset_feasibility.md` and the machine-readable
`data/dataset_feasibility.json` used by the dashboard. The report inventories the
Mendeley dermatology FAERS supplement and the cited Dryad antiepileptic-study
supplement, reports source-name/CID matching and missingness, and assesses each outcome
without treating unreported events as negatives.

The Mendeley supplement contains drug-level aggregate spontaneous-report term counts,
not patient-level outcomes or exposed-pregnancy denominators. The Dryad deposit is a
supplementary DOCX with aggregate cohort characteristics, diagnostic codes, and
study-level ASD/ADHD analyses; it does not contain patient-level outcomes. The local
pregnancy database is a classification catalog, not an outcome-label dataset. No
endpoint is currently trainable, and model training must remain blocked.

## Reproducible dataset integration

Run the source-specific ingestion, identifier standardization, and feasibility pipeline
from the project root:

```powershell
python -m src.acquire_datasets
```

By default, the command works from local source files and cached PubChem mappings; it
does not issue unbounded network requests. To acquire missing pinned source files and
allow a capped number of PubChem lookups:

```powershell
python -m src.acquire_datasets --download-missing --max-pubchem-queries 25 --verbose
```

The pipeline ingests OnSIDES v3.1.1, Mendeley Version 2, the verified Dryad deposit
(DOI `10.5061/dryad.2z34tmpk0`), PubChem, and ReproTox-KG. Study-chemical toxicity
sources are excluded from the master evidence and prediction features. The pipeline
records source versions, URLs, checksums, parse/acquisition
errors, normalized drug names, original source identifiers, PubChem matches, outcome
terms, evidence granularity, trimester, and missingness. OnSIDES RxNorm relations
expose generic ingredient names; single-ingredient products can match PubChem through
that relation, while multi-ingredient products remain intact and their ingredients are
retained as a list rather than split into outcome records. Salt names are not stripped.
Raw OnSIDES and ReproTox downloads stay under the
git-ignored `datasets/raw/` directory. The Dryad public API can require an authenticated
file download; the checked-in pipeline accepts the local file only when its checksum
matches the verified DOI record.

Generated outputs include:

- `data/cleaned/`: source-specific cleaned tables, including separate Mendeley outcomes
  and aggregate report totals.
- `data/standardized_master_evidence.csv`: source-linked evidence records; it does not
  add outcome labels.
- `data/drug_identifier_mapping.csv` and `data/standardized_drug_master.csv`.
- `data/evidence_availability_matrix.csv` and `data/outcome_evidence_matrix.csv`.
- `data/dataset_summary.csv`, `data/drug_overlap.csv`, `data/outcome_summary.csv`,
  `data/trimester_summary.csv`, `data/missingness_summary.csv`, and
  `data/identifier_matching_summary.csv`, plus `data/source_registry.csv`.
- `data/dataset_acquisition_manifest.json`.
- `reports/dataset_integration_feasibility.md` and coverage plots in
  `reports/figures/` for source coverage, pregnancy evidence, trimester data,
  identifier matching, and missingness.

OnSIDES contributes regulatory product-label associations; its `pred0`/`pred1` values
are retained as source scores, not interpreted as confidence. ReproTox-KG links remain
knowledge-graph associations and are not human outcome observations. Neither source
is used as a human pregnancy risk target. Check upstream dataset terms before
redistributing raw source data.

## Filtered evidence dashboard

Run the Streamlit dashboard from the project root with
`python -m streamlit run dashboard/app.py`. The Query Filter Control Panel selects one
medication, one timing window, and either fetal or maternal outcomes. Evidence records
and their provenance are filtered to those selections; trimester timing is never
inferred from an untimed record. `Q_cov` is the fraction of tracked pregnancy-evidence
source families represented in the filtered evidence, not a risk or quality score.
No pregnancy outcome is currently trainable, so risk scores and model variance
(`sigma`) remain unavailable; the dashboard abstains rather than substituting
unvalidated values.

Key processed outputs include:

- `data/drug_mapping.csv`: original and normalized names, PubChem identifiers, InChIKey, SMILES, and resolution status.
- `data/real_adr_evidence.csv` and `data/real_faers_adr.csv`: general SIDER terms and model features.
- `data/real_pregnancy_outcomes.csv` and `data/real_pregnancy_features.csv`: source-reported aggregate outcomes, not patient-level observations.
- `data/real_faers_summary.csv`: report totals for drugs explicitly listed in the supplied supplement.
- `data/source_availability.csv`: source availability per drug.
- `data/pregnancy_classifications.csv`: pregnancy classifications kept separate from observed outcomes.

Fusion receives source availability masks plus source-specific metadata-completeness features. These completeness indicators summarize recorded provenance fields; they are not validated scientific confidence scores.

## Tests, training, and dashboard

```powershell
python -m pytest -q
python -m src.train
streamlit run dashboard/app.py
```

The available sources do not provide validated binary human pregnancy-outcome labels with defensible negative examples. Training therefore remains blocked rather than learning from classifications, aggregate report counts, or fabricated labels. The Mendeley supplement supplies aggregate spontaneous-report terms, not case-level exposed and comparator counts. For that reason a valid 2x2 table for ROR/PRR cannot be constructed, and no such ratios are emitted. TGA pregnancy categories are classifications, not outcome labels; general adverse-effect term associations do not supply a compatible comparator denominator. No continuous fetal or maternal risk target, predictive score, or model confidence can be trained from these data.

Before training, obtain endpoint-specific human pregnancy cohort or case-level pharmacovigilance data with validated exposure/outcome definitions, report-level denominators and comparators, documented exposure timing, linkable drug identifiers, and provenance. Preserve all unknown outcomes as missing. Keep drug/scaffold groups disjoint across training and testing and evaluate only targets with adequate validated support. The dashboard continues to distinguish software validation and evidence availability from model evaluation and prediction uncertainty; it withholds risk scores when no validated model exists.

## Important limitations

- The included pregnancy-outcome supplement reports aggregate spontaneous-report counts; counts are not incidence, patient-level labels, or causal estimates.
- An absent source or report is not evidence that a drug is safe.
- General ADR term associations, pregnancy classifications, and human pregnancy outcomes are distinct evidence types.
- Monte Carlo dropout variance is an uncertainty signal, not a calibrated probability interval.
- Synthetic data generation is isolated under `tests/fixtures/generated` and must not be used for real-data training or validation.
