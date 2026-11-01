# Pregnancy drug-safety research prototype

This repository contains an evidence-processing prototype for drug safety research in pregnancy. It is not a clinical decision-support or prescribing system.

## Windows PowerShell setup

From the project directory:

```powershell
cd "D:\aiml final\aiml-pbl"
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For environments that cannot use Python 3.14, use a Python version supported by the installed PyTorch and RDKit wheels.

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
