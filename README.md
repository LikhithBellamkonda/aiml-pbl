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

This resolves medication names through PubChem PUG REST, caches responses in `data/pubchem_cache.json`, maps SIDER identifiers to actual PubChem CIDs, extracts the aggregate outcome tables present in the supplied JAAD/FAERS supplement, and retains exact-name-linked ToxRefDB study records. Combination products are not resolved as single ingredients. Failed or ambiguous lookups remain explicit; they are not assigned replacement CIDs.

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
Mendeley dermatology FAERS supplement, checks for the Dryad antiepileptic outcome
dataset, reports source-name/CID matching and missingness, and assesses each outcome
without treating unreported events as negatives.

In the current workspace, the Mendeley supplement contains drug-level aggregate
spontaneous-report term counts, not patient-level outcomes or exposed-pregnancy
denominators. The Dryad antiepileptic outcome file is not present. The local pregnancy
database is a classification catalog, not an outcome-label dataset. No endpoint is
currently trainable, and model training must remain blocked.

Key processed outputs include:

- `data/drug_mapping.csv`: original and normalized names, PubChem identifiers, InChIKey, SMILES, and resolution status.
- `data/real_adr_evidence.csv` and `data/real_faers_adr.csv`: general SIDER terms and model features.
- `data/real_toxicity_evidence.csv` and `data/real_toxicity.csv`: ToxRefDB study records and study-type counts, not toxicity-positive labels.
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

The current bundled sources do not provide validated binary human pregnancy-outcome labels with defensible negative examples. Training therefore stops with an explicit explanation rather than learning from classifications, aggregate report counts, or fabricated labels. The Mendeley outcome counts must not be used as both labels and input features. Before training, obtain a documented outcome-specific human cohort or controlled study with outcomes and non-outcomes, suitable follow-up and exposure/comparator denominators, validated drug identifiers, endpoint definitions, and provenance. Add eligible labels to `data/pregnancy_labels.csv` with one row per `drug_id`, `label_<outcome>` and `mask_<outcome>` columns for the supported outcomes (`fetal_death`, `preterm_birth`, `fetal_growth_restriction`, `congenital_anomaly`), plus `source_dataset`, `source_reference`, `outcome_definition`, and `evidence_type`. Masks must be `1` only for verified binary human outcome labels; unknown outcomes must remain masked. Supported evidence types are `human_observed_outcome` and `human_controlled_outcome`. Keep drug/scaffold groups disjoint across training and testing, and validate both classes for each outcome before interpreting metrics.

### Exploratory nonclinical toxicity training

The optional `datasets/Supplementary_S3_Raw data.xlsx` workbook is not bundled. The trainer requires that file at this path and reads five binary molecular assay targets (`TG414`, `TG416`, `TG421t`, `TG422t`, `TG443t`) from its `*_smiles4` sheets. The source citation and label definitions were not established in this project, so obtain and verify the source and redistribution rights before supplying the workbook. If available, run:

```powershell
python -m src.train_reproductive_tox
```

This fits Logistic Regression, Random Forest, and XGBoost baselines on RDKit Morgan fingerprints and reports five-fold Bemis-Murcko scaffold-grouped out-of-fold metrics. Results are saved to `data/reproductive_tox_training_metrics.json`; full-data exploratory artifacts are saved under `models/reproductive_tox/`.

These are **nonclinical assay label predictions**, not human pregnancy outcome predictions. The supplied workbook lacks an accompanying citation/codebook defining how the TG target classes were derived; validate its label semantics before scientific use. The initial scaffold-grouped baselines are only modest and do not beat the majority-class accuracy on the measured endpoints. They do not establish high accuracy or calibration, and they must not be used for clinical risk decisions. The ASD/ADHD anti-seizure supplement is aggregate hazard-ratio output without patient-level event labels, so it remains unsuitable for supervised training.

The repository includes processed dashboard data and audit reports, but does not bundle the additional TG workbook, drug-target interaction archive, or anti-seizure study supplement because their redistribution rights are not established here. The dataset audit handles absent optional sources; training the exploratory TG models requires obtaining the workbook separately. Raw/source data should only be added after provenance and redistribution permissions are confirmed.

The dashboard separates software tests, dataset validation, model training, model evaluation, and prediction uncertainty. Software tests do not establish predictive performance. When a compatible, evaluated checkpoint exists, outcome metrics and predictions can be displayed. Without one, the dashboard displays evidence and availability but withholds scores and risk rankings; prediction uncertainty is unavailable.

## Important limitations

- The included pregnancy-outcome supplement reports aggregate spontaneous-report counts; counts are not incidence, patient-level labels, or causal estimates.
- The supplied ToxRefDB summary contains study metadata, not observed positive/negative toxicity endpoints. Exact-name-linked records remain study evidence only.
- An absent source or report is not evidence that a drug is safe.
- General ADRs, animal study evidence, pregnancy classifications, and human pregnancy outcomes are distinct evidence types.
- Monte Carlo dropout variance is an uncertainty signal, not a calibrated probability interval.
- Synthetic data generation is isolated under `tests/fixtures/generated` and must not be used for real-data training or validation.
