# System Architecture

This document describes what the repository actually executes. It is a
research prototype for evidence-aware prioritisation of pregnancy drug-safety
signals. It does not predict human pregnancy outcomes, because no validated
human pregnancy-outcome labels are available.

## Execution path

```
datasets/ (raw source files)
        │
        ├── src/acquire_datasets.py ──► data/standardized_master_evidence.csv
        │                               data/drug_identifier_mapping.csv
        │                               data/dataset_acquisition_manifest.json
        │
        ├── src/process_real_datasets.py ─► data/drug_mapping.csv
        │                                   data/real_adr_evidence.csv
        │                                   data/real_pregnancy_outcomes.csv
        │                                   data/real_faers_summary.csv
        │                                   data/source_availability.csv
        │
        ├── src/animal_adr/ pipeline ────► data/animal_adr_*.json|csv
        │     acquire → preprocess →        data/cleaned/animal_adr_*.csv
        │     map → features
        │
        ├── src/data_preprocessing.py ───► features + availability masks
        │     PregnancyDataset              (5 sources; 6th when animal enabled)
        │
        └── src/dataset_feasibility.py ──► data/dataset_feasibility.json
                                            reports/pregnancy_dataset_feasibility.md
```

`python -m src.run_pipeline` runs each stage in order, validates its outputs,
records status and details in `data/pipeline_run.json`, and continues past a
failing stage without fabricating data.

## Evidence layers (never merged)

| Layer | Stored in | Can inform a human pregnancy conclusion? |
| --- | --- | --- |
| Human pregnancy-specific evidence | `data/real_pregnancy_outcomes.csv`, `data/standardized_master_evidence.csv` | It is the only pregnancy-specific human layer, but it is aggregate/untimed association evidence, not a label |
| Human spontaneous ADR evidence | `data/real_adr_evidence.csv`, `data/real_faers_summary.csv` | No — general adverse-event terms and reporting volume |
| Animal spontaneous ADR evidence | `data/animal_adr_features.csv`, `data/cleaned/animal_adr_*` | No — supplementary veterinary reporting signal |
| Animal experimental toxicity | `data/real_toxicity_evidence.csv` (OECD TG endpoints) | No — nonclinical assay evidence |

Missing evidence is represented as an availability mask value of 0, an empty
record set and `None`/`"Unavailable"` in responses. It is never imputed and
never converted into a negative label.

## Models

- `src/models/framework.py` — `MultimodalPregnancyADRModel`. Five evidence
  sources by default (ADR, chemical descriptors, molecular graph, pregnancy
  outcomes, FAERS reporting); `include_animal_adr=True` adds the animal branch
  as a sixth source. Fusion is `MaskAwareCrossAttentionGating`, conditioned on
  the availability mask and evidence-quality vector.
- `src/models/animal_adr_encoder.py` — species/reaction embedding MLP. Emits
  exact zeros when no animal evidence exists.
- `src/models/adr_tox_models.py`, `src/models/molecular_gnn.py` — baseline
  encoders used by the ablation harness.

No checkpoint is trained or claimed. `src/train.py` raises when validated binary
human pregnancy-outcome labels are absent, and `src/services/inference.py`
returns `status="abstained"` for every human pregnancy target. The A/B/C/D
ablation (`src/animal_adr_experiments.py`) runs forward passes with randomly
initialised weights only to verify mask behaviour; it records metrics as
blocked. Its key check is that with shared weights and the animal source
masked, the six-source model reproduces the five-source model exactly
(`A_equals_D_mean_probability`).

## Query and presentation layer

```
Streamlit dashboard (dashboard/app.py)
        │  query: name / drug_id / PubChem CID / InChIKey
        ├── src/services/drug_service.py   identity, molecular descriptors,
        │                                  per-source evidence, availability
        ├── src/mepr_reasoning.py          filtered evidence, Q_cov, abstention
        └── src/services/animal_adr_service.py  animal ADR evidence (separate panel)
```

`src/services/inference.py` is the programmatic equivalent for target-level
requests. It abstains on human pregnancy targets, runs a nonclinical toxicity
checkpoint only if one exists on disk, and always attaches animal ADR evidence
as a separate branch.

`src/ml_risk_model.py` contains `PregnancyEvidenceEngine`: observed report
counts plus evidence coverage, with `risk_available=False`. It performs no
scoring.

## Reports

| Report | Produced by |
| --- | --- |
| `data/pipeline_run.json` | `src/run_pipeline.py` |
| `data/dataset_acquisition_manifest.json` | `src/acquire_datasets.py` |
| `data/dataset_feasibility.json`, `reports/pregnancy_dataset_feasibility.md` | `src/dataset_feasibility.py` |
| `data/animal_adr_*.json`, `reports/animal_adr_integration_report.md` | `src/animal_adr/pipeline.py` |
| `reports/animal_adr_ablation_results.csv`, `reports/animal_adr_evaluation_report.json` | `src/animal_adr_experiments.py` |
| `data/ml_risk_metrics.json`, `src/models/ml_risk_model_metadata.json` | `src/ml_risk_model.py` |

Every report is written from the execution that produced it; no metric is
manually entered.
