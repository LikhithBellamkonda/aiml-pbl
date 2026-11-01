# Next Steps: Scientific Audit of Pregnancy-Safety Prediction Data

**Audit date:** 2026-11-01
**Scope:** Read-only scientific and dataset audit of the files currently present in `datasets/`, the saved nonclinical evaluation results, and the relevant training/evaluation code. No code was changed and no new model training was run.

## Executive findings

1. **The five TG columns are names of OECD nonclinical test-guideline studies, not validated human pregnancy outcomes.** The workbook contains binary values in its `*_smiles4` sheets, but no included citation, codebook, or outcome rule explains what `0` and `1` mean. Therefore, their biological direction and label validity are unknown.
2. **The `smiles3` and `smiles4` sheets are not interchangeable.** They have the same records in the same order, but `smiles4` converts all assay-feature columns to 0/1. For TG414 and TG416, it also converts 17 and 2 fractional target values, respectively, into binary values. The observed conversion is consistent with a 0.5 cutoff, but its source and meaning are undocumented. The existing trainer uses only `smiles4` targets and molecular fingerprints; it does not use the workbook's assay-feature columns.
3. **Per-endpoint duplicate handling is sound for exact canonical SMILES:** no canonical duplicates or conflicting canonical labels were found in the rows used for training. However, the current “scaffold” grouping assigns every acyclic compound its own group, so related acyclic compounds can cross folds.
4. **The 5-fold `GroupKFold` split prevents a Bemis–Murcko scaffold appearing in both train and test for cyclic structures.** It is a useful safeguard, but it is not stratified, does not address acyclic analogues, and does not provide external validation or uncertainty intervals.
5. **Current nonclinical results do not support a high-accuracy claim.** The best accuracy for every TG task is below its overall majority-class accuracy. Best AUROC is 0.561–0.707. These are exploratory pooled out-of-fold results on labels whose semantics have not been verified.
6. **No human pregnancy endpoint is trainable from the currently available files.** The pregnancy sources contain aggregate spontaneous-report counts, aggregate hazard ratios, or classifications—not compatible human outcome/non-outcome labels. The pregnancy training safeguard should remain unchanged.

## 1. Dataset provenance and endpoint definitions

### Workbook provenance

The local assay file is [`datasets/Supplementary_S3_Raw data.xlsx`](./datasets/Supplementary_S3_Raw%20data.xlsx). It has ten sheets: one `smiles3` and one `smiles4` sheet for each TG endpoint. The workbook has no usable title, subject, citation, or label-codebook metadata. The local project also does not contain a matching citation or data dictionary for this workbook. The original source and the rules used to produce its labels therefore **could not be established from the supplied artifact**.

The TG identifiers correspond to OECD nonclinical guideline study designs. Their names describe the type of study, not the meaning of this workbook's binary classes:

| Workbook endpoint | OECD guideline / scientific scope | What the workbook does **not** establish |
|---|---|---|
| TG414 | [OECD Test No. 414: Prenatal Developmental Toxicity Study](https://doi.org/10.1787/9789264070820-en). A nonclinical prenatal-development study. | Whether label 1 means developmental toxicity, an observed effect, a threshold exceedance, or another category. |
| TG416 | [OECD Test No. 416: Two-Generation Reproduction Toxicity](https://doi.org/10.1787/9789264070868-en). A nonclinical two-generation reproductive-toxicity study. | Which reproductive/developmental findings define either class, or how mixed findings are aggregated. |
| TG421t | [OECD Test No. 421: Reproduction/Developmental Toxicity Screening Test](https://doi.org/10.1787/9789264264380-en). A nonclinical screening study. | Whether the target is any effect, a specific endpoint, or a derived summary class. |
| TG422t | [OECD Test No. 422: Combined Repeated Dose Toxicity Study with the Reproduction/Developmental Toxicity Screening Test](https://doi.org/10.1787/9789264242715-en). A combined nonclinical repeated-dose and reproductive/developmental screening study. | How general toxicity is distinguished from reproductive/developmental toxicity in the label. |
| TG443t | [OECD Test No. 443: Extended One-Generation Reproductive Toxicity Study](https://doi.org/10.1787/9789264185371-en). A nonclinical extended one-generation reproductive-toxicity study. | Which cohorts, findings, or triggers determine the binary class. |

These are animal/nonclinical study guidelines. They are **not human pregnancy outcome labels**, and the OECD guideline number alone does not provide a binary outcome definition.

### What numeric labels mean in the supplied sheets

The `*_smiles4` target columns contain only `0` and `1`, but the workbook does not label either value “toxic,” “non-toxic,” “positive,” or “negative.” In this report, `label=1` and `label=0` mean only the stored numeric classes. They must not be interpreted as biological positives/negatives until the source definition is obtained.

| Endpoint | `smiles3` target values | `smiles4` target counts (`1` / `0`) | Variant observation |
|---|---:|---:|---|
| TG414 | 357 values of 1, 868 of 0, and 17 fractional values | 370 / 872 | All 17 fractional rows are converted: 13 values at or above 0.5 become 1; four below 0.5 become 0. |
| TG416 | 73 values of 1, 189 of 0, and 2 fractional values | 74 / 190 | The 0.5 value becomes 1 and the 0.3333 value becomes 0. |
| TG421t | 95 values of 1 and 415 of 0 | 95 / 415 | Target labels are unchanged. |
| TG422t | 191 values of 1 and 755 of 0 | 191 / 755 | Target labels are unchanged. |
| TG443t | 30 values of 1 and 115 of 0 | 30 / 115 | Target labels are unchanged; one row is not a valid SMILES and is excluded by the trainer. |

For TG414 and TG416, the observed fractional-value conversion is consistent with thresholding at 0.5. This is an empirical observation about the paired sheets—not proof of the intended label rule. The original fraction meaning, threshold rationale, and class direction remain unknown.

### `smiles3` versus `smiles4`

- Each paired sheet contains the same `SMILES_wash` records in the same order.
- The `smiles3` sheets have 118 non-target assay-feature columns for TG414/TG416 and 109 for TG421t/TG422t/TG443t. Their assay values include continuous/fractional values.
- Every assay-feature column in each `smiles4` sheet contains only 0/1 values. Thus, `smiles4` applies a material feature transformation as well as binary target encoding for TG414/TG416. No transformation documentation was supplied.
- The existing trainer reads only `SMILES_wash` and the target column from `smiles4`, creates RDKit Morgan fingerprints, and ignores all other workbook columns. This avoids direct use of those assay features in current model inputs, but does not validate the target labels.

## 2. Dataset suitability, duplicates, imbalance, and overlap

The counts below refer to the current trainer's usable `smiles4` rows after RDKit parsing and its canonical-SMILES deduplication. “Label 1” is a numeric class only; its biological interpretation is unknown.

| Endpoint | Usable compounds | Label 1 | Label 0 | Label 1 share | Bemis–Murcko groups | Usable for validated science? |
|---|---:|---:|---:|---:|---:|---|
| TG414 | 1,242 | 370 | 872 | 29.8% | 927 | No—provenance and label definition missing |
| TG416 | 264 | 74 | 190 | 28.0% | 215 | No—provenance and label definition missing |
| TG421t | 510 | 95 | 415 | 18.6% | 399 | No—provenance and label definition missing |
| TG422t | 946 | 191 | 755 | 20.2% | 677 | No—provenance and label definition missing |
| TG443t | 144 | 29 | 115 | 20.1% | 117 | No—provenance and label definition missing |

Checks on the rows used by the trainer found:

- No repeated raw `SMILES_wash` values per endpoint.
- No duplicate canonical SMILES per endpoint after RDKit canonicalization.
- No conflicting labels for a canonical structure within any endpoint.
- One invalid SMILES in the 145-row TG443t sheet is excluded; the remaining 144 rows contain 29 label-1 and 115 label-0 compounds.
- These checks cover canonical SMILES identity only. The workbook has no stable substance identifiers, source study IDs, salt/mixture policy, or replicate-level records, so broader substance equivalence and repeated-study conflicts cannot be ruled out.

The same canonical compounds occur in multiple endpoint datasets. Pairwise overlaps are:

| Endpoint pair | Shared canonical structures |
|---|---:|
| TG414 / TG416 | 217 |
| TG414 / TG421t | 184 |
| TG414 / TG422t | 274 |
| TG414 / TG443t | 125 |
| TG416 / TG421t | 28 |
| TG416 / TG422t | 42 |
| TG416 / TG443t | 11 |
| TG421t / TG422t | 105 |
| TG421t / TG443t | 30 |
| TG422t / TG443t | 46 |

The union is 2,244 canonical structures across the five endpoints. Cross-endpoint reuse is not leakage in the current separately trained, separately evaluated endpoint models by itself. It becomes a leakage risk if the endpoints are combined into a multi-task experiment and a structure/scaffold is allowed in both training and test for different tasks.

### Class imbalance

The numeric label-1 class ranges from 18.6% to 29.8%; a majority-class accuracy baseline is therefore 70.2%–81.4%. This imbalance makes accuracy alone unsuitable. The code reports balanced accuracy, AUROC, AUPRC, precision, recall, and F1, but because class meaning is unknown, these remain metrics for unexplained workbook classes, not for toxicity or pregnancy safety.

## 3. Scaffold cross-validation review

The current implementation in [`src/train_reproductive_tox.py`](./src/train_reproductive_tox.py) uses 5-fold `GroupKFold` with a group key derived from the RDKit Bemis–Murcko scaffold. It canonicalizes SMILES, rejects conflicting labels for a canonical structure, removes duplicate canonical structures, and creates fingerprints independently for each compound. Each fold is checked to contain both numeric classes in train and test; a fold that fails this check aborts evaluation.

**What is correct/useful:**

- `GroupKFold` keeps equal group IDs in one fold, so no identical Bemis–Murcko scaffold string is shared across train and test.
- Exact canonical duplicates are removed before splitting, preventing the same canonical structure from appearing on both sides within an endpoint.
- Current model input features are SMILES-derived Morgan fingerprints, not target columns or the workbook's assay-feature columns.
- Metrics are computed from pooled out-of-fold predictions, not in-sample predictions.

**Limitations:**

1. **Acyclic compounds are not scaffold-grouped as a chemical family.** An empty Murcko scaffold is replaced by `acyclic:<canonical SMILES>`, making each distinct acyclic compound its own group. The current datasets contain 634 acyclic compounds for TG414, 138 for TG416, 177 for TG421t, 410 for TG422t, and 75 for TG443t. Similar acyclic molecules can therefore appear across folds.
2. **Folds are not stratified.** The explicit both-class check avoids undefined AUROC, but does not ensure stable positive/negative ratios. For example, TG443t fold test sets contain between 2 and 12 label-1 compounds; such small per-fold positive counts make fold behavior unstable.
3. **Group separation is not a guarantee of chemical independence.** Different Murcko scaffolds can still be highly similar. Report nearest-neighbor similarity across train/test and consider a pre-specified scaffold/cluster split policy.
4. **No independent external test set or temporal/source-held-out validation exists.** Pooled cross-validation results are not external generalization estimates.
5. **No confidence intervals or repeated-split sensitivity analysis are reported.** Small endpoints—especially TG443t—have substantial sampling uncertainty that the current point metrics do not express.
6. **The global majority baseline is computed from all endpoint labels.** A fair fold-level baseline should be derived using each training fold only, then evaluated on that fold's held-out records. The current baseline is descriptive, not a fully fold-independent benchmark.

**Conclusion:** the implementation is a reasonable initial scaffold-grouped exploratory split for cyclic structures, but it is not sufficient to establish robust chemical generalization and should not be called a validated scaffold-held-out benchmark until the acyclic grouping and split diagnostics are addressed.

## 4. Existing model evaluation and weaknesses

The saved results in [`data/reproductive_tox_training_metrics.json`](./data/reproductive_tox_training_metrics.json) are 5-fold pooled out-of-fold estimates. Each model is subsequently fit on the full endpoint dataset and saved as an exploratory artifact. There is no independent holdout or confidence interval. The table shows every saved model. `Acc / majority` compares accuracy with the overall majority-label baseline; all baselines and labels are numeric only.

| Endpoint | Model | Accuracy / majority | Balanced accuracy | AUROC | AUPRC | Precision | Recall | F1 | Brier | ECE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TG414 | Logistic Regression | 0.627 / 0.702 | 0.584 | 0.612 | 0.400 | 0.396 | 0.478 | 0.433 | 0.250 | 0.184 |
| TG414 | Random Forest | 0.647 / 0.702 | 0.570 | 0.641 | 0.422 | 0.402 | 0.378 | 0.390 | 0.211 | 0.108 |
| TG414 | XGBoost | 0.591 / 0.702 | 0.593 | 0.635 | 0.405 | 0.381 | 0.597 | 0.465 | 0.229 | 0.160 |
| TG416 | Logistic Regression | 0.701 / 0.720 | 0.635 | 0.672 | 0.457 | 0.468 | 0.486 | 0.477 | 0.209 | 0.126 |
| TG416 | Random Forest | 0.712 / 0.720 | 0.614 | 0.707 | 0.472 | 0.483 | 0.392 | 0.433 | 0.191 | 0.103 |
| TG416 | XGBoost | 0.708 / 0.720 | 0.657 | 0.701 | 0.525 | 0.482 | 0.541 | 0.510 | 0.200 | 0.122 |
| TG421t | Logistic Regression | 0.718 / 0.814 | 0.546 | 0.606 | 0.271 | 0.257 | 0.274 | 0.265 | 0.199 | 0.151 |
| TG421t | Random Forest | 0.773 / 0.814 | 0.560 | 0.661 | 0.333 | 0.333 | 0.221 | 0.266 | 0.164 | 0.131 |
| TG421t | XGBoost | 0.678 / 0.814 | 0.591 | 0.628 | 0.257 | 0.277 | 0.453 | 0.344 | 0.202 | 0.183 |
| TG422t | Logistic Regression | 0.735 / 0.798 | 0.603 | 0.646 | 0.331 | 0.354 | 0.382 | 0.368 | 0.193 | 0.153 |
| TG422t | Random Forest | 0.775 / 0.798 | 0.548 | 0.670 | 0.369 | 0.372 | 0.168 | 0.231 | 0.163 | 0.110 |
| TG422t | XGBoost | 0.696 / 0.798 | 0.592 | 0.672 | 0.356 | 0.311 | 0.419 | 0.357 | 0.191 | 0.175 |
| TG443t | Logistic Regression | 0.681 / 0.799 | 0.478 | 0.514 | 0.206 | 0.160 | 0.138 | 0.148 | 0.220 | 0.183 |
| TG443t | Random Forest | 0.708 / 0.799 | 0.482 | 0.561 | 0.226 | 0.158 | 0.103 | 0.125 | 0.193 | 0.179 |
| TG443t | XGBoost | 0.660 / 0.799 | 0.452 | 0.534 | 0.230 | 0.115 | 0.103 | 0.109 | 0.216 | 0.181 |

Interpretation and evaluation limitations:

- **Accuracy:** every accuracy is below its majority-class baseline. The observed accuracy is not evidence of high predictive value.
- **Discrimination:** best AUROC by endpoint is 0.641 (TG414), 0.707 (TG416), 0.661 (TG421t), 0.672 (TG422t), and 0.561 (TG443t). These are modest exploratory results, not estimates of clinical or pregnancy-safety performance.
- **AUPRC:** should be compared with the relevant label-1 prevalence (approximately 0.19–0.30), and remains uninterpretable biologically until label direction is verified.
- **Model selection:** three algorithms are compared across five endpoints using the same CV dataset; selecting the best reported result after viewing these metrics introduces selection optimism. No nested CV or pre-registered primary model is used.
- **Thresholding:** precision, recall, and F1 use a fixed probability threshold of 0.5. No operating threshold is selected against a defined use case or within training-only validation.
- **Calibration:** Brier score and 10-bin ECE are descriptive on pooled OOF predictions. No calibration model, uncertainty interval, calibration-in-the-large/slope assessment, or independent calibration cohort is provided. They do not establish calibrated probabilities.
- **Hyperparameters:** the fixed model settings are not selected in nested validation. Class weighting is used, which can affect probability calibration.
- **No comparator uncertainty:** there are no confidence intervals, statistical tests against baselines, external test results, or repeated-seed robustness checks.

Do not display these assay results as pregnancy-model accuracy, confidence, or pregnancy predictions.

## 5. Drug-target interaction data

[`datasets/drug.target.interaction.tsv.gz`](./datasets/drug.target.interaction.tsv.gz) has 19,378 drug-target interaction records, 2,587 drug names, and 2,167 target names. Its fields include target/accession/gene, activity value/type/source, mode of action, and source URLs. The data are pharmacology evidence, not pregnancy outcomes.

Potential role in a future architecture:

- Could provide a drug-to-protein network or mechanism feature after stable drug and target identifier resolution, source/version tracking, and missingness indicators.
- Could complement chemistry or known pharmacology as an input modality, but cannot supply a pregnancy label or establish fetal exposure, placental transfer, trimester-specific risk, or an adverse pregnancy outcome.
- Must be audited for population/source coverage and temporal relevance. The file has 100% missing `ACT_UNIT`, 8.28% missing `ACT_VALUE`, 8.29% missing `ACT_TYPE`, 30.17% missing `ACT_COMMENT`, and 85.9% missing `MOA`; interaction sources are heterogeneous.
- Local name matching is incomplete and is not equivalent to active-ingredient identity resolution. The matching result in the current feasibility audit is a name match, not validation of a pharmacologic record against a pregnancy cohort.
- If used in supervised modeling, exclude any feature derived from the target cohort's outcome or post-outcome evidence. Split all formulations/aliases/active-moiety duplicates together and preserve missing-source masks.

The current reproductive-toxicity trainer does **not** consume this interaction file. Its inclusion in the project's dataset inventory does not make the present toxicity experiments multimodal.

## 6. Pregnancy-related datasets and their suitability

| Local source | Available information / granularity | Why it does not provide a trainable human pregnancy target |
|---|---|---|
| `datasets/FAERS_JAAD_Mendeley Supplement.docx` | Supplement describes 121 dermatology medicines, says 118 were analyzed, and summarizes pregnancy-related FAERS reports from 1970-01-01 through 2024-06-30. Local extraction found 116 report-total drug names, 93 with outcome-term rows, and 2,083 aggregate outcome-term rows. | The supplied supplement does not include case-level patient records or exposed-pregnancy denominators. A spontaneous report is not a validated incidence estimate; no report is not a negative outcome. Counts cannot be interpreted as drug-level positives/negatives. The parsed medication discrepancy also needs source reconciliation. |
| Anti-seizure study supplement, `datasets/Supplemental_tables.docx` | Aggregate cohort tables report 16,205 children (12,661 without ASM; 3,544 with ASM), an epilepsy subgroup of 5,637, and aggregate ASD/ADHD hazard ratios for medication groups including any ASM, valproate, lamotrigine, and carbamazepine. | No patient-level outcome labels, per-drug event/non-event observations, or usable event counts are supplied. Aggregate effect estimates are not per-drug binary labels. |
| Expected Dryad antiepileptic pregnancy-outcome file | The local audit found no patient-level Dryad cohort file. | The missing dataset cannot be used to calculate cohort/drug overlap or supply outcomes. `datasets/supplemental_data.docx` is an infection-diagnosis/drug definition supplement, not the Dryad cohort. |
| `datasets/medicines-pregnancy-current-database-2026-10-21.csv` | 1,778 rows and 1,725 unique normalized names; drug classifications and narrative safety statements. | Classification and narrative categories are not observations of outcomes among exposed and unexposed pregnancies. |
| `datasets/toxrefdb_3_0_study_chem_summary.xlsx` | 6,341 study/chemical summary rows with study type, species, dose, route, guideline, and study metadata. | A study record or study type is not an endpoint label. The supplied summary does not provide a validated binary human pregnancy outcome. |
| `datasets/meddra_all_se.tsv.gz` | MedDRA terminology/supporting dictionary. | Terminology does not provide patient outcome observations or negative controls. |
| TG workbook and drug-target interaction file | Nonclinical assay classes and pharmacology interactions. | Neither is a human pregnancy-outcome cohort. Their labels must not be merged into pregnancy targets. |

The prior local feasibility audit records 0 usable human binary label records for the Mendeley aggregate, absent Dryad cohort, and pregnancy classification catalog. The four Mendeley outcome groups—fetal death, prematurity, fetal growth restriction, and congenital anomaly—must remain separate. Existing training code requires provenance-bearing human observed/controlled outcome labels and rejects aggregate FAERS counts, classifications, and unsupported evidence types; **that safeguard remains necessary**.

### Leakage concerns for a future pregnancy model

- If Mendeley report counts are used as model features while labels are derived from those same reports, the target can be directly leaked into the predictors. Use independent label sources or exclude every feature derived from the target source.
- Do not split patient records from the same pregnancy, drug, active moiety, formulation, or chemical scaffold across training and test.
- Prevent duplicate/overlapping participants across datasets and report dataset provenance, time windows, and linkage rules.
- Use only predictors available at the intended prediction time; exclude post-outcome reporting, treatment changes caused by an outcome, and target-derived classifications.
- Classifications, animal-study outcomes, drug-target records, and FAERS report counts must remain distinct evidence modalities, not substituted for human outcome labels.

## 7. Exact additional data required for intended pregnancy prediction

For **each outcome separately**, obtain a source-backed human pregnancy dataset with:

1. Patient/pregnancy-level records or a defensible, clinically validated drug-level aggregation, including a stable pregnancy/participant key and de-duplication rules.
2. Explicit outcome ascertainment and definitions, timing, gestational age/window, follow-up, and evidence source. A binary label must specify what constitutes an observed event and an observed non-event.
3. Exposed and non-exposed or otherwise valid comparator pregnancies, with denominators, observation periods, and event counts. Unknown or unreported outcomes must remain missing/masked—not be encoded as 0.
4. Medication exposure mapped to active ingredient and stable identifiers (e.g., RxNorm/ATC/DrugBank/PubChem/InChIKey as appropriate), with dose, route, start/stop dates, trimester, combination products, and formulation handling.
5. Relevant confounders and context, including indication, maternal age, comorbidities, co-medications, pregnancy history, and relevant demographic/clinical factors, with a plan for missingness and confounding control.
6. Study/site/time/source identifiers, citations, permissions, a codebook, and enough independent outcome events and non-events across drugs/scaffolds for train, validation, held-out test, and external validation.
7. An explicit leakage plan: source-independent labels/features, participant and drug grouping, chemical scaffold handling, and temporal/site-held-out evaluation.

For TG assays, separately obtain the original citation/download URL, source dataset/version, per-compound identifiers/CAS/DSS/ToxCast IDs, the codebook for `smiles3`/`smiles4`, assay-feature definitions and units, label derivation and threshold, interpretation of fractions, replicate/mixture rules, missingness and exclusions, and confirm whether label 1 is the adverse class. Until then, those models remain an unvalidated exploratory exercise.

## 8. Recommended experiments and prioritized roadmap

### Priority 0 — establish provenance before fitting more models

- Obtain and archive the assay workbook's source paper, original dataset, data dictionary, binary-label derivation, and feature transformation documentation.
- Independently verify the meaning and direction of every TG label against source records. Resolve the 17 TG414 and 2 TG416 fractional-label conversions rather than relying on an inferred cutoff.
- Obtain the missing Dryad cohort and its data dictionary/citation if it is intended to support the antiepileptic target. Confirm participant-level outcomes, outcome event counts, exposure/comparator denominators, and drug identity fields.
- Reconcile the Mendeley medication count (121 reported, 118 analyzed, 116 locally parsed total names) against the full source supplement before using its coverage statistics.

### Priority 1 — strengthen nonclinical validation only after label verification

- Preserve TG endpoints as separate nonclinical targets. Never use these classes as human pregnancy outcomes.
- Build data checks for compound identifiers, stereochemistry/salts, duplicates, repeated-study conflict handling, source-level counts, and missing labels.
- Compare `smiles3` and `smiles4` via a documented transformation rather than assuming either is the canonical analytic table.
- Use a stratified group splitter where feasible, with a defined acyclic-analogue grouping strategy. Add assertions/tests that train/test groups do not overlap and report per-fold class counts.
- Set aside a genuinely external source/temporal cohort if available. Otherwise, report uncertainty intervals with a pre-specified group-aware bootstrap/repeated grouped split and label the estimate as internal validation.
- Add train-fold majority and prevalence baselines, pre-specify the primary metric, conduct threshold selection within training-only validation, and evaluate calibration on a separate calibration/validation set.
- Report applicability domain and nearest-neighbor chemical similarity for each test compound; document which structures are outside training support.

### Priority 2 — build a human outcome cohort before pregnancy-model training

- Construct outcome-specific human datasets with true observed event and non-event labels, source citations, denominator/follow-up, and stable drug identifiers.
- Reconcile duplicated pregnancies and drugs across sources and retain unknowns as masked values.
- Establish drug/scaffold and preferably temporal/site holdouts before model selection. Validate enough events per endpoint and report class counts before selecting estimators.
- Only after this validation, compare simple clinical/chemical baselines (e.g., logistic regression, random forest, gradient boosting) with the missing-evidence-aware fusion architecture. Keep evidence availability separate from outcome probabilities.

### Priority 3 — add pharmacology as a documented feature source, not a label source

- Resolve drug-target records to the cohort's stable active ingredient IDs, harmonize target accessions, preserve activity units/source/date, quantify coverage, and build missingness masks.
- Test whether these features add value over simpler baselines using the same patient/drug/scaffold split. Assess indication and source confounding.

### Priority 4 — clinical evaluation and dashboard governance

- Evaluate externally and report endpoint-specific AUROC, AUPRC, sensitivity/recall, specificity, precision, calibration, confidence intervals, and subgroup performance. Include prevalence and decision thresholds.
- Keep software tests, dataset validation, model fit status, external evaluation, and prediction uncertainty as distinct dashboard statuses.
- Do not show a risk score or confidence estimate unless the relevant human endpoint has a validated model, evaluation provenance, applicability-domain check, and appropriate calibration evidence.

## Final decision

**Do not begin or enable training for the intended human pregnancy prediction model with the currently supplied data.** There is no validated human pregnancy endpoint with both positive and negative observations and adequate provenance. Keep the existing pregnancy-training safeguard unchanged.

The TG sheets provide enough rows to run exploratory numerical classification, and prior scaffold-grouped CV has already been run, but the original dataset provenance and label semantics are unresolved. Do not start further TG model training or present the current TG artifacts as scientifically validated until those definitions are established. Even after verification, those endpoints would remain nonclinical toxicity tasks and would not establish pregnancy-safety prediction.
