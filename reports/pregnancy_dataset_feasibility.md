# Pregnancy dataset feasibility and model-training readiness

Generated: 2026-10-31T08:19:37.562032+00:00

## Executive decision

**No human pregnancy outcome target is trainable.** Preserve the pregnancy-model safeguard; do not train the pregnancy fusion model on aggregate tables. The new workbook does allow a separate, exploratory structure-based nonclinical assay task, but its label source/meaning is not documented. It is not a pregnancy prediction model, and the scaffold-held-out baselines did not achieve high accuracy over the majority-class baseline. No human pregnancy model accuracy, AUROC, AUPRC, precision, recall, F1, Brier/ECE, or calibrated uncertainty results exist.

The inventory contains an anti-seizure medication study supplementary-tables document and an additional raw assay workbook, but no patient-level Dryad cohort file. The anti-seizure document has aggregate study statistics only. The file `datasets/supplemental_data.docx` is an infection-diagnosis/drug definition supplement, while `datasets/medicines-pregnancy-current-database-2026-10-21.csv` contains pregnancy classifications and narrative statements.

## Dataset inventory and granularity

| Local source | Unique drugs / records | Outcome data | Usable binary labels |
| --- | --- | --- | --- |
| Mendeley dermatology FAERS supplement | 116 parsed report-total drug names; 93 with outcome-term rows | 2083 aggregate outcome-term rows | 0 |
| Anti-seizure medication study supplement | 16205 children; 5637 in epilepsy subgroup | Aggregate ASD/ADHD hazard ratios; no individual outcome records | 0 |
| Pregnancy classification catalog | 1725 names; 30 antiepileptic-category names | Classifications / narrative statements, not observations | 0 |
| Nonclinical TG assay workbook | 2244 unique structures | 5 binary TG assay endpoints | Exploratory labels only |

### Mendeley dermatology FAERS supplement

Source: `datasets/FAERS_JAAD_Mendeley Supplement.docx`. The document describes **121** dermatologic medications and states that **118** were analyzed. The local parser found **116** unique names in report-total table headings and **93** names with outcome-term rows. This is a parsing/coverage discrepancy against the source narrative, so the extracted table inventory should not be treated as a complete medication list.

The supplement provides aggregate spontaneous-report counts grouped by drug and MedDRA outcome terms. It does not provide the underlying patient-level rows, a pregnancy-exposure denominator, or validated non-event controls. The paper's methods refer to case-level FAERS review, but those records are not included in the local supplement.

It contains **2083** outcome-term rows. Of these, **2082** have a count, **2082** have a positive reported count, and **0** have zero. These are counts of reported terms, not binary drug-level outcome labels or independent pregnancy counts; one pregnancy may contribute multiple outcome terms.

| Outcome | Term rows | Drugs | Sum of term counts* | Count missing | Binary negative labels |
| --- | --- | --- | --- | --- | --- |
| Fetal death | 283 | 87 | 4819 | 0.0% | Not available |
| Preterm birth / prematurity | 64 | 64 | 1529 | 0.0% | Not available |
| Fetal growth restriction | 123 | 56 | 896 | 0.0% | Not available |
| Congenital anomaly | 1597 | 84 | 4546 | 0.06% | Not available |

\* The term-count sums are descriptive source counts, not incidence, patient totals, or positive-label counts. The source does not identify valid drug-level negatives.

Identifier mapping for the **93** source outcome-drug names: **67 (72.04%)** match a local normalized name key; **51 (54.84%)** have a PubChem CID. In the processed outcome-term CSV, **527 / 2083 (25.3%)** rows have no local `drug_id`.

Missing-value percentages in the processed outcome-term table: `reported_count` 0.05%; outcome category 0.77%; drug ID 25.3%. Unclassified outcome terms are retained separately rather than forced into an endpoint.

#### Source drug names and matched identifiers

| Mendeley source drug name | Outcome terms present | Local drug ID | PubChem CID | Resolution |
| --- | --- | --- | --- | --- |
| Abrocitinib | Yes | CID078323835 | 78323835 | resolved |
| Acyclovir | Yes | Unmatched | Unavailable | not_matched |
| Adalimumab | Yes | NAME:adalimumab | Unavailable | unresolved |
| Aluminum chloride | No | Unmatched | Unavailable | not_matched |
| Amoxicillin | Yes | CID000033613 | 33613 | resolved |
| Anakinra | Yes | NAME:anakinra | Unavailable | unresolved |
| Atovaquone-Proguanil | Yes | Unmatched | Unavailable | not_matched |
| Azathioprine | Yes | CID000002265 | 2265 | resolved |
| Azithromycin | Yes | CID000447043 | 447043 | resolved |
| Baricitinib | Yes | CID044205240 | 44205240 | resolved |
| Beclomethasone | No | CID000020469 | 20469 | resolved |
| Betamethasone | No | Unmatched | Unavailable | not_matched |
| Botulinum Toxin | Yes | Unmatched | Unavailable | not_matched |
| Brodalumab | Yes | Unmatched | Unavailable | not_matched |
| Calcipotriene | Yes | Unmatched | Unavailable | not_matched |
| Canakinumab | Yes | NAME:canakinumab | Unavailable | unresolved |
| Ceftriaxone | Yes | CID005479530 | 5479530 | resolved |
| Cemiplimab | No | NAME:cemiplimab | Unavailable | unresolved |
| Certolizumab pegol | Yes | Unmatched | Unavailable | not_matched |
| Cetirizine | Yes | CID000002678 | 2678 | resolved |
| Chlorpheniramine | Yes | Unmatched | Unavailable | not_matched |
| Ciprofloxacin | Yes | CID000002764 | 2764 | resolved |
| Clarithromycin | Yes | CID000084029 | 84029 | resolved |
| Clascoterone | No | CID011750009 | 11750009 | resolved |
| Clindamycin | Yes | CID000446598 | 446598 | resolved |
| Clobetasol | Yes | CID005311051 | 5311051 | resolved |
| Clotrimazole | Yes | CID000002812 | 2812 | resolved |
| Coal Tar | No | Unmatched | Unavailable | not_matched |
| Colchicine | Yes | CID000006167 | 6167 | resolved |
| Corticosteroid Nos | No | Unmatched | Unavailable | not_matched |
| Cortisone | No | CID000222786 | 222786 | resolved |
| Crisaborole | No | CID044591583 | 44591583 | resolved |
| Cyclosporine | Yes | Unmatched | Unavailable | not_matched |
| Dapsone | No | CID000002955 | 2955 | resolved |
| Deucravacitinib | No | CID134821691 | 134821691 | resolved |
| Dexamethasone | Yes | CID000005743 | 5743 | resolved |
| Dinitrochlorobenzene | No | Unmatched | Unavailable | not_matched |
| Diphenhydramine | Yes | CID000003100 | 3100 | resolved |
| Doxepin | Yes | CID000003158 | 3158 | resolved |
| Doxycycline | Yes | CID054671203 | 54671203 | resolved |
| Dupilumab | Yes | NAME:dupilumab | Unavailable | unresolved |
| Epinepherine | Yes | Unmatched | Unavailable | not_matched |
| Erythromycin | Yes | CID000012560 | 12560 | resolved |
| Etanercept | Yes | NAME:etanercept | Unavailable | unresolved |
| Fenticonazole | No | Unmatched | Unavailable | not_matched |
| Fexofenadine | Yes | CID000003348 | 3348 | resolved |
| Fluconazole | Yes | CID000003365 | 3365 | resolved |
| Fluocinolone | Yes | CID000091488 | 91488 | resolved |
| Fluocinonide | Yes | Unmatched | Unavailable | not_matched |
| Ganciclovir | Yes | CID135398740 | 135398740 | resolved |
| Glycolic acid | No | Unmatched | Unavailable | not_matched |
| Glycopyrronium | No | Unmatched | Unavailable | not_matched |
| Griseofulvin | Yes | CID000441140 | 441140 | resolved |
| Guselkumab | Yes | NAME:guselkumab | Unavailable | unresolved |
| Hyaluronate andderivatives | No | Unmatched | Unavailable | not_matched |
| Hydrocortisone | Yes | CID000005754 | 5754 | resolved |
| Hydroquinone | Yes | Unmatched | Unavailable | not_matched |
| Hydroxychloroquine | Yes | CID000003652 | 3652 | resolved |
| Hydroxyzine | Yes | CID000003658 | 3658 | resolved |
| Infliximab | Yes | NAME:infliximab | Unavailable | unresolved |
| Ipilimumab | Yes | NAME:ipilimumab | Unavailable | unresolved |
| Isoniazid and Rifampin | Yes | Unmatched | Unavailable | not_matched |
| Isotretinoin | Yes | Unmatched | Unavailable | not_matched |
| Itraconazole | Yes | CID000055283 | 55283 | resolved |
| Ixekizumab | Yes | NAME:ixekizumab | Unavailable | unresolved |
| Ketoconazole | Yes | CID000047576 | 47576 | resolved |
| Levocetirizine | Yes | CID001549000 | 1549000 | resolved |
| Levofloxacin | Yes | Unmatched | Unavailable | not_matched |
| Lidocaine | Yes | CID000003676 | 3676 | resolved |
| Loratadine and desloratadine | Yes | Unmatched | Unavailable | not_matched |
| Methotrexate | Yes | CID000126941 | 126941 | resolved |
| Methylprednisolone | Yes | CID000006741 | 6741 | resolved |
| Metronidazole | Yes | CID000004173 | 4173 | resolved |
| Miconazole | Yes | CID000004189 | 4189 | resolved |
| Minoxidil | Yes | CID000004201 | 4201 | resolved |
| Moxifloxacin | Yes | CID000152946 | 152946 | resolved |
| Mycophenolate | Yes | Unmatched | Unavailable | not_matched |
| Nivolumab and Relatlimab | Yes | Unmatched | Unavailable | not_matched |
| Nystatin | Yes | NAME:nystatin | Unavailable | ambiguous |
| Ocrelizumab | Yes | NAME:ocrelizumab | Unavailable | unresolved |
| Omalizumab | Yes | NAME:omalizumab | Unavailable | unresolved |
| Oxacillin | Yes | Unmatched | Unavailable | not_matched |
| Penicillin | No | Unmatched | Unavailable | not_matched |
| Permethrin | Yes | CID000040326 | 40326 | resolved |
| Pimecrolimus | Yes | CID006509979 | 6509979 | resolved |
| Prednisolone | Yes | CID000005755 | 5755 | resolved |
| Prednisone | Yes | CID000005865 | 5865 | resolved |
| Psoralen (PUVA) | No | Unmatched | Unavailable | not_matched |
| Risankizumab | Yes | NAME:risankizumab | Unavailable | unresolved |
| Rituximab | Yes | NAME:rituximab | Unavailable | unresolved |
| Roflumilast | No | Unmatched | Unavailable | not_matched |
| Ruxolitinib | Yes | Unmatched | Unavailable | not_matched |
| Secukinumab | Yes | Unmatched | Unavailable | not_matched |
| Sirukumab | No | Unmatched | Unavailable | not_matched |
| Spironolactone | Yes | CID000005833 | 5833 | resolved |
| Squaric acid dibutylester | No | Unmatched | Unavailable | not_matched |
| Sulfamethoxazole and Trimethoprim | Yes | Unmatched | Unavailable | not_matched |
| Tacrolimus | Yes | CID000445643 | 445643 | resolved |
| Tapinarof | No | Unmatched | Unavailable | not_matched |
| Tazarotene | Yes | CID000005381 | 5381 | resolved |
| Terbinafine | Yes | CID001549008 | 1549008 | resolved |
| Terconazole | Yes | Unmatched | Unavailable | not_matched |
| Tetracycline | Yes | CID054675776 | 54675776 | resolved |
| Tildrakizumab | Yes | NAME:tildrakizumab | Unavailable | unresolved |
| Tioconazole | Yes | Unmatched | Unavailable | not_matched |
| Tirbanibulin | No | CID023635314 | 23635314 | resolved |
| Tofacitinib | Yes | Unmatched | Unavailable | not_matched |
| Tralokinumab | Yes | Unmatched | Unavailable | not_matched |
| Tranexamic acid | Yes | CID000005526 | 5526 | resolved |
| Tretinoin | Yes | CID000444795 | 444795 | resolved |
| Triamcinolone | Yes | Unmatched | Unavailable | not_matched |
| Upadacitinib | Yes | CID058557659 | 58557659 | resolved |
| Ustekinumab | Yes | NAME:ustekinumab | Unavailable | unresolved |
| Valganciclovir | Yes | CID135413535 | 135413535 | resolved |
| Vancomycin | Yes | CID000014969 | 14969 | resolved |
| Vismodegib | Yes | CID024776445 | 24776445 | resolved |

### Anti-seizure medication study supplement

No patient-level Dryad cohort file was found under `datasets/`. A related-looking supplementary document is present: `datasets/Supplemental_tables.docx` (“Anti-seizure medication use during pregnancy and risk of ASD and ADHD in children”). It summarizes **16205** children overall (**12661** without ASM and **12661** with ASM) and reports associations in a subgroup of **5637** children. Its endpoints are autism spectrum disorder and ADHD; the medication groups include any ASM, valproic acid, lamotrigine, and carbamazepine. The tables report hazard ratios and confidence intervals, not event counts, participant-level labels, or drug-level positive/negative observations. It is therefore **not trainable** for either endpoint from the supplied data. The local medication classification catalog has 30 antiepileptic-category names, but these classifications are not outcomes.

### New nonclinical TG assay workbook

Source: `datasets/Supplementary_S3_Raw data.xlsx`. It contains `10` sheets: paired `*_smiles3` and `*_smiles4` versions of five TG endpoints. The training experiment reads the binary target from `*_smiles4` and uses only molecular structure (RDKit Morgan fingerprints) as input. The binary conversion/endpoint semantics are not documented in the workbook or accompanying files, so the models below are exploratory and the dataset is **limited**, not validated for scientific or clinical use. There are 2244 unique canonical structures across endpoints; 11 also exactly match a local PubChem SMILES, corresponding to 11 mapped local drug IDs. This structure overlap does not convert assay labels into human outcomes.

| Endpoint | Usable structures | Encoded positive | Encoded negative | Positive % | Scaffold groups | Readiness |
| --- | --- | --- | --- | --- | --- | --- |
| TG414 | 1242 | 370 | 872 | 29.79% | 927 | limited |
| TG416 | 264 | 74 | 190 | 28.03% | 215 | limited |
| TG421t | 510 | 95 | 415 | 18.63% | 399 | limited |
| TG422t | 946 | 191 | 755 | 20.19% | 677 | limited |
| TG443t | 144 | 29 | 115 | 20.14% | 117 | limited |

### Exploratory scaffold-held-out baseline results

The command `python -m src.train_reproductive_tox` trains Logistic Regression, Random Forest, and XGBoost separately for each TG endpoint, using 5-fold scaffold-grouped out-of-fold predictions. The values below are estimates from that cross-validation, not an external holdout. Accuracy is compared to the majority-class accuracy; high accuracy is not established by these results.

| Endpoint | Model | Accuracy | Majority baseline | AUROC | AUPRC | Precision | Recall | F1 | Brier | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TG414 | logistic_regression | 0.627 | 0.702 | 0.612 | 0.400 | 0.396 | 0.478 | 0.433 | 0.250 | 0.184 |
| TG414 | random_forest | 0.647 | 0.702 | 0.641 | 0.422 | 0.402 | 0.378 | 0.390 | 0.211 | 0.108 |
| TG414 | xgboost | 0.591 | 0.702 | 0.635 | 0.405 | 0.381 | 0.597 | 0.465 | 0.229 | 0.160 |
| TG416 | logistic_regression | 0.701 | 0.720 | 0.672 | 0.457 | 0.468 | 0.486 | 0.477 | 0.209 | 0.126 |
| TG416 | random_forest | 0.712 | 0.720 | 0.707 | 0.472 | 0.483 | 0.392 | 0.433 | 0.191 | 0.103 |
| TG416 | xgboost | 0.708 | 0.720 | 0.701 | 0.525 | 0.482 | 0.541 | 0.510 | 0.200 | 0.122 |
| TG421t | logistic_regression | 0.718 | 0.814 | 0.606 | 0.271 | 0.257 | 0.274 | 0.265 | 0.199 | 0.151 |
| TG421t | random_forest | 0.773 | 0.814 | 0.661 | 0.333 | 0.333 | 0.221 | 0.266 | 0.164 | 0.131 |
| TG421t | xgboost | 0.678 | 0.814 | 0.628 | 0.257 | 0.277 | 0.453 | 0.344 | 0.202 | 0.183 |
| TG422t | logistic_regression | 0.735 | 0.798 | 0.646 | 0.331 | 0.354 | 0.382 | 0.368 | 0.193 | 0.153 |
| TG422t | random_forest | 0.775 | 0.798 | 0.670 | 0.369 | 0.372 | 0.168 | 0.231 | 0.163 | 0.110 |
| TG422t | xgboost | 0.696 | 0.798 | 0.672 | 0.356 | 0.311 | 0.419 | 0.357 | 0.191 | 0.175 |
| TG443t | logistic_regression | 0.681 | 0.799 | 0.514 | 0.206 | 0.160 | 0.138 | 0.148 | 0.220 | 0.183 |
| TG443t | random_forest | 0.708 | 0.799 | 0.561 | 0.226 | 0.158 | 0.103 | 0.125 | 0.193 | 0.179 |
| TG443t | xgboost | 0.660 | 0.799 | 0.534 | 0.230 | 0.115 | 0.103 | 0.109 | 0.216 | 0.181 |

The script saves full-data experimental model files under `models/reproductive_tox/`; those artifacts are strictly for the nonclinical endpoints above and are not loaded by the pregnancy dashboard. Brier and ECE are descriptive only; no calibration validation or confidence interval is provided.

The separate `datasets/drug.target.interaction.tsv.gz` source contains 19378 drug-target records and 2587 drug names (1054 exact normalized-name matches to the local drug map). These interactions are pharmacology features, not pregnancy outcome labels.

### Other local pregnancy classification data

`datasets/medicines-pregnancy-current-database-2026-10-21.csv` contains 1778 rows and 1725 unique normalized names. It includes 30 rows / 30 names classified in antiepileptic categories. It has no patient-level or drug-level observed outcome records and cannot be converted into outcome labels. Missingness is descriptive metadata missingness: safety statement 55.12%, category 0.0%, classification 2 15.58%.

## Compatibility, overlap, and leakage

- Mendeley outcome-drug names overlap the classification catalog by exact normalized name for **67 / 93** names. This is a name-catalog overlap, not shared patient records and not outcome validation.
- Mendeley-versus-patient-level Dryad overlap cannot be calculated because the cohort records are absent. The anti-seizure supplement is not a per-drug outcome cohort.
- The model's current evidence inputs include the aggregate pregnancy/FAERS report-count features. If the same Mendeley counts were used to create target labels, those features would directly leak target information. Exclude source-derived outcome features or use labels from an independent cohort.
- If independent valid labels are obtained, split by drug and preferably chemical scaffold. Never randomly split rows from the same drug across train/test sets.

## Outcome-by-outcome training decision

| Prediction target | Aggregate term rows | Drugs with report terms | Validated positives | Validated negatives | Decision |
| --- | --- | --- | --- | --- | --- |
| Fetal death | 283 | 87 | 0 | 0 | not_trainable |
| Preterm birth / prematurity | 64 | 64 | 0 | 0 | not_trainable |
| Fetal growth restriction | 123 | 56 | 0 | 0 | not_trainable |
| Congenital anomaly | 1597 | 84 | 0 | 0 | not_trainable |
| TG414 encoded assay class | 1242 | 0 | 0 | 0 | limited |
| TG416 encoded assay class | 264 | 0 | 0 | 0 | limited |
| TG421t encoded assay class | 510 | 0 | 0 | 0 | limited |
| TG422t encoded assay class | 946 | 0 | 0 | 0 | limited |
| TG443t encoded assay class | 144 | 0 | 0 | 0 | limited |
| Autism spectrum disorder | 0 | 0 | 0 | 0 | not_trainable |
| ADHD | 0 | 0 | 0 | 0 | not_trainable |
| Dryad antiepileptic study endpoints | 0 | 0 | 0 | 0 | not_trainable |

For each Mendeley outcome, the listed term rows and drug counts are reported evidence only. They are **not** positive training labels, and missing reports cannot define a negative. The ASD and ADHD supplement contains only aggregate hazard-ratio results. Nonclinical TG labels support exploratory modeling only, with unverified label semantics and no clinical interpretation. No unrelated pregnancy outcomes have been combined.

## Required data before training

1. Supply the actual Dryad study data file and its data dictionary/citation, including the cohort and outcome columns.
2. Obtain an outcome-specific human cohort or controlled study with both outcome and non-outcome observations, suitable follow-up, and exposure/comparator denominators.
3. Keep endpoints separate and define ascertainment, pregnancy timing, and follow-up for fetal death, preterm birth, fetal growth restriction, and congenital anomaly.
4. Provide per-record or defensible per-drug aggregation, source references, outcome definitions, and stable identifiers (CID/InChIKey plus the original drug name).
5. Distinguish true observed non-events from unreported/missing outcomes. Encode unknowns as masked, never as zero.
6. Ensure enough independent drug/scaffold groups and positive/negative examples for train, validation, held-out test, and external validation. Check endpoint-specific class balance before selecting metrics.
7. Keep features independent of the target source to prevent target leakage.

## Model status and dashboard interpretation

- **Software pipeline validation:** tests validate implementation behavior only; they do not measure clinical or predictive performance.
- **Dataset validation:** this report records file presence, schemas/granularity, missingness, and identifier coverage.
- **Model training:** blocked because no compatible validated binary labels are present.
- **Model evaluation:** no evaluation metrics exist; do not report accuracy, precision, recall, or confidence as model results.
- **Nonclinical assay training:** exploratory scaffold-held-out models have been trained; their metrics are in the table above and are not pregnancy outcome performance.
- **Stored model files:** checkpoint present=True; metadata present=False; evaluation metrics present=False. A checkpoint by itself is not treated as a trained/evaluated model.
- **Prediction uncertainty:** unavailable for human pregnancy outcomes. Evidence completeness and MC-dropout variance are not calibrated confidence intervals.
