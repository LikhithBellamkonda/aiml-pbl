# Dataset integration and feasibility report

Generated: 2026-11-22T14:51:34.889149+00:00

## Result

The source-specific ingestion pipeline completed without creating training labels. It preserves source granularity and distinguishes unknown outcomes from explicitly reported evidence. No pregnancy endpoint is currently trainable.

## Source coverage

| source_dataset | status | cleaned_record_count | unique_drug_count | pubchem_matched_drug_count |
| --- | --- | --- | --- | --- |
| onsides_general_adr | source_unavailable | 0 | 0 | 0 |
| mendeley_pregnancy_v2 | available | 2199 | 116 | 58 |
| dryad_antiepileptic_pregnancy | available_aggregate_tables_only | 98 | 3 | 2 |
| pubchem | available_cached | 1238 | 1173 | 1173 |

## Provenance and access

| source_dataset | citation_or_doi | version | license_or_access | granularity | limitation |
| --- | --- | --- | --- | --- | --- |
| onsides_general_adr | OnSIDES release v3.1.1 | v3.1.1 | Review upstream data and release terms before redistribution. | regulatory product-label adverse-effect associations | No pregnancy outcome labels; source prediction scores are not confidence. |
| mendeley_pregnancy_v2 | 10.17632/fnf4t4x2dz.2 | Version 2 | CC BY 4.0 (verified against official metadata) | drug-level aggregate spontaneous-report counts | No patient-level outcomes, exposed-pregnancy denominator, or validated negative labels. |
| dryad_antiepileptic_pregnancy | 10.5061/dryad.2z34tmpk0 | 10.5061/dryad.2z34tmpk0 | CC0 1.0 (verified against Dryad metadata) | study-level aggregate supplementary tables | No patient-level observations; ASD/ADHD analyses are not the requested pregnancy endpoints. |
| pubchem | PubChem PUG REST | retrieval time | PubChem public chemical reference service | chemical identifiers and structures | Identity/features only; not outcome evidence. |

## Pairwise drug overlap

| source_a | source_b | unique_drugs_a | unique_drugs_b | overlapping_drug_count |
| --- | --- | --- | --- | --- |
| onsides_general_adr | mendeley_pregnancy_v2 | 0 | 116 | 0 |
| onsides_general_adr | dryad_antiepileptic_pregnancy | 0 | 3 | 0 |
| onsides_general_adr | pubchem | 0 | 1762 | 0 |
| mendeley_pregnancy_v2 | dryad_antiepileptic_pregnancy | 116 | 3 | 0 |
| mendeley_pregnancy_v2 | pubchem | 116 | 1762 | 116 |
| dryad_antiepileptic_pregnancy | pubchem | 3 | 1762 | 3 |

Name matching uses normalized exact names and existing/PubChem CID mappings. An OnSIDES product maps to PubChem through an RxNorm ingredient only when exactly one ingredient is documented and no differing salt form is detected; multiple-ingredient products remain intact with their ingredient list and are not split into outcome records. Salt names are not stripped. Unresolved names retain stable `NAME:` identifiers. The source mapping table records original identifiers, ingredient relations, and match methods.

## Pregnancy prediction targets

- **congenital_anomaly — not_trainable:** Sources provide aggregate spontaneous-report counts, report/label associations, or study-level evidence, but no validated human drug-level binary observations with compatible positive and negative outcomes. Missing reports remain unknown.
- **miscarriage — not_trainable:** Sources provide aggregate spontaneous-report counts, report/label associations, or study-level evidence, but no validated human drug-level binary observations with compatible positive and negative outcomes. Missing reports remain unknown.
- **preterm_birth — not_trainable:** Sources provide aggregate spontaneous-report counts, report/label associations, or study-level evidence, but no validated human drug-level binary observations with compatible positive and negative outcomes. Missing reports remain unknown.
- **fetal_growth_outcome — not_trainable:** Sources provide aggregate spontaneous-report counts, report/label associations, or study-level evidence, but no validated human drug-level binary observations with compatible positive and negative outcomes. Missing reports remain unknown.
- **stillbirth — not_trainable:** Sources provide aggregate spontaneous-report counts, report/label associations, or study-level evidence, but no validated human drug-level binary observations with compatible positive and negative outcomes. Missing reports remain unknown.
- **maternal_adverse_reaction — not_trainable:** Sources provide aggregate spontaneous-report counts, report/label associations, or study-level evidence, but no validated human drug-level binary observations with compatible positive and negative outcomes. Missing reports remain unknown.

### Outcome evidence and label distribution

| source_dataset | outcome_category | evidence_record_count | positive_concern_record_count | observed_negative_record_count | validated_positive_label_count | validated_negative_label_count |
| --- | --- | --- | --- | --- | --- | --- |
| mendeley_pregnancy_v2 | congenital_anomaly | 0 | 0 | 0 | 0 | 0 |
| mendeley_pregnancy_v2 | miscarriage | 0 | 0 | 0 | 0 | 0 |
| mendeley_pregnancy_v2 | preterm_birth | 0 | 0 | 0 | 0 | 0 |
| mendeley_pregnancy_v2 | fetal_growth_outcome | 0 | 0 | 0 | 0 | 0 |
| mendeley_pregnancy_v2 | stillbirth | 0 | 0 | 0 | 0 | 0 |
| mendeley_pregnancy_v2 | maternal_adverse_reaction | 0 | 0 | 0 | 0 | 0 |
| dryad_antiepileptic_pregnancy | congenital_anomaly | 0 | 0 | 0 | 0 | 0 |
| dryad_antiepileptic_pregnancy | miscarriage | 0 | 0 | 0 | 0 | 0 |
| dryad_antiepileptic_pregnancy | preterm_birth | 0 | 0 | 0 | 0 | 0 |
| dryad_antiepileptic_pregnancy | fetal_growth_outcome | 0 | 0 | 0 | 0 | 0 |
| dryad_antiepileptic_pregnancy | stillbirth | 0 | 0 | 0 | 0 | 0 |
| dryad_antiepileptic_pregnancy | maternal_adverse_reaction | 0 | 0 | 0 | 0 | 0 |

No negative labels are inferred from absent reports, omitted endpoints, or missing files. The Dryad supplement is an aggregate DOCX (tables of cohort characteristics, ICD codes, and study-level ASD/ADHD hazard-ratio analyses), not individual participant outcomes. Its ASD/ADHD endpoints are distinct from the six pregnancy endpoints and are not converted into labels.

The Mendeley source contains drug-level aggregate spontaneous-report counts without exposed-pregnancy denominators or patient-level non-events. These are reporting signals only, not incidence or validated binary labels. OnSIDES contributes regulatory product-label adverse-effect associations and source model scores; those scores are not confidence estimates or pregnancy outcome labels. PubChem contributes chemical identity and descriptors only.

## Potential leakage

The version, DOI, data-use note, evidence granularity, and limitations for each source are recorded in `../data/source_registry.csv`. Exact canonical-drug overlaps are reported in `../data/drug_overlap.csv`; this measures entity overlap, not shared patients. No individual-level cohort data are available here to test patient overlap or deduplicate outcomes across studies. Drug-level entity overlap is expected across chemical and safety sources, so any future model evaluation must separate canonical drug IDs (and consider scaffold grouping) across train/test sets. Do not use Mendeley report counts as features when evaluating labels derived from the same reports or cohort.

## Identifier matching

| source_dataset | source_identifier_rows | unique_source_drugs | pubchem_matched_drugs | pubchem_match_percent |
| --- | --- | --- | --- | --- |
| dryad_antiepileptic_pregnancy | 3 | 3 | 2 | 66.67 |
| existing_project_mapping | 1725 | 1725 | 1178 | 68.29 |
| mendeley_pregnancy_v2 | 209 | 116 | 58 | 50.0 |
| pubchem | 1767 | 1767 | 1178 | 66.67 |

## Trimester and missingness

### Recorded trimester availability

| source_dataset | trimester | record_count |
| --- | --- | --- |
| Dryad 10.5061/dryad.2z34tmpk0 | unknown | 98 |
| Mendeley Data | unknown | 116 |
| Mendeley Data 10.17632/fnf4t4x2dz.2 | unknown | 2083 |

### Master missingness (top 15 fields)

| field | missing_count | missing_percent |
| --- | --- | --- |
| gestational_weeks | 2297 | 100.0 |
| source_identifier | 2297 | 100.0 |
| study_endpoint | 2285 | 99.4776 |
| outcome_term_original | 2243 | 97.6491 |
| is_requested_pregnancy_endpoint | 2199 | 95.7336 |
| individual_outcome_observation | 2199 | 95.7336 |
| publication_doi | 2199 | 95.7336 |
| source_table_id | 2199 | 95.7336 |
| table_caption_or_row_text | 2199 | 95.7336 |
| source_row_number | 2199 | 95.7336 |
| outcome_semantics_verified | 2199 | 95.7336 |
| raw_record_json | 2199 | 95.7336 |
| outcomes | 2191 | 95.3853 |
| reports | 2181 | 94.9499 |
| outcome_value_original | 2083 | 90.6835 |

Trimester values are used only when explicitly documented as a trimester or gestational weeks; otherwise they remain `unknown`. See `../data/trimester_summary.csv` and `../data/missingness_summary.csv`. The evidence availability matrices distinguish `source_unavailable`, `source_available_no_relevant_record`, `outcome_not_reported`, and positive concern evidence. No source currently provides validated negative human outcome observations.

## Additional data required

For each target endpoint, obtain a documented human pregnancy cohort or controlled study with:

1. Patient- or pregnancy-level exposure and follow-up records, or validated drug-level event and non-event counts with denominators.
2. Endpoint-specific definitions and observed positive and negative outcomes; do not merge unrelated endpoints.
3. Drug exposure timing, gestational weeks/trimester when documented, comparator definition, and relevant confounders.
4. Linkable drug identifiers and sufficient independent drugs/cohorts for leakage-resistant drug/scaffold-separated evaluation.
5. Provenance, validation procedures, license/permissions, and a clear missing-data mechanism.

Until then, keep the existing pregnancy-model training safeguard enabled. Do not report accuracy, recall, precision, calibration, or confidence for a pregnancy predictor.
