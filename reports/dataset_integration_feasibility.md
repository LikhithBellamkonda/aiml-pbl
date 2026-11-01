# Dataset integration and feasibility report

Generated: 2026-11-01T06:00:39.874029+00:00

## Result

The source-specific ingestion pipeline completed without creating training labels. It preserves source granularity and distinguishes unknown outcomes from explicitly reported evidence. No pregnancy endpoint is currently trainable.

## Source coverage

| source_dataset | status | cleaned_record_count | unique_drug_count | pubchem_matched_drug_count |
| --- | --- | --- | --- | --- |
| onsides_general_adr | available | 380131 | 8225 | 736 |
| mendeley_pregnancy_v2 | available | 2199 | 116 | 62 |
| dryad_antiepileptic_pregnancy | available_aggregate_tables_only | 98 | 3 | 3 |
| reprotox_kg | available | 687 | 309 | 288 |
| pubchem | available_cached | 1249 | 1173 | 1173 |

## Provenance and access

| source_dataset | citation_or_doi | version | license_or_access | granularity | limitation |
| --- | --- | --- | --- | --- | --- |
| onsides_general_adr | OnSIDES release v3.1.1 | v3.1.1 | Review upstream data and release terms before redistribution. | regulatory product-label adverse-effect associations | No pregnancy outcome labels; source prediction scores are not confidence. |
| mendeley_pregnancy_v2 | 10.17632/fnf4t4x2dz.2 | Version 2 | CC BY 4.0 (verified against official metadata) | drug-level aggregate spontaneous-report counts | No patient-level outcomes, exposed-pregnancy denominator, or validated negative labels. |
| dryad_antiepileptic_pregnancy | 10.5061/dryad.2z34tmpk0 | 10.5061/dryad.2z34tmpk0 | CC0 1.0 (verified against Dryad metadata) | study-level aggregate supplementary tables | No patient-level observations; ASD/ADHD analyses are not the requested pregnancy endpoints. |
| pubchem | PubChem PUG REST | retrieval time | PubChem public chemical reference service | chemical identifiers and structures | Identity/features only; not outcome evidence. |
| reprotox_kg | ReproTox-KG graph serialization | version field retained from graph when available | License is not stated in the downloaded graph; review upstream terms. | curated/computational knowledge-graph associations | Associations are not observed patient outcomes or incidence estimates. |

## Pairwise drug overlap

| source_a | source_b | unique_drugs_a | unique_drugs_b | overlapping_drug_count |
| --- | --- | --- | --- | --- |
| onsides_general_adr | mendeley_pregnancy_v2 | 8225 | 116 | 59 |
| onsides_general_adr | dryad_antiepileptic_pregnancy | 8225 | 3 | 2 |
| onsides_general_adr | reprotox_kg | 8225 | 309 | 159 |
| onsides_general_adr | pubchem | 8225 | 10241 | 8225 |
| mendeley_pregnancy_v2 | dryad_antiepileptic_pregnancy | 116 | 3 | 0 |
| mendeley_pregnancy_v2 | reprotox_kg | 116 | 309 | 23 |
| mendeley_pregnancy_v2 | pubchem | 116 | 10241 | 116 |
| dryad_antiepileptic_pregnancy | reprotox_kg | 3 | 309 | 3 |
| dryad_antiepileptic_pregnancy | pubchem | 3 | 10241 | 3 |
| reprotox_kg | pubchem | 309 | 10241 | 309 |

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

The Mendeley source contains drug-level aggregate spontaneous-report counts without exposed-pregnancy denominators or patient-level non-events. These are reporting signals only, not incidence or validated binary labels. OnSIDES contributes regulatory product-label adverse-effect associations and source model scores; those scores are not confidence estimates or pregnancy outcome labels. ReproTox-KG birth-defect edges are curated/computational knowledge-graph associations, not observed pregnancy events. PubChem contributes chemical identity and descriptors only.

## Potential leakage

The version, DOI, data-use note, evidence granularity, and limitations for each source are recorded in `../data/source_registry.csv`. Exact canonical-drug overlaps are reported in `../data/drug_overlap.csv`; this measures entity overlap, not shared patients. No individual-level cohort data are available here to test patient overlap or deduplicate outcomes across studies. Drug-level entity overlap is expected across chemical and safety sources, so any future model evaluation must separate canonical drug IDs (and consider scaffold grouping) across train/test sets. Do not use Mendeley report counts as features when evaluating labels derived from the same reports or cohort.

## Identifier matching

| source_dataset | source_identifier_rows | unique_source_drugs | pubchem_matched_drugs | pubchem_match_percent |
| --- | --- | --- | --- | --- |
| dryad_antiepileptic_pregnancy | 3 | 3 | 3 | 100.0 |
| existing_project_mapping | 1725 | 1725 | 1179 | 68.35 |
| mendeley_pregnancy_v2 | 209 | 116 | 62 | 53.45 |
| onsides_general_adr | 19067 | 12303 | 3782 | 30.74 |
| pubchem | 13202 | 13202 | 4249 | 32.18 |
| reprotox_kg | 310 | 309 | 288 | 93.2 |

## Trimester and missingness

### Recorded trimester availability

| source_dataset | trimester | record_count |
| --- | --- | --- |
| Dryad 10.5061/dryad.2z34tmpk0 | unknown | 98 |
| Mendeley Data | unknown | 116 |
| Mendeley Data 10.17632/fnf4t4x2dz.2 | unknown | 2083 |
| OnSIDES | unknown | 380131 |
| ReproTox-KG | unknown | 687 |

### Master missingness (top 15 fields)

| field | missing_count | missing_percent |
| --- | --- | --- |
| gestational_weeks | 383115 | 100.0 |
| association_value | 383115 | 100.0 |
| sample_label_rxnorm_name | 383115 | 100.0 |
| source_label_source_sample | 383115 | 100.0 |
| source_label_url_sample | 383115 | 100.0 |
| source_product_id_sample | 383115 | 100.0 |
| study_endpoint | 383103 | 99.9969 |
| source_table_id | 383017 | 99.9744 |
| outcome_semantics_verified | 383017 | 99.9744 |
| table_caption_or_row_text | 383017 | 99.9744 |
| is_requested_pregnancy_endpoint | 383017 | 99.9744 |
| individual_outcome_observation | 383017 | 99.9744 |
| source_row_number | 383017 | 99.9744 |
| publication_doi | 383017 | 99.9744 |
| raw_record_json | 383017 | 99.9744 |

Trimester values are used only when explicitly documented as a trimester or gestational weeks; otherwise they remain `unknown`. See `../data/trimester_summary.csv` and `../data/missingness_summary.csv`. The evidence availability matrices distinguish `source_unavailable`, `source_available_no_relevant_record`, `outcome_not_reported`, and positive concern evidence. No source currently provides validated negative human outcome observations.

## Additional data required

For each target endpoint, obtain a documented human pregnancy cohort or controlled study with:

1. Patient- or pregnancy-level exposure and follow-up records, or validated drug-level event and non-event counts with denominators.
2. Endpoint-specific definitions and observed positive and negative outcomes; do not merge unrelated endpoints.
3. Drug exposure timing, gestational weeks/trimester when documented, comparator definition, and relevant confounders.
4. Linkable drug identifiers and sufficient independent drugs/cohorts for leakage-resistant drug/scaffold-separated evaluation.
5. Provenance, validation procedures, license/permissions, and a clear missing-data mechanism.

Until then, keep the existing pregnancy-model training safeguard enabled. Do not report accuracy, recall, precision, calibration, or confidence for a pregnancy predictor.
