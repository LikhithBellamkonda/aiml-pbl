# Pregnancy Evidence Inventory (evidence tiers, not outcome labels)

Generated from the current local data by `python -m src.evidence_inventory`.

This inventory records **what each acquired source contains** for a fixed
set of literature reference drugs. It is an evidence-coverage audit, not a
labelled benchmark: no patient-level pregnancy outcome labels exist in this
repository, so no AUROC/AUPRC or training target is derived from these rows.
The literature reference class is a coverage grouping only and must not be
read as a validated clinical label or as a model target.

- Control drugs listed: 16
- Resolved by the local identifier mapping / evidence index: 16
- Flagged with an evidence gap (fewer than two families or low coverage): 9

## Coverage table

| Drug | Reference class | Mapping | Human pregnancy records | ADR terms | Toxicity records | FAERS reports | Trimesters |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| isotretinoin | established teratogen | name_keyed_evidence_only | 222 | 0 | 0 | unavailable | unknown |
| sodium valproate (valproic acid) | established teratogen | unresolved | 0 | 0 | 0 | unavailable | none/unknown |
| thalidomide | established teratogen | resolved | 0 | 402 | 1 | unavailable | none/unknown |
| warfarin | established teratogen | resolved | 0 | 0 | 0 | unavailable | none/unknown |
| carbamazepine | established teratogen | resolved | 0 | 340 | 3 | unavailable | none/unknown |
| phenytoin sodium | established teratogen | resolved | 0 | 0 | 0 | unavailable | none/unknown |
| lisinopril | fetotoxic in later pregnancy | resolved | 0 | 0 | 0 | unavailable | none/unknown |
| spironolactone | possible anti-androgen effect | resolved | 7 | 56 | 3 | 11 | unknown |
| ibuprofen | trimester-dependent | resolved | 0 | 337 | 0 | unavailable | none/unknown |
| sertraline | trimester-dependent | resolved | 0 | 0 | 0 | unavailable | none/unknown |
| paracetamol | generally considered lower risk | resolved | 0 | 187 | 0 | unavailable | none/unknown |
| amoxicillin | generally considered lower risk | resolved | 34 | 93 | 0 | 109 | unknown |
| metformin | generally considered lower risk | resolved | 0 | 141 | 0 | unavailable | none/unknown |
| heparin | generally considered lower risk | unresolved | 0 | 0 | 0 | unavailable | none/unknown |
| thyroxine (levothyroxine) | generally considered lower risk | unresolved | 0 | 0 | 0 | unavailable | none/unknown |
| insulin glargine | generally considered lower risk | resolved | 0 | 0 | 0 | unavailable | none/unknown |

## Unmatched controls

None: every control drug above resolves in the local mapping.

## What this supports

- Auditing which evidence families cover which drugs, and where the gaps are.
- Selecting a small, manually reviewable subset of drugs for expert review.
- Reproducible coverage comparisons when new sources are added.

## What this does not support

- Any pregnancy risk prediction, ranking or model metric.
- Treating a literature reference class as a supervised label.
- Converting spontaneous-report counts into incidence or causal effects.
