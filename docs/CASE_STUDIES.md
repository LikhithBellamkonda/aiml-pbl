# Case Studies: What the Local Evidence Actually Contains

This document records, for a set of drugs with well-known pregnancy-safety
literature, **what the acquired local sources contain**. It is an evidence
inventory, not a prediction exercise: the project has no validated human
pregnancy-outcome model, so there is no model output to compare with the
literature.

## Method

Every row below was produced by calling the same service the dashboard uses:

```bash
python -c "from src.services.drug_service import build_drug_profile; import json; print(json.dumps(build_drug_profile('isotretinoin'), indent=2, default=str))"
```

Counts are record counts from the local files at the time of the run recorded in
`data/pipeline_run.json`:

- **Human pregnancy records** — rows in `data/real_pregnancy_outcomes.csv`
  (aggregate spontaneous-report terms, not patient-level outcomes).
- **General ADR terms** — distinct terms in `data/real_adr_evidence.csv`.
- **Nonclinical toxicity records** — OECD TG assay rows in
  `data/real_toxicity_evidence.csv`.
- **FAERS reports** — aggregate reporting total in `data/real_faers_summary.csv`.

A zero or "unavailable" cell means the local sources have no record. It is not
evidence of safety, and it is not a negative label.

## Measured coverage

| Drug asked | Resolved identity | Mapping status | Human pregnancy records | General ADR terms | Nonclinical toxicity records | FAERS reports |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| isotretinoin | `NAME:isotretinoin` | name-keyed evidence only | 222 | 0 | 0 | unavailable |
| thalidomide | CID 5426 | resolved | 0 | 402 | 1 | unavailable |
| carbamazepine | CID 2554 | resolved | 0 | 340 | 3 | unavailable |
| spironolactone | CID 5833 | resolved | 7 | 56 | 3 | 11 |
| metformin | CID 4091 | resolved | 0 | 141 | 0 | unavailable |
| amoxicillin | CID 33613 | resolved | 34 | 93 | 0 | 109 |
| ibuprofen | CID 3672 | resolved | 0 | 337 | 0 | unavailable |
| warfarin | CID 54678486 | resolved | 0 | 0 | 0 | unavailable |
| heparin | `NAME:heparin` | unresolved | 0 | 0 | 0 | unavailable |

The following names used in the pregnancy literature are **not present** in the
local identifier mapping under that exact spelling, so a lookup returns
"no local mapping entry matches": `valproic acid` (recorded as
`sodium valproate (valproic acid)`), `phenytoin` (recorded as
`phenytoin sodium`), `levothyroxine` (recorded as `thyroxine (levothyroxine)`),
`acetaminophen` (recorded as `paracetamol`) and `insulin` (recorded only as
specific analogues and combinations). They are listed here so the gap is
explicit rather than hidden.

## Observations

- **Isotretinoin** is the clearest case of human pregnancy evidence in the local
  data: 222 aggregate pregnancy-outcome records keyed by the evidence index name
  `isotretinoin`. It has no PubChem structure in the local mapping, so molecular
  and nonclinical joins are unavailable. The dashboard states this and does not
  attach a CID.
- **Thalidomide, carbamazepine, metformin and ibuprofen** have substantial
  general-ADR coverage but no human pregnancy-outcome records in the local files.
  Absence of a pregnancy record is reported as unavailable.
- **Spironolactone and amoxicillin** are the only rows above with records in
  four of the tracked human/animal families at once; they are useful for
  checking that the dashboard keeps the panels separate.
- **Warfarin** resolves to a PubChem CID but has no records in any local
  evidence table. The identifier mapping and the evidence coverage are
  independent facts, and the dashboard reports both.
- **Heparin** is keyed by name with `unresolved` status; no structure is
  available, so molecular analysis is disabled for it.

## What this does and does not support

Supported:

- Identifier resolution and its status, including ambiguity and name-only keys.
- Per-source evidence availability and record counts.
- Evidence coverage (Q_cov) and evidence-gap flags.
- Separation of human pregnancy, human ADR, animal spontaneous ADR and animal
  experimental toxicity evidence in the interface.

Not supported:

- Any statement that a drug is high or low risk in pregnancy.
- Any ranking of these drugs by predicted risk.
- Any comparison against literature effect sizes; the local records are
  aggregate spontaneous-report counts without denominators or comparators.
- Any causal interpretation. Spontaneous reporting is voluntary, under-reported
  and subject to stimulated reporting.

## Limitations

- Counts are reporting volume. They are not incidence, severity or causal
  effects, and they cannot be converted into a 2×2 table for ROR/PRR because no
  case/non-case denominators are present.
- The local identifier mapping does not cover every synonym in clinical use;
  the unmatched names above are retained explicitly rather than forced onto a
  neighbouring identifier.
- The absence of a record is never treated as a negative outcome, and no
  pregnancy risk score or confidence value is produced anywhere in this build.
