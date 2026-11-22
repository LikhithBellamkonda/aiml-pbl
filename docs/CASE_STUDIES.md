# Case Studies: Known Pregnancy Drug-Safety Signals

This document presents case studies for drugs with well-established pregnancy
safety profiles. These serve as positive controls (known teratogens) and
negative controls (known safe drugs) to demonstrate where the model succeeds,
where it fails, and how evidence reliability scoring behaves.

## Positive Controls (Known Teratogens / High Risk)

### Isotretinoin (CID000444795)
- **Known risk:** Severe congenital anomalies (craniofacial, cardiac, thymic)
- **Most vulnerable trimester:** First Trimester (T1) - organogenesis
- **Evidence level:** High (FAERS + Mendeley + FDA labels)
- **Expected model output:** High foetus risk in T1, evidence quality High
- **Model behavior:** Should predict high risk with high evidence quality

### Valproic Acid / Valproate (CID000002812)
- **Known risk:** Neural tube defects, craniofacial anomalies, developmental delay
- **Most vulnerable trimester:** First Trimester (T1)
- **Evidence level:** High (FAERS + Mendeley + FDA labels)
- **Expected model output:** High foetus risk in T1, evidence quality High

### Thalidomide (CID000004946)
- **Known risk:** Phocomelia (limb reduction defects)
- **Most vulnerable trimester:** First Trimester (T1)
- **Evidence level:** High (historical, FDA labels)
- **Expected model output:** High foetus risk in T1
- **Note:** Historical teratogen; may have limited modern FAERS reports due to strict controls

### Warfarin (CID000004189)
- **Known risk:** Fetal warfarin syndrome (nasal hypoplasia, stippled epiphyses)
- **Most vulnerable trimester:** First Trimester (T1)
- **Evidence level:** High (FAERS + Mendeley + FDA labels)
- **Expected model output:** High foetus risk in T1

### Phenytoin (CID000003365)
- **Known risk:** Fetal hydantoin syndrome (cleft lip/palate, growth restriction)
- **Most vulnerable trimester:** First Trimester (T1)
- **Evidence level:** High (FAERS + Mendeley + FDA labels)

### Carbamazepine (CID0000047576)
- **Known risk:** Spina bifida, craniofacial defects
- **Most vulnerable trimester:** First Trimester (T1)
- **Evidence level:** High (FAERS + Mendeley + FDA labels)

### ACE Inhibitors (e.g., Lisinopril, CID000005526)
- **Known risk:** Fetal renal dysfunction, oligohydramnios, skull hypoplasia
- **Most vulnerable trimester:** Second Trimester (T2) / Third Trimester (T3)
- **Evidence level:** High (FAERS + Mendeley + FDA labels)
- **Expected model output:** High risk in T2/T3, lower in T1

### Spironolactone (CID000005833)
- **Known risk:** Feminization of male fetus (anti-androgen effects)
- **Most vulnerable trimester:** Second Trimester (T2)
- **Evidence level:** Moderate (FAERS + Mendeley)

## Negative Controls (Known Safe)

### Metformin (CID000006741)
- **Known risk:** None established
- **Most vulnerable trimester:** Entire Pregnancy (no specific trimester)
- **Evidence level:** High (FAERS + Mendeley)
- **Expected model output:** Low risk across all trimesters, evidence quality High

### Levothyroxine (CID000006167)
- **Known risk:** None (essential for maternal hypothyroidism)
- **Most vulnerable trimester:** Entire Pregnancy
- **Evidence level:** High
- **Expected model output:** Low risk, evidence quality High

### Heparin (CID000005754)
- **Known risk:** None (does not cross placenta)
- **Most vulnerable trimester:** Entire Pregnancy
- **Evidence level:** High
- **Expected model output:** Low risk, evidence quality High

### Acetaminophen / Paracetamol (CID000003652)
- **Known risk:** None (first-line analgesic in pregnancy)
- **Most vulnerable trimester:** Entire Pregnancy
- **Evidence level:** High
- **Expected model output:** Low risk, evidence quality High

### Amoxicillin (CID000005526)
- **Known risk:** None
- **Most vulnerable trimester:** Entire Pregnancy
- **Evidence level:** High
- **Expected model output:** Low risk, evidence quality High

### Insulin (CID000004946)
- **Known risk:** None (does not cross placenta)
- **Most vulnerable trimester:** Entire Pregnancy
- **Evidence level:** High
- **Expected model output:** Low risk, evidence quality High

## Trimester-Dependent Risk (Different Risk by Trimester)

### Ibuprofen (CID000004201)
- **Known risk:** Oligohydramnios, premature closure of ductus arteriosus
- **Most vulnerable trimester:** Third Trimester (T3)
- **Evidence level:** Moderate
- **Expected model output:** Higher risk in T3, lower in T1/T2

### Fluoxetine / Sertraline (SSRIs, CID000005526, CID000004173)
- **Known risk:** Neonatal adaptation syndrome (mild, transient)
- **Most vulnerable trimester:** Third Trimester (T3)
- **Evidence level:** Moderate
- **Expected model output:** Higher maternal/neonatal risk in T3

## Interpretation Guide

| Model Output | Interpretation | Action |
|-------------|---------------|--------|
| High risk + High evidence quality | Strong signal, well-supported | Clinical review warranted |
| High risk + Low evidence quality | Hypothesis-generating only | Needs investigation, not actionable |
| Low risk + High evidence quality | Reassuring evidence | Can be considered with clinical judgment |
| Low risk + Low evidence quality | Absence of evidence ≠ evidence of absence | Caution, more data needed |
| Evidence gap flag | Understudied drug | Prioritize for data collection |

## Limitations

- These case studies reflect historical and aggregate evidence, not individual patient outcomes
- Spontaneous reporting data (FAERS, OnSIDES) has stimulated reporting and under-reporting biases
- Associations are not causal; the model is a screening/prioritization tool
- "Safe" drugs may still carry individual patient risks; always consult clinical guidelines
- Evidence quality scores are exploratory and not clinically validated

## Known Teratogen Reference List

For expert validation, the following drugs should be reviewed by a clinical
pharmacologist or obstetric expert:

1. Isotretinoin - confirmed human teratogen
2. Valproate - confirmed human teratogen
3. Thalidomide - confirmed human teratogen
4. Warfarin - confirmed human teratogen
5. Phenytoin - confirmed human teratogen
6. Carbamazepine - confirmed human teratogen
7. ACE inhibitors - confirmed fetotoxic in T2/T3
8. Metformin - considered safe
9. Levothyroxine - considered safe
10. Heparin - considered safe (does not cross placenta)
