# Trimester Evaluation Report

Status: **blocked**

No validated human pregnancy-outcome labels (data/pregnancy_labels.csv) and no trained human pregnancy model exist, so trimester-wise AUROC, AUPRC, ΔAUROC, ECE, Brier and Precision@10 cannot be computed. They are reported as unavailable, not as zero.

## Metric specification

- Per-trimester: auroc, auprc, ece, brier, precision_at_10
- Cross-trimester: delta_auroc (max - min across trimesters)
- ΔAUROC > 0 indicates trimester-specific learning; ΔAUROC ≈ 0 means the model collapses to a single-stage predictor.

## Three separate signals

- **prediction**: unavailable — No trained human pregnancy-outcome model.
- **uncertainty**: unavailable — No trained model, so predictive uncertainty is undefined.
- **evidence_reliability**: available
  - Q_cov = mean over source families of (availability mask × source quality), where source quality = 0.5 × populated content fields + 0.5 × log-scaled evidence volume for that drug; drugs=1720, mean=0.3155, median=0.225, evidence-gap fraction=0.5843

## Results

No trimester metrics are reported. This is a blocked evaluation, not a zero-valued one.
