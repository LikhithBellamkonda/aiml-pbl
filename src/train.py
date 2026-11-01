from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader, Subset

from src.data_preprocessing import (
    LABEL_PROVENANCE_COLUMNS,
    OUTCOME_NAMES,
    PregnancyDataset,
    collate_drugs,
)
from src.models.framework import MultimodalPregnancyADRModel


ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("pregnancy_training")


def expected_calibration_error(y_true, probabilities, bins=10):
    y_true = np.asarray(y_true, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if not len(y_true):
        return None
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        in_bin = (probabilities >= lower) & (
            probabilities <= upper if upper == 1 else probabilities < upper
        )
        if in_bin.any():
            error += in_bin.mean() * abs(y_true[in_bin].mean() - probabilities[in_bin].mean())
    return float(error)


def binary_metrics(y_true, probabilities, precision_at=10):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if y_true.size == 0:
        return {"n": 0}
    order = np.argsort(probabilities)[::-1]
    top = order[: min(precision_at, len(order))]
    positives = int(y_true.sum())
    result = {
        "n": int(len(y_true)),
        "f1": float(f1_score(y_true, probabilities >= 0.5, zero_division=0)),
        "precision": float(precision_score(y_true, probabilities >= 0.5, zero_division=0)),
        "recall": float(recall_score(y_true, probabilities >= 0.5, zero_division=0)),
        "precision_at_k": float(y_true[top].mean()) if len(top) else None,
        "recall_at_k": float(y_true[top].sum() / positives) if positives else None,
        "brier_score": float(np.mean((probabilities - y_true) ** 2)),
        "ece": expected_calibration_error(y_true, probabilities),
        "auroc": (
            float(roc_auc_score(y_true, probabilities))
            if len(np.unique(y_true)) == 2
            else None
        ),
        "auprc": (
            float(average_precision_score(y_true, probabilities))
            if len(np.unique(y_true)) == 2
            else None
        ),
    }
    return result


def save_reliability_diagram(y_true, probabilities, path):
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(np.unique(y_true)) < 2:
        raise ValueError("A reliability diagram requires both positive and negative labels")
    observed, predicted = calibration_curve(
        y_true, probabilities, n_bins=10, strategy="uniform"
    )
    figure, axis = plt.subplots()
    axis.plot(predicted, observed, marker="o", label="Model")
    axis.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    axis.set(xlabel="Mean predicted probability", ylabel="Observed fraction")
    axis.legend()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _scaffold_groups(dataset):
    groups = []
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError:
        Chem = None

    for drug_id in dataset.drug_ids:
        smiles = dataset.merged_df.loc[drug_id].get("smiles")
        scaffold = None
        if Chem is not None and isinstance(smiles, str) and smiles:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is not None:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
        groups.append(f"scaffold:{scaffold}" if scaffold else f"drug:{drug_id}")
    return np.asarray(groups)


def grouped_train_validation_test_indices(dataset, seed=42):
    groups = _scaffold_groups(dataset)
    if len(np.unique(groups)) < 3:
        raise RuntimeError(
            "Training requires at least three independent drug/scaffold groups for "
            "train/validation/test evaluation."
        )
    all_indices = np.arange(len(dataset))
    first_split = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    train_val_idx, test_idx = next(first_split.split(all_indices, groups=groups))
    second_split = GroupShuffleSplit(
        n_splits=1, test_size=0.15 / 0.85, random_state=seed + 1
    )
    train_rel, val_rel = next(
        second_split.split(train_val_idx, groups=groups[train_val_idx])
    )
    return train_val_idx[train_rel], train_val_idx[val_rel], test_idx


def _evidence_vector(sample):
    return torch.cat(
        (
            sample.adr_features.flatten(),
            sample.mol_desc.flatten(),
            sample.pregnancy_features.flatten(),
            sample.faers_features.flatten(),
            sample.availability_mask.flatten(),
            sample.evidence_quality.flatten(),
        )
    ).numpy()


def evaluate_classical_baselines(dataset, train_indices, test_indices):
    x_train = np.stack([_evidence_vector(dataset[index]) for index in train_indices])
    x_test = np.stack([_evidence_vector(dataset[index]) for index in test_indices])
    source_slices = {
        "general_adr": slice(0, 20),
        "chemical_descriptors": slice(20, 26),
        "pregnancy_outcomes": slice(26, 30),
        "faers_reporting": slice(30, 31),
    }
    source_indices = {
        "general_adr": 0,
        "chemical_descriptors": 1,
        "pregnancy_outcomes": 3,
        "faers_reporting": 4,
    }
    source_train = {
        name: np.stack(
            [
                torch.cat(
                    (
                        dataset[index].adr_features.flatten(),
                        dataset[index].mol_desc.flatten(),
                        dataset[index].pregnancy_features.flatten(),
                        dataset[index].faers_features.flatten(),
                    )
                ).numpy()[feature_slice]
                for index in train_indices
            ]
        )
        for name, feature_slice in source_slices.items()
    }
    source_test = {
        name: np.stack(
            [
                torch.cat(
                    (
                        dataset[index].adr_features.flatten(),
                        dataset[index].mol_desc.flatten(),
                        dataset[index].pregnancy_features.flatten(),
                        dataset[index].faers_features.flatten(),
                    )
                ).numpy()[feature_slice]
                for index in test_indices
            ]
        )
        for name, feature_slice in source_slices.items()
    }
    train_availability = np.stack(
        [dataset[index].availability_mask.numpy() for index in train_indices]
    )
    test_availability = np.stack(
        [dataset[index].availability_mask.numpy() for index in test_indices]
    )
    results = {}
    for outcome_index, outcome in enumerate(OUTCOME_NAMES):
        train_labels = np.asarray(
            [dataset[index].y[outcome_index].item() for index in train_indices], dtype=int
        )
        train_masks = np.asarray(
            [dataset[index].y_mask[outcome_index].item() for index in train_indices], dtype=bool
        )
        test_labels = np.asarray(
            [dataset[index].y[outcome_index].item() for index in test_indices], dtype=int
        )
        test_masks = np.asarray(
            [dataset[index].y_mask[outcome_index].item() for index in test_indices], dtype=bool
        )
        eligible_train = train_masks
        eligible_test = test_masks
        if len(np.unique(train_labels[eligible_train])) < 2 or not eligible_test.any():
            results[outcome] = {"status": "insufficient labeled training or test data"}
            continue
        for model_name, classifier in (
            (
                "feature_concatenation_logistic_regression",
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
            (
                "feature_concatenation_random_forest",
                RandomForestClassifier(
                    n_estimators=300, random_state=42, class_weight="balanced"
                ),
            ),
        ):
            classifier.fit(x_train[eligible_train], train_labels[eligible_train])
            probabilities = classifier.predict_proba(x_test[eligible_test])[:, 1]
            results.setdefault(outcome, {})[model_name] = binary_metrics(
                test_labels[eligible_test], probabilities
            )
        source_probabilities = []
        for source_name, source_index in source_indices.items():
            available_train = train_masks & train_availability[:, source_index].astype(bool)
            available_test = test_masks & test_availability[:, source_index].astype(bool)
            if (
                available_train.sum() < 2
                or len(np.unique(train_labels[available_train])) < 2
                or not available_test.any()
            ):
                continue
            classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
            classifier.fit(
                source_train[source_name][available_train],
                train_labels[available_train],
            )
            probabilities = classifier.predict_proba(
                source_test[source_name][available_test]
            )[:, 1]
            results.setdefault(outcome, {})[f"individual_{source_name}"] = binary_metrics(
                test_labels[available_test], probabilities
            )
            aligned_probabilities = np.full(len(test_masks), np.nan, dtype=float)
            aligned_probabilities[available_test] = probabilities
            source_probabilities.append(aligned_probabilities)

        if source_probabilities:
            stacked = np.vstack(source_probabilities)
            weights = np.isfinite(stacked).sum(axis=0)
            weighted_scores = np.nansum(stacked, axis=0)
            included = test_masks & (weights > 0)
            results.setdefault(outcome, {})["equal_weight_average"] = binary_metrics(
                test_labels[included],
                weighted_scores[included] / weights[included],
            )

        eligible_train_labels = train_labels[eligible_train]
        training_groups = _scaffold_groups(dataset)[train_indices][eligible_train]
        class_counts = np.bincount(eligible_train_labels, minlength=2)
        unique_groups = np.unique(training_groups)
        if (
            len(eligible_train_labels) >= 20
            and class_counts.min() >= 5
            and len(unique_groups) >= 3
            and eligible_test.any()
        ):
            cv = min(5, len(unique_groups))
            x_stack = x_train[eligible_train]
            group_kfold = GroupKFold(n_splits=cv)
            oof = np.full((len(eligible_train_labels), 2), np.nan)
            valid_folds = True
            for fold_train, fold_validation in group_kfold.split(
                x_stack, eligible_train_labels, groups=training_groups
            ):
                if len(np.unique(eligible_train_labels[fold_train])) < 2:
                    valid_folds = False
                    break
                fold_models = (
                    LogisticRegression(max_iter=1000, class_weight="balanced"),
                    RandomForestClassifier(
                        n_estimators=100, random_state=42, class_weight="balanced"
                    ),
                )
                for model_index, classifier in enumerate(fold_models):
                    classifier.fit(x_stack[fold_train], eligible_train_labels[fold_train])
                    oof[fold_validation, model_index] = classifier.predict_proba(
                        x_stack[fold_validation]
                    )[:, 1]
            has_oof = np.isfinite(oof).all(axis=1)
            if valid_folds and has_oof.any() and len(np.unique(eligible_train_labels[has_oof])) == 2:
                stacker = LogisticRegression(max_iter=1000)
                stacker.fit(oof[has_oof], eligible_train_labels[has_oof])
                full_models = (
                    LogisticRegression(max_iter=1000, class_weight="balanced"),
                    RandomForestClassifier(
                        n_estimators=100, random_state=42, class_weight="balanced"
                    ),
                )
                full_probabilities = []
                for classifier in full_models:
                    classifier.fit(x_stack, eligible_train_labels)
                    full_probabilities.append(
                        classifier.predict_proba(x_test[eligible_test])[:, 1]
                    )
                stacked_probabilities = stacker.predict_proba(
                    np.column_stack(full_probabilities)
                )[:, 1]
                results.setdefault(outcome, {})["grouped_stacking"] = binary_metrics(
                    test_labels[eligible_test], stacked_probabilities
                )
            else:
                results.setdefault(outcome, {})["grouped_stacking"] = {
                    "status": "insufficient class variation across training groups"
                }
        else:
            results.setdefault(outcome, {})["grouped_stacking"] = {
                "status": "insufficient labeled data for grouped stacking"
            }
    return results


def _collect_predictions(model, loader, device):
    model.eval()
    predictions, targets, masks = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions.append(model(batch).cpu())
            targets.append(batch.y.cpu())
            masks.append(batch.y_mask.cpu())
    return torch.cat(predictions), torch.cat(targets), torch.cat(masks)


def evaluate_missing_evidence(model, loader, device):
    allowed_sources = {
        "complete_available_sources": None,
        "adr_and_chemical_only": (0, 1, 2),
        "without_chemical_features": (0, 3, 4),
        "without_pregnancy_outcomes": (0, 1, 2, 4),
        "without_faers": (0, 1, 2, 3),
    }
    collected = {scenario: ([], [], []) for scenario in allowed_sources}
    collected["half_with_pregnancy_source_masked"] = ([], [], [])
    for batch in loader:
        batch = batch.to(device)
        original_mask = batch.availability_mask.clone()
        scenarios = dict(allowed_sources)
        sparse_mask = original_mask.clone()
        sparse_mask[1::2, 3] = 0
        scenarios["half_with_pregnancy_source_masked"] = None
        for scenario, allowed in scenarios.items():
            if scenario == "half_with_pregnancy_source_masked":
                batch.availability_mask = sparse_mask.clone()
            elif allowed is None:
                batch.availability_mask = original_mask.clone()
            else:
                batch.availability_mask = original_mask.clone()
                keep = torch.zeros_like(batch.availability_mask)
                keep[:, list(allowed)] = 1
                batch.availability_mask *= keep
            with torch.no_grad():
                collected[scenario][0].append(model(batch).cpu())
            collected[scenario][1].append(batch.y.cpu())
            collected[scenario][2].append(batch.y_mask.cpu())

    results = {}
    for scenario, (prediction_parts, target_parts, mask_parts) in collected.items():
        predictions = torch.cat(prediction_parts)
        targets = torch.cat(target_parts)
        masks = torch.cat(mask_parts)
        results[scenario] = {}
        for index, outcome in enumerate(OUTCOME_NAMES):
            valid = masks[:, index].bool().numpy()
            results[scenario][outcome] = (
                binary_metrics(
                    targets[valid, index].numpy(), predictions[valid, index].numpy()
                )
                if valid.any()
                else {"n": 0, "status": "no test labels"}
            )
    return results


def train_model(epochs=30, batch_size=32):
    dataset = PregnancyDataset(
        ROOT / "data" / "real_faers_adr.csv",
        ROOT / "data" / "real_pubchem.csv",
        pregnancy_path=ROOT / "data" / "real_pregnancy_features.csv",
        labels_path=ROOT / "data" / "pregnancy_labels.csv",
        faers_path=ROOT / "data" / "real_faers_summary.csv",
    )
    observed = sum(
        int(dataset.merged_df[mask].fillna(0).sum()) if mask in dataset.merged_df else 0
        for mask in dataset.mask_cols
    )
    if observed == 0:
        raise RuntimeError(
            "Training skipped: no validated binary human pregnancy-outcome labels are available. "
            "Aggregate FAERS counts and pregnancy classifications are evidence, not negative or "
            "positive training labels. Add a provenance-backed data/pregnancy_labels.csv first."
        )
    missing_provenance = set(LABEL_PROVENANCE_COLUMNS) - set(dataset.labels_df.columns)
    if missing_provenance:
        raise RuntimeError(
            "Training skipped: pregnancy_labels.csv is missing provenance columns: "
            + ", ".join(sorted(missing_provenance))
        )
    observed_rows = dataset.labels_df[
        [column for column in dataset.mask_cols if column in dataset.labels_df]
    ].fillna(0).astype(bool).any(axis=1)
    provenance_rows = dataset.labels_df.loc[observed_rows, list(LABEL_PROVENANCE_COLUMNS)]
    if provenance_rows.isna().any().any() or provenance_rows.astype(str).apply(
        lambda column: column.str.strip().eq("").any()
    ).any():
        raise RuntimeError(
            "Training skipped: every observed label needs non-empty source dataset, "
            "source reference, outcome definition, and evidence type."
        )
    supported_evidence_types = {"human_observed_outcome", "human_controlled_outcome"}
    if not set(provenance_rows["evidence_type"]).issubset(supported_evidence_types):
        raise RuntimeError(
            "Training skipped: labels must represent observed human pregnancy outcomes; "
            "classifications and spontaneous-report counts are not valid label evidence."
        )

    train_indices, val_indices, test_indices = grouped_train_validation_test_indices(dataset)
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_drugs,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_drugs,
    )
    test_loader = DataLoader(
        Subset(dataset, test_indices),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_drugs,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalPregnancyADRModel(num_outcomes=len(OUTCOME_NAMES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = nn.BCELoss(reduction="none")
    best_val_loss = float("inf")
    checkpoint_path = ROOT / "src" / "models" / "pregnancy_adr_model.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_saved = False

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            loss_values = loss_function(model(batch), batch.y) * batch.y_mask
            denominator = batch.y_mask.sum()
            if denominator.item() == 0:
                continue
            optimizer.zero_grad()
            (loss_values.sum() / denominator).backward()
            optimizer.step()

        model.eval()
        val_loss_total, val_label_total = 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                mask = batch.y_mask
                val_loss_total += (loss_function(model(batch), batch.y) * mask).sum().item()
                val_label_total += mask.sum().item()
        if val_label_total and val_loss_total / val_label_total < best_val_loss:
            best_val_loss = val_loss_total / val_label_total
            torch.save(model.state_dict(), checkpoint_path)
            checkpoint_saved = True
        LOGGER.info("Epoch %d/%d validation masked loss=%.6f", epoch + 1, epochs, best_val_loss)

    if not checkpoint_saved:
        raise RuntimeError("Training completed without a checkpoint; validation labels were unavailable.")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    prediction, target, mask = _collect_predictions(model, test_loader, device)
    final_metrics = {}
    for index, outcome in enumerate(OUTCOME_NAMES):
        valid = mask[:, index].bool().numpy()
        final_metrics[outcome] = (
            binary_metrics(target[valid, index].numpy(), prediction[valid, index].numpy())
            if valid.any()
            else {"n": 0, "status": "no test labels"}
        )

    baseline_metrics = evaluate_classical_baselines(dataset, train_indices, test_indices)
    missing_evidence_metrics = evaluate_missing_evidence(model, test_loader, device)
    result = {
        "split_method": "scaffold when parseable, otherwise drug-level groups",
        "train_drugs": len(train_indices),
        "validation_drugs": len(val_indices),
        "test_drugs": len(test_indices),
        "model": final_metrics,
        "baselines": baseline_metrics,
        "missing_evidence_scenarios": missing_evidence_metrics,
    }
    metrics_path = ROOT / "data" / "training_metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for index, outcome in enumerate(OUTCOME_NAMES):
        valid = mask[:, index].bool().numpy()
        if valid.any() and len(np.unique(target[valid, index].numpy())) == 2:
            save_reliability_diagram(
                target[valid, index].numpy(),
                prediction[valid, index].numpy(),
                ROOT / "data" / f"reliability_{outcome}.png",
            )
    (ROOT / "src" / "models" / "model_metadata.json").write_text(
        json.dumps(
            {
                "model_type": "MEPRank research prototype",
                "outcomes": list(OUTCOME_NAMES),
                "training_labels": "provenance-backed binary human outcomes",
                "split_method": result["split_method"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    LOGGER.info("Saved model to %s", checkpoint_path)
    LOGGER.info("Saved evaluation metrics to %s", metrics_path)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        train_model()
    except RuntimeError as error:
        LOGGER.error("%s", error)
        raise SystemExit(2) from error
