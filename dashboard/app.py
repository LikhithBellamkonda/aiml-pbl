from pathlib import Path
import sys

import pandas as pd
import streamlit as st
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard.data_access import (  # noqa: E402
    find_drug,
    load_dashboard_data,
    records_for_drug,
)
from src.data_preprocessing import OUTCOME_NAMES  # noqa: E402
from src.models.framework import rank_outcomes  # noqa: E402


OUTCOME_LABELS = {
    "fetal_death": "Fetal death",
    "preterm_birth": "Preterm birth / prematurity",
    "fetal_growth_restriction": "Fetal growth restriction",
    "congenital_anomaly": "Congenital anomaly",
}


@st.cache_data
def load_data():
    return load_dashboard_data(ROOT)


def has_evaluated_model(data):
    metadata = data.get("metadata", {})
    metrics = data.get("metrics", {})
    checkpoint_path = ROOT / "src" / "models" / "pregnancy_adr_model.pt"
    return bool(
        metadata
        and metadata.get("outcomes") == list(OUTCOME_LABELS)
        and checkpoint_path.exists()
        and isinstance(metrics.get("model"), dict)
        and metrics["model"]
    )


@st.cache_resource
def load_trained_model():
    from src.data_preprocessing import PregnancyDataset
    from src.models.framework import MultimodalPregnancyADRModel

    metadata_path = ROOT / "src" / "models" / "model_metadata.json"
    checkpoint_path = ROOT / "src" / "models" / "pregnancy_adr_model.pt"
    if not metadata_path.exists() or not checkpoint_path.exists():
        return None, None
    metadata = load_data()["metadata"]
    expected = list(OUTCOME_LABELS)
    if metadata.get("outcomes") != expected:
        return None, None

    model = MultimodalPregnancyADRModel(num_outcomes=len(expected))
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )
    model.eval()
    dataset = PregnancyDataset(
        ROOT / "data" / "real_faers_adr.csv",
        ROOT / "data" / "real_pubchem.csv",
        ROOT / "data" / "real_toxicity.csv",
        pregnancy_path=ROOT / "data" / "real_pregnancy_features.csv",
        labels_path=ROOT / "data" / "pregnancy_labels.csv",
        faers_path=ROOT / "data" / "real_faers_summary.csv",
    )
    return model, dataset


def _show_molecular_data(pubchem_row):
    smiles = pubchem_row.get("smiles")
    if not isinstance(smiles, str) or not smiles.strip():
        st.info("Molecular structure is unavailable; no structure was imputed.")
        return
    st.code(smiles, language=None)
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Draw, rdFingerprintGenerator
    except ImportError:
        st.warning("RDKit is not installed, so molecular descriptors and fingerprints are unavailable.")
        return
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        st.warning("The stored SMILES could not be parsed by RDKit.")
        return
    left, right = st.columns(2)
    with left:
        st.image(Draw.MolToImage(molecule), caption="PubChem molecular structure")
    with right:
        descriptor_values = {
            "Molecular weight": round(Descriptors.MolWt(molecule), 3),
            "LogP": round(Descriptors.MolLogP(molecule), 3),
            "TPSA": round(Descriptors.TPSA(molecule), 3),
            "H-bond donors": Descriptors.NumHDonors(molecule),
            "H-bond acceptors": Descriptors.NumHAcceptors(molecule),
            "Rotatable bonds": Descriptors.NumRotatableBonds(molecule),
        }
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fingerprint = generator.GetFingerprint(molecule)
        descriptor_values["Morgan fingerprint on-bits (2048-bit)"] = int(fingerprint.GetNumOnBits())
        st.dataframe(pd.DataFrame([descriptor_values]), hide_index=True, use_container_width=True)


def main():
    st.set_page_config(page_title="MEPRank Research Dashboard", layout="wide")
    st.title("Pregnancy-Related Drug Safety Research Dashboard")
    st.warning(
        "This application is a research-support prototype. Predictions are not clinical "
        "diagnoses, verified adverse-event findings, or prescribing recommendations."
    )

    data = load_data()
    mapping = data["mapping"]
    if mapping.empty:
        st.error(
            "Processed drug data is missing. From the project root, run "
            "`python src/process_real_datasets.py` first."
        )
        return

    st.markdown("### Validation and model readiness")
    feasibility = data["feasibility"]
    readiness = feasibility.get("training_readiness", {})
    evaluated_model = has_evaluated_model(data)
    software_col, dataset_col, training_col, evaluation_col, uncertainty_col = st.columns(5)
    with software_col:
        st.metric("Software pipeline", "Separate test suite")
    with dataset_col:
        st.metric(
            "Dataset validation",
            "Audit available" if feasibility else "Not generated",
        )
    with training_col:
        labels_present = (ROOT / "data" / "pregnancy_labels.csv").exists()
        training_status = (
            "Trained"
            if evaluated_model
            else "Labels to validate"
            if labels_present
            else "Blocked"
            if str(readiness.get("status", "")).startswith("pregnancy_model_blocked")
            or readiness.get("status") == "blocked"
            else "Not verified"
        )
        st.metric("Model training", training_status)
    with evaluation_col:
        st.metric("Model evaluation", "Available" if evaluated_model else "None")
    with uncertainty_col:
        st.metric("Prediction uncertainty", "Available" if evaluated_model else "N/A")

    with st.expander("What these statuses mean", expanded=True):
        st.write(
            "**Software pipeline validation** checks implementation behavior only. "
            "Run `python -m pytest -q` to execute the tests; passing tests are not model "
            "accuracy or clinical validation."
        )
        if feasibility:
            st.write(
                f"**Dataset validation:** feasibility audit generated "
                f"{feasibility.get('generated_at', 'at an unknown time')}. "
                "See `reports/pregnancy_dataset_feasibility.md` for source counts, "
                "label limitations, identifier coverage, and target decisions."
            )
        else:
            st.warning(
                "No dataset feasibility report is available. Run "
                "`python -m src.dataset_feasibility`."
            )
        if not evaluated_model:
            st.write(
                "**Model training and evaluation:** no compatible evaluated model or "
                "test metrics are available. No accuracy, precision, recall, confidence, "
                "or risk score is inferred."
            )
            if readiness.get("reason"):
                st.warning(readiness["reason"])
            if labels_present:
                st.warning(
                    "`data/pregnancy_labels.csv` exists, but its labels and training "
                    "results have not been verified by this dashboard."
                )
            st.write(
                "**Prediction uncertainty:** unavailable until a model is trained and "
                "evaluated. Evidence availability and metadata completeness are not "
                "confidence scores."
            )
        else:
            st.write(
                "**Prediction uncertainty:** MC-dropout variance is shown as an "
                "uncertainty signal, not a calibrated confidence interval."
            )
            model_metrics = data["metrics"]["model"]
            metric_rows = [
                {"Outcome": OUTCOME_LABELS.get(name, name), **values}
                for name, values in model_metrics.items()
                if isinstance(values, dict)
            ]
            if metric_rows:
                st.markdown("**Held-out model evaluation results**")
                st.dataframe(
                    pd.DataFrame(metric_rows),
                    hide_index=True,
                    use_container_width=True,
                )

    toxicity_metrics = data["toxicity_metrics"]
    if toxicity_metrics.get("endpoints"):
        st.markdown("### Exploratory nonclinical toxicity experiments")
        st.warning(
            "These models predict the binary labels encoded in the supplied TG assay "
            "workbook using molecular fingerprints. They are not human pregnancy outcome "
            "models, and the workbook's label definitions need source validation."
        )
        st.caption(
            "Values below are scaffold-grouped cross-validation estimates, not an "
            "independent external test. Brier score and ECE are descriptive, not "
            "calibration validation."
        )
        toxicity_rows = []
        for endpoint, details in toxicity_metrics["endpoints"].items():
            for model_name, metrics in details.get("models", {}).items():
                toxicity_rows.append(
                    {
                        "Assay endpoint": endpoint,
                        "Model": model_name.replace("_", " ").title(),
                        "n": details.get("records"),
                        "Positive": details.get("positive"),
                        "Negative": details.get("negative"),
                        "Accuracy": metrics.get("accuracy"),
                        "Majority baseline accuracy": metrics.get(
                            "majority_class_accuracy"
                        ),
                        "AUROC": metrics.get("auroc"),
                        "AUPRC": metrics.get("auprc"),
                        "Precision": metrics.get("precision"),
                        "Recall": metrics.get("recall"),
                        "F1": metrics.get("f1"),
                        "Brier": metrics.get("brier_score"),
                        "ECE": metrics.get("ece"),
                    }
                )
        if toxicity_rows:
            st.dataframe(
                pd.DataFrame(toxicity_rows),
                hide_index=True,
                use_container_width=True,
            )
        st.info(
            "Training command: `python -m src.train_reproductive_tox`. "
            "This does not unlock pregnancy-outcome predictions."
        )

    query = st.text_input("Search by drug name or identifier")
    if not query.strip():
        st.info("Enter a drug name to inspect its mapped identity and supporting evidence.")
        return
    drug = find_drug(query, mapping)
    if drug is None:
        st.warning(
            "This name is not in the local mapping. No identifier or safety prediction was invented."
        )
        return

    drug_id = str(drug["drug_id"])
    st.subheader(f"{drug['original_name']}  ·  {drug_id}")
    identity = {
        "Standardized name": drug.get("normalized_name", ""),
        "PubChem CID": (
            int(drug["pubchem_cid"]) if pd.notna(drug.get("pubchem_cid")) else "Unresolved"
        ),
        "InChIKey": drug.get("inchikey") or "Unavailable",
        "Resolution status": drug.get("resolution_status", "unknown"),
    }
    st.dataframe(pd.DataFrame([identity]), hide_index=True, use_container_width=True)

    pubchem_rows = records_for_drug(data["pubchem"], drug_id)
    st.markdown("### Molecular evidence")
    if pubchem_rows.empty:
        st.info("No PubChem record is available for this drug.")
    else:
        _show_molecular_data(pubchem_rows.iloc[0])

    st.markdown("### Evidence availability")
    availability_rows = records_for_drug(data["availability"], drug_id)
    if availability_rows.empty:
        st.info("Source availability has not been processed for this drug.")
    else:
        status_row = availability_rows.iloc[0].to_dict()
        quality_columns = {
            "adr_status": ("adr_metadata_completeness",),
            "molecular_status": (
                "molecular_metadata_completeness",
                "graph_metadata_completeness",
            ),
            "toxicity_status": ("toxicity_metadata_completeness",),
            "pregnancy_outcomes_status": ("pregnancy_outcomes_metadata_completeness",),
            "faers_status": ("faers_metadata_completeness",),
        }
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Evidence source": source.replace("_status", "").replace("_", " ").title(),
                        "Status": str(status).replace("_", " ").title(),
                        "Metadata completeness": ", ".join(
                            f"{float(status_row[column]):.0%}"
                            for column in quality_columns[source]
                            if column in status_row and pd.notna(status_row[column])
                        )
                        or "Unknown",
                    }
                    for source, status in status_row.items()
                    if source.endswith("_status")
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "Metadata completeness reflects whether provenance fields were present; it is not "
            "a validated scientific evidence-quality score."
        )

    left, right = st.columns(2)
    with left:
        st.markdown("### General ADR evidence")
        adr = records_for_drug(data["adr"], drug_id)
        if adr.empty:
            st.info("No matched SIDER association is available. This is not evidence of safety.")
        else:
            st.dataframe(adr, hide_index=True, use_container_width=True)
    with right:
        st.markdown("### Developmental / reproductive study records")
        toxicity = records_for_drug(data["toxicity"], drug_id)
        if toxicity.empty:
            st.info("No exact-name-linked ToxRefDB study record is available.")
        else:
            st.caption(
                "These are study records, not toxicity-positive labels; species and dose details "
                "are preserved where supplied."
            )
            st.dataframe(toxicity, hide_index=True, use_container_width=True)

    st.markdown("### Pregnancy-specific outcome evidence")
    pregnancy = records_for_drug(data["pregnancy"], drug_id)
    if pregnancy.empty:
        st.info(
            "No outcome-level record from the supplied aggregate FAERS supplement is matched "
            "to this drug. Missing reports do not mean low risk."
        )
    else:
        display = pregnancy.copy()
        display["outcome"] = display["outcome"].map(
            lambda value: OUTCOME_LABELS.get(value, value or "Unclassified source category")
        )
        st.dataframe(display, hide_index=True, use_container_width=True)
        st.caption(
            "The supplied JAAD supplement reports aggregate spontaneous-report counts. "
            "They are not patient-level outcomes, incidence estimates, or evidence of causality."
        )

    st.markdown("### Pregnancy classification evidence")
    classifications = records_for_drug(data["classifications"], drug_id)
    if classifications.empty:
        st.info("No pregnancy classification record is available.")
    else:
        st.dataframe(classifications, hide_index=True, use_container_width=True)
        st.caption(
            "Classifications and safety statements are shown as source evidence only; they are "
            "not used as verified pregnancy-outcome labels."
        )

    st.markdown("### Historical reporting summary")
    faers_summary = records_for_drug(data["faers_summary"], drug_id)
    if faers_summary.empty:
        st.info("No aggregate FAERS report total is available for this drug.")
    else:
        report_count = faers_summary.iloc[0].get("faers_report_count")
        st.write(
            f"Reports listed in the supplied study summary: "
            f"{int(report_count) if pd.notna(report_count) else 'not reported'}"
        )
        st.caption("Spontaneous report counts are not incidence estimates and do not establish causality.")

    st.markdown("### Outcome-specific model predictions and ranking")
    model_path = ROOT / "src" / "models" / "pregnancy_adr_model.pt"
    if has_evaluated_model(data) and model_path.exists():
        if st.button("Run outcome predictions"):
            from src.data_preprocessing import collate_drugs

            model, dataset = load_trained_model()
            if model is None or drug_id not in dataset.drug_ids:
                st.error("No compatible trained model or matching model input is available.")
            else:
                sample = dataset[dataset.drug_ids.index(drug_id)]
                batch = collate_drugs([sample])
                if not model.has_predictive_evidence(batch).item():
                    st.warning("Prediction abstained: no supported evidence source is available.")
                else:
                    with torch.no_grad():
                        mean, variance = model.predict_with_uncertainty(
                            batch, num_samples=30
                        )
                    ranked = rank_outcomes(mean[0], OUTCOME_NAMES)
                    rows = [
                        {
                            "Outcome": OUTCOME_LABELS[name],
                            "Predicted probability": float(score),
                            "MC-dropout variance (uncertainty)": float(variance[0, OUTCOME_NAMES.index(name)]),
                        }
                        for name, score in ranked
                    ]
                    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
                    st.caption(
                        "Predictions are model outputs, distinct from observed evidence. "
                        "Uncertainty is MC-dropout variance; calibration is not established."
                    )
    else:
        st.info(
            "No compatible model trained on provenance-backed human pregnancy-outcome labels "
            "is available. Scores and risk rankings are therefore withheld; observed report "
            "counts above are not substituted for predicted risk."
        )


if __name__ == "__main__":
    main()
