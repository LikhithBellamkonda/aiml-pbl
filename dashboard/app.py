from pathlib import Path
import sys

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard.data_access import load_dashboard_data  # noqa: E402
from src.mepr_reasoning import evaluate_drug_safety  # noqa: E402


@st.cache_data
def load_data():
    return load_dashboard_data(ROOT)


@st.cache_data(show_spinner="Filtering evidence for the selected query...")
def _evaluate_query(drug_name: str, drug_id: str, trimester: str, target: str):
    return evaluate_drug_safety(
        drug_name,
        trimester=trimester,
        outcome_target=target,
        drug_id=drug_id,
    )


def has_evaluated_model(data):
    metadata = data.get("metadata", {})
    metrics = data.get("metrics", {})
    decisions = data.get("feasibility", {}).get("target_decisions", [])
    checkpoint_path = ROOT / "src" / "models" / "pregnancy_adr_model.pt"
    expected_outcomes = {
        "fetal_death",
        "preterm_birth",
        "fetal_growth_restriction",
        "congenital_anomaly",
    }
    trainable_outcomes = {
        item.get("outcome")
        for item in decisions
        if item.get("decision") == "trainable"
    }
    return bool(
        metadata
        and set(metadata.get("outcomes", [])) == expected_outcomes
        and expected_outcomes.issubset(trainable_outcomes)
        and checkpoint_path.exists()
        and isinstance(metrics.get("model"), dict)
        and metrics["model"]
    )


def _render_readiness(data):
    feasibility = data["feasibility"]
    readiness = feasibility.get("training_readiness", {})
    if not readiness and feasibility.get("target_decisions"):
        decisions = feasibility["target_decisions"]
        blocked = not any(item.get("decision") == "trainable" for item in decisions)
        readiness = {
            "status": "blocked" if blocked else "not_verified",
            "reason": decisions[0].get("reason", "") if blocked else "",
        }
    evaluated_model = has_evaluated_model(data)
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

    with st.expander("System validation and model readiness", expanded=False):
        st.write(
            "**Software validation** checks implementation behavior only. "
            "Passing tests are not model accuracy or clinical validation."
        )
        st.write(
            f"**Dataset validation:** {'Audit available' if feasibility else 'Not generated'}. "
            "Unreported outcomes are not treated as negative."
        )
        st.write(f"**Model training:** {training_status}.")
        st.write(
            f"**Model evaluation:** {'Available' if evaluated_model else 'Unavailable'}."
        )
        if readiness.get("reason"):
            st.warning(readiness["reason"])
        if labels_present:
            st.warning(
                "`data/pregnancy_labels.csv` exists, but this dashboard does not "
                "treat its labels as validated."
            )
        st.caption(
            "Prediction uncertainty and pregnancy risk scores are withheld unless "
            "supported by a validated human pregnancy-outcome model."
        )

        model_metrics = data.get("metrics", {}).get("model", {})
        metric_rows = [
            {"Outcome": name.replace("_", " ").title(), **values}
            for name, values in model_metrics.items()
            if isinstance(values, dict)
        ]
        if metric_rows:
            st.markdown("**Stored model evaluation metrics**")
            st.dataframe(
                pd.DataFrame(metric_rows),
                hide_index=True,
                width="stretch",
            )


def _render_filtered_summary(result, selected_name: str, target_label: str):
    timing_label = result["trimester"]
    st.subheader(f"{selected_name} · {target_label} · {timing_label}")

    if not result["evidence_source_available"]:
        st.error(
            "The standardized evidence master dataset is unavailable. Run "
            "`python -m src.acquire_datasets` before evaluating this query."
        )
        st.info(result["risk_scoring_status"])
        return

    # Determine if ML prediction is active or AI fallback
    if result.get("ml_prediction_active", True):
        # Existing ML UI rendering
        coverage_column, variance_column = st.columns(2)
        with coverage_column:
            coverage = result["evidence_coverage_score"]
            st.progress(
                coverage,
                text=f"Evidence Coverage Score (Q_cov): {coverage:.0%}",
            )
            st.caption(
                "Q_cov is the fraction of tracked pregnancy-evidence source families "
                "represented in this filtered result; it is not evidence quality or safety."
            )
        with variance_column:
            variance = result["model_prediction_variance"]
            if variance is None:
                st.metric("Model Prediction Variance (sigma)", "Unavailable")
                st.caption(
                    "No validated human pregnancy-outcome model is available. "
                    "Variance is unknown, not zero."
                )
            else:
                st.progress(
                    min(max(float(variance), 0.0), 1.0),
                    text=f"Model Prediction Variance (sigma): {float(variance):.3f}",
                )
    else:
        # AI fallback UI rendering
        synthesis = result.get("ai_evidence_synthesis", {})
        st.info("AI Fallback & Evidence Synthesis Active")
        st.markdown("**Missing evidence sources:**")
        missing = synthesis.get("missing_source_families", [])
        if missing:
            st.write(", ".join(missing))
        else:
            st.write("None")
        st.markdown("**Details:**")
        for key, val in synthesis.get("details", {}).items():
            st.write(f"- **{key}**: {val}")

    if result["abstention_alert"]:
        st.warning(result["abstention_alert"])
    else:
        st.info(result["risk_scoring_status"])

    records = result["records"]
    if records.empty:
        st.info(
            "No endpoint-matched evidence is available for this selection. "
            "Missing evidence is unknown, not evidence of safety."
        )
        return

    st.markdown("**Filtered evidence provenance**")
    for provenance in result["provenance_badges"]:
        source = provenance.get("source_dataset") or "Source not specified"
        details = [
            provenance.get("evidence_granularity"),
            provenance.get("evidence_status"),
        ]
        details = [str(value) for value in details if value]
        st.caption(f"● {source}" + (f" — {'; '.join(details)}" if details else ""))

    display_columns = [
        column
        for column in (
            "outcome_category",
            "outcome_term_original",
            "outcome",
            "reported_term",
            "trimester",
            "gestational_weeks",
            "source_dataset",
            "evidence_granularity",
            "evidence_status",
            "study_endpoint",
            "study_type",
            "species",
            "source_url",
            "source_doi",
            "source_record_id",
        )
        if column in records.columns
    ]
    st.dataframe(
        records[display_columns],
        hide_index=True,
        width="stretch",
    )

    timing_label = result["trimester"]
    st.subheader(f"{selected_name} · {target_label} · {timing_label}")

    if not result["evidence_source_available"]:
        st.error(
            "The standardized evidence master dataset is unavailable. Run "
            "`python -m src.acquire_datasets` before evaluating this query."
        )
        st.info(result["risk_scoring_status"])
        return

    coverage_column, variance_column = st.columns(2)
    with coverage_column:
        coverage = result["evidence_coverage_score"]
        st.progress(
            coverage,
            text=f"Evidence Coverage Score (Q_cov): {coverage:.0%}",
        )
        st.caption(
            "Q_cov is the fraction of tracked pregnancy-evidence source families "
            "represented in this filtered result; it is not evidence quality or safety."
        )
    with variance_column:
        variance = result["model_prediction_variance"]
        if variance is None:
            st.metric("Model Prediction Variance (sigma)", "Unavailable")
            st.caption(
                "No validated human pregnancy-outcome model is available. "
                "Variance is unknown, not zero."
            )
        else:
            st.progress(
                min(max(float(variance), 0.0), 1.0),
                text=f"Model Prediction Variance (sigma): {float(variance):.3f}",
            )

    if result["abstention_alert"]:
        st.warning(result["abstention_alert"])
    else:
        st.info(result["risk_scoring_status"])

    records = result["records"]
    if records.empty:
        st.info(
            "No endpoint-matched evidence is available for this selection. "
            "Missing evidence is unknown, not evidence of safety."
        )
        return

    st.markdown("**Filtered evidence provenance**")
    for provenance in result["provenance_badges"]:
        source = provenance.get("source_dataset") or "Source not specified"
        details = [
            provenance.get("evidence_granularity"),
            provenance.get("evidence_status"),
        ]
        details = [str(value) for value in details if value]
        st.caption(f"● {source}" + (f" — {'; '.join(details)}" if details else ""))

    display_columns = [
        column
        for column in (
            "outcome_category",
            "outcome_term_original",
            "outcome",
            "reported_term",
            "trimester",
            "gestational_weeks",
            "source_dataset",
            "evidence_granularity",
            "evidence_status",
            "study_endpoint",
            "study_type",
            "species",
            "source_url",
            "source_doi",
            "source_record_id",
        )
        if column in records.columns
    ]
    st.dataframe(
        records[display_columns],
        hide_index=True,
        width="stretch",
    )


def main():
    st.set_page_config(page_title="MEPRank Research Dashboard", layout="wide")
    st.title("Pregnancy-Related Drug Safety Research Dashboard")
    st.warning(
        "Research-support prototype only. This dashboard does not provide clinical "
        "diagnoses, prescribing recommendations, or validated pregnancy risk scores."
    )

    data = load_data()
    mapping = data["mapping"]
    if mapping.empty or "original_name" not in mapping.columns:
        st.error(
            "Processed drug data is missing. From the project root, run "
            "`python src/process_real_datasets.py` first."
        )
        return

    medication_names = sorted(
        {
            name.strip()
            for name in mapping["original_name"].dropna().astype(str)
            if name.strip()
        },
        key=str.casefold,
    )
    if not medication_names:
        st.error("No medication names are available in the local drug mapping.")
        return

    with st.container(border=True):
        st.markdown("### Query Filter Control Panel")
        medication_column, timing_column, target_column = st.columns([2, 1, 1])
        selected_name = medication_column.selectbox(
            "Select Medication Name",
            medication_names,
        )
        timing_options = [
            "First Trimester (T1)",
            "Second Trimester (T2)",
            "Third Trimester (T3)",
            "Entire Pregnancy",
            "All Trimesters",
        ]
        selected_timing = timing_column.selectbox(
            "Target Timing",
            timing_options,
        )
        target_label = target_column.radio(
            "View Target",
            ("Fetal Risk", "Maternal Risk"),
        )

    drug_rows = mapping.loc[mapping["original_name"].astype(str).eq(selected_name)]
    drug_id = str(drug_rows.iloc[0].get("drug_id", ""))
    timing_filter = (
        "All" if selected_timing == "All Trimesters" else selected_timing
    )
    target_filter = (
        "Fetal Outcomes Only"
        if target_label == "Fetal Risk"
        else "Maternal Outcomes Only"
    )
    result = _evaluate_query(
        selected_name,
        drug_id,
        timing_filter,
        target_filter,
    )
    _render_filtered_summary(result, selected_name, target_label)

    _render_readiness(data)


if __name__ == "__main__":
    main()
