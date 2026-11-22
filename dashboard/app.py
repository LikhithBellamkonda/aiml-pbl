import base64
from pathlib import Path
import sys

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard.data_access import load_dashboard_data  # noqa: E402
from src.mepr_reasoning import evaluate_drug_safety  # noqa: E402
from src.services.animal_adr_service import evidence_distinction  # noqa: E402
from src.services.drug_service import build_drug_profile  # noqa: E402


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
        # Evidence-synthesis UI: no model produced a value, so the panel
        # explains what the evidence does and does not cover.
        synthesis = result.get("ai_evidence_synthesis", {})
        st.info("Evidence synthesis active — no model prediction is produced")
        st.markdown("**Evidence source families missing for this filter:**")
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

    # Prediction status panel. No validated human pregnancy-outcome model
    # exists, so this panel reports evidence availability and abstention. It
    # never substitutes a heuristic number for a prediction.
    evidence_quality = result.get("evidence_quality")
    availability = result.get("evidence_availability", {}) or {}
    with st.container(border=True):
        st.markdown("### Pregnancy risk assessment")
        risk_col, conf_col, eq_col, model_col = st.columns(4)
        with risk_col:
            st.metric("Predicted foetal risk", "Unavailable")
            st.metric("Predicted maternal risk", "Unavailable")
            st.caption("No validated human pregnancy-outcome model is available.")
        with conf_col:
            st.metric("Prediction confidence", "Unavailable")
            st.caption(
                "Confidence is undefined when no model output exists. "
                "The absence of a prediction is not a low-risk result."
            )
        with eq_col:
            st.metric(
                "Evidence quality",
                "Unavailable"
                if evidence_quality is None
                else f"{float(evidence_quality):.0%}",
            )
            if result.get("evidence_gap", True):
                st.error("Evidence gap: understudied or single-source drug.")
            else:
                st.caption(
                    "Evidence coverage (availability mask × completeness), "
                    "not risk and not prediction confidence."
                )
        with model_col:
            st.metric("Model used", "None (abstained)")
            st.caption("No trained human pregnancy-outcome checkpoint exists.")

        available_sources = availability.get("available_sources") or []
        missing_sources = availability.get("missing_sources") or []
        st.markdown(
            "**Evidence source families with drug-level records:** "
            + (", ".join(available_sources) if available_sources else "None")
        )
        st.markdown(
            "**Evidence source families without drug-level records:** "
            + (", ".join(missing_sources) if missing_sources else "None")
        )

        observed = (result.get("evidence_summary") or {}).get(
            "observed_outcome_reports"
        ) or {}
        with st.expander("Observed pregnancy-outcome records (not risk)", expanded=False):
            if observed:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "outcome": str(outcome).replace("_", " "),
                                "source records": (
                                    "no record" if value is None else value
                                ),
                            }
                            for outcome, value in observed.items()
                        ]
                    ),
                    hide_index=True,
                    width="stretch",
                )
            else:
                st.caption("No pregnancy-outcome record summary is available.")
            st.caption(
                "Counts are source-reported spontaneous-report terms. 'No record' "
                "is missing evidence, not a zero and not a negative outcome."
            )

        with st.expander("Trimester-wise evidence availability", expanded=False):
            trimester_metrics = result.get("trimester_metrics", {}) or {}
            if not trimester_metrics:
                st.caption("No trimester-level evidence summary is available.")
            for trimester_key, entry in trimester_metrics.items():
                eq_value = entry.get("evidence_quality")
                q_cov_value = entry.get("q_cov")
                eq_text = (
                    "unavailable" if eq_value is None else f"{float(eq_value):.0%}"
                )
                q_cov_text = (
                    "unavailable"
                    if q_cov_value is None
                    else f"{float(q_cov_value):.0%}"
                )
                gap_text = "yes" if entry.get("evidence_gap", True) else "no"
                st.markdown(
                    f"**{str(trimester_key).replace('_', ' ').title()}** — "
                    f"evidence quality: {eq_text} · Q_cov: {q_cov_text} · "
                    f"evidence gap: {gap_text}"
                )
        st.caption(
            "Evidence availability and evidence quality describe the underlying "
            "records only. They are not a risk score, and a drug with no records "
            "must not be interpreted as safe."
        )

    # Side effects reported for this drug. Only measured counts and a
    # documented report-volume band are shown; no score is invented.
    side_effects = result.get("side_effects", []) or []
    if side_effects:
        with st.container(border=True):
            st.markdown("### Reported Side Effects")
            st.caption(
                "Observed ADR terms and the number of source records that mention "
                "each. 'Report-volume level' is a band on that count "
                "(High >= 10, Moderate 3-9, Low < 3 records), not clinical "
                "severity, incidence or validated confidence."
            )
            table = pd.DataFrame(
                [
                    {
                        "term": se["term"],
                        "records": se["report_count"],
                        "report-volume level": se["level"],
                        "MEDDRA id": se.get("meddra_id") or "not recorded",
                    }
                    for se in side_effects[:50]
                ]
            )
            st.dataframe(table, hide_index=True, width="stretch")
            if len(side_effects) > 50:
                st.caption(
                    f"Showing the 50 most-recorded terms of {len(side_effects)} "
                    "observed ADR terms."
                )

    _render_animal_adr(result)

    records = result["records"]
    if records.empty:
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


def _render_animal_adr(result):
    """Render the independent Animal ADR evidence branch.

    Findings are reported veterinary adverse events. A missing section is
    reported as unavailable evidence, never as a safe drug.
    """
    evidence = result.get("animal_adr_evidence") or {}
    with st.container(border=True):
        st.markdown("### Animal Adverse Drug Reaction Evidence")
        st.caption(
            "Veterinary spontaneous reports (FDA CVM ADAE, VeDDRA terms). "
            "Supplementary evidence only: not a human pregnancy risk conclusion, "
            "not incidence, and no causality is implied."
        )
        if not evidence.get("available"):
            st.info(
                evidence.get(
                    "message",
                    "Animal ADR evidence unavailable. This does not indicate that "
                    "the drug is safe.",
                )
            )
        else:
            first, second, third = st.columns(3)
            first.metric("Animal ADR reports", evidence.get("report_count", 0))
            first.metric("Reported reaction terms", evidence.get("reaction_count", 0))
            second.metric("Species represented", evidence.get("species_count", 0))
            second.metric("Serious reports", evidence.get("serious_report_count", 0))
            third.metric("Dominant species", evidence.get("top_species") or "Unknown")
            third.metric(
                "Dominant reaction category",
                evidence.get("top_reaction_category") or "Unclassified",
            )
            quality, mapping, dates = st.columns(3)
            quality.metric(
                "Record completeness", f"{evidence.get('record_completeness', 0.0):.0%}"
            )
            mapping.metric(
                "Drug mapping confidence", f"{evidence.get('mapping_confidence', 0.0):.0%}"
            )
            dates.metric(
                "Reporting period",
                f"{evidence.get('report_date_min') or '?'} to "
                f"{evidence.get('report_date_max') or '?'}",
            )
            st.caption(
                f"Source: {evidence.get('sources') or 'FDA CVM ADAE'} · "
                f"mapping: {evidence.get('mapping_status') or 'unknown'} · "
                f"deaths/euthanasia reports: "
                f"{evidence.get('death_or_euthanasia_report_count', 0)}"
            )
            if evidence.get("source_url"):
                st.markdown(f"[Original dataset documentation]({evidence['source_url']})")
            st.caption(
                "Limitations: spontaneous reporting is voluntary and has no "
                "exposure denominator, so counts are neither incidence nor "
                "severity. Reports describe product-level exposure and may "
                "involve combination products."
            )
        with st.expander("Evidence categories kept separate", expanded=False):
            st.dataframe(
                pd.DataFrame(evidence_distinction()),
                hide_index=True,
                width="stretch",
            )


def _render_drug_profile(profile):
    """Render resolved identity and each evidence source separately.

    Nothing in this panel is a risk estimate: absent evidence is reported as
    unavailable rather than scored.
    """
    identity = profile.get("identity") or {}
    counts = profile.get("evidence_counts") or {}
    availability = profile.get("availability") or {}

    with st.container(border=True):
        st.markdown("### Standardized drug identity")
        cid_col, key_col, status_col, id_col = st.columns(4)
        cid_col.metric("PubChem CID", identity.get("pubchem_cid") or "Unavailable")
        key_col.metric("InChIKey", identity.get("inchikey") or "Unavailable")
        status_col.metric(
            "Mapping status", str(identity.get("resolution_status") or "unknown")
        )
        id_col.metric(
            "Project drug id",
            str(identity.get("drug_id") or profile.get("drug_id") or "Unavailable"),
        )
        st.caption(
            "Resolved from: "
            + str(identity.get("resolution_source") or "project identifier mapping")
            + " · name key: "
            + str(profile.get("name_key") or "not applicable")
        )
        if identity.get("smiles"):
            st.caption(f"SMILES: {identity['smiles']}")
        molecular = profile.get("molecular") or {}
        structure_png = molecular.get("structure_png")
        if structure_png:
            st.image(
                base64.b64decode(structure_png.split(",", 1)[-1]),
                caption=(
                    "2D structure drawn locally from the recorded SMILES "
                    f"(scaffold: {molecular.get('scaffold') or 'not available'})."
                ),
                width=340,
            )
        elif molecular.get("reason"):
            st.caption(f"Structure depiction unavailable: {molecular['reason']}")
        for warning in profile.get("warnings") or []:
            st.warning(warning)

    with st.container(border=True):
        st.markdown("### Evidence availability by source")
        rows = [
            {
                "evidence source": "Human pregnancy-specific",
                "category": "human",
                "records": counts.get("pregnancy_records"),
            },
            {
                "evidence source": "General ADR terms",
                "category": "human",
                "records": counts.get("adr_terms"),
            },
            {
                "evidence source": "FAERS reports (aggregate)",
                "category": "human",
                "records": counts.get("faers_reports"),
            },
            {
                "evidence source": "Developmental / reproductive toxicity studies",
                "category": "animal (experimental)",
                "records": counts.get("nonclinical_toxicity_records"),
            },
        ]
        frame = pd.DataFrame(rows)
        frame["records"] = frame["records"].map(
            lambda value: "unavailable" if value is None else value
        )
        st.dataframe(frame, hide_index=True, width="stretch")
        st.caption(
            "Absent evidence is reported as unavailable. It is not zero risk and "
            "not a negative outcome. Animal ADR evidence is shown separately below."
        )
        missing = availability.get("missing_sources") or []
        if missing:
            st.info("Sources without records: " + ", ".join(missing))

    adr = profile.get("adr") or {}
    with st.expander("General adverse-drug-reaction evidence", expanded=False):
        if not adr.get("available"):
            st.info(adr.get("reason") or "No general ADR evidence recorded.")
        else:
            st.caption(
                f"{adr.get('term_count', 0)} distinct terms from "
                f"{', '.join(adr.get('source_datasets') or []) or 'unspecified sources'}. "
                "Term records are not incidence rates and imply no causality."
            )
            if adr.get("terms"):
                st.dataframe(
                    pd.DataFrame(adr["terms"]), hide_index=True, width="stretch"
                )

    molecular = profile.get("molecular") or {}
    with st.expander("Chemical and molecular properties", expanded=False):
        if not molecular.get("available"):
            st.info(molecular.get("reason") or "No molecular structure available.")
        else:
            descriptors = molecular.get("descriptors") or {}
            st.dataframe(
                pd.DataFrame(
                    [
                        {"property": name, "value": value}
                        for name, value in descriptors.items()
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
            fingerprint = molecular.get("fingerprint") or {}
            st.caption(
                f"Scaffold: {molecular.get('scaffold') or 'not available'} · "
                f"{fingerprint.get('type', 'fingerprint')}: "
                f"{fingerprint.get('on_bits', 0)} bits set. "
                "Descriptors describe the structure only; they are not biological predictions."
            )
            if molecular.get("svg"):
                st.markdown(molecular["svg"], unsafe_allow_html=True)

    toxicity = profile.get("reproductive_toxicity") or {}
    with st.expander(
        "Developmental / reproductive toxicity evidence (nonclinical animal studies)",
        expanded=False,
    ):
        st.caption(
            "Controlled animal assay endpoints. Kept separate from animal "
            "spontaneous ADR reports and from human pregnancy evidence."
        )
        if not toxicity.get("available"):
            st.info(
                toxicity.get("reason")
                or "No developmental/reproductive toxicity records."
            )
        else:
            st.caption(
                f"{toxicity.get('record_count', 0)} records · species: "
                f"{', '.join(toxicity.get('species') or []) or 'not recorded'}"
            )
            if toxicity.get("records"):
                st.dataframe(
                    pd.DataFrame(toxicity["records"]),
                    hide_index=True,
                    width="stretch",
                )

    pregnancy = profile.get("pregnancy") or {}
    with st.expander("Human pregnancy-specific evidence", expanded=False):
        st.caption(
            "Human evidence only. Report counts are spontaneous-report volume, "
            "not incidence, and imply no causality."
        )
        if not pregnancy.get("available"):
            st.info(pregnancy.get("reason") or "No human pregnancy evidence recorded.")
        else:
            st.caption(
                f"{pregnancy.get('record_count', 0)} records · sources: "
                f"{', '.join(pregnancy.get('sources') or []) or 'unspecified'} · "
                "trimesters: "
                f"{', '.join(pregnancy.get('trimesters') or []) or 'unknown'}"
            )
            category_counts = pregnancy.get("category_counts") or {}
            if category_counts:
                st.caption(
                    "Records by category: "
                    + ", ".join(
                        f"{name} {value}" for name, value in category_counts.items()
                    )
                )
            if pregnancy.get("records"):
                st.dataframe(
                    pd.DataFrame(pregnancy["records"]),
                    hide_index=True,
                    width="stretch",
                )

    faers = profile.get("faers") or {}
    with st.expander("FAERS reporting summary", expanded=False):
        if not faers.get("available"):
            st.info(faers.get("reason") or "No FAERS reporting summary.")
        else:
            st.caption(
                f"Aggregate reports: {faers.get('report_count')} · "
                f"outcome rows: {faers.get('outcome_count')} · "
                f"source: {faers.get('source_name') or 'not recorded'}. "
                "Reporting volume is not incidence."
            )


def main():
    st.set_page_config(page_title="MEPRank Research Dashboard", layout="wide")
    st.title("Pregnancy-Related Drug Safety Research Dashboard")
    st.caption(
        "Evidence-aware prioritization of potential pregnancy-stage-specific "
        "safety signals: trimester-segmented evidence, explicit coverage and "
        "missing-source reporting. No unvalidated risk score is produced."
    )
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
        search_query = st.text_input(
            "Search by drug name or identifier (drug_id, PubChem CID, InChIKey)",
            value="",
            placeholder="e.g. tretinoin, CID000444795, 444795",
        )
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
            ("Fetal Outcomes", "Maternal Outcomes"),
        )

    profile = None
    if search_query.strip():
        profile = build_drug_profile(search_query.strip())
        if profile.get("ambiguous"):
            st.warning(
                f"'{search_query.strip()}' matches multiple drug identifiers. "
                "Select one explicitly before reading any evidence."
            )
            st.dataframe(
                pd.DataFrame(profile.get("candidates") or []),
                hide_index=True,
                width="stretch",
            )
            _render_readiness(data)
            return
        if not profile.get("matched"):
            st.error(
                f"No local identifier mapping or evidence entry matches "
                f"'{search_query.strip()}'."
            )
            for warning in profile.get("warnings") or []:
                st.caption(warning)
            _render_readiness(data)
            return
        selected_name = str(
            (profile.get("identity") or {}).get("name") or search_query.strip()
        )
        drug_id = str(profile.get("drug_id") or "")
    else:
        drug_rows = mapping.loc[
            mapping["original_name"].astype(str).eq(selected_name)
        ]
        drug_id = str(drug_rows.iloc[0].get("drug_id", ""))

    timing_filter = (
        "All" if selected_timing == "All Trimesters" else selected_timing
    )
    target_filter = (
        "Fetal Outcomes Only"
        if target_label == "Fetal Outcomes"
        else "Maternal Outcomes Only"
    )
    if profile is not None:
        _render_drug_profile(profile)
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
