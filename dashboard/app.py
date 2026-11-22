"""MEPRank research dashboard.

The dashboard is a presentation layer only: all resolution, molecular and
evidence logic lives in ``src/services``. Sections that have no validated
model show an explicit abstention instead of a fabricated score.
"""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard.data_access import load_dashboard_data  # noqa: E402
from src.mepr_reasoning import evaluate_drug_safety  # noqa: E402
from src.services.drug_service import build_drug_profile, clear_cache  # noqa: E402
from src.services.inference import (  # noqa: E402
    PREGNANCY_TARGETS,
    TOX_ENDPOINTS,
    TOX_ENDPOINT_DESCRIPTIONS,
    predict,
)
from src.services.status import system_status  # noqa: E402

try:  # optional: renders the RDKit SVG structure
    import streamlit.components.v1 as components
except Exception:  # pragma: no cover - Streamlit always provides it
    components = None


@st.cache_data
def load_data():
    return load_dashboard_data(ROOT)


@st.cache_data(show_spinner="Resolving drug and assembling evidence...")
def _profile(query: str):
    return build_drug_profile(query, ROOT / "data")


@st.cache_data(show_spinner="Filtering evidence for the selected query...")
def _evaluate_query(drug_name: str, drug_id: str, trimester: str, target: str):
    return evaluate_drug_safety(
        drug_name,
        trimester=trimester,
        outcome_target=target,
        drug_id=drug_id,
    )


@st.cache_data
def _status():
    return system_status(ROOT / "data")


def _available_names(data) -> list[str]:
    mapping = data["mapping"]
    if mapping.empty or "original_name" not in mapping.columns:
        return []
    return sorted(
        {
            str(name).strip()
            for name in mapping["original_name"].dropna()
            if str(name).strip()
        },
        key=str.casefold,
    )


def _metric_row(profile):
    counts = profile.get("evidence_counts") or {}
    columns = st.columns(4)
    columns[0].metric("General ADR terms", counts.get("adr_terms", 0) or 0)
    columns[1].metric("Human pregnancy records", counts.get("pregnancy_records", 0) or 0)
    columns[2].metric(
        "Nonclinical toxicity studies",
        counts.get("nonclinical_toxicity_records", 0) or 0,
    )
    faers = counts.get("faers_reports")
    columns[3].metric(
        "FAERS reports", "Unavailable" if faers is None else f"{int(faers):,}"
    )


def _render_identity(profile):
    resolution = profile
    identity = profile.get("identity") or {}
    st.subheader(str(resolution.get("drug_name") or resolution.get("query")))
    if resolution.get("ambiguous"):
        st.error(
            "The query matches several identifiers. Pick one exact drug before "
            "interpreting evidence."
        )
        st.dataframe(pd.DataFrame(resolution.get("candidates") or []), hide_index=True)
        return
    table = pd.DataFrame(
        [
            ("Drug name", identity.get("name")),
            ("Project drug_id", identity.get("drug_id")),
            ("PubChem CID", identity.get("pubchem_cid")),
            ("InChIKey", identity.get("inchikey")),
            ("Molecular formula", identity.get("molecular_formula")),
            ("Molecular weight (reported)", identity.get("molecular_weight")),
            ("SMILES", identity.get("smiles")),
            ("Resolution status", identity.get("resolution_status")),
            ("Resolution source", identity.get("resolution_source")),
        ],
        columns=["Field", "Value"],
    )
    table["Value"] = table["Value"].map(
        lambda value: "Unavailable" if value is None or value == "" else value
    )
    st.dataframe(table, hide_index=True, width="stretch")
    for warning in resolution.get("warnings") or []:
        st.warning(warning)


def _render_overview(profile):
    _metric_row(profile)
    availability = profile.get("availability") or {}
    st.markdown("#### Source availability")
    sources = availability.get("sources") or {}
    if sources:
        rows = [
            {
                "Source": name.replace("_", " "),
                "Status": details.get("status") or "Unavailable",
                "Metadata completeness": details.get("metadata_completeness"),
            }
            for name, details in sorted(sources.items())
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.info("No per-source availability row exists for this drug.")
    missing = availability.get("missing_sources") or []
    if missing:
        st.warning(
            "Missing or unmatched sources (not zero, not negative evidence): "
            + ", ".join(name.replace("_", " ") for name in missing)
        )

    st.markdown("#### Molecular identity")
    molecular = profile.get("molecular") or {}
    if molecular.get("available"):
        st.dataframe(
            pd.DataFrame(
                [
                    {"Property": key.replace("_", " "), "Value": value}
                    for key, value in molecular["descriptors"].items()
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    else:
        st.info(molecular.get("reason") or "No molecular structure available.")


def _render_adr(profile):
    adr = profile.get("adr") or {}
    st.caption(
        "General adverse-event terms from the acquired pharmacovigilance sources. "
        "These are not pregnancy-specific outcomes and are not pregnancy risk."
    )
    if not adr.get("available"):
        st.info(adr.get("reason") or "No general ADR evidence recorded.")
        return
    st.write(
        f"{adr['term_count']} distinct terms from {adr['record_count']} records "
        f"({', '.join(adr.get('source_datasets') or []) or 'source not stated'})."
    )
    st.dataframe(pd.DataFrame(adr.get("terms") or []), hide_index=True, width="stretch")


def _render_molecular(profile):
    molecular = profile.get("molecular") or {}
    if not molecular.get("available"):
        st.info(molecular.get("reason") or "No valid structure for this drug.")
        return
    left, right = st.columns([1, 1])
    with left:
        st.dataframe(
            pd.DataFrame(
                [
                    {"Descriptor": key.replace("_", " "), "Value": value}
                    for key, value in molecular["descriptors"].items()
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(f"Murcko scaffold: {molecular.get('scaffold') or 'acyclic'}")
    with right:
        fingerprint = molecular.get("fingerprint") or {}
        st.metric("Fingerprint on-bits", fingerprint.get("on_bits", 0))
        st.caption(
            f"{fingerprint.get('type')}; density {fingerprint.get('density', 0):.1%}. "
            "These bits are the exact features used by the toxicity models."
        )
        st.code(molecular.get("canonical_smiles", ""), language="text")
    if molecular.get("svg") and components is not None:
        st.markdown("#### Structure")
        components.html(molecular["svg"], height=280)
    elif molecular.get("svg"):
        st.markdown("#### Structure")
        st.markdown(molecular["svg"], unsafe_allow_html=True)


def _render_reprotox(profile):
    toxicity = profile.get("reproductive_toxicity") or {}
    st.error(
        "Animal and nonclinical study evidence is kept separate from human "
        "pregnancy outcomes. It is not converted into a human label or risk score."
    )
    if not toxicity.get("available"):
        st.info(toxicity.get("reason") or "No nonclinical toxicity records.")
        return
    st.write(
        f"{toxicity['record_count']} study records across species: "
        + ", ".join(toxicity.get("species") or [])
    )
    counts = toxicity.get("counts_by_species") or []
    if counts:
        st.dataframe(pd.DataFrame(counts), hide_index=True, width="stretch")
    st.dataframe(
        pd.DataFrame(toxicity.get("records") or []),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Study citations, species, strain, route and dose are shown as recorded "
        "in " + ", ".join(toxicity.get("sources") or ["the toxicity source"]) + "."
    )


def _render_pregnancy(profile):
    pregnancy = profile.get("pregnancy") or {}
    st.caption(
        "Human pregnancy evidence, separated into maternal, fetal and neonatal "
        "outcomes. Reported counts are report volume, not incidence or risk."
    )
    if not pregnancy.get("available"):
        st.info(pregnancy.get("reason") or "No human pregnancy evidence recorded.")
        return
    counts = pregnancy.get("category_counts") or {}
    columns = st.columns(3)
    for column, category in zip(columns, ("maternal", "fetal", "neonatal")):
        column.metric(category.title(), counts.get(category, 0))
    if pregnancy.get("unknown_trimester"):
        st.warning(
            "Trimester is unknown for every record for this drug. Timing is never "
            "inferred from untimed evidence."
        )
    totals = pregnancy.get("outcome_totals") or []
    if totals:
        st.markdown("#### Outcome totals")
        st.dataframe(pd.DataFrame(totals), hide_index=True, width="stretch")
    st.markdown("#### Recorded outcomes and provenance")
    st.dataframe(
        pd.DataFrame(pregnancy.get("records") or []),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Recorded outcome terms: "
        + ", ".join(pregnancy.get("evidence_types") or ["not stated"])
    )


def _render_filtered_summary(result, selected_name: str, target_label: str):
    """Render one MEPRank evidence query (kept for the existing test suite)."""
    st.subheader(f"{selected_name} · {target_label} · {result.get('trimester', 'All')}")
    if not result.get("evidence_source_available"):
        st.error(
            "The standardized evidence master dataset is unavailable. Run "
            "`python -m src.acquire_datasets` before evaluating this query."
        )
        st.info(result.get("risk_scoring_status", ""))
        return
    coverage = result.get("evidence_coverage_score")
    left, right = st.columns(2)
    with left:
        if coverage is None:
            st.metric("Evidence coverage (Q_cov)", "Unavailable")
        else:
            st.progress(coverage, text=f"Evidence coverage (Q_cov): {coverage:.0%}")
        st.caption(
            "Q_cov is the fraction of tracked human evidence source families in "
            "this result; it is not safety or evidence quality."
        )
    with right:
        st.metric("Risk score", "Unavailable")
        st.metric("Predictive variance (sigma)", "Unavailable")
        st.caption("No validated human pregnancy-outcome model exists.")
    if result.get("abstention_alert"):
        st.warning(result["abstention_alert"])
    else:
        st.info(result.get("risk_scoring_status", ""))
    records = result.get("records")
    if isinstance(records, pd.DataFrame) and not records.empty:
        st.dataframe(records, hide_index=True, width="stretch")
    for provenance in result.get("provenance_badges") or []:
        source = provenance.get("source_dataset") or "Source not specified"
        st.caption(f"• {source}")


def _render_meprank(profile, data):
    identity = profile.get("identity") or {}
    drug_id = identity.get("drug_id")
    st.caption(
        "MEPRank reports a model result only when a trained model supports the "
        "requested task. Otherwise it abstains and states why."
    )
    evaluation = _evaluate_query(
        str(profile.get("drug_name") or ""),
        str(drug_id or ""),
        "All",
        "Both",
    )
    _render_filtered_summary(evaluation, str(profile.get("drug_name") or ""), "Both")
    coverage = evaluation.get("evidence_coverage_score")
    columns = st.columns(3)
    columns[0].metric(
        "Evidence coverage (Q_cov)",
        "Unavailable" if coverage is None else f"{coverage:.0%}",
    )
    columns[1].metric("Evidence records", evaluation.get("evidence_record_count", 0))
    columns[2].metric("Trained pregnancy model", "No")
    if coverage is not None:
        st.progress(coverage, text="Fraction of tracked human evidence sources present")
    else:
        st.info("No standardized evidence master is available for coverage.")

    st.markdown("#### Human pregnancy outcomes")
    for target in PREGNANCY_TARGETS:
        result = predict(drug_id, target)
        with st.container(border=True):
            st.markdown(f"**{target.replace('_', ' ')}** — `{result['status']}`")
            st.write(
                "Prediction: "
                + (
                    f"{result['prediction']:.3f}"
                    if result["prediction"] is not None
                    else "Unavailable (abstained)"
                )
            )
            for reason in result["reasons"]:
                st.caption(reason)
            st.caption(
                "Evidence available: "
                + (", ".join(result["modalities_available"]) or "none")
            )

    st.markdown("#### Nonclinical reproductive-toxicity endpoints")
    st.caption(
        "Separate animal-endpoint models. A probability here is not a human "
        "pregnancy prediction."
    )
    rows = []
    for endpoint in TOX_ENDPOINTS:
        result = predict(drug_id, endpoint)
        rows.append(
            {
                "Endpoint": endpoint,
                "Description": TOX_ENDPOINT_DESCRIPTIONS[endpoint],
                "Status": result["status"],
                "Probability": result["prediction"],
                "Uncertainty": result["uncertainty"],
                "Model": (result.get("model") or {}).get("name") or "—",
                "Reason": " ".join(result["reasons"]) or "—",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.info(
        "No nonclinical checkpoint is present unless the source workbook is "
        "available and `python -m src.train_reproductive_tox` has been run."
    )


def _render_comparison(profile):
    identity = profile.get("identity") or {}
    adr = profile.get("adr") or {}
    pregnancy = profile.get("pregnancy") or {}
    toxicity = profile.get("reproductive_toxicity") or {}
    faers = profile.get("faers") or {}
    rows = [
        {
            "Domain": "General ADR",
            "Task": "Human adverse-event terms",
            "Records": adr.get("record_count", 0),
            "Sources": ", ".join(adr.get("source_datasets") or []) or "—",
            "Model available": "No",
        },
        {
            "Domain": "Pregnancy outcomes",
            "Task": "Human maternal/fetal/neonatal outcomes",
            "Records": pregnancy.get("record_count", 0),
            "Sources": ", ".join(pregnancy.get("sources") or []) or "—",
            "Model available": "No",
        },
        {
            "Domain": "Nonclinical toxicity",
            "Task": "Animal reproductive/developmental studies",
            "Records": toxicity.get("record_count", 0),
            "Sources": ", ".join(toxicity.get("sources") or []) or "—",
            "Model available": "Only if a checkpoint exists",
        },
        {
            "Domain": "FAERS reporting",
            "Task": "Aggregate spontaneous reports",
            "Records": faers.get("report_count") if faers.get("available") else None,
            "Sources": faers.get("source_name") or "—",
            "Model available": "No",
        },
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.markdown(
        f"**Drug:** {identity.get('name')} · `{identity.get('drug_id')}` · "
        f"CID {identity.get('pubchem_cid') or 'unavailable'}"
    )
    for label, payload in (
        ("General ADR", adr),
        ("Pregnancy", pregnancy),
        ("Nonclinical toxicity", toxicity),
        ("FAERS", faers),
    ):
        st.caption(
            f"{label}: "
            + ("available" if payload.get("available") else "not available for this drug")
        )
    st.warning(
        "Absent evidence is not negative evidence. A drug with no pregnancy record "
        "has unknown pregnancy evidence, not a safe outcome."
    )


def _render_status():
    status = _status()
    st.markdown("#### Datasets")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Dataset": item["name"],
                    "File": item["file"],
                    "Present": "Yes" if item["present"] else "No",
                    "Rows": item["rows"],
                }
                for item in status["datasets"]
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    st.markdown("#### Models and training readiness")
    readiness = status["training_readiness"]
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Task": "Human pregnancy outcomes",
                    "Status": readiness["human_pregnancy"]["status"],
                    "Reason": readiness["human_pregnancy"]["reason"],
                },
                {
                    "Task": "Nonclinical reproductive toxicity",
                    "Status": readiness["nonclinical_toxicity"]["status"],
                    "Reason": readiness["nonclinical_toxicity"]["reason"],
                },
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    if status.get("metrics"):
        st.markdown("#### Nonclinical evaluation metrics (out-of-fold)")
        st.json(status["metrics"])
    run = status.get("last_pipeline_run")
    if run:
        st.markdown("#### Last pipeline execution")
        st.json(
            {
                "started_at": run.get("started_at"),
                "finished_at": run.get("finished_at"),
                "status": run.get("status"),
                "stages": [
                    {"stage": item["stage"], "status": item["status"]}
                    for item in run.get("stages", [])
                ],
            }
        )
    else:
        st.info(
            "No pipeline run record found. Run `python -m src.run_pipeline` to "
            "regenerate the data artefacts and record the run."
        )
    if status["missing_datasets"]:
        st.warning("Missing datasets: " + ", ".join(status["missing_datasets"]))
    st.caption("Refreshing this panel does not modify any data.")


def main():
    st.set_page_config(page_title="MEPRank Research Dashboard", layout="wide")
    st.title("Pregnancy-Related Drug Safety Research Dashboard")
    st.warning(
        "Research-support prototype only. This dashboard does not provide clinical "
        "diagnoses, prescribing recommendations, or validated pregnancy risk scores."
    )

    data = load_data()
    names = _available_names(data)

    st.sidebar.header("Drug search")
    if st.sidebar.button("Clear cached data"):
        clear_cache()
        load_data.clear()
        _profile.clear()
        _status.clear()
        st.rerun()
    query = st.sidebar.text_input(
        "Drug name or identifier",
        value="",
        key="drug_query",
        placeholder="e.g. isotretinoin, 444795 or CID000444795",
    )
    selected = st.sidebar.selectbox(
        "…or choose from mapped drugs",
        ["—"] + names,
        key="drug_select",
    )
    if selected and selected != "—":
        query = selected
    st.sidebar.caption(
        f"{len(names)} mapped drug names available."
        if names
        else "No drug mapping found. Run `python src/process_real_datasets.py`."
    )

    if not query.strip():
        st.info(
            "Enter a drug name, PubChem CID, InChIKey or project drug_id in the "
            "sidebar to inspect its evidence."
        )
        st.markdown("## System status")
        _render_status()
        return

    profile = _profile(query)

    if not profile.get("matched") or profile.get("ambiguous"):
        st.error(f"Could not resolve a single drug for '{query}'.")
        _render_identity(profile)
        for warning in profile.get("warnings") or []:
            st.warning(warning)
        st.markdown("## System status")
        _render_status()
        return

    _render_identity(profile)

    tabs = st.tabs(
        [
            "Overview",
            "General ADR",
            "Molecular",
            "Reproductive toxicity",
            "Pregnancy evidence",
            "MEPRank results",
            "Evidence comparison",
            "System status",
        ]
    )
    with tabs[0]:
        _render_overview(profile)
    with tabs[1]:
        _render_adr(profile)
    with tabs[2]:
        _render_molecular(profile)
    with tabs[3]:
        _render_reprotox(profile)
    with tabs[4]:
        _render_pregnancy(profile)
    with tabs[5]:
        _render_meprank(profile, data)
    with tabs[6]:
        _render_comparison(profile)
    with tabs[7]:
        _render_status()


if __name__ == "__main__":
    main()
