import base64
import re
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


@st.cache_data(show_spinner="Resolving drug identity and chemical structure...")
def _build_profile(query: str):
    return build_drug_profile(query)


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
    evaluated_model = has_evaluated_model(data)
    if not readiness and feasibility.get("target_decisions"):
        decisions = feasibility["target_decisions"]
        blocked = not any(item.get("decision") == "trainable" for item in decisions)
        readiness = {
            "status": "blocked" if blocked else "not_verified",
            "reason": decisions[0].get("reason", "") if blocked else "",
        }
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

    status_column, labels_column, model_column = st.columns(3)
    status_column.metric("Training status", training_status)
    labels_column.metric(
        "Validated human labels", "present" if labels_present else "absent"
    )
    model_column.metric(
        "Evaluated model", "available" if evaluated_model else "none"
    )
    reason = readiness.get("reason")
    if reason:
        st.caption(reason)

    decisions = feasibility.get("target_decisions") or []
    if decisions:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "outcome": str(item.get("outcome") or "").replace("_", " "),
                        "decision": item.get("decision") or "unknown",
                        "reason": item.get("reason") or "",
                    }
                    for item in decisions
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    st.caption(
        "No validated human pregnancy-outcome labels exist locally, so no model "
        "is trained and no AUROC/AUPRC/calibration value is reported. Metrics "
        "that have not been measured are not shown as zero."
    )


def _formula_html(formula: str) -> str:
    """Render an RDKit Hill-order formula with proper element subscripts."""
    parts: list[str] = []
    for element, count in re.findall(r"([A-Z][a-z]?)(\d*)", str(formula)):
        if not element:
            continue
        parts.append(
            element + (f"<sub>{count}</sub>" if count and count != "1" else "")
        )
    return (
        "<div style='font-size:2.1rem;font-weight:600;letter-spacing:0.02em;"
        "line-height:1.15;margin:0.1rem 0 0.6rem 0'>"
        + "".join(parts)
        + "</div>"
    )


def _format_count(value: float | int) -> str:
    number = float(value)
    if number.is_integer():
        return f"{int(number):,}"
    return f"{number:,.3f}".rstrip("0").rstrip(".")


def _fmt_score(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.1f}"


def _render_filtered_summary(
    result, selected_name: str, target_label: str, profile: dict | None = None
):
    profile = profile or {}
    timing_label = result["trimester"]
    st.subheader(f"{selected_name} · {target_label} · {timing_label}")

    if not result["evidence_source_available"]:
        st.error(
            "The standardized evidence master dataset is unavailable. Run "
            "`python -m src.acquire_datasets` before evaluating this query."
        )
        st.info(result["risk_scoring_status"])
        return

    pregnancy = profile.get("pregnancy") or {}
    pregnancy_rows = pregnancy.get("records") or []
    records = result.get("records", pd.DataFrame())
    relevant_categories = (
        {"fetal", "neonatal"}
        if target_label == "Fetal Outcomes"
        else {"maternal"}
    )
    if timing_label == "All":
        selected_outcomes = [
            row
            for row in pregnancy.get("outcome_totals", [])
            if row.get("category") in relevant_categories
        ]
        selected_row_count = sum(
            int(row.get("records") or 0) for row in selected_outcomes
        )
        selected_reported_count = sum(
            float(row["reported_count"])
            for row in selected_outcomes
            if row.get("reported_count") is not None
        )
        has_reported_counts = any(
            row.get("reported_count") is not None for row in selected_outcomes
        )
    else:
        selected_row_count = int(len(records))
        selected_outcomes = []
        if not records.empty and "outcome" in records.columns:
            group_columns = [
                column
                for column in ("outcome_category", "outcome")
                if column in records.columns
            ]
            grouped = records.groupby(group_columns, dropna=False)
            for key, rows in grouped:
                values = key if isinstance(key, tuple) else (key,)
                item = dict(zip(group_columns, values))
                counts = (
                    pd.to_numeric(rows["reported_count"], errors="coerce")
                    if "reported_count" in rows.columns
                    else pd.Series(dtype=float)
                )
                item.update(
                    {
                        "records": int(len(rows)),
                        "reported_count": float(counts.sum())
                        if counts.notna().any()
                        else None,
                    }
                )
                selected_outcomes.append(item)
        selected_reported_count = sum(
            float(row["reported_count"])
            for row in selected_outcomes
            if row.get("reported_count") is not None
        )
        has_reported_counts = any(
            row.get("reported_count") is not None for row in selected_outcomes
        )
    trimester_counts = result.get("trimester_evidence_counts") or {}
    stage_values = [
        ("First trimester", "first_trimester"),
        ("Second trimester", "second_trimester"),
        ("Third trimester", "third_trimester"),
        ("Entire pregnancy", "entire_pregnancy"),
        ("Timing not recorded", "unknown"),
    ]
    stage_frame = pd.DataFrame(
        [
            {
                "Timing": label,
                "Evidence rows": int(
                    (trimester_counts.get(key) or {}).get("record_count") or 0
                ),
            }
            for label, key in stage_values
        ]
    )

    availability = result.get("evidence_availability", {}) or {}
    with st.container(border=True):
        st.markdown("### Observed pregnancy evidence")
        records_col, reported_col, model_col = st.columns(3)
        records_col.metric("Source outcome rows", selected_row_count)
        reported_col.metric(
            "Sum across outcome counts",
            _format_count(selected_reported_count)
            if has_reported_counts
            else "No count recorded",
        )
        model_col.metric("Pregnancy outcome model", "Not trained")
        st.caption(
            "Counts are source-reported aggregates; the same case may contribute "
            "to multiple outcomes. They are not unique reports, incidence, probability, "
            "severity, or an individual's risk. No validated pregnancy outcome model "
            "is trained from the current local labels."
        )

        stage_total = int(stage_frame["Evidence rows"].sum())
        unknown_count = int(
            (trimester_counts.get("unknown") or {}).get("record_count") or 0
        )
        if stage_total:
            st.markdown("#### Evidence rows by recorded timing")
            st.bar_chart(stage_frame, x="Timing", y="Evidence rows", height=250)
        if unknown_count and not int(stage_frame.iloc[:4]["Evidence rows"].sum()):
            st.info(
                f"{unknown_count} pregnancy evidence row(s) have no trimester "
                "annotation. No stage is inferred from untimed records."
            )
        elif not stage_total:
            st.info("No source outcome rows match this drug and outcome filter.")

        if selected_outcomes:
            st.markdown("#### Reported outcomes in the local source tables")
            outcome_frame = pd.DataFrame(
                [
                    {
                        "Outcome term": str(row.get("outcome") or "").replace("_", " "),
                        "Source rows": int(row.get("records") or 0),
                        "Reported count": row.get("reported_count"),
                    }
                    for row in selected_outcomes
                ]
            )
            st.dataframe(outcome_frame, hide_index=True, width="stretch")
            plot_rows = outcome_frame.dropna(subset=["Reported count"])
            if not plot_rows.empty:
                st.bar_chart(
                    plot_rows,
                    x="Outcome term",
                    y="Reported count",
                    height=280,
                )
            st.caption(
                "Reported counts are transcribed from the cited aggregate source. "
                "They have no population denominator, may overlap across outcomes, "
                "and are not risk percentages."
            )
        elif pregnancy_rows and timing_label == "All":
            st.info(
                f"{len(pregnancy_rows)} pregnancy source row(s) exist, but none "
                "are categorized under this outcome filter."
            )
        else:
            st.info("No human pregnancy outcome rows are indexed for this drug.")

        breakdown = result.get("evidence_source_breakdown") or []
        with st.expander("Evidence source coverage and measurement details", expanded=False):
            available_sources = availability.get("available_sources") or []
            missing_sources = availability.get("missing_sources") or []
            st.metric(
                "Tracked source families with records",
                f"{len(available_sources)} / "
                f"{len(available_sources) + len(missing_sources)}",
            )
            st.caption(
                "Represented: "
                + (", ".join(available_sources) if available_sources else "None")
            )
            st.caption(
                "No drug-level record in: "
                + (", ".join(missing_sources) if missing_sources else "None")
            )
            if breakdown:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "source family": item.get("label") or item.get("source"),
                                "available": "yes" if item.get("available") else "no",
                                "records": item.get("records"),
                                "observed count total": item.get("observed_total"),
                                "completeness": (
                                    "not measured"
                                    if item.get("completeness") is None
                                    else f"{float(item['completeness']):.1%}"
                                ),
                                "volume component": (
                                    "not measured"
                                    if item.get("volume_score") is None
                                    else f"{float(item['volume_score']):.1%}"
                                ),
                                "source quality": (
                                    "not measured"
                                    if item.get("quality") is None
                                    else f"{float(item['quality']):.1%}"
                                ),
                            }
                            for item in breakdown
                        ]
                    ),
                    hide_index=True,
                    width="stretch",
                )
            else:
                st.caption("No per-source measurement rows were generated.")
            if result.get("evidence_quality_method"):
                st.caption(result["evidence_quality_method"])
            st.caption(
                "These are evidence completeness/volume measurements, not clinical "
                "confidence or a predicted pregnancy outcome."
            )

        # --- Trimester-wise evidence-priority prioritization ---
        maternal = result.get("maternal_priority", {}) or {}
        fetal = result.get("fetal_priority", {}) or {}
        overall = result.get("overall_priority", {}) or {}
        evidence_status = result.get("evidence_status", "score_unavailable")
        status_detail = result.get("status_detail", "")

        with st.container(border=True):
            st.markdown("### Trimester-wise evidence-priority index")
            st.caption(
                "Exploratory evidence-priority index (0-100) computed from actual "
                "recorded evidence only. Not a clinical risk probability. "
                "A score is unavailable when no eligible human evidence exists "
                "for a trimester."
            )

            # Status badge
            status_colors = {
                "human_pregnancy_evidence_available": "#2e7d32",
                "human_and_nonclinical_evidence_available": "#1565c0",
                "limited_pregnancy_evidence": "#f9a825",
                "single_source_evidence": "#f9a825",
                "conflicting_evidence": "#ef6c00",
                "insufficient_evidence": "#ef6c00",
                "score_unavailable": "#9e9e9e",
            }
            status_color = status_colors.get(evidence_status, "#9e9e9e")
            st.markdown(
                f"<span style='color:{status_color};font-weight:bold'>"
                f"Evidence status: {evidence_status.replace('_', ' ').title()}</span>",
                unsafe_allow_html=True,
            )
            if status_detail:
                st.caption(status_detail)

            # Trimester-wise table
            trimester_order = [
                ("first_trimester", "T1 (0-13 wks)"),
                ("second_trimester", "T2 (14-27 wks)"),
                ("third_trimester", "T3 (28+ wks)"),
                ("entire_pregnancy", "Entire Pregnancy"),
            ]
            table_rows = []
            for tkey, tlabel in trimester_order:
                m = maternal.get(tkey, {})
                f = fetal.get(tkey, {})
                o = overall.get(tkey, {})
                table_rows.append(
                    {
                        "Trimester": tlabel,
                        "Maternal priority": _fmt_score(m.get("score")),
                        "Maternal records": m.get("eligible_record_count", 0),
                        "Fetal priority": _fmt_score(f.get("score")),
                        "Fetal records": f.get("eligible_record_count", 0),
                        "Overall priority": _fmt_score(o.get("score")),
                        "Overall records": o.get("eligible_record_count", 0),
                    }
                )
            st.dataframe(
                pd.DataFrame(table_rows),
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "Score is unavailable (shown as —) when no eligible human evidence "
                "exists for that trimester. Overall = 0.5 × Maternal + 0.5 × Fetal; "
                "if either component is unavailable, overall is unavailable."
            )

            # Detailed trimester breakdown
            with st.expander("Trimester detail", expanded=False):
                for tkey, tlabel in trimester_order:
                    m = maternal.get(tkey, {})
                    f = fetal.get(tkey, {})
                    o = overall.get(tkey, {})
                    st.markdown(f"#### {tlabel}")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Maternal",
                            _fmt_score(m.get("score")),
                            f"{m.get('eligible_record_count', 0)} records",
                        )
                    with col2:
                        st.metric(
                            "Fetal",
                            _fmt_score(f.get("score")),
                            f"{f.get('eligible_record_count', 0)} records",
                        )
                    with col3:
                        st.metric(
                            "Overall",
                            _fmt_score(o.get("score")),
                            f"{o.get('eligible_record_count', 0)} records",
                        )
                    if m.get("source_counts"):
                        st.caption(
                            "Maternal sources: "
                            + ", ".join(
                                f"{k} ({v})" for k, v in m["source_counts"].items()
                            )
                        )
                    if f.get("source_counts"):
                        st.caption(
                            "Fetal sources: "
                            + ", ".join(
                                f"{k} ({v})" for k, v in f["source_counts"].items()
                            )
                        )
                    if f.get("has_nonclinical_evidence"):
                        st.caption(
                            "Nonclinical (animal) evidence also available — "
                            "reported separately, not merged into human score."
                        )

        st.caption(
            "Trimester sections show only records whose source explicitly supplies "
            "timing. The dashboard does not invent different trimester risk scores."
        )

    side_effects = result.get("side_effects", []) or []
    if side_effects:
        with st.container(border=True):
            st.markdown("### General ADR term index")
            st.caption(
                "The source contains drug-to-term mappings, not patient-level "
                "reports. Entry counts below are source-table rows and may include "
                "multiple MedDRA concept types for the same term."
            )
            table = pd.DataFrame(
                [
                    {
                        "term": se["term"],
                        "source entries": se["source_entries"],
                        "MedDRA concept types": se.get("concept_types") or "not recorded",
                        "MedDRA id": se.get("meddra_id") or "not recorded",
                        "source": se.get("source_dataset") or "not recorded",
                    }
                    for se in side_effects[:50]
                ]
            )
            st.dataframe(table, hide_index=True, width="stretch")
            if len(side_effects) > 50:
                st.caption(
                    f"Showing 50 of {len(side_effects)} distinct indexed ADR terms."
                )



def _render_provenance(result):
    """Render the filtered record provenance and the records themselves."""
    records = result["records"]
    if records.empty:
        st.info("No records match the current filter.")
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
                    "No matching animal ADR records in the local snapshot.",
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
                "Record completeness",
                f"{evidence.get('record_completeness', 0.0):.2f} / 1",
            )
            mapping.metric("Drug mapping status", evidence.get("mapping_status") or "Not mapped")
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

    Nothing in this panel is a risk estimate: absent evidence is described as
    missing from the local sources rather than scored.
    """
    identity = profile.get("identity") or {}
    counts = profile.get("evidence_counts") or {}
    availability = profile.get("availability") or {}

    with st.container(border=True):
        st.markdown("### Standardized drug identity")
        cid_col, key_col, status_col, id_col = st.columns(4)
        cid_col.metric("PubChem CID", identity.get("pubchem_cid") or "No CID mapped")
        key_col.metric("InChIKey", identity.get("inchikey") or "No InChIKey mapped")
        status_col.metric(
            "Mapping status", str(identity.get("resolution_status") or "unknown")
        )
        id_col.metric(
            "Project drug id",
            str(identity.get("drug_id") or profile.get("drug_id") or "Not assigned"),
        )
        st.caption(
            "Resolved from: "
            + str(identity.get("resolution_source") or "project identifier mapping")
            +             " · indexed name: "
            + str(profile.get("name_key") or "not recorded")
        )
        for warning in profile.get("warnings") or []:
            st.warning(warning)

    molecular = profile.get("molecular") or {}
    formula = identity.get("molecular_formula") or molecular.get("molecular_formula")
    structure_png = molecular.get("structure_png")
    with st.container(border=True):
        st.markdown("### Chemical representation")
        formula_column, image_column = st.columns([1, 1])
        with formula_column:
            st.caption("Molecular formula (Hill order, computed from the recorded SMILES)")
            if formula:
                st.markdown(_formula_html(formula), unsafe_allow_html=True)
            else:
                st.info("Formula is not recorded for this structure.")
            weight = identity.get("molecular_weight")
            if weight not in (None, ""):
                try:
                    st.metric("Molecular weight (g/mol)", f"{float(weight):.2f}")
                except (TypeError, ValueError):
                    st.metric("Molecular weight (g/mol)", str(weight))
            smiles = identity.get("smiles") or molecular.get("canonical_smiles")
            if smiles:
                st.caption("SMILES (linear symbolic representation)")
                st.code(str(smiles), language="text")
            else:
                st.caption("SMILES is not mapped for this drug identity.")
        with image_column:
            if structure_png:
                st.image(
                    base64.b64decode(structure_png.split(",", 1)[-1]),
                    caption=(
                        "2D structure drawn locally from the recorded SMILES "
                        f"(scaffold: {molecular.get('scaffold') or 'not classified'})."
                    ),
                    width=340,
                )
            elif molecular.get("reason"):
                st.info(f"Structure not drawn: {molecular['reason']}")
            else:
                st.info("No structure is mapped to this drug identity.")

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
            lambda value: "No indexed record" if value is None else value
        )
        st.dataframe(frame, hide_index=True, width="stretch")
        st.caption(
            "A missing row means this local source has no linked record; it is not "
            "a zero-risk or negative-outcome result. Animal ADR evidence is separate."
        )
        missing = availability.get("missing_sources") or []
        if missing:
            st.info("No linked record in: " + ", ".join(missing))

    research_priority = profile.get("research_priority") or {}
    if research_priority.get("available"):
        with st.container(border=True):
            st.markdown("### Research-priority index — evidence gaps only")
            score_column, detail_column = st.columns([1, 3])
            score_column.metric(
                "Missing evidence domains (out of 3)",
                f"{research_priority['score']} / {research_priority['max_score']}",
            )
            detail_column.info(
                "One point is added for each missing domain. A higher score means "
                "more local evidence gaps—not higher fetal or maternal risk. "
                "This is not endpoint-specific, is not a prediction, and has no "
                "prediction-confidence value."
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Evidence domain": component["domain"],
                            "Status": "Available" if component["covered"] else "Missing",
                            "Index points": component["gap_points"],
                        }
                        for component in research_priority.get("components") or []
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "The index records source coverage only; available aggregate records "
                "are not treated as incidence or proof of safety. Maternal- and "
                "fetal-risk estimates remain unavailable."
            )

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
            st.info(molecular.get("reason") or "No molecular descriptors for this identity.")
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
                f"Scaffold: {molecular.get('scaffold') or 'not classified'} · "
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
                or "No linked developmental/reproductive toxicity rows."
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
            st.info(pregnancy.get("reason") or "No linked human pregnancy rows.")
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

    if not pregnancy.get("available"):
        related = profile.get("related_pregnancy_evidence") or {}
        with st.container(border=True):
            st.markdown("### Research leads — direct pregnancy data missing")
            st.warning(
                "Research needed: no direct human pregnancy-outcome rows are linked "
                "to this drug. The nearest-structure records below belong to those "
                "compounds only; weak similarity does not establish a useful "
                "analogue or a shared effect. No risk probability or trimester-specific "
                "result is inferred."
            )
            molecular = profile.get("molecular") or {}
            descriptors = molecular.get("descriptors") or {}
            property_specs = [
                ("Calculated logP", "clogp"),
                ("Topological polar surface area", "tpsa"),
                ("Hydrogen-bond donors", "h_bond_donors"),
                ("Hydrogen-bond acceptors", "h_bond_acceptors"),
            ]
            property_specs = [
                (label, descriptors[key])
                for label, key in property_specs
                if descriptors.get(key) is not None
            ]
            if property_specs:
                st.markdown("#### Computed chemistry context")
                property_columns = st.columns(len(property_specs))
                for column, (label, value) in zip(property_columns, property_specs):
                    column.metric(label, f"{value:g}")
                st.caption(
                    "Structure-derived properties can inform follow-up research, "
                    "but they are not validated predictors of placental transfer "
                    "or pregnancy outcome."
                )
            toxicity = profile.get("reproductive_toxicity") or {}
            if toxicity.get("available"):
                st.caption(
                    f"This drug also has {toxicity.get('record_count', 0)} "
                    "nonclinical toxicity study records; these are not human outcomes."
                )
            if related.get("available"):
                rows = []
                for candidate in related.get("candidates") or []:
                    outcome_labels = []
                    for outcome in candidate.get("outcomes") or []:
                        label = outcome["outcome"]
                        count = outcome.get("reported_count")
                        if count is not None:
                            label += f" (source-reported count {count:g})"
                        outcome_labels.append(label)
                    rows.append(
                        {
                            "Nearest compound in local evidence": candidate["drug_name"],
                            "Structural similarity (Tanimoto)": (
                                f"{candidate['similarity']:.3f}"
                            ),
                            "Pregnancy source rows": candidate["source_record_count"],
                            "Source-reported outcomes": (
                                "; ".join(outcome_labels) or "No named outcome term"
                            ),
                            "Timing": candidate["timing"],
                            "Sources": ", ".join(candidate.get("sources") or [])
                            or "Not specified",
                        }
                    )
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                st.caption(
                    f"{related.get('similarity_method')}. Similarity is a chemical "
                    "comparison, not a validated safety measure. Aggregate source "
                    "counts can overlap and do not show incidence or causality. "
                    "No similarity cutoff is validated here; low values may not "
                    "identify meaningful analogues."
                )
            else:
                st.info(
                    related.get("reason")
                    or "No structurally related compounds with linked pregnancy records."
                )
            st.caption(
                "The local pregnancy records do not provide gestational timing, so "
                "trimester-specific conclusions cannot be supported. FDA pregnancy "
                "labeling calls for drug-specific risk summaries, clinical "
                "considerations, and data."
            )
            st.markdown(
                "[FDA Pregnancy and Lactation Labeling Resources]"
                "(https://www.fda.gov/drugs/labeling-information-drug-products/"
                "pregnancy-and-lactation-labeling-resources)"
            )

    faers = profile.get("faers") or {}
    with st.expander("FAERS reporting summary", expanded=False):
        if not faers.get("available"):
            st.info(faers.get("reason") or "No FAERS summary row is linked.")
        else:
            st.caption(
                f"Aggregate reports: {faers.get('report_count')} · "
                f"outcome rows: {faers.get('outcome_count')} · "
                f"source: {faers.get('source_name') or 'not recorded'}. "
                "Reporting volume is not incidence."
            )


def main():
    st.set_page_config(page_title="MEPRank Research Dashboard", layout="wide")
    st.markdown(
        """
        <style>
        .main .block-container {max-width: 1500px; padding-top: 1.6rem;}
        [data-testid="stAppViewContainer"] {color: var(--text-color);}
        [data-testid="stMetric"] {
            background-color: var(--secondary-background-color);
            border: 1px solid rgba(128, 128, 128, 0.28);
            border-radius: 14px;
            padding: 0.8rem 1rem;
        }
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"],
        [data-testid="stMetricDelta"],
        [data-testid="stMetricLabel"] *,
        [data-testid="stMetricValue"] *,
        [data-testid="stMetricDelta"] * {
            color: var(--text-color) !important;
        }
        [data-testid="stTabs"] button {font-weight: 600;}
        [data-testid="stSidebar"] {border-right: 1px solid rgba(128, 128, 128, 0.28);}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Pregnancy Drug Evidence Explorer")
    st.caption(
        "Explore observed source records, molecular structure, and explicitly "
        "timed pregnancy evidence. No unvalidated risk score is shown."
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

    with st.sidebar:
        st.markdown("## Search")
        search_query = st.text_input(
            "Drug name or identifier",
            value="",
            placeholder="e.g. tretinoin, CID000444795, 444795",
            help="Matches a drug name, drug_id, PubChem CID or InChIKey.",
        )
        selected_name = st.selectbox(
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
        selected_timing = st.selectbox("Target timing", timing_options)
        target_label = st.radio(
            "View target",
            ("Fetal Outcomes", "Maternal Outcomes"),
            horizontal=False,
        )
        st.divider()
        st.markdown("## Sections")
        st.caption(
            "Overview → identity and chemistry.  Pregnancy evidence → what the "
            "sources record.  Animal ADR → separate veterinary branch.  "
            "Provenance → raw filtered records.  Data readiness → model status."
        )

    if search_query.strip():
        profile = _build_profile(search_query.strip())
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
            st.info(f"No exact keyword match for '{search_query.strip()}' in the local drug index.")
            suggestions = [
                name
                for name in medication_names
                if search_query.strip().casefold() in name.casefold()
            ]
            if suggestions:
                st.markdown("**Similar indexed drug names**")
                st.dataframe(
                    pd.DataFrame({"Drug name": suggestions[:12]}),
                    hide_index=True,
                    width="stretch",
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
        profile = _build_profile(drug_id or selected_name)
        if profile.get("ambiguous"):
            st.warning(
                f"The selected name '{selected_name}' maps to multiple drug "
                "identifiers. Search by CID or project drug ID to select one."
            )
            st.dataframe(
                pd.DataFrame(profile.get("candidates") or []),
                hide_index=True,
                width="stretch",
            )
            return
        if profile.get("matched"):
            drug_id = str(profile.get("drug_id") or drug_id)
            selected_name = str(
                (profile.get("identity") or {}).get("name") or selected_name
            )

    timing_filter = (
        "All" if selected_timing == "All Trimesters" else selected_timing
    )
    target_filter = (
        "Fetal Outcomes Only"
        if target_label == "Fetal Outcomes"
        else "Maternal Outcomes Only"
    )
    result = _evaluate_query(
        selected_name,
        drug_id,
        timing_filter,
        target_filter,
    )

    overview_tab, evidence_tab, animal_tab, provenance_tab, readiness_tab = st.tabs(
        [
            "Overview",
            "Pregnancy evidence",
            "Animal ADR",
            "Provenance & records",
            "Data readiness",
        ]
    )
    with overview_tab:
        _render_drug_profile(profile)
    with evidence_tab:
        _render_filtered_summary(result, selected_name, target_label, profile)
    with animal_tab:
        _render_animal_adr(result)
    with provenance_tab:
        _render_provenance(result)
    with readiness_tab:
        _render_readiness(data)


if __name__ == "__main__":
    main()
