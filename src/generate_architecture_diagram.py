import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def generate_diagram():
    """Generate a publication-ready 4-tier Human-Only System Architecture diagram.

    Renders on a solid white background using matplotlib (no graphviz system
    binary required). Saved as PNG at reports/figures/human_system_architecture.png.
    """
    fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 70)
    ax.axis("off")

    # Tier color palette (human-only architecture)
    tier_colors = {
        1: ("#FEFCBF", "#D69E2E"),   # Data Ingestion - pale yellow / amber
        2: ("#C6F6D5", "#38A169"),   # Processing - pale green / green
        3: ("#EBF8FF", "#3182CE"),   # Reasoning - pale blue / blue
        4: ("#FED7D7", "#C53030"),   # Presentation - pale red / dark red
    }

    tier_specs = [
        {
            "tier": 1,
            "label": "Tier 1 — Data Ingestion Layer\n(datasets/raw  →  src/ingestion/)",
            "nodes": [
                ("data_sources", "Raw Human Datasets\n(OnSIDES Label Side-Effects ·\nMendeley Pregnancy Outcome Counts ·\nTGA Safety Classifications ·\nFDA FAERS Reports ·\nPubChem SMILES/CID Mapping)", 14, 52),
                ("ingestion_pipe", "src/ingestion/pipeline.py — Orchestrator\n(pubchem.py · mendeley.py · onsides.py · dryad.py)", 14, 30),
            ],
        },
        {
            "tier": 2,
            "label": "Tier 2 — Processing & Standardization\nLayer (src/)",
            "nodes": [
                ("preprocessing", "src/data_preprocessing.py\n• Generic Name → CID / InChIKey / SMILES\n• Trimester Exposure Schema\n• Disproportionality Signals (ROR / PRR)\n• Availability Mask (M) & Quality Vector (Q)", 50, 48),
                ("process_real", "src/process_real_datasets.py\n• Human-only ingestion coordination\n• Master evidence standardization\n• Source acquisition manifest", 50, 26),
            ],
        },
        {
            "tier": 3,
            "label": "Tier 3 — ML / AI Reasoning Layer\n(src/models/  &  src/mepr_reasoning.py)",
            "nodes": [
                ("fusion_model", "src/models/fusion.py\n• Mask-Aware Cross-Attention Gating\n• Conditioned on Availability Mask M\n• Conditioned on Quality Vector Q\n• Inputs: SMILES, OnSIDES, FAERS", 86, 52),
                ("train_safeguard", "src/train.py — Scientific Safeguard\nBlocks unvalidated supervised outputs\nwhen human baseline labels are absent", 86, 34),
                ("mepr_engine", "src/mepr_reasoning.py — 2-Stage Evaluation\n• Stage 1: Drug × Trimester × Outcome Target\n• Stage 2: Q_cov < 0.40 → AI Fallback\n• ml_prediction_active dynamic routing\n• Foetus (T1/T2/T3/Entire) + Maternal risk", 86, 14),
            ],
        },
        {
            "tier": 4,
            "label": "Tier 4 — Presentation & Validation\nLayer (dashboard/  &  tests/)",
            "nodes": [
                ("streamlit_app", "dashboard/app.py — Streamlit UI\n• Query Filter Panel (Drug · Trimester · Fetal/Maternal)\n• Filtered Evidence Cards & Provenance Badges\n• AI Fallback & Evidence Synthesis Indicator", 50, 5),
                ("test_suite", "tests/test_full_system_components.py\n• 34+ Automated System Component Tests", 14, 5),
            ],
        },
    ]

    # Draw tier backgrounds
    for spec in tier_specs:
        tier = spec["tier"]
        bg, border = tier_colors[tier]
        # Determine bounding box for this tier's nodes
        node_x = [n[2] for n in spec["nodes"]]
        node_y = [n[3] for n in spec["nodes"]]
        cx = sum(node_x) / len(node_x)
        cy = sum(node_y) / len(node_y)
        width = 36 if tier in (1, 2, 4) else 32
        height = 30 if tier == 3 else 20
        left = cx - width / 2
        bottom = cy - height / 2

        tier_box = FancyBboxPatch(
            (left, bottom),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            linewidth=2,
            edgecolor=border,
            facecolor=bg,
            alpha=0.85,
            zorder=1,
        )
        ax.add_patch(tier_box)

        # Tier label (placed inside top of tier box)
        ax.text(
            cx,
            bottom + height - 1.2,
            spec["label"],
            ha="center",
            va="top",
            fontsize=10.5,
            fontweight="bold",
            color=border,
            fontname="Arial",
            zorder=4,
        )

    # Draw nodes
    node_positions = {}
    for spec in tier_specs:
        for node_id, node_text, x, y in spec["nodes"]:
            tier = spec["tier"]
            _, border = tier_colors[tier]
            # Determine node style based on content
            if "Orchestrator" in node_text or "standardization" in node_text:
                fill = "#2D3748"
                text_color = "#FFFFFF"
            elif "Safeguard" in node_text:
                fill = "#C53030"
                text_color = "#FFFFFF"
            elif "Evaluation" in node_text:
                fill = "#2D3748"
                text_color = "#FFFFFF"
            else:
                fill = "#FFFFFF"
                text_color = "#1A202C"

            box = FancyBboxPatch(
                (x - 8, y - 4.5),
                16,
                9,
                boxstyle="round,pad=0.01,rounding_size=0.05",
                linewidth=1.5,
                edgecolor=border,
                facecolor=fill,
                alpha=0.95,
                zorder=2,
            )
            ax.add_patch(box)

            # Multi-line text with smaller font for code nodes
            is_code = fill in ("#2D3748", "#C53030")
            fontsize = 8.2 if is_code else 9.0
            lines = node_text.split("\n")
            for i, line in enumerate(lines):
                ax.text(
                    x,
                    y + 1.8 - i * 1.75,
                    line,
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    color=text_color,
                    fontname="Arial",
                    fontweight="bold" if is_code else "normal",
                    zorder=3,
                )
            node_positions[node_id] = (x, y)

    # Draw edges
    edges = [
        ("ingestion_pipe", "preprocessing", "Raw Human\nData Streams"),
        ("process_real", "fusion_model", "Master Dataset\n& Features"),
        ("mepr_engine", "streamlit_app", "Filtered Evidence\n& Risk Scores"),
        ("test_suite", "mepr_engine", "validates"),
        ("test_suite", "streamlit_app", "validates"),
    ]

    for src, dst, label in edges:
        x1, y1 = node_positions[src]
        x2, y2 = node_positions[dst]

        # Determine curve direction
        if x1 < x2:
            # Left to right
            rad = 0.15
        elif x1 > x2:
            # Right to left
            rad = -0.15
        else:
            rad = 0.0

        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            linewidth=1.8,
            color="#4A5568",
            alpha=0.8,
            mutation_scale=14,
            zorder=0,
        )
        ax.add_patch(arrow)

        # Edge label at midpoint
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        # Offset label slightly perpendicular to edge
        if rad != 0:
            mx += 0 if x1 == x2 else (1 if x1 < x2 else -1) * 1.5
            my += 1.2 if rad > 0 else -1.2
        ax.text(
            mx,
            my,
            label,
            ha="center",
            va="center",
            fontsize=7.5,
            color="#2D3748",
            fontname="Arial",
            fontweight="bold",
            zorder=3,
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="#FFFFFF",
                edgecolor="#CBD5E0",
                alpha=0.85,
                linewidth=0.5,
            ),
        )

    # Add dashed edge for fusion_model -> train_safeguard (blocked)
    x1, y1 = node_positions["fusion_model"]
    x2, y2 = node_positions["train_safeguard"]
    arrow = FancyArrowPatch(
        (x1, y1 - 1),
        (x2, y2 + 1),
        connectionstyle="arc3,rad=0.0",
        arrowstyle="-|>",
        linewidth=1.8,
        color="#C53030",
        alpha=0.7,
        mutation_scale=14,
        linestyle="--",
        zorder=0,
    )
    ax.add_patch(arrow)
    ax.text(
        (x1 + x2) / 2,
        (y1 + y2) / 2 + 0.5,
        "blocked",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#C53030",
        fontname="Arial",
        fontweight="bold",
        zorder=3,
    )

    # Footer note
    ax.text(
        50,
        -2.5,
        "4-Tier Human-Only System Architecture — Animal/nonclinical data (ReproTox-KG, ToxRefDB)\n"
        "excluded from preprocessing, fusion, training, and UI. Scientific safeguard blocks "
        "unvalidated supervised outputs.",
        ha="center",
        va="top",
        fontsize=8.5,
        color="#4A5568",
        fontname="Arial",
        style="italic",
    )

    plt.tight_layout(pad=0.4)

    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "human_system_architecture.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig)
    print(f"Saved architecture diagram to {output_path}")


if __name__ == "__main__":
    generate_diagram()
