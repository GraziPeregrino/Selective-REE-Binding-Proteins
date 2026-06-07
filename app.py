"""Streamlit UI for the REE Selectivity Predictor (Week 4 Block 4.4).

Single-sequence predictor for Lanmodulin-like protein sequences.
Loads the deployed XGBoost model via the inference layer and renders:
  - Sequence input with example button
  - Ranked predictions table
  - Selectivity profile bar chart (sorted by atomic number)
  - Per-sequence and per-REE warnings
  - About panel with model provenance and limitations

Per the inference contract (docs/block_3_5_inference_contract.md),
all output language describes predictions as on-resin normalized
selectivity scores, NEVER as binding affinity or Kd.

Run locally:
    streamlit run app.py
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from agentic_ai.inference.predictor import (
    get_model_info,
    predict_profile,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="REE Selectivity Predictor",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Mex-LanM (o-621): a real MOESM3 ortholog used as the example sequence.
# Length 113, canonical 4 EF-hand motifs, well-documented in literature.
_EXAMPLE_SEQUENCE = (
    "MAPTTTTKVDIAAFDPDKDGTIDLKEALAAGSAAFDKLDPDKDGTLDAKELKGRVSEADLKKL"
    "DPDNDGTLDKKEYLAAVEAQFKAANPDNDGTIDARELASPAGSALVNLIR"
)

# Element symbol mapping for compact chart x-axis labels
_ELEMENT_SYMBOLS = {
    "Lanthanum": "La", "Cerium": "Ce", "Praseodymium": "Pr",
    "Neodymium": "Nd", "Samarium": "Sm", "Europium": "Eu",
    "Gadolinium": "Gd", "Terbium": "Tb", "Dysprosium": "Dy",
    "Holmium": "Ho", "Erbium": "Er", "Thulium": "Tm",
    "Ytterbium": "Yb", "Lutetium": "Lu", "Yttrium": "Y",
}

_GITHUB_URL = "https://github.com/linlin-husky/Selective-REE-Binding-Proteins"


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

if "sequence_input" not in st.session_state:
    st.session_state.sequence_input = ""
if "result" not in st.session_state:
    st.session_state.result = None


def _load_example():
    """Loads the Mex-LanM example sequence into the input area."""
    st.session_state.sequence_input = _EXAMPLE_SEQUENCE
    st.session_state.result = None  # clear prior results on example load

def _erase_input():
    """Clears the sequence input and any prior results."""
    st.session_state.sequence_input = ""
    st.session_state.result = None
# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("REE Selectivity Predictor")
st.markdown(
    "Predict the rare earth element (REE) selectivity profile of a "
    "Lanmodulin-like protein from its amino acid sequence."
)

# ---------------------------------------------------------------------------
# Input section
# ---------------------------------------------------------------------------

st.subheader("Sequence input")

sequence_input = st.text_area(
    label="Protein sequence (single-letter amino acid codes)",
    height=160,
    placeholder=(
        "Paste a protein sequence here, e.g.:\n"
        "MAPTTTTKVDIAAFDPDKDGTIDLKEALAAGSAAFDKLDPDKDGT..."
    ),
    key="sequence_input",
    help=(
        "Single-letter amino acid codes. FASTA headers are tolerated "
        "(one record only). Supported length: 80-400 residues."
    ),
)

col1, col2, col3, _ = st.columns([1, 1, 1, 2])
with col1:
    st.button(
        "Load example",
        on_click=_load_example,
        help="Loads Mex-LanM (o-621), a real LanM ortholog from MOESM3.",
    )
with col2:
    st.button(
        "Erase",
        on_click=_erase_input,
        help="Clears the sequence input and any prior results.",
    )
with col3:
    predict_clicked = st.button(
        "Predict",
        type="primary",
        help="Run inference on the entered sequence.",
    )

# Run inference when Predict is clicked
if predict_clicked:
    if not sequence_input.strip():
        st.warning("Please enter a sequence before clicking Predict.")
        st.session_state.result = None
    else:
        with st.spinner("Computing prediction..."):
            st.session_state.result = predict_profile(sequence_input)

# ---------------------------------------------------------------------------
# Validation summary
# ---------------------------------------------------------------------------

result = st.session_state.result

if result is not None:
    if result["is_valid"]:
        st.success(
            f"Valid {result['length']}-residue sequence  •  "
            f"{result['n_motifs']} EF-hand motif"
            f"{'s' if result['n_motifs'] != 1 else ''} detected"
        )
    else:
        st.error(f"Invalid sequence: {result['error']}")

# ---------------------------------------------------------------------------
# Output caveat (always shown above results, per Block 3.4)
# ---------------------------------------------------------------------------

if result is not None and result["is_valid"]:
    st.info(
        "Predictions are normalized on-resin selectivity scores, not "
        "binding affinities or Kd values. See the About panel for "
        "interpretation guidance."
    )

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

if result is not None and result["is_valid"]:
    predictions_df = result["predictions"].copy()
    predictions_df["element_symbol"] = predictions_df["target_element"].map(
        _ELEMENT_SYMBOLS
    )

    # ----- Ranked table -----
    st.subheader("Ranked predictions")

    display_df = predictions_df[[
        "rank", "target_element", "element_symbol",
        "predicted_normalized_logD", "confidence_note",
    ]].copy()
    display_df.columns = ["Rank", "Element", "Symbol", "Predicted", "Notes"]
    display_df["Predicted"] = display_df["Predicted"].round(4)

    st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True,
    )

    # ----- Selectivity profile chart (sorted by atomic number) -----
    st.subheader("Selectivity profile")
    st.caption("Bars sorted by atomic number (Y shown alongside Ln series).")

    chart_df = predictions_df.copy()

    # Color: highlight Samarium differently to reinforce the confidence note
    chart_df["color_category"] = chart_df["target_element"].apply(
        lambda e: "Samarium (reduced confidence)" if e == "Samarium"
        else "Other REEs"
    )

    chart = alt.Chart(chart_df).mark_bar().encode(
        x=alt.X(
            "element_symbol:N",
            sort=alt.SortField(field="atomic_number", order="ascending"),
            title="REE (sorted by atomic number)",
            axis=alt.Axis(labelAngle=0),
        ),
        y=alt.Y(
            "predicted_normalized_logD:Q",
            title="Predicted normalized on-resin logD",
            scale=alt.Scale(domain=[0, 1.05]),
        ),
        color=alt.Color(
            "color_category:N",
            scale=alt.Scale(
                domain=["Other REEs", "Samarium (reduced confidence)"],
                range=["#4C78A8", "#E45756"],
            ),
            legend=alt.Legend(title=None, orient="top"),
        ),
        tooltip=[
            alt.Tooltip("target_element:N", title="Element"),
            alt.Tooltip("atomic_number:Q", title="Atomic number"),
            alt.Tooltip(
                "predicted_normalized_logD:Q",
                title="Predicted score",
                format=".4f",
            ),
            alt.Tooltip("rank:Q", title="Rank"),
            alt.Tooltip("confidence_note:N", title="Note"),
        ],
    ).properties(
        height=320,
    )

    st.altair_chart(chart, use_container_width=True)

    # ----- Warnings -----
    if result["warnings"]:
        st.subheader("Interpretation notes")
        for warning in result["warnings"]:
            st.warning(warning)

# ---------------------------------------------------------------------------
# About panel
# ---------------------------------------------------------------------------

with st.expander("About this model"):
    info = get_model_info()
    st.markdown(f"""
**Model**: {info['model_name']}

**Version**: `{info['model_version']}`

**Training data**: 442 Lanmodulin orthologs from the MOESM3 corpus
of Diep et al. (2026). The model was trained on 6,630 (variant × REE)
measurements and evaluated on 1,860 held-out measurements from 124
unseen variants.

**Held-out test metrics**:
- Macro per-element Spearman: **0.557** (rank candidate proteins for a target REE)
- Macro per-variant Spearman: **0.938** (recover a protein's selectivity profile)
- R²: **0.851** (row-level variance explained)
- RMSE: **0.128** (target range 0–1)

**What is predicted**: Per-REE normalized on-resin logD as defined
in MOESM3 of Diep et al. (2026). This is a relative selectivity
score in [0, 1] within each protein, NOT an absolute binding
affinity, NOT a Kd value, and NOT directly comparable across
proteins without context.

**Samarium confidence caveat**: 44.1% of LanM orthologs in the
training data have Sm = 1.0 due to per-variant normalization;
81.5% have Sm ≥ 0.95. This compression reduces the model's ability
to rank proteins by Sm preference. See Block 3.3 in the project
repository for the full investigation.

**Cross-assay caveat**: An exploratory comparison against literature
in-solution Kd measurements (Block 3.4) showed partial directional
concordance with important rank-order differences for some variants.
On-resin selectivity and in-solution binding affinity are related
but distinct quantities; outputs here describe the former.

**Limitations**:
- Trained on natural LanM orthologs only. Predictions for engineered
  chelators, fusion proteins, or point mutants are unvalidated.
- No prediction uncertainty (point estimates only).
- Outside the validated input regime (80–400 residues, canonical
  amino acids, LanM-like architecture), behavior is undefined.

**Source code**: [{_GITHUB_URL}]({_GITHUB_URL})
    """)

st.caption(
    "Built with Streamlit. For research and educational use only."
)
