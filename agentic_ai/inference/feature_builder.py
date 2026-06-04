"""Feature matrix builder for single-sequence inference (Week 4 Block 4.2).

This module is a thin adapter over agentic_ai/features/. It must
produce features that are bit-identical to what training produced
for the same sequence. Any divergence here produces silently wrong
predictions at inference time.

The training pipeline operates on the MOESM3 corpus and produces a
9240-row x 134-column DataFrame (128 features + 6 metadata). For
inference we need the same 128 columns in the same order, for a
single sequence x 15 REEs.

The public entrypoint:
    build_feature_matrix_for_sequence(sequence: str) -> pd.DataFrame
returns a (15, 128) DataFrame ready for model.predict().

The persisted encoder and schema are loaded once via caching and
reused across calls to avoid repeated disk reads.
"""
from __future__ import annotations

import json
import pickle
from functools import lru_cache
from pathlib import Path
from typing import List

import pandas as pd

from agentic_ai.features.ef_hand_motifs import compute_ef_hand_features
from agentic_ai.features.encoding import (
    align_columns_to_schema,
    build_feature_dataframe,
    transform_features,
)
from agentic_ai.features.ree_features import get_ree_features
from agentic_ai.features.sequence_features import compute_basic_features

# The 15 REEs from MOESM3. Order is internal convention only; the model
# receives one row per REE and ordering does not affect predictions.
# Display ordering (atomic number for bar chart, predicted value for
# ranked table) is applied in the UI layer, not here.
_MOESM3_ELEMENTS = [
    "Lanthanum", "Cerium", "Praseodymium", "Neodymium",
    "Samarium", "Europium", "Gadolinium", "Terbium",
    "Dysprosium", "Holmium", "Erbium", "Thulium",
    "Ytterbium", "Lutetium", "Yttrium",
]

_ENCODER_PATH = Path("data/processed/ml_ready_features_encoder.pkl")
_SCHEMA_PATH = Path("data/processed/ml_ready_features_schema.json")

# Prefix used to identify one-hot encoded EF-hand categorical columns.
# Schema columns starting with this prefix may legitimately be absent
# from a single sequence's encoded features (e.g. if its EF-hand motifs
# contain residues the training encoder didn't see). Schema columns NOT
# starting with this prefix are core numeric features that should never
# be missing; their absence indicates a bug, not legitimate variation.
_ONE_HOT_COLUMN_PREFIX = "ef"


@lru_cache(maxsize=1)
def _load_encoder():
    """
    Loads the persisted train-only encoder from disk. Cached so
    repeat inference calls don't re-read the pickle.
    return : OneHotEncoder fitted on training data during Block 2.5.
    raises : FileNotFoundError if the encoder pickle is missing.
    """
    if not _ENCODER_PATH.exists():
        raise FileNotFoundError(
            f"Encoder not found at {_ENCODER_PATH}. "
            f"Run `python -m agentic_ai.features.build_matrix` to regenerate."
        )
    with open(_ENCODER_PATH, "rb") as fh:
        return pickle.load(fh)


@lru_cache(maxsize=1)
def _load_schema() -> List[str]:
    """
    Loads the persisted feature column schema from disk. Cached.
    return : List of 128 feature column names in training order.
    raises : FileNotFoundError if the schema JSON is missing.
    """
    if not _SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"Schema not found at {_SCHEMA_PATH}. "
            f"Run `python -m agentic_ai.features.build_matrix` to regenerate."
        )
    return json.loads(_SCHEMA_PATH.read_text())


def build_feature_matrix_for_sequence(sequence: str) -> pd.DataFrame:
    """
    Constructs the inference-ready feature matrix for one protein
    sequence across all 15 REEs.

    The returned DataFrame has the same 128 feature columns as the
    training matrix, in the same order, with the same encoded
    categorical representation. Each row corresponds to (sequence x REE).

    @param sequence: Validated, normalized protein sequence (uppercase,
                     no whitespace, canonical AAs only). Use
                     sequence_validator.validate_sequence first.
    return : pandas DataFrame of shape (15, 128), dtype consistent with
             the training matrix. Index is reset; ordering matches
             _MOESM3_ELEMENTS.
    raises : FileNotFoundError if encoder/schema artifacts are missing.
             ValueError if sequence is invalid (delegated to feature
             extractors) or if core numeric features are missing
             (indicates a bug).
    """
    # Compute sequence-derived features once (constant across REEs)
    basic = compute_basic_features(sequence)
    ef_hand = compute_ef_hand_features(sequence)
    seq_features = {**basic, **ef_hand}

    # Build 15 rows: one per REE, with sequence features repeated and
    # REE-specific features joined per row.
    rows = []
    for element in _MOESM3_ELEMENTS:
        ree_features = get_ree_features(element)
        if ree_features is None:
            raise ValueError(
                f"REE features unavailable for '{element}'. "
                f"This indicates a code-vs-data version mismatch."
            )
        row = {**seq_features, **ree_features}
        rows.append(row)

    # Combine into a DataFrame using the same primitive used during training.
    raw_df = build_feature_dataframe(rows)
    # Coerce aggregate EF-hand columns to float64 to match training dtype.
    # When a sequence has fewer than 2 motifs, these aggregates return None,
    # which pandas would infer as object dtype here even though training
    # inferred float64 (because the full training corpus had mixed
    # numeric/None values). Coercing at the source prevents downstream
    # dtype divergence between inference and training paths.
    _AGGREGATE_EF_HAND_COLUMNS = [
        "ef_hand_mean_spacing",
        "ef_hand_spacing_stdev",
        "ef_hand_span_fraction",
    ]
    for col in _AGGREGATE_EF_HAND_COLUMNS:
        if col in raw_df.columns:
            raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")
    # Apply the pre-fitted encoder. NEVER fit a new encoder at inference
    # time; that would silently produce different one-hot columns than
    # training and corrupt predictions.
    encoder = _load_encoder()
    encoded_df = transform_features(raw_df, fitted_encoder=encoder)

    # Strict pre-alignment check: schema may contain one-hot columns
    # absent from this sequence's encoded features (those legitimately
    # fill as 0), but core numeric features should NEVER be absent.
    # Catching this here makes bugs loud instead of silent.
    schema = _load_schema()
    missing_numeric = [
        col for col in schema
        if col not in encoded_df.columns
        and not col.startswith(_ONE_HOT_COLUMN_PREFIX)
    ]
    if missing_numeric:
        raise ValueError(
            f"Inference produced a feature matrix missing core numeric "
            f"columns: {missing_numeric}. This indicates a feature-code "
            f"vs schema mismatch and would silently corrupt predictions. "
            f"Investigate before deploying."
        )

    # Reorder + fill missing one-hot columns to match the persisted schema.
    # This guarantees the model receives column names in the exact
    # order it saw during fit().
    aligned_df = align_columns_to_schema(encoded_df, schema_columns=schema)

    # Final shape and ordering sanity checks
    if len(aligned_df) != len(_MOESM3_ELEMENTS):
        raise ValueError(
            f"Expected {len(_MOESM3_ELEMENTS)} rows for the 15 REEs, "
            f"got {len(aligned_df)}."
        )
    if list(aligned_df.columns) != schema:
        raise ValueError(
            "Feature matrix columns do not match schema after alignment. "
            "This indicates a bug in align_columns_to_schema."
        )

        # Final dtype check: every feature column must be numeric. Catches
        # any future feature additions that accidentally introduce object
        # dtypes. NaN is fine; non-numeric types are not.
    non_numeric = aligned_df.select_dtypes(exclude=["number"]).columns.tolist()
    if non_numeric:
        raise ValueError(
            f"Feature matrix contains non-numeric columns after alignment: "
            f"{non_numeric}. XGBoost requires all features to be numeric. "
            f"NaN is acceptable for missing values."
        )

    return aligned_df.reset_index(drop=True)


def get_element_order() -> List[str]:
    """
    Returns the canonical element order used in inference output. The
    i-th row of build_feature_matrix_for_sequence's output corresponds
    to the i-th element in this list.
    return : List of 15 element name strings.
    """
    return list(_MOESM3_ELEMENTS)
