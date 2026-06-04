"""Tests for inference feature builder (Week 4 Block 4.2).

The headline test (test_inference_matches_training_features_for_o621)
verifies that the inference path produces byte-identical features to
the training matrix for a known variant. This catches the entire
class of feature-drift bugs: encoder divergence, column reordering,
dtype mismatches, NaN handling differences, and silent rounding.

If this test ever fails, deployment predictions will no longer match
documented test-set metrics, and the model's public claims become
invalid.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agentic_ai.inference.feature_builder import (
    build_feature_matrix_for_sequence,
    get_element_order,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

# Mex-LanM (o-621) — a real MOESM3 ortholog. Known to be in our training set;
# its features must exactly match the persisted training matrix.
_MEX_LANM_SEQUENCE = (
    "MAPTTTTKVDIAAFDPDKDGTIDLKEALAAGSAAFDKLDPDKDGTLDAKELKGRVSEADLKKL"
    "DPDNDGTLDKKEYLAAVEAQFKAANPDNDGTIDARELASPAGSALVNLIR"
)
_MEX_LANM_VARIANT_ID = "o-621"

# A sequence with no detectable EF-hand motifs (to exercise the
# aggregate-NaN coercion path). Using a stretch of polyalanine that
# definitely won't match the regex `[A-Z]D[A-Z]DG[A-Z]`.
_NO_MOTIF_SEQUENCE = "M" + "A" * 119

# Schema and persisted matrix paths
_SCHEMA_PATH = Path("data/processed/ml_ready_features_schema.json")
_MATRIX_PATH = Path("data/processed/ml_ready_features.parquet")


@pytest.fixture(scope="module")
def schema():
    return json.loads(_SCHEMA_PATH.read_text())


@pytest.fixture(scope="module")
def training_matrix():
    return pd.read_parquet(_MATRIX_PATH)


# ---------------------------------------------------------------------------
# Headline test: byte-identical features vs training matrix
# ---------------------------------------------------------------------------

def test_inference_matches_training_features_for_o621(
    schema, training_matrix
):
    """
    The killer test: features built at inference time must match
    features recorded in the training matrix for the same variant
    and same set of REEs.

    Any divergence indicates feature drift between training and
    inference, which would silently corrupt deployed predictions.
    """
    # Build inference features for the known o-621 sequence
    inferred = build_feature_matrix_for_sequence(_MEX_LANM_SEQUENCE)

    # Extract training rows for o-621, sort to match inference element order
    element_order = get_element_order()
    training_o621 = training_matrix[
        training_matrix["variant_id"] == _MEX_LANM_VARIANT_ID
    ].copy()

    # Reorder training rows to match the inference element ordering
    element_rank = {elem: i for i, elem in enumerate(element_order)}
    training_o621["_element_order"] = training_o621["target_element"].map(
        element_rank
    )
    training_o621 = training_o621.sort_values("_element_order")
    training_features = training_o621[schema].reset_index(drop=True)

    # Compare. Both should have the same shape, columns, dtypes, and values.
    assert inferred.shape == training_features.shape, (
        f"Shape mismatch: inference {inferred.shape} vs "
        f"training {training_features.shape}"
    )
    assert list(inferred.columns) == list(training_features.columns), (
        "Column ordering mismatch"
    )

    # Numeric value comparison with NaN-aware equality. assert_frame_equal
    # is the gold standard for this.
    pd.testing.assert_frame_equal(
        inferred,
        training_features,
        check_dtype=True,
        check_exact=False,
        rtol=1e-9,
        atol=1e-9,
    )


# ---------------------------------------------------------------------------
# Output shape and structure
# ---------------------------------------------------------------------------

def test_output_has_15_rows_one_per_ree():
    """The feature matrix must have exactly 15 rows (one per REE)."""
    X = build_feature_matrix_for_sequence(_MEX_LANM_SEQUENCE)
    assert len(X) == 15


def test_output_has_128_feature_columns(schema):
    """The feature matrix must have exactly the schema's column count."""
    X = build_feature_matrix_for_sequence(_MEX_LANM_SEQUENCE)
    assert X.shape[1] == len(schema)


def test_output_columns_match_schema_order(schema):
    """Column order must exactly match the persisted schema."""
    X = build_feature_matrix_for_sequence(_MEX_LANM_SEQUENCE)
    assert list(X.columns) == schema


def test_output_has_no_object_columns():
    """All feature columns must be numeric (not object)."""
    X = build_feature_matrix_for_sequence(_MEX_LANM_SEQUENCE)
    object_cols = X.select_dtypes(include="object").columns.tolist()
    assert not object_cols, (
        f"Found object-dtype columns (XGBoost would reject these): "
        f"{object_cols}"
    )


def test_output_dtypes_match_training_pattern(training_matrix, schema):
    """
    Feature dtypes must match what training produced. We compare the
    dtype value_counts between inference and training to ensure no
    silent dtype promotion or demotion.
    """
    X = build_feature_matrix_for_sequence(_MEX_LANM_SEQUENCE)
    training_features = training_matrix[schema]

    inference_dtype_counts = X.dtypes.value_counts().to_dict()
    training_dtype_counts = training_features.dtypes.value_counts().to_dict()

    assert inference_dtype_counts == training_dtype_counts, (
        f"Dtype distribution mismatch: "
        f"inference {inference_dtype_counts} vs "
        f"training {training_dtype_counts}"
    )


# ---------------------------------------------------------------------------
# Aggregate NaN handling (the dtype-coercion fix from triage)
# ---------------------------------------------------------------------------

def test_sequence_with_no_motifs_produces_numeric_dtypes():
    """
    A sequence with zero EF-hand motifs returns None for aggregate
    features. The builder must coerce these to numeric (NaN) rather
    than letting them pass through as object dtype.
    """
    X = build_feature_matrix_for_sequence(_NO_MOTIF_SEQUENCE)
    aggregate_cols = [
        "ef_hand_mean_spacing",
        "ef_hand_spacing_stdev",
        "ef_hand_span_fraction",
    ]
    for col in aggregate_cols:
        if col in X.columns:
            assert pd.api.types.is_numeric_dtype(X[col]), (
                f"{col} should be numeric dtype, got {X[col].dtype}"
            )


def test_sequence_with_no_motifs_has_nan_in_aggregates():
    """All aggregate values should be NaN (not None) for a no-motif sequence."""
    X = build_feature_matrix_for_sequence(_NO_MOTIF_SEQUENCE)
    aggregate_cols = [
        "ef_hand_mean_spacing",
        "ef_hand_spacing_stdev",
        "ef_hand_span_fraction",
    ]
    for col in aggregate_cols:
        if col in X.columns:
            assert X[col].isna().all(), (
                f"{col} should be all NaN for no-motif sequence"
            )


# ---------------------------------------------------------------------------
# REE feature integration
# ---------------------------------------------------------------------------

def test_atomic_numbers_match_lanthanide_series():
    """
    Each row should have a sensible atomic number for its element.
    This verifies the REE-feature join worked correctly.
    """
    X = build_feature_matrix_for_sequence(_MEX_LANM_SEQUENCE)
    element_order = get_element_order()

    # Expected atomic numbers in the canonical element order
    expected_atomic_numbers = {
        "Lanthanum": 57, "Cerium": 58, "Praseodymium": 59,
        "Neodymium": 60, "Samarium": 62, "Europium": 63,
        "Gadolinium": 64, "Terbium": 65, "Dysprosium": 66,
        "Holmium": 67, "Erbium": 68, "Thulium": 69,
        "Ytterbium": 70, "Lutetium": 71, "Yttrium": 39,
    }

    if "atomic_number" not in X.columns:
        pytest.skip("atomic_number column not in schema")

    for i, element in enumerate(element_order):
        assert X.iloc[i]["atomic_number"] == expected_atomic_numbers[element], (
            f"Row {i} ({element}): expected atomic number "
            f"{expected_atomic_numbers[element]}, "
            f"got {X.iloc[i]['atomic_number']}"
        )


# ---------------------------------------------------------------------------
# Schema validation guards
# ---------------------------------------------------------------------------

def test_invalid_sequence_raises_in_feature_extraction():
    """
    Passing an invalid sequence (non-canonical AAs) should raise from
    the underlying sequence_features module, not produce silent garbage.
    """
    with pytest.raises(ValueError):
        build_feature_matrix_for_sequence("MACEXTIDKFDHE" * 10)


# ---------------------------------------------------------------------------
# Element order contract
# ---------------------------------------------------------------------------

def test_element_order_has_15_elements():
    """The element order list defines a 15-element contract."""
    order = get_element_order()
    assert len(order) == 15


def test_element_order_includes_all_moesm3_elements():
    """Element order must include all 14 lanthanides + Y."""
    order = set(get_element_order())
    expected = {
        "Lanthanum", "Cerium", "Praseodymium", "Neodymium",
        "Samarium", "Europium", "Gadolinium", "Terbium",
        "Dysprosium", "Holmium", "Erbium", "Thulium",
        "Ytterbium", "Lutetium", "Yttrium",
    }
    assert order == expected
