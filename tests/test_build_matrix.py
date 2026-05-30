"""Tests for final ML feature-matrix assembly (Week 2 Block 2.5)."""
from __future__ import annotations

import json
import pickle

import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder

from agentic_ai.features.build_matrix import (
    _assign_train_test_split,
    _compute_element_features_cached,
    _join_features_to_rows,
    build_feature_matrix,
    get_feature_columns,
    save_feature_matrix,
)


def test_build_feature_matrix_has_expected_shape_and_no_group_leakage():
    """
    Verifies the persisted training-table contract and ensures that no
    variant appears in both train and test.
    """
    df = build_feature_matrix()
    feature_columns = get_feature_columns(df)
    train_variants = set(df.loc[df["split"] == "train", "variant_id"])
    test_variants = set(df.loc[df["split"] == "test", "variant_id"])

    assert df.shape == (9240, 134)
    assert len(feature_columns) == 128
    assert train_variants.isdisjoint(test_variants)
    assert not df["split"].isna().any()
    assert {"selectivity_cluster", "value", "variant_id"}.isdisjoint(
        feature_columns
    )


def test_element_cache_raises_for_unknown_element():
    """
    Prevents an unknown element from silently becoming all-NaN REE
    features in the final matrix.
    """
    df = pd.DataFrame({"target_element": ["Neodymium", "Unobtainium"]})

    with pytest.raises(ValueError, match="Unobtainium"):
        _compute_element_features_cached(df)


def test_row_join_raises_for_missing_sequence_features():
    """
    Prevents measurements without sequence-derived features from
    silently producing incomplete rows.
    """
    df = pd.DataFrame({
        "variant_id": ["o-missing"],
        "target_element": ["Neodymium"],
    })

    with pytest.raises(ValueError, match="o-missing"):
        _join_features_to_rows(
            df,
            sequence_features_by_id={},
            element_features_by_name={"Neodymium": {"atomic_number": 60}},
        )


def test_split_raises_for_missing_cluster():
    """
    Prevents variants without a cluster from receiving NaN split
    assignments.
    """
    df = pd.DataFrame({
        "variant_id": ["o-1", "o-2"],
        "selectivity_cluster": [0, None],
    })

    with pytest.raises(ValueError, match="o-2"):
        _assign_train_test_split(df, test_size=0.2, random_state=42)


def test_save_feature_matrix_writes_schema_and_encoder(tmp_path):
    """
    Verifies persistence of the preprocessing artifacts required by
    Week 3 training and Week 4 inference.
    """
    df = pd.DataFrame({
        "measurement_id": ["m-1"],
        "variant_id": ["o-1"],
        "target_element": ["Neodymium"],
        "selectivity_cluster": [0],
        "split": ["train"],
        "value": [0.5],
        "feature_a": [1.0],
    })
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(pd.DataFrame({"residue": ["D"]}))
    output_path = tmp_path / "matrix.parquet"

    save_feature_matrix(
        df=df,
        output_path=output_path,
        fitted_encoder=encoder,
    )

    schema_path = tmp_path / "matrix_schema.json"
    encoder_path = tmp_path / "matrix_encoder.pkl"
    assert json.loads(schema_path.read_text()) == ["feature_a"]
    assert isinstance(pickle.loads(encoder_path.read_bytes()), OneHotEncoder)
