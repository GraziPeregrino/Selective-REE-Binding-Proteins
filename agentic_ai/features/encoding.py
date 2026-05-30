"""Feature encoding utilities for ML-ready matrix construction
(Week 2 Block 2.2c).

Layer separation:
  - ef_hand_motifs.py is the biological layer: detects motifs and
    returns a dict of raw biological features per sequence.
  - encoding.py is the ML preprocessing layer: takes lists of those
    dicts and produces a sklearn-compatible feature matrix.

Categorical features (per-motif residue identities) become one-hot
columns via sklearn's OneHotEncoder. Missing residues (absent EF-hands)
are encoded as an explicit 'MISSING' category, making absence a
learnable signal rather than relying on tree models' default-split
behavior for NaN categoricals.

Numeric aggregate features pass through unchanged. NaN remains NaN
because XGBoost handles missing numeric values natively at split time.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Sentinel string substituted for None/NaN in categorical columns
# before one-hot encoding. Becomes a regular feature column
# (e.g. 'ef4_motif_pos0_MISSING') that the model can learn from.
_MISSING_SENTINEL = "MISSING"

# Categorical column names produced by compute_ef_hand_features.
# These get one-hot encoded; everything else passes through.
_CATEGORICAL_FEATURE_NAMES = tuple(
    f"ef{ef_number}_motif_pos{pos}"
    for ef_number in range(1, 5)
    for pos in (0, 2, 5)
)


def build_feature_dataframe(
    feature_dicts: List[Dict] = None,
) -> pd.DataFrame:
    """
    Combines per-sequence feature dicts into a single pandas DataFrame.
    Preserves the original column ordering from the feature extractor;
    rows are ordered by input position.
    @param feature_dicts: List of dicts (output of
                          compute_ef_hand_features). Empty list or
                          None returns an empty DataFrame.
    return : pandas DataFrame with one row per input dict. Categorical
             columns retain string residue codes (or None for missing
             motifs). Numeric columns retain int/float (or NaN for
             undefined aggregates). No encoding is applied here; use
             encode_categorical_features() for the ML-ready matrix.
    """
    if not feature_dicts:
        return pd.DataFrame()

    return pd.DataFrame(feature_dicts)


def encode_categorical_features(
    df: pd.DataFrame = None,
    fitted_encoder: OneHotEncoder = None,
) -> Tuple[pd.DataFrame, OneHotEncoder]:
    """
    One-hot encodes the categorical per-motif residue columns. Numeric
    aggregate columns pass through unchanged. NaN/None in categorical
    columns becomes an explicit 'MISSING' category column.
    @param df: DataFrame from build_feature_dataframe.
    @param fitted_encoder: Optional pre-fitted OneHotEncoder. When
                           provided, only transforms (used for inference
                           with the train-time encoder). When None,
                           fits a new encoder on df and returns it for
                           later reuse.
    return : Tuple of (encoded DataFrame, fitted OneHotEncoder). The
             encoder must be persisted alongside the trained model so
             that inference on new sequences uses the same column
             schema.
    raises : ValueError if df is None or empty.
    """
    if df is None or df.empty:
        raise ValueError("Cannot encode empty DataFrame")

    categorical_cols = [
        c for c in _CATEGORICAL_FEATURE_NAMES if c in df.columns
    ]
    numeric_cols = [c for c in df.columns if c not in categorical_cols]

    # Replace None/NaN in categorical columns with the sentinel so
    # OneHotEncoder treats 'missing motif' as a learnable category.
    categorical_df = df[categorical_cols].fillna(_MISSING_SENTINEL)

    if fitted_encoder is None:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
        encoded_array = encoder.fit_transform(categorical_df)
    else:
        encoder = fitted_encoder
        encoded_array = encoder.transform(categorical_df)

    # Build column names like 'ef1_motif_pos0_P', 'ef1_motif_pos0_K', ...
    encoded_column_names = encoder.get_feature_names_out(categorical_cols)
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoded_column_names,
        index=df.index,
    )

    # Concatenate the encoded categoricals with the untouched numerics
    numeric_df = df[numeric_cols]
    result = pd.concat([encoded_df, numeric_df], axis=1)

    return result, encoder


def fit_encoder(df: pd.DataFrame = None) -> OneHotEncoder:
    """
    Convenience wrapper: fits and returns the OneHotEncoder without
    transforming. Useful when you want to fit on training data and
    inspect the categories before applying.
    @param df: DataFrame from build_feature_dataframe.
    return : Fitted OneHotEncoder.
    """
    _, encoder = encode_categorical_features(df=df)
    return encoder


def transform_features(
    df: pd.DataFrame = None,
    fitted_encoder: OneHotEncoder = None,
) -> pd.DataFrame:
    """
    Applies a pre-fitted encoder to new data. Use this for Week 4
    Streamlit inference: the encoder is loaded from disk alongside the
    XGBoost model, then applied to user-submitted sequences.
    @param df: DataFrame from build_feature_dataframe.
    @param fitted_encoder: OneHotEncoder previously fit on training data.
    return : Encoded DataFrame ready for XGBoost prediction.
    raises : ValueError if fitted_encoder is None.
    """
    if fitted_encoder is None:
        raise ValueError(
            "transform_features requires a fitted_encoder. "
            "Use encode_categorical_features() for the train-time path."
        )

    encoded_df, _ = encode_categorical_features(
        df=df, fitted_encoder=fitted_encoder,
    )
    return encoded_df


def align_columns_to_schema(
    df: pd.DataFrame = None,
    schema_columns: list = None,
) -> pd.DataFrame:
    """
    Reindexes a DataFrame to match a pre-saved column schema, filling
    missing columns with 0 and dropping any extra columns. Used at
    inference time (Week 4 Streamlit) to guarantee the model receives
    the same column ordering and presence it saw during training.
    The training pipeline must persist the schema explicitly:
        encoded_df, encoder = encode_categorical_features(train_df)
        schema_columns = encoded_df.columns.tolist()
        # save schema_columns + encoder to disk alongside the model
    Then at inference:
        new_encoded_df = transform_features(new_df, fitted_encoder)
        aligned_df = align_columns_to_schema(new_encoded_df, schema_columns)
        prediction = model.predict(aligned_df)
    @param df: A DataFrame produced by transform_features.
    @param schema_columns: The column list saved at training time.
    return : A DataFrame with exactly the schema_columns, in the same
             order, with missing columns filled as 0.
    raises : ValueError if either argument is None.
    """
    if df is None or schema_columns is None:
        raise ValueError(
            "Both df and schema_columns are required. "
            "schema_columns must be persisted at training time."
        )

    return df.reindex(columns=schema_columns, fill_value=0)
