"""Final feature matrix assembly for ML training (Week 2 Block 2.5).

Combines all three feature families (basic sequence, EF-hand motif,
REE physicochemical) into a single DataFrame matching the row
structure of MOESM3's selectivity dataset. Then encodes categorical
features and adds a train/test split column.

Output: data/processed/ml_ready_features.parquet
Default shape: 9240 rows x (128 features + 6 metadata columns)

The encoded feature count depends on the train/test split because the
one-hot encoder is fit on training variants only. With the default seed,
11 residue categories appear only in held-out variants and are treated
as unknown categories during transformation.

Caching policy:
  - Sequence-derived features (Block 2.1 + 2.2) are computed once per
    unique sequence and joined to all measurement rows for that variant.
  - Element-derived features (Block 2.3) are computed once per element
    and joined to all measurement rows targeting that element.

Train/test split:
  - 80/20 grouped split by variant_id, stratified by selectivity_cluster.
  - Every row for a given variant receives the same split assignment.
  - selectivity_cluster is preserved as metadata but excluded from
    the feature matrix to prevent target leakage.

CLI smoke test:
    python -m agentic_ai.features.build_matrix
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder

from agentic_ai.features.ef_hand_motifs import compute_ef_hand_features
from agentic_ai.features.encoding import (
    build_feature_dataframe,
    encode_categorical_features,
)
from agentic_ai.features.ree_features import get_ree_features
from agentic_ai.features.sequence_features import compute_basic_features
from agentic_ai.loaders.dataset_assembly import assemble_moesm3_dataframe
from agentic_ai.loaders.xlsx_loader import load_moesm3


# Metadata columns preserved alongside features but never used as input
# to the model. These let us trace predictions back to variants and
# slice results by selectivity cluster during evaluation.
_METADATA_COLUMNS = (
    "measurement_id",
    "variant_id",
    "target_element",
    "selectivity_cluster",
    "split",
    "value",  # the ML target itself, kept here for convenience
)


_DEFAULT_OUTPUT_PATH = Path("data/processed/ml_ready_features.parquet")


def build_feature_matrix(
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Assembles the full ML-ready feature matrix from MOESM3 data.
    @param test_size: Fraction of variants held out for testing (not
                      fraction of rows). 0.2 = 80/20 split.
    @param random_state: Seed for reproducible train/test split.
    return : DataFrame with 9240 rows. Columns: metadata + encoded
             feature columns + split assignment. With the default
             split, there are 128 model features and 6 metadata
             columns.
    """
    final_df, _ = _build_feature_matrix_with_encoder(
        test_size=test_size,
        random_state=random_state,
    )
    return final_df


def _build_feature_matrix_with_encoder(
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, OneHotEncoder]:
    """
    Assembles the ML-ready feature matrix and returns its train-fitted
    categorical encoder for persistence alongside the matrix.
    """
    # Load source data
    corpus = load_moesm3()
    moesm3_df = assemble_moesm3_dataframe(moesm3=corpus)

    # Join selectivity_cluster from the variant records (not in the
    # long-form CSV, but needed for stratification and metadata).
    cluster_lookup = {v.variant_id: v.selectivity_cluster
                      for v in corpus.variants}
    moesm3_df["selectivity_cluster"] = moesm3_df["variant_id"].map(
        cluster_lookup
    )

    # Cache per-sequence features (computed once per unique variant)
    sequence_features_by_id = _compute_sequence_features_cached(corpus)

    # Cache per-element features (computed once per unique element)
    element_features_by_name = _compute_element_features_cached(moesm3_df)

    # Compute train/test split assignment (one assignment per variant)
    split_by_variant = _assign_train_test_split(
        moesm3_df, test_size=test_size, random_state=random_state,
    )

    # Build the per-row raw feature dicts by joining caches
    raw_feature_dicts = _join_features_to_rows(
        moesm3_df,
        sequence_features_by_id,
        element_features_by_name,
    )

    # Convert to DataFrame and fit the categorical encoder on training
    # variants only. Transforming the complete matrix afterward keeps a
    # fixed schema without allowing held-out variants to shape preprocessing.
    raw_features_df = build_feature_dataframe(raw_feature_dicts)
    split_series = moesm3_df["variant_id"].map(split_by_variant)
    if split_series.isna().any():
        missing_variants = sorted(
            moesm3_df.loc[split_series.isna(), "variant_id"].unique()
        )
        raise ValueError(
            "Missing train/test split assignment for variants: "
            f"{missing_variants}"
        )
    _, encoder = encode_categorical_features(
        df=raw_features_df.loc[split_series == "train"],
    )
    encoded_features_df, _ = encode_categorical_features(
        df=raw_features_df,
        fitted_encoder=encoder,
    )

    # Build the final DataFrame: metadata + features + split
    metadata_df = moesm3_df[[
        "measurement_id", "variant_id", "target_element",
        "selectivity_cluster", "value",
    ]].copy()
    metadata_df["split"] = metadata_df["variant_id"].map(split_by_variant)

    # Reset indices to align before concatenating
    metadata_df = metadata_df.reset_index(drop=True)
    encoded_features_df = encoded_features_df.reset_index(drop=True)

    final_df = pd.concat([metadata_df, encoded_features_df], axis=1)

    return final_df, encoder


def save_feature_matrix(
    df: pd.DataFrame = None,
    output_path: Optional[Path] = None,
    fitted_encoder: Optional[OneHotEncoder] = None,
) -> Path:
    """
    Writes the feature matrix to Parquet and its ordered model-feature
    schema to JSON. When supplied, writes the fitted categorical encoder
    to a sibling pickle file for training and inference reuse. Creates
    the parent directory if needed.
    @param df: The assembled feature matrix.
    @param output_path: Destination path. Defaults to
                        data/processed/ml_ready_features.parquet.
    return : The resolved output path.
    """
    if df is None or df.empty:
        raise ValueError("Cannot save empty feature matrix")

    if output_path is None:
        output_path = _DEFAULT_OUTPUT_PATH

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    schema_path = output_path.with_name(
        f"{output_path.stem}_schema.json"
    )
    schema_path.write_text(
        json.dumps(get_feature_columns(df), indent=2) + "\n",
        encoding="utf-8",
    )

    if fitted_encoder is not None:
        encoder_path = output_path.with_name(
            f"{output_path.stem}_encoder.pkl"
        )
        encoder_path.write_bytes(
            pickle.dumps(fitted_encoder)
        )

    return output_path


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Returns the list of column names that are model input features
    (excluding metadata). Used by training scripts to select X from
    the full DataFrame.
    @param df: The feature matrix from build_feature_matrix.
    return : List of column names safe to pass to model.fit(X).
    """
    return [c for c in df.columns if c not in _METADATA_COLUMNS]


def _compute_sequence_features_cached(corpus) -> Dict[str, dict]:
    """
    Computes basic + EF-hand features once per unique sequence, keyed
    by variant_id for later joining.
    @param corpus: The MOESM3 CorpusRecords.
    return : Dict mapping variant_id to a flat feature dict combining
             basic sequence features and raw EF-hand features.
    """
    sequence_features_by_id = {}
    features_by_sequence = {}

    for variant in corpus.variants:
        if not variant.sequence:
            continue

        if variant.sequence not in features_by_sequence:
            basic = compute_basic_features(variant.sequence)
            ef_hand = compute_ef_hand_features(variant.sequence)
            features_by_sequence[variant.sequence] = {**basic, **ef_hand}

        sequence_features_by_id[variant.variant_id] = features_by_sequence[
            variant.sequence
        ]

    return sequence_features_by_id


def _compute_element_features_cached(
    moesm3_df: pd.DataFrame,
) -> Dict[str, dict]:
    """
    Computes REE features once per unique element appearing in the
    dataset, keyed by element name for later joining.
    @param moesm3_df: The MOESM3 long-form DataFrame.
    return : Dict mapping element_name to its 5-feature REE dict.
    """
    unique_elements = moesm3_df["target_element"].unique()
    features_by_name = {
        element: get_ree_features(element)
        for element in unique_elements
    }
    unknown_elements = sorted(
        element for element, features in features_by_name.items()
        if features is None
    )
    if unknown_elements:
        raise ValueError(
            "Missing REE feature lookup entries for elements: "
            f"{unknown_elements}"
        )
    return features_by_name


def _join_features_to_rows(
    moesm3_df: pd.DataFrame,
    sequence_features_by_id: Dict[str, dict],
    element_features_by_name: Dict[str, dict],
) -> List[dict]:
    """
    Builds one raw feature dict per row by joining the cached
    sequence and element features.
    @param moesm3_df: The MOESM3 long-form DataFrame.
    @param sequence_features_by_id: Cache from
                                    _compute_sequence_features_cached.
    @param element_features_by_name: Cache from
                                     _compute_element_features_cached.
    return : List of raw feature dicts, one per row in moesm3_df.
    """
    raw_dicts = []

    for _, row in moesm3_df.iterrows():
        variant_id = row["variant_id"]
        target_element = row["target_element"]
        if variant_id not in sequence_features_by_id:
            raise ValueError(
                f"Missing sequence-derived features for variant: {variant_id}"
            )
        if target_element not in element_features_by_name:
            raise ValueError(
                f"Missing REE-derived features for element: {target_element}"
            )
        sequence_features = sequence_features_by_id[variant_id]
        element_features = element_features_by_name[target_element]
        raw_dicts.append({**sequence_features, **element_features})

    return raw_dicts


def _assign_train_test_split(
    moesm3_df: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> Dict[str, str]:
    """
    Assigns each variant to train or test using a grouped split
    stratified by selectivity_cluster. Returns a dict mapping
    variant_id to 'train' or 'test' so all rows for one variant
    get the same assignment.
    @param moesm3_df: The MOESM3 DataFrame with selectivity_cluster
                      already joined.
    @param test_size: Fraction of variants in the test set.
    @param random_state: Random seed.
    return : Dict mapping variant_id to 'train' or 'test'.
    """
    # Get one row per variant (variant_id -> cluster) for stratification.
    variants_df = (
        moesm3_df[["variant_id", "selectivity_cluster"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    missing_cluster_variants = sorted(
        variants_df.loc[
            variants_df["selectivity_cluster"].isna(), "variant_id"
        ].tolist()
    )
    if missing_cluster_variants:
        raise ValueError(
            "Missing selectivity_cluster for variants: "
            f"{missing_cluster_variants}"
        )

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    train_idx, test_idx = next(splitter.split(
        variants_df,
        variants_df["selectivity_cluster"],
    ))

    train_variants = set(variants_df.iloc[train_idx]["variant_id"])
    test_variants = set(variants_df.iloc[test_idx]["variant_id"])

    return {
        **{v: "train" for v in train_variants},
        **{v: "test" for v in test_variants},
    }


def main() -> int:
    """
    CLI smoke test: build, validate, and save the feature matrix.
    """
    df, encoder = _build_feature_matrix_with_encoder()

    feature_cols = get_feature_columns(df)
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]
    train_variants = set(train_df["variant_id"])
    test_variants = set(test_df["variant_id"])

    print("=== Feature matrix assembled ===")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Feature columns: {len(feature_cols)}")
    print(f"Metadata columns: {df.shape[1] - len(feature_cols)}")
    print()
    print("=== Train/test split ===")
    print(f"Train rows: {len(train_df):,} ({len(train_variants)} variants)")
    print(f"Test rows:  {len(test_df):,} ({len(test_variants)} variants)")
    print(f"Variant overlap: {len(train_variants & test_variants)} "
          f"(must be 0)")
    print()
    print("=== Leakage checks ===")
    print(f"  selectivity_cluster in features: "
          f"{'selectivity_cluster' in feature_cols} (must be False)")
    print(f"  value (target) in features:      "
          f"{'value' in feature_cols} (must be False)")
    print(f"  variant_id in features:          "
          f"{'variant_id' in feature_cols} (must be False)")
    print()
    print("=== Cluster distribution in train/test ===")
    print("Train clusters:")
    print(train_df["selectivity_cluster"].value_counts().sort_index()
          .to_string())
    print("Test clusters:")
    print(test_df["selectivity_cluster"].value_counts().sort_index()
          .to_string())

    output_path = save_feature_matrix(df=df, fitted_encoder=encoder)
    print()
    print("=== Written ===")
    print(f"  {output_path}")
    print(f"  {output_path.with_name(f'{output_path.stem}_schema.json')}")
    print(f"  {output_path.with_name(f'{output_path.stem}_encoder.pkl')}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
