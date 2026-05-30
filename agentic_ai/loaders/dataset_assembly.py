"""Dataset assembly: build two source-specific long-form DataFrames from
MOESM3 and literature extractions (Week 1 Block 5).

The two sources measure fundamentally different quantities:
  - MOESM3: normalized_logD selectivity scores (unitless, 0-1 range)
  - Literature: actual binding constants (Kd, EC50, etc. in molar)

They cannot be merged into a single target column without corrupting
the science. This module produces two separate DataFrames and CSVs.
Week 3 trains its primary model on MOESM3 selectivity; literature data
serves as orthogonal validation.

CLI smoke test:
    python -m agentic_ai.loaders.dataset_assembly
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from agentic_ai.agents.extraction_models import PaperExtraction
from agentic_ai.loaders.extraction_io import load_extractions
from agentic_ai.loaders.xlsx_loader import CorpusRecords, load_moesm3
from agentic_ai.schemas import BindingMeasurement, ProteinVariant


# Column order for the MOESM3 selectivity DataFrame. Every column has
# a single semantic meaning: value is always a normalized_logD score.
MOESM3_COLUMNS = [
    "measurement_id",
    "variant_id",
    "construct_type",
    "parent_scaffold",
    "source_organism",
    "sequence",
    "target_element",
    "value",
    "value_units",
    "value_type",
    "source_paper",
]


# Column order for the literature binding DataFrame. value can be Kd,
# Kd_app, EC50, etc., reported in arbitrary units. value_in_molar holds
# the normalized form when units are mass-action (M, nM, mM, etc.).
LITERATURE_COLUMNS = [
    "measurement_id",
    "variant_id",
    "construct_type",
    "parent_scaffold",
    "source_organism",
    "sequence",
    "target_element",
    "value",
    "value_units",
    "value_type",
    "value_in_molar",
    "conditions_pH",
    "source_paper",
]


# Default CSV output paths. Both live under data/processed/ alongside
# the persisted JSON extractions.
_DEFAULT_OUTPUT_DIR = Path("data/processed")
_MOESM3_CSV_NAME = "moesm3_selectivity_data.csv"
_LITERATURE_CSV_NAME = "literature_binding_data.csv"


def assemble_moesm3_dataframe(
    moesm3: Optional[CorpusRecords] = None,
) -> pd.DataFrame:
    """
    Builds the MOESM3 selectivity DataFrame. One row per (variant, REE)
    measurement. value is a unitless normalized_logD score.
    @param moesm3: Pre-loaded MOESM3 records. If None, loads from disk.
    return : A pandas DataFrame with the columns listed in
             MOESM3_COLUMNS, sorted by (variant_id, target_element).
    """
    if moesm3 is None:
        moesm3 = load_moesm3()

    rows = _build_moesm3_rows(moesm3)

    df = pd.DataFrame(rows, columns=MOESM3_COLUMNS)
    df = df.sort_values(
        by=["variant_id", "target_element"],
        kind="stable",
    ).reset_index(drop=True)

    return df


def assemble_literature_dataframe(
    literature: Optional[Dict[str, PaperExtraction]] = None,
) -> pd.DataFrame:
    """
    Builds the literature binding DataFrame. One row per measurement
    extracted from the curated literature corpus, joined to its parent
    ProteinVariant for sequence and construct metadata.
    @param literature: Pre-loaded extractions keyed by paper_id. If
                       None, loads from data/processed/extractions/.
    return : A pandas DataFrame with the columns listed in
             LITERATURE_COLUMNS, sorted by (source_paper, variant_id,
             target_element).
    """
    if literature is None:
        literature = load_extractions()

    rows = _build_literature_rows(literature)

    df = pd.DataFrame(rows, columns=LITERATURE_COLUMNS)
    df = df.sort_values(
        by=["source_paper", "variant_id", "target_element"],
        kind="stable",
    ).reset_index(drop=True)

    return df


def save_datasets(
    moesm3_df: pd.DataFrame = None,
    literature_df: pd.DataFrame = None,
    output_dir: Path = None,
) -> Dict[str, Path]:
    """
    Writes both DataFrames to CSV files in the output directory.
    Creates the directory if it does not exist.
    @param moesm3_df: The MOESM3 selectivity DataFrame.
    @param literature_df: The literature binding DataFrame.
    @param output_dir: Directory to write CSVs to. Defaults to
                       data/processed/.
    return : Dict mapping 'moesm3' and 'literature' to the resolved
             output paths.
    """
    if output_dir is None:
        output_dir = _DEFAULT_OUTPUT_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    if moesm3_df is not None:
        moesm3_path = output_dir / _MOESM3_CSV_NAME
        moesm3_df.to_csv(moesm3_path, index=False)
        paths["moesm3"] = moesm3_path

    if literature_df is not None:
        lit_path = output_dir / _LITERATURE_CSV_NAME
        literature_df.to_csv(lit_path, index=False)
        paths["literature"] = lit_path

    return paths


def _build_moesm3_rows(corpus: CorpusRecords) -> List[dict]:
    """
    Builds one row per MOESM3 BindingMeasurement, joining each
    measurement to its parent ProteinVariant by variant_id.
    @param corpus: The loaded MOESM3 CorpusRecords.
    return : List of dicts ready for DataFrame construction.
    """
    variants_by_id = {v.variant_id: v for v in corpus.variants}
    rows = []

    for i, measurement in enumerate(corpus.measurements, start=1):
        variant = variants_by_id.get(measurement.variant_id)
        rows.append({
            "measurement_id":   f"moesm3_{i:05d}",
            "variant_id":       measurement.variant_id,
            "construct_type":   variant.construct_type if variant else None,
            "parent_scaffold":  variant.parent_scaffold if variant else None,
            "source_organism":  variant.source_organism if variant else None,
            "sequence":         variant.sequence if variant else None,
            "target_element":   measurement.target_element,
            "value":            measurement.value,
            "value_units":      measurement.units,
            "value_type":       measurement.measurement_type,
            "source_paper":     measurement.source_paper,
        })

    return rows


def _build_literature_rows(
    literature: Dict[str, PaperExtraction],
) -> List[dict]:
    """
    Builds one row per literature BindingMeasurement, joining each
    measurement to its parent ProteinVariant within the same paper.
    @param literature: Dict mapping paper_id to PaperExtraction.
    return : List of dicts ready for DataFrame construction.
    """
    rows = []
    counter = 1

    for paper_id in sorted(literature.keys()):
        extraction = literature[paper_id]
        variants_by_id = {v.variant_id: v for v in extraction.variants}

        for measurement in extraction.measurements:
            variant = variants_by_id.get(measurement.variant_id)
            rows.append({
                "measurement_id":   f"lit_{counter:04d}",
                "variant_id":       measurement.variant_id,
                "construct_type":   variant.construct_type if variant else None,
                "parent_scaffold":  variant.parent_scaffold if variant else None,
                "source_organism":  variant.source_organism if variant else None,
                "sequence":         variant.sequence if variant else None,
                "target_element":   measurement.target_element,
                "value":            measurement.value,
                "value_units":      measurement.units,
                "value_type":       measurement.measurement_type,
                "value_in_molar":   measurement.value_in_molar,
                "conditions_pH":    measurement.conditions_pH,
                "source_paper":     measurement.source_paper,
            })
            counter += 1

    return rows


def main() -> int:
    """
    CLI smoke test: assemble both DataFrames, print summary statistics,
    and write CSV files to data/processed/.
    return : Shell exit code (0 on success).
    """
    moesm3_df = assemble_moesm3_dataframe()
    literature_df = assemble_literature_dataframe()

    print(f"=== MOESM3 selectivity DataFrame ===")
    print(f"Shape: {moesm3_df.shape[0]:,} rows x {moesm3_df.shape[1]} columns")
    print(f"Unique variants:  {moesm3_df['variant_id'].nunique():,}")
    print(f"Unique elements:  {moesm3_df['target_element'].nunique()}")
    print(f"value_type values: {moesm3_df['value_type'].unique().tolist()}")
    print(f"value range:      [{moesm3_df['value'].min():.3f}, "
          f"{moesm3_df['value'].max():.3f}]")

    print()
    print(f"=== Literature binding DataFrame ===")
    print(f"Shape: {literature_df.shape[0]:,} rows x "
          f"{literature_df.shape[1]} columns")
    print(f"Unique variants:  {literature_df['variant_id'].nunique()}")
    print(f"Unique elements:  {literature_df['target_element'].nunique()}")
    print(f"value_type top 5:")
    print(literature_df["value_type"].value_counts().head(5).to_string())
    print()
    print(f"value_in_molar coverage: "
          f"{literature_df['value_in_molar'].notna().sum()} of "
          f"{len(literature_df)} rows "
          f"({literature_df['value_in_molar'].notna().mean()*100:.1f}%)")

    print()
    print(f"=== construct_type distribution (literature) ===")
    print(literature_df["construct_type"].value_counts(dropna=False).to_string())

    print()
    print(f"=== parent_scaffold distribution (literature) ===")
    print(literature_df["parent_scaffold"].value_counts(dropna=False).to_string())

    paths = save_datasets(moesm3_df=moesm3_df, literature_df=literature_df)
    print()
    print(f"=== Written CSVs ===")
    for source, path in paths.items():
        print(f"  {source:<12} {path}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
