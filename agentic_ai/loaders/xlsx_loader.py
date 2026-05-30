"""XLSX loader for the Diep et al. 2026 master dataset (MOESM3).

The MOESM3 supplementary spreadsheet contains 616 LanM ortholog records with
full sequences, organism metadata, taxonomy, raw ICP-MS measurements (15
REEs x 3 replicates), and 8-cluster selectivity assignments. This module
reads that file and produces validated CorpusRecords ready for downstream
feature engineering.

Run as a script for a quick smoke test:
    python -m agentic_ai.loaders.xlsx_loader
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from agentic_ai.schemas import (
    BindingMeasurement,
    CorpusRecords,
    ProteinVariant,
)

# Map the REE column prefixes used in MOESM3 to canonical element names.
_REE_PREFIX_TO_ELEMENT = {
    "Y":  "Yttrium",
    "La": "Lanthanum",
    "Ce": "Cerium",
    "Pr": "Praseodymium",
    "Nd": "Neodymium",
    "Sm": "Samarium",
    "Eu": "Europium",
    "Gd": "Gadolinium",
    "Tb": "Terbium",
    "Dy": "Dysprosium",
    "Ho": "Holmium",
    "Er": "Erbium",
    "Tm": "Thulium",
    "Yb": "Ytterbium",
    "Lu": "Lutetium",
}

# Rename map applied to MOESM3 columns so they become valid Python identifiers.
# Keys = original messy column names, values = clean attribute names.
_COLUMN_RENAMES = {
    "ID-4": "id_4",
    "ID-6": "id_6",
    "Sequence..SP.Removed.": "sequence_sp_removed",
}

# Suffix used by MOESM3 for the normalized logD mean columns (e.g. La.9).
# These are renamed to "<prefix>_norm_logd" (e.g. "La_norm_logd") for access.
_NORMALIZED_LOGD_SUFFIX = ""

# Source citation for every record produced by this loader.
_SOURCE_PAPER = "Diep_2026_NCB_MOESM3"

# Default location of the MOESM3 file inside the project.
_DEFAULT_XLSX_PATH = Path(
    "data/raw/supplementary/41589_2026_2176_MOESM3_ESM.xlsx"
)


def load_moesm3(
    xlsx_path: Path = None,
    sheet_name: str = "Data",
) -> CorpusRecords:
    """
    Loads the Diep et al. 2026 MOESM3 supplementary spreadsheet into
    validated ProteinVariant and BindingMeasurement records.
    @param xlsx_path: Path to the MOESM3 xlsx file. Defaults to the
                      canonical project location under data/raw/supplementary/.
    @param sheet_name: Excel sheet to read. MOESM3 has one data sheet named
                       'Data' alongside a 'ReadMe' sheet.
    return : A CorpusRecords object containing 616 variants and ~9240
             measurements (one per variant x element pair).
    raises : FileNotFoundError if the xlsx file does not exist.
    raises : ValueError if the file does not contain the expected columns.
    """
    if xlsx_path is None:
        xlsx_path = _DEFAULT_XLSX_PATH

    if not xlsx_path.exists():
        raise FileNotFoundError(
            f"MOESM3 not found at {xlsx_path}. "
            f"Download from "
            f"https://www.nature.com/articles/s41589-026-02176-3 "
            f"and place in data/raw/supplementary/."
        )

    dataframe = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    dataframe = _rename_columns_for_safe_access(dataframe)
    _validate_expected_columns(dataframe)

    variants: List[ProteinVariant] = []
    measurements: List[BindingMeasurement] = []

    for _, row in dataframe.iterrows():
        variant = _row_to_variant(row)
        if variant is None:
            continue

        variants.append(variant)
        measurements.extend(_row_to_measurements(row, variant.variant_id))

    return CorpusRecords(variants=variants, measurements=measurements)


def _rename_columns_for_safe_access(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Renames MOESM3 columns to Python-identifier-safe names so they can be
    accessed by dict-style lookup without hyphens, dots, or other surprises.
    @param dataframe: The raw DataFrame as loaded from MOESM3.
    return : A new DataFrame with renamed columns.
    """
    renames = dict(_COLUMN_RENAMES)

    # Rename the normalized logD columns: "La.9" -> "La_norm_logd", etc.
    for prefix in _REE_PREFIX_TO_ELEMENT:
        if prefix in dataframe.columns:
            renames[prefix] = f"{prefix}_norm_logd"
        sd_col = f"{prefix}.1"
        if sd_col in dataframe.columns:
            renames[sd_col] = f"{prefix}_norm_logd_sd"

    return dataframe.rename(columns=renames)


def _validate_expected_columns(dataframe: pd.DataFrame) -> None:
    """
    Confirms that the spreadsheet has the columns this loader depends on
    after renaming. Fails fast with a clear message if the schema has drifted.
    @param dataframe: The renamed DataFrame to inspect.
    raises : ValueError listing the missing column names.
    """
    required = {
        "id_6", "sequence_sp_removed", "source_original",
        "EFhands", "agglomerative_cluster",
    }
    required.update(
        f"{prefix}_norm_logd"
        for prefix in _REE_PREFIX_TO_ELEMENT
    )

    missing = required - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"MOESM3 is missing expected columns after rename: "
            f"{sorted(missing)}."
        )


def _row_to_variant(row: pd.Series) -> Optional[ProteinVariant]:
    """
    Converts one MOESM3 row to a ProteinVariant. Skips rows with missing
    sequence or organism since those are unusable downstream.
    @param row: A pandas Series representing one row.
    return : A ProteinVariant instance, or None if the row is unusable.
    """
    sequence = _normalize_cell(row.get("sequence_sp_removed"))
    organism = _normalize_cell(row.get("source_original"))
    ortholog_index = _safe_int(row.get("id_6"))

    if not sequence or not organism or ortholog_index is None:
        return None

    cluster = _safe_int(row.get("agglomerative_cluster"))
    ef_hands = _safe_int(row.get("EFhands"))
    taxonomy = _build_taxonomy_string(row)

    return ProteinVariant(
        variant_id=f"o-{ortholog_index}",
        source_organism=organism,
        sequence=sequence,
        taxonomy=taxonomy,
        ef_hand_count=ef_hands,
        selectivity_cluster=cluster,
        construct_type="ortholog",
        parent_scaffold="Lanmodulin",
        source_paper=_SOURCE_PAPER,
    )


def _row_to_measurements(
    row: pd.Series,
    variant_id: str,
) -> Iterable[BindingMeasurement]:
    """
    Produces one BindingMeasurement per (variant, element) pair from the
    normalized logD columns of MOESM3.
    @param row: A pandas Series representing one row.
    @param variant_id: The variant_id the measurements belong to.
    return : Generator yielding BindingMeasurement instances. Skips
             elements with missing or NaN values.
    """
    for prefix, element in _REE_PREFIX_TO_ELEMENT.items():
        raw_value = row.get(f"{prefix}_norm_logd")

        if raw_value is None or pd.isna(raw_value):
            continue

        yield BindingMeasurement(
            variant_id=variant_id,
            target_element=element,
            measurement_type="normalized_logD",
            value=float(raw_value),
            units="unitless",
            value_in_molar=None,
            conditions_pH=3.0,
            conditions_notes=(
                "SpyCI-LAMBS, normalized logD, mean of 3 replicates"
            ),
            value_source_type="primary",
            source_paper=_SOURCE_PAPER,
        )


def _normalize_cell(value: object) -> Optional[str]:
    """
    Coerces a pandas cell value to a clean string or returns None.
    @param value: Raw cell value from pandas (str, float, NaT, or NaN).
    return : Stripped string, or None when the cell is empty/NaN.
    """
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()
    return text if text else None


def _safe_int(value: object) -> Optional[int]:
    """
    Coerces a pandas cell value to an int or returns None on failure.
    @param value: Raw cell value.
    return : Integer value, or None if conversion fails or value is missing.
    """
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _build_taxonomy_string(row: pd.Series) -> Optional[str]:
    """
    Builds a compact taxonomy string from the available phylogeny columns.
    @param row: A pandas Series representing one row.
    return : Pipe-separated taxonomy string, or None if all fields are empty.
    """
    fields = [
        ("phylum",  _normalize_cell(row.get("phylum_original"))),
        ("class",   _normalize_cell(row.get("class_original"))),
        ("order",   _normalize_cell(row.get("order_original"))),
        ("family",  _normalize_cell(row.get("family_original"))),
        ("genus",   _normalize_cell(row.get("genus_original"))),
    ]
    parts = [f"{label}={value}" for label, value in fields if value]
    return " | ".join(parts) if parts else None


def main() -> int:
    """
    Loads MOESM3 and prints a summary. Used as a smoke test.
    return : Shell exit code (0 on success, 1 on file-not-found).
    """
    try:
        corpus = load_moesm3()
    except FileNotFoundError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print(
        f"Loaded MOESM3: "
        f"{len(corpus.variants)} variants, "
        f"{len(corpus.measurements)} measurements"
    )

    references = {
        "o-36": "Melba-LanM",
        "o-180": "Hans-LanM",
        "o-621": "Mex-LanM",
    }
    for ref_id, name in references.items():
        match = [v for v in corpus.variants if v.variant_id == ref_id]
        if match:
            variant = match[0]
            seq_len = len(variant.sequence) if variant.sequence else "n/a"
            print(
                f"  {ref_id} ({name}): {variant.source_organism}, "
                f"{seq_len} residues, cluster={variant.selectivity_cluster}"
            )

    cluster_counts: dict = {}
    for variant in corpus.variants:
        cluster_counts[variant.selectivity_cluster] = (
            cluster_counts.get(variant.selectivity_cluster, 0) + 1
        )
    sorted_clusters = sorted(
        cluster_counts.items(),
        key=lambda pair: (pair[0] is None, pair[0]),
    )
    print(f"  Cluster distribution: {dict(sorted_clusters)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
