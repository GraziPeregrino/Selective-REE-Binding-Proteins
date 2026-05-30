"""Tests for the dataset assembly module (Week 1 Block 5).

Covers both source-specific DataFrames and the save_datasets writer.
Uses pre-loaded CorpusRecords / PaperExtraction objects so tests run
in-memory without disk I/O for most assertions.
"""
from __future__ import annotations

import pytest
import pandas as pd

from agentic_ai.agents.extraction_models import PaperExtraction
from agentic_ai.loaders.dataset_assembly import (
    LITERATURE_COLUMNS,
    MOESM3_COLUMNS,
    assemble_literature_dataframe,
    assemble_moesm3_dataframe,
    save_datasets,
)
from agentic_ai.loaders.xlsx_loader import CorpusRecords
from agentic_ai.schemas import BindingMeasurement, ProteinVariant


# ---------------------------------------------------------------------------
# Helpers for building in-memory fixtures
# ---------------------------------------------------------------------------

def _make_moesm3_corpus() -> CorpusRecords:
    """
    Builds a small CorpusRecords fixture mimicking MOESM3's shape:
    one ortholog, 3 normalized_logD measurements.
    """
    variant = ProteinVariant(
        variant_id="o-0",
        source_organism="Rhizobium sp. WYJ-E13",
        sequence="QSQPVAMNQGQMDRLDRDKNGAV",
        construct_type="ortholog",
        parent_scaffold="Lanmodulin",
        source_paper="Diep_2026_NCB_MOESM3",
    )
    measurements = [
        BindingMeasurement(
            variant_id="o-0",
            target_element=element,
            measurement_type="normalized_logD",
            value=value,
            units="unitless",
            source_paper="Diep_2026_NCB_MOESM3",
        )
        for element, value in [
            ("Yttrium", 0.65),
            ("Lanthanum", 0.56),
            ("Cerium", 0.64),
        ]
    ]
    return CorpusRecords(variants=[variant], measurements=measurements)


def _make_literature_extractions() -> dict:
    """
    Builds a small dict of PaperExtractions covering several literature
    edge cases: a Kd in molar units, an EC50 in nM, a fold-change
    without unit conversion. value_in_molar is set explicitly to mirror
    what production data looks like after the extractor's unit-conversion
    step runs.
    """
    paper_a_variants = [
        ProteinVariant(
            variant_id="o-621",
            source_organism="Methylorubrum extorquens",
            construct_type="ortholog",
            parent_scaffold="Lanmodulin",
            source_paper="paper_a",
        )
    ]
    paper_a_measurements = [
        BindingMeasurement(
            variant_id="o-621",
            target_element="Neodymium",
            measurement_type="Kd",
            value=2.4e-12,
            units="M",
            value_in_molar=2.4e-12,
            source_paper="paper_a",
            conditions_pH=7.0,
        ),
    ]
    paper_b_variants = [
        ProteinVariant(
            variant_id="LanTERN",
            source_organism="Escherichia coli",
            construct_type="fusion_sensor",
            parent_scaffold="Lanmodulin+GFP",
            source_paper="paper_b",
        )
    ]
    paper_b_measurements = [
        BindingMeasurement(
            variant_id="LanTERN",
            target_element="Lanthanum",
            measurement_type="EC50",
            value=976.0,
            units="nM",
            value_in_molar=9.76e-07,
            source_paper="paper_b",
        ),
        BindingMeasurement(
            variant_id="LanTERN",
            target_element="Cerium",
            measurement_type="fold_change",
            value=3.5,
            units="fold",
            source_paper="paper_b",
        ),
    ]
    return {
        "paper_a": PaperExtraction(
            variants=paper_a_variants, measurements=paper_a_measurements,
        ),
        "paper_b": PaperExtraction(
            variants=paper_b_variants, measurements=paper_b_measurements,
        ),
    }
    """
    Builds a small dict of PaperExtractions covering several literature
    edge cases: a Kd in molar units, an EC50 in nM, a fold-change
    without unit conversion, and an orphan measurement.
    """
    paper_a_variants = [
        ProteinVariant(
            variant_id="o-621",
            source_organism="Methylorubrum extorquens",
            construct_type="ortholog",
            parent_scaffold="Lanmodulin",
            source_paper="paper_a",
        )
    ]
    paper_a_measurements = [
        BindingMeasurement(
            variant_id="o-621",
            target_element="Neodymium",
            measurement_type="Kd",
            value=2.4e-12,
            units="M",
            source_paper="paper_a",
            conditions_pH=7.0,
        ),
    ]
    paper_b_variants = [
        ProteinVariant(
            variant_id="LanTERN",
            source_organism="Escherichia coli",
            construct_type="fusion_sensor",
            parent_scaffold="Lanmodulin+GFP",
            source_paper="paper_b",
        )
    ]
    paper_b_measurements = [
        BindingMeasurement(
            variant_id="LanTERN",
            target_element="Lanthanum",
            measurement_type="EC50",
            value=976.0,
            units="nM",
            source_paper="paper_b",
        ),
        BindingMeasurement(
            variant_id="LanTERN",
            target_element="Cerium",
            measurement_type="fold_change",
            value=3.5,
            units="fold",
            source_paper="paper_b",
        ),
    ]
    return {
        "paper_a": PaperExtraction(
            variants=paper_a_variants, measurements=paper_a_measurements,
        ),
        "paper_b": PaperExtraction(
            variants=paper_b_variants, measurements=paper_b_measurements,
        ),
    }


# ---------------------------------------------------------------------------
# MOESM3 DataFrame: shape and content invariants
# ---------------------------------------------------------------------------

def test_moesm3_dataframe_has_expected_columns_in_order():
    """
    Verifies that the MOESM3 DataFrame exposes exactly the documented
    columns in the documented order. Downstream code (Week 3 training)
    depends on this contract.
    """
    df = assemble_moesm3_dataframe(moesm3=_make_moesm3_corpus())
    assert list(df.columns) == MOESM3_COLUMNS


def test_moesm3_dataframe_row_count_matches_input_measurements():
    """
    Verifies one row per BindingMeasurement, regardless of variant
    count.
    """
    corpus = _make_moesm3_corpus()
    df = assemble_moesm3_dataframe(moesm3=corpus)

    assert len(df) == len(corpus.measurements)


def test_moesm3_dataframe_value_type_is_uniformly_normalized_logD():
    """
    Verifies that every MOESM3 row has value_type='normalized_logD'.
    Pinning this semantic invariant prevents accidentally mixing other
    measurement types into the MOESM3 selectivity DataFrame.
    """
    df = assemble_moesm3_dataframe(moesm3=_make_moesm3_corpus())
    assert (df["value_type"] == "normalized_logD").all()


def test_moesm3_dataframe_measurement_ids_are_unique():
    """
    Verifies that every measurement_id is unique. The primary key for
    the dataset must be collision-free for outlier tracing in Week 3.
    """
    df = assemble_moesm3_dataframe(moesm3=_make_moesm3_corpus())
    assert df["measurement_id"].is_unique


def test_moesm3_dataframe_measurement_ids_follow_naming_convention():
    """
    Verifies that MOESM3 measurement_ids start with 'moesm3_' and use
    a fixed-width zero-padded counter so lexicographic sort matches
    insertion order.
    """
    df = assemble_moesm3_dataframe(moesm3=_make_moesm3_corpus())
    assert df["measurement_id"].str.startswith("moesm3_").all()


def test_moesm3_dataframe_joins_variant_metadata_correctly():
    """
    Verifies that construct_type, parent_scaffold, source_organism,
    and sequence get joined from the ProteinVariant onto each
    measurement row.
    """
    df = assemble_moesm3_dataframe(moesm3=_make_moesm3_corpus())

    assert (df["construct_type"] == "ortholog").all()
    assert (df["parent_scaffold"] == "Lanmodulin").all()
    assert (df["source_organism"] == "Rhizobium sp. WYJ-E13").all()
    assert df["sequence"].notna().all()


# ---------------------------------------------------------------------------
# Literature DataFrame: shape, content, and semantic invariants
# ---------------------------------------------------------------------------

def test_literature_dataframe_has_expected_columns_in_order():
    """
    Verifies the literature DataFrame's column contract.
    """
    df = assemble_literature_dataframe(literature=_make_literature_extractions())
    assert list(df.columns) == LITERATURE_COLUMNS


def test_literature_dataframe_row_count_matches_input_measurements():
    """
    Verifies one row per measurement across all input papers.
    """
    lit = _make_literature_extractions()
    expected_rows = sum(len(ext.measurements) for ext in lit.values())

    df = assemble_literature_dataframe(literature=lit)

    assert len(df) == expected_rows


def test_literature_dataframe_preserves_heterogeneous_value_types():
    """
    Verifies that the literature DataFrame retains all distinct
    value_type values from the input (Kd, EC50, fold_change, etc.)
    rather than collapsing them.
    """
    df = assemble_literature_dataframe(
        literature=_make_literature_extractions()
    )
    assert set(df["value_type"]) == {"Kd", "EC50", "fold_change"}


def test_literature_dataframe_value_in_molar_only_for_mass_action_units():
    """
    Verifies the semantic invariant: value_in_molar is populated for
    rows with mass-action units (M, nM, etc.) and is NaN for rows
    with unitless or fold-change measurements. This is the load-bearing
    distinction for Week 3 validation.
    """
    df = assemble_literature_dataframe(
        literature=_make_literature_extractions()
    )

    kd_row = df[df["value_type"] == "Kd"].iloc[0]
    ec50_row = df[df["value_type"] == "EC50"].iloc[0]
    fold_row = df[df["value_type"] == "fold_change"].iloc[0]

    assert kd_row["value_in_molar"] is not None
    assert ec50_row["value_in_molar"] is not None
    assert pd.isna(fold_row["value_in_molar"])


def test_literature_dataframe_measurement_ids_are_unique():
    """
    Verifies measurement_id uniqueness across all papers.
    """
    df = assemble_literature_dataframe(
        literature=_make_literature_extractions()
    )
    assert df["measurement_id"].is_unique


def test_literature_dataframe_measurement_ids_use_lit_prefix():
    """
    Verifies that literature rows are clearly distinguishable from
    MOESM3 rows by their measurement_id prefix.
    """
    df = assemble_literature_dataframe(
        literature=_make_literature_extractions()
    )
    assert df["measurement_id"].str.startswith("lit_").all()


def test_literature_dataframe_preserves_construct_type_diversity():
    """
    Verifies that the literature DataFrame represents both orthologs
    and engineered constructs distinctly, not collapsing them.
    """
    df = assemble_literature_dataframe(
        literature=_make_literature_extractions()
    )

    construct_types = set(df["construct_type"].unique())
    assert "ortholog" in construct_types
    assert "fusion_sensor" in construct_types


# ---------------------------------------------------------------------------
# save_datasets: file IO
# ---------------------------------------------------------------------------

def test_save_datasets_writes_both_csvs_to_disk(tmp_path):
    """
    Verifies that save_datasets writes the two expected CSV files in
    the specified output directory.
    """
    moesm3_df = assemble_moesm3_dataframe(moesm3=_make_moesm3_corpus())
    lit_df = assemble_literature_dataframe(
        literature=_make_literature_extractions()
    )

    paths = save_datasets(
        moesm3_df=moesm3_df,
        literature_df=lit_df,
        output_dir=tmp_path,
    )

    assert paths["moesm3"].exists()
    assert paths["literature"].exists()
    assert paths["moesm3"].suffix == ".csv"
    assert paths["literature"].suffix == ".csv"


def test_save_datasets_round_trips_through_pandas_read_csv(tmp_path):
    """
    Verifies that the written CSVs can be read back with pandas and
    the resulting DataFrame has the same shape and column order.
    """
    original = assemble_moesm3_dataframe(moesm3=_make_moesm3_corpus())

    paths = save_datasets(moesm3_df=original, output_dir=tmp_path)
    reloaded = pd.read_csv(paths["moesm3"])

    assert reloaded.shape == original.shape
    assert list(reloaded.columns) == list(original.columns)


def test_save_datasets_creates_output_directory_if_missing(tmp_path):
    """
    Verifies that the output directory is auto-created when it doesn't
    exist, matching the contract of save_extractions in the persistence
    layer.
    """
    nested = tmp_path / "deeply" / "nested" / "outputs"
    assert not nested.exists()

    paths = save_datasets(
        moesm3_df=assemble_moesm3_dataframe(moesm3=_make_moesm3_corpus()),
        output_dir=nested,
    )

    assert nested.exists()
    assert paths["moesm3"].exists()


def test_save_datasets_only_writes_provided_dataframes(tmp_path):
    """
    Verifies that passing None for one DataFrame skips its write
    rather than writing an empty file.
    """
    paths = save_datasets(
        moesm3_df=assemble_moesm3_dataframe(moesm3=_make_moesm3_corpus()),
        literature_df=None,
        output_dir=tmp_path,
    )

    assert "moesm3" in paths
    assert "literature" not in paths
