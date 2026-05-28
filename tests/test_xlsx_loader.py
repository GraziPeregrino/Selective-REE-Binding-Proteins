"""Tests for the MOESM3 XLSX loader (Week 1 Block 3.2)."""
from __future__ import annotations

from pathlib import Path

import pytest

from agentic_ai.loaders.xlsx_loader import load_moesm3
from agentic_ai.schemas import BindingMeasurement, CorpusRecords, ProteinVariant

# Path to the real MOESM3 file. Tests will skip if it's not on disk so the
# suite still runs in environments where supplementary data hasn't been
# downloaded (e.g. fresh clones, CI).
_MOESM3_PATH = Path(
    "data/raw/supplementary/41589_2026_2176_MOESM3_ESM.xlsx"
)

# Skip marker applied to every test in this file when the data is missing.
pytestmark = pytest.mark.skipif(
    not _MOESM3_PATH.exists(),
    reason=f"MOESM3 not present at {_MOESM3_PATH}; download from Nature paper.",
)


@pytest.fixture(scope="module")
def corpus() -> CorpusRecords:
    """
    Module-scoped fixture: load MOESM3 once and reuse across tests.
    return : The loaded CorpusRecords from MOESM3.
    """
    return load_moesm3()


def test_load_moesm3_yields_expected_variant_count(corpus):
    """
    Verifies that all 616 LanM orthologs in MOESM3 are loaded as variants.
    """
    assert len(corpus.variants) == 616


def test_load_moesm3_yields_expected_measurement_count(corpus):
    """
    Verifies that exactly 616 x 15 = 9240 BindingMeasurement records are
    produced (one per variant per rare earth element).
    """
    assert len(corpus.measurements) == 9240


def test_load_moesm3_all_records_are_correct_types(corpus):
    """
    Verifies that every entry in CorpusRecords is the expected Pydantic
    type, not a dict or raw row.
    """
    assert all(isinstance(v, ProteinVariant) for v in corpus.variants)
    assert all(isinstance(m, BindingMeasurement) for m in corpus.measurements)


def test_load_moesm3_includes_canonical_reference_orthologs(corpus):
    """
    Verifies that the canonical reference orthologs from the literature
    (Mex-LanM, Hans-LanM, Melba-LanM) appear with the correct organisms
    and cluster assignments.
    """
    references = {
        "o-621": ("Methylorubrum extorquens", 0),  # Mex-LanM, cluster C0
        "o-180": ("Hansschlegelia quercus",  4),   # Hans-LanM, cluster C4
        "o-36":  ("Methylobacterium",        5),   # Melba-LanM, cluster C5
    }

    variants_by_id = {v.variant_id: v for v in corpus.variants}

    for ref_id, (expected_organism_prefix, expected_cluster) in references.items():
        assert ref_id in variants_by_id, f"Missing reference variant {ref_id}"
        variant = variants_by_id[ref_id]
        assert variant.source_organism.startswith(expected_organism_prefix), (
            f"{ref_id}: expected organism starting with "
            f"{expected_organism_prefix!r}, got {variant.source_organism!r}"
        )
        assert variant.selectivity_cluster == expected_cluster, (
            f"{ref_id}: expected cluster {expected_cluster}, "
            f"got {variant.selectivity_cluster}"
        )


def test_load_moesm3_every_variant_has_15_measurements(corpus):
    """
    Verifies the 1:15 variant-to-measurement ratio holds: every variant
    has exactly one measurement per rare earth element.
    """
    measurement_counts = {}
    for measurement in corpus.measurements:
        measurement_counts[measurement.variant_id] = (
            measurement_counts.get(measurement.variant_id, 0) + 1
        )

    expected_count = 15
    off_count = {
        variant_id: count
        for variant_id, count in measurement_counts.items()
        if count != expected_count
    }

    assert not off_count, (
        f"Variants with measurement count != {expected_count}: {off_count}"
    )


def test_load_moesm3_measurement_values_in_normalized_range(corpus):
    """
    Verifies that normalized logD values fall within the expected [0, 1]
    range from min-max normalization. Catches any silent unit/column drift.
    """
    out_of_range = [
        m for m in corpus.measurements
        if not (0.0 <= m.value <= 1.0)
    ]

    assert not out_of_range, (
        f"{len(out_of_range)} measurements outside [0, 1]. "
        f"First three: {out_of_range[:3]}"
    )


def test_load_moesm3_all_measurements_have_provenance(corpus):
    """
    Verifies every measurement carries source_paper and measurement_type
    fields populated correctly.
    """
    for measurement in corpus.measurements:
        assert measurement.source_paper == "Diep_2026_NCB_MOESM3"
        assert measurement.measurement_type == "normalized_logD"
        assert measurement.value_source_type == "primary"


def test_load_moesm3_raises_clear_error_on_missing_file(tmp_path):
    """
    Verifies the loader fails with a clear FileNotFoundError that tells
    the user where to obtain the missing file.
    """
    missing_path = tmp_path / "nonexistent.xlsx"

    with pytest.raises(FileNotFoundError, match="Download from"):
        load_moesm3(xlsx_path=missing_path)
