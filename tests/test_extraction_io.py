"""Tests for the extraction persistence and enrichment layer
(Week 1 Block 4.5).

Covers:
  - save_extractions / load_extractions round trip
  - enrich_persisted_extractions: canonicalization, drops,
    measurement propagation, orphan auto-creation, dedup
"""
from __future__ import annotations

from agentic_ai.agents.extraction_models import PaperExtraction
from agentic_ai.loaders.extraction_io import (
    enrich_persisted_extractions,
    load_extractions,
    save_extractions,
)
from agentic_ai.schemas import BindingMeasurement, ProteinVariant


# ---------------------------------------------------------------------------
# Helpers for building minimal valid PaperExtractions in tests
# ---------------------------------------------------------------------------

def _make_variant(
    variant_id: str = "Mex-LanM",
    source_organism: str = "Methylorubrum extorquens",
    source_paper: str = "test_paper",
    **overrides,
) -> ProteinVariant:
    """Builds a minimally-valid ProteinVariant for testing."""
    payload = {
        "variant_id": variant_id,
        "source_organism": source_organism,
        "source_paper": source_paper,
    }
    payload.update(overrides)
    return ProteinVariant(**payload)


def _make_measurement(
    variant_id: str = "Mex-LanM",
    target_element: str = "Neodymium",
    measurement_type: str = "Kd",
    value: float = 1.0e-9,
    units: str = "M",
    source_paper: str = "test_paper",
    **overrides,
) -> BindingMeasurement:
    """Builds a minimally-valid BindingMeasurement for testing."""
    payload = {
        "variant_id": variant_id,
        "target_element": target_element,
        "measurement_type": measurement_type,
        "value": value,
        "units": units,
        "source_paper": source_paper,
    }
    payload.update(overrides)
    return BindingMeasurement(**payload)


def _make_extraction(
    variants: list = None,
    measurements: list = None,
) -> PaperExtraction:
    """
    Builds a PaperExtraction wrapping the given variants/measurements.
    paper_id is the dict key when saved, not a field on the model.
    """
    return PaperExtraction(
        variants=variants or [],
        measurements=measurements or [],
    )


# ---------------------------------------------------------------------------
# save / load round trip
# ---------------------------------------------------------------------------

def test_save_extractions_returns_zero_for_empty_input(tmp_path):
    """Verifies that save with no extractions writes nothing and returns 0."""
    written = save_extractions(extractions={}, output_dir=tmp_path)
    assert written == 0
    assert list(tmp_path.glob("*.json")) == []


def test_save_extractions_returns_zero_for_none_input(tmp_path):
    """Verifies that save with None input is a no-op rather than crashing."""
    written = save_extractions(extractions=None, output_dir=tmp_path)
    assert written == 0


def test_save_extractions_writes_one_file_per_paper(tmp_path):
    """Verifies that one JSON file per paper_id is written."""
    extractions = {
        "paper_a": _make_extraction(),
        "paper_b": _make_extraction(),
        "paper_c": _make_extraction(),
    }
    written = save_extractions(extractions=extractions, output_dir=tmp_path)

    assert written == 3
    files = sorted(p.name for p in tmp_path.glob("*.json"))
    assert files == ["paper_a.json", "paper_b.json", "paper_c.json"]


def test_save_extractions_creates_output_directory_if_missing(tmp_path):
    """Verifies that save creates the output directory automatically."""
    nested = tmp_path / "deeply" / "nested" / "extractions"
    assert not nested.exists()

    save_extractions(
        extractions={"p": _make_extraction()},
        output_dir=nested,
    )

    assert nested.exists()
    assert (nested / "p.json").exists()


def test_load_extractions_returns_empty_dict_when_directory_missing(tmp_path):
    """
    Verifies that load returns an empty dict (not None, not raise) when
    the directory doesn't exist. Downstream code can treat 'no data' as
    a normal state.
    """
    missing = tmp_path / "does_not_exist"
    assert load_extractions(input_dir=missing) == {}


def test_load_extractions_returns_empty_dict_for_empty_directory(tmp_path):
    """Verifies that an empty directory loads as an empty dict."""
    assert load_extractions(input_dir=tmp_path) == {}


def test_save_load_round_trip_preserves_paper_extraction(tmp_path):
    """
    Verifies that a PaperExtraction with variants and measurements
    survives a save/load cycle byte-equal in semantic content.
    """
    original = _make_extraction(
        variants=[_make_variant(variant_id="Mex-LanM")],
        measurements=[_make_measurement(variant_id="Mex-LanM")],
    )
    save_extractions(
        extractions={"round_trip_test": original},
        output_dir=tmp_path,
    )

    loaded = load_extractions(input_dir=tmp_path)

    # The paper_id lives as the dict key (the filename stem), not as a
    # field on the PaperExtraction object itself.
    assert "round_trip_test" in loaded
    reloaded = loaded["round_trip_test"]
    assert len(reloaded.variants) == 1
    assert reloaded.variants[0].variant_id == "Mex-LanM"
    assert reloaded.variants[0].source_organism == "Methylorubrum extorquens"
    assert len(reloaded.measurements) == 1
    assert reloaded.measurements[0].value == 1.0e-9
    assert reloaded.measurements[0].target_element == "Neodymium"


def test_load_extractions_keys_match_filename_stems(tmp_path):
    """Verifies the dict key for each extraction is the filename stem."""
    save_extractions(
        extractions={"my_paper_id_123": _make_extraction()},
        output_dir=tmp_path,
    )

    loaded = load_extractions(input_dir=tmp_path)
    assert list(loaded.keys()) == ["my_paper_id_123"]


# ---------------------------------------------------------------------------
# enrich_persisted_extractions: variant canonicalization
# ---------------------------------------------------------------------------

def test_enrich_canonicalizes_mex_lanm_to_moesm3_id(tmp_path):
    """
    Verifies end-to-end that 'Mex-LanM' becomes 'o-621' on disk after
    enrichment runs.
    """
    save_extractions(
        extractions={"p": _make_extraction(

            variants=[_make_variant(variant_id="Mex-LanM", source_paper="p")],
        )},
        output_dir=tmp_path,
    )

    summary = enrich_persisted_extractions(
        input_dir=tmp_path,
        output_dir=tmp_path,
        dry_run=False,
    )
    loaded = load_extractions(input_dir=tmp_path)

    assert summary["variants_canonicalized"] == 1
    assert loaded["p"].variants[0].variant_id == "o-621"


def test_enrich_populates_construct_type_and_scaffold(tmp_path):
    """
    Verifies that enrichment writes construct_type and parent_scaffold
    onto the variant record on disk.
    """
    save_extractions(
        extractions={"p": _make_extraction(
            variants=[_make_variant(variant_id="LanTERN", source_paper="p")],
        )},
        output_dir=tmp_path,
    )

    enrich_persisted_extractions(
        input_dir=tmp_path, output_dir=tmp_path, dry_run=False,
    )
    loaded = load_extractions(input_dir=tmp_path)

    v = loaded["p"].variants[0]
    assert v.construct_type == "fusion_sensor"
    assert v.parent_scaffold == "Lanmodulin+GFP"


def test_enrich_drops_variants_in_drop_list(tmp_path):
    """
    Verifies that bare 'LanM' (drop-list entry) does not survive
    enrichment.
    """
    save_extractions(
        extractions={"p": _make_extraction(
            variants=[
                _make_variant(variant_id="LanM", source_paper="p"),
                _make_variant(variant_id="Mex-LanM", source_paper="p"),
            ],
        )},
        output_dir=tmp_path,
    )

    summary = enrich_persisted_extractions(
        input_dir=tmp_path, output_dir=tmp_path, dry_run=False,
    )
    loaded = load_extractions(input_dir=tmp_path)

    assert summary["variants_dropped"] == 1
    variant_ids = [v.variant_id for v in loaded["p"].variants]
    assert "LanM" not in variant_ids
    assert "o-621" in variant_ids  # Mex-LanM canonicalized


def test_enrich_dry_run_does_not_write_to_disk(tmp_path):
    """
    Verifies that dry_run=True returns a populated summary without
    modifying the persisted JSON files.
    """
    original_variant = _make_variant(variant_id="Mex-LanM", source_paper="p")
    save_extractions(
        extractions={"p": _make_extraction(
            variants=[original_variant],
        )},
        output_dir=tmp_path,
    )

    summary = enrich_persisted_extractions(
        input_dir=tmp_path, output_dir=tmp_path, dry_run=True,
    )

    assert summary["variants_canonicalized"] == 1
    loaded = load_extractions(input_dir=tmp_path)
    # Disk state unchanged: Mex-LanM still on disk, not o-621.
    assert loaded["p"].variants[0].variant_id == "Mex-LanM"


# ---------------------------------------------------------------------------
# enrich_persisted_extractions: measurement handling
# ---------------------------------------------------------------------------

def test_enrich_propagates_canonical_ids_to_measurements(tmp_path):
    """
    Verifies that when a variant is renamed (Mex-LanM -> o-621), every
    measurement referencing the old name gets updated to the new name.
    """
    save_extractions(
        extractions={"p": _make_extraction(

            variants=[_make_variant(variant_id="Mex-LanM", source_paper="p")],
            measurements=[
                _make_measurement(variant_id="Mex-LanM", source_paper="p"),
                _make_measurement(
                    variant_id="Mex-LanM",
                    target_element="Dysprosium",
                    source_paper="p",
                ),
            ],
        )},
        output_dir=tmp_path,
    )

    enrich_persisted_extractions(
        input_dir=tmp_path, output_dir=tmp_path, dry_run=False,
    )
    loaded = load_extractions(input_dir=tmp_path)

    for m in loaded["p"].measurements:
        assert m.variant_id == "o-621"


def test_enrich_drops_measurements_for_dropped_variants(tmp_path):
    """
    Verifies that measurements whose only parent variant got dropped
    are also removed from the dataset.
    """
    save_extractions(
        extractions={"p": _make_extraction(

            variants=[_make_variant(variant_id="LanM", source_paper="p")],
            measurements=[
                _make_measurement(variant_id="LanM", source_paper="p"),
            ],
        )},
        output_dir=tmp_path,
    )

    summary = enrich_persisted_extractions(
        input_dir=tmp_path, output_dir=tmp_path, dry_run=False,
    )
    loaded = load_extractions(input_dir=tmp_path)

    assert summary["measurements_dropped"] == 1
    assert len(loaded["p"].measurements) == 0


def test_enrich_auto_creates_variant_for_orphan_measurement(tmp_path):
    """
    Verifies the LBT-style case: a measurement references a classifiable
    variant_id but no parent ProteinVariant exists. Enrichment should
    auto-create the missing variant rather than dropping the measurement.
    """
    save_extractions(
        extractions={"p": _make_extraction(

            variants=[],  # No variants at all
            measurements=[
                _make_measurement(variant_id="LBT", source_paper="p"),
            ],
        )},
        output_dir=tmp_path,
    )

    summary = enrich_persisted_extractions(
        input_dir=tmp_path, output_dir=tmp_path, dry_run=False,
    )
    loaded = load_extractions(input_dir=tmp_path)

    assert summary["variants_auto_created"] == 1
    assert summary["measurements_kept"] == 1
    variant_ids = [v.variant_id for v in loaded["p"].variants]
    assert "LBT" in variant_ids


def test_enrich_handles_measurement_variant_id_mismatch_within_paper(tmp_path):
    """
    Verifies the R100K case: variant is listed as 'R100K' but the
    measurement refers to 'Hans-LanM(R100K)' in the same paper. Both
    should normalize to the same canonical form and the measurement
    should attach correctly.
    """
    save_extractions(
        extractions={"p": _make_extraction(

            variants=[_make_variant(variant_id="R100K", source_paper="p")],
            measurements=[_make_measurement(
                variant_id="Hans-LanM(R100K)", source_paper="p",
            )],
        )},
        output_dir=tmp_path,
    )

    enrich_persisted_extractions(
        input_dir=tmp_path, output_dir=tmp_path, dry_run=False,
    )
    loaded = load_extractions(input_dir=tmp_path)

    assert len(loaded["p"].variants) == 1
    assert loaded["p"].variants[0].variant_id == "Hans-LanM(R100K)"
    assert len(loaded["p"].measurements) == 1
    assert loaded["p"].measurements[0].variant_id == "Hans-LanM(R100K)"


# ---------------------------------------------------------------------------
# enrich_persisted_extractions: deduplication
# ---------------------------------------------------------------------------

def test_enrich_merges_duplicate_variants_after_canonicalization(tmp_path):
    """
    Verifies that two variants in the same paper that canonicalize to
    the same ID (e.g. 'Mex-LanM' and 'o-621') get merged into one
    record.
    """
    save_extractions(
        extractions={"p": _make_extraction(

            variants=[
                _make_variant(variant_id="Mex-LanM", source_paper="p"),
                _make_variant(variant_id="o-621", source_paper="p"),
            ],
        )},
        output_dir=tmp_path,
    )

    summary = enrich_persisted_extractions(
        input_dir=tmp_path, output_dir=tmp_path, dry_run=False,
    )
    loaded = load_extractions(input_dir=tmp_path)

    assert summary["variants_merged_duplicates"] == 1
    assert len(loaded["p"].variants) == 1
    assert loaded["p"].variants[0].variant_id == "o-621"


def test_enrich_merges_combine_non_overlapping_fields(tmp_path):
    """
    Verifies that field-merging dedup preserves data: one duplicate has
    a sequence, the other has notes — the merged record has both.
    """
    save_extractions(
        extractions={"p": _make_extraction(

            variants=[
                _make_variant(
                    variant_id="Mex-LanM",
                    sequence="MAEKVKVAVLGAAGGIGQPLSLLLKNS",
                    source_paper="p",
                ),
                _make_variant(
                    variant_id="o-621",
                    notes="Reference wild-type for selectivity comparison",
                    source_paper="p",
                ),
            ],
        )},
        output_dir=tmp_path,
    )

    enrich_persisted_extractions(
        input_dir=tmp_path, output_dir=tmp_path, dry_run=False,
    )
    loaded = load_extractions(input_dir=tmp_path)

    merged = loaded["p"].variants[0]
    assert merged.sequence is not None and len(merged.sequence) >= 20
    assert merged.notes is not None
    assert "selectivity" in merged.notes


def test_enrich_summary_includes_all_expected_keys(tmp_path):
    """
    Verifies that the summary dict returned by enrich_persisted_extractions
    always includes every counter, even when zero.
    """
    summary = enrich_persisted_extractions(
        input_dir=tmp_path,  # empty
        output_dir=tmp_path,
        dry_run=True,
    )

    expected_keys = {
        "papers_processed",
        "variants_kept",
        "variants_dropped",
        "variants_canonicalized",
        "variants_auto_created",
        "measurements_kept",
        "measurements_dropped",
    }
    assert expected_keys.issubset(summary.keys())
