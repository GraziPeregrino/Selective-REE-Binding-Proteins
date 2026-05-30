"""Tests for the Path C two-tier schemas (Week 1 Block 3)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentic_ai.schemas import (
    BindingMeasurement,
    CorpusRecords,
    ProteinVariant,
)

# A realistic Lanmodulin-style fragment using only valid amino acid letters
_VALID_SEQUENCE = "MKKLLFAIPLVVPFYSHSAAQNNDGDGKVGV"


# ---------------------------------------------------------------------------
# ProteinVariant tests
# ---------------------------------------------------------------------------

def _build_valid_variant_payload() -> dict:
    """
    Builds a known-good payload for ProteinVariant.
    return : A dict suitable for `ProteinVariant(**payload)`.
    """
    return {
        "variant_id": "Hans-LanM",
        "source_organism": "Hansschlegelia quercus",
        "sequence": _VALID_SEQUENCE,
        "source_paper": "nature_s41586-023-05945-5.txt",
    }


def test_variant_accepts_minimal_valid_payload():
    """
    Verifies that ProteinVariant constructs with just the required fields
    and sensible defaults for the optional ones.
    """
    variant = ProteinVariant(**_build_valid_variant_payload())

    assert variant.variant_id == "Hans-LanM"
    assert variant.source_organism == "Hansschlegelia quercus"
    assert variant.sequence == _VALID_SEQUENCE
    assert variant.parent_variant_id is None
    assert variant.mutations == []
    assert variant.selectivity_cluster is None


def test_variant_construct_type_defaults_to_ortholog():
    """
    Verifies that ProteinVariant defaults construct_type to 'ortholog'
    when not specified, matching the most common case in the corpus.
    """
    variant = ProteinVariant(**_build_valid_variant_payload())

    assert variant.construct_type == "ortholog"


def test_variant_accepts_known_construct_types():
    """
    Verifies that every documented construct_type value is accepted.
    """
    valid_types = (
        "ortholog", "point_mutant", "fusion_sensor",
        "engineered_chelator", "chimera", "unknown",
    )

    for construct_type in valid_types:
        payload = _build_valid_variant_payload()
        payload["construct_type"] = construct_type
        variant = ProteinVariant(**payload)
        assert variant.construct_type == construct_type


def test_variant_rejects_unknown_construct_type():
    """
    Verifies that construct_type values outside the controlled
    vocabulary are rejected with a clear error.
    """
    payload = _build_valid_variant_payload()
    payload["construct_type"] = "magical_widget"

    with pytest.raises(ValidationError, match="Unknown construct_type"):
        ProteinVariant(**payload)


def test_variant_parent_scaffold_defaults_to_none():
    """
    Verifies that parent_scaffold is None by default rather than
    silently assuming Lanmodulin lineage. Loaders must set it
    explicitly when they know the scaffold.
    """
    variant = ProteinVariant(**_build_valid_variant_payload())

    assert variant.parent_scaffold is None


def test_variant_accepts_known_scaffolds():
    """
    Verifies that documented parent_scaffold values are accepted.
    """
    for scaffold in ("Lanmodulin", "lanpepsy", "Calmodulin"):
        payload = _build_valid_variant_payload()
        payload["parent_scaffold"] = scaffold
        variant = ProteinVariant(**payload)
        assert variant.parent_scaffold == scaffold


def test_variant_accepts_unknown_scaffold_with_warning(capsys):
    """
    Verifies that novel parent_scaffold values are accepted but emit
    a warning on stderr, so the curator notices the addition without
    the validation failing.
    """
    payload = _build_valid_variant_payload()
    payload["parent_scaffold"] = "ExoticScaffoldXYZ"

    variant = ProteinVariant(**payload)
    captured = capsys.readouterr()

    assert variant.parent_scaffold == "ExoticScaffoldXYZ"
    assert "ExoticScaffoldXYZ" in captured.err
    assert "not in the known vocabulary" in captured.err


def test_variant_rejects_empty_parent_scaffold():
    """
    Verifies that an empty parent_scaffold string is rejected (None is
    the way to encode 'not set').
    """
    payload = _build_valid_variant_payload()
    payload["parent_scaffold"] = "   "

    with pytest.raises(ValidationError, match="cannot be an empty string"):
        ProteinVariant(**payload)


def test_variant_accepts_notes_field():
    """
    Verifies that the free-form notes field accepts arbitrary text
    and defaults to None.
    """
    default_variant = ProteinVariant(**_build_valid_variant_payload())
    assert default_variant.notes is None

    payload = _build_valid_variant_payload()
    payload["notes"] = "His10-tagged recombinant construct"
    variant = ProteinVariant(**payload)
    assert variant.notes == "His10-tagged recombinant construct"


def test_variant_accepts_mutant_with_parent_reference():
    """
    Verifies that mutant variants can declare a parent and mutation list.
    """
    payload = _build_valid_variant_payload()
    payload["variant_id"] = "Hans-LanM(R100K)"
    payload["parent_variant_id"] = "Hans-LanM"
    payload["mutations"] = ["R100K"]

    variant = ProteinVariant(**payload)

    assert variant.parent_variant_id == "Hans-LanM"
    assert variant.mutations == ["R100K"]


def test_variant_accepts_none_sequence_for_unreported_papers():
    """
    Verifies that ProteinVariant.sequence can be None when the source
    paper does not report a full continuous sequence (only motifs).
    """
    payload = _build_valid_variant_payload()
    payload["sequence"] = None

    variant = ProteinVariant(**payload)

    assert variant.sequence is None


def test_variant_normalizes_sequence_case_and_whitespace():
    """
    Verifies that sequence input is stripped and uppercased.
    """
    payload = _build_valid_variant_payload()
    payload["sequence"] = f"  {_VALID_SEQUENCE.lower()}  "

    variant = ProteinVariant(**payload)

    assert variant.sequence == _VALID_SEQUENCE


def test_variant_rejects_invalid_amino_acids_in_sequence():
    """
    Verifies that non-standard characters in the sequence trigger a
    ValidationError.
    """
    payload = _build_valid_variant_payload()
    payload["sequence"] = "MKKLLFA1PLVVPFYSHSAAQ"

    with pytest.raises(ValidationError, match="invalid characters"):
        ProteinVariant(**payload)


def test_variant_rejects_missing_source_paper():
    """
    Verifies that source_paper is required for provenance.
    """
    payload = _build_valid_variant_payload()
    del payload["source_paper"]

    with pytest.raises(ValidationError):
        ProteinVariant(**payload)


def test_variant_accepts_cluster_assignment():
    """
    Verifies that the agglomerative_cluster value from Diep et al. 2026
    is accepted within the 0-7 expected range.
    """
    payload = _build_valid_variant_payload()
    payload["selectivity_cluster"] = 4

    variant = ProteinVariant(**payload)

    assert variant.selectivity_cluster == 4


def test_variant_rejects_out_of_range_cluster():
    """
    Verifies that absurd cluster numbers are rejected.
    """
    payload = _build_valid_variant_payload()
    payload["selectivity_cluster"] = 99

    with pytest.raises(ValidationError):
        ProteinVariant(**payload)


# ---------------------------------------------------------------------------
# BindingMeasurement tests
# ---------------------------------------------------------------------------

def _build_valid_measurement_payload() -> dict:
    """
    Builds a known-good payload for BindingMeasurement.
    return : A dict suitable for `BindingMeasurement(**payload)`.
    """
    return {
        "variant_id": "Hans-LanM",
        "target_element": "Neodymium",
        "measurement_type": "Kd_app",
        "value": 9.1e-11,
        "units": "M",
        "value_in_molar": 9.1e-11,
        "conditions_pH": 5.0,
        "source_paper": "nature_s41586-023-05945-5.txt",
    }


def test_measurement_accepts_valid_payload():
    """
    Verifies that BindingMeasurement constructs with a typical Kd record.
    """
    measurement = BindingMeasurement(**_build_valid_measurement_payload())

    assert measurement.variant_id == "Hans-LanM"
    assert measurement.target_element == "Neodymium"
    assert measurement.measurement_type == "Kd_app"
    assert measurement.value_in_molar == 9.1e-11


def test_measurement_normalizes_element_capitalization():
    """
    Verifies that element name input is case-normalized to canonical form.
    """
    payload = _build_valid_measurement_payload()
    payload["target_element"] = "neodymium"

    measurement = BindingMeasurement(**payload)

    assert measurement.target_element == "Neodymium"


def test_measurement_accepts_oxidation_state_notation():
    """
    Verifies that scientific oxidation-state forms are normalized to
    canonical element names. Covers Roman numerals, parens, and
    Unicode superscripts the agent might produce.
    """
    forms = ["Nd", "NdIII", "Nd(III)", "Nd3+", "Nd³⁺", "nd(iii)"]

    for form in forms:
        payload = _build_valid_measurement_payload()
        payload["target_element"] = form
        measurement = BindingMeasurement(**payload)
        assert measurement.target_element == "Neodymium", (
            f"Form {form!r} did not normalize to 'Neodymium'"
        )


def test_measurement_rejects_group_level_labels():
    """
    Verifies that group-level mentions like 'Light REEs' are rejected
    with a clear error explaining the granularity mismatch.
    """
    payload = _build_valid_measurement_payload()
    payload["target_element"] = "Light REEs"

    with pytest.raises(ValidationError, match="group-level label"):
        BindingMeasurement(**payload)


def test_measurement_accepts_actinide_elements():
    """
    Verifies that actinides used in LanM comparison studies (e.g.
    Curium, Americium) are accepted as target elements.
    """
    payload = _build_valid_measurement_payload()
    payload["target_element"] = "Cm(III)"

    measurement = BindingMeasurement(**payload)

    assert measurement.target_element == "Curium"


def test_measurement_rejects_unknown_element():
    """
    Verifies that elements outside the known REE set are rejected.
    """
    payload = _build_valid_measurement_payload()
    payload["target_element"] = "Unobtanium"

    with pytest.raises(ValidationError, match="Unknown target_element"):
        BindingMeasurement(**payload)


def test_measurement_accepts_unitless_logd():
    """
    Verifies that logD measurements with no value_in_molar are accepted.
    """
    payload = _build_valid_measurement_payload()
    payload["measurement_type"] = "logD"
    payload["value"] = 0.45
    payload["units"] = "unitless"
    payload["value_in_molar"] = None

    measurement = BindingMeasurement(**payload)

    assert measurement.measurement_type == "logD"
    assert measurement.value_in_molar is None


def test_measurement_rejects_implausible_value_in_molar():
    """
    Verifies that a value_in_molar above the biological plausibility
    ceiling (1e-2 M) is rejected, guarding against scientific-notation
    extraction failures.
    """
    payload = _build_valid_measurement_payload()
    payload["value_in_molar"] = 2.4  # what happens when LLM drops 'e-12'

    with pytest.raises(ValidationError):
        BindingMeasurement(**payload)


def test_measurement_rejects_unknown_value_source_type():
    """
    Verifies that value_source_type uses only the canonical vocabulary.
    """
    payload = _build_valid_measurement_payload()
    payload["value_source_type"] = "made_up"

    with pytest.raises(ValidationError):
        BindingMeasurement(**payload)


def test_measurement_accepts_cited_from_earlier_work():
    """
    Verifies that secondary-source records are correctly flagged.
    """
    payload = _build_valid_measurement_payload()
    payload["value_source_type"] = "cited_from_earlier_work"

    measurement = BindingMeasurement(**payload)

    assert measurement.value_source_type == "cited_from_earlier_work"


def test_measurement_rejects_out_of_range_pH():
    """
    Verifies that pH values outside 0-14 are rejected.
    """
    payload = _build_valid_measurement_payload()
    payload["conditions_pH"] = 20.0

    with pytest.raises(ValidationError):
        BindingMeasurement(**payload)


# ---------------------------------------------------------------------------
# CorpusRecords tests
# ---------------------------------------------------------------------------

def test_corpus_defaults_to_empty():
    """
    Verifies that CorpusRecords can be instantiated with no records.
    """
    corpus = CorpusRecords()

    assert len(corpus) == 0
    assert corpus.variants == []
    assert corpus.measurements == []


def test_corpus_holds_variants_and_measurements():
    """
    Verifies that CorpusRecords correctly stores both record types and
    reports total length across both tiers.
    """
    variant = ProteinVariant(**_build_valid_variant_payload())
    measurement = BindingMeasurement(**_build_valid_measurement_payload())

    corpus = CorpusRecords(
        variants=[variant, variant],
        measurements=[measurement, measurement, measurement],
    )

    assert len(corpus.variants) == 2
    assert len(corpus.measurements) == 3
    assert len(corpus) == 5
