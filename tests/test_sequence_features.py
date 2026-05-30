"""Tests for basic sequence feature extraction (Week 2 Block 2.1).

Most assertions use the Mex-LanM (o-621) sequence as a known reference
since it has a well-characterized acidic-LanM profile. Generic input
validation tests use synthetic sequences.
"""
from __future__ import annotations

import math

import pytest

from agentic_ai.features.sequence_features import (
    compute_basic_features,
    compute_basic_features_batch,
)

# Mex-LanM (o-621) — the canonical reference Lanmodulin sequence.
# Source: Diep et al. 2026 MOESM3, o-621 entry.
MEX_LANM_SEQUENCE = (
    "MAPTTTTKVDIAAFDPDKDGTIDLKEALAAGSAAFDKLDPDKDGTLDAKELKGRVSEADL"
    "KKLDPDNDGTLDKKEYLAAVEAQFKAANPDNDGTIDARELASPAGSALVNLIR"
)


# ---------------------------------------------------------------------------
# compute_basic_features — output shape and feature presence
# ---------------------------------------------------------------------------

def test_compute_returns_15_features():
    """
    Verifies that compute_basic_features returns exactly 15 features,
    matching the documented contract for Block 2.1.
    """
    features = compute_basic_features(MEX_LANM_SEQUENCE)
    assert len(features) == 15


def test_compute_returns_all_documented_feature_names():
    """
    Verifies that every documented feature name appears in the output.
    Pinning these names prevents accidental rename regressions.
    """
    features = compute_basic_features(MEX_LANM_SEQUENCE)
    expected_keys = {
        "length", "molecular_weight", "instability_index",
        "isoelectric_point", "charge_at_pH7", "aromaticity",
        "gravy", "hydrophobicity_pct",
        "pct_D", "pct_E", "pct_N", "pct_T", "pct_K", "pct_R",
        "acidic_basic_ratio",
    }
    assert set(features.keys()) == expected_keys


def test_compute_returns_numerical_values_only():
    """
    Verifies that every feature value is an int or float, not None or
    a string. The output is consumed directly by ML model training.
    """
    features = compute_basic_features(MEX_LANM_SEQUENCE)
    for name, value in features.items():
        assert isinstance(value, (int, float)), f"{name} is {type(value)}"


# ---------------------------------------------------------------------------
# compute_basic_features — Mex-LanM reference values
# ---------------------------------------------------------------------------

def test_mex_lanm_length_is_113():
    """
    Verifies the length feature matches the known Mex-LanM sequence.
    """
    features = compute_basic_features(MEX_LANM_SEQUENCE)
    assert features["length"] == 113


def test_mex_lanm_is_strongly_acidic():
    """
    Verifies that Mex-LanM exhibits the canonical Lanmodulin acidic
    profile: pI < 5.0 and negative charge at pH 7.
    """
    features = compute_basic_features(MEX_LANM_SEQUENCE)
    assert features["isoelectric_point"] < 5.0
    assert features["charge_at_pH7"] < 0.0


def test_mex_lanm_has_high_aspartate_content():
    """
    Verifies that Mex-LanM's Asp percentage exceeds 10% — well above
    the typical protein average of ~6%. This is the textbook signature
    of a Lanmodulin ortholog and a major REE-coordination feature.
    """
    features = compute_basic_features(MEX_LANM_SEQUENCE)
    assert features["pct_D"] > 0.10


def test_mex_lanm_has_more_acidic_than_basic_residues():
    """
    Verifies the acidic/basic ratio exceeds 1.0, confirming the
    overall negative charge expected for a REE-binding protein.
    """
    features = compute_basic_features(MEX_LANM_SEQUENCE)
    assert features["acidic_basic_ratio"] > 1.0


def test_mex_lanm_molecular_weight_within_known_range():
    """
    Verifies MW is approximately 12 kDa, consistent with published
    Mex-LanM characterization.
    """
    features = compute_basic_features(MEX_LANM_SEQUENCE)
    assert 11_000 < features["molecular_weight"] < 13_000


# ---------------------------------------------------------------------------
# compute_basic_features — fraction-based features
# ---------------------------------------------------------------------------

def test_composition_features_are_fractions_not_percents():
    """
    Verifies that composition features are in the [0, 1] range, not
    [0, 100]. Catches the silent percent-vs-fraction mismatch we saw
    during smoke testing.
    """
    features = compute_basic_features(MEX_LANM_SEQUENCE)
    composition_features = [
        "pct_D", "pct_E", "pct_N", "pct_T", "pct_K", "pct_R",
        "aromaticity", "hydrophobicity_pct",
    ]
    for name in composition_features:
        assert 0.0 <= features[name] <= 1.0, (
            f"{name}={features[name]} is outside [0, 1]"
        )


def test_composition_fractions_sum_correctly_for_known_sequence():
    """
    Verifies that pct_D + pct_E for Mex-LanM matches the known acidic
    composition. 18 D residues + 6 E residues out of 113 total.
    """
    features = compute_basic_features(MEX_LANM_SEQUENCE)
    expected_acidic = (18 + 6) / 113

    assert math.isclose(
        features["pct_D"] + features["pct_E"],
        expected_acidic,
        abs_tol=1e-6,
    )


# ---------------------------------------------------------------------------
# compute_basic_features — input validation
# ---------------------------------------------------------------------------

def test_compute_raises_on_empty_sequence():
    """
    Verifies that an empty string raises ValueError rather than
    returning a feature dict with zero/None values.
    """
    with pytest.raises(ValueError, match="non-empty"):
        compute_basic_features("")


def test_compute_raises_on_none_input():
    """
    Verifies that None input raises ValueError.
    """
    with pytest.raises(ValueError, match="non-empty"):
        compute_basic_features(None)


def test_compute_raises_on_non_standard_amino_acids():
    """
    Verifies that sequences with selenocysteine (U), pyrrolysine (O),
    or wildcard codes (X, B, Z) are rejected. Mex-LanM contaminated
    with a single X residue should fail clearly.
    """
    contaminated = MEX_LANM_SEQUENCE[:50] + "X" + MEX_LANM_SEQUENCE[51:]
    with pytest.raises(ValueError, match="non-standard"):
        compute_basic_features(contaminated)


def test_compute_raises_on_stop_codon():
    """
    Verifies that sequences containing a stop codon symbol (*) are
    rejected.
    """
    with pytest.raises(ValueError, match="non-standard"):
        compute_basic_features(MEX_LANM_SEQUENCE[:50] + "*")


def test_compute_strips_whitespace_and_handles_lowercase():
    """
    Verifies that real-world copy-paste artifacts (leading/trailing
    whitespace, mixed case) are normalized rather than rejected.
    """
    messy = "  " + MEX_LANM_SEQUENCE.lower() + "\n"
    features = compute_basic_features(messy)

    assert features["length"] == 113


# ---------------------------------------------------------------------------
# compute_basic_features — edge cases
# ---------------------------------------------------------------------------

def test_compute_handles_minimal_valid_sequence():
    """
    Verifies that a single-residue sequence does not crash. Useful
    for downstream robustness even though it has no biological meaning.
    """
    features = compute_basic_features("A")
    assert features["length"] == 1
    assert features["pct_D"] == 0.0


def test_compute_acidic_basic_ratio_handles_zero_basic_residues():
    """
    Verifies that a sequence with no K/R/H residues yields the acidic
    count rather than infinity or crashing on division by zero.
    """
    all_aspartate = "DDDDDDDDDD"
    features = compute_basic_features(all_aspartate)
    assert features["acidic_basic_ratio"] == 10.0


# ---------------------------------------------------------------------------
# compute_basic_features_batch
# ---------------------------------------------------------------------------

def test_batch_returns_one_dict_per_sequence():
    """
    Verifies that the batch function returns a list of the same length
    as its input.
    """
    results = compute_basic_features_batch([
        MEX_LANM_SEQUENCE,
        "DDDDDDDDDD",
        "AAAAAAAAAAAAAA",
    ])
    assert len(results) == 3


def test_batch_returns_empty_list_for_empty_input():
    """
    Verifies that an empty input list does not crash.
    """
    assert compute_basic_features_batch([]) == []
    assert compute_basic_features_batch(None) == []
