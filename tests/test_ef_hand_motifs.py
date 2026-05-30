"""Tests for EF-hand motif detection (Week 2 Block 2.2a)."""
from __future__ import annotations

import pytest

from agentic_ai.features.ef_hand_motifs import (
    EF_HAND_MOTIF_LENGTH,
    EFHandMotif,
    find_ef_hand_motifs,
)

# Canonical Mex-LanM (o-621) sequence with 4 known EF-hand motifs at
# positions 15, 39, 64, 88. The detector should find all four.
MEX_LANM_SEQUENCE = (
    "MAPTTTTKVDIAAFDPDKDGTIDLKEALAAGSAAFDKLDPDKDGTLDAKELKGRVSEADL"
    "KKLDPDNDGTLDKKEYLAAVEAQFKAANPDNDGTIDARELASPAGSALVNLIR"
)


# ---------------------------------------------------------------------------
# Mex-LanM ground-truth tests
# ---------------------------------------------------------------------------

def test_mex_lanm_yields_exactly_4_motifs():
    """
    Verifies the canonical Lanmodulin architecture: 4 EF-hands. This
    is the gold-standard test for the detector since Mex-LanM is the
    most-characterized member of the family.
    """
    motifs = find_ef_hand_motifs(MEX_LANM_SEQUENCE)
    assert len(motifs) == 4


def test_mex_lanm_motifs_are_at_expected_positions():
    """
    Verifies that the 4 motifs land at the expected positions 15, 39,
    64, 88. These positions correspond to the start of each EF-hand
    coordination loop's conserved DxDG core. Spacing of ~24-25 residues
    between motifs reflects the HLH bundle architecture.
    """
    motifs = find_ef_hand_motifs(MEX_LANM_SEQUENCE)
    positions = [m.start_index for m in motifs]
    assert positions == [15, 39, 64, 88]


def test_mex_lanm_motifs_are_canonical_lanmodulin_sequences():
    """
    Verifies that the matched 6-residue motifs correspond to the
    expected LanM EF-hand coordination loop strings.
    """
    motifs = find_ef_hand_motifs(MEX_LANM_SEQUENCE)
    assert [m.motif for m in motifs] == [
        "PDKDGT", "PDKDGT", "PDNDGT", "PDNDGT",
    ]


def test_mex_lanm_motifs_are_correctly_spaced():
    """
    Verifies that consecutive motifs are spaced approximately 24-25
    residues apart, matching the canonical HLH bundle geometry.
    """
    motifs = find_ef_hand_motifs(MEX_LANM_SEQUENCE)
    positions = [m.start_index for m in motifs]
    intervals = [positions[i+1] - positions[i] for i in range(len(positions)-1)]

    for interval in intervals:
        assert 20 <= interval <= 30


# ---------------------------------------------------------------------------
# Output structure tests
# ---------------------------------------------------------------------------

def test_motifs_are_returned_as_ef_hand_motif_named_tuples():
    """
    Verifies that the output is a list of EFHandMotif named tuples
    with start_index and motif fields, not raw tuples or dicts.
    """
    motifs = find_ef_hand_motifs(MEX_LANM_SEQUENCE)
    for m in motifs:
        assert isinstance(m, EFHandMotif)
        assert hasattr(m, "start_index")
        assert hasattr(m, "motif")


def test_motif_strings_are_exactly_six_residues_long():
    """
    Verifies that every matched motif is exactly EF_HAND_MOTIF_LENGTH
    (6) residues. Pins the regex's match length contract.
    """
    motifs = find_ef_hand_motifs(MEX_LANM_SEQUENCE)
    for m in motifs:
        assert len(m.motif) == EF_HAND_MOTIF_LENGTH


def test_motifs_returned_in_order_of_occurrence():
    """
    Verifies that motifs appear in the order they occur in the
    sequence. Downstream feature naming (ef1_*, ef2_*, etc.) depends
    on this ordering.
    """
    motifs = find_ef_hand_motifs(MEX_LANM_SEQUENCE)
    positions = [m.start_index for m in motifs]
    assert positions == sorted(positions)


# ---------------------------------------------------------------------------
# Edge cases and input validation
# ---------------------------------------------------------------------------

def test_empty_sequence_returns_empty_list():
    """
    Verifies that an empty string returns an empty list rather than
    raising an exception.
    """
    assert find_ef_hand_motifs("") == []


def test_none_sequence_returns_empty_list():
    """
    Verifies that None input returns an empty list. Downstream code
    can rely on a list always being returned.
    """
    assert find_ef_hand_motifs(None) == []


def test_sequence_with_no_ef_hands_returns_empty_list():
    """
    Verifies that a real but non-EF-hand sequence (poly-alanine)
    produces zero motifs rather than spurious matches.
    """
    assert find_ef_hand_motifs("A" * 50) == []


def test_lowercase_sequence_is_handled():
    """
    Verifies that lowercase input is normalized and motifs are still
    found. Real-world sequences sometimes arrive lowercase from
    parsing tools.
    """
    motifs = find_ef_hand_motifs(MEX_LANM_SEQUENCE.lower())
    assert len(motifs) == 4


def test_outer_whitespace_is_stripped():
    """
    Verifies tolerance for whitespace artifacts from copy-paste or
    file reading.
    """
    motifs = find_ef_hand_motifs("  " + MEX_LANM_SEQUENCE + "\n")
    assert len(motifs) == 4


def test_non_standard_characters_are_rejected():
    """
    Verifies that sequences with non-standard amino acid codes (X, B,
    Z, U, O, *) raise ValueError rather than silently returning
    partial or wrong results.
    """
    contaminated = MEX_LANM_SEQUENCE[:50] + "X" + MEX_LANM_SEQUENCE[51:]
    with pytest.raises(ValueError, match="non-standard"):
        find_ef_hand_motifs(contaminated)


# ---------------------------------------------------------------------------
# Detector robustness against synthetic edge cases
# ---------------------------------------------------------------------------

def test_minimum_viable_motif_is_detected():
    """
    Verifies that the shortest possible sequence containing exactly
    one EF-hand motif yields a single hit at position 0.
    """
    motifs = find_ef_hand_motifs("PDKDGT")
    assert len(motifs) == 1
    assert motifs[0] == EFHandMotif(start_index=0, motif="PDKDGT")


def test_overlapping_motifs_are_not_double_counted():
    """
    Verifies that the detector does not produce overlapping matches.
    re.finditer is non-overlapping by default; this test pins that
    behavior since overlapping EF-hands would corrupt downstream
    counting and spacing features.
    """
    # Construct a sequence where two motifs would overlap if the
    # detector allowed it.
    overlapping = "PDKDGTDKDGT"  # 11 chars, two potential overlapping matches
    motifs = find_ef_hand_motifs(overlapping)

    # Verify the second match starts at least 6 positions after the
    # first (no overlap).
    if len(motifs) >= 2:
        assert motifs[1].start_index >= motifs[0].start_index + EF_HAND_MOTIF_LENGTH


def test_motif_at_end_of_sequence_is_detected():
    """
    Verifies that a motif occurring at the very end of a sequence is
    still found.
    """
    sequence = "AAAAAA" + "PDKDGT"
    motifs = find_ef_hand_motifs(sequence)
    assert len(motifs) == 1
    assert motifs[0].start_index == 6
