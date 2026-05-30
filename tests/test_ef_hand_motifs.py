"""Tests for EF-hand motif detection (Week 2 Block 2.2a)."""
from __future__ import annotations

import pytest

from agentic_ai.features.ef_hand_motifs import (
    EF_HAND_MOTIF_LENGTH,
    EFHandMotif,
    compute_ef_hand_features,
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


# ---------------------------------------------------------------------------
# Block 2.2b: compute_ef_hand_features
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Output shape and contract
# ---------------------------------------------------------------------------

def test_features_returns_exactly_17_keys():
    """
    Verifies the documented contract: 12 per-motif + 5 aggregate = 17
    features, regardless of how many motifs were actually found.
    """
    features = compute_ef_hand_features(MEX_LANM_SEQUENCE)
    assert len(features) == 17


def test_features_returns_all_documented_keys():
    """
    Verifies that every documented feature name appears in the output.
    Pins the feature schema so accidental renames break the test.
    """
    features = compute_ef_hand_features(MEX_LANM_SEQUENCE)
    expected_per_motif = {
        f"ef{i}_motif_pos{pos}"
        for i in range(1, 5)
        for pos in (0, 2, 5)
    }
    expected_aggregate = {
        "ef_hand_count",
        "ef_hand_mean_spacing",
        "ef_hand_spacing_stdev",
        "ef_hand_span_fraction",
        "has_four_ef_hands",
    }
    assert set(features.keys()) == expected_per_motif | expected_aggregate


def test_features_returns_17_keys_even_when_no_motifs_found():
    """
    Verifies that a sequence with zero EF-hand motifs still returns
    all 17 keys, with per-motif and undefined aggregate features as
    None. The fixed-column contract is what enables consistent
    DataFrame construction in Block 2.5.
    """
    features = compute_ef_hand_features("A" * 50)
    assert len(features) == 17


# ---------------------------------------------------------------------------
# Mex-LanM reference values for per-motif features
# ---------------------------------------------------------------------------

def test_mex_lanm_ef1_motif_residues_match_expected():
    """
    Verifies the residues at the three encoded positions of EF1
    (PDKDGT) — position 0=P, position 2=K, position 5=T.
    """
    features = compute_ef_hand_features(MEX_LANM_SEQUENCE)
    assert features["ef1_motif_pos0"] == "P"
    assert features["ef1_motif_pos2"] == "K"
    assert features["ef1_motif_pos5"] == "T"


def test_mex_lanm_ef3_and_ef4_share_n_at_position_2():
    """
    Verifies the biologically meaningful EF1/EF2 vs EF3/EF4 split:
    the N-terminal EF-hands have K at position 2 while the C-terminal
    EF-hands have N. Pins this dataset-specific signature.
    """
    features = compute_ef_hand_features(MEX_LANM_SEQUENCE)
    assert features["ef1_motif_pos2"] == "K"
    assert features["ef2_motif_pos2"] == "K"
    assert features["ef3_motif_pos2"] == "N"
    assert features["ef4_motif_pos2"] == "N"


def test_mex_lanm_all_motifs_have_p_at_position_0_and_t_at_position_5():
    """
    Verifies the canonical pattern: every Mex-LanM EF-hand starts
    with P and ends with T at the encoded positions.
    """
    features = compute_ef_hand_features(MEX_LANM_SEQUENCE)
    for i in range(1, 5):
        assert features[f"ef{i}_motif_pos0"] == "P"
        assert features[f"ef{i}_motif_pos5"] == "T"


# ---------------------------------------------------------------------------
# Mex-LanM reference values for aggregate features
# ---------------------------------------------------------------------------

def test_mex_lanm_ef_hand_count_is_four():
    features = compute_ef_hand_features(MEX_LANM_SEQUENCE)
    assert features["ef_hand_count"] == 4


def test_mex_lanm_has_four_ef_hands_indicator_is_one():
    features = compute_ef_hand_features(MEX_LANM_SEQUENCE)
    assert features["has_four_ef_hands"] == 1


def test_mex_lanm_mean_spacing_is_in_canonical_range():
    """
    Verifies that mean spacing reflects the HLH bundle geometry
    (motifs ~24-25 residues apart).
    """
    features = compute_ef_hand_features(MEX_LANM_SEQUENCE)
    assert 23.0 <= features["ef_hand_mean_spacing"] <= 26.0


def test_mex_lanm_spacing_stdev_is_low():
    """
    Verifies that Mex-LanM's EF-hand spacing is highly regular
    (very low standard deviation), as expected for a textbook
    Lanmodulin.
    """
    features = compute_ef_hand_features(MEX_LANM_SEQUENCE)
    assert features["ef_hand_spacing_stdev"] < 1.5


def test_mex_lanm_span_fraction_is_in_expected_range():
    """
    Verifies that the EF-hand region occupies ~60-70% of the sequence,
    consistent with Lanmodulin's compact bundle architecture.
    """
    features = compute_ef_hand_features(MEX_LANM_SEQUENCE)
    assert 0.55 <= features["ef_hand_span_fraction"] <= 0.75


# ---------------------------------------------------------------------------
# Missing-motif handling
# ---------------------------------------------------------------------------

def test_sequence_with_fewer_than_4_motifs_pads_with_none():
    """
    Verifies that sequences with only 2 motifs have None for the
    missing EF3 and EF4 positions, preserving the fixed-column
    contract.
    """
    # Synthetic sequence with exactly 2 EF-hand motifs ~20aa apart
    two_motif_seq = "PDKDGT" + "A" * 14 + "PDNDGT" + "A" * 20
    features = compute_ef_hand_features(two_motif_seq)

    # EF1 and EF2 should be populated
    assert features["ef1_motif_pos0"] == "P"
    assert features["ef2_motif_pos0"] == "P"
    # EF3 and EF4 should be None
    assert features["ef3_motif_pos0"] is None
    assert features["ef3_motif_pos2"] is None
    assert features["ef3_motif_pos5"] is None
    assert features["ef4_motif_pos0"] is None
    assert features["ef4_motif_pos2"] is None
    assert features["ef4_motif_pos5"] is None


def test_zero_motif_sequence_has_all_per_motif_features_as_none():
    """
    Verifies that a sequence with no EF-hand motifs has all 12
    per-motif features set to None.
    """
    features = compute_ef_hand_features("A" * 50)

    for i in range(1, 5):
        for pos in (0, 2, 5):
            assert features[f"ef{i}_motif_pos{pos}"] is None


def test_more_than_four_motifs_keeps_only_first_four():
    """
    Verifies that a sequence with 5+ motifs encodes only the first 4
    in the feature dict. The fifth motif is reflected in the count
    aggregate but not the per-motif features.
    """
    # 5 motifs in a row with 10aa spacers
    five_motif_seq = (
        "PDKDGT" + "A" * 10 +
        "PDKDGT" + "A" * 10 +
        "PDNDGT" + "A" * 10 +
        "PDNDGT" + "A" * 10 +
        "PDADGT" + "A" * 10
    )
    features = compute_ef_hand_features(five_motif_seq)

    assert features["ef_hand_count"] == 5
    # All 4 named EF positions populated
    for i in range(1, 5):
        assert features[f"ef{i}_motif_pos0"] is not None
    # has_four_ef_hands should be 0 because count != 4
    assert features["has_four_ef_hands"] == 0


# ---------------------------------------------------------------------------
# Aggregate features when fewer than 2 motifs (undefined spacing)
# ---------------------------------------------------------------------------

def test_zero_motif_sequence_has_none_for_spacing_features():
    """
    Verifies that spacing-derived aggregate features are None (not
    0.0) when fewer than 2 motifs exist. Distinguishes 'no spacing
    to measure' from 'spacing of zero variance'.
    """
    features = compute_ef_hand_features("A" * 50)
    assert features["ef_hand_count"] == 0
    assert features["ef_hand_mean_spacing"] is None
    assert features["ef_hand_spacing_stdev"] is None
    assert features["ef_hand_span_fraction"] is None
    assert features["has_four_ef_hands"] == 0


def test_single_motif_sequence_has_none_for_spacing_features():
    """
    Verifies that a sequence with exactly 1 motif still has undefined
    spacing features. Cannot compute spacing from a single point.
    """
    features = compute_ef_hand_features("A" * 20 + "PDKDGT" + "A" * 20)
    assert features["ef_hand_count"] == 1
    assert features["ef_hand_mean_spacing"] is None
    assert features["ef_hand_spacing_stdev"] is None
    assert features["ef_hand_span_fraction"] is None
    assert features["has_four_ef_hands"] == 0


def test_two_motif_sequence_has_well_defined_spacing_features():
    """
    Verifies that 2 motifs are sufficient to compute mean_spacing,
    spacing_stdev (which equals 0 for a single spacing), and
    span_fraction.
    """
    # 2 motifs 26 chars apart, in a 50-char sequence
    seq = "PDKDGT" + "A" * 20 + "PDNDGT" + "A" * 18
    features = compute_ef_hand_features(seq)

    assert features["ef_hand_count"] == 2
    assert features["ef_hand_mean_spacing"] == 26.0
    # Only one spacing measurement, so variance is 0
    assert features["ef_hand_spacing_stdev"] == 0.0
    assert features["ef_hand_span_fraction"] is not None


# ---------------------------------------------------------------------------
# has_four_ef_hands indicator
# ---------------------------------------------------------------------------

def test_has_four_ef_hands_is_one_only_when_count_is_exactly_four():
    """
    Verifies the indicator is 1 only when exactly 4 motifs are found,
    not 3, not 5. Edge cases must be airtight since this is a
    biologically meaningful flag.
    """
    # 4 motifs -> 1
    assert compute_ef_hand_features(MEX_LANM_SEQUENCE)["has_four_ef_hands"] == 1
    # 0 motifs -> 0
    assert compute_ef_hand_features("A" * 50)["has_four_ef_hands"] == 0
    # 1 motif -> 0
    assert compute_ef_hand_features("PDKDGT" + "A" * 30)["has_four_ef_hands"] == 0


# ---------------------------------------------------------------------------
# Input validation propagation
# ---------------------------------------------------------------------------

def test_features_returns_empty_motif_features_for_empty_input():
    """
    Verifies that empty input gracefully produces a feature dict with
    all per-motif features as None and count=0, rather than crashing.
    """
    features = compute_ef_hand_features("")
    assert features["ef_hand_count"] == 0
    assert features["has_four_ef_hands"] == 0
    assert all(features[f"ef{i}_motif_pos{p}"] is None
               for i in range(1, 5) for p in (0, 2, 5))


def test_features_returns_empty_motif_features_for_none_input():
    """
    Verifies that None input also produces the empty-features dict.
    """
    features = compute_ef_hand_features(None)
    assert features["ef_hand_count"] == 0
    assert features["has_four_ef_hands"] == 0


def test_features_raises_on_non_standard_amino_acids():
    """
    Verifies that input validation propagates from find_ef_hand_motifs.
    """
    contaminated = MEX_LANM_SEQUENCE[:50] + "X" + MEX_LANM_SEQUENCE[51:]
    with pytest.raises(ValueError, match="non-standard"):
        compute_ef_hand_features(contaminated)
