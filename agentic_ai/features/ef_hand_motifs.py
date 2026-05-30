"""EF-hand motif detection in protein sequences (Week 2 Block 2.2a).

Provides find_ef_hand_motifs() which locates Lanmodulin-family EF-hand
coordination loops within a protein sequence. The detector uses a
permissive regex empirically validated against the 616-sequence MOESM3
corpus (recovers exactly 4 motifs in 67% of orthologs and 3-or-4 in
78%, vs ~3% with the strict canonical motif).

This module only finds motifs and reports their positions. Feature
extraction from those motifs (per-position residues, aggregate
spacing, etc.) lives in Block 2.2b.

The motif length is 6 residues, matching the conserved core of the
canonical 12-residue EF-hand loop (positions 1-6 of the loop, which
include both anchoring Asps and the conserved DG dinucleotide).
"""
from __future__ import annotations

import re
from typing import List, NamedTuple

# Empirically-derived EF-hand seed pattern.
# Positions 1, 3, 6 are variable; positions 2, 4, 5 are conserved (D-D-G).
# Tested against all 616 MOESM3 sequences: recovers 4 motifs in 67% of
# orthologs and 3-or-4 in 78%, far outperforming a strict canonical
# regex which only matched 3% of the dataset.
_EF_HAND_PATTERN = re.compile(r"[A-Z]D[A-Z]DG[A-Z]")

# Conserved core length. EF-hand loops are 12 residues total; the core
# DxDxDG signature spans the first 6 and is what the regex matches.
EF_HAND_MOTIF_LENGTH = 6


class EFHandMotif(NamedTuple):
    """
    One EF-hand motif located in a sequence.
    @param start_index: Zero-based index of the first residue of the
                        motif within the parent sequence.
    @param motif: The 6-residue motif string itself (e.g. 'DPDKDG').
    """
    start_index: int
    motif: str


def find_ef_hand_motifs(sequence: str = None) -> List[EFHandMotif]:
    """
    Locates every EF-hand-like coordination motif in a protein sequence
    using the validated seed pattern.
    @param sequence: A protein amino acid sequence. Tolerates lowercase
                     and outer whitespace; rejects non-standard amino
                     acid characters.
    return : A list of EFHandMotif tuples in order of occurrence.
             Empty list when no motifs are found or when input is
             empty/None.
    raises : ValueError if the sequence contains non-standard amino
             acid characters.
    """
    if sequence is None or not sequence:
        return []

    sequence = sequence.strip().upper()
    _validate_sequence_characters(sequence)

    motifs = []
    for match in _EF_HAND_PATTERN.finditer(sequence):
        motifs.append(EFHandMotif(
            start_index=match.start(),
            motif=match.group(),
        ))

    return motifs


def _validate_sequence_characters(sequence: str) -> None:
    """
    Raises ValueError if the sequence contains characters outside the
    standard 20 amino acid alphabet.
    @param sequence: An uppercase, whitespace-stripped sequence.
    """
    valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
    invalid = set(sequence) - valid_aas

    if invalid:
        raise ValueError(
            f"Sequence contains non-standard characters: "
            f"{sorted(invalid)}. Only the 20 standard amino acids "
            f"(ACDEFGHIKLMNPQRSTVWY) are accepted."
        )


# ---------------------------------------------------------------------------
# Block 2.2b: Per-motif and aggregate features
# ---------------------------------------------------------------------------

# Positions within the 6-residue regex match that vary across the 616
# MOESM3 orthologs. Positions 1, 3, 4 (D, D, G) are essentially invariant
# and provide no ML signal, so we exclude them. Position numbering is
# regex-relative, not classical EF-hand loop numbering:
#
#     P D K D G T
#     0 1 2 3 4 5
#
# Encoded positions: 0, 2, 5
# Classical-literature positions: 1, 3, 6 (each +1 from regex indexing)
_VARIABLE_MOTIF_POSITIONS = (0, 2, 5)

# Lanmodulin's defining architecture is 4 EF-hand motifs. We pin the
# feature schema to exactly 4 motifs, padding with None when a sequence
# has fewer and ignoring extras when a sequence has more.
_CANONICAL_EF_HAND_COUNT = 4


def compute_ef_hand_features(sequence: str = None) -> dict:
    """
    Computes 17 EF-hand-related features from a protein sequence:
      - 12 per-motif categorical features (3 positions x 4 motifs)
      - 4 aggregate architectural features
      - 1 binary canonical-architecture indicator
    Per-motif features are populated for the first 4 motifs found.
    Sequences with fewer than 4 motifs have None for the missing
    motif positions; pandas will convert these to NaN on DataFrame
    construction and tree models handle NaN natively at split time.
    @param sequence: A protein amino acid sequence. Tolerates lowercase
                     and outer whitespace; rejects non-standard amino
                     acid characters.
    return : Dict of 17 features. Categorical features (per-motif
             positions) are single-character strings or None.
             Numerical features (count, spacing, span_fraction) are
             ints or floats, or None when undefined.
    """
    motifs = find_ef_hand_motifs(sequence)

    features = {}

    # Per-motif categorical features. We always emit 12 keys regardless
    # of how many motifs were found, padding with None for missing motifs.
    for ef_index in range(_CANONICAL_EF_HAND_COUNT):
        ef_number = ef_index + 1  # 1-indexed in feature names for clarity
        motif_string = (
            motifs[ef_index].motif if ef_index < len(motifs) else None
        )
        for position in _VARIABLE_MOTIF_POSITIONS:
            feature_name = f"ef{ef_number}_motif_pos{position}"
            features[feature_name] = (
                motif_string[position] if motif_string else None
            )

    # Aggregate architectural features
    features.update(_compute_aggregate_features(motifs, sequence))

    return features


def _compute_aggregate_features(motifs: list, sequence: str = None) -> dict:
    """
    Computes the 5 aggregate features describing overall EF-hand
    architecture for a sequence.
    @param motifs: List of EFHandMotif tuples (output of
                   find_ef_hand_motifs).
    @param sequence: The original sequence; used only for span_fraction.
    return : Dict of 5 aggregate features. Spacing-related features
             are None when fewer than 2 motifs are found.
    """
    count = len(motifs)
    seq_length = len(sequence.strip()) if sequence else 0

    if count < 2:
        return {
            "ef_hand_count":          count,
            "ef_hand_mean_spacing":   None,
            "ef_hand_spacing_stdev":  None,
            "ef_hand_span_fraction":  None,
            "has_four_ef_hands":      int(count == _CANONICAL_EF_HAND_COUNT),
        }

    positions = [m.start_index for m in motifs]
    spacings = [positions[i+1] - positions[i] for i in range(len(positions)-1)]

    mean_spacing = sum(spacings) / len(spacings)

    # Population standard deviation (we have the full set of spacings
    # for this sequence, not a sample of them).
    variance = sum((s - mean_spacing) ** 2 for s in spacings) / len(spacings)
    stdev = variance ** 0.5

    span = positions[-1] - positions[0]
    span_fraction = span / seq_length if seq_length else None

    return {
        "ef_hand_count":          count,
        "ef_hand_mean_spacing":   mean_spacing,
        "ef_hand_spacing_stdev":  stdev,
        "ef_hand_span_fraction":  span_fraction,
        "has_four_ef_hands":      int(count == _CANONICAL_EF_HAND_COUNT),
    }
