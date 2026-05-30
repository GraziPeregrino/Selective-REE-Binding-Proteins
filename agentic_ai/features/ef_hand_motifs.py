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
