"""Basic sequence features computed from amino acid sequences
(Week 2 Block 2.1).

Uses Biopython's ProtParam for established physicochemical properties.
Domain-specific features (EF-hand motif detection, REE coordination
signatures) live in subsequent modules.

The output of compute_basic_features() is a flat dict of 15 numerical
features per sequence — ready to be joined onto the MOESM3 DataFrame
by Block 2.5's assembly step.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from Bio.SeqUtils.ProtParam import ProteinAnalysis

# Hydrophobic residues per the Kyte-Doolittle scale (gravy > 0).
# Used for hydrophobicity_pct feature.
_HYDROPHOBIC_AAS = set("AILMFWVY")

# REE-coordinating and other residues with mechanistic relevance for
# Lanmodulin EF-hand binding chemistry.
_ACIDIC_AAS = set("DE")        # carboxylate side chains, coordinate REE3+
_BASIC_AAS = set("KRH")        # positively charged side chains
_COORDINATING_OH_AAS = set("ST")  # hydroxyl-bearing residues at EF-hand pos 5/9


def compute_basic_features(sequence: str = None) -> Dict[str, float]:
    """
    Computes 15 basic physicochemical features for a protein sequence.
    @param sequence: A protein amino acid sequence. Must contain only
                     standard 20 amino acid characters. Stop codons
                     (*) and non-standard residues (X, U, B, Z) raise
                     a ValueError.
    return : Dict mapping feature name to numerical value.
    raises : ValueError if the sequence is None, empty, or contains
             non-standard characters.
    """
    if sequence is None or not sequence:
        raise ValueError("sequence must be a non-empty string")

    # Biopython's ProteinAnalysis is strict about uppercase characters.
    sequence = sequence.strip().upper()
    _validate_sequence_characters(sequence)

    analyzer = ProteinAnalysis(sequence)
    composition = {aa: pct / 100.0
                   for aa, pct in analyzer.amino_acids_percent.items()}

    return {
        # Bulk properties
        "length":             len(sequence),
        "molecular_weight":   analyzer.molecular_weight(),
        "instability_index":  analyzer.instability_index(),

        # Charge and pH behavior
        "isoelectric_point":  analyzer.isoelectric_point(),
        "charge_at_pH7":      analyzer.charge_at_pH(7.0),
        "aromaticity":        analyzer.aromaticity(),

        # Hydrophobicity
        "gravy":              analyzer.gravy(),
        "hydrophobicity_pct": _fraction_in_set(sequence, _HYDROPHOBIC_AAS),

        # Composition (LanM-informative residues)
        "pct_D":              composition.get("D", 0.0),
        "pct_E":              composition.get("E", 0.0),
        "pct_N":              composition.get("N", 0.0),
        "pct_T":              composition.get("T", 0.0),
        "pct_K":              composition.get("K", 0.0),
        "pct_R":              composition.get("R", 0.0),
        "acidic_basic_ratio": _acidic_basic_ratio(sequence),
    }


def compute_basic_features_batch(
    sequences: List[str] = None,
) -> List[Dict[str, float]]:
    """
    Vectorized batch version of compute_basic_features. Processes
    multiple sequences and returns one dict per input. Sequences that
    raise validation errors yield None in their position rather than
    aborting the batch.
    @param sequences: List of protein sequences. None or empty list
                      returns an empty list.
    return : List of feature dicts (or None for any failed sequence)
             in the same order as the input list.
    """
    if not sequences:
        return []

    results = []
    for seq in sequences:
        try:
            results.append(compute_basic_features(seq))
        except ValueError:
            results.append(None)

    return results


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


def _fraction_in_set(sequence: str, allowed: set) -> float:
    """
    Returns the fraction of residues in the sequence that belong to
    the given set, in [0.0, 1.0].
    @param sequence: An uppercase amino acid sequence.
    @param allowed: A set of single-letter amino acid codes.
    return : float in [0, 1].
    """
    if not sequence:
        return 0.0
    matches = sum(1 for aa in sequence if aa in allowed)
    return matches / len(sequence)


def _acidic_basic_ratio(sequence: str) -> float:
    """
    Returns the ratio of acidic residues (D+E) to basic residues
    (K+R+H). High values are characteristic of LanM-like REE binders.
    Returns the count of acidic residues when there are no basic
    residues (rather than infinity or zero) since this is a more
    informative encoding for ML.
    @param sequence: An uppercase amino acid sequence.
    return : float
    """
    acidic = sum(1 for aa in sequence if aa in _ACIDIC_AAS)
    basic = sum(1 for aa in sequence if aa in _BASIC_AAS)

    if basic == 0:
        return float(acidic)

    return acidic / basic
