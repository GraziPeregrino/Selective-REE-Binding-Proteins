"""Sequence validation for inference inputs (Week 4 Block 4.1).

Accepts a protein sequence as a string, normalizes whitespace and
casing, and validates that the result contains only canonical amino
acids within the supported app input range.

The validator does NOT enforce that the sequence is a true LanM
ortholog. That distinction is impossible to make from sequence alone
without alignment/HMM tooling beyond Week 4 scope. The size bounds
are intentionally permissive: too-short sequences fall outside the
model's validated input regime (training data was 113-232 residues);
too-long sequences are rejected to prevent the app from churning on
unreasonable input.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# IUPAC canonical 20 amino acids. We reject U (selenocysteine),
# O (pyrrolysine), B/Z/J (ambiguity codes), X (unknown), and gaps.
_CANONICAL_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

# Length bounds informed by MOESM3 LanM orthologs:
#   min length in MOESM3: 113 residues
#   max length in MOESM3: 232 residues
# We use generous bounds on either side to allow for near-LanM
# proteins (engineered variants, fragments) without enabling
# pathological inputs.
_MIN_LENGTH = 80
_MAX_LENGTH = 400


@dataclass
class ValidationResult:
    """Outcome of sequence validation.

    On success: is_valid=True, sequence=normalized form, error=None.
    On failure: is_valid=False, sequence=normalized (best-effort),
                error=human-readable.
    """
    is_valid: bool
    sequence: str
    error: Optional[str] = None
    length: int = 0


def validate_sequence(raw_sequence: str) -> ValidationResult:
    """
    Validates and normalizes a protein sequence input.

    Normalization steps:
      - Strip all whitespace (spaces, newlines, tabs)
      - Convert to uppercase
      - Remove a single FASTA header line if present (starts with >)

    Validation:
      - Must not contain multiple FASTA records (silent concatenation
        would produce nonsense input)
      - Must not be empty after normalization
      - Length must be in [_MIN_LENGTH, _MAX_LENGTH]
      - All characters must be in the 20 canonical amino acids

    @param raw_sequence: Untrusted user input string.
    return : ValidationResult with is_valid, normalized sequence,
             and a human-readable error message if invalid.
    """
    if raw_sequence is None:
        return ValidationResult(
            is_valid=False, sequence="", error="No sequence provided.",
        )

    # Strip FASTA header lines (anything starting with >).
    # Reject multiple FASTA records: Week 4 v1 accepts a single sequence
    # only. Silently concatenating records would produce nonsense input.
    stripped_lines = [
        line.strip() for line in raw_sequence.strip().splitlines()
    ]
    header_count = sum(
        1 for line in stripped_lines if line.startswith(">")
    )
    if header_count > 1:
        return ValidationResult(
            is_valid=False, sequence="",
            error=f"Multiple FASTA records detected ({header_count}). "
                  f"This app accepts one sequence at a time. "
                  f"Remove extra records and try again.",
        )

    sequence_lines = [
        line for line in stripped_lines if not line.startswith(">")
    ]
    normalized = "".join("".join(sequence_lines).split()).upper()

    if not normalized:
        return ValidationResult(
            is_valid=False, sequence="",
            error="No sequence content found after removing whitespace "
                  "and headers.",
        )

    # Length check
    length = len(normalized)
    if length < _MIN_LENGTH:
        return ValidationResult(
            is_valid=False, sequence=normalized, length=length,
            error=f"Sequence is too short ({length} residues). "
                  f"This app supports {_MIN_LENGTH}-{_MAX_LENGTH} residues. "
                  f"For reference, LanM orthologs in the training data "
                  f"range from 113 to 232 residues.",
        )
    if length > _MAX_LENGTH:
        return ValidationResult(
            is_valid=False, sequence=normalized, length=length,
            error=f"Sequence is too long ({length} residues). "
                  f"This app supports {_MIN_LENGTH}-{_MAX_LENGTH} residues. "
                  f"For reference, LanM orthologs in the training data "
                  f"range from 113 to 232 residues.",
        )

    # Canonical amino acid check
    invalid_chars = sorted(set(normalized) - _CANONICAL_AMINO_ACIDS)
    if invalid_chars:
        return ValidationResult(
            is_valid=False, sequence=normalized, length=length,
            error=f"Sequence contains non-canonical amino acids: "
                  f"{invalid_chars}. Only the 20 standard amino acids "
                  f"(ACDEFGHIKLMNPQRSTVWY) are accepted.",
        )

    return ValidationResult(
        is_valid=True, sequence=normalized, length=length, error=None,
    )
