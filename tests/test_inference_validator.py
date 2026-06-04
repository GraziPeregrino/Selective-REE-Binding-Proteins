"""Tests for sequence validator (Week 4 Block 4.1)."""
from __future__ import annotations

import pytest

from agentic_ai.inference.sequence_validator import (
    ValidationResult,
    validate_sequence,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

# A reasonable LanM-sized sequence using only canonical amino acids.
# 150 residues, well within [80, 400] bounds.
_VALID_SEQUENCE = (
    "MACETIDKFDHEMACETIDKFDHEMACETIDKFDHEMACETIDKFDHE"
    "MACETIDKFDHEMACETIDKFDHEMACETIDKFDHEMACETIDKFDHE"
    "MACETIDKFDHEMACETIDKFDHEMACETIDKFDHEMACETIDKFDH"
)
assert 80 <= len(_VALID_SEQUENCE) <= 400


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_valid_sequence_passes():
    """A valid sequence returns is_valid=True with no error."""
    result = validate_sequence(_VALID_SEQUENCE)
    assert result.is_valid is True
    assert result.error is None
    assert result.length == len(_VALID_SEQUENCE)
    assert result.sequence == _VALID_SEQUENCE


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def test_lowercase_input_is_normalized_to_uppercase():
    """Lowercase amino acid letters are uppercased."""
    lowercase = _VALID_SEQUENCE.lower()
    result = validate_sequence(lowercase)
    assert result.is_valid is True
    assert result.sequence == _VALID_SEQUENCE


def test_whitespace_is_stripped():
    """Internal whitespace (spaces, tabs, newlines) is stripped."""
    spaced = " ".join(_VALID_SEQUENCE)  # space between each residue
    result = validate_sequence(spaced)
    assert result.is_valid is True
    assert result.sequence == _VALID_SEQUENCE


def test_newlines_are_stripped():
    """Newlines (e.g. from FASTA-style wrapping) are stripped."""
    wrapped = "\n".join(
        _VALID_SEQUENCE[i:i+60] for i in range(0, len(_VALID_SEQUENCE), 60)
    )
    result = validate_sequence(wrapped)
    assert result.is_valid is True
    assert result.sequence == _VALID_SEQUENCE


def test_single_fasta_header_is_stripped():
    """A single FASTA header line is removed; remaining sequence kept."""
    fasta = f">test_sequence description\n{_VALID_SEQUENCE}"
    result = validate_sequence(fasta)
    assert result.is_valid is True
    assert result.sequence == _VALID_SEQUENCE


def test_indented_fasta_header_is_recognized():
    """A FASTA header with leading whitespace is still detected."""
    fasta = f"   >test_sequence\n{_VALID_SEQUENCE}"
    result = validate_sequence(fasta)
    assert result.is_valid is True
    assert result.sequence == _VALID_SEQUENCE


# ---------------------------------------------------------------------------
# Rejection: multiple FASTA records
# ---------------------------------------------------------------------------

def test_multiple_fasta_records_are_rejected():
    """Two FASTA records would silently concatenate; reject explicitly."""
    multi_fasta = f">seq1\n{_VALID_SEQUENCE}\n>seq2\n{_VALID_SEQUENCE}"
    result = validate_sequence(multi_fasta)
    assert result.is_valid is False
    assert "Multiple FASTA records" in result.error
    assert "2" in result.error


# ---------------------------------------------------------------------------
# Rejection: empty / None
# ---------------------------------------------------------------------------

def test_none_input_is_rejected():
    """None input returns is_valid=False with informative error."""
    result = validate_sequence(None)
    assert result.is_valid is False
    assert "No sequence provided" in result.error


def test_empty_string_is_rejected():
    """Empty string is rejected."""
    result = validate_sequence("")
    assert result.is_valid is False
    assert "No sequence content found" in result.error


def test_whitespace_only_input_is_rejected():
    """Input with only whitespace is rejected after normalization."""
    result = validate_sequence("   \n\t  ")
    assert result.is_valid is False
    assert "No sequence content found" in result.error


def test_fasta_header_only_is_rejected():
    """Input with only a FASTA header and no sequence content is rejected."""
    result = validate_sequence(">my_sequence\n")
    assert result.is_valid is False
    assert "No sequence content" in result.error


# ---------------------------------------------------------------------------
# Rejection: length bounds
# ---------------------------------------------------------------------------

def test_too_short_sequence_is_rejected():
    """A sequence shorter than _MIN_LENGTH is rejected."""
    short = "M" * 50  # below the 80 minimum
    result = validate_sequence(short)
    assert result.is_valid is False
    assert "too short" in result.error
    assert "50" in result.error
    assert result.length == 50


def test_too_long_sequence_is_rejected():
    """A sequence longer than _MAX_LENGTH is rejected."""
    too_long = "M" * 500  # above the 400 maximum
    result = validate_sequence(too_long)
    assert result.is_valid is False
    assert "too long" in result.error
    assert "500" in result.error


def test_exactly_at_min_length_is_accepted():
    """A sequence of exactly _MIN_LENGTH residues is accepted."""
    at_min = "MACETIDKFD" * 8  # 80 residues
    result = validate_sequence(at_min)
    assert result.is_valid is True
    assert result.length == 80


def test_exactly_at_max_length_is_accepted():
    """A sequence of exactly _MAX_LENGTH residues is accepted."""
    at_max = "M" * 400
    result = validate_sequence(at_max)
    assert result.is_valid is True
    assert result.length == 400


# ---------------------------------------------------------------------------
# Rejection: invalid characters
# ---------------------------------------------------------------------------

def test_sequence_with_X_is_rejected():
    """X (unknown residue) is not in the canonical alphabet."""
    invalid = _VALID_SEQUENCE[:75] + "X" + _VALID_SEQUENCE[76:]
    result = validate_sequence(invalid)
    assert result.is_valid is False
    assert "non-canonical amino acids" in result.error
    assert "X" in result.error


def test_sequence_with_selenocysteine_is_rejected():
    """U (selenocysteine) is not in the canonical 20."""
    invalid = _VALID_SEQUENCE[:75] + "U" + _VALID_SEQUENCE[76:]
    result = validate_sequence(invalid)
    assert result.is_valid is False
    assert "U" in result.error


def test_sequence_with_digit_is_rejected():
    """Numeric characters in the middle of a sequence are rejected."""
    invalid = _VALID_SEQUENCE[:75] + "2" + _VALID_SEQUENCE[76:]
    result = validate_sequence(invalid)
    assert result.is_valid is False
    assert "non-canonical" in result.error


def test_sequence_with_gap_character_is_rejected():
    """Gap characters ('-' from alignments) are rejected."""
    invalid = _VALID_SEQUENCE[:75] + "-" + _VALID_SEQUENCE[76:]
    result = validate_sequence(invalid)
    assert result.is_valid is False
    assert "non-canonical" in result.error


def test_multiple_invalid_characters_all_reported():
    """All distinct invalid characters are listed in the error message."""
    invalid = _VALID_SEQUENCE[:50] + "X" + _VALID_SEQUENCE[51:75] + "U" + _VALID_SEQUENCE[76:]
    result = validate_sequence(invalid)
    assert result.is_valid is False
    assert "U" in result.error
    assert "X" in result.error
