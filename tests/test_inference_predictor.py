"""Tests for the prediction orchestrator (Week 4 Block 4.3)."""
from __future__ import annotations

import re

import pandas as pd
import pytest

from agentic_ai.inference.predictor import (
    get_model_info,
    predict_profile,
)


# ---------------------------------------------------------------------------
# Fixtures: known sequences
# ---------------------------------------------------------------------------

# Mex-LanM (o-621): a real MOESM3 ortholog, used as the ground truth
# for inference verification.
_MEX_LANM_SEQUENCE = (
    "MAPTTTTKVDIAAFDPDKDGTIDLKEALAAGSAAFDKLDPDKDGTLDAKELKGRVSEADLKKL"
    "DPDNDGTLDKKEYLAAVEAQFKAANPDNDGTIDARELASPAGSALVNLIR"
)

# A sequence with no detectable EF-hand motifs, for testing the
# low-motif warning path.
_NO_MOTIF_SEQUENCE = "M" + "A" * 119


# ---------------------------------------------------------------------------
# Happy path: valid sequence with full structure
# ---------------------------------------------------------------------------

def test_valid_sequence_returns_complete_response():
    """All expected keys are present in the response for a valid sequence."""
    result = predict_profile(_MEX_LANM_SEQUENCE)
    required_keys = {
        "is_valid", "error", "sequence", "length", "n_motifs",
        "predictions", "warnings", "model_name", "model_version",
    }
    assert set(result.keys()) == required_keys


def test_valid_sequence_marked_valid():
    result = predict_profile(_MEX_LANM_SEQUENCE)
    assert result["is_valid"] is True
    assert result["error"] is None


def test_valid_sequence_returns_15_predictions():
    """The predictions DataFrame must have 15 rows (one per REE)."""
    result = predict_profile(_MEX_LANM_SEQUENCE)
    assert isinstance(result["predictions"], pd.DataFrame)
    assert len(result["predictions"]) == 15


def test_predictions_have_required_columns():
    """The predictions DataFrame must have all documented columns."""
    result = predict_profile(_MEX_LANM_SEQUENCE)
    expected_columns = {
        "target_element", "atomic_number", "predicted_normalized_logD",
        "reduced_confidence", "confidence_note", "rank",
    }
    assert set(result["predictions"].columns) == expected_columns


def test_predictions_have_15_distinct_ranks():
    """Ranks should be 1 through 15 with no duplicates."""
    result = predict_profile(_MEX_LANM_SEQUENCE)
    ranks = sorted(result["predictions"]["rank"].tolist())
    assert ranks == list(range(1, 16))


def test_predictions_sorted_by_rank():
    """Predictions DataFrame should be pre-sorted by rank ascending."""
    result = predict_profile(_MEX_LANM_SEQUENCE)
    ranks = result["predictions"]["rank"].tolist()
    assert ranks == sorted(ranks)


def test_predictions_in_normalized_logd_range():
    """
    Predicted values should be within or near [0, 1].
    Tree models can extrapolate slightly outside in extreme cases,
    so we use generous bounds.
    """
    result = predict_profile(_MEX_LANM_SEQUENCE)
    values = result["predictions"]["predicted_normalized_logD"]
    assert values.min() > -0.2
    assert values.max() < 1.2


# ---------------------------------------------------------------------------
# Samarium confidence flag and warning
# ---------------------------------------------------------------------------

def test_samarium_marked_reduced_confidence():
    """The Samarium row must have reduced_confidence=True."""
    result = predict_profile(_MEX_LANM_SEQUENCE)
    sm_row = result["predictions"][
        result["predictions"]["target_element"] == "Samarium"
    ]
    assert sm_row["reduced_confidence"].iloc[0] is True or sm_row["reduced_confidence"].iloc[0] == True


def test_samarium_has_confidence_note():
    """The Samarium row must have a non-empty confidence_note."""
    result = predict_profile(_MEX_LANM_SEQUENCE)
    sm_row = result["predictions"][
        result["predictions"]["target_element"] == "Samarium"
    ]
    note = sm_row["confidence_note"].iloc[0]
    assert "Reduced confidence" in note
    assert "saturation" in note.lower()


def test_non_samarium_have_no_confidence_note():
    """All non-Samarium rows must have empty confidence_note."""
    result = predict_profile(_MEX_LANM_SEQUENCE)
    non_sm = result["predictions"][
        result["predictions"]["target_element"] != "Samarium"
    ]
    assert (non_sm["confidence_note"] == "").all()


def test_samarium_warning_included():
    """The warnings list must include the Sm-specific caveat."""
    result = predict_profile(_MEX_LANM_SEQUENCE)
    warning_text = " ".join(result["warnings"])
    assert "Samarium" in warning_text
    assert "saturation" in warning_text.lower() or "saturated" in warning_text.lower()


def test_samarium_warning_includes_both_stats():
    """Warning should reference both 44.1% and 81.5% statistics."""
    result = predict_profile(_MEX_LANM_SEQUENCE)
    warning_text = " ".join(result["warnings"])
    assert "44.1" in warning_text
    assert "81.5" in warning_text


# ---------------------------------------------------------------------------
# Low-motif warning path
# ---------------------------------------------------------------------------

def test_low_motif_sequence_produces_warning():
    """A sequence with <2 motifs should produce a low-motif warning."""
    result = predict_profile(_NO_MOTIF_SEQUENCE)
    if not result["is_valid"]:
        pytest.skip("Test sequence failed validation; skipping warning check")

    warning_text = " ".join(result["warnings"])
    assert "Low EF-hand motif" in warning_text or "outside the validated" in warning_text


def test_low_motif_sequence_still_returns_predictions():
    """Low-motif sequences should still produce predictions (with warnings)."""
    result = predict_profile(_NO_MOTIF_SEQUENCE)
    if not result["is_valid"]:
        pytest.skip("Test sequence failed validation; skipping")

    assert result["predictions"] is not None
    assert len(result["predictions"]) == 15


def test_canonical_motif_count_no_low_motif_warning():
    """Mex-LanM has 4 canonical motifs; should not trigger low-motif warning."""
    result = predict_profile(_MEX_LANM_SEQUENCE)
    warning_text = " ".join(result["warnings"])
    assert "Low EF-hand motif" not in warning_text


# ---------------------------------------------------------------------------
# Invalid sequence handling
# ---------------------------------------------------------------------------

def test_invalid_sequence_marked_invalid():
    """A sequence with non-canonical AAs must be rejected."""
    result = predict_profile("MACEXTIDKFDHE" * 10)
    assert result["is_valid"] is False
    assert result["error"] is not None


def test_invalid_sequence_returns_no_predictions():
    """An invalid sequence must not have a predictions DataFrame."""
    result = predict_profile("XXXXX" * 30)
    assert result["predictions"] is None


def test_invalid_sequence_still_includes_model_metadata():
    """
    Even on validation failure, model_name and model_version must be
    populated so the UI can display provenance consistently.
    """
    result = predict_profile("INVALID")
    assert result["model_name"]
    assert result["model_version"]


def test_too_short_sequence_returns_clear_error():
    """A too-short sequence should fail with a descriptive error."""
    result = predict_profile("MACE" * 5)  # 20 residues, below 80 minimum
    assert result["is_valid"] is False
    assert "too short" in result["error"].lower()


def test_empty_sequence_handled_gracefully():
    """An empty sequence input should not crash the predictor."""
    result = predict_profile("")
    assert result["is_valid"] is False


# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------

def test_model_version_is_sha_format():
    """model_version should match the expected sha256 format."""
    result = predict_profile(_MEX_LANM_SEQUENCE)
    assert re.match(
        r"^xgb_baseline_json_sha256:[0-9a-f]{12}$",
        result["model_version"],
    )


def test_model_name_is_human_readable():
    """model_name should be a descriptive label."""
    result = predict_profile(_MEX_LANM_SEQUENCE)
    assert "baseline" in result["model_name"].lower()


def test_get_model_info_returns_metadata_without_inference():
    """get_model_info() should work without running predictions."""
    info = get_model_info()
    assert "model_name" in info
    assert "model_version" in info
    assert "model_path" in info


def test_model_version_is_deterministic():
    """The same model file should always produce the same version string."""
    v1 = get_model_info()["model_version"]
    v2 = predict_profile(_MEX_LANM_SEQUENCE)["model_version"]
    assert v1 == v2


# ---------------------------------------------------------------------------
# Sequence echo (normalization round-trip)
# ---------------------------------------------------------------------------

def test_sequence_echoed_in_normalized_form():
    """The returned sequence should be the normalized (uppercase) version."""
    result = predict_profile(_MEX_LANM_SEQUENCE.lower())
    assert result["sequence"] == _MEX_LANM_SEQUENCE.upper()


def test_length_matches_normalized_sequence():
    """The length field should match the normalized sequence length."""
    result = predict_profile(_MEX_LANM_SEQUENCE)
    assert result["length"] == len(_MEX_LANM_SEQUENCE)


def test_n_motifs_for_canonical_lanm_is_four():
    """Mex-LanM has 4 canonical EF-hand motifs."""
    result = predict_profile(_MEX_LANM_SEQUENCE)
    assert result["n_motifs"] == 4
