"""Single-sequence prediction orchestrator (Week 4 Block 4.3).

Ties together sequence validation, feature construction, and the
trained XGBoost model. The public `predict_profile()` function is
the single entrypoint Streamlit (or any other caller) uses for
inference.

The deployed model is the Block 3.1 baseline, selected per the
inference contract in docs/block_3_5_inference_contract.md because
the tuned model did not improve the pre-declared headline metric
(macro per-element Spearman).

Per-REE warnings:
  - Samarium: ranking confidence reduced due to target compression
    in training data. 44.1% of orthologs are saturated at Sm = 1.0;
    81.5% are at Sm >= 0.95. Documented in
    docs/block_3_3_samarium_investigation.md.

Per-sequence warnings:
  - Fewer than 2 detected EF-hand motifs: the sequence falls outside
    the validated LanM input regime, even if structurally valid.

Column naming convention:
  - target_element: matches the training-matrix column name
  - predicted_normalized_logD: explicit about what's predicted
    (normalized on-resin selectivity, NOT raw logD or affinity)
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Dict

import pandas as pd
import xgboost as xgb

from agentic_ai.features.ef_hand_motifs import find_ef_hand_motifs
from agentic_ai.features.ree_features import get_ree_features
from agentic_ai.inference.feature_builder import (
    build_feature_matrix_for_sequence,
    get_element_order,
)
from agentic_ai.inference.sequence_validator import validate_sequence

_MODEL_PATH = Path("models/xgb_baseline.json")
_MODEL_NAME = "Block 3.1 baseline XGBoost"

# Per-REE confidence flags. Block 3.3 documented Sm's reduced ranking
# confidence due to target compression. Other REEs do not require
# special caveats given current evidence.
_REDUCED_CONFIDENCE_ELEMENTS = {"Samarium"}
_SAMARIUM_CONFIDENCE_TEXT = "Reduced confidence: target saturation"

# Threshold below which we warn about low EF-hand motif count.
# 2 is the minimum for any aggregate spacing/spread features to be
# meaningful, and is a soft lower bound for "this sequence resembles
# a canonical LanM ortholog."
_MIN_EXPECTED_MOTIFS = 2


@lru_cache(maxsize=1)
def _load_model() -> xgb.XGBRegressor:
    """
    Loads the deployed XGBoost model from disk. Cached so repeat
    predictions don't re-deserialize.
    return : XGBRegressor with weights loaded from models/xgb_baseline.json.
    raises : FileNotFoundError if the model file is missing.
    """
    if not _MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {_MODEL_PATH}. "
            f"Run `python -m agentic_ai.models.train` to regenerate, "
            f"then run the Block 3.5 JSON export."
        )
    model = xgb.XGBRegressor()
    model.load_model(_MODEL_PATH)
    return model


@lru_cache(maxsize=1)
def _compute_model_version() -> str:
    """
    Computes a deterministic version identifier from the model file's
    SHA256. The first 12 hex characters give 48 bits of entropy, which
    is more than enough to detect any meaningful model change.
    return : String of the form 'xgb_baseline_json_sha256:abc123def456'.
    """
    if not _MODEL_PATH.exists():
        return "xgb_baseline_json_sha256:unknown"
    sha = hashlib.sha256(_MODEL_PATH.read_bytes()).hexdigest()[:12]
    return f"xgb_baseline_json_sha256:{sha}"


def predict_profile(sequence: str) -> Dict:
    """
    Predicts the REE selectivity profile for a single protein sequence.

    Pipeline:
      1. Validate and normalize the sequence
      2. If invalid, return a structured error response
      3. Build the 15-row inference feature matrix
      4. Load the trained model
      5. Predict normalized_logD for each REE
      6. Attach per-REE atomic numbers, confidence flags, and ranks
      7. Compute per-sequence warnings (low motif count, etc.)
      8. Return the full result dict

    @param sequence: Raw user input. Will be validated and normalized.
    return : Dict with keys:
      - "is_valid":      bool
      - "error":         str or None (set when is_valid=False)
      - "sequence":      validated/normalized sequence string
      - "length":        int (validated sequence length)
      - "n_motifs":      int (count of detected EF-hand motifs)
      - "predictions":   pd.DataFrame or None (15 rows when valid)
      - "warnings":      list of str
      - "model_name":    str (human-readable)
      - "model_version": str (deterministic SHA-based identifier)

    Predictions DataFrame columns:
      - target_element:            REE name (matches training column)
      - atomic_number:             int
      - predicted_normalized_logD: float
      - rank:                      int (1 = highest predicted value)
      - reduced_confidence:        bool
      - confidence_note:           str ("" or warning text)

    Rows are sorted by rank ascending (i.e. highest predicted first).
    """
    # Always include model identifiers, even on error paths, so the UI
    # can show provenance consistently.
    response = {
        "is_valid":      False,
        "error":         None,
        "sequence":      "",
        "length":        0,
        "n_motifs":      0,
        "predictions":   None,
        "warnings":      [],
        "model_name":    _MODEL_NAME,
        "model_version": _compute_model_version(),
    }

    # Step 1: validate
    validation = validate_sequence(sequence)
    response["sequence"] = validation.sequence
    response["length"] = validation.length

    if not validation.is_valid:
        response["error"] = validation.error
        return response

    response["is_valid"] = True

    # Step 2: count motifs (for the low-motif warning and UI display)
    motifs = find_ef_hand_motifs(validation.sequence)
    response["n_motifs"] = len(motifs)
    if len(motifs) < _MIN_EXPECTED_MOTIFS:
        response["warnings"].append(
            f"Low EF-hand motif count ({len(motifs)} detected). LanM "
            f"orthologs in the training data have 4 canonical motifs. "
            f"Predictions for this sequence are outside the validated "
            f"input regime and should be interpreted with caution."
        )

    # Step 3: build features
    X = build_feature_matrix_for_sequence(validation.sequence)

    # Step 4 & 5: load model, predict
    model = _load_model()
    y_pred = model.predict(X)

    # Step 6: assemble predictions DataFrame with all metadata
    element_order = get_element_order()
    predictions_df = pd.DataFrame({
        "target_element": element_order,
        "atomic_number": [
            get_ree_features(e)["atomic_number"] for e in element_order
        ],
        "predicted_normalized_logD": y_pred,
        "reduced_confidence": [
            e in _REDUCED_CONFIDENCE_ELEMENTS for e in element_order
        ],
        "confidence_note": [
            _SAMARIUM_CONFIDENCE_TEXT if e in _REDUCED_CONFIDENCE_ELEMENTS else ""
            for e in element_order
        ],
    })

    # Add rank column based on predicted value (1 = highest)
    predictions_df["rank"] = (
        predictions_df["predicted_normalized_logD"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    # Sort by rank for caller convenience; chart code can resort by
    # atomic_number when needed
    predictions_df = predictions_df.sort_values("rank").reset_index(drop=True)

    response["predictions"] = predictions_df

    # Step 7: per-REE warnings (always include Sm caveat when valid)
    response["warnings"].append(
        "Samarium ranking confidence is reduced due to target "
        "compression in training data: 44.1% of orthologs are "
        "saturated at Sm = 1.0; 81.5% are at Sm >= 0.95. "
        "See Block 3.3 investigation for details."
    )

    return response


def get_model_info() -> Dict:
    """
    Returns model metadata without running inference. Useful for the
    Streamlit About panel.
    return : Dict with model_name, model_version, model_path.
    """
    return {
        "model_name":    _MODEL_NAME,
        "model_version": _compute_model_version(),
        "model_path":    str(_MODEL_PATH),
    }
