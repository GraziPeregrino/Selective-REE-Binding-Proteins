"""Hyperparameter tuning via randomized search (Week 3 Block 3.2).

Locked design:
  - 50 randomly sampled hyperparameter combinations via
    sklearn ParameterSampler (fixed seed for reproducibility)
  - 5-fold StratifiedGroupKFold outer CV (variant_id grouped,
    selectivity_cluster stratified)
  - 5 fixed inner-holdout splits (one per outer fold) for
    early stopping, materialized once and reused across all trials
  - Headline metric: macro per-element Spearman across 5 folds
  - Selection rule (pre-stated): argmax of mean CV score
  - Retraining rule (pre-stated): median best_iteration + 1 across
    winning trial's outer folds; retrain on full training set
  - Test set untouched until the single final evaluation
  - Trial-level checkpoint writes after each completed trial

CLI:
    python -m agentic_ai.models.tune
"""
from __future__ import annotations

import json
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform, spearmanr
from sklearn.model_selection import ParameterSampler, StratifiedGroupKFold
from xgboost import XGBRegressor

import xgboost as xgb_module
import sklearn as sklearn_module
from agentic_ai.models.metrics import compute_all_metrics


# -- Locked configuration ----------------------------------------------------
_RANDOM_SEED = 42
_N_TRIALS = 50
_N_OUTER_FOLDS = 5
_INNER_HOLDOUT_FRACTION = 0.15
_N_ESTIMATORS_CEILING = 1000
_EARLY_STOPPING_ROUNDS = 40


def _search_space() -> Dict:
    return {
        "learning_rate":     loguniform(0.02, 0.2),
        "max_depth":         randint(2, 8),         # [2, 7] inclusive
        "subsample":         uniform(0.6, 0.4),     # 0.6 to 1.0
        "colsample_bytree":  uniform(0.5, 0.5),     # 0.5 to 1.0
        "min_child_weight":  randint(1, 16),        # [1, 15] inclusive
        "reg_alpha":         loguniform(1e-4, 10),
        "reg_lambda":        loguniform(1e-2, 20),
    }


_MODEL_PATH = Path("models/xgb_tuned.json")
_METRICS_PATH = Path("models/xgb_tuned_metrics.json")
_IMPORTANCE_PATH = Path("models/xgb_tuned_feature_importance.csv")
_TUNING_RESULTS_PATH = Path("models/xgb_tuning_results.csv")
_ELEMENT_SCORES_PATH = Path("models/xgb_tuning_element_scores.csv")
_CONFIG_PATH = Path("models/xgb_tuning_config.json")
_BASELINE_METRICS_PATH = Path("models/xgb_baseline_metrics.json")

_FEATURE_MATRIX_PATH = Path("data/processed/ml_ready_features.parquet")
_FEATURE_SCHEMA_PATH = Path("data/processed/ml_ready_features_schema.json")


def tune_xgboost(
    n_trials: int = _N_TRIALS,
    seed: int = _RANDOM_SEED,
) -> Dict:
    t_start = time.time()

    df = pd.read_parquet(_FEATURE_MATRIX_PATH)
    feature_cols = json.loads(_FEATURE_SCHEMA_PATH.read_text())
    _validate_schema(df, feature_cols)

    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

    # Materialize outer folds AND inner splits ONCE before the search.
    # All trials see identical fold compositions and identical inner
    # train/val splits.
    outer_fold_assignments = _materialize_outer_folds(train_df, seed=seed)
    inner_splits_per_fold = _materialize_inner_splits(
        train_df, outer_fold_assignments, seed=seed,
    )

    _persist_config(
        n_trials=n_trials,
        seed=seed,
        outer_fold_assignments=outer_fold_assignments,
        inner_splits_per_fold=inner_splits_per_fold,
        train_df=train_df,
    )

    # Clear prior checkpoint files so this run starts fresh
    if _TUNING_RESULTS_PATH.exists():
        _TUNING_RESULTS_PATH.unlink()
    if _ELEMENT_SCORES_PATH.exists():
        _ELEMENT_SCORES_PATH.unlink()

    sampler = ParameterSampler(
        _search_space(), n_iter=n_trials, random_state=seed,
    )
    parameter_grid = [_normalize_params(p) for p in sampler]

    results = _run_search(
        parameter_grid=parameter_grid,
        train_df=train_df,
        feature_cols=feature_cols,
        outer_fold_assignments=outer_fold_assignments,
        inner_splits_per_fold=inner_splits_per_fold,
        seed=seed,
    )

    # Pick winner per the pre-stated rule (argmax mean CV score)
    results_df = pd.read_csv(_TUNING_RESULTS_PATH)
    results_df["rank"] = (
        results_df["mean_cv_score"].rank(method="min", ascending=False).astype(int)
    )
    results_df.sort_values("rank", inplace=True)
    results_df.to_csv(_TUNING_RESULTS_PATH, index=False)

    winner = results_df.iloc[0]
    winning_params = json.loads(winner["params_json"])
    winning_fold_iterations = json.loads(winner["best_iteration_per_fold"])
    # CRITICAL: best_iteration is zero-indexed; tree count is +1
    final_n_estimators = int(np.median(winning_fold_iterations)) + 1

    final_model, test_metrics, importance_series = _retrain_and_evaluate(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        winning_params=winning_params,
        final_n_estimators=final_n_estimators,
        seed=seed,
    )

    final_model.save_model(_MODEL_PATH)

    metrics_for_json = _make_json_safe(test_metrics)
    metrics_for_json.update({
        "winning_params":            winning_params,
        "final_n_estimators":        final_n_estimators,
        "best_iteration_per_fold":   winning_fold_iterations,
        "mean_cv_score":             float(winner["mean_cv_score"]),
        "std_cv_score":              float(winner["std_cv_score"]),
        "n_trials":                  n_trials,
        "n_outer_folds":             _N_OUTER_FOLDS,
        "seed":                      seed,
    })

    with open(_METRICS_PATH, "w") as fh:
        json.dump(metrics_for_json, fh, indent=2)

    importance_series.to_csv(_IMPORTANCE_PATH, header=True)

    runtime = time.time() - t_start

    return {
        "winning_params":          winning_params,
        "final_n_estimators":      final_n_estimators,
        "best_iteration_per_fold": winning_fold_iterations,
        "mean_cv_score":           float(winner["mean_cv_score"]),
        "std_cv_score":            float(winner["std_cv_score"]),
        "test_metrics":            test_metrics,
        "runtime_seconds":         runtime,
        "n_trials":                n_trials,
    }


def _run_search(
    parameter_grid: List[Dict],
    train_df: pd.DataFrame,
    feature_cols: List[str],
    outer_fold_assignments: List[Tuple[np.ndarray, np.ndarray]],
    inner_splits_per_fold: List[Tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> None:
    """
    Runs all trials. Appends to checkpoint CSVs after each trial so
    progress is recoverable on interruption.
    """
    n_trials = len(parameter_grid)
    expected_elements = set(train_df["target_element"].unique())
    print(f"Starting search: {n_trials} trials x {_N_OUTER_FOLDS} folds = "
          f"{n_trials * _N_OUTER_FOLDS} model fits")
    print(f"Expected REEs per fold: {len(expected_elements)}")
    print()

    for trial_idx, params in enumerate(parameter_grid):
        trial_start = time.time()
        fold_scores = []
        fold_iterations = []
        trial_element_rows = []

        for fold_idx, (outer_train_idx, outer_val_idx) in enumerate(
            outer_fold_assignments
        ):
            outer_train = train_df.iloc[outer_train_idx]
            outer_val = train_df.iloc[outer_val_idx]

            # Use the pre-materialized inner split for this outer fold
            inner_train_local_idx, inner_val_local_idx = (
                inner_splits_per_fold[fold_idx]
            )
            inner_train = outer_train.iloc[inner_train_local_idx]
            inner_val = outer_train.iloc[inner_val_local_idx]

            model = XGBRegressor(
                **params,
                n_estimators=_N_ESTIMATORS_CEILING,
                early_stopping_rounds=_EARLY_STOPPING_ROUNDS,
                objective="reg:squarederror",
                tree_method="hist",
                eval_metric="rmse",
                random_state=seed,
            )

            model.fit(
                inner_train[feature_cols], inner_train["value"],
                eval_set=[(inner_val[feature_cols], inner_val["value"])],
                verbose=False,
            )

            outer_val_with_pred = outer_val.copy()
            outer_val_with_pred["predicted"] = model.predict(
                outer_val[feature_cols]
            )

            per_element_result = _per_element_spearman_with_logging(
                outer_val_with_pred,
                expected_elements=expected_elements,
                trial_idx=trial_idx,
                fold_idx=fold_idx,
            )

            fold_scores.append(per_element_result["macro"])
            fold_iterations.append(int(model.best_iteration))

            for element, rho in per_element_result["per_element"].items():
                trial_element_rows.append({
                    "trial":      trial_idx,
                    "outer_fold": fold_idx,
                    "element":    element,
                    "spearman":   rho,
                })

        mean_score = float(np.mean(fold_scores))
        std_score = float(np.std(fold_scores))
        elapsed = time.time() - trial_start

        trial_row = {
            "trial":                    trial_idx,
            "params_json":              json.dumps(params),
            "fold_scores_json":         json.dumps(fold_scores),
            "best_iteration_per_fold":  json.dumps(fold_iterations),
            "mean_cv_score":            mean_score,
            "std_cv_score":             std_score,
            "elapsed_seconds":          elapsed,
            **params,
        }

        # Checkpoint: append this trial to both CSVs immediately
        _append_to_csv(_TUNING_RESULTS_PATH, [trial_row])
        _append_to_csv(_ELEMENT_SCORES_PATH, trial_element_rows)

        print(
            f"Trial {trial_idx+1:3d}/{n_trials}  "
            f"score={mean_score:+.4f} std={std_score:.4f}  "
            f"iters={fold_iterations}  time={elapsed:.1f}s"
        )


def _materialize_outer_folds(train_df, seed):
    splitter = StratifiedGroupKFold(
        n_splits=_N_OUTER_FOLDS, shuffle=True, random_state=seed,
    )
    return list(splitter.split(
        train_df,
        y=train_df["selectivity_cluster"],
        groups=train_df["variant_id"],
    ))


def _materialize_inner_splits(
    train_df: pd.DataFrame,
    outer_fold_assignments: List[Tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generates ONE inner train/val split per outer fold, fixed for the
    entire search. Each trial reuses these same 5 inner splits.
    Returns indices that are local to each outer training fold.
    """
    n_inner_splits = max(2, int(round(1.0 / _INNER_HOLDOUT_FRACTION)))
    inner_splits = []

    for fold_idx, (outer_train_idx, _) in enumerate(outer_fold_assignments):
        outer_train = train_df.iloc[outer_train_idx]
        # Inner seed: derive deterministically from the base seed + fold,
        # but fix it once per fold (NOT per trial)
        inner_splitter = StratifiedGroupKFold(
            n_splits=n_inner_splits,
            shuffle=True,
            random_state=seed + fold_idx,
        )
        train_local, val_local = next(inner_splitter.split(
            outer_train,
            y=outer_train["selectivity_cluster"],
            groups=outer_train["variant_id"],
        ))
        inner_splits.append((train_local, val_local))

    return inner_splits


def _per_element_spearman_with_logging(
    df: pd.DataFrame,
    expected_elements: set,
    trial_idx: int,
    fold_idx: int,
) -> Dict:
    """
    Computes macro per-element Spearman with explicit skip-logging.
    Warns if any expected element is missing from the per-element
    breakdown so we can audit trial comparability.
    """
    per_element = {}
    skipped = []

    for element, group in df.groupby("target_element"):
        if len(group) < 2:
            skipped.append(element)
            continue
        rho, _ = spearmanr(group["value"], group["predicted"])
        if np.isnan(rho):
            skipped.append(element)
            continue
        per_element[element] = float(rho)

    # If any expected REE is missing, warn but don't crash — the trial
    # might still be informative (just less directly comparable)
    missing = expected_elements - set(per_element.keys())
    if missing:
        warnings.warn(
            f"Trial {trial_idx} fold {fold_idx}: "
            f"skipped {len(missing)} elements from per-element score: "
            f"{sorted(missing)}",
            UserWarning,
        )

    macro = float(np.mean(list(per_element.values()))) if per_element else float("nan")
    return {"macro": macro, "per_element": per_element, "skipped": skipped}


def _retrain_and_evaluate(
    train_df, test_df, feature_cols, winning_params, final_n_estimators, seed,
):
    model = XGBRegressor(
        **winning_params,
        n_estimators=final_n_estimators,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=seed,
    )
    model.fit(train_df[feature_cols], train_df["value"], verbose=False)

    test_with_pred = test_df.copy()
    test_with_pred["predicted"] = model.predict(test_df[feature_cols])
    test_metrics = compute_all_metrics(test_with_pred)

    importance_series = pd.Series(
        model.feature_importances_,
        index=feature_cols,
        name="importance",
    ).sort_values(ascending=False)

    return model, test_metrics, importance_series


def _validate_schema(df, feature_cols):
    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise ValueError(
            f"Feature matrix is missing schema columns: {sorted(missing)}"
        )


def _normalize_params(params: Dict) -> Dict:
    """
    Converts NumPy scalars (returned by scipy distributions) to Python
    primitives. Critical: max_depth and min_child_weight come back as
    numpy.int64 which would silently become strings if we relied on
    default=str during JSON serialization.
    """
    out = {}
    for k, v in params.items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def _append_to_csv(path: Path, rows: List[Dict]) -> None:
    """Appends rows to a CSV, writing the header only on first write."""
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    new_df.to_csv(path, mode="a", header=write_header, index=False)


def _persist_config(
    n_trials, seed,
    outer_fold_assignments, inner_splits_per_fold, train_df,
):
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    fold_assignments = []
    for fold_idx, (outer_train_idx, outer_val_idx) in enumerate(
        outer_fold_assignments
    ):
        val_variants = sorted(
            train_df.iloc[outer_val_idx]["variant_id"].unique().tolist()
        )
        outer_train = train_df.iloc[outer_train_idx]
        inner_train_local, inner_val_local = inner_splits_per_fold[fold_idx]
        inner_val_variants = sorted(
            outer_train.iloc[inner_val_local]["variant_id"].unique().tolist()
        )
        fold_assignments.append({
            "outer_fold":                  fold_idx,
            "outer_validation_variants":   val_variants,
            "n_outer_validation_variants": len(val_variants),
            "inner_validation_variants":   inner_val_variants,
            "n_inner_validation_variants": len(inner_val_variants),
        })

    config = {
        "timestamp_iso":          datetime.now().isoformat(timespec="seconds"),
        "search_method":          "RandomizedSearch (sklearn ParameterSampler)",
        "n_trials":               n_trials,
        "n_outer_folds":          _N_OUTER_FOLDS,
        "inner_holdout_fraction": _INNER_HOLDOUT_FRACTION,
        "n_estimators_ceiling":   _N_ESTIMATORS_CEILING,
        "early_stopping_rounds":  _EARLY_STOPPING_ROUNDS,
        "headline_metric":        "macro_per_element_spearman",
        "selection_rule":         "argmax(mean_cv_score)",
        "retraining_rule":        "median(best_iteration) + 1 across winning trial's outer folds",
        "search_space": {
            "learning_rate":     "loguniform(0.02, 0.2)",
            "max_depth":         "randint(2, 8)",
            "subsample":         "uniform(0.6, 1.0)",
            "colsample_bytree":  "uniform(0.5, 1.0)",
            "min_child_weight":  "randint(1, 16)",
            "reg_alpha":         "loguniform(1e-4, 10)",
            "reg_lambda":        "loguniform(1e-2, 20)",
        },
        "seeds": {
            "sampler_seed":     seed,
            "outer_fold_seed":  seed,
            "inner_split_seed": f"seed + outer_fold_idx (fixed, materialized once)",
            "model_seed":       seed,
        },
        "library_versions": {
            "xgboost":  xgb_module.__version__,
            "sklearn":  sklearn_module.__version__,
            "numpy":    np.__version__,
            "pandas":   pd.__version__,
        },
        "fold_assignments": fold_assignments,
    }

    with open(_CONFIG_PATH, "w") as fh:
        json.dump(config, fh, indent=2)


def _make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_safe(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _print_baseline_comparison(test_metrics: Dict) -> None:
    """Prints delta table comparing tuned model to persisted baseline."""
    if not _BASELINE_METRICS_PATH.exists():
        print(f"(No baseline metrics found at {_BASELINE_METRICS_PATH}; "
              f"skipping comparison.)")
        return

    baseline = json.load(open(_BASELINE_METRICS_PATH))
    baseline_pes = baseline["per_element_spearman"]["macro"]
    baseline_pvs = baseline["per_variant_spearman"]["macro"]
    baseline_rmse = baseline["rmse"]
    baseline_mae = baseline["mae"]
    baseline_r2 = baseline["r2"]

    tuned_pes = test_metrics["per_element_spearman"]["macro"]
    tuned_pvs = test_metrics["per_variant_spearman"]["macro"]

    print(f"=== Baseline vs Tuned (test-set comparison) ===")
    print(f"  Metric                          Baseline   Tuned     Delta")
    print(f"  per-element Spearman (HEADLINE) {baseline_pes:+.4f}    "
          f"{tuned_pes:+.4f}   {tuned_pes - baseline_pes:+.4f}")
    print(f"  per-variant Spearman            {baseline_pvs:+.4f}    "
          f"{tuned_pvs:+.4f}   {tuned_pvs - baseline_pvs:+.4f}")
    print(f"  RMSE                            {baseline_rmse:.4f}     "
          f"{test_metrics['rmse']:.4f}    {test_metrics['rmse'] - baseline_rmse:+.4f}")
    print(f"  MAE                             {baseline_mae:.4f}     "
          f"{test_metrics['mae']:.4f}    {test_metrics['mae'] - baseline_mae:+.4f}")
    print(f"  R^2                             {baseline_r2:.4f}     "
          f"{test_metrics['r2']:.4f}    {test_metrics['r2'] - baseline_r2:+.4f}")


def main() -> int:
    print(f"=== Block 3.2 randomized search ===")
    print(f"  Trials: {_N_TRIALS}")
    print(f"  Outer folds: {_N_OUTER_FOLDS}")
    print(f"  Seed: {_RANDOM_SEED}")
    print()

    result = tune_xgboost()

    test_metrics = result["test_metrics"]
    pes = test_metrics["per_element_spearman"]
    pvs = test_metrics["per_variant_spearman"]

    print()
    print(f"=== Search complete in {result['runtime_seconds']:.1f}s ===")
    print()
    print(f"=== Winning configuration ===")
    print(f"  Mean CV score (per-element Spearman macro): "
          f"{result['mean_cv_score']:+.4f} (+/- {result['std_cv_score']:.4f})")
    print(f"  Best iteration per fold:    {result['best_iteration_per_fold']}")
    print(f"  Selected n_estimators:      {result['final_n_estimators']}")
    print(f"  Hyperparameters:")
    for k, v in result["winning_params"].items():
        if isinstance(v, float):
            print(f"    {k:<22} {v:.6f}")
        else:
            print(f"    {k:<22} {v}")
    print()
    print(f"=== Test-set evaluation (single pass) ===")
    print(f"  HEADLINE per-element Spearman: {pes['macro']:+.4f}")
    print(f"  per-variant Spearman:          {pvs['macro']:+.4f}")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  MAE:  {test_metrics['mae']:.4f}")
    print(f"  R^2:  {test_metrics['r2']:.4f}")
    print()
    _print_baseline_comparison(test_metrics)
    print()
    print(f"=== Artifacts persisted ===")
    print(f"  {_MODEL_PATH}")
    print(f"  {_METRICS_PATH}")
    print(f"  {_IMPORTANCE_PATH}")
    print(f"  {_TUNING_RESULTS_PATH}")
    print(f"  {_ELEMENT_SCORES_PATH}")
    print(f"  {_CONFIG_PATH}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
