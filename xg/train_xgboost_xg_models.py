"""Tune XGBoost xG hyperparameters with grid search and match-level CV.

The script is standalone for the xG workflow. It tunes a
single XGBoost model family on the two full training views used to choose
hyperparameters:

- men_full: all men's 2015/16 top-league shots;
- women_train: women's non-test shots after the same match-level test split as
  the xG notebook.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

import xgboost_model
from model_contract import ModelSpec, make_feature_frame
from xg_data import (
    CV_FOLDS,
    DEFAULT_OUTPUT_DIR,
    RANDOM_SEED,
    TARGET_COLUMN,
    WOMEN_TEST_SHARE,
    load_xg_test_and_rest,
    make_match_level_cv_splits,
)

SCORERS = {
    "neg_log_loss": "neg_log_loss",
    "neg_brier": "neg_brier_score",
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
}

CV_SEED_OFFSETS = {
    "men_full": 1000,
    "women_train": 3000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--selection-metric", choices=sorted(SCORERS), default="neg_log_loss")
    parser.add_argument("--cv-folds", type=int, default=CV_FOLDS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--quick", action="store_true", help="Run only the first few candidates for smoke tests.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["men_full", "women_train"],
        default=["men_full", "women_train"],
    )
    return parser.parse_args()


def tune_model(
    spec: ModelSpec,
    dataset: str,
    frame: pd.DataFrame,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    selection_metric: str,
) -> tuple[GridSearchCV, pd.DataFrame]:
    x_frame = make_feature_frame(frame)
    y_frame = frame[TARGET_COLUMN].astype(int)
    search = GridSearchCV(
        estimator=spec.estimator,
        param_grid=spec.param_grid,
        scoring=SCORERS,
        refit=selection_metric,
        cv=cv_splits,
        n_jobs=spec.search_n_jobs,
        return_train_score=False,
        verbose=1,
    )
    search.fit(x_frame, y_frame)
    cv_results = pd.DataFrame(search.cv_results_)
    cv_results.insert(0, "dataset", dataset)
    cv_results.insert(0, "model_family", spec.name)
    print(f"[selected] {dataset}: params={search.best_params_} {selection_metric}={search.best_score_:.6f}")
    return search, cv_results


def json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_xg_test_and_rest()
    dataset_frames = {
        "men_full": data.men_full,
        "women_train": data.women_train,
    }
    print("\n[split summary]")
    print(pd.DataFrame(data.split_summary).T.to_string())

    spec = xgboost_model.get_model_spec(quick=args.quick)
    selected_candidates: dict[str, dict] = {}
    cv_result_frames: list[pd.DataFrame] = []
    cv_split_counts: dict[str, int] = {}

    for dataset in args.datasets:
        frame = dataset_frames[dataset]
        cv_splits = make_match_level_cv_splits(frame, args.cv_folds, args.seed + CV_SEED_OFFSETS[dataset], dataset)
        cv_split_counts[dataset] = len(cv_splits)
        search, cv_results = tune_model(spec, dataset, frame, cv_splits, args.selection_metric)
        selected_candidates[dataset] = {
            "best_params": search.best_params_,
            "best_score": float(search.best_score_),
            "best_index": int(search.best_index_),
        }
        cv_result_frames.append(cv_results)

    results = {
        "model_family": spec.name,
        "selection_metric": args.selection_metric,
        "candidate_count": len(spec.param_grid),
        "quick": args.quick,
        "cv_folds": args.cv_folds,
        "cv_split_counts": cv_split_counts,
        "data_seed": RANDOM_SEED,
        "cv_seed": args.seed,
        "women_test_share": WOMEN_TEST_SHARE,
        "split_summary": data.split_summary,
        "selected_candidates": selected_candidates,
    }

    results_path = args.output_dir / "xgboost_xg_tuning_results.json"
    cv_results_path = args.output_dir / "validation_candidate_metrics.csv"
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    pd.concat(cv_result_frames, ignore_index=True).to_csv(cv_results_path, index=False)

    print(f"\n[wrote] {results_path}")
    print(f"[wrote] {cv_results_path}")


if __name__ == "__main__":
    main()
