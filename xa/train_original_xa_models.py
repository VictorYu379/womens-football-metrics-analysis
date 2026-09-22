"""Train and compare original-definition xA model families.

The intentionally small orchestration here is:
1. load match-level train/test data from xa_data.py;
2. build model-family sklearn estimators;
3. let GridSearchCV tune hyperparameters and refit the best model;
4. evaluate men-trained and women-trained models on the isolated women test set.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV

import logistic_model
import xgboost_model
from model_contract import ModelSpec, make_feature_frame
from xa_data import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    CV_FOLDS,
    DEFAULT_OUTPUT_DIR,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    RANDOM_SEED,
    TARGET_COLUMN,
    VALIDATION_SHARE,
    WOMEN_TEST_SHARE,
    load_xa_test_and_rest,
    make_match_level_cv_splits,
    summarize_split,
)

PREDICTION_EPSILON = 1e-7
MODEL_REGISTRY = {
    logistic_model.MODEL_FAMILY: logistic_model.get_model_spec,
    xgboost_model.MODEL_FAMILY: xgboost_model.get_model_spec,
}
OUTPUT_ARTIFACT_NAMES = {
    "xa_original_training_results.json",
    "validation_candidate_metrics.csv",
    "women_final_test_xa_predictions.csv",
}
OUTPUT_ARTIFACT_GLOBS = ("*_xa_model.joblib",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train original-definition xA models.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-family", choices=["logistic", "xgboost", "all"], default="all")
    parser.add_argument("--validation-share", type=float, default=VALIDATION_SHARE)
    parser.add_argument("--tuning-mode", choices=["cv", "holdout"], default="cv")
    parser.add_argument("--selection-metric", choices=["xa_accuracy", "average_precision", "roc_auc", "neg_log_loss", "neg_brier"], default="neg_log_loss")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--quick", action="store_true", help="Use the smaller candidate grid for a fast smoke run.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing xA training artifacts in --output-dir.")
    return parser.parse_args()


def model_families_from_arg(model_family: str) -> list[str]:
    return list(MODEL_REGISTRY) if model_family == "all" else [model_family]


def output_artifacts(output_dir: Path) -> list[Path]:
    if not output_dir.exists():
        return []
    artifacts = [output_dir / name for name in OUTPUT_ARTIFACT_NAMES if (output_dir / name).exists()]
    for pattern in OUTPUT_ARTIFACT_GLOBS:
        artifacts.extend(output_dir.glob(pattern))
    return sorted(set(artifacts))


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    artifacts = output_artifacts(output_dir)
    if artifacts and not overwrite:
        names = ", ".join(path.name for path in artifacts[:6])
        raise FileExistsError(f"{output_dir} already contains xA artifacts ({names}). Use a fresh directory or --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for artifact in artifacts:
            artifact.unlink()


def _git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=Path(__file__).resolve().parents[1], check=True, text=True, capture_output=True)
    except Exception:
        return None
    return result.stdout.strip()


def code_metadata() -> dict:
    status = _git(["status", "--short"])
    return {
        "training_script": str(Path(__file__).resolve()),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_dirty": bool(status),
        "git_status_short": status,
    }


def make_holdout_split(frame: pd.DataFrame, share: float, seed: int, name: str) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[np.ndarray, np.ndarray]]]:
    games = np.array(sorted(frame["game_id"].unique()))
    split_size = max(1, min(len(games) - 1, int(round(len(games) * share))))
    positives_by_game = frame.groupby("game_id")[TARGET_COLUMN].sum()
    for offset in range(1000):
        rng = np.random.default_rng(seed + offset)
        validation_games = set(int(game_id) for game_id in rng.choice(games, size=split_size, replace=False))
        train_games = set(int(game_id) for game_id in games) - validation_games
        if positives_by_game.reindex(list(validation_games), fill_value=0).sum() > 0 and positives_by_game.reindex(list(train_games), fill_value=0).sum() > 0:
            is_validation = frame["game_id"].isin(validation_games).to_numpy()
            train_indices = np.flatnonzero(~is_validation)
            validation_indices = np.flatnonzero(is_validation)
            return frame.iloc[train_indices].copy(), frame.iloc[validation_indices].copy(), [(train_indices, validation_indices)]
    raise ValueError(f"Could not make a positive-label holdout split for {name}.")


def _positive_probability(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities)
    if probabilities.ndim == 2:
        return probabilities[:, 1]
    return probabilities


def xa_accuracy_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    probabilities = np.clip(_positive_probability(probabilities), PREDICTION_EPSILON, 1.0 - PREDICTION_EPSILON)
    y_true = np.asarray(y_true, dtype=int)
    prevalence = float(y_true.mean()) if len(y_true) else 0.0
    if prevalence <= 0.0 or prevalence >= 1.0:
        return float("nan")
    brier = brier_score_loss(y_true, probabilities)
    baseline = brier_score_loss(y_true, np.full_like(probabilities, prevalence, dtype=float))
    return float(1.0 - brier / baseline)


def neg_brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    probabilities = np.clip(_positive_probability(probabilities), PREDICTION_EPSILON, 1.0 - PREDICTION_EPSILON)
    return -float(brier_score_loss(y_true, probabilities))


def scorers() -> dict:
    return {
        "xa_accuracy": make_scorer(xa_accuracy_score, response_method="predict_proba"),
        "average_precision": "average_precision",
        "roc_auc": "roc_auc",
        "neg_log_loss": "neg_log_loss",
        "neg_brier": make_scorer(neg_brier_score, response_method="predict_proba"),
    }


def probability_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), PREDICTION_EPSILON, 1.0 - PREDICTION_EPSILON)
    prevalence = float(y_true.mean()) if len(y_true) else 0.0
    baseline = np.full_like(probabilities, prevalence, dtype=float)
    brier = float(brier_score_loss(y_true, probabilities))
    baseline_brier = float(brier_score_loss(y_true, baseline)) if 0.0 < prevalence < 1.0 else float("nan")
    return {
        "rows": int(len(y_true)),
        "positives": int(y_true.sum()),
        "prevalence": prevalence,
        "predicted_xa_sum": float(probabilities.sum()),
        "observed_assists": int(y_true.sum()),
        "calibration_error_sum": float(probabilities.sum() - y_true.sum()),
        "brier": brier,
        "baseline_brier": baseline_brier,
        "xa_accuracy": float(1.0 - brier / baseline_brier) if baseline_brier and not pd.isna(baseline_brier) else float("nan"),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "average_precision": float(average_precision_score(y_true, probabilities)) if y_true.sum() else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if 0 < y_true.sum() < len(y_true) else float("nan"),
    }


def tune_model(spec: ModelSpec, dataset: str, frame: pd.DataFrame, cv_splits: list[tuple[np.ndarray, np.ndarray]], selection_metric: str) -> tuple[GridSearchCV, pd.DataFrame]:
    x_frame = make_feature_frame(frame)
    y_frame = frame[TARGET_COLUMN].astype(int)
    search = GridSearchCV(
        estimator=spec.estimator,
        param_grid=spec.param_grid,
        scoring=scorers(),
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
    print(f"[selected] {spec.name} {dataset}: rank=1 params={search.best_params_} {selection_metric}={search.best_score_:.6f}")
    return search, cv_results


def predict_xa(search: GridSearchCV, frame: pd.DataFrame) -> np.ndarray:
    return np.clip(search.best_estimator_.predict_proba(make_feature_frame(frame))[:, 1], 0.0, 1.0)


def prediction_families(prediction_columns: dict[str, np.ndarray]) -> list[str]:
    families: set[str] = set()
    for column in prediction_columns:
        if column.endswith("_men_model_xa"):
            families.add(column.removesuffix("_men_model_xa"))
        elif column.endswith("_women_model_xa"):
            families.add(column.removesuffix("_women_model_xa"))
    return sorted(families)


def write_predictions(output_dir: Path, women_test: pd.DataFrame, prediction_columns: dict[str, np.ndarray]) -> None:
    base_columns = ["event_id", "game_id", "player_id", "player_name", "team_name", "position_name", "league", "start_x", "start_y", "end_x", "end_y", "pass_type", "pass_height", "pass_body_part", "pass_technique", "play_pattern", "pass_context", "end_zone", TARGET_COLUMN]
    predictions = women_test[[column for column in base_columns if column in women_test.columns]].copy()
    for column_name, values in prediction_columns.items():
        predictions[column_name] = values
    for family in prediction_families(prediction_columns):
        men_column = f"{family}_men_model_xa"
        women_column = f"{family}_women_model_xa"
        if men_column in predictions and women_column in predictions:
            predictions[f"{family}_xa_delta_men_minus_women"] = predictions[men_column] - predictions[women_column]
    predictions.to_csv(output_dir / "women_final_test_xa_predictions.csv", index=False)


def json_default(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")



def main() -> None:
    args = parse_args()
    prepare_output_dir(args.output_dir, args.overwrite)

    data = load_xa_test_and_rest()
    men_frame = data.men_rest
    women_frame = data.women_rest
    women_test = data.women_test

    if args.tuning_mode == "cv":
        men_cv = make_match_level_cv_splits(men_frame, CV_FOLDS, args.seed + 1000, "men")
        women_cv = make_match_level_cv_splits(women_frame, CV_FOLDS, args.seed + 2000, "women")
        split_summary = {
            "men_cv_modeling": summarize_split(men_frame),
            "women_cv_modeling_non_test": summarize_split(women_frame),
            "women_final_test": summarize_split(women_test),
        }
    else:
        men_train, men_validation, men_cv = make_holdout_split(men_frame, args.validation_share, args.seed + 1000, "men")
        women_train, women_validation, women_cv = make_holdout_split(women_frame, args.validation_share, args.seed + 2000, "women")
        split_summary = {
            "men_train": summarize_split(men_train),
            "men_validation": summarize_split(men_validation),
            "women_train": summarize_split(women_train),
            "women_validation": summarize_split(women_validation),
            "women_final_test": summarize_split(women_test),
        }

    print("\n[split summary]")
    print(pd.DataFrame(split_summary).T.to_string())

    final_test_metrics: dict[str, dict] = {}
    selected_candidates: dict[str, dict] = {}
    prediction_columns: dict[str, np.ndarray] = {}
    cv_result_frames: list[pd.DataFrame] = []
    saved_models: dict[str, str] = {}
    y_test = women_test[TARGET_COLUMN].astype(int).to_numpy()

    for family in model_families_from_arg(args.model_family):
        spec = MODEL_REGISTRY[family](quick=args.quick)
        selected_candidates[family] = {}

        men_search, men_cv_results = tune_model(spec, "men", men_frame, men_cv, args.selection_metric)
        women_search, women_cv_results = tune_model(spec, "women", women_frame, women_cv, args.selection_metric)
        cv_result_frames.extend([men_cv_results, women_cv_results])

        men_probabilities = predict_xa(men_search, women_test)
        women_probabilities = predict_xa(women_search, women_test)
        final_test_metrics[f"{family}_men_model_on_women_final_test"] = probability_metrics(y_test, men_probabilities)
        final_test_metrics[f"{family}_women_model_on_women_final_test"] = probability_metrics(y_test, women_probabilities)
        prediction_columns[f"{family}_men_model_xa"] = men_probabilities
        prediction_columns[f"{family}_women_model_xa"] = women_probabilities

        selected_candidates[family]["men"] = {"best_params": men_search.best_params_, "best_score": float(men_search.best_score_)}
        selected_candidates[family]["women"] = {"best_params": women_search.best_params_, "best_score": float(women_search.best_score_)}

        men_path = args.output_dir / f"men_{family}_xa_model.joblib"
        women_path = args.output_dir / f"women_{family}_xa_model.joblib"
        joblib.dump(men_search.best_estimator_, men_path)
        joblib.dump(women_search.best_estimator_, women_path)
        saved_models[f"men_{family}"] = str(men_path)
        saved_models[f"women_{family}"] = str(women_path)

    print("\n[final women test metrics]")
    print(pd.DataFrame(final_test_metrics).T[["rows", "positives", "predicted_xa_sum", "observed_assists", "xa_accuracy", "average_precision", "roc_auc", "brier", "log_loss"]].to_string())

    results = {
        "definition": "xA = P(completed pass becomes a goal assist)",
        "target_column": TARGET_COLUMN,
        "model_family": args.model_family,
        "model_families_trained": model_families_from_arg(args.model_family),
        "selection_metric": args.selection_metric,
        "tuning_mode": args.tuning_mode,
        "cv_folds": CV_FOLDS if args.tuning_mode == "cv" else None,
        "data_seed": RANDOM_SEED,
        "cv_seed": args.seed,
        "quick": args.quick,
        "women_test_share": WOMEN_TEST_SHARE,
        "validation_share": args.validation_share if args.tuning_mode == "holdout" else None,
        "split_summary": split_summary,
        "selected_candidates": selected_candidates,
        "final_test_metrics": final_test_metrics,
        "saved_models": saved_models,
        "feature_columns": {"numeric": NUMERIC_FEATURES, "boolean": BOOLEAN_FEATURES, "categorical": CATEGORICAL_FEATURES},
        "code_metadata": code_metadata(),
    }

    (args.output_dir / "xa_original_training_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    pd.concat(cv_result_frames, ignore_index=True).to_csv(args.output_dir / "validation_candidate_metrics.csv", index=False)
    write_predictions(args.output_dir, women_test, prediction_columns)
    print(f"\n[wrote] {args.output_dir}")


if __name__ == "__main__":
    main()
