"""Tune and train VAEP probability model families.

VAEP uses two binary classifiers: ``scores`` and ``concedes``. This script
keeps those targets separate while sharing the same prepared action rows,
match-level CV policy, model-family registry, and output shape used by the xG
and xA training scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV, ParameterGrid

import hist_gradient_boosting_model
from model_contract import ModelSpec, make_feature_frame, silence_common_sklearn_warnings
from vaep_data import (
    CV_FOLDS,
    DEFAULT_OUTPUT_DIR,
    N_PREVIOUS_ACTIONS,
    RANDOM_SEED,
    TARGET_COLUMNS,
    VAEP_LABEL_HORIZON,
    VALIDATION_SHARE,
    WOMEN_TEST_SHARE,
    load_vaep_test_and_rest,
    make_match_level_cv_splits,
    summarize_split,
)

PREDICTION_EPSILON = 1e-7
MODEL_REGISTRY = {
    hist_gradient_boosting_model.MODEL_FAMILY: hist_gradient_boosting_model.get_model_spec,
}
OUTPUT_ARTIFACT_NAMES = {
    "progress_log.csv",
    "vaep_training_results.json",
    "validation_candidate_metrics.csv",
    "women_final_test_vaep_predictions.csv",
}
OUTPUT_ARTIFACT_GLOBS = ("*_vaep_model.joblib", "*_vaep_models.joblib")
PROGRESS_LOG_NAME = "progress_log.csv"
PROGRESS_LOG_COLUMNS = [
    "timestamp",
    "event",
    "status",
    "model_family",
    "target",
    "dataset",
    "candidates",
    "folds",
    "fits_total",
    "elapsed_seconds",
    "best_score",
    "best_params",
    "message",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune and train VAEP probability models.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-family", choices=[*MODEL_REGISTRY, "all"], default="hist_gradient_boosting")
    parser.add_argument("--target", choices=[*TARGET_COLUMNS, "both"], default="both")
    parser.add_argument("--validation-share", type=float, default=VALIDATION_SHARE)
    parser.add_argument("--tuning-mode", choices=["cv", "holdout"], default="cv")
    parser.add_argument(
        "--selection-metric",
        choices=["average_precision", "roc_auc", "neg_log_loss", "neg_brier"],
        default="neg_log_loss",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--search-n-jobs",
        type=int,
        default=None,
        help="Override GridSearchCV n_jobs. The HGB family defaults to 1 for Windows memory safety.",
    )
    parser.add_argument(
        "--grid-verbose",
        type=int,
        default=3,
        help="GridSearchCV verbose level. Default 3 prints candidate/fold progress in the console.",
    )
    parser.add_argument("--quick", action="store_true", help="Use a smaller candidate grid for a fast smoke run.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing VAEP training artifacts in --output-dir.")
    return parser.parse_args()


def model_families_from_arg(model_family: str) -> list[str]:
    return list(MODEL_REGISTRY) if model_family == "all" else [model_family]


def target_columns_from_arg(target: str) -> list[str]:
    return list(TARGET_COLUMNS) if target == "both" else [target]


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
        raise FileExistsError(f"{output_dir} already contains VAEP artifacts ({names}). Use a fresh directory or --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for artifact in artifacts:
            artifact.unlink()


def _git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            text=True,
            capture_output=True,
        )
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


def append_progress_row(output_dir: Path, **row: object) -> None:
    """Append one durable progress event for long VAEP grid-search runs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / PROGRESS_LOG_NAME
    full_row = {column: "" for column in PROGRESS_LOG_COLUMNS}
    full_row.update(row)
    full_row["timestamp"] = datetime.now().isoformat(timespec="seconds")
    for key in ("best_params", "message"):
        if isinstance(full_row.get(key), (dict, list, tuple)):
            full_row[key] = json.dumps(full_row[key], ensure_ascii=False, default=json_default)

    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROGRESS_LOG_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(full_row)

    console_parts = [
        f"[{full_row['timestamp']}]",
        "[progress]",
        str(full_row.get("event", "")),
        str(full_row.get("status", "")),
    ]
    context = " ".join(
        str(full_row.get(column, ""))
        for column in ("model_family", "dataset", "target")
        if full_row.get(column, "") != ""
    )
    if context:
        console_parts.append(context)
    if full_row.get("fits_total", "") != "":
        console_parts.append(f"fits={full_row['fits_total']}")
    if full_row.get("elapsed_seconds", "") != "":
        console_parts.append(f"elapsed={full_row['elapsed_seconds']}s")
    if full_row.get("best_score", "") != "":
        console_parts.append(f"best_score={full_row['best_score']}")
    if full_row.get("message", "") != "":
        console_parts.append(f"| {full_row['message']}")
    print(" ".join(console_parts), flush=True)


def make_holdout_split(
    frame: pd.DataFrame,
    share: float,
    seed: int,
    name: str,
    target_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[np.ndarray, np.ndarray]]]:
    games = np.array(sorted(frame["game_id"].unique()))
    split_size = max(1, min(len(games) - 1, int(round(len(games) * share))))
    positives_by_game = frame.groupby("game_id")[target_column].sum()
    rows_by_game = frame.groupby("game_id").size()

    for offset in range(1000):
        rng = np.random.default_rng(seed + offset)
        validation_games = set(int(game_id) for game_id in rng.choice(games, size=split_size, replace=False))
        train_games = set(int(game_id) for game_id in games) - validation_games
        validation_positives = int(positives_by_game.reindex(list(validation_games), fill_value=0).sum())
        train_positives = int(positives_by_game.reindex(list(train_games), fill_value=0).sum())
        validation_rows = int(rows_by_game.reindex(list(validation_games), fill_value=0).sum())
        train_rows = int(rows_by_game.reindex(list(train_games), fill_value=0).sum())
        if 0 < validation_positives < validation_rows and 0 < train_positives < train_rows:
            is_validation = frame["game_id"].isin(validation_games).to_numpy()
            train_indices = np.flatnonzero(~is_validation)
            validation_indices = np.flatnonzero(is_validation)
            return frame.iloc[train_indices].copy(), frame.iloc[validation_indices].copy(), [(train_indices, validation_indices)]
    raise ValueError(f"Could not make a binary-label holdout split for {name} {target_column}.")


def _positive_probability(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities)
    if probabilities.ndim == 2:
        return probabilities[:, 1]
    return probabilities


def neg_brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    probabilities = np.clip(_positive_probability(probabilities), PREDICTION_EPSILON, 1.0 - PREDICTION_EPSILON)
    return -float(brier_score_loss(y_true, probabilities))


def scorers() -> dict:
    return {
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
        "predicted_positive_sum": float(probabilities.sum()),
        "calibration_error_sum": float(probabilities.sum() - y_true.sum()),
        "brier": brier,
        "baseline_brier": baseline_brier,
        "brier_skill": float(1.0 - brier / baseline_brier) if baseline_brier and not pd.isna(baseline_brier) else float("nan"),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "average_precision": float(average_precision_score(y_true, probabilities)) if y_true.sum() else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if 0 < y_true.sum() < len(y_true) else float("nan"),
    }


def tune_model(
    spec: ModelSpec,
    target_column: str,
    dataset: str,
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    selection_metric: str,
    output_dir: Path,
    grid_verbose: int,
) -> tuple[GridSearchCV, pd.DataFrame]:
    x_frame = make_feature_frame(frame, feature_columns)
    y_frame = frame[target_column].astype(int)
    candidate_count = len(list(ParameterGrid(spec.param_grid)))
    fold_count = len(cv_splits)
    fits_total = candidate_count * fold_count
    append_progress_row(
        output_dir,
        event="grid_search",
        status="started",
        model_family=spec.name,
        target=target_column,
        dataset=dataset,
        candidates=candidate_count,
        folds=fold_count,
        fits_total=fits_total,
        message=f"rows={len(frame):,}; features={len(feature_columns):,}; selection_metric={selection_metric}",
    )
    search = GridSearchCV(
        estimator=spec.estimator,
        param_grid=spec.param_grid,
        scoring=scorers(),
        refit=selection_metric,
        cv=cv_splits,
        n_jobs=spec.search_n_jobs,
        return_train_score=False,
        verbose=grid_verbose,
    )
    started_at = time.perf_counter()
    try:
        search.fit(x_frame, y_frame)
    except Exception as error:
        append_progress_row(
            output_dir,
            event="grid_search",
            status="failed",
            model_family=spec.name,
            target=target_column,
            dataset=dataset,
            candidates=candidate_count,
            folds=fold_count,
            fits_total=fits_total,
            elapsed_seconds=round(time.perf_counter() - started_at, 1),
            message=repr(error),
        )
        raise
    cv_results = pd.DataFrame(search.cv_results_)
    cv_results.insert(0, "dataset", dataset)
    cv_results.insert(0, "target", target_column)
    cv_results.insert(0, "model_family", spec.name)
    print(f"[best] {spec.name} {dataset} {target_column}: {selection_metric}={search.best_score_:.6f}")
    print(search.best_params_)
    append_progress_row(
        output_dir,
        event="grid_search",
        status="completed",
        model_family=spec.name,
        target=target_column,
        dataset=dataset,
        candidates=candidate_count,
        folds=fold_count,
        fits_total=fits_total,
        elapsed_seconds=round(time.perf_counter() - started_at, 1),
        best_score=float(search.best_score_),
        best_params=search.best_params_,
    )
    return search, cv_results


def predict_probability(model: object, frame: pd.DataFrame, feature_columns: tuple[str, ...]) -> np.ndarray:
    features = make_feature_frame(frame, feature_columns)
    return np.clip(model.predict_proba(features)[:, 1], 0.0, 1.0)


def write_predictions(output_dir: Path, women_test: pd.DataFrame, prediction_columns: dict[str, np.ndarray]) -> None:
    metadata_columns = [
        "game_id",
        "action_id",
        "player_id",
        "team_id",
        "type_name",
        "result_name",
        "start_x",
        "start_y",
        "end_x",
        "end_y",
        *TARGET_COLUMNS,
    ]
    prediction_frame = women_test[[column for column in metadata_columns if column in women_test.columns]].copy()
    for column, values in prediction_columns.items():
        prediction_frame[column] = values
    prediction_frame.to_csv(output_dir / "women_final_test_vaep_predictions.csv", index=False)


def assert_reserved_test_set_is_excluded(women_modeling_frame: pd.DataFrame, women_test_frame: pd.DataFrame) -> None:
    """Guarantee the final women test matches are not used for tuning or fitting."""
    modeling_games = set(int(game_id) for game_id in women_modeling_frame["game_id"].dropna().unique())
    test_games = set(int(game_id) for game_id in women_test_frame["game_id"].dropna().unique())
    overlap = modeling_games & test_games
    if overlap:
        sample = sorted(overlap)[:10]
        raise AssertionError(f"Reserved women final-test games leaked into VAEP modeling data: {sample}")


def json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def build_cv_plan(
    targets: list[str],
    men_frame: pd.DataFrame,
    women_frame: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]], dict[str, dict]]:
    cv_plan: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = {}
    split_summary: dict[str, dict] = {
        "men_modeling": summarize_split(men_frame),
        "women_modeling_non_test": summarize_split(women_frame),
    }

    for target_column in targets:
        if args.tuning_mode == "cv":
            cv_plan[target_column] = {
                "men_full": make_match_level_cv_splits(men_frame, CV_FOLDS, args.seed + 1000, f"men_full_{target_column}", target_column),
                "women_train": make_match_level_cv_splits(women_frame, CV_FOLDS, args.seed + 2000, f"women_train_{target_column}", target_column),
            }
        else:
            men_train, men_validation, men_cv = make_holdout_split(
                men_frame,
                args.validation_share,
                args.seed + 1000,
                "men_full",
                target_column,
            )
            women_train, women_validation, women_cv = make_holdout_split(
                women_frame,
                args.validation_share,
                args.seed + 2000,
                "women_train",
                target_column,
            )
            cv_plan[target_column] = {"men_full": men_cv, "women_train": women_cv}
            split_summary[f"men_{target_column}_train"] = summarize_split(men_train)
            split_summary[f"men_{target_column}_validation"] = summarize_split(men_validation)
            split_summary[f"women_{target_column}_train"] = summarize_split(women_train)
            split_summary[f"women_{target_column}_validation"] = summarize_split(women_validation)

    return cv_plan, split_summary


def main() -> None:
    args = parse_args()
    silence_common_sklearn_warnings()
    prepare_output_dir(args.output_dir, args.overwrite)
    append_progress_row(args.output_dir, event="run", status="started", message="VAEP training run started")

    data = load_vaep_test_and_rest()
    men_frame = data.men_full
    women_frame = data.women_train
    women_test = data.women_test
    assert_reserved_test_set_is_excluded(women_frame, women_test)
    targets = target_columns_from_arg(args.target)
    cv_plan, split_summary = build_cv_plan(targets, men_frame, women_frame, args)
    split_summary["women_final_test"] = summarize_split(women_test)
    append_progress_row(
        args.output_dir,
        event="data",
        status="ready",
        message={
            "men_full_rows": len(men_frame),
            "women_train_rows": len(women_frame),
            "women_final_test_rows": len(women_test),
            "feature_columns": len(data.feature_columns),
            "targets": targets,
        },
    )

    print("\n[split summary]")
    print(pd.DataFrame(split_summary).T.to_string())

    final_test_metrics: dict[str, dict] = {}
    selected_candidates: dict[str, dict] = {}
    prediction_columns: dict[str, np.ndarray] = {}
    cv_result_frames: list[pd.DataFrame] = []
    saved_models: dict[str, str] = {}
    model_bundles: dict[str, dict[str, object]] = {}

    for family in model_families_from_arg(args.model_family):
        spec = MODEL_REGISTRY[family](quick=args.quick)
        if args.search_n_jobs is not None:
            spec = ModelSpec(
                name=spec.name,
                estimator=spec.estimator,
                param_grid=spec.param_grid,
                search_n_jobs=args.search_n_jobs,
            )
        selected_candidates[family] = {}

        for target_column in targets:
            selected_candidates[family][target_column] = {}
            men_search, men_cv_results = tune_model(
                spec,
                target_column,
                "men_full",
                men_frame,
                data.feature_columns,
                cv_plan[target_column]["men_full"],
                args.selection_metric,
                args.output_dir,
                args.grid_verbose,
            )
            women_search, women_cv_results = tune_model(
                spec,
                target_column,
                "women_train",
                women_frame,
                data.feature_columns,
                cv_plan[target_column]["women_train"],
                args.selection_metric,
                args.output_dir,
                args.grid_verbose,
            )
            cv_result_frames.extend([men_cv_results, women_cv_results])

            y_test = women_test[target_column].astype(int).to_numpy()
            men_probabilities = predict_probability(men_search.best_estimator_, women_test, data.feature_columns)
            women_probabilities = predict_probability(women_search.best_estimator_, women_test, data.feature_columns)
            final_test_metrics[f"{family}_{target_column}_men_full_model_on_women_final_test"] = probability_metrics(y_test, men_probabilities)
            final_test_metrics[f"{family}_{target_column}_women_model_on_women_final_test"] = probability_metrics(y_test, women_probabilities)
            prediction_columns[f"{family}_men_full_model_p_{target_column}"] = men_probabilities
            prediction_columns[f"{family}_women_model_p_{target_column}"] = women_probabilities

            selected_candidates[family][target_column]["men_full"] = {
                "best_params": men_search.best_params_,
                "best_score": float(men_search.best_score_),
            }
            selected_candidates[family][target_column]["women_train"] = {
                "best_params": women_search.best_params_,
                "best_score": float(women_search.best_score_),
            }

            men_path = args.output_dir / f"men_full_{family}_{target_column}_vaep_model.joblib"
            women_path = args.output_dir / f"women_train_{family}_{target_column}_vaep_model.joblib"
            joblib.dump(men_search.best_estimator_, men_path)
            joblib.dump(women_search.best_estimator_, women_path)
            saved_models[f"men_full_{family}_{target_column}"] = str(men_path)
            saved_models[f"women_train_{family}_{target_column}"] = str(women_path)

            model_bundles.setdefault(f"men_full_{family}", {})[target_column] = men_search.best_estimator_
            model_bundles.setdefault(f"women_train_{family}", {})[target_column] = women_search.best_estimator_

    for bundle_name, models in model_bundles.items():
        bundle_path = args.output_dir / f"{bundle_name}_vaep_models.joblib"
        joblib.dump(models, bundle_path)
        saved_models[f"{bundle_name}_bundle"] = str(bundle_path)

    print("\n[final women test metrics]")
    metric_columns = ["rows", "positives", "predicted_positive_sum", "brier_skill", "average_precision", "roc_auc", "brier", "log_loss"]
    print(pd.DataFrame(final_test_metrics).T[metric_columns].to_string())

    results = {
        "definition": f"VAEP probability models: P(scores within {VAEP_LABEL_HORIZON} actions) and P(concedes within {VAEP_LABEL_HORIZON} actions)",
        "target_columns": list(TARGET_COLUMNS),
        "trained_targets": targets,
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
        "n_previous_actions": N_PREVIOUS_ACTIONS,
        "label_horizon": VAEP_LABEL_HORIZON,
        "split_summary": split_summary,
        "selected_candidates": selected_candidates,
        "final_test_metrics": final_test_metrics,
        "saved_models": saved_models,
        "feature_columns": list(data.feature_columns),
        "feature_column_count": len(data.feature_columns),
        "code_metadata": code_metadata(),
    }

    (args.output_dir / "vaep_training_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    pd.concat(cv_result_frames, ignore_index=True).to_csv(args.output_dir / "validation_candidate_metrics.csv", index=False)
    write_predictions(args.output_dir, women_test, prediction_columns)
    append_progress_row(args.output_dir, event="run", status="completed", message="VAEP training run completed")
    print(f"\n[wrote] {args.output_dir}")


if __name__ == "__main__":
    main()
