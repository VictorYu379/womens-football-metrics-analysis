"""XGBoost xA model family."""

from __future__ import annotations

from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from model_contract import ModelSpec, make_preprocessor

MODEL_FAMILY = "xgboost"
XGB_THREADS_PER_FIT = 4
XGB_GRIDSEARCH_JOBS = 2


def _candidate(**params: object) -> dict:
    return {f"model__{key}": [value] for key, value in params.items()}


def get_model_spec(quick: bool = False) -> ModelSpec:
    quick_candidates = [
        _candidate(n_estimators=300, max_depth=3, learning_rate=0.035, min_child_weight=15, subsample=0.90, colsample_bytree=0.90, reg_lambda=8.0, reg_alpha=0.15),
        _candidate(n_estimators=450, max_depth=3, learning_rate=0.030, min_child_weight=20, subsample=0.92, colsample_bytree=0.92, reg_lambda=12.0, reg_alpha=0.30),
    ]
    full_candidates = quick_candidates + [
        _candidate(n_estimators=250, max_depth=3, learning_rate=0.050, min_child_weight=12, subsample=0.90, colsample_bytree=0.90, reg_lambda=6.0, reg_alpha=0.10),
        _candidate(n_estimators=600, max_depth=3, learning_rate=0.020, min_child_weight=25, subsample=0.92, colsample_bytree=0.92, reg_lambda=14.0, reg_alpha=0.50),
        _candidate(n_estimators=350, max_depth=4, learning_rate=0.035, min_child_weight=10, subsample=0.90, colsample_bytree=0.88, reg_lambda=8.0, reg_alpha=0.20),
        _candidate(n_estimators=500, max_depth=4, learning_rate=0.025, min_child_weight=12, subsample=0.90, colsample_bytree=0.88, reg_lambda=10.0, reg_alpha=0.30),
        _candidate(n_estimators=700, max_depth=4, learning_rate=0.015, min_child_weight=15, subsample=0.92, colsample_bytree=0.90, reg_lambda=12.0, reg_alpha=0.40),
        _candidate(n_estimators=350, max_depth=5, learning_rate=0.030, min_child_weight=18, subsample=0.86, colsample_bytree=0.84, reg_lambda=12.0, reg_alpha=0.40),
        _candidate(n_estimators=500, max_depth=5, learning_rate=0.020, min_child_weight=20, subsample=0.86, colsample_bytree=0.84, reg_lambda=16.0, reg_alpha=0.60),
        _candidate(n_estimators=300, max_depth=6, learning_rate=0.025, min_child_weight=25, subsample=0.84, colsample_bytree=0.82, reg_lambda=20.0, reg_alpha=0.80),
        _candidate(n_estimators=450, max_depth=3, learning_rate=0.030, min_child_weight=40, subsample=0.95, colsample_bytree=0.95, reg_lambda=20.0, reg_alpha=1.00),
        _candidate(n_estimators=650, max_depth=4, learning_rate=0.018, min_child_weight=30, subsample=0.88, colsample_bytree=0.86, reg_lambda=24.0, reg_alpha=1.20),
    ]

    estimator = Pipeline(
        [
            ("preprocess", make_preprocessor(scale_numeric=False)),
            (
                "model",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=42,
                    n_jobs=XGB_THREADS_PER_FIT,
                ),
            ),
        ]
    )
    return ModelSpec(
        name=MODEL_FAMILY,
        estimator=estimator,
        param_grid=quick_candidates if quick else full_candidates,
        search_n_jobs=XGB_GRIDSEARCH_JOBS,
    )
