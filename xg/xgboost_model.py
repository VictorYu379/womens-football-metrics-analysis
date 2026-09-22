"""XGBoost xG model family and candidate grid."""

from __future__ import annotations

from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from model_contract import ModelSpec, make_preprocessor

MODEL_FAMILY = "xgboost"
XGB_THREADS_PER_FIT = 4
XGB_GRIDSEARCH_JOBS = 2


def _candidate(**params: object) -> dict:
    return {f"model__{key}": [value] for key, value in params.items()}


def _candidate_grid() -> list[dict]:
    """Return explicit xG candidates rather than a broad cartesian grid."""
    return [
        _candidate(n_estimators=200, max_depth=2, learning_rate=0.080, min_child_weight=1, subsample=1.00, colsample_bytree=1.00, reg_lambda=1.0, reg_alpha=0.00),
        _candidate(n_estimators=250, max_depth=2, learning_rate=0.060, min_child_weight=3, subsample=0.98, colsample_bytree=0.98, reg_lambda=1.5, reg_alpha=0.00),
        _candidate(n_estimators=350, max_depth=2, learning_rate=0.045, min_child_weight=5, subsample=0.95, colsample_bytree=0.95, reg_lambda=2.0, reg_alpha=0.02),
        _candidate(n_estimators=500, max_depth=2, learning_rate=0.030, min_child_weight=8, subsample=0.92, colsample_bytree=0.92, reg_lambda=3.0, reg_alpha=0.05),
        _candidate(n_estimators=700, max_depth=2, learning_rate=0.020, min_child_weight=12, subsample=0.90, colsample_bytree=0.90, reg_lambda=4.0, reg_alpha=0.10),
        _candidate(n_estimators=220, max_depth=3, learning_rate=0.070, min_child_weight=1, subsample=1.00, colsample_bytree=1.00, reg_lambda=1.0, reg_alpha=0.00),
        _candidate(n_estimators=300, max_depth=3, learning_rate=0.050, min_child_weight=3, subsample=0.98, colsample_bytree=0.98, reg_lambda=1.5, reg_alpha=0.00),
        _candidate(n_estimators=400, max_depth=3, learning_rate=0.040, min_child_weight=5, subsample=0.95, colsample_bytree=0.95, reg_lambda=2.0, reg_alpha=0.02),
        _candidate(n_estimators=550, max_depth=3, learning_rate=0.028, min_child_weight=8, subsample=0.92, colsample_bytree=0.92, reg_lambda=3.0, reg_alpha=0.05),
        _candidate(n_estimators=750, max_depth=3, learning_rate=0.020, min_child_weight=12, subsample=0.90, colsample_bytree=0.90, reg_lambda=4.0, reg_alpha=0.10),
        _candidate(n_estimators=260, max_depth=4, learning_rate=0.055, min_child_weight=3, subsample=0.96, colsample_bytree=0.96, reg_lambda=2.0, reg_alpha=0.02),
        _candidate(n_estimators=380, max_depth=4, learning_rate=0.040, min_child_weight=5, subsample=0.94, colsample_bytree=0.94, reg_lambda=3.0, reg_alpha=0.05),
        _candidate(n_estimators=520, max_depth=4, learning_rate=0.030, min_child_weight=8, subsample=0.92, colsample_bytree=0.92, reg_lambda=4.0, reg_alpha=0.10),
        _candidate(n_estimators=700, max_depth=4, learning_rate=0.022, min_child_weight=12, subsample=0.90, colsample_bytree=0.90, reg_lambda=5.0, reg_alpha=0.15),
        _candidate(n_estimators=350, max_depth=5, learning_rate=0.035, min_child_weight=8, subsample=0.90, colsample_bytree=0.90, reg_lambda=5.0, reg_alpha=0.15),
        _candidate(n_estimators=550, max_depth=5, learning_rate=0.025, min_child_weight=12, subsample=0.88, colsample_bytree=0.88, reg_lambda=6.0, reg_alpha=0.20),
        _candidate(n_estimators=800, max_depth=5, learning_rate=0.018, min_child_weight=18, subsample=0.86, colsample_bytree=0.86, reg_lambda=8.0, reg_alpha=0.30),
        _candidate(n_estimators=300, max_depth=3, learning_rate=0.045, min_child_weight=15, subsample=0.92, colsample_bytree=0.92, reg_lambda=8.0, reg_alpha=0.20),
        _candidate(n_estimators=450, max_depth=3, learning_rate=0.030, min_child_weight=20, subsample=0.90, colsample_bytree=0.90, reg_lambda=10.0, reg_alpha=0.30),
        _candidate(n_estimators=650, max_depth=3, learning_rate=0.022, min_child_weight=25, subsample=0.88, colsample_bytree=0.88, reg_lambda=12.0, reg_alpha=0.40),
        _candidate(n_estimators=400, max_depth=4, learning_rate=0.032, min_child_weight=20, subsample=0.90, colsample_bytree=0.88, reg_lambda=12.0, reg_alpha=0.40),
        _candidate(n_estimators=650, max_depth=4, learning_rate=0.020, min_child_weight=25, subsample=0.88, colsample_bytree=0.86, reg_lambda=16.0, reg_alpha=0.60),
        _candidate(n_estimators=450, max_depth=2, learning_rate=0.035, min_child_weight=25, subsample=0.95, colsample_bytree=0.95, reg_lambda=8.0, reg_alpha=0.20),
        _candidate(n_estimators=650, max_depth=2, learning_rate=0.024, min_child_weight=35, subsample=0.94, colsample_bytree=0.94, reg_lambda=12.0, reg_alpha=0.40),
    ]


def get_model_spec(quick: bool = False) -> ModelSpec:
    full_candidates = _candidate_grid()
    quick_candidates = full_candidates[:4]
    estimator = Pipeline(
        [
            ("preprocess", make_preprocessor()),
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
