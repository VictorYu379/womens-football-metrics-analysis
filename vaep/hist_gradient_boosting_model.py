"""Histogram-gradient-boosting VAEP model family.

This is the first VAEP model family because the current VAEP notebook trains
``HistGradientBoostingClassifier`` models for both ``scores`` and ``concedes``.
"""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from model_contract import ModelSpec, make_preprocessor

MODEL_FAMILY = "hist_gradient_boosting"
# Keep the default serial on Windows: VAEP has millions of action rows, and
# joblib process parallelism copies the full feature matrix per worker.
HGB_GRIDSEARCH_JOBS = 1


def _candidate(**params: object) -> dict:
    return {f"model__{key}": [value] for key, value in params.items()}


def _candidate_grid() -> list[dict]:
    """Return explicit VAEP HGB candidates centered on the notebook baseline."""
    return [
        _candidate(max_iter=300, learning_rate=0.050, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0.00),
        _candidate(max_iter=220, learning_rate=0.070, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0.00),
        _candidate(max_iter=500, learning_rate=0.030, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0.00),
        _candidate(max_iter=300, learning_rate=0.050, max_leaf_nodes=15, min_samples_leaf=25, l2_regularization=0.00),
        _candidate(max_iter=450, learning_rate=0.035, max_leaf_nodes=15, min_samples_leaf=30, l2_regularization=0.10),
        _candidate(max_iter=300, learning_rate=0.050, max_leaf_nodes=63, min_samples_leaf=20, l2_regularization=0.10),
        _candidate(max_iter=550, learning_rate=0.025, max_leaf_nodes=31, min_samples_leaf=35, l2_regularization=0.10),
        _candidate(max_iter=350, learning_rate=0.045, max_leaf_nodes=31, min_samples_leaf=50, l2_regularization=0.50),
        _candidate(max_iter=650, learning_rate=0.020, max_leaf_nodes=63, min_samples_leaf=35, l2_regularization=0.50),
        _candidate(max_iter=400, learning_rate=0.035, max_leaf_nodes=63, min_samples_leaf=50, l2_regularization=1.00),
    ]


def get_model_spec(quick: bool = False) -> ModelSpec:
    full_candidates = _candidate_grid()
    quick_candidates = full_candidates[:2]
    estimator = Pipeline(
        [
            ("preprocess", make_preprocessor()),
            (
                "model",
                HistGradientBoostingClassifier(
                    random_state=42,
                ),
            ),
        ]
    )
    return ModelSpec(
        name=MODEL_FAMILY,
        estimator=estimator,
        param_grid=quick_candidates if quick else full_candidates,
        search_n_jobs=HGB_GRIDSEARCH_JOBS,
    )
