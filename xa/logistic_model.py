"""Logistic-regression xA model family."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from model_contract import ModelSpec, make_preprocessor

MODEL_FAMILY = "logistic"


def get_model_spec(quick: bool = False) -> ModelSpec:
    c_values = [0.03, 0.10] if quick else [0.005, 0.01, 0.03, 0.10, 0.30, 1.00, 3.00, 10.00]
    estimator = Pipeline(
        [
            ("preprocess", make_preprocessor(scale_numeric=True)),
            (
                "model",
                LogisticRegression(
                    solver="lbfgs",
                    class_weight=None,
                    max_iter=1500,
                    tol=1e-5,
                ),
            ),
        ]
    )
    return ModelSpec(
        name=MODEL_FAMILY,
        estimator=estimator,
        param_grid=[{"model__C": c_values}],
        search_n_jobs=4,
    )
