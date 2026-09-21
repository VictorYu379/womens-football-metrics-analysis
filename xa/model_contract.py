"""Small shared pieces for sklearn-style xA model families."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xa_data import BOOLEAN_FEATURES, CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object
    param_grid: list[dict]
    search_n_jobs: int = 1


def make_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame[FEATURE_COLUMNS].copy()
    for column in NUMERIC_FEATURES:
        features[column] = pd.to_numeric(features[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    for column in BOOLEAN_FEATURES:
        features[column] = features[column].fillna(False).astype(int)
    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].fillna("Unknown").astype(str)
    return features


def make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_steps = [("impute", SimpleImputer(strategy="constant", fill_value=0.0))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    boolean_steps = [("impute", SimpleImputer(strategy="constant", fill_value=0))]
    categorical_steps = [
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
            ("boolean", Pipeline(boolean_steps), BOOLEAN_FEATURES),
            ("categorical", Pipeline(categorical_steps), CATEGORICAL_FEATURES),
        ],
        sparse_threshold=1.0,
    )
