"""Small shared pieces for sklearn-style VAEP model families."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object
    param_grid: list[dict]
    search_n_jobs: int = 1


def make_feature_frame(frame: pd.DataFrame, feature_columns: tuple[str, ...]) -> pd.DataFrame:
    """Prepare numeric VAEP features for sklearn model families."""
    missing_columns = [column for column in feature_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"VAEP frame missing feature columns: {missing_columns[:20]}")

    features = frame.loc[:, list(feature_columns)].copy()
    for column in feature_columns:
        features[column] = pd.to_numeric(features[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return features.fillna(0.0).astype(np.float32, copy=False)


def make_preprocessor() -> SimpleImputer:
    """Build the lightweight preprocessing step used by numeric VAEP features."""
    return SimpleImputer(strategy="constant", fill_value=0.0)


def silence_common_sklearn_warnings() -> None:
    """Suppress benign pandas/sklearn feature-name warnings during large grid searches."""
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
