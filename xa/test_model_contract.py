from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "xa"))

from model_contract import ModelSpec, make_feature_frame, make_preprocessor
from xa_data import BOOLEAN_FEATURES, CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES


class ModelContractTests(unittest.TestCase):
    def test_make_feature_frame_returns_model_columns(self) -> None:
        row = {column: None for column in FEATURE_COLUMNS}
        for column in NUMERIC_FEATURES:
            row[column] = "1.5"
        for column in BOOLEAN_FEATURES:
            row[column] = None
        for column in CATEGORICAL_FEATURES:
            row[column] = None

        features = make_feature_frame(pd.DataFrame([row]))

        self.assertEqual(list(features.columns), FEATURE_COLUMNS)
        self.assertTrue(all(pd.api.types.is_numeric_dtype(features[column]) for column in NUMERIC_FEATURES))
        self.assertTrue(all(pd.api.types.is_integer_dtype(features[column]) for column in BOOLEAN_FEATURES))
        self.assertEqual(set(features[CATEGORICAL_FEATURES[0]]), {"Unknown"})

    def test_model_spec_is_small_contract(self) -> None:
        spec = ModelSpec(name="dummy", estimator=object(), param_grid=[{"a": [1]}])

        self.assertEqual(spec.name, "dummy")
        self.assertEqual(spec.param_grid, [{"a": [1]}])

    def test_make_preprocessor_has_expected_transformers(self) -> None:
        preprocessor = make_preprocessor(scale_numeric=True)

        self.assertEqual([name for name, _, _ in preprocessor.transformers], ["numeric", "boolean", "categorical"])


if __name__ == "__main__":
    unittest.main()
