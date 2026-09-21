from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "xa"))

import logistic_model
import train_original_xa_models as training
import xgboost_model


class TrainingHelperTests(unittest.TestCase):
    def test_prediction_families_preserves_underscores(self) -> None:
        columns = {
            "new_model_men_model_xa": np.array([0.1]),
            "new_model_women_model_xa": np.array([0.2]),
        }

        self.assertEqual(training.prediction_families(columns), ["new_model"])

    def test_probability_metrics_do_not_select_test_threshold(self) -> None:
        metrics = training.probability_metrics(np.array([0, 1, 0, 1]), np.array([0.05, 0.8, 0.2, 0.7]))

        self.assertIn("xa_accuracy", metrics)
        self.assertNotIn("threshold", metrics)
        self.assertNotIn("f1_at_threshold", metrics)

    def test_logistic_spec_is_unweighted_and_tightly_converged(self) -> None:
        spec = logistic_model.get_model_spec(quick=False)
        model = spec.estimator.named_steps["model"]

        self.assertEqual(spec.name, "logistic")
        self.assertIsNone(model.class_weight)
        self.assertEqual(model.solver, "lbfgs")
        self.assertLessEqual(model.tol, 1e-5)
        self.assertEqual(spec.param_grid, [{"model__C": [0.005, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]}])
        self.assertEqual(spec.search_n_jobs, 4)

    def test_xgboost_spec_tunes_tree_count_with_gridsearch(self) -> None:
        spec = xgboost_model.get_model_spec(quick=True)

        self.assertEqual(spec.name, "xgboost")
        self.assertTrue(all("model__n_estimators" in candidate for candidate in spec.param_grid))
        self.assertEqual(len(spec.param_grid), 2)
        self.assertEqual(spec.search_n_jobs, 2)
        self.assertEqual(spec.estimator.named_steps["model"].n_jobs, 4)

    def test_prepare_output_dir_refuses_existing_artifacts_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            artifact = output_dir / "xa_original_training_results.json"
            artifact.write_text("{}", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                training.prepare_output_dir(output_dir, overwrite=False)

            training.prepare_output_dir(output_dir, overwrite=True)
            self.assertFalse(artifact.exists())


if __name__ == "__main__":
    unittest.main()
