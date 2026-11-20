import ast
import json
from pathlib import Path
import sys
import unittest

import joblib
import nbformat
import numpy as np
import pandas as pd

from analysis import atomic_values, player_values, ranking_summary, shortlists, spatial_summary
from inference import apply_adapter, load_pair, predict_pair
from export_models import comparison_metrics, comparison_seeds, representative_seeds


HOME = Path(__file__).resolve().parents[1]


class InferenceTests(unittest.TestCase):
    def representative_directory(self, method):
        catalog = pd.read_csv(HOME / "model_catalog.csv")
        return catalog.loc[(catalog.source == "C") & (catalog.method == method) & (catalog.budget == 540), "directory"].item()

    def test_adapter_equations_match_exported_training_code(self):
        sys.path.insert(0, str(HOME / "provenance"))
        from training_code_at_export.adapters import apply_calibration
        probability = np.array([0., 1., 1e-10, .001, .02, .1, .3, .8])
        action_types = np.array(["pass", "receival", "interception", "dribble", "clearance", "tackle", "shot", "goal"])
        for method in ("intercept", "logistic", "beta", "context"):
            for target in ("scores", "concedes"):
                with self.subTest(method=method, target=target):
                    parameters = joblib.load(HOME / self.representative_directory(method) / f"{target}.joblib")
                    np.testing.assert_allclose(apply_adapter(probability, parameters, action_types),
                                               apply_calibration(probability, parameters, action_types), rtol=0, atol=1e-14)

    def test_exported_models_reproduce_saved_predictions(self):
        features = np.load(HOME / "inputs/test_features.npy", mmap_mode="r")
        actions = pd.read_parquet(HOME / "inputs/test_actions.parquet", columns=["type_name"])
        indices = np.linspace(0, len(features) - 1, 2048, dtype=int)
        manifest = json.loads((HOME / "export_manifest.json").read_text(encoding="utf-8"))
        for relative in manifest["reference_sha256"]:
            with self.subTest(model=relative):
                pair = load_pair(HOME, relative)
                predicted = predict_pair(pair, features[indices], actions.type_name.to_numpy()[indices])
                expected = np.load(HOME / "references" / relative / "test_predictions.npy", mmap_mode="r")[indices]
                np.testing.assert_allclose(predicted, expected, rtol=1e-11, atol=1e-13)

    def test_context_adapter_requires_action_types(self):
        parameters = joblib.load(HOME / self.representative_directory("context") / "scores.joblib")
        with self.assertRaises(ValueError):
            apply_adapter(np.array([.1, .2]), parameters)


class SeedSelectionTests(unittest.TestCase):
    def metrics(self):
        rows = []
        for seed, factor in zip((17, 29, 43, 71), (.8, .99, 1.01, 1.2)):
            for target, baseline in (("scores", .04), ("concedes", .01)):
                rows.append({"source": "C", "method": "beta", "budget": 540, "seed": seed,
                             "target": target, "log_loss": baseline * factor})
        return pd.DataFrame(rows)

    def test_selects_nearest_pair_not_best_seed(self):
        selected = representative_seeds(self.metrics())
        self.assertEqual(selected.loc[selected.selected, "seed"].tolist(), [29])
        self.assertEqual(selected.loc[selected.scores.idxmin(), "seed"], 17)
        self.assertEqual(selected.selected.sum(), 1)

    def test_comparison_export_excludes_lower_budgets_and_legacy_continuation(self):
        full_budget = self.metrics()
        lower_budget = full_budget.assign(budget=500)
        legacy = full_budget.assign(method="continued")
        base = full_budget.assign(method="direct", budget=0)
        selection = comparison_seeds(pd.concat([full_budget, lower_budget, legacy, base], ignore_index=True))
        self.assertEqual(len(selection), 4)
        self.assertEqual(set(selection.budget), {540})
        self.assertEqual(selection.loc[selection.selected, "seed"].tolist(), [29])

    def test_comparison_metrics_keep_bases_and_all_full_budget_seeds(self):
        full_budget = self.metrics()
        continuation = full_budget.assign(method="continued_match")
        bases = full_budget.iloc[:2].assign(method="direct", budget=0, seed=42)
        women = bases.assign(source="D", budget=540)
        metrics = pd.concat([bases, women, full_budget, continuation,
                             full_budget.assign(budget=200), full_budget.assign(method="continued")], ignore_index=True)
        actual = comparison_metrics(metrics).reset_index(drop=True)
        expected = pd.concat([bases, women, full_budget, continuation], ignore_index=True)
        pd.testing.assert_frame_equal(actual, expected)

    def test_target_scale_does_not_change_selection(self):
        metrics = self.metrics()
        metrics.loc[metrics.target == "scores", "log_loss"] *= 100
        selected = representative_seeds(metrics)
        self.assertEqual(selected.loc[selected.selected, "seed"].tolist(), [29])

    def test_identical_seeds_choose_smallest(self):
        metrics = self.metrics()
        metrics["log_loss"] = np.where(metrics.target == "scores", .04, .01)
        selected = representative_seeds(metrics)
        self.assertEqual(selected.loc[selected.selected, "seed"].tolist(), [17])

    def test_missing_or_duplicate_seed_results_are_rejected(self):
        metrics = self.metrics()
        for invalid in (metrics.iloc[1:], metrics.loc[metrics.seed != 71], pd.concat([metrics, metrics.iloc[:1]])):
            with self.assertRaises(ValueError):
                representative_seeds(invalid)


class AnalysisTests(unittest.TestCase):
    def actions(self):
        return pd.DataFrame({
            "game_id": [10, 10, 10, 20, 20], "action_id": [0, 1, 2, 0, 1],
            "team_id": [1, 2, 2, 3, 3], "period_id": [1, 1, 1, 1, 1],
            "time_seconds": [1., 2., 3., 1., 2.], "type_name": ["pass", "goal", "receival", "pass", "shot"],
        })

    def test_match_boundaries_team_changes_and_goal_reset(self):
        probabilities = np.array([[.1, .02], [.8, .03], [.05, .01], [.4, .1], [.5, .09]])
        values = atomic_values(self.actions(), probabilities)
        np.testing.assert_allclose(values.iloc[[0, 3]].to_numpy(), 0, atol=1e-14)
        np.testing.assert_allclose(values.iloc[1].to_numpy(), [.78, .07, .85], atol=1e-14)
        np.testing.assert_allclose(values.iloc[2].to_numpy(), [.05, -.01, .04], atol=1e-14)

    def test_atomic_values_match_original_notebook_function(self):
        original_path = HOME.parent / "enriched_atomic_vaep_features_competition_events.ipynb"
        if not original_path.exists():
            self.skipTest("Original notebook is outside this exported bundle")
        notebook = nbformat.read(original_path, as_version=4)
        source = next(cell.source for cell in notebook.cells if cell.cell_type == "code" and "def calculate_vaep_values_by_game" in cell.source)
        function = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "calculate_vaep_values_by_game")
        probabilities = np.array([[.1, .02], [.8, .03], [.05, .01], [.4, .1], [.5, .09]])
        actual = atomic_values(self.actions(), probabilities)
        from socceraction.atomic.vaep import formula
        namespace = {"np": np, "pd": pd, "vaep_formula": formula}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "original_notebook", "exec"), namespace)
        expected = namespace["calculate_vaep_values_by_game"](self.actions(), probabilities[:, 0], probabilities[:, 1])
        pd.testing.assert_frame_equal(actual, expected)

    def test_reordered_duplicate_or_invalid_actions_are_rejected(self):
        probabilities = np.tile([.1, .01], (5, 1))
        reordered = self.actions().iloc[[1, 0, 2, 3, 4]]
        with self.assertRaises(ValueError):
            atomic_values(reordered, probabilities)
        duplicated = self.actions()
        duplicated.loc[1, "action_id"] = 0
        with self.assertRaises(ValueError):
            atomic_values(duplicated, probabilities)
        with self.assertRaises(ValueError):
            atomic_values(self.actions(), np.full((5, 2), np.nan))

    def test_player_totals_preserve_negative_values_and_use_all_minutes(self):
        actions = pd.DataFrame({"player_id": [1, 1, 2, 3]})
        values = pd.DataFrame({"offensive_value": [.2, -.1, 5., -.2], "defensive_value": [0., 0., 0., 0.],
                               "vaep_value": [.2, -.1, 5., -.2]})
        context = pd.DataFrame({"player_id": [1, 2, 3, 4], "player_name": ["One", "Keeper", "Three", "No actions"],
                                "position_bucket": ["Forward", "Goalkeeper", "Forward", "Forward"],
                                "test_minutes": [180., 90., 90., 90.]})
        all_players, players = player_values(actions, {"C": values, "D": values * 2}, context, 1, 45)
        self.assertEqual(set(players.player_id), {1, 3})
        self.assertEqual(all_players.loc[all_players.player_id == 4, "action_count"].iloc[0], 0)
        indexed = players.set_index("player_id")
        self.assertAlmostEqual(indexed.loc[1, "C_per90"], .05)
        self.assertAlmostEqual(indexed.loc[3, "C_total"], -.2)
        self.assertAlmostEqual(indexed.C_total.sum(), -.1)

    def test_ties_and_small_shortlists_are_explicit(self):
        players = pd.DataFrame({"player_id": [3, 2, 1], "player_name": ["Three", "Two", "One"],
                                "position_bucket": ["Forward"] * 3, "test_minutes": [90.] * 3,
                                "action_count": [100] * 3, "C_per90": [1., 1., 1.], "D_per90": [1., 2., 3.]})
        lists = shortlists(players, ["C", "D"], top_k=2)
        self.assertEqual(lists.loc[lists.model == "C", "player_id"].tolist(), [1, 2, 3])
        self.assertTrue(lists.loc[lists.model == "C", "boundary_tie"].all())
        self.assertTrue(shortlists(players, ["C", "D"], top_k=15).empty)
        summary = ranking_summary(players, ["C", "D"], top_k=15)
        self.assertTrue(summary.shortlist_is_entire_group.all())
        self.assertTrue(summary.top_overlap.eq(3).all())
        self.assertTrue(summary.spearman.isna().all())
        self.assertTrue(summary.top_overlap_fraction.isna().all())

    def test_spatial_bin_counts_cover_boundary_actions(self):
        actions = pd.DataFrame({"x": [0., 105.], "y": [0., 68.], "type_name": ["pass", "pass"]})
        values = {"C": pd.DataFrame({"vaep_value": [1., -2.]}), "D": pd.DataFrame({"vaep_value": [0., 0.]})}
        summary = spatial_summary(actions, values, [("C", "D")], event_types=("pass",))
        self.assertEqual(summary.actions.sum(), 2)
        self.assertAlmostEqual((summary.mean_delta * summary.actions).sum(), -1.)


class NotebookTests(unittest.TestCase):
    def test_main_probability_plot_has_readable_tick_spacing(self):
        from plotting import probability_plot
        import matplotlib.pyplot as plt
        metrics = pd.DataFrame([
            {"model": model, "target": target, "log_loss": loss}
            for model, target, loss in [("D", "scores", .034910), ("C_adapter", "scores", .034471),
                                        ("D", "concedes", .008529), ("C_adapter", "concedes", .008454)]
        ])
        figure = probability_plot(metrics, ["D", "C_adapter"])
        figure.canvas.draw()
        for axis in figure.axes:
            labels = axis.get_xticklabels()
            for left, right in zip(labels, labels[1:]):
                self.assertFalse(left.get_window_extent().overlaps(right.get_window_extent()))
        plt.close(figure)

    def test_export_catalog_contains_only_bases_and_full_budget_runs(self):
        catalog = pd.read_csv(HOME / "model_catalog.csv")
        manifest = json.loads((HOME / "export_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(catalog), 19)
        self.assertEqual(manifest["model_pairs"], len(catalog))
        self.assertEqual(set(catalog.loc[catalog.method == "direct", "source"]), {"A", "B", "C", "D"})
        self.assertTrue(catalog.loc[catalog.method != "direct", "budget"].eq(540).all())
        self.assertNotIn("continued", set(catalog.method))
        expected = {f"{directory}/{target}.joblib" for directory in catalog.directory for target in ("scores", "concedes")}
        actual = {path.relative_to(HOME).as_posix() for branch in ("models", "runs") for path in (HOME / branch).rglob("*.joblib")}
        self.assertEqual(actual, expected)
        selection = pd.read_csv(HOME / "seed_selection.csv")
        self.assertEqual(len(selection), 60)
        self.assertEqual(selection.selected.sum(), 15)
        self.assertTrue(selection.budget.eq(540).all())

    def test_provenance_contains_only_required_training_snapshot_files(self):
        provenance = HOME / "provenance"
        self.assertFalse(list(provenance.rglob("*.ipynb")))
        snapshot_files = {path.name for path in (provenance / "training_code_at_export").glob("*.py")}
        self.assertEqual(snapshot_files, {"__init__.py", "adapters.py", "config.py"})

    def test_provenance_metrics_cover_only_exported_setups_and_all_seeds(self):
        catalog = pd.read_csv(HOME / "model_catalog.csv")
        metrics = pd.concat([pd.read_csv(HOME / "provenance/results" / name)
                             for name in ("metrics.csv", "continued_match_metrics.csv")], ignore_index=True)
        keys = ["source", "method", "budget"]
        self.assertEqual(set(metrics[keys].itertuples(index=False, name=None)),
                         set(catalog[keys].itertuples(index=False, name=None)))
        self.assertFalse(metrics.duplicated([*keys, "seed", "target"]).any())
        for setup, group in metrics.groupby(keys):
            expected_seeds = {42} if setup[1] == "direct" else {17, 29, 43, 71}
            self.assertEqual(set(group.seed), expected_seeds)
            self.assertEqual(len(group), 2 * len(expected_seeds))
            for _, target_pair in group.groupby("seed"):
                self.assertEqual(set(target_pair.target), {"scores", "concedes"})

    def test_notebook_main_comparison_is_adapter_against_women_only(self):
        notebook = nbformat.read(HOME / "01_compare_atomic_vaep.ipynb", as_version=4)
        configuration = next(cell.source for cell in notebook.cells if cell.cell_type == "code")
        settings = {}
        for node in ast.parse(configuration).body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id in {"MAIN_MODELS", "MAIN_PAIRS", "WOMEN_BUDGET"}:
                    settings[node.targets[0].id] = ast.literal_eval(node.value)
        self.assertEqual(settings["WOMEN_BUDGET"], 540)
        self.assertEqual(settings["MAIN_MODELS"], ["D", "C_adapter"])
        self.assertEqual(settings["MAIN_PAIRS"], [("C_adapter", "D")])

    def test_readme_explains_every_exported_model_pair(self):
        catalog = pd.read_csv(HOME / "model_catalog.csv")
        readme = (HOME / "README.md").read_text(encoding="utf-8")
        for directory in catalog.directory:
            with self.subTest(directory=directory):
                self.assertIn(f"`{directory}/`", readme)

    def test_single_comparison_action_plot_centers_points(self):
        from plotting import action_plot
        import matplotlib.pyplot as plt
        summary = pd.DataFrame([{"left": "C_adapter", "right": "D", "type_name": "pass",
                                 "actions": 1000, "mean_delta": .001, "mean_abs_delta": .002}])
        figure = action_plot(summary, [("C_adapter", "D")])
        for axis in figure.axes:
            np.testing.assert_allclose(axis.collections[0].get_offsets()[:, 1], [0.])
        plt.close(figure)

    def test_action_plot_keeps_goal_atoms_on_separate_scales(self):
        from plotting import action_plot
        import matplotlib.pyplot as plt
        pairs = [("C", "D"), ("C_adapter", "D"), ("C_adapter", "C")]
        summary = pd.DataFrame([
            {"left": left, "right": right, "type_name": name, "actions": 1000,
             "mean_delta": difference, "mean_abs_delta": abs(difference)}
            for left, right in pairs for name, difference in (("goal", -.05), ("pass", .0001))
        ])
        figure = action_plot(summary, pairs)
        self.assertEqual(len(figure.axes), 4)
        points = sum(len(collection.get_offsets()) for axis in figure.axes for collection in axis.collections)
        self.assertEqual(points, 2 * len(summary))
        self.assertEqual(len(figure.legends), 1)
        plt.close(figure)

    def test_notebook_is_valid_and_has_no_training_calls(self):
        notebook = nbformat.read(HOME / "01_compare_atomic_vaep.ipynb", as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                compile(cell.source, "comparison_notebook", "exec")
                for node in ast.walk(ast.parse(cell.source)):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                        self.assertNotIn(node.func.attr, {"fit", "partial_fit"})


if __name__ == "__main__":
    unittest.main()
