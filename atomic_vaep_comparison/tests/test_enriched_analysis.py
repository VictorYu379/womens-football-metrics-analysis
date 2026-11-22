import ast
from pathlib import Path
import unittest

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from scipy import stats

from analysis import (attach_pass_heights, compare_top_rankings, distribution_summary, normalization_top_comparison,
                      rank_movement_counts, ranking_metrics, shortlist_scope, spatial_summary)
from plotting import action_distribution_plot, player_plot, rank_movement_plot, spatial_plot


HOME = Path(__file__).resolve().parents[1]


class EnrichedAnalysisTests(unittest.TestCase):
    def original_function(self, name):
        path = HOME.parent / "enriched_atomic_vaep_features_competition_events.ipynb"
        if not path.exists():
            self.skipTest("Reference notebook is outside the shareable bundle")
        notebook = nbformat.read(path, as_version=4)
        namespace = {"np": np, "pd": pd, "stats": stats, "RANK_COMPARISON_TOP_K": 10, "POSITION_TOP_K": 15}
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            for node in ast.parse(cell.source).body:
                if isinstance(node, ast.FunctionDef) and node.name == name:
                    exec(compile(ast.Module(body=[node], type_ignores=[]), "reference_analysis", "exec"), namespace)
                    return namespace[name]
        self.fail(f"Reference function not found: {name}")

    def players(self, count=30):
        return pd.DataFrame({"player_id": np.arange(count), "player_name": [f"Player {index}" for index in range(count)],
                             "position_bucket": ["Forward"] * count, "test_minutes": [90.] * count,
                             "action_count": [100] * count, "C_adapter_total": np.arange(count, dtype=float),
                             "D_total": np.arange(count, dtype=float)[::-1],
                             "C_adapter_per90": np.arange(count, dtype=float)[::-1],
                             "D_per90": np.arange(count, dtype=float)})

    def test_ranking_metrics_match_original_including_ties(self):
        players = self.players()
        players.loc[:3, "C_adapter_total"] = 28.
        expected = self.original_function("ranking_comparison_metrics")(players, "C_adapter_total", "D_total")
        actual = ranking_metrics(players.C_adapter_total, players.D_total)
        mapping = {"kendall_tau_b": "kendall_tau_b", "mean_abs_rank_delta": "mean_abs_rank_change",
                   "mean_normalized_abs_rank_delta": "mean_abs_rank_change_fraction",
                   "exact_rank_agreement_pct": "exact_rank_agreement_pct",
                   "within_2_rank_agreement_pct": "within_2_rank_agreement_pct",
                   "within_5_rank_agreement_pct": "within_5_rank_agreement_pct"}
        for original, current in mapping.items():
            np.testing.assert_allclose(actual[current], expected[original], equal_nan=True)
        np.testing.assert_allclose(100 * actual["top_overlap_fraction"], expected["top_k_overlap_pct"], equal_nan=True)

    def test_singleton_metrics_are_explicitly_undefined(self):
        result = ranking_metrics([1.], [2.])
        for column in ("spearman", "kendall_tau_b", "mean_abs_rank_change_fraction", "top_overlap_fraction"):
            self.assertTrue(np.isnan(result[column]))

    def test_top15_matches_original_complete_union_and_boundary_ties(self):
        players = self.players(20)
        players.loc[:7, "C_adapter_total"] = 6.
        players["adapter_rank"] = players.C_adapter_total.rank(ascending=False, method="min")
        players["D_rank"] = players.D_total.rank(ascending=False, method="min")
        expected, original_union = self.original_function("compare_top_rankings")(players, "adapter_rank", "D_rank")
        actual, union = compare_top_rankings(players, "C_adapter_total", "D_total")
        self.assertTrue(actual["boundary_tie"])
        for column in expected:
            np.testing.assert_allclose(actual[column], expected[column], equal_nan=True)
        self.assertEqual(union.player_id.tolist(), original_union.player_id.tolist())
        np.testing.assert_array_equal(union.left_rank, original_union.left_rank)
        self.assertGreater(union.right_rank.max(), 15)

    def test_small_groups_are_excluded_and_minimum_overlap_is_not_chance_adjusted(self):
        scope = shortlist_scope(self.players(14))
        self.assertFalse(scope.included.item())
        with self.assertRaises(ValueError):
            compare_top_rankings(self.players(14), "C_adapter_total", "D_total")
        scope = shortlist_scope(self.players(15))
        self.assertEqual(scope.min_possible_overlap_pct.item(), 100)

    def test_normalization_reports_entrants_exits_and_full_ranks(self):
        summary, union = normalization_top_comparison(self.players(), ["C_adapter"])
        self.assertEqual(summary.retained_players.item(), 0)
        self.assertEqual(summary.entrants.item(), 15)
        self.assertEqual(summary.exits.item(), 15)
        self.assertEqual(len(union), 30)
        np.testing.assert_array_equal(union.rank_improvement_per90, union.total_rank - union.per90_rank)

    def test_rank_heatmap_does_not_move_outliers_to_boundaries(self):
        players = self.players()
        counts = rank_movement_counts(players, "C_adapter", "D", "total")
        self.assertEqual(counts['count'].sum() + counts.outside_display.iloc[0], len(players))
        self.assertEqual(counts.loc[counts.rank_delta.abs().eq(10), "count"].sum(), 0)
        self.assertGreater(counts.outside_display.iloc[0], 0)
        self.assertEqual(set(counts.rank_delta), set(range(-10, 11)))

    def test_rank_heatmap_uses_shared_reference_color_scale(self):
        counts = rank_movement_counts(self.players(), "C_adapter", "D", "total")
        figure = rank_movement_plot(counts, ("C_adapter", "D"), "total", 224)
        image = figure.axes[0].images[0]
        self.assertEqual(image.get_cmap().name, "Blues")
        self.assertEqual(image.get_clim(), (0, 224))
        plt.close(figure)

    def test_passing_metadata_excludes_direct_shots_and_keeps_unknown(self):
        actions = pd.DataFrame({"game_id": [1, 1, 1, 1], "action_id": [0, 1, 2, 3],
                                "original_event_id": ["one", "two", "three", "one"],
                                "type_name": ["pass", "freekick", "freekick", "receival"],
                                "vaep_delta": [.1, .2, -.3, .4]})
        events = pd.DataFrame({"game_id": [1, 1, 1], "original_event_id": ["one", "two", "three"],
                               "event_type": ["Pass", "Shot", "Pass"], "pass_height": ["Unknown", None, "High Pass"]})
        passed = attach_pass_heights(actions, events)
        self.assertEqual(passed.action_id.tolist(), [0, 2])
        np.testing.assert_allclose(passed.vaep_delta, [.1, -.3])
        self.assertEqual(passed.pass_height.astype(str).tolist(), ["Unknown", "High Pass"])
        with self.assertRaises(ValueError):
            attach_pass_heights(actions, events.iloc[1:])

    def test_distribution_outliers_remain_in_statistics(self):
        actions = pd.DataFrame({"type_name": ["pass"] * 3, "vaep_delta": [-.02, .001, .03],
                                "left_value": [.1, .2, .3], "right_value": [.12, .199, .27],
                                "offensive_delta": [-.02, .001, .03], "defensive_delta": [0., 0., 0.]})
        summary = distribution_summary(actions, ["type_name"])
        self.assertEqual(summary.actions.item(), 3)
        self.assertEqual(summary.outside_display.item(), 2)
        self.assertAlmostEqual(summary.total_delta.item(), .011)
        self.assertAlmostEqual(summary.std_delta.item(), actions.vaep_delta.std())
        figure = action_distribution_plot(actions, summary, "type_name", ("C_adapter", "D"), "Test")
        np.testing.assert_allclose(figure.axes[0].get_xlim(), [-.01, .01])
        plt.close(figure)

    def test_spatial_panels_share_scale_and_have_distinct_shot_threshold(self):
        actions = pd.DataFrame({"x": [10.] * 10 + [20.] * 19, "y": [20.] * 29,
                                "type_name": ["shot"] * 10 + ["pass"] * 19})
        values = {"C_adapter": pd.DataFrame({"vaep_value": [.1] * 29}), "D": pd.DataFrame({"vaep_value": [0.] * 29})}
        summary = spatial_summary(actions, values, [("C_adapter", "D")])
        figure = spatial_plot(summary, [("C_adapter", "D")])
        images = [image for axis in figure.axes for image in axis.images]
        self.assertEqual(len(images), 6)
        self.assertEqual(len({image.get_clim() for image in images}), 1)
        for image in images:
            self.assertEqual(image.get_cmap().name, "coolwarm")
            self.assertEqual(image.get_clim(), (-.03, .03))
            self.assertEqual(image.get_alpha(), .86)
            np.testing.assert_array_equal(image.get_cmap().get_bad(), [1., 1., 1., 0.])
        self.assertEqual(images[0].get_array().count(), 1)
        self.assertAlmostEqual(float(images[0].get_array().compressed()[0]), .1)
        self.assertEqual(images[1].get_array().count(), 0)
        plt.close(figure)

    def test_notebooks_share_fixed_heatmap_limits(self):
        paths = [HOME / "01_compare_atomic_vaep.ipynb",
                 HOME.parent / "enriched_atomic_vaep_features_competition_events.ipynb"]
        for path in paths:
            if not path.exists():
                continue
            constants = {}
            for cell in nbformat.read(path, as_version=4).cells:
                if cell.cell_type != "code":
                    continue
                for node in ast.parse(cell.source).body:
                    if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
                        continue
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            constants[target.id] = node.value.value
            with self.subTest(notebook=path.name):
                self.assertEqual(constants["SPATIAL_COLOR_LIMIT"], .03)
                self.assertEqual(constants["RANK_MOVEMENT_COLOR_MAX"], 224)

    def test_scatter_labels_do_not_overlap_or_cross_axis_border(self):
        figure = player_plot(self.players(), [("C_adapter", "D")], "total", annotate=True)
        figure.canvas.draw()
        axis = figure.axes[0]
        labels = [text.get_bbox_patch().get_window_extent() for text in axis.texts]
        self.assertEqual(len(labels), 8)
        for index, bounds in enumerate(labels):
            self.assertTrue(axis.bbox.contains(bounds.x0, bounds.y0))
            self.assertTrue(axis.bbox.contains(bounds.x1, bounds.y1))
            self.assertFalse(any(bounds.overlaps(other) for other in labels[index + 1:]))
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
