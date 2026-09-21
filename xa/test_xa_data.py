from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "xa"))

import xa_data


def make_pass_row(game_id: int, row_id: int, *, is_goal_assist: bool, pass_outcome: str = "Complete") -> dict:
    start_x = 30.0 + (row_id % 20)
    start_y = 20.0 + (row_id % 30)
    end_x = start_x + 12.0
    end_y = start_y + (row_id % 5) - 2.0
    delta_x = end_x - start_x
    delta_y = end_y - start_y
    pass_length = float(np.hypot(delta_x, delta_y))
    return {
        "event_id": f"event-{game_id}-{row_id}",
        "game_id": game_id,
        "pass_outcome": pass_outcome,
        "is_goal_assist": is_goal_assist,
        "start_x": start_x,
        "start_y": start_y,
        "end_x": end_x,
        "end_y": end_y,
        "delta_x": delta_x,
        "delta_y": delta_y,
        "pass_length": pass_length,
        "pass_angle": float(np.arctan2(delta_y, delta_x)),
        "start_distance_to_goal": 80.0 - row_id * 0.1,
        "end_distance_to_goal": 70.0 - row_id * 0.1,
        "start_angle_to_goal": 0.20,
        "end_angle_to_goal": 0.25,
        "distance_reduction_to_goal": 10.0,
        "under_pressure": False,
        "is_cross": False,
        "is_cut_back": False,
        "is_switch": False,
        "is_through_ball": row_id % 2 == 0,
        "is_inswinging": False,
        "is_outswinging": False,
        "end_in_box": False,
        "end_in_six_yard_box": False,
        "pass_type": "Regular Pass",
        "pass_height": "Ground Pass",
        "pass_body_part": "Right Foot",
        "pass_technique": "None",
        "play_pattern": "Regular Play",
        "pass_context": "Open Play",
        "end_zone": "Middle Third",
    }


def make_raw_passes(game_ids: list[int]) -> pd.DataFrame:
    rows = []
    for game_id in game_ids:
        rows.append(make_pass_row(game_id, game_id * 10, is_goal_assist=True))
        rows.append(make_pass_row(game_id, game_id * 10 + 1, is_goal_assist=False))
        rows.append(
            make_pass_row(
                game_id,
                game_id * 10 + 2,
                is_goal_assist=True,
                pass_outcome="Incomplete",
            )
        )
    return pd.DataFrame(rows)


class XaDataTests(unittest.TestCase):
    def write_cache(self, data_dir: Path) -> None:
        make_raw_passes(list(range(1001, 1009))).to_pickle(
            data_dir / "women_xa_passes_competition_events_method.pkl"
        )
        make_raw_passes(list(range(2001, 2007))).to_pickle(
            data_dir / "men2015_xa_passes_competition_events_method.pkl"
        )

    def test_public_api_does_not_export_internal_helpers(self) -> None:
        exported = set(xa_data.__all__)
        self.assertIn("load_xa_test_and_rest", exported)
        self.assertIn("make_match_level_cv_splits", exported)
        self.assertNotIn("read_cached_passes", exported)
        self.assertNotIn("prepare_original_xa_rows", exported)
        self.assertNotIn("_read_cached_passes", exported)
        self.assertNotIn("_prepare_original_xa_rows", exported)

    def test_load_xa_test_and_rest_filters_and_labels_completed_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            self.write_cache(data_dir)

            bundle = xa_data.load_xa_test_and_rest(
                data_dir=data_dir,
                women_test_share=0.50,
                seed=42,
                minimum_positive_test_games=3,
            )

        self.assertEqual(bundle.target_column, xa_data.TARGET_COLUMN)
        self.assertEqual(bundle.men_rest["game_id"].nunique(), 6)
        self.assertEqual(bundle.women_rest["game_id"].nunique(), 4)
        self.assertEqual(bundle.women_test["game_id"].nunique(), 4)
        self.assertTrue(set(bundle.women_test["game_id"].unique()).issubset(bundle.women_test_games))
        self.assertTrue(set(bundle.women_rest["game_id"]).isdisjoint(bundle.women_test_games))

        for frame in [bundle.men_rest, bundle.women_rest, bundle.women_test]:
            self.assertTrue(frame["pass_outcome"].eq("Complete").all())
            expected_target = frame["is_goal_assist"].fillna(False).astype(int)
            pd.testing.assert_series_equal(
                frame[xa_data.TARGET_COLUMN].astype(int),
                expected_target,
                check_names=False,
            )
            self.assertIn("abs_delta_y", frame.columns)
            self.assertIn("context_height", frame.columns)

        self.assertEqual(bundle.split_summary["men_rest"]["rows"], 12)
        self.assertEqual(bundle.split_summary["women_rest"]["rows"], 8)
        self.assertEqual(bundle.split_summary["women_final_test"]["rows"], 8)
        self.assertEqual(bundle.split_summary["women_final_test"]["positives"], 4)

    def test_make_match_level_cv_splits_keeps_matches_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            self.write_cache(data_dir)
            bundle = xa_data.load_xa_test_and_rest(
                data_dir=data_dir,
                women_test_share=0.50,
                seed=42,
                minimum_positive_test_games=3,
            )

        splits = xa_data.make_match_level_cv_splits(
            bundle.women_rest,
            cv_folds=2,
            seed=7,
            dataset="women_test_fixture",
        )

        self.assertEqual(len(splits), 2)
        validation_indices_seen: set[int] = set()
        for train_indices, validation_indices in splits:
            train_games = set(bundle.women_rest.iloc[train_indices]["game_id"])
            validation_games = set(bundle.women_rest.iloc[validation_indices]["game_id"])
            self.assertTrue(train_games.isdisjoint(validation_games))
            self.assertGreater(int(bundle.women_rest.iloc[validation_indices][xa_data.TARGET_COLUMN].sum()), 0)
            validation_indices_seen.update(int(index) for index in validation_indices)

        self.assertEqual(validation_indices_seen, set(range(len(bundle.women_rest))))


if __name__ == "__main__":
    unittest.main()
