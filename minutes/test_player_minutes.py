from __future__ import annotations

import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from minutes.player_minutes import (
    MATCHES_FILENAME,
    METADATA_FILENAME,
    TOTALS_FILENAME,
    _timestamp_ms,
    build_minutes_tables,
    calculate_match_minutes,
    load_test_player_minutes,
    main,
)


def make_event(event_type: str, period: int, timestamp: str, player_id: int | None = None, **fields) -> dict:
    event = {"type": {"name": event_type}, "period": period, "timestamp": timestamp, "team": {"id": 10}}
    if player_id is not None:
        event["player"] = {"id": player_id}
    event.update(fields)
    return event


def substitution(outgoing: int, incoming: int, period: int, timestamp: str) -> dict:
    return make_event("Substitution", period, timestamp, outgoing, substitution={"replacement": {"id": incoming}})


def make_match(extra_events: list[dict] | None = None, period_ends: dict[int, str] | None = None) -> tuple[list[dict], list[dict]]:
    extra_events = extra_events or []
    period_ends = period_ends or {1: "00:47:00.500", 2: "00:50:00.250"}
    replacements = {
        event["substitution"]["replacement"]["id"]
        for event in extra_events
        if event["type"]["name"] == "Substitution"
    }
    lineups = []
    events = []
    for team_id, player_base in [(10, 1000), (20, 2000)]:
        starter_ids = list(range(player_base + 1, player_base + 12))
        roster_ids = starter_ids + list(range(player_base + 101, player_base + 104))
        lineups.append({
            "team_id": team_id,
            "lineup": [
                {
                    "player_id": player_id,
                    "player_name": f"Player {player_id}",
                    "positions": [{}] if player_id in starter_ids or player_id in replacements else [],
                }
                for player_id in roster_ids
            ],
        })
        events.append(make_event(
            "Starting XI", 1, "00:00:00.000", team={"id": team_id},
            tactics={"lineup": [{"player": {"id": player_id}} for player_id in starter_ids]},
        ))
    events.extend(extra_events)
    events.extend(make_event("Half End", period, timestamp) for period, timestamp in period_ends.items())
    events.sort(key=lambda event: (event["period"], _timestamp_ms(event["timestamp"])))
    return lineups, events


class PlayerMinutesTests(unittest.TestCase):
    def calculate(self, extra_events=None, period_ends=None):
        lineups, events = make_match(extra_events, period_ends)
        frame, summary = calculate_match_minutes(123, lineups, events)
        return frame.set_index("player_id"), summary

    def test_full_match_includes_stoppage_without_halftime_break(self):
        frame, summary = self.calculate()
        self.assertEqual(len(frame), 22)
        self.assertNotIn(1103, frame.index)
        self.assertEqual(frame.loc[1001, "milliseconds_played"], 5_820_750)
        self.assertEqual(frame.loc[1001, "minutes_played"], 97.0125)
        self.assertEqual(summary["match_minutes"], 97.0125)

    def test_halftime_substitution_uses_period_not_cumulative_event_minute(self):
        frame, _ = self.calculate([substitution(1001, 1101, 2, "00:00:00.000")])
        self.assertEqual(frame.loc[1001, "milliseconds_played"], 2_820_500)
        self.assertEqual(frame.loc[1101, "milliseconds_played"], 3_000_250)
        self.assertFalse(frame.loc[1101, "is_starter"])

    def test_substitute_can_be_substituted_later(self):
        frame, _ = self.calculate([
            substitution(1001, 1101, 2, "00:15:00.125"),
            substitution(1101, 1102, 2, "00:30:00.125"),
        ])
        self.assertEqual(frame.loc[1101, "milliseconds_played"], 900_000)
        self.assertEqual(frame.loc[1001, "milliseconds_played"], 3_720_625)
        self.assertEqual(frame.loc[1102, "milliseconds_played"], 1_200_125)

    def test_red_and_second_yellow_end_appearance(self):
        for card, field in [("Red Card", "foul_committed"), ("Second Yellow", "bad_behaviour")]:
            with self.subTest(card=card):
                frame, _ = self.calculate([
                    make_event("Foul Committed", 1, "00:30:00.250", 1001, **{field: {"card": {"name": card}}}),
                    make_event("Tactical Shift", 2, "00:10:00.000"),
                ])
                self.assertEqual(frame.loc[1001, "milliseconds_played"], 1_800_250)

    def test_bench_red_does_not_create_an_appearance(self):
        frame, _ = self.calculate([
            make_event("Bad Behaviour", 2, "00:10:00.000", 1103, bad_behaviour={"card": {"name": "Red Card"}}),
        ])
        self.assertEqual(len(frame), 22)
        self.assertNotIn(1103, frame.index)

    def test_temporary_absences_count_but_permanent_departures_stop_time(self):
        frame, _ = self.calculate([
            make_event("Player Off", 2, "00:05:00.000", 1001, player_off={"permanent": False}),
            make_event("Player On", 2, "00:06:00.000", 1001),
            make_event("Player Off", 2, "00:10:00.000", 1002, player_off={"permanent": True}),
        ])
        self.assertEqual(frame.loc[1001, "milliseconds_played"], 5_820_750)
        self.assertEqual(frame.loc[1002, "milliseconds_played"], 3_420_500)

    def test_substitution_after_permanent_departure_does_not_double_count(self):
        frame, _ = self.calculate([
            make_event("Player Off", 2, "00:10:00.000", 1001, player_off={"permanent": True}),
            substitution(1001, 1101, 2, "00:11:00.000"),
        ])
        self.assertEqual(frame.loc[1001, "milliseconds_played"], 3_420_500)
        self.assertEqual(frame.loc[1101, "milliseconds_played"], 2_340_250)

    def test_extra_time_counts_but_shootouts_do_not(self):
        frame, summary = self.calculate(
            [substitution(1001, 1101, 3, "00:00:00.000"), make_event("Half End", 5, "00:10:00.000")],
            {1: "00:47:00.000", 2: "00:50:00.000", 3: "00:15:00.000", 4: "00:16:00.000"},
        )
        self.assertEqual(summary["match_minutes"], 128)
        self.assertEqual(frame.loc[1001, "minutes_played"], 97)
        self.assertEqual(frame.loc[1101, "minutes_played"], 31)

    def test_duplicate_half_end_events_are_not_added_twice(self):
        frame, _ = self.calculate([make_event("Half End", 1, "00:47:00.500")])
        self.assertEqual(frame.loc[1001, "milliseconds_played"], 5_820_750)

    def test_short_recorded_match_is_not_padded_to_90_minutes(self):
        frame, _ = self.calculate(period_ends={1: "00:45:00.000", 2: "00:07:00.000"})
        self.assertEqual(frame.loc[1001, "minutes_played"], 52)

    def test_missing_half_end_fails(self):
        lineups, events = make_match()
        events = [event for event in events if not (event["type"]["name"] == "Half End" and event["period"] == 1)]
        with self.assertRaisesRegex(ValueError, "Half End"):
            calculate_match_minutes(123, lineups, events)

    def test_invalid_substitution_fails(self):
        with self.assertRaisesRegex(ValueError, "not active"):
            self.calculate([substitution(1102, 1101, 2, "00:10:00.000")])

    def test_event_after_period_end_fails(self):
        with self.assertRaisesRegex(ValueError, "after period"):
            self.calculate([make_event("Pass", 2, "00:51:00.000", 1001)])

    def test_totals_include_zero_action_matches_and_exclude_unselected_games(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "lineups").mkdir()
            (root / "events").mkdir()
            lineups, events = make_match()
            for game_id in [1, 2, 3]:
                (root / "lineups" / f"{game_id}.json").write_text(json.dumps(lineups), encoding="utf-8")
                (root / "events" / f"{game_id}.json").write_text(json.dumps(events), encoding="utf-8")
            totals, player_matches, summaries = build_minutes_tables({1, 2}, root)
            player = totals.set_index("player_id").loc[1001]
            self.assertEqual(player["test_games_played"], 2)
            self.assertEqual(player["test_milliseconds"], 11_641_500)
            self.assertEqual(player["test_minutes"], 194.025)
            self.assertEqual(set(player_matches["game_id"]), {1, 2})
            self.assertEqual(len(player_matches), 44)
            self.assertEqual(len(summaries), 2)
            self.assertTrue(totals["player_id"].is_unique)
            self.assertNotIn(1103, set(totals["player_id"]))

            totals.to_csv(root / TOTALS_FILENAME, index=False)
            (root / METADATA_FILENAME).write_text(json.dumps({"test_game_ids": [1, 2]}), encoding="utf-8")
            loaded = load_test_player_minutes({1, 2}, root)
            self.assertEqual(len(loaded), 22)
            with self.assertRaisesRegex(ValueError, "different test split"):
                load_test_player_minutes({1, 3}, root)

    def test_duplicate_test_match_ids_fail_before_reading_data(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            build_minutes_tables([1, 1])

    def test_shared_and_xa_lookups_use_separate_flat_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for prefix, game_ids, minutes in [("", [1, 2], 100), ("xa_", [2, 3], 150)]:
                (root / (prefix + TOTALS_FILENAME)).write_text(
                    f"player_id,test_minutes\n1001,{minutes}\n", encoding="utf-8"
                )
                (root / (prefix + METADATA_FILENAME)).write_text(
                    json.dumps({"test_game_ids": game_ids}), encoding="utf-8"
                )
            shared = load_test_player_minutes({1, 2}, root)
            xa = load_test_player_minutes({2, 3}, root, split="xa")
            self.assertEqual(shared.loc[0, "test_minutes"], 100)
            self.assertEqual(xa.loc[0, "test_minutes"], 150)
            with self.assertRaisesRegex(ValueError, "different test split"):
                load_test_player_minutes({2, 3}, root)
            with self.assertRaisesRegex(ValueError, "different test split"):
                load_test_player_minutes({1, 2}, root, split="xa")

    def test_unknown_split_fails_before_reading_files(self):
        with self.assertRaisesRegex(ValueError, "Unknown minutes split"):
            load_test_player_minutes({1}, split="unknown")

    def test_generation_uses_canonical_split_and_preserves_other_lookup(self):
        for split, argv, module_name, loader_name, prefix in [
            ("shared", [], "xg.xg_data", "load_xg_test_and_rest", ""),
            ("xa", ["--split", "xa"], "xa.xa_data", "load_xa_test_and_rest", "xa_"),
        ]:
            with self.subTest(split=split), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "lineups").mkdir()
                (root / "events").mkdir()
                lineups, events = make_match()
                for game_id in [1, 2]:
                    (root / "lineups" / f"{game_id}.json").write_text(json.dumps(lineups), encoding="utf-8")
                    (root / "events" / f"{game_id}.json").write_text(json.dumps(events), encoding="utf-8")
                tables = build_minutes_tables({1, 2}, root)
                for cache_name in [
                    "women_matches_competition_events_method.pkl",
                    "women_xg_shots_competition_events_method.pkl",
                    "men2015_xg_shots_competition_events_method.pkl",
                    "women_xa_passes_competition_events_method.pkl",
                    "men2015_xa_passes_competition_events_method.pkl",
                ]:
                    (root / cache_name).touch()
                other_prefix = "" if split == "xa" else "xa_"
                for filename in [TOTALS_FILENAME, MATCHES_FILENAME, METADATA_FILENAME]:
                    (root / (other_prefix + filename)).write_text("unchanged", encoding="utf-8")
                data_module = importlib.import_module(module_name)
                with (
                    patch("minutes.player_minutes.OUTPUT_DIR", root),
                    patch.object(data_module, "DEFAULT_DATA_DIR", root),
                    patch.object(data_module, loader_name, return_value=SimpleNamespace(women_test_games={2, 1})) as loader,
                    patch("minutes.player_minutes.build_minutes_tables", return_value=tables) as build_tables,
                    patch("builtins.print"),
                ):
                    main(argv)
                loader.assert_called_once_with()
                build_tables.assert_called_once_with([1, 2])
                metadata = json.loads((root / (prefix + METADATA_FILENAME)).read_text(encoding="utf-8"))
                self.assertEqual(metadata["split_source"], f"{module_name}.{loader_name}")
                self.assertEqual(metadata["seed"], data_module.RANDOM_SEED)
                self.assertEqual(metadata["test_share"], data_module.WOMEN_TEST_SHARE)
                self.assertEqual(metadata["test_game_ids"], [1, 2])
                self.assertEqual(metadata["players"], 22)
                self.assertEqual(metadata["player_matches"], 44)
                self.assertEqual(len(load_test_player_minutes({1, 2}, root, split=split)), 22)
                for filename in [TOTALS_FILENAME, MATCHES_FILENAME, METADATA_FILENAME]:
                    self.assertTrue((root / (prefix + filename)).is_file())
                    self.assertEqual((root / (other_prefix + filename)).read_text(encoding="utf-8"), "unchanged")
                self.assertFalse((root / "xa").exists())


if __name__ == "__main__":
    unittest.main()
