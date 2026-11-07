"""Build reusable player minutes for shared or xA women's test matches."""

from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPEN_DATA_DIR = PROJECT_ROOT / "open-data" / "data"
OUTPUT_DIR = Path(__file__).resolve().parent
TOTALS_FILENAME = "women_test_player_minutes.csv"
MATCHES_FILENAME = "women_test_player_match_minutes.csv"
METADATA_FILENAME = "women_test_minutes_metadata.json"


def _timestamp_ms(timestamp: str) -> int:
    hours, minutes, seconds = timestamp.split(":")
    seconds = Decimal(seconds)
    if int(hours) < 0 or not 0 <= int(minutes) < 60 or not 0 <= seconds < 60:
        raise ValueError(f"Invalid event timestamp: {timestamp}")
    milliseconds = (int(hours) * 3600 + int(minutes) * 60 + seconds) * 1000
    if milliseconds != int(milliseconds):
        raise ValueError(f"Timestamp exceeds millisecond precision: {timestamp}")
    return int(milliseconds)


def _period_lengths(events: list[dict]) -> dict[int, int]:
    lengths = {}
    played_periods = {event["period"] for event in events if event["period"] <= 4}
    for event in events:
        period = event["period"]
        if period <= 4 and event["type"]["name"] == "Half End":
            lengths[period] = max(lengths.get(period, 0), _timestamp_ms(event["timestamp"]))
    if not lengths or set(lengths) != played_periods:
        raise ValueError("Every played period must have a Half End event.")
    if sorted(lengths) != list(range(1, max(lengths) + 1)):
        raise ValueError("Match periods must be consecutive, starting at period 1.")
    return lengths


def calculate_match_minutes(game_id: int, lineups: list[dict], events: list[dict]) -> tuple[pd.DataFrame, dict]:
    """Count appearance time, including stoppage time and temporary treatment absences."""
    lengths = _period_lengths(events)
    offsets = {
        period: sum(duration for previous, duration in lengths.items() if previous < period)
        for period in lengths
    }
    match_ms = sum(lengths.values())
    roster = {}
    for team in lineups:
        for player in team["lineup"]:
            player_id = int(player["player_id"])
            if player_id in roster:
                raise ValueError(f"Game {game_id}: duplicate roster player {player_id}.")
            roster[player_id] = (int(team["team_id"]), player["player_name"])

    active = {}
    played_ms = {}
    starters = {}
    departures = {}
    starting_teams = set()

    def start_player(player_id: int, team_id: int, elapsed_ms: int, is_starter: bool) -> None:
        if player_id not in roster or roster[player_id][0] != team_id:
            raise ValueError(f"Game {game_id}: player {player_id} is absent from team {team_id}'s roster.")
        if player_id in played_ms:
            raise ValueError(f"Game {game_id}: player {player_id} appears more than once.")
        if sum(roster[active_id][0] == team_id for active_id in active) >= 11:
            raise ValueError(f"Game {game_id}: team {team_id} has more than 11 active players.")
        active[player_id] = elapsed_ms
        played_ms[player_id] = 0
        starters[player_id] = is_starter

    def finish_player(player_id: int, elapsed_ms: int, reason: str) -> None:
        if player_id not in active:
            return
        started_ms = active.pop(player_id)
        if elapsed_ms < started_ms:
            raise ValueError(f"Game {game_id}: negative playing time for player {player_id}.")
        played_ms[player_id] += elapsed_ms - started_ms
        departures[player_id] = reason

    for event in events:
        period = event["period"]
        if period > 4:
            continue
        local_ms = _timestamp_ms(event["timestamp"])
        if local_ms > lengths[period]:
            raise ValueError(f"Game {game_id}: event occurs after period {period}'s Half End.")
        elapsed_ms = offsets[period] + local_ms
        event_type = event["type"]["name"]
        if event_type == "Starting XI":
            team_id = int(event["team"]["id"])
            if period != 1 or team_id in starting_teams or len(event["tactics"]["lineup"]) != 11:
                raise ValueError(f"Game {game_id}: invalid Starting XI for team {team_id}.")
            starting_teams.add(team_id)
            for player in event["tactics"]["lineup"]:
                start_player(int(player["player"]["id"]), team_id, 0, True)
        elif event_type == "Substitution":
            outgoing_id = int(event["player"]["id"])
            incoming_id = int(event["substitution"]["replacement"]["id"])
            team_id = int(event["team"]["id"])
            if outgoing_id not in active and departures.get(outgoing_id) != "permanent departure":
                raise ValueError(f"Game {game_id}: substituted player {outgoing_id} is not active.")
            if roster[outgoing_id][0] != team_id:
                raise ValueError(f"Game {game_id}: substituted player {outgoing_id} belongs to another team.")
            finish_player(outgoing_id, elapsed_ms, "substitution")
            start_player(incoming_id, team_id, elapsed_ms, False)
        else:
            card = event.get("foul_committed", {}).get("card", event.get("bad_behaviour", {}).get("card", {}))
            permanent_off = event_type == "Player Off" and event.get("player_off", {}).get("permanent", False)
            if card.get("name") in {"Red Card", "Second Yellow"}:
                finish_player(int(event["player"]["id"]), elapsed_ms, "dismissal")
            elif permanent_off:
                finish_player(int(event["player"]["id"]), elapsed_ms, "permanent departure")

    if starting_teams != {int(team["team_id"]) for team in lineups} or len(starting_teams) != 2:
        raise ValueError(f"Game {game_id}: both teams need Starting XI events.")
    for player_id in list(active):
        finish_player(player_id, match_ms, "final whistle")

    rows = []
    for player_id, milliseconds in sorted(played_ms.items()):
        if not 0 <= milliseconds <= match_ms:
            raise ValueError(f"Game {game_id}: out-of-bounds minutes for player {player_id}.")
        team_id, player_name = roster[player_id]
        rows.append({
            "game_id": game_id,
            "team_id": team_id,
            "player_id": player_id,
            "player_name": player_name,
            "is_starter": starters[player_id],
            "milliseconds_played": milliseconds,
            "minutes_played": milliseconds / 60000,
        })
    frame = pd.DataFrame(rows)
    if (frame.groupby("team_id")["milliseconds_played"].sum() > 11 * match_ms).any():
        raise ValueError(f"Game {game_id}: team minutes exceed 11 times the match duration.")
    summary = {
        "game_id": game_id,
        "period_milliseconds": lengths,
        "match_minutes": match_ms / 60000,
        "players": len(frame),
    }
    return frame, summary


def build_minutes_tables(test_game_ids, open_data_dir: Path = OPEN_DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Read only the selected matches, without requiring any player action or prediction."""
    game_ids = sorted(int(game_id) for game_id in test_game_ids)
    if not game_ids or len(game_ids) != len(set(game_ids)):
        raise ValueError("Test match IDs must be nonempty and unique.")
    frames = []
    match_summaries = []
    for game_id in game_ids:
        lineups = json.loads((open_data_dir / "lineups" / f"{game_id}.json").read_text(encoding="utf-8"))
        events = json.loads((open_data_dir / "events" / f"{game_id}.json").read_text(encoding="utf-8"))
        frame, summary = calculate_match_minutes(game_id, lineups, events)
        expected_players = {
            int(player["player_id"])
            for team in lineups
            for player in team["lineup"]
            if player["positions"]
        }
        if set(frame["player_id"]) != expected_players:
            raise ValueError(f"Game {game_id}: event-based appearances disagree with the lineup roster.")
        frames.append(frame)
        match_summaries.append(summary)

    player_matches = pd.concat(frames, ignore_index=True).sort_values(["game_id", "player_id"]).reset_index(drop=True)
    if player_matches.duplicated(["game_id", "player_id"]).any():
        raise ValueError("Duplicate player-match minutes.")
    totals = player_matches.groupby("player_id", as_index=False).agg(
        player_name=("player_name", "first"),
        test_games_played=("game_id", "nunique"),
        test_milliseconds=("milliseconds_played", "sum"),
    )
    totals["test_minutes"] = totals["test_milliseconds"] / 60000
    return totals, player_matches, match_summaries


def _minutes_filenames(split: str) -> dict[str, str]:
    if split not in {"shared", "xa"}:
        raise ValueError(f"Unknown minutes split: {split}")
    prefix = "xa_" if split == "xa" else ""
    return {
        "totals": prefix + TOTALS_FILENAME,
        "matches": prefix + MATCHES_FILENAME,
        "metadata": prefix + METADATA_FILENAME,
    }


def load_test_player_minutes(test_game_ids, data_dir: Path = OUTPUT_DIR, *, split: str = "shared") -> pd.DataFrame:
    """Load the selected lookup only if its exact test match IDs match the caller's split."""
    filenames = _minutes_filenames(split)
    metadata = json.loads((data_dir / filenames["metadata"]).read_text(encoding="utf-8"))
    if set(map(int, test_game_ids)) != set(metadata["test_game_ids"]):
        raise ValueError("Saved minutes belong to a different test split. Select the matching lookup or regenerate it.")
    totals = pd.read_csv(data_dir / filenames["totals"])
    if totals["player_id"].isna().any() or totals["player_id"].duplicated().any():
        raise ValueError("Saved minutes must have exactly one row per player ID.")
    return totals


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", choices=["shared", "xa"], default="shared",
        help="Generate the shared xG/xT/VAEP/Atomic-VAEP lookup or the separate xA lookup.",
    )
    args = parser.parse_args(argv)
    filenames = _minutes_filenames(args.split)
    if args.split == "xa":
        from xa.xa_data import DEFAULT_DATA_DIR, RANDOM_SEED, WOMEN_TEST_SHARE, load_xa_test_and_rest as load_test_data

        split_source = "xa.xa_data.load_xa_test_and_rest"
        required_caches = [
            "women_xa_passes_competition_events_method.pkl",
            "men2015_xa_passes_competition_events_method.pkl",
        ]
    else:
        from xg.xg_data import DEFAULT_DATA_DIR, RANDOM_SEED, WOMEN_TEST_SHARE, load_xg_test_and_rest as load_test_data

        split_source = "xg.xg_data.load_xg_test_and_rest"
        required_caches = [
            "women_matches_competition_events_method.pkl",
            "women_xg_shots_competition_events_method.pkl",
            "men2015_xg_shots_competition_events_method.pkl",
        ]
    for filename in required_caches:
        if not (DEFAULT_DATA_DIR / filename).exists():
            raise FileNotFoundError(f"Prepare the existing {args.split} split caches before running this script: {filename}")
    test_game_ids = sorted(load_test_data().women_test_games)
    totals, player_matches, match_summaries = build_minutes_tables(test_game_ids)
    metadata = {
        "split_source": split_source,
        "seed": RANDOM_SEED,
        "test_share": WOMEN_TEST_SHARE,
        "test_game_ids": test_game_ids,
        "test_games": len(test_game_ids),
        "players": len(totals),
        "player_matches": len(player_matches),
        "minutes_definition": {
            "include_stoppage_time": True,
            "include_extra_time": True,
            "include_temporary_treatment_absences": True,
            "include_half_time_breaks": False,
            "include_penalty_shootouts": False,
            "end_appearance_at": "substitution, dismissal, permanent Player Off, or recorded final whistle",
            "clock": "period-relative event timestamps, accumulated across played periods",
            "precision": "integer milliseconds; sum before converting to minutes",
        },
        "matches": match_summaries,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    totals.to_csv(OUTPUT_DIR / filenames["totals"], index=False, float_format="%.9f")
    player_matches.to_csv(OUTPUT_DIR / filenames["matches"], index=False, float_format="%.9f")
    (OUTPUT_DIR / filenames["metadata"]).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(totals):,} players across {len(test_game_ids)} test matches ({len(player_matches):,} appearances).")
    print(OUTPUT_DIR / filenames["totals"])


if __name__ == "__main__":
    main()
