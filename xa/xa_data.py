"""Stable data interface for original-definition xA modeling.

Public API:
- load_xa_test_and_rest(): prepare men/women data and lock the women final test set.
- make_match_level_cv_splits(...): build match-level folds for training scripts.

All lower-level cache loading, feature engineering, split assertions, and label checks
are private implementation details of this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from statsbombpy import sb

RANDOM_SEED = 42
WOMEN_TEST_SHARE = 0.20
VALIDATION_SHARE = 0.15
CV_FOLDS = 4
TARGET_COLUMN = "target_goal_assist"

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
DEFAULT_DATA_DIR = _PROJECT_ROOT / "vaep_data"
DEFAULT_OUTPUT_DIR = _SCRIPT_DIR / "xa_original_models"

WOMEN_COMPETITIONS = [
    {
        "competition_id": 37,
        "season_id": 281,
        "country": "England",
        "division": "FA Women's Super League",
        "season": "2023/2024",
        "gender": "female",
        "league": "WSL",
    },
    {
        "competition_id": 135,
        "season_id": 281,
        "country": "Germany",
        "division": "Frauen Bundesliga",
        "season": "2023/2024",
        "gender": "female",
        "league": "Frauen Bundesliga",
    },
    {
        "competition_id": 182,
        "season_id": 281,
        "country": "Spain",
        "division": "Liga F",
        "season": "2023/2024",
        "gender": "female",
        "league": "Liga F",
    },
    {
        "competition_id": 131,
        "season_id": 281,
        "country": "Italy",
        "division": "Serie A Women",
        "season": "2023/2024",
        "gender": "female",
        "league": "Serie A Women",
    },
    {
        "competition_id": 49,
        "season_id": 107,
        "country": "United States of America",
        "division": "NWSL",
        "season": "2023",
        "gender": "female",
        "league": "NWSL",
    },
]

MEN_COMPETITIONS = [
    {
        "competition_id": 11,
        "season_id": 27,
        "country": "Spain",
        "division": "La Liga",
        "season": "2015/2016",
        "gender": "male",
        "league": "La Liga",
    },
    {
        "competition_id": 2,
        "season_id": 27,
        "country": "England",
        "division": "Premier League",
        "season": "2015/2016",
        "gender": "male",
        "league": "Premier League",
    },
    {
        "competition_id": 12,
        "season_id": 27,
        "country": "Italy",
        "division": "Serie A",
        "season": "2015/2016",
        "gender": "male",
        "league": "Serie A",
    },
    {
        "competition_id": 7,
        "season_id": 27,
        "country": "France",
        "division": "Ligue 1",
        "season": "2015/2016",
        "gender": "male",
        "league": "Ligue 1",
    },
]

COMPETITIONS_BY_CACHE_TAG = {
    "women": WOMEN_COMPETITIONS,
    "men2015": MEN_COMPETITIONS,
}

FLAT_PASS_REQUIRED_COLUMNS = [
    "id",
    "match_id",
    "player_id",
    "player",
    "team",
    "position",
    "location",
    "pass_end_location",
    "pass_type",
    "pass_height",
    "pass_body_part",
    "pass_technique",
    "pass_outcome",
]
FLAT_SHOT_REQUIRED_COLUMNS = ["id", "shot_statsbomb_xg"]

NUMERIC_FEATURES = [
    "start_x", "start_y", "end_x", "end_y", "delta_x", "delta_y", "pass_length", "pass_angle",
    "start_distance_to_goal", "end_distance_to_goal", "start_angle_to_goal", "end_angle_to_goal",
    "distance_reduction_to_goal", "abs_delta_y", "abs_pass_angle", "start_distance_to_center",
    "end_distance_to_center", "goal_angle_gain", "progress_to_goal_rate", "end_x_squared",
    "end_y_centered_squared",
]

BOOLEAN_FEATURES = [
    "under_pressure", "is_cross", "is_cut_back", "is_switch", "is_through_ball", "is_inswinging",
    "is_outswinging", "end_in_box", "end_in_six_yard_box", "is_forward_pass", "is_backward_pass",
    "is_central_end", "is_final_third_end",
]

CATEGORICAL_FEATURES = [
    "pass_type", "pass_height", "pass_body_part", "pass_technique", "play_pattern", "pass_context",
    "end_zone", "start_x_band", "end_x_band", "start_y_band", "end_y_band", "distance_reduction_band",
    "pass_length_band", "pass_angle_band", "context_height", "context_body_part", "context_end_zone",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES

_DERIVED_FEATURES = {
    "abs_delta_y", "abs_pass_angle", "start_distance_to_center", "end_distance_to_center",
    "goal_angle_gain", "progress_to_goal_rate", "end_x_squared", "end_y_centered_squared",
    "is_forward_pass", "is_backward_pass", "is_central_end", "is_final_third_end", "start_x_band",
    "end_x_band", "start_y_band", "end_y_band", "distance_reduction_band", "pass_length_band",
    "pass_angle_band", "context_height", "context_body_part", "context_end_zone",
}


@dataclass
class XaDataBundle:
    men_rest: pd.DataFrame
    women_rest: pd.DataFrame
    women_test: pd.DataFrame
    women_test_games: set[int]
    split_summary: dict[str, dict]
    target_column: str = TARGET_COLUMN
    feature_columns: tuple[str, ...] = tuple(FEATURE_COLUMNS)
    numeric_features: tuple[str, ...] = tuple(NUMERIC_FEATURES)
    boolean_features: tuple[str, ...] = tuple(BOOLEAN_FEATURES)
    categorical_features: tuple[str, ...] = tuple(CATEGORICAL_FEATURES)


__all__ = [
    "BOOLEAN_FEATURES", "CATEGORICAL_FEATURES", "CV_FOLDS", "DEFAULT_DATA_DIR", "DEFAULT_OUTPUT_DIR",
    "FEATURE_COLUMNS", "NUMERIC_FEATURES", "RANDOM_SEED", "TARGET_COLUMN", "VALIDATION_SHARE",
    "WOMEN_TEST_SHARE", "XaDataBundle", "load_xa_test_and_rest", "make_match_level_cv_splits",
    "summarize_split",
]


def load_xa_test_and_rest() -> XaDataBundle:
    """Load prepared xA rows using the single canonical data/split policy."""
    return _load_xa_test_and_rest(
        data_dir=DEFAULT_DATA_DIR,
        women_test_share=WOMEN_TEST_SHARE,
        seed=RANDOM_SEED,
        minimum_positive_test_games=3,
    )


def _load_xa_test_and_rest(
    data_dir: Path = DEFAULT_DATA_DIR,
    women_test_share: float = WOMEN_TEST_SHARE,
    seed: int = RANDOM_SEED,
    minimum_positive_test_games: int = 3,
) -> XaDataBundle:
    """Load prepared xA rows and isolate the women final test set by match."""
    women = _prepare_original_xa_rows(_read_cached_passes(data_dir, "women"))
    men = _prepare_original_xa_rows(_read_cached_passes(data_dir, "men2015"))

    women_test_games = _deterministic_game_split(
        women,
        share=women_test_share,
        seed=seed,
        label="women_final_test",
        minimum_positive_games=minimum_positive_test_games,
    )
    women_rest = women.loc[~women["game_id"].isin(women_test_games)].copy()
    women_test = women.loc[women["game_id"].isin(women_test_games)].copy()

    _assert_disjoint_game_sets({"women_rest": _game_ids(women_rest), "women_final_test": _game_ids(women_test)})
    _assert_no_event_overlap({"women_rest": women_rest, "women_final_test": women_test})
    _assert_label_definition(men)
    _assert_label_definition(women_rest)
    _assert_label_definition(women_test)

    split_summary = {
        "men_rest": summarize_split(men),
        "women_rest": summarize_split(women_rest),
        "women_final_test": summarize_split(women_test),
    }

    return XaDataBundle(
        men_rest=men,
        women_rest=women_rest,
        women_test=women_test,
        women_test_games=women_test_games,
        split_summary=split_summary,
    )


def make_match_level_cv_splits(
    frame: pd.DataFrame,
    cv_folds: int,
    seed: int,
    dataset: str,
    target_column: str = TARGET_COLUMN,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create CV folds where whole matches are assigned to each validation fold."""
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2.")
    group_count = frame["game_id"].nunique()
    if cv_folds > group_count:
        raise ValueError(f"cv_folds={cv_folds} exceeds {dataset} match count={group_count}.")

    _assert_label_definition(frame)
    y_values = frame[target_column].astype(int).to_numpy()
    groups = frame["game_id"].astype(int).to_numpy()
    splitters = [
        StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=seed),
        GroupKFold(n_splits=cv_folds),
    ]
    last_error: ValueError | None = None
    for splitter in splitters:
        try:
            splits = list(splitter.split(frame, y_values, groups=groups))
            _validate_match_level_cv_splits(splits, groups, y_values, dataset)
            return splits
        except ValueError as error:
            last_error = error

    message = f"Could not create {cv_folds} valid match-level folds for {dataset}."
    if last_error is not None:
        message = f"{message} Last error: {last_error}"
    raise ValueError(message)


def _validate_match_level_cv_splits(
    splits: list[tuple[np.ndarray, np.ndarray]],
    groups: np.ndarray,
    y_values: np.ndarray,
    dataset: str,
) -> None:
    for fold_index, (train_indices, validation_indices) in enumerate(splits, start=1):
        train_games = set(int(game_id) for game_id in groups[train_indices])
        validation_games = set(int(game_id) for game_id in groups[validation_indices])
        _assert_disjoint_game_sets(
            {
                f"{dataset}_fold_{fold_index}_train": train_games,
                f"{dataset}_fold_{fold_index}_validation": validation_games,
            }
        )
        validation_positives = int(y_values[validation_indices].sum())
        if validation_positives <= 0:
            raise ValueError(f"{dataset} fold {fold_index} has no positive assist labels; reduce cv_folds.")


def _is_missing_value(value) -> bool:
    """Treat None/NaN/pandas missing scalars as missing, but leave lists/dicts alone."""
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if missing is pd.NA:
        return True
    if isinstance(missing, (bool, np.bool_)):
        return bool(missing)
    return False


def _value_or_default(row: pd.Series, column_name: str, default=None):
    """Return a flat dataframe value, falling back only when missing."""
    if column_name not in row.index:
        return default
    value = row[column_name]
    return default if _is_missing_value(value) else value


def _require_columns(frame: pd.DataFrame, required_columns: list[str], frame_label: str) -> None:
    """Fail fast when competition_events does not return the expected flat schema."""
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"{frame_label} missing expected flat columns: {missing_columns}")


def _location_xy(location) -> tuple[float, float]:
    """Extract x/y from the expected StatsBomb [x, y] location."""
    return float(location[0]), float(location[1])


def _distance_and_angle_from_goal_statsbomb(x_coordinate: float, y_coordinate: float) -> tuple[float, float]:
    """Compute distance and visible goal angle from StatsBomb 120x80 coordinates."""
    spadl_x = x_coordinate * 105 / 120
    spadl_y = y_coordinate * 68 / 80
    dx = 105 - spadl_x
    dy = 34 - spadl_y

    distance = np.hypot(dx, dy)
    angle = abs(np.arctan2(7.32 * dx, dx**2 + dy**2 - (7.32 / 2) ** 2))
    return distance, angle


def _boolean_flag(value) -> bool:
    """Convert optional StatsBomb boolean fields to clean booleans."""
    return False if _is_missing_value(value) else bool(value)


def _pass_context(pass_type: str, play_pattern: str) -> str:
    """Collapse pass type and play pattern into broad chance-creation contexts."""
    if pass_type == "Corner" or play_pattern == "From Corner":
        return "Corner"
    if pass_type == "Free Kick" or play_pattern == "From Free Kick":
        return "Free Kick"
    if pass_type == "Throw-in" or play_pattern == "From Throw In":
        return "Throw-in"
    if pass_type in {"Goal Kick", "Kick Off"} or play_pattern in {"From Goal Kick", "From Kick Off"}:
        return "Restart"
    if play_pattern == "Regular Play" or pass_type == "Regular Pass":
        return "Open Play"
    return "Other"


def _pass_end_zone(end_x: float, end_y: float) -> str:
    """Bucket the pass end location into broad attacking zones."""
    if end_x >= 102 and 18 <= end_y <= 62:
        return "Penalty Box"
    if end_x >= 80:
        if end_y < 26.7:
            return "Final Third Left"
        if end_y > 53.3:
            return "Final Third Right"
        return "Final Third Central"
    if end_x >= 40:
        return "Middle Third"
    return "Defensive Third"


def _build_shot_xg_lookups(shot_events: pd.DataFrame) -> tuple[dict, dict]:
    """Build shot-id -> xG and key-pass-id -> xG lookup tables."""
    shot_xg_by_id = {}
    key_pass_xg_by_id = {}

    for _, shot_event in shot_events.iterrows():
        statsbomb_xg = shot_event["shot_statsbomb_xg"]
        if _is_missing_value(statsbomb_xg):
            continue
        statsbomb_xg = float(statsbomb_xg)

        shot_xg_by_id[shot_event["id"]] = statsbomb_xg

        key_pass_id = _value_or_default(shot_event, "shot_key_pass_id")
        if not _is_missing_value(key_pass_id):
            key_pass_xg_by_id[key_pass_id] = statsbomb_xg

    return shot_xg_by_id, key_pass_xg_by_id


def _event_to_xa_row(event: pd.Series, league_name: str, shot_xg_by_id: dict, key_pass_xg_by_id: dict) -> dict:
    """Convert one StatsBomb pass event into one xA modeling row."""
    start_x, start_y = _location_xy(event["location"])
    end_x, end_y = _location_xy(event["pass_end_location"])

    pass_length = _value_or_default(event, "pass_length")
    if _is_missing_value(pass_length):
        pass_length = float(np.hypot(end_x - start_x, end_y - start_y))
    else:
        pass_length = float(pass_length)

    pass_angle = _value_or_default(event, "pass_angle")
    if _is_missing_value(pass_angle):
        pass_angle = float(np.arctan2(end_y - start_y, end_x - start_x))
    else:
        pass_angle = float(pass_angle)

    start_distance, start_goal_angle = _distance_and_angle_from_goal_statsbomb(start_x, start_y)
    end_distance, end_goal_angle = _distance_and_angle_from_goal_statsbomb(end_x, end_y)

    play_pattern = _value_or_default(event, "play_pattern", "Unknown") or "Unknown"
    pass_type = _value_or_default(event, "pass_type", "Regular Pass") or "Regular Pass"
    pass_height = _value_or_default(event, "pass_height", "Unknown") or "Unknown"
    pass_body_part = _value_or_default(event, "pass_body_part", "Unknown") or "Unknown"
    pass_technique = _value_or_default(event, "pass_technique", "None") or "None"
    pass_outcome = _value_or_default(event, "pass_outcome", "Complete") or "Complete"

    pass_id = event["id"]
    assisted_shot_id = _value_or_default(event, "pass_assisted_shot_id")

    assisted_shot_xg = 0.0
    if not _is_missing_value(assisted_shot_id) and assisted_shot_id in shot_xg_by_id:
        assisted_shot_xg = shot_xg_by_id[assisted_shot_id]
    elif pass_id in key_pass_xg_by_id:
        assisted_shot_xg = key_pass_xg_by_id[pass_id]

    shot_assist_flag = _boolean_flag(_value_or_default(event, "pass_shot_assist", False))
    goal_assist_flag = _boolean_flag(_value_or_default(event, "pass_goal_assist", False))

    return {
        "event_id": pass_id,
        "assisted_shot_id": assisted_shot_id,
        "game_id": int(event["match_id"]),
        "player_id": event["player_id"],
        "player_name": _value_or_default(event, "player", "Unknown") or "Unknown",
        "team_name": _value_or_default(event, "team", "Unknown") or "Unknown",
        "position_name": _value_or_default(event, "position", "Unknown") or "Unknown",
        "league": league_name,
        "start_x": start_x,
        "start_y": start_y,
        "end_x": end_x,
        "end_y": end_y,
        "delta_x": end_x - start_x,
        "delta_y": end_y - start_y,
        "pass_length": pass_length,
        "pass_angle": pass_angle,
        "start_distance_to_goal": start_distance,
        "end_distance_to_goal": end_distance,
        "start_angle_to_goal": start_goal_angle,
        "end_angle_to_goal": end_goal_angle,
        "distance_reduction_to_goal": start_distance - end_distance,
        "pass_type": pass_type,
        "pass_height": pass_height,
        "pass_body_part": pass_body_part,
        "pass_technique": pass_technique,
        "pass_outcome": pass_outcome,
        "play_pattern": play_pattern,
        "pass_context": _pass_context(pass_type, play_pattern),
        "end_zone": _pass_end_zone(end_x, end_y),
        "under_pressure": _boolean_flag(_value_or_default(event, "under_pressure", False)),
        "is_cross": _boolean_flag(_value_or_default(event, "pass_cross", False)),
        "is_cut_back": _boolean_flag(_value_or_default(event, "pass_cut_back", False)),
        "is_switch": _boolean_flag(_value_or_default(event, "pass_switch", False)),
        "is_through_ball": _boolean_flag(_value_or_default(event, "pass_through_ball", False)),
        "is_inswinging": _boolean_flag(_value_or_default(event, "pass_inswinging", False)),
        "is_outswinging": _boolean_flag(_value_or_default(event, "pass_outswinging", False)),
        "end_in_box": end_x >= 102 and 18 <= end_y <= 62,
        "end_in_six_yard_box": end_x >= 114 and 30 <= end_y <= 50,
        "is_shot_assist": bool(assisted_shot_xg > 0 or shot_assist_flag or goal_assist_flag),
        "is_goal_assist": goal_assist_flag,
        "assisted_shot_xg": float(assisted_shot_xg),
    }


def _load_competition_event_xa_passes(data_dir: Path, competitions: list[dict], cache_tag: str) -> pd.DataFrame:
    """Load pass + shot events with competition_events and cache xA rows."""
    data_dir.mkdir(parents=True, exist_ok=True)
    passes_cache = data_dir / f"{cache_tag}_xa_passes_competition_events_method.pkl"
    if passes_cache.exists():
        return pd.read_pickle(passes_cache).copy()

    pass_rows = []
    for competition in competitions:
        try:
            competition_passes = sb.competition_events(
                country=competition["country"],
                division=competition["division"],
                season=competition["season"],
                gender=competition["gender"],
                fmt="dataframe",
                filters={"type": "Pass"},
            )
            competition_shots = sb.competition_events(
                country=competition["country"],
                division=competition["division"],
                season=competition["season"],
                gender=competition["gender"],
                fmt="dataframe",
                filters={"type": "Shot"},
            )
        except Exception as error:
            raise RuntimeError(f"Failed to load pass/shot events for {competition['league']}") from error

        _require_columns(competition_passes, FLAT_PASS_REQUIRED_COLUMNS, f"{competition['league']} pass events")
        _require_columns(competition_shots, FLAT_SHOT_REQUIRED_COLUMNS, f"{competition['league']} shot events")
        shot_xg_by_id, key_pass_xg_by_id = _build_shot_xg_lookups(competition_shots)

        print(
            f"[load competition_events passes] {competition['league']}: "
            f"{len(competition_passes):,} passes | {len(competition_shots):,} shots"
        )

        league_rows = [
            _event_to_xa_row(pass_event, competition["league"], shot_xg_by_id, key_pass_xg_by_id)
            for _, pass_event in competition_passes.iterrows()
        ]
        pass_rows.extend(league_rows)

    if not pass_rows:
        raise FileNotFoundError(f"Could not rebuild xA pass cache: {passes_cache}")

    passes = pd.DataFrame(pass_rows)
    if passes[["game_id", "player_id"]].isna().any().any():
        raise ValueError("Loaded pass rows contain missing game_id or player_id.")

    passes.to_pickle(passes_cache)
    print(f"[cache] wrote {cache_tag} competition_events xA passes: {len(passes):,} passes")
    return passes.copy()


def _read_cached_passes(data_dir: Path, cache_tag: str) -> pd.DataFrame:
    path = data_dir / f"{cache_tag}_xa_passes_competition_events_method.pkl"
    if not path.exists():
        competitions = COMPETITIONS_BY_CACHE_TAG.get(cache_tag)
        if competitions is None:
            raise FileNotFoundError(f"Missing xA pass cache and no competition list for {cache_tag}: {path}")
        frame = _load_competition_event_xa_passes(data_dir, competitions, cache_tag)
    else:
        frame = pd.read_pickle(path)
    required = set(FEATURE_COLUMNS) - _DERIVED_FEATURES
    required.update({"game_id", "pass_outcome", "is_goal_assist"})
    missing = sorted(column for column in required if column not in frame.columns)
    if missing:
        raise ValueError(f"{path} missing expected columns: {missing}")
    return frame


def _bin_feature(series: pd.Series, bins: Iterable[float], labels: Iterable[str]) -> pd.Series:
    return pd.cut(series, bins=list(bins), labels=list(labels), include_lowest=True).astype("object").fillna("Unknown")


def _prepare_original_xa_rows(raw_passes: pd.DataFrame) -> pd.DataFrame:
    """Keep completed passes and label each row by whether it became a goal assist."""
    passes = raw_passes.loc[raw_passes["pass_outcome"].eq("Complete")].copy()
    passes[TARGET_COLUMN] = passes["is_goal_assist"].fillna(False).astype(int)

    passes["abs_delta_y"] = passes["delta_y"].abs()
    passes["abs_pass_angle"] = passes["pass_angle"].abs()
    passes["start_distance_to_center"] = (passes["start_y"] - 40.0).abs()
    passes["end_distance_to_center"] = (passes["end_y"] - 40.0).abs()
    passes["goal_angle_gain"] = passes["end_angle_to_goal"] - passes["start_angle_to_goal"]
    passes["progress_to_goal_rate"] = passes["distance_reduction_to_goal"] / passes["pass_length"].replace(0, np.nan)
    passes["progress_to_goal_rate"] = passes["progress_to_goal_rate"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    passes["end_x_squared"] = (passes["end_x"] / 120.0) ** 2
    passes["end_y_centered_squared"] = ((passes["end_y"] - 40.0) / 40.0) ** 2

    passes["is_forward_pass"] = passes["delta_x"] > 0
    passes["is_backward_pass"] = passes["delta_x"] < 0
    passes["is_central_end"] = passes["end_y"].between(26.7, 53.3)
    passes["is_final_third_end"] = passes["end_x"] >= 80.0

    x_labels = ["defensive", "middle", "advanced", "final_third", "box_edge", "box"]
    x_bins = [-0.001, 40.0, 70.0, 90.0, 102.0, 114.0, 120.001]
    y_labels = ["left_wide", "left_half", "central", "right_half", "right_wide"]
    y_bins = [-0.001, 16.0, 29.0, 51.0, 64.0, 80.001]
    passes["start_x_band"] = _bin_feature(passes["start_x"], x_bins, x_labels)
    passes["end_x_band"] = _bin_feature(passes["end_x"], x_bins, x_labels)
    passes["start_y_band"] = _bin_feature(passes["start_y"], y_bins, y_labels)
    passes["end_y_band"] = _bin_feature(passes["end_y"], y_bins, y_labels)
    passes["distance_reduction_band"] = _bin_feature(
        passes["distance_reduction_to_goal"],
        [-np.inf, -15.0, -3.0, 3.0, 10.0, 25.0, np.inf],
        ["large_negative", "negative", "flat", "small_gain", "progressive", "major_progressive"],
    )
    passes["pass_length_band"] = _bin_feature(
        passes["pass_length"],
        [-0.001, 5.0, 15.0, 30.0, 45.0, np.inf],
        ["very_short", "short", "medium", "long", "very_long"],
    )
    passes["pass_angle_band"] = _bin_feature(
        np.degrees(passes["pass_angle"]),
        [-181.0, -120.0, -45.0, 45.0, 120.0, 181.0],
        ["backward_negative", "wide_negative", "forward", "wide_positive", "backward_positive"],
    )

    for column in ["pass_type", "pass_height", "pass_body_part", "pass_technique", "play_pattern", "pass_context", "end_zone"]:
        passes[column] = passes[column].fillna("Unknown").astype(str)

    passes["context_height"] = passes["pass_context"] + "|" + passes["pass_height"]
    passes["context_body_part"] = passes["pass_context"] + "|" + passes["pass_body_part"]
    passes["context_end_zone"] = passes["pass_context"] + "|" + passes["end_zone"]
    _assert_label_definition(passes)
    return passes


def _assert_label_definition(frame: pd.DataFrame) -> None:
    if not frame["pass_outcome"].eq("Complete").all():
        raise AssertionError("xA modeling rows must contain only completed passes.")
    expected_target = frame["is_goal_assist"].fillna(False).astype(int)
    actual_target = frame[TARGET_COLUMN].astype(int)
    if not actual_target.equals(expected_target):
        raise AssertionError(f"{TARGET_COLUMN} must exactly equal is_goal_assist as a binary target.")
    invalid_values = sorted(set(actual_target.unique()) - {0, 1})
    if invalid_values:
        raise AssertionError(f"{TARGET_COLUMN} contains non-binary values: {invalid_values}")


def _game_ids(frame: pd.DataFrame) -> set[int]:
    return set(int(game_id) for game_id in frame["game_id"].unique())


def _assert_disjoint_game_sets(named_game_sets: dict[str, set[int]]) -> None:
    names = list(named_game_sets)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            overlap = named_game_sets[left_name] & named_game_sets[right_name]
            if overlap:
                sample = sorted(overlap)[:10]
                raise AssertionError(f"Match-level leakage between {left_name} and {right_name}: {sample}")


def _assert_no_event_overlap(named_frames: dict[str, pd.DataFrame]) -> None:
    frames_with_event_ids = {
        name: set(frame["event_id"].dropna().astype(str))
        for name, frame in named_frames.items()
        if "event_id" in frame.columns
    }
    names = list(frames_with_event_ids)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            overlap = frames_with_event_ids[left_name] & frames_with_event_ids[right_name]
            if overlap:
                sample = sorted(overlap)[:10]
                raise AssertionError(f"Pass-event leakage between {left_name} and {right_name}: {sample}")


def _deterministic_game_split(
    frame: pd.DataFrame,
    share: float,
    seed: int,
    label: str,
    forbidden_game_ids: set[int] | None = None,
    minimum_positive_games: int = 1,
) -> set[int]:
    forbidden_game_ids = forbidden_game_ids or set()
    available_games = np.array(sorted(set(frame["game_id"].unique()) - forbidden_game_ids))
    if len(available_games) == 0:
        raise ValueError(f"No games available for {label} split.")
    split_size = max(1, int(round(len(available_games) * share)))
    split_size = min(split_size, len(available_games))

    positive_by_game = frame.groupby("game_id")[TARGET_COLUMN].sum()
    for offset in range(1000):
        rng = np.random.default_rng(seed + offset)
        selected = set(int(game_id) for game_id in rng.choice(available_games, size=split_size, replace=False))
        positive_games = int((positive_by_game.reindex(list(selected), fill_value=0) > 0).sum())
        positives = int(positive_by_game.reindex(list(selected), fill_value=0).sum())
        if positive_games >= minimum_positive_games and positives > 0:
            return selected
    raise ValueError(f"Could not create {label} split with positive examples after 1000 attempts.")


def summarize_split(frame: pd.DataFrame) -> dict:
    """Row/match/positive counts for one split, used in run summaries."""
    positives = int(frame[TARGET_COLUMN].sum())
    rows = int(len(frame))
    return {
        "rows": rows,
        "games": int(frame["game_id"].nunique()),
        "positives": positives,
        "prevalence": float(positives / rows) if rows else 0.0,
    }
