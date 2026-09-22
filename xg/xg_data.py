"""Stable data interface for xG hyperparameter tuning.

This module is intentionally self-contained for the xG workflow. It reads the
shot caches produced by the xG notebooks and exposes match-level train/test and
cross-validation helpers for XGBoost tuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from statsbombpy import sb

RANDOM_SEED = 42
WOMEN_TEST_SHARE = 0.30
CV_FOLDS = 4
TARGET_COLUMN = "goal"

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
DEFAULT_DATA_DIR = _PROJECT_ROOT / "vaep_data"
DEFAULT_OUTPUT_DIR = _SCRIPT_DIR / "xgboost_tuning_results"

NUMERIC_FEATURES = ["distance", "angle"]
BOOLEAN_FEATURES = ["under_pressure", "first_time"]
CATEGORICAL_FEATURES = ["body_part", "shot_technique", "shot_type", "play_pattern", "shot_context"]
FEATURE_COLUMNS = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES

REQUIRED_COLUMNS = ["game_id", "player_id", "shot_x", "shot_y", "statsbomb_xg", TARGET_COLUMN] + FEATURE_COLUMNS

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


@dataclass
class XgDataBundle:
    men_full: pd.DataFrame
    women_train: pd.DataFrame
    women_test: pd.DataFrame
    women_test_games: set[int]
    split_summary: dict[str, dict]
    target_column: str = TARGET_COLUMN
    feature_columns: tuple[str, ...] = tuple(FEATURE_COLUMNS)
    numeric_features: tuple[str, ...] = tuple(NUMERIC_FEATURES)
    boolean_features: tuple[str, ...] = tuple(BOOLEAN_FEATURES)
    categorical_features: tuple[str, ...] = tuple(CATEGORICAL_FEATURES)


def load_xg_test_and_rest() -> XgDataBundle:
    """Load prepared xG rows using the single canonical data/split policy."""
    return _load_xg_test_and_rest(
        data_dir=DEFAULT_DATA_DIR,
        women_test_share=WOMEN_TEST_SHARE,
        seed=RANDOM_SEED,
    )


def _load_xg_test_and_rest(
    data_dir: Path = DEFAULT_DATA_DIR,
    women_test_share: float = WOMEN_TEST_SHARE,
    seed: int = RANDOM_SEED,
) -> XgDataBundle:
    """Load prepared xG shot rows and isolate the women test set by match."""
    women = _read_cached_shots(data_dir, "women")
    men = _read_cached_shots(data_dir, "men2015")
    _assert_xg_rows(women, "women")
    _assert_xg_rows(men, "men2015")

    women_games = _read_women_game_ids(data_dir, women)
    women_test_games = _deterministic_game_split(women_games, women_test_share, seed)
    women_train = women.loc[~women["game_id"].isin(women_test_games)].copy()
    women_test = women.loc[women["game_id"].isin(women_test_games)].copy()
    _assert_disjoint_game_sets({"women_train": _game_ids(women_train), "women_test": _game_ids(women_test)})

    split_summary = {
        "men_full": summarize_split(men),
        "women_train": summarize_split(women_train),
        "women_test": summarize_split(women_test),
    }
    return XgDataBundle(
        men_full=men,
        women_train=women_train,
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
        message += f" Last error: {last_error}"
    raise ValueError(message)


def summarize_split(frame: pd.DataFrame) -> dict:
    """Row/match/goal counts for one split."""
    goals = int(frame[TARGET_COLUMN].sum())
    rows = int(len(frame))
    return {
        "rows": rows,
        "games": int(frame["game_id"].nunique()),
        "goals": goals,
        "goal_rate": float(goals / rows) if rows else 0.0,
    }


def _load_remote_matches(data_dir: Path, competitions: list[dict], cache_tag: str) -> pd.DataFrame:
    """Load StatsBomb match lists and cache them for canonical split creation."""
    data_dir.mkdir(parents=True, exist_ok=True)
    matches_cache = data_dir / f"{cache_tag}_matches_competition_events_method.pkl"
    if matches_cache.exists():
        return pd.read_pickle(matches_cache).copy()

    match_frames = []
    for competition in competitions:
        try:
            matches = sb.matches(
                competition_id=competition["competition_id"],
                season_id=competition["season_id"],
            )
        except Exception as error:
            print(f"  ! matches {competition['league']}: {error}")
            continue

        matches = matches.copy()
        matches["league"] = competition["league"]
        match_frames.append(matches)
        print(f"[load matches] {competition['league']}: {len(matches):,} games")

    if not match_frames:
        raise FileNotFoundError(f"Could not rebuild xG match cache: {matches_cache}")

    matches = pd.concat(match_frames, ignore_index=True)
    matches.to_pickle(matches_cache)
    print(f"[cache] wrote {cache_tag} matches: {len(matches):,} games")
    return matches.copy()


def _is_missing_value(value) -> bool:
    """Treat None/NaN/pandas missing scalars as missing, but leave lists/dicts alone."""
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, (bool, np.bool_)):
        return bool(missing)
    return False


def _statsbomb_name(value):
    """Return the StatsBomb object name, or the value itself if it is already flat."""
    if isinstance(value, dict):
        return value.get("name")
    return None if _is_missing_value(value) else value


def _first_available_value(row: pd.Series, column_names: list[str], default=None):
    """Return the first non-missing value among candidate dataframe columns."""
    for column_name in column_names:
        if column_name in row.index:
            value = row[column_name]
            if not _is_missing_value(value):
                return value
    return default


def _shot_coordinates(row: pd.Series) -> tuple[float, float]:
    """Extract StatsBomb shot x/y coordinates from either list or flattened columns."""
    location = _first_available_value(row, ["location"])
    if isinstance(location, (list, tuple, np.ndarray)) and len(location) >= 2:
        return location[0], location[1]

    return (
        _first_available_value(row, ["location_x", "x", "start_x"], np.nan),
        _first_available_value(row, ["location_y", "y", "start_y"], np.nan),
    )


def _distance_and_angle_from_statsbomb_coordinates(shot_x: float, shot_y: float) -> tuple[float, float]:
    """Compute distance/angle from StatsBomb 120x80 coordinates."""
    spadl_x = shot_x * 105 / 120
    spadl_y = shot_y * 68 / 80
    dx = 105 - spadl_x
    dy = 34 - spadl_y

    distance = np.hypot(dx, dy)
    angle = abs(np.arctan2(7.32 * dx, dx**2 + dy**2 - (7.32 / 2) ** 2))
    return distance, angle


def _shot_context(shot_type: str, play_pattern: str) -> str:
    """Collapse shot type and play pattern into broad shot context."""
    if shot_type == "Penalty":
        return "Penalty"
    if shot_type == "Free Kick" or play_pattern == "From Free Kick":
        return "Free Kick"
    if play_pattern in {"From Corner", "From Throw In", "From Keeper", "From Goal Kick", "From Kick Off"}:
        return "Set Piece"
    if shot_type == "Open Play" or play_pattern == "Regular Play":
        return "Open Play"
    return "Other"


def _boolean_flag(value) -> bool:
    """Convert optional StatsBomb boolean fields to clean booleans."""
    return False if _is_missing_value(value) else bool(value)


def _event_to_xg_row(event: pd.Series, league_name: str) -> dict:
    """Convert one StatsBomb shot event into one xG modeling row."""
    shot_x, shot_y = _shot_coordinates(event)
    distance, angle = _distance_and_angle_from_statsbomb_coordinates(shot_x, shot_y)

    shot_type = _statsbomb_name(_first_available_value(event, ["shot_type"], "Unknown"))
    play_pattern = _statsbomb_name(_first_available_value(event, ["play_pattern"], "Unknown"))
    body_part = _statsbomb_name(_first_available_value(event, ["shot_body_part"], "Unknown"))
    technique = _statsbomb_name(_first_available_value(event, ["shot_technique"], "Unknown"))
    outcome = _statsbomb_name(_first_available_value(event, ["shot_outcome"], "Unknown"))
    statsbomb_xg = _first_available_value(event, ["shot_statsbomb_xg"], np.nan)

    return {
        "game_id": int(_first_available_value(event, ["match_id"])),
        "player_id": _first_available_value(event, ["player_id"]),
        "league": league_name,
        "shot_x": shot_x,
        "shot_y": shot_y,
        "distance": distance,
        "angle": angle,
        "body_part": body_part or "Unknown",
        "shot_technique": technique or "Unknown",
        "shot_type": shot_type or "Unknown",
        "play_pattern": play_pattern or "Unknown",
        "shot_context": _shot_context(shot_type, play_pattern),
        "under_pressure": _boolean_flag(_first_available_value(event, ["under_pressure"], False)),
        "first_time": _boolean_flag(_first_available_value(event, ["shot_first_time"], False)),
        "statsbomb_xg": np.nan if _is_missing_value(statsbomb_xg) else float(statsbomb_xg),
        TARGET_COLUMN: int(outcome == "Goal"),
    }


def _load_competition_event_shots(data_dir: Path, competitions: list[dict], cache_tag: str) -> pd.DataFrame:
    """Load shot events with sb.competition_events and cache xG modeling rows."""
    data_dir.mkdir(parents=True, exist_ok=True)
    shots_cache = data_dir / f"{cache_tag}_xg_shots_competition_events_method.pkl"
    if shots_cache.exists():
        return pd.read_pickle(shots_cache).copy()

    shot_rows = []
    for competition in competitions:
        try:
            competition_shots = sb.competition_events(
                country=competition["country"],
                division=competition["division"],
                season=competition["season"],
                gender=competition["gender"],
                filters={"type": "Shot"},
            )
        except Exception as error:
            print(f"  ! shots {competition['league']}: {error}")
            continue

        print(f"[load competition_events shots] {competition['league']}: {len(competition_shots):,} shots")
        for _, shot_event in competition_shots.iterrows():
            shot_rows.append(_event_to_xg_row(shot_event, competition["league"]))

    if not shot_rows:
        raise FileNotFoundError(f"Could not rebuild xG shot cache: {shots_cache}")

    shots = pd.DataFrame(shot_rows)
    shots.to_pickle(shots_cache)
    print(f"[cache] wrote {cache_tag} competition_events xG shots: {len(shots):,} shots")
    return shots.copy()


def _read_cached_shots(data_dir: Path, cache_tag: str) -> pd.DataFrame:
    path = data_dir / f"{cache_tag}_xg_shots_competition_events_method.pkl"
    if not path.exists():
        competitions = COMPETITIONS_BY_CACHE_TAG.get(cache_tag)
        if competitions is None:
            raise FileNotFoundError(f"Missing xG shot cache and no competition list for {cache_tag}: {path}")
        return _load_competition_event_shots(data_dir, competitions, cache_tag)
    return pd.read_pickle(path).copy()


def _read_women_game_ids(data_dir: Path, women_shots: pd.DataFrame) -> np.ndarray:
    matches_path = data_dir / "women_matches_competition_events_method.pkl"
    if matches_path.exists():
        matches = pd.read_pickle(matches_path)
        return np.sort(matches["match_id"].dropna().astype(int).unique())
    try:
        matches = _load_remote_matches(data_dir, WOMEN_COMPETITIONS, "women")
        return np.sort(matches["match_id"].dropna().astype(int).unique())
    except FileNotFoundError:
        pass
    return np.sort(women_shots["game_id"].dropna().astype(int).unique())


def _deterministic_game_split(game_ids: np.ndarray, share: float, seed: int) -> set[int]:
    if not 0 < share < 1:
        raise ValueError("women_test_share must be between 0 and 1.")
    split_size = int(len(game_ids) * share)
    split_size = max(1, min(split_size, len(game_ids) - 1))
    rng = np.random.default_rng(seed)
    return set(int(game_id) for game_id in rng.choice(game_ids, size=split_size, replace=False))


def _assert_xg_rows(frame: pd.DataFrame, label: str) -> None:
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} xG rows missing columns: {missing}")
    invalid_targets = sorted(set(frame[TARGET_COLUMN].dropna().astype(int).unique()) - {0, 1})
    if invalid_targets:
        raise ValueError(f"{label} target contains non-binary values: {invalid_targets}")
    if frame[TARGET_COLUMN].isna().any():
        raise ValueError(f"{label} target contains missing values.")


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


def _validate_match_level_cv_splits(
    splits: list[tuple[np.ndarray, np.ndarray]],
    groups: np.ndarray,
    y_values: np.ndarray,
    dataset: str,
) -> None:
    validation_indices_seen: set[int] = set()
    for fold_index, (train_indices, validation_indices) in enumerate(splits, start=1):
        train_games = set(int(game_id) for game_id in groups[train_indices])
        validation_games = set(int(game_id) for game_id in groups[validation_indices])
        _assert_disjoint_game_sets(
            {
                f"{dataset}_fold_{fold_index}_train": train_games,
                f"{dataset}_fold_{fold_index}_validation": validation_games,
            }
        )
        validation_goals = int(y_values[validation_indices].sum())
        if validation_goals <= 0:
            raise ValueError(f"{dataset} fold {fold_index} has no goals; reduce cv_folds.")
        validation_indices_seen.update(int(index) for index in validation_indices)
    if validation_indices_seen != set(range(len(groups))):
        raise ValueError(f"{dataset} CV validation folds do not cover every row exactly once.")
