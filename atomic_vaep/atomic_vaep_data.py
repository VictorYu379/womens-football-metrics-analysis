"""Stable data interface for Atomic-VAEP model tuning.

Atomic-VAEP uses full Atomic-SPADL action sequences. This module prepares
features and labels on complete match sequences before any row sampling or model
training can happen, then reserves the women final test set at match level.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np

if not hasattr(np, "string_"):
    np.string_ = np.bytes_
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128

import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from socceraction.data.statsbomb import StatsBombLoader
import socceraction.spadl as spadl
import socceraction.atomic.spadl as atomic_spadl
import socceraction.atomic.vaep.features as fs
import socceraction.atomic.vaep.labels as lab

warnings.filterwarnings("ignore", category=FutureWarning)
_CHAINED_ASSIGNMENT_ERROR = getattr(pd.errors, "ChainedAssignmentError", None)
if _CHAINED_ASSIGNMENT_ERROR is not None:
    warnings.filterwarnings("ignore", category=_CHAINED_ASSIGNMENT_ERROR)

RANDOM_SEED = 42
WOMEN_TEST_SHARE = 0.30
VALIDATION_SHARE = 0.20
CV_FOLDS = 4

ALIGN_ATTACKING_DIRECTION = True
ATTACKING_DIRECTION = "left_to_right"
N_PREVIOUS_ACTIONS = 5
VAEP_LABEL_HORIZON = 10
TARGET_COLUMNS = ("scores", "concedes")

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
DEFAULT_DATA_DIR = _PROJECT_ROOT / "vaep_data"
DEFAULT_OPEN_DATA_DIR = _PROJECT_ROOT / "open-data" / "data"
DEFAULT_OUTPUT_DIR = _SCRIPT_DIR / "hist_gradient_boosting_tuning_results"

WOMEN_CACHE_TAG = "women"
MEN_CACHE_TAG = "men2015"

WOMEN_COMPETITIONS = [
    {"competition_id": 37, "season_id": 281, "league": "WSL"},
    {"competition_id": 135, "season_id": 281, "league": "Frauen Bundesliga"},
    {"competition_id": 182, "season_id": 281, "league": "Liga F"},
    {"competition_id": 131, "season_id": 281, "league": "Serie A Women"},
    {"competition_id": 49, "season_id": 107, "league": "NWSL"},
]

MEN_COMPETITIONS = [
    {"competition_id": 11, "season_id": 27, "league": "La Liga"},
    {"competition_id": 2, "season_id": 27, "league": "Premier League"},
    {"competition_id": 12, "season_id": 27, "league": "Serie A"},
    {"competition_id": 7, "season_id": 27, "league": "Ligue 1"},
]

COMPETITIONS_BY_CACHE_TAG = {
    WOMEN_CACHE_TAG: WOMEN_COMPETITIONS,
    MEN_CACHE_TAG: MEN_COMPETITIONS,
}

ATOMIC_VAEP_FEATURE_FUNCTIONS = [
    fs.actiontype,
    fs.actiontype_onehot,
    fs.bodypart,
    fs.bodypart_onehot,
    fs.time,
    fs.team,
    fs.time_delta,
    fs.location,
    fs.polar,
    fs.movement_polar,
    fs.direction,
    fs.goalscore,
]

ACTION_ROW_COLUMNS = (
    "game_id",
    "original_event_id",
    "action_id",
    "period_id",
    "time_seconds",
    "team_id",
    "player_id",
    "x",
    "y",
    "dx",
    "dy",
    "type_id",
    "bodypart_id",
    "league",
    "type_name",
    "bodypart_name",
)
ROW_METADATA_COLUMNS = ("dataset", "player_id_key") + ACTION_ROW_COLUMNS
NON_FEATURE_COLUMNS = frozenset(ROW_METADATA_COLUMNS).union(TARGET_COLUMNS)


@dataclass(frozen=True)
class AtomicVaepDataBundle:
    men_full: pd.DataFrame
    women_train: pd.DataFrame
    women_test: pd.DataFrame
    women_test_games: set[int]
    split_summary: dict[str, dict]
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...] = TARGET_COLUMNS
    n_previous_actions: int = N_PREVIOUS_ACTIONS
    label_horizon: int = VAEP_LABEL_HORIZON
    split_unit: str = "match"
    attacking_direction: str = ATTACKING_DIRECTION


def load_atomic_vaep_test_and_rest() -> AtomicVaepDataBundle:
    """Load prepared Atomic-VAEP rows using the canonical split policy."""
    return _load_atomic_vaep_test_and_rest(
        data_dir=DEFAULT_DATA_DIR,
        open_data_dir=DEFAULT_OPEN_DATA_DIR,
        women_test_share=WOMEN_TEST_SHARE,
        seed=RANDOM_SEED,
    )


def _load_atomic_vaep_test_and_rest(
    data_dir: Path = DEFAULT_DATA_DIR,
    open_data_dir: Path = DEFAULT_OPEN_DATA_DIR,
    women_test_share: float = WOMEN_TEST_SHARE,
    seed: int = RANDOM_SEED,
    minimum_positive_test_games: int = 1,
) -> AtomicVaepDataBundle:
    """Load Atomic-VAEP rows and reserve women final test matches up front."""
    women = _read_or_build_atomic_vaep_rows(data_dir, open_data_dir, WOMEN_CACHE_TAG)
    men = _read_or_build_atomic_vaep_rows(data_dir, open_data_dir, MEN_CACHE_TAG)
    men, women, feature_columns = _align_feature_columns(men, women)

    women_test_games = _deterministic_game_split(
        women,
        share=women_test_share,
        seed=seed,
        label="women_final_test",
        target_columns=TARGET_COLUMNS,
        minimum_positive_games=minimum_positive_test_games,
    )
    women_train = women.loc[~women["game_id"].isin(women_test_games)].copy()
    women_test = women.loc[women["game_id"].isin(women_test_games)].copy()

    _assert_match_level_holdout(women_train, women_test, "women_train", "women_final_test")
    _assert_atomic_vaep_rows(men, "men_full", feature_columns)
    _assert_atomic_vaep_rows(women_train, "women_train", feature_columns)
    _assert_atomic_vaep_rows(women_test, "women_final_test", feature_columns)

    split_summary = {
        "men_full": summarize_split(men),
        "women_train": summarize_split(women_train),
        "women_final_test": summarize_split(women_test),
    }

    return AtomicVaepDataBundle(
        men_full=men,
        women_train=women_train,
        women_test=women_test,
        women_test_games=women_test_games,
        split_summary=split_summary,
        feature_columns=feature_columns,
    )


def load_atomic_actions(
    competitions: list[dict],
    cache_tag: str,
    data_dir: Path = DEFAULT_DATA_DIR,
    open_data_dir: Path = DEFAULT_OPEN_DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Atomic-SPADL actions, rebuilding from aligned SPADL when needed."""
    data_dir.mkdir(parents=True, exist_ok=True)
    atomic_cache = _atomic_cache_path(data_dir, cache_tag)
    positions_cache = data_dir / f"{cache_tag}_pos.parquet"
    if atomic_cache.exists():
        print(f"[cache] {cache_tag} Atomic-SPADL actions: {atomic_cache.name}")
        actions = pd.read_parquet(atomic_cache).copy()
        positions = pd.read_parquet(positions_cache).copy() if positions_cache.exists() else _empty_positions()
        return add_atomic_names_if_needed(actions), positions

    actions = rebuild_atomic_cache_from_spadl(data_dir, open_data_dir, competitions, cache_tag)
    positions = pd.read_parquet(positions_cache).copy() if positions_cache.exists() else _empty_positions()
    return add_atomic_names_if_needed(actions), positions


def rebuild_atomic_cache_from_spadl(
    data_dir: Path,
    open_data_dir: Path,
    competitions: list[dict],
    cache_tag: str,
) -> pd.DataFrame:
    """Rebuild Atomic-SPADL cache from the left-to-right SPADL cache."""
    spadl_cache = _spadl_cache_path(data_dir, cache_tag)
    if spadl_cache.exists():
        print(f"[rebuild] {cache_tag}: reading aligned SPADL cache {spadl_cache.name}")
        spadl_actions = pd.read_parquet(spadl_cache).copy()
    else:
        spadl_actions, _ = load_spadl_actions(
            competitions,
            cache_tag,
            data_dir=data_dir,
            open_data_dir=open_data_dir,
        )

    league_by_game = (
        spadl_actions.drop_duplicates("game_id").set_index("game_id")["league"].to_dict()
        if "league" in spadl_actions.columns
        else {}
    )
    atomic_frames = []
    game_ids = spadl_actions["game_id"].drop_duplicates().tolist()
    for game_index, (game_id, game_actions) in enumerate(spadl_actions.groupby("game_id", sort=False), start=1):
        game_actions = game_actions.sort_values(["period_id", "time_seconds", "action_id"]).reset_index(drop=True)
        atomic_actions = atomic_spadl.convert_to_atomic(game_actions)
        if league_by_game:
            atomic_actions["league"] = league_by_game.get(game_id)
        atomic_frames.append(atomic_actions)
        if game_index % 50 == 0 or game_index == len(game_ids):
            print(f"    atomic converted {game_index}/{len(game_ids)} games")

    atomic_actions = add_atomic_names_if_needed(pd.concat(atomic_frames, ignore_index=True))
    atomic_cache = _atomic_cache_path(data_dir, cache_tag)
    atomic_actions.to_parquet(atomic_cache, index=False)
    print(f"[cache] wrote {atomic_cache.name}: {len(atomic_actions):,} Atomic-SPADL actions")
    return atomic_actions.copy()


def load_spadl_actions(
    competitions: list[dict],
    cache_tag: str,
    data_dir: Path = DEFAULT_DATA_DIR,
    open_data_dir: Path = DEFAULT_OPEN_DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load left-to-right SPADL actions needed to rebuild Atomic-SPADL."""
    if not ALIGN_ATTACKING_DIRECTION:
        raise ValueError("Atomic-VAEP preparation requires left-to-right attacking-direction alignment.")
    spadl_cache = _spadl_cache_path(data_dir, cache_tag)
    positions_cache = data_dir / f"{cache_tag}_pos.parquet"
    if spadl_cache.exists():
        actions = pd.read_parquet(spadl_cache).copy()
        positions = pd.read_parquet(positions_cache).copy() if positions_cache.exists() else _empty_positions()
        return add_spadl_names_if_needed(actions), positions

    statsbomb_loader = _statsbomb_loader(open_data_dir)
    action_frames = []
    position_frames = []
    for competition in competitions:
        league = competition["league"]
        try:
            games = statsbomb_loader.games(
                competition_id=competition["competition_id"],
                season_id=competition["season_id"],
            )
        except Exception as error:
            print(f"  ! {league}: {error}")
            continue

        print(f"[load] {league}: {len(games)} games")
        for game_index, (_, game) in enumerate(games.iterrows(), start=1):
            try:
                events = statsbomb_loader.events(game.game_id)
                events = make_events_compatible_with_socceraction(events)
                actions = spadl.statsbomb.convert_to_actions(events, home_team_id=game.home_team_id)
                actions = spadl.play_left_to_right(actions, home_team_id=game.home_team_id)
                actions["league"] = league
                action_frames.append(actions)
                if "position_name" in events.columns:
                    position_frames.append(events[["player_id", "position_name"]].dropna())
                if game_index % 25 == 0 or game_index == len(games):
                    print(f"    converted {game_index}/{len(games)} games")
            except Exception as error:
                print(f"  ! skipped game {game.game_id}: {error}")

    if not action_frames:
        raise RuntimeError(f"No SPADL actions loaded for {cache_tag}.")
    actions = pd.concat(action_frames, ignore_index=True)
    positions = pd.concat(position_frames, ignore_index=True) if position_frames else _empty_positions()
    actions.to_parquet(spadl_cache, index=False)
    positions.to_parquet(positions_cache, index=False)
    print(f"[cache] wrote {cache_tag} SPADL actions: {len(actions):,} rows")
    return add_spadl_names_if_needed(actions), positions


def make_atomic_vaep_rows(actions: pd.DataFrame, dataset_label: str) -> pd.DataFrame:
    """Build Atomic-VAEP rows from complete game sequences."""
    actions = add_atomic_names_if_needed(actions).reset_index(drop=True)
    row_frames = []
    for game_id, game_actions in actions.groupby("game_id", sort=False):
        game_actions = game_actions.reset_index(drop=True)
        gamestates = fs.gamestates(game_actions, N_PREVIOUS_ACTIONS)
        game_features = pd.concat(
            [feature_function(gamestates) for feature_function in ATOMIC_VAEP_FEATURE_FUNCTIONS],
            axis=1,
        ).fillna(0)
        game_labels = pd.concat(
            [
                lab.scores(game_actions, VAEP_LABEL_HORIZON),
                lab.concedes(game_actions, VAEP_LABEL_HORIZON),
            ],
            axis=1,
        ).fillna(0)
        game_labels = game_labels.loc[:, list(TARGET_COLUMNS)].astype(int)

        if len(game_features) != len(game_labels) or len(game_features) != len(game_actions):
            raise ValueError(
                f"Atomic-VAEP alignment mismatch in game {game_id}: "
                f"actions={len(game_actions)}, features={len(game_features)}, labels={len(game_labels)}"
            )

        metadata = game_actions.copy()
        metadata.insert(0, "dataset", dataset_label)
        metadata["player_id_key"] = normalize_player_id_series(metadata["player_id"])
        reserved_columns = set(metadata.columns).union(TARGET_COLUMNS)
        overlap = set(game_features.columns).intersection(reserved_columns)
        if overlap:
            raise ValueError(f"Atomic-VAEP feature columns overlap metadata/targets: {sorted(overlap)}")

        row_frames.append(
            pd.concat(
                [
                    metadata.reset_index(drop=True),
                    game_labels.reset_index(drop=True),
                    compact_feature_dtypes(game_features).reset_index(drop=True),
                ],
                axis=1,
            )
        )

    if not row_frames:
        raise ValueError(f"No Atomic-VAEP rows created for {dataset_label}.")
    return pd.concat(row_frames, ignore_index=True)


def add_atomic_names_if_needed(actions: pd.DataFrame) -> pd.DataFrame:
    """Ensure numeric Atomic-SPADL ids also have readable name columns."""
    required_name_columns = {"type_name", "bodypart_name"}
    if required_name_columns.issubset(actions.columns):
        return actions.copy()
    return atomic_spadl.add_names(actions.copy())


def add_spadl_names_if_needed(actions: pd.DataFrame) -> pd.DataFrame:
    """Ensure numeric SPADL ids also have readable name columns."""
    required_name_columns = {"type_name", "result_name", "bodypart_name"}
    if required_name_columns.issubset(actions.columns):
        return actions.copy()
    return spadl.add_names(actions.copy())


def make_events_compatible_with_socceraction(events: pd.DataFrame) -> pd.DataFrame:
    """Return an events copy that socceraction can convert under pandas/NumPy 2.x."""
    events = events.copy()
    for column in events.columns:
        if pd.api.types.is_string_dtype(events[column].dtype):
            events[column] = events[column].astype(object)
    return events


def compact_feature_dtypes(features: pd.DataFrame) -> pd.DataFrame:
    """Reduce Atomic-VAEP feature-memory footprint before writing caches."""
    features = features.copy()
    float_columns = features.select_dtypes(include=["float64"]).columns
    integer_columns = features.select_dtypes(include=["int64", "int32"]).columns
    if len(float_columns):
        features[float_columns] = features[float_columns].astype("float32")
    for column in integer_columns:
        features[column] = pd.to_numeric(features[column], downcast="integer")
    return features


def action_rows_from_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return Atomic-SPADL action columns from a prepared frame."""
    columns = [column for column in ACTION_ROW_COLUMNS if column in frame.columns]
    return frame.loc[:, columns].copy()


def normalize_player_id_series(player_ids: pd.Series) -> pd.Series:
    """Convert player ids from different loaders/caches to one nullable integer key."""
    return pd.to_numeric(player_ids, errors="coerce").astype("Int64")


def load_position_rows(data_dir: Path = DEFAULT_DATA_DIR, cache_tag: str = WOMEN_CACHE_TAG) -> pd.DataFrame:
    """Load cached player-position observations for a dataset."""
    candidates = [
        data_dir / f"{cache_tag}_pos.parquet",
        data_dir / f"{cache_tag}_positions_competition_events_method.pkl",
        data_dir / f"{cache_tag}_positions_statsbombpy_remote.pkl",
    ]
    for path in candidates:
        if path.exists():
            positions = pd.read_parquet(path).copy() if path.suffix == ".parquet" else pd.read_pickle(path).copy()
            break
    else:
        return _empty_positions()
    required_columns = {"player_id", "position_name"}
    missing_columns = required_columns.difference(positions.columns)
    if missing_columns:
        raise ValueError(f"Position cache missing columns: {sorted(missing_columns)}")
    positions = positions.loc[:, ["player_id", "position_name"]].dropna().copy()
    positions["player_id_key"] = normalize_player_id_series(positions["player_id"])
    positions["position_name"] = positions["position_name"].astype(str)
    return positions.dropna(subset=["player_id_key", "position_name"])


def build_primary_position_map(position_rows: pd.DataFrame) -> dict[int, str]:
    """Choose the most frequent observed position for each player."""
    if position_rows.empty:
        return {}
    positions = position_rows.copy()
    if "player_id_key" not in positions.columns:
        positions["player_id_key"] = normalize_player_id_series(positions["player_id"])
    return (
        positions.dropna(subset=["player_id_key"])
        .groupby("player_id_key")["position_name"]
        .agg(lambda values: values.value_counts().index[0])
        .to_dict()
    )


def bucket_position(position_name: str) -> str:
    """Map granular StatsBomb positions to broad tactical buckets."""
    position_name = str(position_name)
    if "Goalkeeper" in position_name:
        return "Goalkeeper"
    if "Center Back" in position_name:
        return "Center Back"
    if "Wing Back" in position_name or position_name in {"Left Back", "Right Back"}:
        return "Fullback / Wing Back"
    if "Defensive Midfield" in position_name:
        return "Defensive Midfield"
    if "Center Midfield" in position_name:
        return "Central Midfield"
    if "Attacking Midfield" in position_name:
        return "Attacking Midfield"
    if position_name in {"Left Midfield", "Right Midfield", "Left Wing", "Right Wing"}:
        return "Wide Midfielder / Winger"
    if "Center Forward" in position_name or position_name == "Striker":
        return "Forward"
    return "Other / Unknown"


def make_match_level_cv_splits(
    frame: pd.DataFrame,
    cv_folds: int,
    seed: int,
    dataset: str,
    target_column: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create CV folds where whole matches are assigned to validation folds."""
    if target_column not in TARGET_COLUMNS:
        raise ValueError(f"Unknown Atomic-VAEP target column: {target_column}")
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2.")
    group_count = frame["game_id"].nunique()
    if cv_folds > group_count:
        raise ValueError(f"cv_folds={cv_folds} exceeds {dataset} match count={group_count}.")

    _assert_binary_target(frame, target_column, dataset)
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
            _validate_match_level_cv_splits(splits, groups, y_values, dataset, target_column)
            return splits
        except ValueError as error:
            last_error = error

    message = f"Could not create {cv_folds} valid match-level folds for {dataset} {target_column}."
    if last_error is not None:
        message = f"{message} Last error: {last_error}"
    raise ValueError(message)


def summarize_split(frame: pd.DataFrame) -> dict:
    """Row/match/label counts for one split."""
    rows = int(len(frame))
    summary = {
        "rows": rows,
        "games": int(frame["game_id"].nunique()),
    }
    for target_column in TARGET_COLUMNS:
        positives = int(frame[target_column].sum())
        summary[f"{target_column}_positives"] = positives
        summary[f"{target_column}_prevalence"] = float(positives / rows) if rows else 0.0
    return summary


def feature_columns_from_frame(frame: pd.DataFrame) -> tuple[str, ...]:
    """Infer Atomic-VAEP model-feature columns from a prepared frame."""
    return tuple(str(column) for column in frame.columns if column not in NON_FEATURE_COLUMNS)


def _read_or_build_atomic_vaep_rows(data_dir: Path, open_data_dir: Path, cache_tag: str) -> pd.DataFrame:
    rows_cache = _atomic_vaep_rows_cache_path(data_dir, cache_tag)
    if rows_cache.exists():
        print(f"[cache] {cache_tag} Atomic-VAEP rows: {rows_cache.name}")
        return pd.read_parquet(rows_cache).copy()
    competitions = COMPETITIONS_BY_CACHE_TAG.get(cache_tag)
    if competitions is None:
        raise FileNotFoundError(f"Missing Atomic-VAEP row cache and no competitions for {cache_tag}: {rows_cache}")
    actions, _ = load_atomic_actions(competitions, cache_tag, data_dir=data_dir, open_data_dir=open_data_dir)
    rows = make_atomic_vaep_rows(actions, cache_tag)
    data_dir.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(rows_cache, index=False)
    print(f"[cache] wrote {cache_tag} Atomic-VAEP rows: {len(rows):,} actions")
    return rows.copy()


def _align_feature_columns(men: pd.DataFrame, women: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    feature_columns = tuple(pd.Index(feature_columns_from_frame(women)).union(feature_columns_from_frame(men)))
    aligned = []
    for frame in (men.copy(), women.copy()):
        for column in feature_columns:
            if column not in frame.columns:
                frame[column] = 0
        aligned.append(frame)
    return aligned[0], aligned[1], feature_columns


def _statsbomb_loader(open_data_dir: Path) -> StatsBombLoader:
    if open_data_dir.exists():
        print(f"using local StatsBomb open data: {open_data_dir}")
        return StatsBombLoader(getter="local", root=str(open_data_dir))
    print("using remote StatsBomb open data; first SPADL load can be slow")
    return StatsBombLoader(getter="remote", creds={})


def _atomic_cache_path(data_dir: Path, cache_tag: str) -> Path:
    direction_suffix = "ltr" if ALIGN_ATTACKING_DIRECTION else "home_perspective"
    return data_dir / f"{cache_tag}_atomic_spadl_{direction_suffix}.parquet"


def _spadl_cache_path(data_dir: Path, cache_tag: str) -> Path:
    direction_suffix = "ltr" if ALIGN_ATTACKING_DIRECTION else "home_perspective"
    return data_dir / f"{cache_tag}_spadl_{direction_suffix}.parquet"


def _atomic_vaep_rows_cache_path(data_dir: Path, cache_tag: str) -> Path:
    direction_suffix = "ltr" if ALIGN_ATTACKING_DIRECTION else "home_perspective"
    return data_dir / f"{cache_tag}_atomic_vaep_rows_{direction_suffix}_n{N_PREVIOUS_ACTIONS}_h{VAEP_LABEL_HORIZON}.parquet"


def _empty_positions() -> pd.DataFrame:
    return pd.DataFrame(columns=["player_id", "position_name"])


def _deterministic_game_split(
    frame: pd.DataFrame,
    share: float,
    seed: int,
    label: str,
    target_columns: tuple[str, ...],
    minimum_positive_games: int = 1,
) -> set[int]:
    if not 0 < share < 1:
        raise ValueError("share must be between 0 and 1.")
    available_games = np.array(sorted(frame["game_id"].dropna().astype(int).unique()))
    if len(available_games) < 2:
        raise ValueError(f"Need at least two games for {label} split.")
    split_size = max(1, int(round(len(available_games) * share)))
    split_size = min(split_size, len(available_games) - 1)
    positive_by_game = frame.groupby("game_id")[list(target_columns)].sum()
    for offset in range(1000):
        rng = np.random.default_rng(seed + offset)
        selected = set(int(game_id) for game_id in rng.choice(available_games, size=split_size, replace=False))
        remaining = set(int(game_id) for game_id in available_games) - selected
        if _split_has_label_coverage(positive_by_game, selected, target_columns, minimum_positive_games) and _split_has_label_coverage(
            positive_by_game,
            remaining,
            target_columns,
            minimum_positive_games,
        ):
            return selected
    raise ValueError(f"Could not create {label} split with positives for all Atomic-VAEP targets after 1000 attempts.")


def _split_has_label_coverage(
    positive_by_game: pd.DataFrame,
    game_ids: set[int],
    target_columns: tuple[str, ...],
    minimum_positive_games: int,
) -> bool:
    if not game_ids:
        return False
    positives = positive_by_game.reindex(list(game_ids), fill_value=0)
    for target_column in target_columns:
        if int(positives[target_column].sum()) <= 0:
            return False
        if int((positives[target_column] > 0).sum()) < minimum_positive_games:
            return False
    return True


def _assert_match_level_holdout(train_frame: pd.DataFrame, test_frame: pd.DataFrame, train_name: str, test_name: str) -> None:
    _assert_disjoint_game_sets({train_name: _game_ids(train_frame), test_name: _game_ids(test_frame)})
    if train_frame.empty or test_frame.empty:
        raise ValueError(f"{train_name}/{test_name} match-level split produced an empty frame.")


def _assert_atomic_vaep_rows(frame: pd.DataFrame, label: str, feature_columns: tuple[str, ...]) -> None:
    required_columns = {"game_id", *TARGET_COLUMNS, *feature_columns}
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} Atomic-VAEP rows missing columns: {missing[:20]}")
    for target_column in TARGET_COLUMNS:
        _assert_binary_target(frame, target_column, label)
    if frame["game_id"].isna().any():
        raise ValueError(f"{label} contains rows without game_id.")


def _assert_binary_target(frame: pd.DataFrame, target_column: str, label: str) -> None:
    if frame[target_column].isna().any():
        raise ValueError(f"{label} {target_column} target contains missing values.")
    target_values = pd.to_numeric(frame[target_column], errors="coerce")
    invalid_targets = sorted(set(target_values.dropna().astype(int).unique()) - {0, 1})
    if invalid_targets:
        raise ValueError(f"{label} {target_column} target contains non-binary values: {invalid_targets}")
    positives = int(target_values.sum())
    if positives <= 0 or positives >= len(target_values):
        raise ValueError(f"{label} {target_column} target must contain both classes.")


def _game_ids(frame: pd.DataFrame) -> set[int]:
    return set(int(game_id) for game_id in frame["game_id"].dropna().unique())


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
    target_column: str,
) -> None:
    validation_indices_seen: set[int] = set()
    for fold_index, (train_indices, validation_indices) in enumerate(splits, start=1):
        train_games = set(int(game_id) for game_id in groups[train_indices])
        validation_games = set(int(game_id) for game_id in groups[validation_indices])
        _assert_disjoint_game_sets(
            {
                f"{dataset}_{target_column}_fold_{fold_index}_train": train_games,
                f"{dataset}_{target_column}_fold_{fold_index}_validation": validation_games,
            }
        )
        _assert_fold_has_both_classes(y_values[train_indices], dataset, target_column, fold_index, "train")
        _assert_fold_has_both_classes(y_values[validation_indices], dataset, target_column, fold_index, "validation")
        validation_indices_seen.update(int(index) for index in validation_indices)
    if validation_indices_seen != set(range(len(groups))):
        raise ValueError(f"{dataset} {target_column} CV validation folds do not cover every row exactly once.")


def _assert_fold_has_both_classes(
    y_values: np.ndarray,
    dataset: str,
    target_column: str,
    fold_index: int,
    split_name: str,
) -> None:
    positives = int(y_values.sum())
    if positives <= 0:
        raise ValueError(f"{dataset} {target_column} fold {fold_index} {split_name} split has no positives.")
    if positives >= len(y_values):
        raise ValueError(f"{dataset} {target_column} fold {fold_index} {split_name} split has no negatives.")
