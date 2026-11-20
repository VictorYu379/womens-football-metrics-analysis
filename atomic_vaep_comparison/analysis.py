import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from inference import TARGETS, checksum


PASS_ACTION_TYPES = ("pass", "cross", "throw_in", "freekick", "corner", "goalkick")
PASS_HEIGHT_ORDER = ("Ground Pass", "Low Pass", "High Pass", "Unknown")


def load_inputs(home):
    home = Path(home)
    exported = json.loads((home / "export_manifest.json").read_text(encoding="utf-8"))
    if checksum(home / "inputs/manifest.json") != exported["input_manifest_sha256"]:
        raise ValueError("Input manifest changed after export")
    manifest = json.loads((home / "inputs/manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files_sha256"].items():
        if checksum(home / "inputs" / name) != expected:
            raise ValueError(f"Analysis input changed: {name}")
    features = np.load(home / "inputs/test_features.npy", mmap_mode="r")
    actions = pd.read_parquet(home / "inputs/test_actions.parquet")
    players = pd.read_parquet(home / "inputs/player_context.parquet")
    if features.shape != (len(actions), 256) or len(actions) != manifest["actions"]:
        raise ValueError("Action and feature rows do not align")
    if set(actions.game_id) != set(manifest["test_games"]):
        raise ValueError("Wrong test matches")
    if set(manifest["test_games"]) & set(manifest["development_games"]):
        raise ValueError("Test and development games overlap")
    if actions.duplicated(["game_id", "action_id"]).any() or players.player_id.duplicated().any():
        raise ValueError("Duplicate action/player identifiers")
    if not set(actions.player_id).issubset(set(players.player_id)):
        raise ValueError("Missing player context")
    return actions, features, players, manifest


def probability_metrics(actions, probabilities, model):
    rows = []
    for column, target in enumerate(TARGETS):
        truth = actions[target].to_numpy(dtype=int)
        probability = probabilities[:, column]
        rows.append({"model": model, "target": target, "actions": len(truth),
                     "positive_rate": truth.mean(), "log_loss": log_loss(truth, probability, labels=[0, 1]),
                     "brier": brier_score_loss(truth, probability), "roc_auc": roc_auc_score(truth, probability),
                     "error_rate_at_05": np.mean((probability >= .5) != truth),
                     "always_negative_error_rate": truth.mean()})
    return rows


def atomic_values(actions, probabilities):
    for name, value in (("string_", np.bytes_), ("float_", np.float64), ("complex_", np.complex128)):
        if not hasattr(np, name):
            setattr(np, name, value)
    from socceraction.atomic.vaep import formula
    actions = actions.reset_index(drop=True)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(actions), 2) or len(actions) == 0:
        raise ValueError("Each action needs scoring and conceding probabilities")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Invalid probabilities")
    identifiers = ["game_id", "action_id", "team_id", "period_id", "time_seconds"]
    if actions[identifiers].isna().any().any() or actions.duplicated(["game_id", "action_id"]).any():
        raise ValueError("Invalid atomic sequence identifiers")
    values = np.full((len(actions), 3), np.nan)
    for _, game in actions.groupby("game_id", sort=False):
        if not game.action_id.is_monotonic_increasing:
            raise ValueError("Actions must retain their full within-match order")
        positions = game.index.to_numpy()
        result = formula.value(game.reset_index(drop=True), pd.Series(probabilities[positions, 0]),
                               pd.Series(probabilities[positions, 1]))
        values[positions] = result[["offensive_value", "defensive_value", "vaep_value"]].to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite atomic valuations")
    np.testing.assert_allclose(values[:, 0] + values[:, 1], values[:, 2], atol=1e-14)
    return pd.DataFrame(values, columns=["offensive_value", "defensive_value", "vaep_value"])


def player_values(actions, valuations, context, min_actions=100, min_minutes=45):
    grouped = actions.groupby("player_id").size().rename("action_count").to_frame()
    for model, values in valuations.items():
        if len(values) != len(actions):
            raise ValueError("Valuations and actions do not align")
        totals = values.assign(player_id=actions.player_id.to_numpy()).groupby("player_id").sum()
        for column, suffix in (("vaep_value", "total"), ("offensive_value", "offensive"), ("defensive_value", "defensive")):
            grouped[f"{model}_{suffix}"] = totals[column]
    all_players = context.merge(grouped, on="player_id", how="left", validate="one_to_one")
    all_players[grouped.columns] = all_players[grouped.columns].fillna(0)
    eligible = all_players.loc[all_players.action_count.ge(min_actions)
                               & ~all_players.position_bucket.eq("Goalkeeper")].copy()
    if eligible.position_bucket.isna().any() or (~np.isfinite(eligible.test_minutes) | eligible.test_minutes.le(0)).any():
        raise ValueError("Eligible players need valid positions and minutes")
    eligible = eligible.loc[eligible.test_minutes.ge(min_minutes)].copy()
    if eligible.empty:
        raise ValueError("No players meet the common action/minutes thresholds")
    for model in valuations:
        eligible[f"{model}_per90"] = eligible[f"{model}_total"] * 90 / eligible.test_minutes
        for component in ("offensive", "defensive"):
            eligible[f"{model}_{component}_per90"] = eligible[f"{model}_{component}"] * 90 / eligible.test_minutes
        for measure in ("total", "per90"):
            eligible[f"{model}_{measure}_rank"] = eligible[f"{model}_{measure}"].rank(ascending=False, method="min")
            eligible[f"{model}_{measure}_position_rank"] = eligible.groupby("position_bucket")[f"{model}_{measure}"].rank(ascending=False, method="min")
    return all_players, eligible.sort_values("player_id").reset_index(drop=True)


def spearman(left, right):
    if len(left) < 2 or pd.Series(left).nunique() < 2 or pd.Series(right).nunique() < 2:
        return np.nan
    return float(spearmanr(left, right).statistic)


def top_players(players, column, top_k=15):
    if not isinstance(top_k, (int, np.integer)) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    ranks = players[column].rank(ascending=False, method="min")
    return players.loc[ranks.le(top_k)].sort_values([column, "player_id"], ascending=[False, True])


def ranking_metrics(left_scores, right_scores, top_k=10):
    if not isinstance(top_k, (int, np.integer)) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if len(left_scores) != len(right_scores) or not np.isfinite(np.column_stack([left_scores, right_scores])).all():
        raise ValueError("Rankings require aligned, finite scores")
    left_scores = pd.Series(np.asarray(left_scores))
    right_scores = pd.Series(np.asarray(right_scores))
    left_rank = left_scores.rank(ascending=False, method="min")
    right_rank = right_scores.rank(ascending=False, method="min")
    delta = left_rank - right_rank
    left_top, right_top = left_rank.le(top_k), right_rank.le(top_k)
    fixed_size = left_top.sum() == top_k and right_top.sum() == top_k
    overlap = int((left_top & right_top).sum())
    correlation_available = len(left_scores) >= 2 and left_scores.nunique() > 1 and right_scores.nunique() > 1
    return {
        "spearman": spearman(left_scores, right_scores),
        "kendall_tau_b": float(kendalltau(left_scores, right_scores, variant="b").statistic) if correlation_available else np.nan,
        "mean_abs_rank_change": delta.abs().mean(),
        "mean_abs_rank_change_fraction": delta.abs().mean() / (len(delta) - 1) if len(delta) > 1 else np.nan,
        "exact_rank_agreement_pct": 100 * delta.eq(0).mean(),
        "within_2_rank_agreement_pct": 100 * delta.abs().le(2).mean(),
        "within_5_rank_agreement_pct": 100 * delta.abs().le(5).mean(),
        "mean_value_difference": (left_scores - right_scores).mean(),
        "median_value_difference": (left_scores - right_scores).median(),
        "mean_abs_value_difference": (left_scores - right_scores).abs().mean(),
        "top_k": top_k, "top_overlap": overlap,
        "top_overlap_fraction": overlap / top_k if fixed_size and len(delta) > top_k else np.nan,
        "boundary_tie": len(delta) >= top_k and not fixed_size,
        "shortlist_is_entire_group": len(delta) <= top_k,
    }


def ranking_summary(players, models, measure="per90", top_k=10):
    rows = []
    for position, group in [("All eligible", players), *players.groupby("position_bucket", sort=True)]:
        for left, right in combinations(models, 2):
            left_scores, right_scores = group[f"{left}_{measure}"], group[f"{right}_{measure}"]
            rows.append({"measure": measure, "position": position, "left": left, "right": right,
                         "players": len(group), "scored_actions": int(group.action_count.sum()),
                         **ranking_metrics(left_scores, right_scores, top_k)})
    return pd.DataFrame(rows)


def shortlists(players, models, measure="per90", top_k=15):
    parts = []
    for _, group in players.groupby("position_bucket", sort=True):
        if len(group) < top_k:
            continue
        for model in models:
            selected = top_players(group, f"{model}_{measure}", top_k)
            part = selected[["player_id", "player_name", "position_bucket", "test_minutes", "action_count"]].copy()
            part["model"] = model
            part["measure"] = measure
            part["list_order"] = np.arange(1, len(part) + 1)
            part["value"] = selected[f"{model}_{measure}"].to_numpy()
            part["position_rank"] = group[f"{model}_{measure}"].rank(ascending=False, method="min").loc[selected.index].to_numpy()
            part["boundary_tie"] = len(part) > top_k
            parts.append(part)
    columns = ["player_id", "player_name", "position_bucket", "test_minutes", "action_count", "model", "measure",
               "list_order", "value", "position_rank", "boundary_tie"]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)


def comparison_actions(actions, valuations, left, right):
    work = actions.reset_index(drop=True).copy()
    for model in (left, right):
        if len(valuations[model]) != len(work):
            raise ValueError("Action values must align with the complete action table")
    work["left_value"] = valuations[left].vaep_value.to_numpy()
    work["right_value"] = valuations[right].vaep_value.to_numpy()
    work["vaep_delta"] = work.left_value - work.right_value
    for component in ("offensive", "defensive"):
        work[f"{component}_delta"] = (valuations[left][f"{component}_value"].to_numpy()
                                      - valuations[right][f"{component}_value"].to_numpy())
    return work


def distribution_summary(actions, group_columns, display_limit=.01):
    work = actions.assign(absolute_delta=actions.vaep_delta.abs(),
                          outside_display=actions.vaep_delta.abs().gt(display_limit))
    return work.groupby(group_columns, observed=True, dropna=False).agg(
        actions=("vaep_delta", "size"), total_delta=("vaep_delta", "sum"),
        mean_delta=("vaep_delta", "mean"), median_delta=("vaep_delta", "median"), std_delta=("vaep_delta", "std"),
        mean_abs_delta=("absolute_delta", "mean"), p95_abs_delta=("absolute_delta", lambda values: values.quantile(.95)),
        left_mean=("left_value", "mean"), right_mean=("right_value", "mean"),
        offensive_delta=("offensive_delta", "mean"), defensive_delta=("defensive_delta", "mean"),
        outside_display=("outside_display", "sum"),
    ).reset_index()


def attach_pass_heights(actions, events):
    candidates = actions.loc[actions.type_name.isin(PASS_ACTION_TYPES)].copy()
    keys = ["game_id", "original_event_id"]
    if candidates.original_event_id.isna().any() or candidates.duplicated(["game_id", "action_id"]).any():
        raise ValueError("Passing atoms need unique action IDs and original event references")
    matched = candidates.merge(events, on=keys, how="left", validate="many_to_one", indicator=True, sort=False)
    if matched._merge.ne("both").any() or not matched.event_type.isin(["Pass", "Shot"]).all():
        raise ValueError("Missing or invalid original events for candidate passing atoms")
    if (matched.event_type.eq("Shot") & matched.type_name.ne("freekick")).any():
        raise ValueError("Only freekick atoms can reference shots among passing types")
    passed = matched.loc[matched.event_type.eq("Pass")].drop(columns="_merge").copy()
    if not passed.pass_height.isin(PASS_HEIGHT_ORDER).all():
        raise ValueError("Pass height must be Ground, Low, High or explicit Unknown")
    passed["type_name"] = pd.Categorical(passed.type_name, categories=PASS_ACTION_TYPES, ordered=True)
    passed["pass_height"] = pd.Categorical(passed.pass_height, categories=PASS_HEIGHT_ORDER, ordered=True)
    return passed


def shortlist_scope(players, top_k=15):
    scope = players.groupby("position_bucket").size().rename("players").reset_index()
    scope["included"] = scope.players.ge(top_k)
    scope["top_k"] = top_k
    scope["min_possible_overlap_pct"] = np.where(scope.included, 100 * (2 * top_k - scope.players).clip(lower=0) / top_k, np.nan)
    return scope.sort_values(["players", "position_bucket"], ascending=[False, True]).reset_index(drop=True)


def compare_top_rankings(players, left_column, right_column, top_k=15):
    if not isinstance(top_k, (int, np.integer)) or top_k < 1 or len(players) < top_k:
        raise ValueError("Top-list comparisons need a positive top_k and a sufficiently large position group")
    if players.player_id.isna().any() or players.player_id.duplicated().any():
        raise ValueError("Top-list comparisons require unique player IDs")
    if not np.isfinite(players[[left_column, right_column]].to_numpy()).all():
        raise ValueError("Top-list comparisons require finite scores")
    work = players[["player_id", "player_name", "position_bucket", "test_minutes", "action_count"]].copy()
    work["left_value"], work["right_value"] = players[left_column], players[right_column]
    work["left_rank"] = players[left_column].rank(ascending=False, method="min")
    work["right_rank"] = players[right_column].rank(ascending=False, method="min")
    work["rank_delta"] = work.left_rank - work.right_rank
    work["left_in_top"], work["right_in_top"] = work.left_rank.le(top_k), work.right_rank.le(top_k)
    left_count, right_count = int(work.left_in_top.sum()), int(work.right_in_top.sum())
    shared = int((work.left_in_top & work.right_in_top).sum())
    boundary_tie = left_count != top_k or right_count != top_k
    summary = {"players": len(work), "top_k": top_k, "left_selected": left_count, "right_selected": right_count,
               "shared_players": shared, "left_only_players": left_count - shared, "right_only_players": right_count - shared,
               "overlap_pct": 100 * shared / top_k if not boundary_tie else np.nan, "boundary_tie": boundary_tie,
               "min_possible_overlap_pct": 100 * max(0, 2 * top_k - len(work)) / top_k}
    union = work.loc[work.left_in_top | work.right_in_top].copy()
    union["best_rank"] = union[["left_rank", "right_rank"]].min(axis=1)
    return summary, union.sort_values(["best_rank", "left_rank", "right_rank", "player_id"])


def positional_top_comparison(players, left, right, measure, top_k=15):
    summaries, details = [], []
    for position, group in players.groupby("position_bucket", sort=True):
        if len(group) < top_k:
            continue
        summary, union = compare_top_rankings(group, f"{left}_{measure}", f"{right}_{measure}", top_k)
        labels = {"position_bucket": position, "measure": measure, "left": left, "right": right}
        summaries.append({**labels, **summary})
        details.append(union.assign(**labels))
    return pd.DataFrame(summaries), pd.concat(details, ignore_index=True) if details else pd.DataFrame()


def normalization_top_comparison(players, models, top_k=15):
    summaries, details = [], []
    for position, group in players.groupby("position_bucket", sort=True):
        if len(group) < top_k:
            continue
        for model in models:
            summary, union = compare_top_rankings(group, f"{model}_total", f"{model}_per90", top_k)
            summaries.append({"position_bucket": position, "model": model, **summary,
                              "retained_players": summary["shared_players"], "entrants": summary["right_only_players"],
                              "exits": summary["left_only_players"]})
            union = union.rename(columns={"left_rank": "total_rank", "right_rank": "per90_rank",
                                          "left_value": "total_value", "right_value": "per90_value",
                                          "left_in_top": "in_total_top", "right_in_top": "in_per90_top",
                                          "rank_delta": "rank_improvement_per90"})
            details.append(union.assign(model=model))
    return pd.DataFrame(summaries), pd.concat(details, ignore_index=True) if details else pd.DataFrame()


def rank_movement_counts(players, left, right, measure, limit=10):
    records = []
    for position, group in players.groupby("position_bucket", sort=True):
        left_rank = group[f"{left}_{measure}"].rank(ascending=False, method="min")
        right_rank = group[f"{right}_{measure}"].rank(ascending=False, method="min")
        movement = (left_rank - right_rank).astype(int)
        counts = movement.value_counts().reindex(range(-limit, limit + 1), fill_value=0)
        for delta, count in counts.items():
            records.append({"position_bucket": position, "measure": measure, "rank_delta": delta, "count": int(count),
                            "players": len(group), "outside_display": int(movement.abs().gt(limit).sum())})
    return pd.DataFrame(records)


def action_summary(actions, valuations, pairs):
    parts = []
    for left, right in pairs:
        work = comparison_actions(actions, valuations, left, right)
        summary = distribution_summary(work, ["type_name"])
        summary["left"], summary["right"] = left, right
        parts.append(summary)
    return pd.concat(parts, ignore_index=True)


def spatial_summary(actions, valuations, pairs, event_types=("shot", "pass", "receival", "dribble", "cross", "take_on"), columns=16, rows=12):
    if not actions.x.between(0, 105).all() or not actions.y.between(0, 68).all():
        raise ValueError("Spatial plots expect corrected 105x68 own-team coordinates")
    if not isinstance(columns, int) or not isinstance(rows, int) or min(columns, rows) < 1:
        raise ValueError("Spatial grid dimensions must be positive integers")
    parts = []
    grid = pd.MultiIndex.from_product([event_types, range(rows), range(columns)], names=["type_name", "row", "column"])
    for left, right in pairs:
        work = pd.DataFrame({"type_name": actions.type_name.to_numpy(),
                             "column": np.floor(actions.x.to_numpy() / 105 * columns).astype(int).clip(0, columns - 1),
                             "row": np.floor(actions.y.to_numpy() / 68 * rows).astype(int).clip(0, rows - 1),
                             "left_value": valuations[left].vaep_value.to_numpy(),
                             "right_value": valuations[right].vaep_value.to_numpy()})
        work["delta"] = work.left_value - work.right_value
        summary = work.groupby(["type_name", "row", "column"]).agg(
            actions=("delta", "size"), total_delta=("delta", "sum"), mean_delta=("delta", "mean"),
            median_delta=("delta", "median"), left_mean=("left_value", "mean"), right_mean=("right_value", "mean"),
        ).reindex(grid).reset_index()
        summary["actions"] = summary.actions.fillna(0).astype(int)
        summary["total_delta"] = summary.total_delta.fillna(0)
        summary["left"], summary["right"] = left, right
        summary["minimum_actions"] = np.where(summary.type_name.eq("shot"), 10, 20)
        parts.append(summary)
    return pd.concat(parts, ignore_index=True)


def player_changes(players, pairs, measure="per90"):
    parts = []
    for left, right in pairs:
        part = players[["player_id", "player_name", "position_bucket", "test_minutes", "action_count"]].copy()
        part["left"], part["right"], part["measure"] = left, right, measure
        part["left_value"] = players[f"{left}_{measure}"]
        part["right_value"] = players[f"{right}_{measure}"]
        part["value_delta"] = part.left_value - part.right_value
        part["left_position_rank"] = players[f"{left}_{measure}_position_rank"]
        part["right_position_rank"] = players[f"{right}_{measure}_position_rank"]
        part["rank_gain_for_left"] = part.right_position_rank - part.left_position_rank
        part["rank_delta"] = part.left_position_rank - part.right_position_rank
        part["absolute_rank_change"] = part.rank_gain_for_left.abs()
        part["position_players"] = players.groupby("position_bucket").player_id.transform("size")
        part["normalized_rank_change"] = part.absolute_rank_change / (part.position_players - 1).clip(lower=1)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)
