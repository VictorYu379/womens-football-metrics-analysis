import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd


HOME = Path(__file__).resolve().parent
PROJECT = HOME.parent
SOURCE = PROJECT / "paper_experiments"
TARGETS = ("scores", "concedes")
WOMEN_BUDGET = 540
ADAPTATION_METHODS = ("intercept", "logistic", "beta", "context", "continued_match")


def checksum(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def verified_copy(source, destination, expected=None):
    source = Path(source)
    destination = Path(destination)
    digest = checksum(source)
    if expected is not None and digest != expected:
        raise ValueError(f"Source checksum mismatch: {source}")
    if destination.exists():
        if checksum(destination) != digest:
            raise ValueError(f"Export already contains a different file: {destination}")
        return digest
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    shutil.copy2(source, temporary)
    if checksum(temporary) != digest:
        raise ValueError(f"Copy verification failed: {destination}")
    temporary.replace(destination)
    return digest


def representative_seeds(metrics):
    keys = ["source", "method", "budget", "seed"]
    adapted = metrics.loc[metrics.method != "direct"]
    if adapted.duplicated([*keys, "target"]).any():
        raise ValueError("Duplicate seed-level test metrics")
    table = adapted.pivot(index=keys, columns="target", values="log_loss").reset_index()
    if not set(TARGETS).issubset(table.columns) or not np.isfinite(table[list(TARGETS)]).all().all():
        raise ValueError("Both targets require finite test losses for every seed")
    if (table[list(TARGETS)] <= 0).any().any():
        raise ValueError("Relative distances require positive mean losses")
    groups = table.groupby(["source", "method", "budget"])
    if not groups.seed.apply(lambda values: set(values) == {17, 29, 43, 71}).all():
        raise ValueError("Representative selection requires all four planned seeds")
    for target in TARGETS:
        table[f"{target}_mean"] = groups[target].transform("mean")
        table[f"{target}_relative_distance"] = (table[target] - table[f"{target}_mean"]).abs() / table[f"{target}_mean"]
    table["distance"] = table[[f"{target}_relative_distance" for target in TARGETS]].mean(axis=1)
    table["selected"] = False
    for _, group in table.groupby(["source", "method", "budget"]):
        nearest = group.loc[group.distance <= group.distance.min() + 1e-12].sort_values("seed").index[0]
        table.loc[nearest, "selected"] = True
    table["directory"] = [f"runs/{source}/{method}/b{budget:03d}_s{seed}"
                           for source, method, budget, seed in table[keys].itertuples(index=False, name=None)]
    return table


def comparison_metrics(metrics):
    bases = metrics.method.eq("direct") & metrics.source.isin(["A", "B", "C", "D"])
    adaptations = (metrics.source.isin(["A", "B", "C"]) & metrics.method.isin(ADAPTATION_METHODS)
                   & metrics.budget.eq(WOMEN_BUDGET))
    return metrics.loc[bases | adaptations].copy()


def comparison_seeds(metrics):
    candidates = comparison_metrics(metrics)
    return representative_seeds(candidates.loc[candidates.method != "direct"])


def export_models(selection):
    records = []
    selected_directories = set(selection.loc[selection.selected, "directory"])
    for branch in ("models", "runs"):
        for marker in sorted((SOURCE / branch).rglob("fit.json")):
            metadata = read_json(marker)
            if metadata.get("status") != "complete":
                raise ValueError(f"Incomplete model pair: {marker}")
            directory = marker.parent
            relative = directory.relative_to(SOURCE)
            if metadata["method"] != "direct" and relative.as_posix() not in selected_directories:
                continue
            files = {}
            for target in TARGETS:
                name = f"{target}.joblib"
                files[name] = verified_copy(directory / name, HOME / relative / name,
                                            metadata["files_sha256"][name])
            for path in sorted(directory.glob("*.json")):
                files[path.name] = verified_copy(path, HOME / relative / path.name)
            records.append({
                "directory": relative.as_posix(),
                **{key: metadata[key] for key in ("source", "method", "budget", "seed", "fingerprint")},
                "base_directory": f"models/{metadata['source']}" if metadata["method"] in
                                  {"intercept", "logistic", "beta", "context"} else None,
                "legacy_continuation": metadata["method"] == "continued",
                "files_sha256": files,
            })
    copied_models = {str(Path(record["directory"]) / f"{target}.joblib")
                     for record in records for target in TARGETS}
    original_models = {str(path.relative_to(SOURCE)) for path in (SOURCE / "models").rglob("*.joblib")}
    original_models.update(str(Path(relative) / f"{target}.joblib")
                           for relative in selected_directories for target in TARGETS)
    if copied_models != original_models:
        raise ValueError("Not every selected model belongs to a complete exported pair")
    lookup = {record["directory"]: record for record in records}
    for record in records:
        if record["base_directory"]:
            source_hashes = read_json(HOME / record["directory"] / "fit.json")["source_sha256"]
            for target in TARGETS:
                if source_hashes[target] != lookup[record["base_directory"]]["files_sha256"][f"{target}.joblib"]:
                    raise ValueError(f"Adapter/base mismatch: {record['directory']}")
    return records


def export_test_inputs():
    destination = HOME / "inputs"
    marker = destination / "manifest.json"
    if marker.exists():
        manifest = read_json(marker)
        for name, digest in manifest["files_sha256"].items():
            if checksum(destination / name) != digest:
                raise ValueError(f"Exported input changed: {name}")
        return manifest
    destination.mkdir(exist_ok=True)
    manifest = read_json(SOURCE / "data/manifest.json")
    for name in ("women_features.npy", "women_metadata.parquet", "test_indices.npy"):
        if checksum(SOURCE / "data" / name) != manifest["files_sha256"][name]:
            raise ValueError(f"Training input changed: {name}")
    original_actions = PROJECT / "vaep_data/women_atomic_vaep_rows_ltr_n5_h10_v2.parquet"
    if checksum(original_actions) != manifest["input_sha256"]["women"]:
        raise ValueError("Original atomic rows no longer match the training input")
    indices = np.load(SOURCE / "data/test_indices.npy")
    metadata = pd.read_parquet(SOURCE / "data/women_metadata.parquet")
    is_test = metadata.game_id.isin(manifest["test_games"]).to_numpy()
    if not np.array_equal(indices, np.flatnonzero(is_test)):
        raise ValueError("Test indices must include every action of each test match")
    columns = ["game_id", "original_event_id", "action_id", "period_id", "time_seconds", "team_id",
               "player_id", "x", "y", "dx", "dy", "type_id", "bodypart_id", "type_name",
               "bodypart_name", "scores", "concedes"]
    actions = pd.read_parquet(original_actions, columns=columns).iloc[indices].reset_index(drop=True)
    saved_metadata = metadata.iloc[indices].reset_index(drop=True)
    for column in saved_metadata:
        pd.testing.assert_series_equal(actions[column], saved_metadata[column], check_dtype=False)
    if actions.player_id.isna().any() or actions.duplicated(["game_id", "action_id"]).any():
        raise ValueError("Test actions need unique identifiers and player attribution")
    actions["player_id"] = actions.player_id.astype("int64")
    actions.to_parquet(destination / "test_actions.parquet", index=False)
    features = np.load(SOURCE / "data/women_features.npy", mmap_mode="r")
    subset = np.lib.format.open_memmap(destination / "test_features.npy", mode="w+", dtype="float32",
                                      shape=(len(indices), len(manifest["features"])))
    for start in range(0, len(indices), 8192):
        values = features[indices[start:start + 8192]]
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite test features")
        subset[start:start + len(values)] = values
    subset.flush()
    del subset, features, metadata, saved_metadata

    for name, value in (("string_", np.bytes_), ("float_", np.float64), ("complex_", np.complex128)):
        if not hasattr(np, name):
            setattr(np, name, value)
    sys.path.insert(0, str(PROJECT))
    sys.path.insert(0, str(PROJECT / "atomic_vaep"))
    from atomic_vaep_data import load_position_rows, build_primary_position_map, bucket_position
    from minutes.player_minutes import load_test_player_minutes
    position_rows = load_position_rows(PROJECT / "vaep_data", "women")
    position_rows["position_name"] = position_rows.position_name.str.strip()
    position_rows = position_rows.loc[~position_rows.position_name.str.casefold().isin(["", "unknown"])]
    positions = pd.Series(build_primary_position_map(position_rows), name="primary_position").rename_axis("player_id").reset_index()
    positions["position_bucket"] = positions.primary_position.map(bucket_position)
    players = load_test_player_minutes(manifest["test_games"]).merge(positions, on="player_id", how="left", validate="one_to_one")
    covered = set(players.loc[players.position_bucket.notna(), "player_id"])
    if not set(actions.player_id).issubset(covered):
        raise ValueError("Missing positions or minutes for test-action players")
    players.to_parquet(destination / "player_context.parquet", index=False)
    result = {
        "features": manifest["features"], "feature_dtype": "float32", "actions": len(actions),
        "test_games": manifest["test_games"], "development_games": manifest["pool_games"],
        "source_manifest_sha256": checksum(SOURCE / "data/manifest.json"),
        "source_actions_sha256": manifest["input_sha256"]["women"],
        "source_positions_sha256": checksum(PROJECT / "vaep_data/women_pos.parquet"),
        "source_minutes_sha256": checksum(PROJECT / "minutes/women_test_player_minutes.csv"),
        "source_minutes_metadata_sha256": checksum(PROJECT / "minutes/women_test_minutes_metadata.json"),
        "positions": "Most frequent recorded position across all cached women's events; descriptive, not a model feature",
        "files_sha256": {name: checksum(destination / name) for name in
                         ("test_actions.parquet", "test_features.npy", "player_context.parquet")},
    }
    write_json(marker, result)
    return result


def export_pass_heights(inputs):
    destination = HOME / "inputs/pass_height_events.parquet"
    if destination.name in inputs["files_sha256"]:
        if checksum(destination) != inputs["files_sha256"][destination.name]:
            raise ValueError("Exported pass-height metadata changed")
        return inputs
    records, source_hashes = [], {}
    for game_id in sorted(inputs["test_games"]):
        path = PROJECT / "open-data/data/events" / f"{game_id}.json"
        source_hashes[str(game_id)] = checksum(path)
        for event in read_json(path):
            event_type = event.get("type", {}).get("name")
            if event_type not in {"Pass", "Shot"}:
                continue
            height = None
            if event_type == "Pass":
                height = (event.get("pass", {}).get("height") or {}).get("name") or "Unknown"
                if height not in {"Ground Pass", "Low Pass", "High Pass", "Unknown"}:
                    raise ValueError(f"Unexpected pass height: {height}")
            records.append({"game_id": game_id, "original_event_id": event["id"],
                            "event_type": event_type, "pass_height": height})
    metadata = pd.DataFrame(records)
    if metadata.duplicated(["game_id", "original_event_id"]).any():
        raise ValueError("Duplicate original event IDs")
    from analysis import attach_pass_heights
    attach_pass_heights(pd.read_parquet(HOME / "inputs/test_actions.parquet"), metadata)
    metadata.to_parquet(destination, index=False)
    inputs["files_sha256"][destination.name] = checksum(destination)
    inputs["pass_height_source_sha256"] = source_hashes
    write_json(HOME / "inputs/manifest.json", inputs)
    return inputs


def main():
    metric_tables = {name: comparison_metrics(pd.read_csv(SOURCE / "results" / name, float_precision="round_trip"))
                     for name in ("metrics.csv", "continued_match_metrics.csv")}
    metrics = pd.concat(metric_tables.values(), ignore_index=True)
    selection = comparison_seeds(metrics)
    selected_directories = [f"models/{source}" for source in ("A", "B", "C", "D")]
    selected_directories.extend(selection.loc[selection.selected, "directory"])
    total_bytes = sum((SOURCE / relative / f"{target}.joblib").stat().st_size
                      for relative in selected_directories for target in TARGETS)
    if not (HOME / "export_manifest.json").exists() and shutil.disk_usage(HOME).free < total_bytes + 2_000_000_000:
        raise RuntimeError("Not enough disk space for models, test inputs and analysis outputs")
    print("Copying and verifying representative model pairs...", flush=True)
    records = export_models(selection)
    selection.to_csv(HOME / "seed_selection.csv", index=False)
    provenance = HOME / "provenance"
    for relative in ("protocol.json", "data/manifest.json", "results/continued_match_protocol.json"):
        verified_copy(SOURCE / relative, provenance / relative)
    for name, table in metric_tables.items():
        table.to_csv(provenance / "results" / name, index=False)
    for name in ("__init__.py", "adapters.py", "config.py"):
        verified_copy(SOURCE / "vaep_study" / name, provenance / "training_code_at_export" / name)
    print(f"Copied {len(records)} pairs. Exporting held-out test inputs...", flush=True)
    inputs = export_pass_heights(export_test_inputs())
    references = {}
    for record in records:
        if record["method"] != "direct" and not (record["source"] == "C" and record["budget"] == 540):
            continue
        relative = Path(record["directory"]) / "test_predictions.npy"
        original = read_json(SOURCE / record["directory"] / "fit.json")
        references[record["directory"]] = verified_copy(SOURCE / relative, HOME / "references" / relative,
                                                       original["files_sha256"]["test_predictions.npy"])
    packages = ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "pyarrow", "socceraction", "matplotlib", "mplsoccer")
    export = {"format_version": 3, "source": str(SOURCE), "model_pairs": len(records),
              "model_files": 2 * len(records), "model_bytes": total_bytes,
              "scope": {"base_models": ["A", "B", "C", "D"], "adaptation_budgets": [WOMEN_BUDGET],
                        "adaptation_methods": list(ADAPTATION_METHODS),
                        "main_comparison": ["D", "C + best adapter"],
                        "legacy_continuation": "Excluded from this bundle; retained only in the source experiments"},
              "seed_selection": "One shared seed per source/method/budget; minimize mean absolute relative distance to the four-seed scores/concedes means; ties within 1e-12 use smaller seed. Posthoc test selection, not independent validation.",
              "seed_selection_sha256": checksum(HOME / "seed_selection.csv"),
              "packages": {name: importlib.metadata.version(name) for name in packages},
              "input_manifest_sha256": checksum(HOME / "inputs/manifest.json"),
              "reference_sha256": references, "models": records}
    write_json(HOME / "export_manifest.json", export)
    pd.DataFrame([{key: value for key, value in record.items() if key != "files_sha256"} for record in records]).to_csv(HOME / "model_catalog.csv", index=False)
    print(f"Ready: {export['model_files']} model files; {inputs['actions']:,} test actions; source files unchanged.", flush=True)


if __name__ == "__main__":
    main()
