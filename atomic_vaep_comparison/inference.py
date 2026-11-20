from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit, logit
import sklearn
from threadpoolctl import threadpool_limits


TARGETS = ("scores", "concedes")
CONTEXT_GROUPS = ("pass", "receival", "interception", "dribble", "clearance", "tackle")


def checksum(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def apply_adapter(probability, parameters, action_types=None):
    probability = np.clip(np.asarray(probability, dtype=float), 1e-8, 1 - 1e-8)
    if probability.ndim != 1 or not np.isfinite(probability).all():
        raise ValueError("Finite, one-dimensional probabilities required")
    constant = np.ones(len(probability))
    mode = parameters["mode"]
    offset = np.zeros(len(probability))
    if mode == "intercept":
        matrix = constant[:, None]
        offset = logit(probability)
    elif mode == "beta":
        matrix = np.column_stack([constant, np.log(probability), -np.log1p(-probability)])
    elif mode in ("logistic", "context"):
        columns = [constant, logit(probability)]
        if mode == "context":
            action_types = np.asarray(action_types)
            if action_types.shape != probability.shape:
                raise ValueError("Action-type adapter requires aligned type names")
            columns.extend((action_types == group).astype(float) for group in CONTEXT_GROUPS)
        matrix = np.column_stack(columns)
    else:
        raise ValueError(f"Unknown adapter: {mode}")
    coefficients = np.asarray(parameters["coefficients"])
    if coefficients.shape != (matrix.shape[1],):
        raise ValueError("Adapter parameter count does not match its design")
    return expit(offset + matrix @ coefficients)


@dataclass
class ProbabilityModel:
    estimator: object
    adapter: dict | None = None

    def predict_proba(self, features, action_types=None):
        probability = self.estimator.predict_proba(features)[:, 1]
        if self.adapter is not None:
            probability = apply_adapter(probability, self.adapter, action_types)
        return np.column_stack([1 - probability, probability])


def load_pair(home, relative):
    home = Path(home)
    manifest = json.loads((home / "export_manifest.json").read_text(encoding="utf-8"))
    if sklearn.__version__ != manifest["packages"]["scikit-learn"]:
        raise RuntimeError("Use the exported scikit-learn version to load these HGB models")
    records = {record["directory"]: record for record in manifest["models"]}
    record = records[Path(relative).as_posix()]
    pair = {}
    for target in TARGETS:
        name = f"{target}.joblib"
        path = home / record["directory"] / name
        if checksum(path) != record["files_sha256"][name]:
            raise ValueError(f"Exported model changed: {path}")
        saved = joblib.load(path)
        if record["base_directory"] is None:
            estimator, adapter = saved, None
        else:
            base_record = records[record["base_directory"]]
            base = home / base_record["directory"] / name
            if checksum(base) != base_record["files_sha256"][name]:
                raise ValueError(f"Exported adapter base changed: {base}")
            estimator, adapter = joblib.load(base), saved
        if estimator.n_features_in_ != 256 or not np.array_equal(estimator.classes_, [0, 1]):
            raise ValueError("Unexpected model feature/class contract")
        pair[target] = ProbabilityModel(estimator, adapter)
    return pair


def predict_pair(pair, features, action_types, batch_size=8192, threads=6):
    action_types = np.asarray(action_types)
    if features.ndim != 2 or features.shape[1] != 256 or len(features) != len(action_types):
        raise ValueError("Expected 256 ordered features and aligned action types")
    probabilities = np.empty((len(features), len(TARGETS)))
    with threadpool_limits(limits=threads):
        for start in range(0, len(features), batch_size):
            block = np.asarray(features[start:start + batch_size], dtype=np.float64, order="F")
            if not np.isfinite(block).all():
                raise ValueError("Nonfinite model input")
            for column, target in enumerate(TARGETS):
                probabilities[start:start + len(block), column] = pair[target].predict_proba(
                    block, action_types[start:start + len(block)])[:, 1]
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Invalid model probabilities")
    return probabilities
