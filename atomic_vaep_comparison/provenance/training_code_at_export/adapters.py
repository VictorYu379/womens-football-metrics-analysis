"""Probability-only calibration and a regularized action-context update."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from .config import CONTEXT_GROUPS, PARAMETER_COUNTS


def design(probability, mode, action_types=None):
    probability = np.clip(np.asarray(probability, dtype=float), 1e-8, 1 - 1e-8)
    if probability.ndim != 1 or not np.isfinite(probability).all():
        raise ValueError("Finite one-dimensional probabilities required")
    constant = np.ones(len(probability))
    if mode == "intercept":
        return constant[:, None], logit(probability), [0.], [(-20., 20.)]
    if mode == "beta":
        matrix = np.column_stack([constant, np.log(probability), -np.log1p(-probability)])
        return matrix, np.zeros(len(probability)), [0., 1., 1.], [(-20., 20.), (0., 20.), (0., 20.)]
    if mode not in {"logistic", "context"}:
        raise ValueError(f"Unknown calibration: {mode}")
    columns = [constant, logit(probability)]
    initial = [0., 1.]
    bounds = [(-20., 20.), (.0001, 20.)]
    if mode == "context":
        action_types = np.asarray(action_types)
        if action_types.shape != probability.shape:
            raise ValueError("Aligned action types required")
        columns.extend((action_types == group).astype(float) for group in CONTEXT_GROUPS)
        initial.extend([0.] * len(CONTEXT_GROUPS))
        bounds.extend([(-20., 20.)] * len(CONTEXT_GROUPS))
    return np.column_stack(columns), np.zeros(len(probability)), initial, bounds


def fit_calibration(probability, labels, mode, action_types=None, penalty=0.):
    matrix, offset, initial, bounds = design(probability, mode, action_types)
    labels = np.asarray(labels, dtype=float)
    if labels.shape != offset.shape or not np.isin(labels, [0, 1]).all() or labels.min() == labels.max():
        raise ValueError("Aligned binary labels with both classes required")
    penalties = np.zeros(matrix.shape[1])
    if mode == "context":
        penalties[2:] = penalty

    def objective(coefficients):
        linear = offset + matrix @ coefficients
        residual = expit(linear) - labels
        loss = np.mean(np.logaddexp(0., linear) - labels * linear) + .5 * np.sum(penalties * coefficients ** 2)
        gradient = matrix.T @ residual / len(labels) + penalties * coefficients
        return loss, gradient

    fitted = minimize(objective, initial, method="L-BFGS-B", jac=True, bounds=bounds, options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 2000, "maxls": 50})
    if not fitted.success:
        raise RuntimeError(f"Calibration did not converge: {fitted.message}")
    binding = [index for index, (value, limits) in enumerate(zip(fitted.x, bounds)) if min(abs(value - limits[0]), abs(value - limits[1])) < 1e-5]
    return {"mode": mode, "coefficients": fitted.x.tolist(), "parameter_count": PARAMETER_COUNTS[mode], "penalty": float(penalty), "success": bool(fitted.success), "objective": float(fitted.fun), "binding_parameters": binding, "rows": len(labels), "positive_labels": int(labels.sum())}


def apply_calibration(probability, parameters, action_types=None):
    matrix, offset, _, _ = design(probability, parameters["mode"], action_types)
    coefficients = np.asarray(parameters["coefficients"])
    if matrix.shape[1] != len(coefficients):
        raise ValueError("Calibration coefficient count mismatch")
    return expit(offset + matrix @ coefficients)
