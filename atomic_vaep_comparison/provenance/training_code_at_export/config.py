from pathlib import Path

HOME = Path(__file__).resolve().parents[1]
ROOT = HOME.parent
TARGETS = ("scores", "concedes")
BUDGETS = (*range(50, 501, 50), 540)
SEEDS = (17, 29, 43, 71)
SOURCE_ROWS = {"A": 1_586_819, "B": 3_085_260, "C": 4_583_701}
ADAPTERS = ("intercept", "logistic", "beta", "context")
METHODS = ("continued", *ADAPTERS)
CONTEXT_GROUPS = ("pass", "receival", "interception", "dribble", "clearance", "tackle")
PENALTIES = (0., 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
PARAMETER_COUNTS = {"intercept": 1, "logistic": 2, "beta": 3, "context": 8}
THREADS = 6
MODEL_PARAMETERS = {
    "scores": {"learning_rate": .016, "max_leaf_nodes": 127, "min_samples_leaf": 35, "max_iter": 800},
    "concedes": {"learning_rate": .010, "max_leaf_nodes": 63, "min_samples_leaf": 50, "max_iter": 1200},
}


def parameters(target):
    return {
        **MODEL_PARAMETERS[target], "l2_regularization": .75, "random_state": 42,
        "early_stopping": "auto", "validation_fraction": .1,
        "n_iter_no_change": 10, "tol": 1e-7,
    }


def run_specs():
    for budget in BUDGETS:
        for seed in SEEDS:
            for source in SOURCE_ROWS:
                for method in METHODS:
                    yield source, method, budget, seed


def run_directory(source, method, budget, seed):
    return HOME / "runs" / source / method / f"b{budget:03d}_s{seed}"


def design():
    return {
        "version": 2, "source_rows": SOURCE_ROWS, "women_reference_games": 540,
        "execution": "manual notebook; train all models here without importing earlier results",
        "budgets": list(BUDGETS), "sample_seeds": list(SEEDS),
        "hgb_parameters": {target: parameters(target) for target in TARGETS},
        "adapters": list(ADAPTERS), "context_groups": list(CONTEXT_GROUPS),
        "context_penalties": list(PENALTIES), "threads": THREADS,
        "male_sampling": "sorted prefixes of RandomState(42).permutation; A subset B subset C",
        "women_sampling": "nested default_rng(seed) permutations of development games",
        "full_budget": "four runs use the same 540 games; model seed remains 42",
        "continuation": "keep source trees; append up to 800/1200 target trees",
        "validation": "HGB: native 10% stratified action split; context: 20% match split, then refit all",
    }
