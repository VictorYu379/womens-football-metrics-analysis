# VAEP Hyperparameter Tuning

This folder moves the VAEP data preparation and probability-model tuning logic
out of `enriched_vaep_features_competition_events.ipynb`, following the same
pattern as the `xg/` and `xa/` folders.

## Scope

- Uses cached SPADL rows in `vaep_data/*_spadl_ltr.parquet`, rebuilding from StatsBomb open data when missing.
- Builds and caches corrected prepared VAEP rows as `vaep_data/*_vaep_rows_ltr_n3_h10_v2.parquet`.
- Handles both cache-hit and no-cache paths: missing prepared rows rebuild from SPADL; missing SPADL rebuilds from local or remote StatsBomb open data.
- Uses a 3-action feature window (the current action plus two previous actions), all in the current team's attacking frame. Opponent historical actions are reflected before computing spatial features; original SPADL metadata is unchanged.
- Labels cover the current action plus the following nine actions, stopping at each match boundary. Own goals, including `bad_touch / owngoal`, count as conceding for the acting team and scoring for its opponent.
- Tunes the two VAEP probability targets separately: `scores` and `concedes`.
- Uses match-level women final-test splits and match-level CV folds to avoid leakage.
- Reserves the women final test matches inside `vaep_data.py`; those games are excluded from hyperparameter search and model fitting.
- Enforces left-to-right attacking-direction alignment through the `*_spadl_ltr.parquet` cache path and `spadl.play_left_to_right` on rebuild.
- Starts with the current notebook model family: `HistGradientBoostingClassifier`.

## Model Families

Model families are replaceable through the registry in `train_vaep_probability_models.py`.
Each family module should expose:

- `MODEL_FAMILY`: string key used by `--model-family`.
- `get_model_spec(quick: bool) -> ModelSpec`: sklearn estimator plus explicit candidate grids.

Current family:

- `hist_gradient_boosting`: sklearn `HistGradientBoostingClassifier`, matching the VAEP notebook baseline.

## Run

```powershell
.\.venv\Scripts\python.exe vaep\train_vaep_probability_models.py
```

Useful smoke test with the smaller grid:

```powershell
.\.venv\Scripts\python.exe vaep\train_vaep_probability_models.py --quick
```

Tune only one VAEP target:

```powershell
.\.venv\Scripts\python.exe vaep\train_vaep_probability_models.py --target scores --quick
```

Progress visibility is printed directly to the terminal. The default
`--grid-verbose 3` prints candidate/fold progress, while `[progress]` lines mark
each grid-search start, completion, or failure.

## Outputs

Default output directory: `vaep/hist_gradient_boosting_tuning_results/`

- `vaep_training_results.json`: selected params, data version, split summary, CV settings, saved model paths. Existing results without a data version are legacy version 1.
- `progress_log.csv`: durable start/end/failure events for the run and each grid search.
- `validation_candidate_metrics.csv`: raw `GridSearchCV.cv_results_` rows for every target/dataset.
- `women_final_test_vaep_predictions.csv`: final-test probabilities for `scores` and `concedes`.
- `*_vaep_model.joblib`: target-specific best estimators.
- `*_vaep_models.joblib`: bundled `{scores, concedes}` estimators for reuse; the comparison notebook fits its own models rather than loading these bundles.

## Notes

Data version 2 corrects own-goal labels and cross-team coordinate alignment in
feature windows. Unversioned prepared-row caches are ignored and can be deleted
after the matching version-2 caches have been validated. Keep the reusable SPADL
caches and the separate Atomic-VAEP caches.
Missing version-2 caches rebuild one match at a time from the existing SPADL
cache, using a temporary parquet file published only after all matches finish.
This avoids concatenating every match's feature matrix during cache generation.
The SPADL caches, feature names, match split and shared minutes lookup are unchanged.

Restart the notebook kernel and run all cells to fit fresh models and recompute
results. Existing saved hyperparameters were selected on legacy data: the notebook
keeps them fixed for comparison, reports the version mismatch, and does not load
old joblib estimators. These are not parameters retuned on version 2. A new tuning
run records its data version; old tuning artifacts are not relabeled or overwritten
by notebook execution.

Downsampling remains a notebook experiment step, as in xG/xA. Hyperparameter
search uses the full men modeling set and the non-test women modeling set.
The VAEP HGB grid search defaults to `n_jobs=1` because Windows process-based
parallelism can duplicate multi-million-row feature matrices and exhaust system
resources. Use `--search-n-jobs` only when the machine has enough RAM.

## Notebook Analysis

The notebook keeps the balanced-row experiment: downsample prepared men's rows to
the women's training size, then fit both groups with the corresponding best
`men_full` parameters. It predicts probabilities once per dataset/model pair and
computes signed VAEP on complete test-game sequences before any player filtering.

Total and per-90 comparisons use the same outfield players with at least 50 scored
test actions and 45 cumulative test minutes. All scored SPADL actions count, including
failures and zero/negative contributions. Denominators come from the shared
`minutes/` lookup, validated against the exact test match IDs. The notebook includes
overall and within-position ranking metrics, annotated rank-movement heatmaps, and
positional Top 15 comparisons. The action-type analysis remains separate from the
player eligibility filters.

Section 5.1.1 also splits passing actions by their original StatsBomb pass height,
joining local test-match event JSON by match/event ID. It retains failures and
original SPADL passing types for matching. Sections 5.1 and 5.1.1 pool
`freekick_short` and `freekick_crossed` as `freekick`, aggregating the underlying
actions before computing statistics or applying the 500-action threshold.
`shot_freekick` remains separate. This grouping and height are analysis-only
metadata, not model features; predictions, action values and player rankings do not change.
All observed type-height groups are tabulated, while distribution plots require
at least 500 actions per group, matching the action-type plot threshold.

Run data-preparation and notebook regression tests without fitting full-size models:

```powershell
.\.venv\Scripts\python.exe -B -m unittest vaep.test_vaep_data vaep.test_vaep_notebook -v
```
