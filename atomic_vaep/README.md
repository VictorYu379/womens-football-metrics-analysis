# Atomic-VAEP Hyperparameter Tuning

This folder moves Atomic-VAEP data preparation and probability-model tuning out
of `enriched_atomic_vaep_features_competition_events.ipynb`, following the same
pattern as `xg/`, `xa/`, and `vaep/`.

## Scope

- Uses coordinate-corrected Atomic-SPADL rows in `vaep_data/*_atomic_spadl_ltr_v2.parquet`, rebuilding from left-to-right SPADL cache when missing.
- Restores a common coordinate frame before Atomic conversion, then normalizes each generated atom to its own team's attacking direction. If the source SPADL cache is also missing, rebuilds it from local or remote StatsBomb open data.
- Builds prepared rows as `vaep_data/*_atomic_vaep_rows_ltr_n5_h10_v2.parquet`, expressing historical positions and movement vectors in the current team's frame.
- Uses the current atomic action plus four previous actions as features, and the current plus following nine atomic actions as labels, within each match.
- Tunes `scores` and `concedes` separately.
- Reserves women final-test matches inside `atomic_vaep_data.py`; those games are excluded from hyperparameter search and model fitting.
- Uses match-level CV folds to avoid leakage.
- Starts with the current notebook model family: `HistGradientBoostingClassifier`.

## Run

First-round grid:

```powershell
.\.venv\Scripts\python.exe atomic_vaep\train_atomic_vaep_probability_models.py --overwrite --search-n-jobs 1
```

Second-round grid around the first-round winner:

```powershell
.\.venv\Scripts\python.exe atomic_vaep\train_atomic_vaep_probability_models.py --candidate-grid round2 --output-dir atomic_vaep\hist_gradient_boosting_tuning_results_round2 --overwrite --search-n-jobs 1
```

Second-round grid-search only, recommended on memory-constrained machines:

```powershell
.\.venv\Scripts\python.exe atomic_vaep\train_atomic_vaep_probability_models.py --candidate-grid round2 --output-dir atomic_vaep\hist_gradient_boosting_tuning_results_round2 --overwrite --search-n-jobs 1 --skip-final-fit
```

Useful smoke test:

```powershell
.\.venv\Scripts\python.exe atomic_vaep\train_atomic_vaep_probability_models.py --candidate-grid round2 --target scores --quick --output-dir atomic_vaep\_smoke_results --overwrite --search-n-jobs 1
```

Progress is printed directly to the terminal. The default `--grid-verbose 3`
prints candidate/fold progress, while `[progress]` lines mark each grid-search
start, completion, final-fit completion, or failure. `GridSearchCV` uses
`refit=False`; final estimators are fitted explicitly after CV so completed CV
results are saved even if a large final fit runs out of memory. Use
`--skip-final-fit` when you only need hyperparameter-search results.

## Outputs

Default output directory: `atomic_vaep/hist_gradient_boosting_tuning_results/`

- `atomic_vaep_training_results.json`: selected params, split summary, CV settings, saved model paths.
- `progress_log.csv`: durable start/end/failure events for the run and each grid search.
- `validation_candidate_metrics.csv`: raw `GridSearchCV.cv_results_` rows for every target/dataset.
- `women_final_test_atomic_vaep_predictions.csv`: final-test probabilities for `scores` and `concedes`.
- `*_atomic_vaep_model.joblib`: target-specific best estimators.
- `*_atomic_vaep_models.joblib`: bundled `{scores, concedes}` estimators; the comparison notebook fits fresh models rather than loading these bundles.

## Notes

Downsampling remains a notebook experiment step. Hyperparameter search uses the
full men modeling set and the non-test women modeling set.

Do not downsample before Atomic-VAEP features/labels are generated: features and
labels rely on complete match sequences. Prepared rows are generated first on
full game sequences; any later row sampling only selects already-computed rows.

The `round2` grid keeps the first-round winner and expands around it with 6
candidates: lower learning rates, more iterations, one larger leaf-node budget,
and stronger regularization. It is intentionally saved to a separate output directory in the
recommended command so first-round results remain available for comparison.

## Coordinate-Corrected Data v2

The old conversion fed individually left-to-right SPADL actions directly into
the Atomic converter. An inferred interception could inherit the passing team's
coordinates despite belonging to the opponent, producing incorrect locations
and spurious or missing connecting dribbles. v2 converts each complete match
in one common frame before normalizing each atomic action. Original pass lengths
are preserved; inferred action counts can change.

Feature windows then rotate historical opponent positions and negate both
movement-vector components to match the current team's frame. Raw atomic metadata
stays individually left-to-right for valuation and spatial analysis. Labels are
regenerated on the corrected complete sequences; explicit own-goal handling and
the 5-action feature / 10-action label windows are unchanged.

Both atomic caches are versioned, so unversioned files are never silently reused.
Cache construction streams complete matches and publishes each file only after
successful completion. The source SPADL, position and shared minutes caches are
unchanged. To build corrected caches without loading all feature rows or training:

```powershell
.\.venv\Scripts\python.exe -B -c "from atomic_vaep.atomic_vaep_data import DEFAULT_DATA_DIR, DEFAULT_OPEN_DATA_DIR, ensure_atomic_vaep_rows_cache; [ensure_atomic_vaep_rows_cache(DEFAULT_DATA_DIR, DEFAULT_OPEN_DATA_DIR, tag) for tag in ('women', 'men2015')]"
```

Old fitted models, predictions and reported metrics do not represent v2. Existing
round2 search artifacts remain historical results; their hyperparameters may be
reused for fresh fits but have not been retuned on corrected data. Future training
manifests record `data_version` explicitly.

## Notebook Analysis

The comparison notebook uses the corrected v2 caches and the round2 parameters:
downsample men's prepared atomic rows to the women's training size with seed 42,
then fit scores/concedes classifiers for both groups using the respective best
men-full round2 parameters. Window lengths and feature names/order must match the
tuning manifest, but identical feature names do not imply identical coordinate
semantics. The notebook reports preparation/search versions and fits fresh models;
it does not retune hyperparameters. Restart the kernel and run all cells after the
coordinate correction; saved notebook outputs are cleared.

To limit notebook RAM use, Section 3 retains compact cached feature dtypes and
releases the full prepared-row tables/bundle before training. HGB uses a single
temporary float64 matrix per training group, with columnwise float32 rounding to
preserve the previous inputs; the constant imputer does not copy that matrix.
The matrix is shared by the scores/concedes fits and released before the next
group. Predictions use batches of at most 50,000 rows. Samples, row order,
random seeds, tuned model parameters and early stopping are unchanged.
After a MemoryError, restart the kernel before Run All to release arrays held by
the failed traceback. Re-running Section 3 alone requires reloading Section 2.

Probability diagnostics compare each model to binary labels on its own training
data and both models on the same women's test data. Test predictions are reused
to compute offensive/defensive Atomic-VAEP with the atomic formula on complete,
ordered game sequences, before any player exclusion. Outcome atoms, goalkeepers,
and zero/negative values remain during valuation.

Total and per-90 player analyses share one cohort: at least 100 scored atomic test
actions and 45 cumulative test minutes, excluding primary-position goalkeepers.
These are atomic-action counts, not ordinary SPADL-action counts. The shared
`minutes/women_test_*` lookup must match the exact 231 test match IDs (30%, seed 42).
Per 90 uses each signed total divided by all test-appearance minutes times 90,
including appearances with no atomic actions. It does not average per-match rates.
After changing player thresholds, rerun Section 1 and Section 6 onward in the
existing kernel to refresh total/per-90, positional and Top 15 outputs. Existing
scored actions are reused without refitting models or rebuilding caches.

The notebook includes overall/positional Spearman, Kendall tau-b and normalized
absolute rank differences, plus cohort-size-dependent rank-tolerance and Top-K
diagnostics. Rank-movement heatmaps use individual integer columns from -10 to +10,
annotate omitted outliers, and show position-bucket sizes. Top 15 comparisons
report overlap, model-only selections, full-bucket rank differences, and changes
after normalization; buckets with fewer than 15 players are excluded and boundary
ties are retained and flagged. Position-distribution colors retain the VAEP convention.

Atomic action-type, original pass-height and existing spatial-grid analyses precede
player filtering. Atomic `freekick` combines passing and shooting variants, so the
height analysis explicitly matches original StatsBomb Pass events and excludes
direct free-kick shots and separate outcome atoms. Six spatial maps show shot, pass,
receival, dribble, cross and take_on on the existing 12-by-16 grid. Shot cells require
at least 10 shots, matching xG; the other event types retain the 20-action display
threshold. Take-on maps include successful and unsuccessful attempts and use the
action start location. Thresholds affect display only, not player values or summary
tables. Rerun Section 5.2 to refresh saved plots without refitting models. These are
action-level views, not per-90 maps.

Run notebook regression tests without full model training:

```powershell
.\.venv\Scripts\python.exe -B -m unittest atomic_vaep.test_atomic_vaep_data atomic_vaep.test_atomic_vaep_notebook minutes.test_player_minutes -v
```
