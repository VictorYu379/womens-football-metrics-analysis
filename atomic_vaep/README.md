# Atomic-VAEP Hyperparameter Tuning

This folder moves Atomic-VAEP data preparation and probability-model tuning out
of `enriched_atomic_vaep_features_competition_events.ipynb`, following the same
pattern as `xg/`, `xa/`, and `vaep/`.

## Scope

- Uses cached Atomic-SPADL rows in `vaep_data/*_atomic_spadl_ltr.parquet`, rebuilding from left-to-right SPADL cache when missing.
- If SPADL cache is also missing, rebuilds SPADL from local or remote StatsBomb open data, aligns attacking direction left-to-right, then converts to Atomic-SPADL.
- Builds and caches prepared rows as `vaep_data/*_atomic_vaep_rows_ltr_n5_h10.parquet`.
- Uses 5 previous atomic actions and a 10-atomic-action label horizon.
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
- `*_atomic_vaep_models.joblib`: bundled `{scores, concedes}` estimators for notebook loading.

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
