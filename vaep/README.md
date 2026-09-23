# VAEP Hyperparameter Tuning

This folder moves the VAEP data preparation and probability-model tuning logic
out of `enriched_vaep_features_competition_events.ipynb`, following the same
pattern as the `xg/` and `xa/` folders.

## Scope

- Uses cached SPADL rows in `vaep_data/*_spadl_ltr.parquet`, rebuilding from StatsBomb open data when missing.
- Builds and caches prepared VAEP rows as `vaep_data/*_vaep_rows_ltr_n3_h10.parquet`.
- Handles both cache-hit and no-cache paths: missing prepared rows rebuild from SPADL; missing SPADL rebuilds from local or remote StatsBomb open data.
- Uses the notebook's VAEP setup: 3 previous actions and a 10-action label horizon.
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

- `vaep_training_results.json`: selected params, split summary, CV settings, saved model paths.
- `progress_log.csv`: durable start/end/failure events for the run and each grid search.
- `validation_candidate_metrics.csv`: raw `GridSearchCV.cv_results_` rows for every target/dataset.
- `women_final_test_vaep_predictions.csv`: final-test probabilities for `scores` and `concedes`.
- `*_vaep_model.joblib`: target-specific best estimators.
- `*_vaep_models.joblib`: bundled `{scores, concedes}` estimators for notebook loading.

## Notes

Downsampling remains a notebook experiment step, as in xG/xA. Hyperparameter
search uses the full men modeling set and the non-test women modeling set.
The VAEP HGB grid search defaults to `n_jobs=1` because Windows process-based
parallelism can duplicate multi-million-row feature matrices and exhaust system
resources. Use `--search-n-jobs` only when the machine has enough RAM.
