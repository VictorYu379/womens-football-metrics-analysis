# xA Modeling

This folder contains a compact original-definition expected assist (xA) training pipeline.

## Definition

```text
xA = P(goal assist | completed pass)
target_goal_assist = is_goal_assist.astype(int)
```

The data layer filters to completed passes before modeling, so incomplete passes are not part of the xA target.

## Structure

- `xa_data.py`
  - Loads cached pass data, rebuilding it from `statsbombpy.sb.competition_events` when missing, builds the original xA rows, isolates the women final test set by match, and creates match-level CV splits.
- `model_contract.py`
  - Small sklearn-oriented contract: `ModelSpec`, `make_feature_frame(...)`, and `make_preprocessor(...)`.
- `logistic_model.py`
  - Logistic regression model family and parameter grid.
- `xgboost_model.py`
  - XGBoost model family and parameter grid.
- `train_original_xa_models.py`
  - Main runner. Uses sklearn `GridSearchCV` for tuning, model selection, and final refit.
- `test_*.py`
  - Standard-library `unittest` tests for data behavior, model specs, and training helpers.

## Splitting Rules

- Women final test set is isolated first by match (`game_id`) and is never used for tuning or training.
- CV uses fixed `CV_FOLDS = 4` match-level folds from `xa_data.py`.
- Holdout tuning also splits by match.
- Each pass label is `target_goal_assist`, exactly derived from `is_goal_assist`.

## Model Tuning

The runner intentionally delegates tuning to sklearn:

```text
GridSearchCV(estimator, param_grid, scoring=..., refit=selection_metric, cv=match_level_splits)
```

This keeps the training loop understandable:

1. Prepare men and women non-test data.
2. Build each model family's sklearn pipeline and parameter grid.
3. Run `GridSearchCV` with match-level splits.
4. Let sklearn refit the best model on all available non-test rows.
5. Evaluate men-trained and women-trained models on the isolated women final test set.

Supported model families:

```text
--model-family logistic
--model-family xgboost
--model-family all
```

Current full candidate grids:

```text
logistic: 8 C values
xgboost: 12 hand-picked parameter candidates
quick mode: 2 candidates per family
```

Parallelization defaults are set inside each model family:

```text
logistic: GridSearchCV search_n_jobs=4
xgboost: GridSearchCV search_n_jobs=2, XGBClassifier n_jobs=4
```

Default selection metric is `neg_log_loss`, which is a standard probability-model tuning objective. Final reports still include `xA_accuracy`, AP, ROC-AUC, Brier, log loss, and aggregate xA calibration.

Supported selection metrics:

```text
xa_accuracy
average_precision
roc_auc
neg_log_loss     (default)
neg_brier
```

All five metrics are always computed and written to `validation_candidate_metrics.csv`; only the
selection metric decides which candidate is refit and saved. `neg_log_loss` is the default because
at this prevalence (~0.27% positives) squared-error metrics saturate on the positives: giving a real
assist 1% vs 5% moves Brier by 8% but log loss by 35%, so `xa_accuracy` cannot resolve candidates
that differ only slightly. Measured on a held-out fold, `xa_accuracy` selected 300 trees where the
true validation-logloss optimum was ~450.

## Commands

Train and compare all current model families with 4-fold match-level CV:

```powershell
.\.venv\Scripts\python.exe -u .\xa\train_original_xa_models.py --model-family all --output-dir .\xa\xa_all_models_cv4
```

Run a faster smoke test with smaller candidate grids:

```powershell
.\.venv\Scripts\python.exe -u .\xa\train_original_xa_models.py --model-family all --quick --tuning-mode holdout --output-dir $env:TEMP\xa_smoke
```

The script refuses to overwrite existing xA artifacts unless `--overwrite` is passed. Prefer a fresh `--output-dir` for formal runs.

## Outputs

A run writes raw artifacts only. After training finishes, we can generate a separate comparison markdown from these artifacts.

A run writes:

- `xa_original_training_results.json`
  - Split summary, selected candidates, final test metrics, saved model paths, and git metadata.
- `validation_candidate_metrics.csv`
  - Raw `GridSearchCV.cv_results_` rows for each model family and dataset.
- `women_final_test_xa_predictions.csv`
  - Per-pass women final test predictions from each trained model.
- `men_<model_family>_xa_model.joblib`
- `women_<model_family>_xa_model.joblib`

## Adding A Model Family

Create a file with a `get_model_spec(quick: bool) -> ModelSpec` function, then register it in `MODEL_REGISTRY` inside `train_original_xa_models.py`.

A model family only needs to provide:

```python
MODEL_FAMILY = "my_model"

def get_model_spec(quick: bool = False) -> ModelSpec:
    return ModelSpec(
        name=MODEL_FAMILY,
        estimator=sklearn_estimator_or_pipeline,
        param_grid=[...],
        search_n_jobs=1,
    )
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s xa -p "test*.py" -v
```

