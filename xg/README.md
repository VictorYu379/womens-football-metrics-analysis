# xG Hyperparameter Tuning

This folder tunes XGBoost xG model hyperparameters for the xG workflow.

## Scope

- Uses cached shot rows in `vaep_data/*_xg_shots_competition_events_method.pkl`, rebuilding them from `statsbombpy.sb.competition_events` when missing.
- Uses the original xG notebook target: `goal`.
- Uses sklearn `GridSearchCV` with match-level CV splits by `game_id`.
- Tunes only XGBoost candidates; there is no logistic-regression model family here.

## Training Views

The tuning script evaluates the two full training views used to select hyperparameters:

- `men_full`: all men 2015/16 top-league shots.
- `women_train`: women non-test shots after the match-level women test split.

Downsampling is intentionally not part of hyperparameter search. It belongs in
the later notebook experiment where men-trained and women-trained models are
compared under equal sample sizes.

## Run

```powershell
.\.venv\Scripts\python.exe xg\train_xgboost_xg_models.py
```

Useful smoke test:

```powershell
.\.venv\Scripts\python.exe xg\train_xgboost_xg_models.py --quick
```

## Outputs

Default output directory: `xg/xgboost_tuning_results/`

- `xgboost_xg_tuning_results.json`: selected params, split summary, CV settings.
- `validation_candidate_metrics.csv`: raw `GridSearchCV.cv_results_` rows.
