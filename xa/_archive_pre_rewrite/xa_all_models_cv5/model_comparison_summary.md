# xA Model Comparison Summary

This report compares XGBoost and Logistic xA models trained with the original xA target: `P(goal assist | completed pass)`.

## Run Setup

- Output directory: `xa/xa_all_models_cv5`
- Tuning mode: `cv`
- CV folds: `5`
- Selection metric: `xa_accuracy`
- Final training policy: `cv_refit_all_non_test_rows`
- Women final test share: `0.1`
- Seed: `42`

## Split Summary

| split | rows | games | positives | prevalence |
| --- | --- | --- | --- | --- |
| women_cv_modeling_non_test | 489432.000000 | 694.000000 | 1310.000000 | 0.002677 |
| women_final_test | 54832.000000 | 77.000000 | 150.000000 | 0.002736 |
| men_cv_modeling | 1136687.000000 | 1517.000000 | 2596.000000 | 0.002284 |

## Selected CV Models

| model_family | dataset | candidate | metrics.xa_accuracy | metrics.average_precision | metrics.roc_auc | metrics.brier | metrics.predicted_xa_sum | metrics.observed_assists | best_iteration |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xgboost | men | xgb_logloss_depth4_slow | 0.126517 | 0.198987 | 0.976742 | 0.001990 | 2553.604483 | 2596 | 365.000000 |
| xgboost | women | xgb_logloss_depth4_mid | 0.139905 | 0.222674 | 0.979428 | 0.002296 | 1308.231119 | 1310 | 298.000000 |
| logistic | men | logistic_l2_c0_05 | 0.125732 | 0.208434 | 0.975757 | 0.001992 | 2592.023112 | 2596 |  |
| logistic | women | logistic_l2_c0_05 | 0.141516 | 0.233971 | 0.979527 | 0.002292 | 1317.675462 | 1310 |  |

## Women Final Test Metrics

Women final test set contains completed passes only. `observed_assists` is the true number of goal assists in the test set.

| family | train_gender | xa_accuracy | average_precision | roc_auc | brier | log_loss | predicted_xa_sum | observed_assists | xA_error_vs_observed | f1_at_threshold | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xgboost | men | 0.141407 | 0.238673 | 0.981498 | 0.002342 | 0.009998 | 157.775980 | 150.000000 | 7.775980 | 0.319249 | 0.147503 |
| xgboost | women | 0.147996 | 0.231013 | 0.978459 | 0.002324 | 0.010037 | 153.853102 | 150.000000 | 3.853102 | 0.338542 | 0.148008 |
| logistic | men | 0.134973 | 0.226767 | 0.977664 | 0.002360 | 0.010181 | 156.292643 | 150.000000 | 6.292643 | 0.319783 | 0.139044 |
| logistic | women | 0.141199 | 0.229282 | 0.977841 | 0.002343 | 0.010085 | 155.615439 | 150.000000 | 5.615439 | 0.328125 | 0.131287 |

## Highlights

- Best final-test xA accuracy: `xgboost` trained on `women` (`0.147996`).
- Best final-test average precision: `xgboost` trained on `men` (`0.238673`).
- Best final-test ROC-AUC: `xgboost` trained on `men` (`0.981498`).
- Closest aggregate xA calibration: `xgboost` trained on `women` (`xA error = 3.853102`).

## Interpretation

- Logistic is the closer original-definition / interpretable baseline.
- XGBoost is the performance-oriented model family.
- On this final test set, the women-trained XGBoost model has the best xA accuracy and aggregate calibration.
- Logistic remains competitive in CV, especially on the women model, but trails XGBoost slightly on final-test xA accuracy.

## Source Files

- Full JSON results: `xa/xa_all_models_cv5/xa_original_training_results.json`
- Candidate CV metrics: `xa/xa_all_models_cv5/validation_candidate_metrics.csv`
- Per-pass final-test predictions: `xa/xa_all_models_cv5/women_final_test_xa_predictions.csv`
