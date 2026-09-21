# xA Model Comparison Summary

This report is generated after training from raw artifacts only. The training script itself does not generate this markdown file.

## Run Setup
| Item | Value |
| --- | --- |
| Output directory | `xa/xa_all_models_cv4` |
| Model families | logistic, xgboost |
| Tuning mode | cv |
| CV folds | 4 |
| Selection metric | neg_log_loss |
| Quick mode | False |
| Seed | 42 |
| Women final test share | 0.1 |
| Target | `target_goal_assist` |
| Definition | xA = P(completed pass becomes a goal assist) |

## Data Splits
| Split | Rows | Games | Positives | Prevalence |
| --- | --- | --- | --- | --- |
| men_cv_modeling | 1136687 | 1517 | 2596 | 0.002284 |
| women_cv_modeling_non_test | 489432 | 694 | 1310 | 0.002677 |
| women_final_test | 54832 | 77 | 150 | 0.002736 |

## Selected Models
| Family | Dataset | Best validation neg_log_loss | Best params |
| --- | --- | --- | --- |
| logistic | men | -0.008707 | C=0.3 |
| logistic | women | -0.009801 | C=0.1 |
| xgboost | men | -0.008607 | colsample_bytree=0.88, learning_rate=0.035, max_depth=4, min_child_weight=10, n_estimators=350, reg_alpha=0.2, reg_lambda=8.0, subsample=0.9 |
| xgboost | women | -0.009775 | colsample_bytree=0.9, learning_rate=0.015, max_depth=4, min_child_weight=15, n_estimators=700, reg_alpha=0.4, reg_lambda=12.0, subsample=0.92 |

## Validation Best-Per-Family
| Dataset | Family | Rank | Validation neg_log_loss | Validation log_loss | xA accuracy | AP | ROC-AUC | Brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| men | logistic | 1 | -0.008707 | 0.008707 | 0.126180 | 0.212838 | 0.978388 | 0.001991 |
| women | logistic | 1 | -0.009801 | 0.009801 | 0.142614 | 0.240017 | 0.979333 | 0.002289 |
| men | xgboost | 1 | -0.008607 | 0.008607 | 0.133295 | 0.214506 | 0.979287 | 0.001975 |
| women | xgboost | 1 | -0.009775 | 0.009775 | 0.143483 | 0.231848 | 0.979395 | 0.002286 |

## Women Final Test
| Model | Rows | Positives | Predicted xA sum | Observed assists | xA sum error | xA accuracy | AP | ROC-AUC | Brier | Log loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_men_model_on_women_final_test | 54832 | 150 | 156.037 | 150 | 6.037 | 0.132718 | 0.224390 | 0.978076 | 0.002366 | 0.010191 |
| logistic_women_model_on_women_final_test | 54832 | 150 | 155.070 | 150 | 5.070 | 0.139373 | 0.227364 | 0.977340 | 0.002348 | 0.010108 |
| xgboost_men_model_on_women_final_test | 54832 | 150 | 159.354 | 150 | 9.354 | 0.139131 | 0.234528 | 0.980792 | 0.002349 | 0.010019 |
| xgboost_women_model_on_women_final_test | 54832 | 150 | 154.445 | 150 | 4.445 | 0.146422 | 0.227727 | 0.978482 | 0.002329 | 0.010047 |

## Final-Test Winners
| Metric | Winner | Value |
| --- | --- | --- |
| xA accuracy | xgboost_women_model_on_women_final_test | 0.146422 |
| Average precision | xgboost_men_model_on_women_final_test | 0.234528 |
| ROC-AUC | xgboost_men_model_on_women_final_test | 0.980792 |
| Brier | xgboost_women_model_on_women_final_test | 0.002329 |
| Log loss | xgboost_men_model_on_women_final_test | 0.010019 |
| Absolute xA sum error | xgboost_women_model_on_women_final_test | 4.445104 |

## Candidate Search Details
| Dataset | Family | Rank by neg_log_loss | Mean neg_log_loss | Mean xA accuracy | Mean AP | Mean ROC-AUC | Params |
| --- | --- | --- | --- | --- | --- | --- | --- |
| men | logistic | 1 | -0.008707 | 0.126180 | 0.212838 | 0.978388 | {'C': 0.3} |
| men | logistic | 2 | -0.008707 | 0.125878 | 0.212035 | 0.978407 | {'C': 1.0} |
| men | logistic | 3 | -0.008708 | 0.126770 | 0.213751 | 0.978150 | {'C': 0.1} |
| men | logistic | 4 | -0.008719 | 0.125061 | 0.210634 | 0.978304 | {'C': 3.0} |
| men | logistic | 5 | -0.008719 | 0.125106 | 0.210620 | 0.978265 | {'C': 10.0} |
| men | logistic | 6 | -0.008739 | 0.127212 | 0.214145 | 0.977313 | {'C': 0.03} |
| men | logistic | 7 | -0.008807 | 0.126558 | 0.212451 | 0.976188 | {'C': 0.01} |
| men | logistic | 8 | -0.008884 | 0.124165 | 0.208912 | 0.975369 | {'C': 0.005} |
| men | xgboost | 1 | -0.008607 | 0.133295 | 0.214506 | 0.979287 | {'colsample_bytree': 0.88, 'learning_rate': 0.035, 'max_depth': 4, 'min_child_weight': 10, 'n_estimators': 350, 'reg_alpha': 0.2, 'reg_lambda': 8.0, 'subsample': 0.9} |
| men | xgboost | 2 | -0.008622 | 0.135058 | 0.217956 | 0.978481 | {'colsample_bytree': 0.9, 'learning_rate': 0.015, 'max_depth': 4, 'min_child_weight': 15, 'n_estimators': 700, 'reg_alpha': 0.4, 'reg_lambda': 12.0, 'subsample': 0.92} |
| men | xgboost | 3 | -0.008625 | 0.135428 | 0.219240 | 0.978312 | {'colsample_bytree': 0.86, 'learning_rate': 0.018, 'max_depth': 4, 'min_child_weight': 30, 'n_estimators': 650, 'reg_alpha': 1.2, 'reg_lambda': 24.0, 'subsample': 0.88} |
| men | xgboost | 4 | -0.008627 | 0.135319 | 0.220500 | 0.978398 | {'colsample_bytree': 0.84, 'learning_rate': 0.02, 'max_depth': 5, 'min_child_weight': 20, 'n_estimators': 500, 'reg_alpha': 0.6, 'reg_lambda': 16.0, 'subsample': 0.86} |
| men | xgboost | 5 | -0.008633 | 0.133688 | 0.215181 | 0.978439 | {'colsample_bytree': 0.88, 'learning_rate': 0.025, 'max_depth': 4, 'min_child_weight': 12, 'n_estimators': 500, 'reg_alpha': 0.3, 'reg_lambda': 10.0, 'subsample': 0.9} |
| men | xgboost | 6 | -0.008634 | 0.133980 | 0.213961 | 0.978164 | {'colsample_bytree': 0.95, 'learning_rate': 0.03, 'max_depth': 3, 'min_child_weight': 40, 'n_estimators': 450, 'reg_alpha': 1.0, 'reg_lambda': 20.0, 'subsample': 0.95} |
| men | xgboost | 7 | -0.008634 | 0.134228 | 0.214816 | 0.978211 | {'colsample_bytree': 0.92, 'learning_rate': 0.02, 'max_depth': 3, 'min_child_weight': 25, 'n_estimators': 600, 'reg_alpha': 0.5, 'reg_lambda': 14.0, 'subsample': 0.92} |
| men | xgboost | 8 | -0.008634 | 0.134037 | 0.214515 | 0.978380 | {'colsample_bytree': 0.92, 'learning_rate': 0.03, 'max_depth': 3, 'min_child_weight': 20, 'n_estimators': 450, 'reg_alpha': 0.3, 'reg_lambda': 12.0, 'subsample': 0.92} |
| men | xgboost | 9 | -0.008640 | 0.135336 | 0.220807 | 0.978131 | {'colsample_bytree': 0.82, 'learning_rate': 0.025, 'max_depth': 6, 'min_child_weight': 25, 'n_estimators': 300, 'reg_alpha': 0.8, 'reg_lambda': 20.0, 'subsample': 0.84} |
| men | xgboost | 10 | -0.008640 | 0.134017 | 0.214969 | 0.978229 | {'colsample_bytree': 0.9, 'learning_rate': 0.035, 'max_depth': 3, 'min_child_weight': 15, 'n_estimators': 300, 'reg_alpha': 0.15, 'reg_lambda': 8.0, 'subsample': 0.9} |
| men | xgboost | 11 | -0.008645 | 0.133765 | 0.216573 | 0.978295 | {'colsample_bytree': 0.84, 'learning_rate': 0.03, 'max_depth': 5, 'min_child_weight': 18, 'n_estimators': 350, 'reg_alpha': 0.4, 'reg_lambda': 12.0, 'subsample': 0.86} |
| men | xgboost | 12 | -0.008655 | 0.133219 | 0.215037 | 0.978153 | {'colsample_bytree': 0.9, 'learning_rate': 0.05, 'max_depth': 3, 'min_child_weight': 12, 'n_estimators': 250, 'reg_alpha': 0.1, 'reg_lambda': 6.0, 'subsample': 0.9} |
| women | logistic | 1 | -0.009801 | 0.142614 | 0.240017 | 0.979333 | {'C': 0.1} |
| women | logistic | 2 | -0.009817 | 0.141137 | 0.238427 | 0.979123 | {'C': 0.3} |
| women | logistic | 3 | -0.009827 | 0.143442 | 0.240867 | 0.979110 | {'C': 0.03} |
| women | logistic | 4 | -0.009860 | 0.139280 | 0.235956 | 0.978713 | {'C': 1.0} |
| women | logistic | 5 | -0.009883 | 0.138559 | 0.235627 | 0.978436 | {'C': 3.0} |
| women | logistic | 6 | -0.009915 | 0.137947 | 0.233588 | 0.978071 | {'C': 10.0} |
| women | logistic | 7 | -0.009932 | 0.140749 | 0.238286 | 0.978609 | {'C': 0.01} |
| women | logistic | 8 | -0.010077 | 0.135011 | 0.230524 | 0.978080 | {'C': 0.005} |
| women | xgboost | 1 | -0.009775 | 0.143483 | 0.231848 | 0.979395 | {'colsample_bytree': 0.9, 'learning_rate': 0.015, 'max_depth': 4, 'min_child_weight': 15, 'n_estimators': 700, 'reg_alpha': 0.4, 'reg_lambda': 12.0, 'subsample': 0.92} |
| women | xgboost | 2 | -0.009776 | 0.143209 | 0.229964 | 0.979403 | {'colsample_bytree': 0.92, 'learning_rate': 0.03, 'max_depth': 3, 'min_child_weight': 20, 'n_estimators': 450, 'reg_alpha': 0.3, 'reg_lambda': 12.0, 'subsample': 0.92} |
| women | xgboost | 3 | -0.009779 | 0.142162 | 0.229449 | 0.979433 | {'colsample_bytree': 0.84, 'learning_rate': 0.02, 'max_depth': 5, 'min_child_weight': 20, 'n_estimators': 500, 'reg_alpha': 0.6, 'reg_lambda': 16.0, 'subsample': 0.86} |
| women | xgboost | 4 | -0.009779 | 0.142829 | 0.229802 | 0.979232 | {'colsample_bytree': 0.9, 'learning_rate': 0.035, 'max_depth': 3, 'min_child_weight': 15, 'n_estimators': 300, 'reg_alpha': 0.15, 'reg_lambda': 8.0, 'subsample': 0.9} |
| women | xgboost | 5 | -0.009781 | 0.142332 | 0.228540 | 0.979663 | {'colsample_bytree': 0.92, 'learning_rate': 0.02, 'max_depth': 3, 'min_child_weight': 25, 'n_estimators': 600, 'reg_alpha': 0.5, 'reg_lambda': 14.0, 'subsample': 0.92} |
| women | xgboost | 6 | -0.009790 | 0.141943 | 0.229340 | 0.979345 | {'colsample_bytree': 0.88, 'learning_rate': 0.025, 'max_depth': 4, 'min_child_weight': 12, 'n_estimators': 500, 'reg_alpha': 0.3, 'reg_lambda': 10.0, 'subsample': 0.9} |
| women | xgboost | 7 | -0.009791 | 0.141609 | 0.229509 | 0.979406 | {'colsample_bytree': 0.84, 'learning_rate': 0.03, 'max_depth': 5, 'min_child_weight': 18, 'n_estimators': 350, 'reg_alpha': 0.4, 'reg_lambda': 12.0, 'subsample': 0.86} |
| women | xgboost | 8 | -0.009792 | 0.142378 | 0.228838 | 0.979282 | {'colsample_bytree': 0.9, 'learning_rate': 0.05, 'max_depth': 3, 'min_child_weight': 12, 'n_estimators': 250, 'reg_alpha': 0.1, 'reg_lambda': 6.0, 'subsample': 0.9} |
| women | xgboost | 9 | -0.009794 | 0.141621 | 0.229726 | 0.979440 | {'colsample_bytree': 0.88, 'learning_rate': 0.035, 'max_depth': 4, 'min_child_weight': 10, 'n_estimators': 350, 'reg_alpha': 0.2, 'reg_lambda': 8.0, 'subsample': 0.9} |
| women | xgboost | 10 | -0.009795 | 0.141262 | 0.227161 | 0.979508 | {'colsample_bytree': 0.82, 'learning_rate': 0.025, 'max_depth': 6, 'min_child_weight': 25, 'n_estimators': 300, 'reg_alpha': 0.8, 'reg_lambda': 20.0, 'subsample': 0.84} |
| women | xgboost | 11 | -0.009811 | 0.140736 | 0.227954 | 0.979425 | {'colsample_bytree': 0.86, 'learning_rate': 0.018, 'max_depth': 4, 'min_child_weight': 30, 'n_estimators': 650, 'reg_alpha': 1.2, 'reg_lambda': 24.0, 'subsample': 0.88} |
| women | xgboost | 12 | -0.009858 | 0.139915 | 0.228022 | 0.978702 | {'colsample_bytree': 0.95, 'learning_rate': 0.03, 'max_depth': 3, 'min_child_weight': 40, 'n_estimators': 450, 'reg_alpha': 1.0, 'reg_lambda': 20.0, 'subsample': 0.95} |

## Notes
- Higher is better for `neg_log_loss`, `xA accuracy`, `AP`, and `ROC-AUC`; lower is better for final-test `Brier`, `log loss`, and absolute xA sum error.
- The final test set here is the isolated women final test set. Men-trained models are evaluated cross-domain on that same women test set; this run does not create a separate men final test set.
- By validation `neg_log_loss`, XGBoost is selected for both men and women, with a larger margin on men and a very small margin on women.
- On the women final test, `xgboost_men_model_on_women_final_test` has the best AP, ROC-AUC, and log loss; `xgboost_women_model_on_women_final_test` has the best xA accuracy, Brier score, and xA-sum calibration.
