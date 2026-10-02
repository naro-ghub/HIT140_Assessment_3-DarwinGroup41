# HIT140 Assessment 3 — Linear Regression 2.1

## Objective

Predict the **goal difference between two opposing teams** in each of the 104 FIFA World Cup 2026 matches.

`goal_difference = Team 1 goals - Team 2 goals`

Penalty-shootout totals are excluded; the football match score after regulation/extra time is used.

## Dataset

`match_goal_difference_104_rows.csv` contains exactly **104 rows**, one per World Cup match, and exactly **8 explanatory variables**.

### Eight pre-match explanatory variables

All variables compare Team 1 with Team 2 and are calculated from each team's **five matches completed before the match being predicted**.

1. `win_rate_diff_last5` — Team 1 win rate minus Team 2 win rate.
2. `clean_sheet_rate_diff_last5` — Team 1 clean-sheet rate minus Team 2 clean-sheet rate.
3. `avg_goals_scored_diff_last5` — difference in average goals scored.
4. `avg_goals_conceded_diff_last5` — difference in average goals conceded.
5. `draw_rate_diff_last5` — difference in draw rate.
6. `scoring_rate_diff_last5` — difference in the proportion of matches in which the team scored at least once.
7. `concede_2plus_rate_diff_last5` — difference in the proportion of matches conceding at least two goals.
8. `goal_diff_std_diff_last5` — difference in the standard deviation of match goal difference over the previous five matches.

The first four share concepts with Regression 2.2. Predictors 5–8 are distinct, satisfying the assignment rule that a maximum of four explanatory variables can be shared between the two regression tasks.

## Data sources

Only the assignment-approved sources are used in the project data workflow:

- FIFA Official Website — https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/statistics
- The Stats Don't Lie — https://www.thestatsdontlie.com/football/world-cup-2026/
- FBref — https://fbref.com/en/

The supplied `world_cup_2026_data.csv` is the 104-match World Cup results file used by the group workflow.

## Validation

The code checks that:

- there are exactly 104 World Cup match rows;
- match IDs are unique;
- exactly eight predictors are present;
- no predictor values are missing;
- every pre-match feature uses only matches dated before the match being predicted;
- penalty-shootout totals are not used as goals;
- the train/test split is chronological;
- validation is chronological using `TimeSeriesSplit`.

The World Cup match list/results were cross-checked against FIFA and FBref's 2026 World Cup results pages.

## Modelling

The code compares:

- Mean baseline
- Ordinary linear regression
- Ridge regression with alpha = 0.1, 1, 10 and 100

Model selection is performed on chronological validation data using RMSE. The held-out test data is used only for final evaluation.

For the supplied data, **Ridge alpha=10.0** is selected by validation.

### Final held-out test performance

| Model             | Test R² | Test MAE | Test RMSE |
| ----------------- | ------: | -------: | --------: |
| Mean baseline     |  -0.166 |    1.466 |     1.768 |
| Linear regression |  -0.140 |    1.404 |     1.748 |
| Ridge alpha=10.0  |  -0.013 |    1.349 |     1.648 |

The negative test R² indicates that predicting exact match goal difference remains difficult with these eight recent-form variables. Ridge nevertheless has the lowest held-out RMSE and MAE among the tested models.

## Main files

- `linear_regression_2_1.py` — complete analysis code
- `match_goal_difference_104_rows.csv` — final modelling dataset
- `validation_results_2_1.csv` — fold-by-fold chronological validation
- `validation_summary_2_1.csv` — pooled validation comparison
- `test_results_2_1.csv` — final held-out metrics
- `test_predictions_2_1.csv` — match-level test predictions and residuals
- `training_goal_difference_distribution.png` — training-target distribution
- `test_prediction_diagnostics_2_1.png` — actual/predicted and residual plots
- `validation_report_2_1.txt` — automated validation report
