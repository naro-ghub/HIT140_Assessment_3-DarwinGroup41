# HIT140 Assessment 3 — Linear Regression 2.2

## Objective

Build and evaluate a linear regression model that predicts the **number of goals scored by a team in a single FIFA World Cup 2026 match**.

`target = goals_scored` (goals scored by the team in regulation and extra time; penalty shootouts excluded).

## Dataset

`team_matches_208_rows.csv` contains exactly **208 rows** (each representing one team's statistics for one of the 104 World Cup matches; 2 rows per match) and exactly **8 explanatory variables** (excluding identifiers and target variables).

### Eight Pre-Match Explanatory Variables

All explanatory variables use information available **strictly before kickoff**, computed from each team's previous five completed matches:

1. `is_knockout` — Binary indicator (0 = Group stage match, 1 = Knockout stage match). Represents match intensity and defensive tactics in elimination rounds.
2. `team_win_rate_last5` — Team's win proportion over its previous five completed matches.
3. `opponent_clean_sheet_rate_last5` — Opponent's proportion of matches with zero goals conceded over its previous five matches (measures opposing defensive resilience).
4. `team_scored_last5` — Average goals scored by the team per match over its previous five matches (team attacking potency).
5. `team_conceded_last5` — Average goals conceded by the team per match over its previous five matches.
6. `opponent_scored_last5` — Average goals scored by the opponent per match over its previous five matches (possession pressure / offensive threat).
7. `opponent_conceded_last5` — Average goals conceded by the opponent per match over its previous five matches (defensive vulnerability of opponent).
8. `team_scoring_std_last5` — Sample standard deviation of the team's goals scored over its previous five matches (scoring consistency/volatility).

### Shared Explanatory Variables Compliance

The assignment specification states that **a maximum of 4 explanatory variables can be shared between Linear Regression 2.1 and Linear Regression 2.2**.

- **Exact Column Level**: Zero column names are identical. Regression 2.1 constructs dyadic difference metrics between Team 1 and Team 2 (`..._diff_last5`), whereas Regression 2.2 constructs individual team and opponent features.
- **Conceptual Dimension Level**: Exactly 4 core historical performance dimensions are bridged between tasks (Win Rate, Clean Sheet Rate, Goals Scored, and Goals Conceded). The remaining 4 dimensions in Regression 2.2 (`is_knockout`, opponent offensive threat, opponent defensive vulnerability, and single-team scoring volatility) are distinct from the unique predictors in Regression 2.1 (draw rate, scoring rate, 2+ conceded rate, and goal difference volatility).

## Data Sources

All data was sourced strictly from the assignment-approved sources:
- **FIFA Official Website**: [FIFA 2026 Statistics](https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/statistics)
- **The Stats Don't Lie**: [World Cup 2026](https://www.thestatsdontlie.com/football/world-cup-2026/)
- **FBref**: [FBref Football Statistics](https://fbref.com/en/)

Pre-tournament form was compiled from official 2025–2026 international fixtures (qualifiers across UEFA, AFC, CAF, CONCACAF, Africa Cup of Nations, and international friendlies).

## Data Validation

Automated pipeline checks confirm:
- Exactly 208 team-match rows (104 matches × 2 teams).
- Exactly 104 unique matches; each match appears exactly twice.
- No duplicate team-match records.
- Exactly 8 explanatory variables passed to models.
- Zero missing values across all predictors.
- Temporal integrity: all form metrics enforce `date < match_date`.
- Regulation/extra-time scores used; shootout totals excluded.
- Matches played on the same date are kept together in partitions to prevent data leakage.

## Modelling Methodology & Model Selection

### Chronological Train/Test Split
- **Training Set**: Matches before `2026-07-02` (82 matches = 164 team-match rows, ~79%).
- **Held-Out Test Set**: Matches on or after `2026-07-02` (22 matches = 44 team-match rows, ~21%).

### Chronological Cross-Validation (`TimeSeriesSplit`)
A 3-fold chronological `TimeSeriesSplit` across training match dates was used to tune regularization parameter $\alpha$:

| Model | Validation RMSE | Validation MAE |
| :--- | :---: | :---: |
| **Ridge alpha=100.0** | **1.318** | **1.048** |
| Random forest (benchmark) | 1.354 | 1.076 |
| Ridge alpha=10.0 | 1.369 | 1.063 |
| Mean baseline | 1.395 | 1.159 |
| Ridge alpha=1.0 | 1.458 | 1.116 |
| Ridge alpha=0.1 | 1.485 | 1.134 |
| Linear regression (OLS) | 1.489 | 1.137 |

**Model Selection Decision**: **Ridge alpha=100.0** achieved the lowest validation RMSE and MAE among all linear models and was selected for final held-out test evaluation.

### Final Held-Out Test Performance

| Model | Test R² | Test MAE | Test RMSE |
| :--- | :---: | :---: | :---: |
| **Ridge alpha=100.0** | **0.052** | **1.000** | **1.288** |
| Linear regression (OLS) | 0.037 | 0.974 | 1.298 |
| Mean baseline | -0.002 | 1.065 | 1.324 |
| Random forest (benchmark) | -0.001 | 1.002 | 1.324 |

## Key Findings & Statistical Discussion

1. **Multicollinearity & Interpretability**:
   - Unlike Regression 2.1, all variance inflation factors (VIFs) in Regression 2.2 are below 5.0 (highest: 4.49 for `team_win_rate_last5` and 4.13 for `team_scored_last5`).
   - All regression coefficients exhibit intuitive directions:
     - `team_scored_last5` (+0.494) and `opponent_conceded_last5` (+0.307) positively drive goals.
     - `opponent_scored_last5` (-0.467) and `opponent_clean_sheet_rate_last5` (-0.179) reduce expected goals.
     - `is_knockout` (-0.142) confirms elimination matches are lower-scoring.
2. **Count Target Dynamics**:
   - `goals_scored` is a non-negative integer count (0 to 7) with right skew.
   - Shapiro-Wilk testing rejects residual normality ($W = 0.918, p = 0.00416$). In the written report, Poisson and Negative Binomial regression are discussed as theoretical benchmarks for count modeling.
3. **Variance Explained ($R^2 = 5.2\%$)**:
   - In international football, single-match goal events carry high stochastic variance; an $R^2$ of 5% with pre-match form features aligns with established sports analytics literature.

## Main Files

- `linear_regression_2_2.py` — Complete data pipeline, modeling, and evaluation code.
- `Results/team_matches_208_rows.csv` — Final modeling dataset (208 rows × 8 predictors).
- `Results/validation_results.csv` — Fold-by-fold chronological validation metrics.
- `Results/validation_summary.csv` — Pooled cross-validation summary table.
- `Results/test_results.csv` — Final test set metrics.
- `Results/test_predictions.csv` — Match-level predictions and residuals.
- `Results/validation_report_2_2.txt` — Automated text validation report.
- `Figures/training_goals_distribution.png` — Distribution of goals scored in training data.
- `Figures/test_prediction_diagnostics.png` — Actual vs. predicted and residual diagnostic plots.
