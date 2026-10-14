# HIT140 Assessment 3 — FIFA World Cup 2026 Prediction Project
**Charles Darwin University | HIT140 Foundations of Data Science**  
**Group:** Darwin Group 41  
**Focus:** Objective 2 — Linear Regression Analysis (Tasks 2.1 & 2.2)

---

## Project Overview

This repository contains the complete analytical pipeline, datasets, statistical models, diagnostic figures, and validation results for **Objective 2** of HIT140 Assessment 3. 

The project investigates match outcomes and scoring dynamics in the **FIFA World Cup 2026** using regularized linear regression models built with Python (`pandas`, `numpy`, `scikit-learn`, `matplotlib`).

### Objective 2 Summary Matrix

| Metric / Dimension | Linear Regression 2.1 | Linear Regression 2.2 |
| :--- | :--- | :--- |
| **Objective** | Predict the **goal difference** between two opposing teams | Predict the **number of goals scored** by a single team |
| **Target Variable** | `goal_difference = Team 1 goals - Team 2 goals` | `goals_scored` (Team goals in regulation/extra time) |
| **Unit of Analysis** | Match-level (1 row per match) | Team-match level (2 rows per match; 1 per team perspective) |
| **Total Rows** | **Exactly 104 rows** | **Exactly 208 rows** |
| **Explanatory Variables** | **Exactly 8 variables** (pair differences) | **Exactly 8 variables** (context, team, and opponent features) |
| **Pre-Match Timing** | Strictly prior to kickoff (`date < match_date`) | Strictly prior to kickoff (`date < match_date`) |
| **Validation Strategy** | Chronological 3-fold `TimeSeriesSplit` | Chronological 3-fold `TimeSeriesSplit` (grouped by date) |
| **Train / Test Split** | Chronological (~79% train / ~21% held-out test) | Chronological (~79% train / ~21% held-out test) |
| **Selected Model** | **Ridge Regression (alpha = 10.0)** | **Ridge Regression (alpha = 100.0)** |
| **Held-Out Test Metrics** | Test RMSE = 1.648 \| Test MAE = 1.349 | Test RMSE = 1.288 \| Test MAE = 1.000 \| Test R² = 0.052 |

---

## Compliance with the "Maximum 4 Shared Explanatory Variables" Rule

The assignment brief specifies:
> *"Only a maximum of 4 explanatory variables can be shared between Linear Regression 2.1 and Linear Regression 2.2"*

To ensure strict adherence, Darwin Group 41 structured the two predictor sets as follows:

```
Regression 2.1 Predictors (Match Dyad Level — 104 Rows):
  1. win_rate_diff_last5            [Diff: Team 1 - Team 2]   <-- Shared Concept 1
  2. clean_sheet_rate_diff_last5    [Diff: Team 1 - Team 2]   <-- Shared Concept 2
  3. avg_goals_scored_diff_last5    [Diff: Team 1 - Team 2]   <-- Shared Concept 3
  4. avg_goals_conceded_diff_last5  [Diff: Team 1 - Team 2]   <-- Shared Concept 4
  5. draw_rate_diff_last5           [Diff: Team 1 - Team 2]   [UNIQUE to 2.1]
  6. scoring_rate_diff_last5        [Diff: Team 1 - Team 2]   [UNIQUE to 2.1]
  7. concede_2plus_rate_diff_last5  [Diff: Team 1 - Team 2]   [UNIQUE to 2.1]
  8. goal_diff_std_diff_last5       [Diff: Team 1 - Team 2]   [UNIQUE to 2.1]

Regression 2.2 Predictors (Team-Match Level — 208 Rows):
  1. is_knockout                    [Match context: Group vs Knockout] [UNIQUE to 2.2]
  2. team_win_rate_last5            [Single team win proportion]       <-- Shared Concept 1
  3. opponent_clean_sheet_rate_last5[Opponent defensive solidity]      <-- Shared Concept 2
  4. team_scored_last5              [Team attacking potency]           <-- Shared Concept 3
  5. team_conceded_last5            [Team defensive vulnerability]     <-- Shared Concept 4
  6. opponent_scored_last5          [Opponent attacking threat]        [UNIQUE to 2.2]
  7. opponent_conceded_last5        [Opponent defensive leakiness]     [UNIQUE to 2.2]
  8. team_scoring_std_last5         [Single team scoring consistency]  [UNIQUE to 2.2]
```

### Justification:
1. **Mathematical Distinctness**: Zero column names are shared between the two datasets. Regression 2.1 employs pairwise difference operators $\Delta X = X_{\text{team1}} - X_{\text{team2}}$, while Regression 2.2 operates on single-team observations.
2. **Conceptual Boundary**: Exactly **four core performance concepts** bridge both tasks (Win Rate, Clean Sheet Rate, Goals Scored, Goals Conceded). Each task incorporates **four entirely distinct variables** tailored to its prediction objective:
   - Regression 2.1 introduces match draw rate, frequency of scoring $\ge 1$, frequency of conceding $\ge 2$, and volatility of goal margin.
   - Regression 2.2 introduces tournament round type (`is_knockout`), opponent attacking threat, opponent defensive leakiness, and single-team scoring volatility.

---

## Data Pipeline & Pre-Match Integrity

### Data Sources
Data was compiled exclusively from the assignment-approved repositories:
- **FIFA Official Website**: 2026 tournament statistics and match outcomes.
- **The Stats Don't Lie**: Detailed form metrics and historical match statistics.
- **FBref**: Official schedule, match reports, and international fixtures (2025–2026).

### Temporal Integrity & Feature Engineering
- **No In-Match Leakage**: All features reflect historical performance completed **strictly prior to the match date** (`date < match_date`). In-match metrics (e.g., possession percentage, shots on target) were rigorously excluded.
- **Rolling Window**: Each team's form is computed across its **five most recent matches** (combining pre-tournament qualifiers, friendlies, and completed World Cup games).
- **Penalty Shootouts**: Matches decided by penalty shootouts are treated as draws (using the score at the end of regulation or extra time), adhering to FIFA and IFAB standards.

---

## Modelling Methodology & Results

### 1. Linear Regression 2.1 (Goal Difference)
- **Training Set**: 82 matches (prior to `2026-07-02`).
- **Held-Out Test Set**: 22 matches (`2026-07-02` onwards; Round of 32 through Final).
- **Model Selection**: 3-fold chronological `TimeSeriesSplit`. Ridge ($\alpha=10.0$) selected with lowest validation RMSE (1.840).

| Model | Test R² | Test MAE | Test RMSE |
| :--- | :---: | :---: | :---: |
| **Ridge (alpha = 10.0)** | **-0.013** | **1.349** | **1.648** |
| Linear Regression (OLS) | -0.140 | 1.404 | 1.748 |
| Mean Baseline | -0.166 | 1.466 | 1.768 |

**Key Statistical Finding**:
- **Multicollinearity & Sign Flip**: `win_rate_diff_last5` exhibited severe multicollinearity (VIF = 18.53). In OLS, its coefficient flipped to $-1.347$ despite a strong positive correlation ($+0.504$) with goal difference. Ridge regularisation successfully penalised extreme weights, providing stable coefficients and lowest test error.
- **Target Shift**: In group stages (training data), the designated "Home" team enjoyed an average $+0.622$ goal margin, whereas knockout games (test data) were evenly balanced ($-0.045$). This domain shift explains the negative test $R^2$.

### 2. Linear Regression 2.2 (Goals Scored)
- **Training Set**: 164 rows (82 matches).
- **Held-Out Test Set**: 44 rows (22 matches).
- **Model Selection**: 3-fold chronological `TimeSeriesSplit`. Ridge ($\alpha=100.0$) selected with lowest validation RMSE (1.318).

| Model | Test R² | Test MAE | Test RMSE |
| :--- | :---: | :---: | :---: |
| **Ridge (alpha = 100.0)** | **0.052** | **1.000** | **1.288** |
| Linear Regression (OLS) | 0.037 | 0.974 | 1.298 |
| Random Forest (Benchmark) | -0.001 | 1.002 | 1.324 |
| Mean Baseline | -0.002 | 1.065 | 1.324 |

**Key Statistical Finding**:
- **Collinearity Health**: All VIF values in 2.2 are $< 5.0$. Coefficients adhere to football intuition (e.g., higher recent scoring $+0.494$, tighter knockout defense $-0.142$).
- **Count Data Nature**: Goals scored follows a discrete, right-skewed distribution. Shapiro-Wilk testing rejects residual normality ($p = 0.00416$). $R^2 = 5.2\%$ is realistic for single-match goal count variance in international football.

---

## Repository Structure

```
HIT140_Assessment_3-DarwinGroup41/
├── README.md                              <- Main project documentation (this file)
├── Report.md                              <- Academic evaluation and audit report
├── Regression 2.1/                        <- Task 2.1: Goal Difference Prediction
│   ├── README.md                          <- Detailed Task 2.1 documentation
│   ├── linear_regression_2_1.py           <- Complete analysis script for Task 2.1
│   ├── Data/                              <- 10 raw CSV match and qualifier datasets
│   │   └── world_cup_2026_data.csv        <- 104 World Cup match results
│   ├── Figures/                           <- Visualizations
│   │   ├── training_goal_difference_distribution.png
│   │   └── test_prediction_diagnostics_2_1.png
│   └── Results/                           <- Model outputs and validation tables
│       ├── match_goal_difference_104_rows.csv
│       ├── validation_results_2_1.csv
│       ├── validation_summary_2_1.csv
│       ├── test_results_2_1.csv
│       ├── test_predictions_2_1.csv
│       └── validation_report_2_1.txt
└── Regression 2.2/                        <- Task 2.2: Team Goals Scored Prediction
    ├── README.md                          <- Detailed Task 2.2 documentation
    ├── linear_regression_2_2.py           <- Complete analysis script for Task 2.2
    ├── Data/                              <- 9 raw CSV match and qualifier datasets
    │   └── world_cup_2026_data.csv        <- 104 World Cup match results
    ├── Figures/                           <- Visualizations
    │   ├── training_goals_distribution.png
    │   └── test_prediction_diagnostics.png
    └── Results/                           <- Model outputs and validation tables
        ├── team_matches_208_rows.csv
        ├── validation_results.csv
        ├── validation_summary.csv
        ├── test_results.csv
        ├── test_predictions.csv
        └── validation_report_2_2.txt
```

---

## Instructions to Reproduce

### Prerequisites
Ensure Python 3.10+ is installed with the following packages:
```bash
pip install pandas numpy scikit-learn matplotlib
```

### Running the Analysis
To execute Regression 2.1:
```bash
cd "Regression 2.1"
python linear_regression_2_1.py
```

To execute Regression 2.2:
```bash
cd "Regression 2.2"
python linear_regression_2_2.py
```

Both scripts run non-interactively, perform all automated validation checks, and output tables, metrics, and high-resolution figures into their respective `Results/` and `Figures/` folders.
