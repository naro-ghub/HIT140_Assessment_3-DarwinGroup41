# HIT140 Assessment 3 — Objective 2 Comprehensive Evaluation Report

**Darwin Group 41 Codebase Audit: Regression 2.1 & Regression 2.2**

---

## 1.Summary

Both **Regression 2.1** and **Regression 2.2** have achieved the core analytical, data-wrangling, and modelling objectives set out in Assessment 3. The underlying data pipeline, historical match integration, rolling feature calculation, chronological train/validation/test splitting, and regularised regression modelling show strong technical execution.

However, several **compliance vulnerabilities**, **statistical issues**, and **project hygiene discrepancies** exist between the two sub-projects. If left unaddressed in the final group submission and written report, these issues could result in mark deductions.

### Highlights:

1. **Assignment Specification Compliance**:
   - **Row Counts**: Exact compliance. Regression 2.1 has exactly **104 rows** (match-level); Regression 2.2 has exactly **208 rows** (team-match level).
   - **Predictor Counts**: Exact compliance. Both models feature exactly **8 explanatory variables** (excluding identifiers).
   - **Pre-match Timing**: Exact compliance. Both pipelines strictly enforce `date < match_date` over the previous 5 completed matches, avoiding data leakage from in-match events.
   - **Shared Predictors Constraint**: **BORDERLINE RISK**. The brief allows a maximum of 4 shared explanatory variables. While 2.1 differences metrics and 2.2 separates team/opponent metrics, 6 of the 8 variables in 2.2 stem from the same 4 underlying metrics (goals scored, goals conceded, win rate, clean sheet rate) as 2.1. This must be justified carefully.
2. **Modelling & Statistical Rigor**:
   - **Multicollinearity in 2.1**: Severe multicollinearity (VIF up to 18.5) causes coefficient sign reversal in OLS linear regression (`win_rate_diff_last5` is -1.347 despite a +0.504 correlation with goal difference). Ridge regression ($\alpha=10$) resolves this, which is a major analytical finding that must be explained.
   - **Non-Linear Model in 2.2**: `linear_regression_2_2.py` includes a `RandomForestRegressor`. The brief specifically requires building and evaluating a _linear regression model_. While Ridge ($\alpha=100$) was ultimately selected, keeping Random Forest in the linear regression task introduces ambiguity.
   - **Count Target in 2.2**: Goals scored is a discrete non-negative integer count. Linear regression residuals significantly fail normality (Shapiro-Wilk $p = 0.004$).
   - **Negative $R^2$ in 2.1**: The test $R^2$ is negative across all models in 2.1 (-0.013 for Ridge, -0.140 for OLS) due to a domain-specific shift in the target distribution between training and test sets (neutral venue / knockout dynamics).
3. **Documentation & Repository Discrepancies**:
   - **Documentation Imbalance**: 2.1 has a complete, professional 88-line README with metric tables and explanations. 2.2 has a 21-line placeholder README with zero results, no validation explanation, and no discussion.
   - **Folder Naming & Loose Files**: Inconsistent capitalization (`Data`/`Figures`/`Results` vs `data`/`figures`/`results`). In addition, running the scripts dumps loose images and CSVs directly into root directories alongside subdirectories.
   - **Unused Files**: `Mexico_FIFA2026stats.csv` is present in `Regression 2.2/data/` but never referenced in code.
   - **Blocking GUI Calls**: `linear_regression_2_2.py` calls `plt.show()`, which blocks script execution in standard interactive environments, whereas `linear_regression_2_1.py` cleanly calls `plt.close()`.

---

## 2. Requirement-by-Requirement Compliance Matrix

| Assignment Requirement (`Document.pdf`) | Linear Regression 2.1 Status                       | Linear Regression 2.2 Status                     | Compliance Assessment & Auditor Notes                                                                                                                                                               |
| :-------------------------------------- | :------------------------------------------------- | :----------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Prediction Objective**                | Predict goal difference between two opposing teams | Predict number of goals scored by a single team  | **PASS**. Both mathematical targets are correctly formulated (`team1_goals - team2_goals` vs `goals_scored`).                                                                                       |
| **Dataset Size**                        | Exactly 104 rows                                   | Exactly 208 rows                                 | **PASS**. Exact row counts verified in code assertions and saved CSV outputs.                                                                                                                       |
| **Number of Explanatory Variables**     | Exactly 8 explanatory variables                    | Exactly 8 explanatory variables                  | **PASS**. Excluding match/team ID and target variables, exactly 8 feature columns are passed to models.                                                                                             |
| **Pre-match Timing Constraint**         | All variables available before match               | All variables available before match             | **PASS**. Rolling 5-match window strictly filters `date < match_date`. No in-match variables (shots, possession) are used.                                                                          |
| **Max 4 Shared Predictors Constraint**  | Differences across 8 metric pairs                  | Team/Opponent individual metrics + `is_knockout` | **BORDERLINE RISK**. The conceptual overlap between 2.1 and 2.2 requires explicit defense to avoid marker scrutiny.                                                                                 |
| **Data Source Attribution**             | Approved sources cited and used                    | Approved sources cited and used                  | **PASS**. Uses FBref, FIFA, and The Stats Don't Lie fixtures and tournament qualifiers.                                                                                                             |
| **Linear Regression Methodology**       | OLS Linear Regression & Ridge                      | OLS Linear Regression, Ridge, & Random Forest    | **PARTIAL PASS (2.2)**. 2.2 includes Random Forest, which is non-linear. The final selected model is Ridge (linear), but Random Forest should be contextualised merely as an exploratory benchmark. |

---

## 3. Deep-Dive Audit: Linear Regression 2.1

### 3.1 Dataset & Target Formulation

- **File**: `match_goal_difference_104_rows.csv` (104 rows $\times$ 16 columns total; 8 predictors).
- **Target**: `goal_difference = team1_goals - team2_goals`.
- **Target Distribution**:
  - Training mean: $+0.622$ goals (std: $2.09$)
  - Test mean: $-0.045$ goals (std: $1.68$)
- **Domain Observation**: In the 2026 World Cup fixture schedule, most matches are neutral-site games. The arbitrary assignment of teams into "Home" (`team1`) and "Away" (`team2`) resulted in an artificial $+0.622$ goal difference in the group stage (training set), while knockout games (test set) were evenly matched ($-0.045$). This explains why OLS learned an intercept of $+0.391$ and why test $R^2$ is negative.
- **Symmetry Consideration**: Because an intercept is fitted ($\beta_0 = 0.391$), swapping Team 1 and Team 2 changes the prediction non-symmetrically ($\hat{y}(T_1, T_2) \neq -\hat{y}(T_2, T_1)$). For head-to-head prediction, fitting without intercept (`fit_intercept=False`) or randomising team order is standard theoretical best practice.

### 3.2 Explanatory Variables & Multicollinearity

The 8 predictors used in Regression 2.1 are:

1. `win_rate_diff_last5`
2. `clean_sheet_rate_diff_last5`
3. `avg_goals_scored_diff_last5`
4. `avg_goals_conceded_diff_last5`
5. `draw_rate_diff_last5`
6. `scoring_rate_diff_last5`
7. `concede_2plus_rate_diff_last5`
8. `goal_diff_std_diff_last5`

#### Variance Inflation Factor (VIF) Diagnostic:

| Predictor                       | Correlation with Goal Difference | OLS Coefficient |    VIF    | Diagnosis                               |
| :------------------------------ | :------------------------------: | :-------------: | :-------: | :-------------------------------------- |
| `win_rate_diff_last5`           |            **+0.504**            |   **-1.347**    | **18.53** | **Severe multicollinearity; sign flip** |
| `avg_goals_conceded_diff_last5` |            **-0.378**            |   **+0.579**    | **11.75** | **Severe multicollinearity; sign flip** |
| `avg_goals_scored_diff_last5`   |              +0.460              |     +1.117      |   9.04    | High collinearity                       |
| `concede_2plus_rate_diff_last5` |              -0.333              |     -2.221      |   7.26    | Substantial collinearity                |
| `clean_sheet_rate_diff_last5`   |              +0.233              |     +1.289      |   4.54    | Moderate collinearity                   |
| `draw_rate_diff_last5`          |              -0.050              |     -1.264      |   4.08    | Moderate collinearity                   |
| `scoring_rate_diff_last5`       |              +0.469              |     +0.806      |   2.67    | Acceptable                              |
| `goal_diff_std_diff_last5`      |              -0.229              |     -0.883      |   2.56    | Acceptable                              |

**Critical Insight for the Report**:
Notice how `win_rate_diff_last5` has a strong positive correlation ($+0.504$) with goal difference, yet its OLS regression coefficient is negative ($-1.347$). Similarly, `avg_goals_conceded_diff_last5` has a negative correlation ($-0.378$), but its OLS coefficient is positive ($+0.579$).
This is textbook variance inflation caused by inter-correlated predictors (win rate is a direct mathematical consequence of goals scored and conceded).

- **Ridge Regression to the Rescue**: Ridge regression ($\alpha=10.0$) shrinks these unstable weights and achieves the lowest validation RMSE ($1.840$) and test RMSE ($1.648$).
- **Student Action Item**: This must be highlighted in the written report as evidence of statistical maturity and as the explicit justification for why Ridge was chosen over OLS.

### 3.3 Residual Diagnostics & Performance

- **Validation**: 3-fold chronological `TimeSeriesSplit`. Ridge $\alpha=10$ selected.
- **Held-out Test Metrics**:
  - Mean baseline: $R^2 = -0.166$, $\text{MAE} = 1.466$, $\text{RMSE} = 1.768$
  - Linear regression (OLS): $R^2 = -0.140$, $\text{MAE} = 1.404$, $\text{RMSE} = 1.748$
  - Ridge ($\alpha=10.0$): $R^2 = -0.013$, $\text{MAE} = 1.349$, $\text{RMSE} = 1.648$
- **Normality**: Shapiro-Wilk test on test residuals yields $W = 0.968, p = 0.654$. Residuals are normally distributed, confirming the linearity assumption holds reasonably well for goal differences.

---

## 4. Deep-Dive Audit: Linear Regression 2.2

### 4.1 Dataset & Target Formulation

- **File**: `team_matches_208_rows.csv` (208 rows $\times$ 14 columns total; 8 predictors).
- **Target**: `goals_scored` (integer count: 0 to 7).
- **Data Structure**: 104 matches $\times$ 2 teams = 208 rows.
  - Rows from the same match share temporal and situational context.
  - The script correctly handles this during train/test split and cross-validation by splitting on unique dates (`training_dates`), ensuring both teams from any given match remain in the same partition.

### 4.2 Explanatory Variables

The 8 predictors used in Regression 2.2 are:

1. `is_knockout` (binary: 0 = group stage, 1 = knockout stage)
2. `team_win_rate_last5`
3. `opponent_clean_sheet_rate_last5`
4. `team_scored_last5`
5. `team_conceded_last5`
6. `opponent_scored_last5`
7. `opponent_conceded_last5`
8. `team_scoring_std_last5`

#### VIF and Coefficient Health:

| Predictor                         | Correlation with Goals Scored | Coefficient | VIF  | Directional Intuition                                    |
| :-------------------------------- | :---------------------------: | :---------: | :--: | :------------------------------------------------------- |
| `team_scored_last5`               |            +0.216             |   +0.494    | 4.13 | Positive (More recent goals $\to$ higher expected goals) |
| `opponent_conceded_last5`         |            +0.263             |   +0.307    | 2.43 | Positive (Weaker opponent defence $\to$ more goals)      |
| `team_win_rate_last5`             |            +0.216             |   +0.133    | 4.49 | Positive (Better form $\to$ more goals)                  |
| `is_knockout`                     |            -0.045             |   -0.142    | 1.26 | Negative (Knockout matches are tighter)                  |
| `opponent_clean_sheet_rate_last5` |            -0.168             |   -0.179    | 2.23 | Negative (Solid opponent defence $\to$ fewer goals)      |
| `team_conceded_last5`             |            -0.166             |   -0.190    | 1.81 | Negative (Vulnerable teams score slightly fewer)         |
| `team_scoring_std_last5`          |            +0.031             |   -0.336    | 1.57 | Negative (Volatile scoring lowers median output)         |
| `opponent_scored_last5`           |            -0.309             |   -0.467    | 1.23 | Negative (Strong opponents dominate possession)          |

- **Health Assessment**: Unlike 2.1, all VIFs in 2.2 are safely below 5.0 (maximum 4.49). Every single coefficient sign matches its intuitive football direction.

### 4.3 Methodological Violations & Model Selection

1. **Inclusion of Non-Linear Model**: `linear_regression_2_2.py` trains `RandomForestRegressor`.
   - The brief states: _"Build and evaluate a second linear regression model that predicts the number of goals scored..."_
   - While exploring Random Forest demonstrates curiosity, it must be clearly marked as an auxiliary baseline in the report, not as the primary assignment task.
   - Fortunately, the chronological validation selected **Ridge alpha=100.0** (Validation RMSE = 1.318 vs Random Forest RMSE = 1.354), keeping the final submitted model within the linear family.
2. **Residual Normality & Count Nature**:
   - `goals_scored` takes integer values $\{0, 1, 2, 3, 4, 5, 6, 7\}$ with strong skew (44 zeros, 58 ones).
   - Shapiro-Wilk test on test residuals yields $W = 0.918, p = 0.00416$ (rejecting normality at $\alpha = 0.01$).
   - Linear regression on count data produces continuous non-integer predictions (e.g., $1.42$ goals). In the report, the group should discuss this limitation and mention Poisson regression or negative binomial regression as the theoretical ideal for goal modeling.
3. **Held-out Test Performance**:
   - Mean baseline: $R^2 = -0.002, \text{MAE} = 1.065, \text{RMSE} = 1.324$
   - Random Forest: $R^2 = -0.001, \text{MAE} = 1.002, \text{RMSE} = 1.324$
   - Linear Regression (OLS): $R^2 = 0.037, \text{MAE} = 0.974, \text{RMSE} = 1.298$
   - Ridge ($\alpha=100.0$): $R^2 = 0.052, \text{MAE} = 1.000, \text{RMSE} = 1.288$
   - An $R^2$ of $5.2\%$ is realistic in single-match football goal prediction. Ridge slightly outperformed OLS on RMSE.

---

## 5. Critical Shared Variable Constraint Audit

The assignment brief states:

> _"Only a maximum of 4 explanatory variables can be shared between Linear Regression 2.1 and Linear Regression 2.2"_

### Comparative Analysis of Predictor Sets:

```
Regression 2.1 Predictors (104 rows):
  1. win_rate_diff_last5            [Diff: Team 1 - Team 2]
  2. clean_sheet_rate_diff_last5    [Diff: Team 1 - Team 2]
  3. avg_goals_scored_diff_last5    [Diff: Team 1 - Team 2]
  4. avg_goals_conceded_diff_last5  [Diff: Team 1 - Team 2]
  5. draw_rate_diff_last5           [Diff: Team 1 - Team 2]  (Unique to 2.1)
  6. scoring_rate_diff_last5        [Diff: Team 1 - Team 2]  (Unique to 2.1)
  7. concede_2plus_rate_diff_last5  [Diff: Team 1 - Team 2]  (Unique to 2.1)
  8. goal_diff_std_diff_last5       [Diff: Team 1 - Team 2]  (Unique to 2.1)

Regression 2.2 Predictors (208 rows):
  1. is_knockout                    [Match context]          (Unique to 2.2)
  2. team_win_rate_last5            [Team individual]        (Shares concept with 2.1 #1)
  3. opponent_clean_sheet_rate_last5[Opponent individual]    (Shares concept with 2.1 #2)
  4. team_scored_last5              [Team individual]        (Shares concept with 2.1 #3)
  5. team_conceded_last5            [Team individual]        (Shares concept with 2.1 #4)
  6. opponent_scored_last5          [Opponent individual]    (Shares concept with 2.1 #3)
  7. opponent_conceded_last5        [Opponent individual]    (Shares concept with 2.1 #4)
  8. team_scoring_std_last5         [Team individual]        (Variance concept)
```

### Risk Assessment & How to Protect Marks:

- **Literal Variable Level**: Zero column names are identical. 2.1 columns are difference scores ($\Delta$ metrics for match dyads); 2.2 columns are single-team and opponent metrics. By this strict technical definition, zero variables are shared.
- **Conceptual Domain Level**:
  - The README for 2.1 claims: _"The first four share concepts with Regression 2.2. Predictors 5–8 are distinct, satisfying the assignment rule that a maximum of four explanatory variables can be shared between the two regression tasks."_
  - **The Risk**: A marker reviewing 2.2 will observe that features 4, 5, 6, and 7 represent four separate columns (`team_scored`, `team_conceded`, `opponent_scored`, `opponent_conceded`). Adding `team_win_rate` and `opponent_clean_sheet_rate` brings the total to **6 variables** derived from the same base statistics.
  - **Recommendation**: In the report, the group must explicitly define the term _"shared variable"_. Emphasise that:
    1. The mathematical definition of a variable in 2.1 is a **pairwise difference operator** $\Delta X = X_1 - X_2$, which is mathematically distinct from an individual team statistic.
    2. Exactly 4 core historical performance dimensions are bridged between tasks (Win Rate, Clean Sheet Rate, Goals Scored, Goals Conceded), while each task introduces 4 completely distinct dimensions (2.1: Draw rate, scoring rate, 2+ conceded rate, goal diff volatility; 2.2: Tournament stage, opponent scoring, opponent conceding, single-team scoring volatility).

---

## 6. Code Quality, Project Hygiene & Directory Structure

### 6.1 Directory & File Inconsistencies

| Item                     | Regression 2.1                                                                   | Regression 2.2                                                           | Recommended Standardisation                                                                                         |
| :----------------------- | :------------------------------------------------------------------------------- | :----------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------ |
| **Directory Casing**     | Capitalized: `Data/`, `Figures/`, `Results/`                                     | Lowercase: `data/`, `figures/`, `results/`                               | Use consistent lowercase across both (`data/`, `figures/`, `results/`).                                             |
| **Output Pathing**       | Creates `results_2_1/` and `figures_2_1/`, but duplicates files into root folder | Outputs some files to `results/` and some directly to root (`base_path`) | Route all generated figures strictly to `figures/` and all CSVs strictly to `results/`. Remove root folder clutter. |
| **Orphan Data Files**    | None                                                                             | `Mexico_FIFA2026stats.csv` (never used in script)                        | Delete or document `Mexico_FIFA2026stats.csv`.                                                                      |
| **Matplotlib Execution** | Clean `plt.close(fig)` (non-blocking)                                            | `plt.show()` on lines 485 and 740 (blocks script execution in GUI)       | Replace `plt.show()` with `plt.close(fig)` in `linear_regression_2_2.py`.                                           |
| **README Quality**       | Comprehensive (88 lines, tables, methodology, full file index)                   | Sparse placeholder (21 lines, only lists 8 bullets, no results)          | Expand `Regression 2.2/README.md` to match 2.1's professional standard.                                             |
| **Text Report Output**   | Outputs `validation_report_2_1.txt`                                              | None                                                                     | Add automated text summary generation to 2.2.                                                                       |

---

## 7. Actionable Remediation Checklist for the Group

To ensure full marks on Objective 2, the group should complete the following actions:

- [ ] **3. Upgrade `Regression 2.2/README.md`**:
  - Document the chronological train/test split (82 training matches / 22 test matches; 164 train rows / 44 test rows).
  - Add tables showing validation results across folds and final held-out test metrics.
  - Explain the selection of Ridge ($\alpha=100.0$).
- [ ] **4. Address the Shared Predictors in the Written Report**:
  - Dedicate a specific subsection to explaining the exact difference between the 8 variables in 2.1 and the 8 variables in 2.2, directly addressing the "maximum 4 shared variables" rubric criteria.
- [ ] **5. Include Key Statistical Insights in the Report**:
  - **Multicollinearity in 2.1**: Explain why OLS coefficient signs flipped and why Ridge regression resolved this.
  - **Count Data Nature in 2.2**: Discuss that goals scored is discrete and Poisson-like, explaining residual non-normality and modest $R^2$.
  - **Neutral Venue Intercept Shift in 2.1**: Explain why the training set had an artificial positive goal difference (+0.62) while the test set was even (-0.05), causing negative test $R^2$.
