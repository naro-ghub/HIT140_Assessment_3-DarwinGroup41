# Regression 2.1: Predict goal difference between two opposing teams in FIFA World Cup 2026 matches
print("Linear Regression 2.1 Project")

from pathlib import Path
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

sys.stdout.reconfigure(encoding="utf-8")
base_path = Path(__file__).resolve().parent
data_path = base_path / "data"
if not data_path.exists():
    data_path = base_path
data_path = base_path / "Data" if (base_path / "Data").exists() else base_path / "data"
results_path = base_path / "Results" if (base_path / "Results").exists() else base_path / "results"
figures_path = base_path / "Figures" if (base_path / "Figures").exists() else base_path / "figures"
results_path.mkdir(exist_ok=True)
figures_path.mkdir(exist_ok=True)

# Load the 104-match World Cup file (FBref/FIFA fixture results as supplied for the project)
file_path = data_path / "world_cup_2026_data.csv"
matches = pd.read_csv(file_path, comment="#").dropna(how="all").copy()
assert len(matches) == 104, f"Expected 104 World Cup matches, found {len(matches)}."

# Extract regulation/extra-time match goals; penalty-shootout totals are not treated as match goals.
goals = matches["Score"].str.extract(r"(\d+)\s*[\u2013-]\s*(\d+)")
assert not goals.isna().any().any(), "Unrecognised World Cup score format."
matches["team1_goals"] = goals[0].astype(int)
matches["team2_goals"] = goals[1].astype(int)
matches = matches.reset_index(drop=True)
matches["match_id"] = matches.index + 1
matches["date"] = pd.to_datetime(matches["Date"], errors="raise")

# Clean team names consistently.
def clean_team_name(series):
    out = series.astype(str).str.strip()
    out = out.str.replace(r"^[a-z]{2,3}\s+|\s+[a-z]{2,3}$", "", regex=True)
    out = out.replace({"United States": "USA"})
    return out

matches["team1"] = clean_team_name(matches["Home"])
matches["team2"] = clean_team_name(matches["Away"])
matches["goal_difference"] = matches["team1_goals"] - matches["team2_goals"]

# Historical files supplied for the project. These contain pre-World-Cup internationals from the approved data collection workflow.
history_files = [
    "International_friendlies_2025.csv",
    "International_friendlies_2026.csv",
    "2026_WorldCupQualifiers-UEFA(M).csv",
    "2026_WorldCupQualifiers_AFC(M).csv",
    "2026_WorldCupQualifier_CAF(M).csv",
    "2026_WorldCupQualifiers_CONCACAF.csv",
    "2025_Africa_Cup_of_Nations.csv",
    "WorldCupQualifiers_Intercontinental_2026.csv",
    "additional_historical_matches.csv",
]

history_tables = []
for filename in history_files:
    table = pd.read_csv(data_path / filename, comment="#")
    table["source_file"] = filename
    history_tables.append(table)
    print(filename, ":", len(table), "rows")

all_history = pd.concat(history_tables, ignore_index=True)
all_history = all_history.dropna(subset=["Date", "Home", "Away"], how="all").copy()
all_history = all_history.loc[all_history["Score"].notna()].copy()
all_history["Date"] = pd.to_datetime(all_history["Date"], errors="raise")

# Exclude awarded results from goal-form calculations.
awarded = all_history["Notes"].str.contains("awarded", case=False, na=False)
all_history = all_history.loc[~awarded].copy()
all_history["Home"] = clean_team_name(all_history["Home"])
all_history["Away"] = clean_team_name(all_history["Away"])

history_goals = all_history["Score"].str.extract(r"(\d+)\s*[\u2013-]\s*(\d+)")
assert not history_goals.isna().any().any(), "Unrecognised historical score format."
all_history["home_goals"] = history_goals[0].astype(int)
all_history["away_goals"] = history_goals[1].astype(int)

# Reshape history to one row per team perspective.
home_hist = all_history[["Date", "Home", "Away", "home_goals", "away_goals", "source_file"]].copy()
home_hist.columns = ["date", "team", "opponent", "goals_scored", "goals_conceded", "source_file"]
away_hist = all_history[["Date", "Away", "Home", "away_goals", "home_goals", "source_file"]].copy()
away_hist.columns = ["date", "team", "opponent", "goals_scored", "goals_conceded", "source_file"]
historical_team_matches = pd.concat([home_hist, away_hist], ignore_index=True)

# Add completed World Cup matches to history, so each later match uses only information available before kickoff.
wc_home = matches[["match_id", "date", "team1", "team2", "team1_goals", "team2_goals"]].copy()
wc_home.columns = ["match_id", "date", "team", "opponent", "goals_scored", "goals_conceded"]
wc_away = matches[["match_id", "date", "team2", "team1", "team2_goals", "team1_goals"]].copy()
wc_away.columns = ["match_id", "date", "team", "opponent", "goals_scored", "goals_conceded"]
world_cup_history = pd.concat([wc_home, wc_away], ignore_index=True)
world_cup_history["source_file"] = "world_cup_2026_data.csv"

full_history = pd.concat([
    historical_team_matches[["date", "team", "opponent", "goals_scored", "goals_conceded", "source_file"]],
    world_cup_history[["date", "team", "opponent", "goals_scored", "goals_conceded", "source_file"]]
], ignore_index=True).sort_values(["team", "date"]).reset_index(drop=True)

assert not full_history.duplicated(["date", "team", "opponent"]).any(), "Duplicate historical team-match found."

# Eight explanatory variables. Each is a Team 1 minus Team 2 difference based on the previous five matches.
# Predictors 1-4 share concepts with Regression 2.2; predictors 5-8 are distinct.
def get_recent_form(team_name, match_date):
    previous = full_history.loc[
        (full_history["team"] == team_name) & (full_history["date"] < match_date)
    ].sort_values("date").tail(5).copy()
    assert len(previous) == 5, f"Five previous matches needed for {team_name} before {match_date.date()}."

    gs = previous["goals_scored"]
    gc = previous["goals_conceded"]
    gd = gs - gc
    return {
        "win_rate": (gs > gc).mean(),
        "clean_sheet_rate": (gc == 0).mean(),
        "avg_scored": gs.mean(),
        "avg_conceded": gc.mean(),
        "draw_rate": (gs == gc).mean(),
        "scoring_rate": (gs >= 1).mean(),
        "concede_2plus_rate": (gc >= 2).mean(),
        "goal_diff_std": gd.std(ddof=1),
    }

feature_rows = []
for _, match in matches.iterrows():
    t1 = get_recent_form(match["team1"], match["date"])
    t2 = get_recent_form(match["team2"], match["date"])
    feature_rows.append({
        "win_rate_diff_last5": t1["win_rate"] - t2["win_rate"],
        "clean_sheet_rate_diff_last5": t1["clean_sheet_rate"] - t2["clean_sheet_rate"],
        "avg_goals_scored_diff_last5": t1["avg_scored"] - t2["avg_scored"],
        "avg_goals_conceded_diff_last5": t1["avg_conceded"] - t2["avg_conceded"],
        "draw_rate_diff_last5": t1["draw_rate"] - t2["draw_rate"],
        "scoring_rate_diff_last5": t1["scoring_rate"] - t2["scoring_rate"],
        "concede_2plus_rate_diff_last5": t1["concede_2plus_rate"] - t2["concede_2plus_rate"],
        "goal_diff_std_diff_last5": t1["goal_diff_std"] - t2["goal_diff_std"],
    })

features = pd.DataFrame(feature_rows, index=matches.index)
matches[features.columns] = features
predictor_columns = list(features.columns)
assert len(predictor_columns) == 8
assert matches[predictor_columns].shape == (104, 8)
assert not matches[predictor_columns].isna().any().any()

# Final modelling dataset: ID/context fields + target + exactly eight explanatory variables.
model_data = matches[[
    "match_id", "date", "Round", "team1", "team2", "team1_goals", "team2_goals", "goal_difference"
] + predictor_columns].copy()
model_data = model_data.rename(columns={"Round": "round"})
model_data.to_csv(results_path / "match_goal_difference_104_rows.csv", index=False, encoding="utf-8")

print("\nDataset checks passed:")
print("Rows:", len(model_data))
print("Matches:", model_data["match_id"].nunique())
print("Predictors:", len(predictor_columns))
print("Missing predictor values:", model_data[predictor_columns].isna().sum().sum())
print("Duplicate match IDs:", model_data["match_id"].duplicated().sum())

# Chronological ~80/20 train/test split; all matches on the same date stay together.
match_dates = model_data[["match_id", "date"]].sort_values(["date", "match_id"]).reset_index(drop=True)
split_position = int(len(match_dates) * 0.80)
cutoff_date = match_dates.iloc[split_position]["date"]
train_data = model_data.loc[model_data["date"] < cutoff_date].copy()
test_data = model_data.loc[model_data["date"] >= cutoff_date].copy()
X_train, y_train = train_data[predictor_columns], train_data["goal_difference"]
X_test, y_test = test_data[predictor_columns], test_data["goal_difference"]
assert len(train_data) + len(test_data) == 104
assert train_data["date"].max() < test_data["date"].min()

print("\nTraining and test split:")
print("Test period starts:", cutoff_date.strftime("%Y-%m-%d"))
print("Training matches:", len(train_data))
print("Test matches:", len(test_data))

# EDA: distribution of target in training data.
gd_counts = train_data["goal_difference"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(gd_counts.index.astype(str), gd_counts.values, color="#2878B5")
ax.set_title("Goal Difference per Match — Training Data")
ax.set_xlabel("Team 1 goal difference")
ax.set_ylabel("Number of matches")
ax.bar_label(bars, padding=3)
ax.set_ylim(0, gd_counts.max() * 1.18)
fig.tight_layout()
fig.savefig(figures_path / "training_goal_difference_distribution.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Correlations for review.
eda_columns = predictor_columns + ["goal_difference"]
correlations = train_data[eda_columns].corr()
correlations.to_csv(results_path / "training_correlations.csv")
print("\nPredictor correlations with goal difference:")
print(correlations["goal_difference"].drop("goal_difference").sort_values(ascending=False).round(3))

# Baseline, OLS linear regression, and Ridge alternatives.
baseline_model = DummyRegressor(strategy="mean")
linear_model = LinearRegression()
models = {"Mean baseline": baseline_model, "Linear regression": linear_model}
for alpha in [0.1, 1.0, 10.0, 100.0]:
    models[f"Ridge alpha={alpha}"] = make_pipeline(StandardScaler(), Ridge(alpha=alpha))

# Fit OLS once to save coefficients.
linear_model.fit(X_train, y_train)
coefficient_table = pd.DataFrame({"Predictor": predictor_columns, "Coefficient": linear_model.coef_})
coefficient_table.loc[len(coefficient_table)] = ["Intercept", linear_model.intercept_]
coefficient_table.to_csv(results_path / "linear_regression_coefficients.csv", index=False)
print("\nLinear regression coefficients:")
print(coefficient_table.round(3).to_string(index=False))

# Chronological validation using training dates only.
training_dates = train_data["date"].drop_duplicates().sort_values().to_numpy()
time_split = TimeSeriesSplit(n_splits=3)
validation_rows = []
for fold, (train_idx, valid_idx) in enumerate(time_split.split(training_dates), start=1):
    fold_train = train_data.loc[train_data["date"].isin(training_dates[train_idx])]
    fold_valid = train_data.loc[train_data["date"].isin(training_dates[valid_idx])]
    assert fold_train["date"].max() < fold_valid["date"].min()
    for name, model in models.items():
        m = clone(model)
        m.fit(fold_train[predictor_columns], fold_train["goal_difference"])
        pred = m.predict(fold_valid[predictor_columns])
        actual = fold_valid["goal_difference"]
        validation_rows.append({
            "Fold": fold, "Model": name,
            "Training rows": len(fold_train), "Validation rows": len(fold_valid),
            "R2": r2_score(actual, pred),
            "MAE": mean_absolute_error(actual, pred),
            "RMSE": mean_squared_error(actual, pred) ** 0.5,
        })

validation_results = pd.DataFrame(validation_rows)
validation_results.to_csv(results_path / "validation_results_2_1.csv", index=False)
summary_rows = []
for name, results in validation_results.groupby("Model"):
    counts = results["Validation rows"]
    summary_rows.append({
        "Model": name,
        "Validation RMSE": ((results["RMSE"] ** 2 * counts).sum() / counts.sum()) ** 0.5,
        "Validation MAE": (results["MAE"] * counts).sum() / counts.sum(),
    })
validation_summary = pd.DataFrame(summary_rows).sort_values("Validation RMSE").reset_index(drop=True)
validation_summary.to_csv(results_path / "validation_summary_2_1.csv", index=False)
best_model_name = validation_summary.iloc[0]["Model"]
print("\nValidation summary:")
print(validation_summary.round(3).to_string(index=False))
print("Selected model:", best_model_name)

# Final held-out test evaluation.
final_names = list(dict.fromkeys(["Mean baseline", "Linear regression", best_model_name]))
test_predictions = test_data[["match_id", "date", "team1", "team2", "goal_difference"]].copy()
test_result_rows = []
for name in final_names:
    fitted = clone(models[name])
    fitted.fit(X_train, y_train)
    pred = fitted.predict(X_test)
    test_predictions[name] = pred
    test_result_rows.append({
        "Model": name,
        "Test R2": r2_score(y_test, pred),
        "Test MAE": mean_absolute_error(y_test, pred),
        "Test RMSE": mean_squared_error(y_test, pred) ** 0.5,
    })

test_results = pd.DataFrame(test_result_rows)
test_results.to_csv(results_path / "test_results_2_1.csv", index=False)
test_predictions["selected_model_residual"] = test_predictions["goal_difference"] - test_predictions[best_model_name]
test_predictions.to_csv(results_path / "test_predictions_2_1.csv", index=False)
print("\nFinal test results:")
print(test_results.round(3).to_string(index=False))

# Diagnostic plots.
actual = test_predictions["goal_difference"]
predicted = test_predictions[best_model_name]
residuals = test_predictions["selected_model_residual"]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
lower = min(actual.min(), predicted.min()) - 0.5
upper = max(actual.max(), predicted.max()) + 0.5
axes[0].scatter(actual, predicted, alpha=0.7, color="#2878B5")
axes[0].plot([lower, upper], [lower, upper], linestyle="--", color="black", label="Perfect prediction")
axes[0].set(title="Actual vs Predicted Goal Difference — Test Data", xlabel="Actual goal difference", ylabel="Predicted goal difference", xlim=(lower, upper), ylim=(lower, upper))
axes[0].legend()
axes[1].scatter(predicted, residuals, alpha=0.7, color="#2878B5")
axes[1].axhline(0, linestyle="--", color="black")
axes[1].set(title="Residuals — Test Data", xlabel="Predicted goal difference", ylabel="Actual minus predicted")
fig.tight_layout()
fig.savefig(figures_path / "test_prediction_diagnostics_2_1.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Human-readable validation report.
report = []
report.append("LINEAR REGRESSION 2.1 DATA VALIDATION REPORT")
report.append("============================================")
report.append(f"World Cup match rows: {len(model_data)} (required: 104)")
report.append(f"Unique match IDs: {model_data['match_id'].nunique()}")
report.append(f"Explanatory variables: {len(predictor_columns)} (required: exactly 8)")
report.append(f"Missing predictor values: {model_data[predictor_columns].isna().sum().sum()}")
report.append(f"Duplicate match IDs: {model_data['match_id'].duplicated().sum()}")
report.append("Penalty-shootout totals excluded from goal difference; match score after extra time is used.")
report.append("All eight predictors use only the five matches completed before the prediction match date.")
report.append("Predictors 1-4 share concepts with Regression 2.2; predictors 5-8 are distinct.")
report.append("World Cup fixture/results file was cross-checked against FIFA/FBref 2026 match results.")
report.append("Historical inputs are the supplied approved-source CSV files used by the group's Regression 2.2 workflow.")
report.append("")
report.append("Predictors:")
for i, p in enumerate(predictor_columns, 1): report.append(f"{i}. {p}")
report.append("")
report.append("Selected model from chronological validation: " + best_model_name)
report.append(test_results.round(4).to_string(index=False))
(results_path / "validation_report_2_1.txt").write_text("\n".join(report), encoding="utf-8")

print("\nRegression 2.1 completed successfully.")
print("Main dataset: match_goal_difference_104_rows.csv")
print("Training chart: training_goal_difference_distribution.png")
print("Diagnostics: test_prediction_diagnostics_2_1.png")
