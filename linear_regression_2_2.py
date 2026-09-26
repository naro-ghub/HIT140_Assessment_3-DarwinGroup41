# Regression 2.2: Predict goals scored by one team per FIFA World Cup 2026 match
print("Linear Regression 2.2 Project")

# Import tools and enable special characters in terminal output
from pathlib import Path
import pandas as pd
import sys
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")

# Import tools and enable special characters in terminal output
file_path = Path(__file__).parent / "world_cup_2026_data.csv"
matches = pd.read_csv(file_path, comment="#")
print(matches.head())
print("Raw rows", len(matches))

# Remove completely empty rows and check the number of matches in each round
print("Empty rows:", matches.isna().all(axis=1).sum())
matches = matches.dropna(how="all").copy()

print("Match rows:", len(matches))
print(matches["Round"].value_counts())

# Inspect score formats and match notes before extracting goals
print("\nScore formats:")
print(matches["Score"].unique())
print(matches[["Home", "Score", "Away", "Notes"]].tail(8).to_string(index=False))

# Extract home and away goals, excluding shootout totals, and check for unmatched scores
goals = matches["Score"].str.extract(r"(\d+)\s*[\u2013-]\s*(\d+)")
unmatched = goals.isna().any(axis=1)
print("Unmatched scores:", unmatched.sum())
print(matches.loc[unmatched, "Score"].apply(repr).to_string())

# Store the extracted goals as integers and inspect the results
matches["home_goals"] = goals[0].astype(int)
matches["away_goals"] = goals[1].astype(int)

print(matches[["Score", "home_goals", "away_goals"]].tail(15).to_string(index=False))

# Reset the row index and assign a unique ID to each match
matches = matches.reset_index(drop=True)
matches["match_id"] = matches.index + 1

# Create one dataset from the home team's perspective and another from the away team's
home_rows = matches[["match_id", "Date", "Round", "Home", "Away", "home_goals"]].copy()
home_rows.columns = ["match_id", "date", "round", "team", "opponent", "goals_scored"]

away_rows = matches[["match_id", "Date", "Round", "Away", "Home", "away_goals"]].copy()
away_rows.columns = ["match_id", "date", "round", "team", "opponent", "goals_scored"]

# Combine both datasets, group rows by match ID and check that each match has two rows
team_matches = pd.concat([home_rows, away_rows], ignore_index=True)
team_matches = team_matches.sort_values("match_id", kind="stable").reset_index(drop=True)
print("Team-match rows:", len(team_matches))
print(team_matches.head(6).to_string(index=False))
print("Two rows per match:", team_matches.groupby("match_id").size().eq(2).all())

# Remove extra spaces and country codes from team and opponent names
for column in ["team", "opponent"]:
    team_matches[column] = team_matches[column].str.strip()
    team_matches[column] = team_matches[column].str.replace(
        r"^[a-z]{2,3}\s+|\s+[a-z]{2,3}$", "", regex=True
    )

# Preview the cleaned names and count the unique teams
print("\nCleaned team names:")
print(team_matches.head(6).to_string(index=False))
print("Unique teams:", team_matches["team"].nunique())

# Predictor 1: Identify group-stage and knockout-stage matches
team_matches["is_knockout"] = (
    team_matches["round"] != "Group stage"
).astype(int)

print("\nPredictor 1: Knockout stage (0 = group stage, 1 = knockout)")
print(team_matches["is_knockout"].value_counts())

# Validate dates, row counts, duplicates and missing values
team_matches["date"] = pd.to_datetime(team_matches["date"], errors="raise")

assert len(team_matches) == 208, "Expected exactly 208 team-match rows."
assert team_matches["match_id"].nunique() == 104, "Expected 104 matches."
assert team_matches.groupby("match_id").size().eq(2).all(), "Each match needs two rows."

assert not team_matches.duplicated(["match_id", "team"]).any(), "Duplicate team in a match."
assert not team_matches.isna().any().any(), "Missing values found in the base dataset."

print("\nBase dataset checks passed.")



# Check pre-match predictor calculations using Mexico as an example
# Load Mexico's historical results
history_path = Path(__file__).parent / "Mexico_FIFA2026stats.csv"
history = pd.read_csv(history_path, comment="#")
history["Date"] = pd.to_datetime(history["Date"], errors="raise")

# Calculate goal-based predictors from the five preceding matches
def calculate_recent_form(team_history, match_date):
    previous = team_history.loc[
        (team_history["Date"] < match_date)
        & team_history["GF"].notna()
        & team_history["GA"].notna()
    ].sort_values("Date").tail(5)

    assert len(previous) == 5, "Five previous matches are required."

    return {
        "team_scored_last5": previous["GF"].mean(),
        "team_conceded_last5": previous["GA"].mean(),
        "team_scoring_std_last5": previous["GF"].std(ddof=1)
    }

# Check the predictors for each of Mexico's World Cup matches
world_cup_dates = history.loc[
    history["Comp"] == "World Cup", "Date"
].sort_values()

for match_date in world_cup_dates:
    form = calculate_recent_form(history, match_date)

    print("\nMexico — match date:", match_date.strftime("%Y-%m-%d"))
    print(
        "Predictor 4: Average goals scored:",
        round(form["team_scored_last5"], 2)
    )
    print(
        "Predictor 5: Average goals conceded:",
        round(form["team_conceded_last5"], 2)
    )
    print(
        "Predictor 8: Scoring consistency (SD):",
        round(form["team_scoring_std_last5"], 2)
    )
    
# Load historical datasets used to calculate pre-match predictors
base_path = Path(__file__).parent

history_files = [
    "International_friendlies_2025.csv",
    "International_friendlies_2026.csv",
    "2026_WorldCupQualifiers-UEFA(M).csv",
    "2026_WorldCupQualifiers_AFC(M).csv",
    "2026_WorldCupQualifier_CAF(M).csv",
    "2026_WorldCupQualifiers_CONCACAF.csv",
    "2025_Africa_Cup_of_Nations.csv",
    "WorldCupQualifiers_Intercontinental_2026.csv",
    "additional_historical_matches.csv"
]

history_tables = []

for filename in history_files:
    table = pd.read_csv(base_path / filename, comment="#")
    table["source_file"] = filename
    history_tables.append(table)
    print(filename, ":", len(table), "rows")

# Combine the tables and clean historical data
all_history = pd.concat(history_tables, ignore_index=True)

print("\nCombined raw historical rows:", len(all_history))

before_cleaning = len(all_history)

all_history = all_history.dropna(
    subset=["Date", "Home", "Away"], how="all"
).copy()

print("\nSeparator rows removed:", before_cleaning - len(all_history))

# Remove fixtures without scores, including cancelled and unplayed games and convert dates to select matches played before each World Cup game
missing_scores = all_history["Score"].isna()
print("Fixtures without scores removed:", missing_scores.sum())

all_history = all_history.loc[~missing_scores].copy()

all_history["Date"] = pd.to_datetime(
    all_history["Date"], errors="raise"
)

print("Historical rows with scores:", len(all_history))

# Identify results that may differ from goals actually scored
awarded = all_history["Notes"].str.contains(
    "awarded", case=False, na=False
)

print("\nAwarded results requiring review:", awarded.sum())

print(
    all_history.loc[
        awarded, ["Date", "Home", "Score", "Away", "Notes"]
    ].to_string(index=False)
)

# Exclude awarded scores from goal-based predictor calculations and keeping the original records unchanged in the raw CSV files
all_history = all_history.loc[~awarded].copy()

print("Historical rows after excluding awarded results:", len(all_history))

# Clean historical team names so they match the World Cup dataset
for column in ["Home", "Away"]:
    all_history[column] = all_history[column].str.strip()
    all_history[column] = all_history[column].str.replace(
        r"^[a-z]{2,3}\s+|\s+[a-z]{2,3}$", "", regex=True
    )
    all_history[column] = all_history[column].replace(
        {"United States": "USA"}
    )

# Extract match goals, excluding penalty shootout numbers
history_goals = all_history["Score"].str.extract(
    r"(\d+)\s*[\u2013-]\s*(\d+)"
)

# Stop if any score could not be understood
assert not history_goals.isna().any().any(), "Unrecognised historical score."

all_history["home_goals"] = history_goals[0].astype(int)
all_history["away_goals"] = history_goals[1].astype(int)

print("\nCleaned historical scores:")
print(
    all_history[
        ["Date", "Home", "Away", "Score", "home_goals", "away_goals"]
    ].head(6).to_string(index=False)
)

# Reshape historical matches into one row per team per match
history_home = all_history[
    ["Date", "Home", "Away", "home_goals", "away_goals", "source_file"]
].copy()

history_home.columns = [
    "date", "team", "opponent", "goals_scored", "goals_conceded", "source_file"
]

history_away = all_history[
    ["Date", "Away", "Home", "away_goals", "home_goals", "source_file"]
].copy()

history_away.columns = [
    "date", "team", "opponent", "goals_scored", "goals_conceded", "source_file"
]

# Combine both teams perspectives and put them according to the dates
historical_team_matches = pd.concat(
    [history_home, history_away], ignore_index=True
)

historical_team_matches = historical_team_matches.sort_values(
    ["team", "date"]
).reset_index(drop=True)

print("\nHistorical team-match rows:", len(historical_team_matches))
print(historical_team_matches.head(6).to_string(index=False))

# Find each team opening World Cup match date
opening_dates = (
    team_matches.groupby("team")["date"]
    .min()
    .rename("opening_date")
    .reset_index()
)

# Keep historical records for World Cup teams and attach their opening dates
opening_history = historical_team_matches.merge(
    opening_dates, on="team", how="inner", validate="many_to_one"
)

# Only use results from before the team's opening match
opening_history = opening_history.loc[
    opening_history["date"] < opening_history["opening_date"]
].copy()

# Select the five latest results per team for review
opening_last5 = (
    opening_history.sort_values(["team", "date"])
    .groupby("team")
    .tail(5)
)

print("\nOpening-match history review:")
print("Teams:", opening_last5["team"].nunique())
print("Rows:", len(opening_last5))

print(
    opening_last5.loc[
        opening_last5["team"] == "Mexico",
        ["date", "team", "opponent", "goals_scored", "goals_conceded"]
    ].to_string(index=False)
)

# Look up the opponent goals to obtain each team's goals conceded
opponent_results = team_matches[
    ["match_id", "team", "goals_scored"]
].rename(columns={
    "team": "opponent",
    "goals_scored": "goals_conceded"
})

world_cup_history = team_matches.merge(
    opponent_results,
    on=["match_id", "opponent"],
    validate="one_to_one"
)

# Give World Cup history the same columns as the other historical records
world_cup_history["source_file"] = "world_cup_2026_data.csv"

history_columns = [
    "date", "team", "opponent",
    "goals_scored", "goals_conceded", "source_file"
]

# Combine historical internationals and World Cup results
full_history = pd.concat(
    [
        historical_team_matches[history_columns],
        world_cup_history[history_columns]
    ],
    ignore_index=True
).sort_values(["team", "date"]).reset_index(drop=True)

# Check that no team-match appears twice
assert not full_history.duplicated(
    ["date", "team", "opponent"]
).any(), "Duplicate historical team-match found."

print("\nCombined team-match history:", len(full_history))

# Calculate recent form using only the five matches before the prediction date
def get_recent_form(team_name, match_date):
    previous = full_history.loc[
        (full_history["team"] == team_name)
        & (full_history["date"] < match_date)
    ].sort_values("date").tail(5)

    assert len(previous) == 5, f"Five previous matches needed for {team_name}."

    # Draws decided by penalty shootouts remain draws for this calculation.
    wins = previous["goals_scored"] > previous["goals_conceded"]

    # A clean sheet means conceding zero goals.
    clean_sheets = previous["goals_conceded"] == 0

    return {
        "scored": previous["goals_scored"].mean(),
        "conceded": previous["goals_conceded"].mean(),
        "scoring_std": previous["goals_scored"].std(ddof=1),
        "win_rate": wins.mean(),
        "clean_sheet_rate": clean_sheets.mean()
    }

# Check that Mexico's first World Cup result contributes to its second match
mexico_form = get_recent_form("Mexico", pd.Timestamp("2026-06-18"))

print("\nMexico: predictors before 18 June 2026")
print("Predictor 4: Average goals scored:", round(mexico_form["scored"], 2))
print("Predictor 5: Average goals conceded:", round(mexico_form["conceded"], 2))
print("Predictor 8: Scoring consistency (SD):", round(mexico_form["scoring_std"], 2))

# Calculate predictors 2–8 for every World Cup team-match row
form_rows = []

for _, match in team_matches.iterrows():
    team_form = get_recent_form(match["team"], match["date"])
    opponent_form = get_recent_form(match["opponent"], match["date"])

    form_rows.append({
        # Predictor 2: Team's win rate over its previous five matches
        "team_win_rate_last5": team_form["win_rate"],

        # Predictor 3: Opponent's clean-sheet rate over its previous five matches
        "opponent_clean_sheet_rate_last5": opponent_form["clean_sheet_rate"],

        # Predictors 4–7: Average goals scored and conceded
        "team_scored_last5": team_form["scored"],
        "team_conceded_last5": team_form["conceded"],
        "opponent_scored_last5": opponent_form["scored"],
        "opponent_conceded_last5": opponent_form["conceded"],

        # Predictor 8: Standard deviation of the team's goals scored
        "team_scoring_std_last5": team_form["scoring_std"]
    })

# Attach the calculated predictors to the corresponding rows
form_table = pd.DataFrame(form_rows, index=team_matches.index)
team_matches[form_table.columns] = form_table

assert not form_table.isna().any().any(), "Missing recent-form values."

print("\nRecent-form rows calculated:", len(form_table))
print(form_table.head(6).round(2).to_string(index=False))

# Review the selected historical matches before each team's World Cup opener
teams_to_review = [
    "Australia", "Cabo Verde", "Curaçao", "New Zealand",
    "Uzbekistan", "Qatar", "Jordan", "Iraq", "IR Iran"
]

for team_name in teams_to_review:
    selected_matches = opening_last5.loc[
        opening_last5["team"] == team_name,
        ["date", "opponent", "goals_scored", "goals_conceded"]
    ].sort_values("date")

    print("\nOpening-match history:", team_name)
    print(selected_matches.to_string(index=False))

# Select exactly eight explanatory variables for modelling
predictor_columns = [
    "is_knockout",
    "team_win_rate_last5",
    "opponent_clean_sheet_rate_last5",
    "team_scored_last5",
    "team_conceded_last5",
    "opponent_scored_last5",
    "opponent_conceded_last5",
    "team_scoring_std_last5"
]

# Check that all 208 rows have all eight predictors
assert team_matches[predictor_columns].shape == (208, 8)
assert not team_matches[predictor_columns].isna().any().any()

# Both rates must be proportions between zero and one
assert team_matches["team_win_rate_last5"].between(0, 1).all()
assert team_matches["opponent_clean_sheet_rate_last5"].between(0, 1).all()

# Print a numbered label for each predictor
print("\nCompleted explanatory variables:")
for number, column in enumerate(predictor_columns, start=1):
    print(f"Predictor {number}: {column}")

print("\nAll eight predictors populated for 208 rows.")

output_path = Path(__file__).parent / "team_matches_208_rows.csv"
team_matches.to_csv(output_path, index=False, encoding="utf-8")
print("Saved:", output_path)

# training/test split code
# Arrange matches by date and choose an approximate 80% training boundary
match_dates = (
    team_matches[["match_id", "date"]]
    .drop_duplicates()
    .sort_values(["date", "match_id"])
    .reset_index(drop=True)
)

split_position = int(len(match_dates) * 0.80)
cutoff_date = match_dates.iloc[split_position]["date"]

# Split earlier and later matches, keeping matches on the same date together
train_data = team_matches.loc[
    team_matches["date"] < cutoff_date
].copy()

test_data = team_matches.loc[
    team_matches["date"] >= cutoff_date
].copy()

# Select the eight predictors (X) and actual goals scored (y)
X_train = train_data[predictor_columns]
y_train = train_data["goals_scored"]

X_test = test_data[predictor_columns]
y_test = test_data["goals_scored"]

# Check that all rows are retained and no match appears in both sets
assert len(train_data) + len(test_data) == 208
assert train_data["date"].max() < test_data["date"].min()
assert not train_data["match_id"].isin(test_data["match_id"]).any()

# Display the training and test dataset sizes
print("\nTraining and test split:")
print("Test period starts:", cutoff_date.strftime("%Y-%m-%d"))
print("Training matches:", train_data["match_id"].nunique())
print("Test matches:", test_data["match_id"].nunique())
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# Summarise the predictors and goals scored using only the training data
eda_columns = predictor_columns + ["goals_scored"]

print("\nTraining data summary:")
print(train_data[eda_columns].describe().round(2).to_string())

# Count how often teams scored each number of goals in the training data
print("\nGoals scored frequency:")
print(train_data["goals_scored"].value_counts().sort_index())

# Plot how often each goal total occurs in the training data
goal_counts = (
    train_data["goals_scored"]
    .value_counts()
    .sort_index()
)

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    goal_counts.index,
    goal_counts.values,
    color="#2878B5"
)

ax.set_title("Goals Scored per Team-Match — Training Data")
ax.set_xlabel("Goals scored")
ax.set_ylabel("Number of team-match rows")
ax.set_xticks(goal_counts.index)
ax.bar_label(bars, padding=3)
ax.set_ylim(0, goal_counts.max() * 1.15)

# Save the chart for the report and display it
fig.tight_layout()
fig.savefig(
    base_path / "training_goals_distribution.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# Examine linear relationships between the predictors and goals scored
correlations = train_data[eda_columns].corr()

print("\nPredictor correlations with goals scored:")
print(
    correlations["goals_scored"]
    .drop("goals_scored")
    .sort_values(ascending=False)
    .round(3)
)

# Check correlations between predictors for potentially overlapping information
predictor_correlations = train_data[predictor_columns].corr()

print("\nStrong predictor correlations (absolute correlation >= 0.70):")
strong_pairs = []

for i, first in enumerate(predictor_columns):
    for second in predictor_columns[i + 1:]:
        correlation = predictor_correlations.loc[first, second]

        if abs(correlation) >= 0.70:
            strong_pairs.append({
                "Predictor 1": first,
                "Predictor 2": second,
                "Correlation": round(correlation, 3)
            })

if strong_pairs:
    print(pd.DataFrame(strong_pairs).to_string(index=False))
else:
    print("No predictor pairs reach this threshold.")