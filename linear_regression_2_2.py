# Regression 2.2: Predict goals scored by one team per FIFA World Cup 2026 match
print("Linear Regression 2.2 Project")

from pathlib import Path
import pandas as pd
import sys

sys.stdout.reconfigure(encoding="utf-8")

file_path = Path(__file__).parent / "world_cup_2026_data.csv"
matches = pd.read_csv(file_path, comment="#")
print(matches.head())
print("Raw rows", len(matches))

#cleaning data
print("Empty rows:", matches.isna().all(axis=1).sum())
matches = matches.dropna(how="all").copy()

print("Match rows:", len(matches))
print(matches["Round"].value_counts())

print("\nScore formats:")
print(matches["Score"].unique())
print(matches[["Home", "Score", "Away", "Notes"]].tail(8).to_string(index=False))

goals = matches["Score"].str.extract(r"(\d+)\s*[\u2013-]\s*(\d+)")
unmatched = goals.isna().any(axis=1)
print("Unmatched scores:", unmatched.sum())
print(matches.loc[unmatched, "Score"].apply(repr).to_string())

matches["home_goals"] = goals[0].astype(int)
matches["away_goals"] = goals[1].astype(int)

print(matches[["Score", "home_goals", "away_goals"]].tail(15).to_string(index=False))

matches = matches.reset_index(drop=True)
matches["match_id"] = matches.index + 1

home_rows = matches[["match_id", "Date", "Round", "Home", "Away", "home_goals"]].copy()
home_rows.columns = ["match_id", "date", "round", "team", "opponent", "goals_scored"]

away_rows = matches[["match_id", "Date", "Round", "Away", "Home", "away_goals"]].copy()
away_rows.columns = ["match_id", "date", "round", "team", "opponent", "goals_scored"]

team_matches = pd.concat([home_rows, away_rows], ignore_index=True)
team_matches = team_matches.sort_values("match_id", kind="stable").reset_index(drop=True)
print("Team-match rows:", len(team_matches))
print(team_matches.head(6).to_string(index=False))
print("Two rows per match:", team_matches.groupby("match_id").size().eq(2).all())

for column in ["team", "opponent"]:
    team_matches[column] = team_matches[column].str.strip()
    team_matches[column] = team_matches[column].str.replace(
        r"^[a-z]{2,3}\s+|\s+[a-z]{2,3}$", "", regex=True
    )

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

output_path = Path(__file__).parent / "team_matches_208_rows.csv"
team_matches.to_csv(output_path, index=False, encoding="utf-8")
print("Saved:", output_path)

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
    "WorldCupQualifiers_Intercontinental_2026.csv"
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