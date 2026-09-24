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