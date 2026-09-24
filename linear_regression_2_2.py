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