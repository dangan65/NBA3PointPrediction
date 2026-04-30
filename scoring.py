"""
Score upcoming player over/under 3pt props using a trained model and compute edge vs sportsbook odds.

Module Purpose:
  Load a trained joblib model, read a CSV of upcoming player props, compute implied probability from odds,
  generate model predictions, calculate edge and expected value, and output scored results to CSV.

Inputs:
  - Trained model bundle (joblib file)
  - CSV of upcoming props with player info and sportsbook odds

Outputs:
  - Scored CSV with implied probabilities, model predictions, edge, expected value, and bet recommendations
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score player over 1.5 threes props and compute edge vs sportsbook odds."
    )
    parser.add_argument("model_path", type=Path, help="Saved joblib model bundle.")
    parser.add_argument("csv_path", type=Path, help="CSV of upcoming props to score.")
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=0.05,
        help="Minimum probability edge required before flagging a bet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/scored_props.csv"),
        help="Where to write the scored CSV.",
    )
    return parser.parse_args()


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()

    required_for_hot = {"last_3_games_3pm", "season_avg_3pm"}
    if required_for_hot.issubset(df.columns):
        df["hot_streak_score"] = df["last_3_games_3pm"] - df["season_avg_3pm"]

    required_for_form = {"last_5_games_3pm_avg", "last_5_games_3pa_avg"}
    if required_for_form.issubset(df.columns):
        attempts = df["last_5_games_3pa_avg"].replace(0, np.nan)
        df["three_point_form"] = df["last_5_games_3pm_avg"] / attempts

    if "back_to_back" not in df.columns and "rest_days" in df.columns:
        df["back_to_back"] = (df["rest_days"] == 0).astype(int)

    for categorical in [
        "home_game",
        "back_to_back",
        "position",
        "starter_flag",
        "team_offensive_role",
        "defensive_scheme_tendency",
        "primary_defender_pos",
    ]:
        if categorical in df.columns:
            df[categorical] = df[categorical].fillna("UNKNOWN").astype(str)

    return df


def decimal_from_american(american_odds: pd.Series) -> pd.Series:
    odds = pd.to_numeric(american_odds, errors="coerce")
    positive = odds > 0
    decimal = pd.Series(np.nan, index=odds.index, dtype=float)
    decimal.loc[positive] = 1 + (odds.loc[positive] / 100.0)
    decimal.loc[~positive] = 1 + (100.0 / odds.loc[~positive].abs())
    return decimal


def ensure_decimal_odds(df: pd.DataFrame) -> pd.Series:
    if "decimal_odds" in df.columns:
        return pd.to_numeric(df["decimal_odds"], errors="coerce")
    if "american_odds" in df.columns:
        return decimal_from_american(df["american_odds"])
    raise ValueError("Props CSV must include either 'decimal_odds' or 'american_odds'.")


def main() -> None:
    # 1. Load model and parse arguments
    args = parse_args()
    bundle = joblib.load(args.model_path)
    model = bundle["model"]

    # 2. Read props CSV and engineer features
    df = pd.read_csv(args.csv_path)
    df = add_engineered_features(df)

    # 3. Extract and convert odds to decimal format
    decimal_odds = ensure_decimal_odds(df)
    implied_prob = 1.0 / decimal_odds

    # 4. Generate model predictions and compute probability
    probs = model.predict_proba(df)[:, 1]

    # 5. Calculate edge (model prob - implied prob) and expected value
    edge = probs - implied_prob
    expected_value = (probs * (decimal_odds - 1.0)) - (1.0 - probs)

    # 6. Build scored output dataframe with all metrics
    scored = df.copy()
    scored["decimal_odds"] = decimal_odds
    scored["implied_prob"] = implied_prob
    scored["model_prob_over_1_5"] = probs
    scored["edge"] = edge
    scored["expected_value_per_unit"] = expected_value
    scored["bet_recommendation"] = np.where(
        (edge >= args.edge_threshold) & (expected_value > 0),
        "BET",
        "PASS",
    )

    # 7. Write scored results to output CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output, index=False)

    # 8. Print top edge opportunities for user review
    printable_columns = [
        col
        for col in [
            "player_name",
            "team",
            "opponent",
            "decimal_odds",
            "implied_prob",
            "model_prob_over_1_5",
            "edge",
            "expected_value_per_unit",
            "bet_recommendation",
        ]
        if col in scored.columns
    ]

    print(scored[printable_columns].sort_values("edge", ascending=False).head(15).to_string(index=False))
    print(f"\nSaved scored props to: {args.output}")


if __name__ == "__main__":
    main()
