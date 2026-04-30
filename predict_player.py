"""
Interactive CLI tool for projecting a single player's 3-point make distribution for an upcoming matchup.

Purpose:
  - Query player game history and opponent profiles from CSV
  - Build a feature row for the next game (opponent + player state)
  - Use trained models to estimate the player's expected 3PM (lambda parameter)
  - Generate Poisson distribution of exact makes (0, 1, 2, ..., max_made+)
  - Output over/under thresholds and model-based probabilities
  - Support both interactive and batch (--player --opponent) modes

Inputs:
  - Player game history CSV with opponent, game_date, player_made_3pt, and feature columns
  - Optional binary model bundle (over 1.5 three-point threshold)
  - Optional count model bundle (exact 3PM prediction)
  - Player name, opponent team abbreviation, max exact bucket

Outputs:
  - Projected mean 3PM (lambda)
  - Probability distribution by exact makes (0, 1, 2, ...)
  - Over/under probabilities for common lines (0.5, 1.5, 2.5, 3.5)
  - Model confidence probabilities when models are available

Helper functions used by rank_props.py for batch processing.
"""

from __future__ import annotations

import argparse
import difflib
import math
import re
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scoring import add_engineered_features


PROFILE_COLUMNS = [
    "opp_3pa_allowed",
    "opp_3pm_allowed",
    "opp_def_rating",
    "opp_pace",
    "opp_pace_differential",
    "opp_position_3pt_rank",
    "opp_3pa_allowed_by_position",
    "opp_team_wide_open_3pa_allowed",
    "opp_team_above_break_3pa_allowed",
    "opp_team_corner_3pa_allowed",
    "corner_3_allowed",
    "above_break_3_allowed",
    "defensive_scheme_tendency",
]

NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
PLAYER_NAME_ALIASES = {
    # Common odds-feed variants vs history feed naming.
    "moe wagner": "moritz wagner",
    "carlton carrington": "bub carrington",
    "herb jones": "herbert jones",
    "ron holland": "ronald holland ii",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate a player's three-point make distribution for a requested opponent."
    )
    parser.add_argument(
        "--player",
        help="Player name. If omitted, the script prompts for it.",
    )
    parser.add_argument(
        "--opponent",
        help="Opponent team abbreviation (for example BOS). If omitted, the script prompts for it.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/player_game_3pt_features.csv"),
        help="Historical player-game dataset.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Optional trained model bundle. If supplied, use it to price over 1.5 and anchor the distribution.",
    )
    parser.add_argument(
        "--count-model",
        type=Path,
        help="Optional trained count model bundle. If supplied, use it to predict expected made threes directly.",
    )
    parser.add_argument(
        "--max-made",
        type=int,
        default=6,
        help="Highest exact made-3 bucket to print before the final plus bucket.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Keep prompting for players and opponents until you exit.",
    )
    return parser.parse_args()


def prompt_if_missing(value: str | None, label: str) -> str:
    if value:
        return value.strip()
    return input(f"{label}: ").strip()


def prompt_query() -> tuple[str, str]:
    player_name = input("Player name (blank to exit): ").strip()
    if not player_name:
        return "", ""
    opponent = input("Opponent (team abbreviation): ").strip()
    return player_name, opponent


def require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def normalize_name(text: str) -> str:
    raw = str(text or "")
    ascii_text = unicodedata.normalize("NFKD", raw)
    ascii_text = "".join(char for char in ascii_text if not unicodedata.combining(char))
    lowered = ascii_text.casefold()
    # Remove punctuation artifacts so "C.J.", "CJ", and "C J" collapse together.
    lowered = re.sub(r"[.'`’-]", "", lowered)
    lowered = re.sub(r"[^a-z0-9 ]+", " ", lowered)
    normalized = " ".join(lowered.split())
    normalized = PLAYER_NAME_ALIASES.get(normalized, normalized)
    return normalized


def strip_name_suffix(name: str) -> str:
    tokens = name.split()
    while tokens and tokens[-1] in NAME_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def select_player_rows(frame: pd.DataFrame, player_name: str) -> pd.DataFrame:
    # 1. Normalize the input player name and create suffix-stripped version.
    query = normalize_name(player_name)
    query_no_suffix = strip_name_suffix(query)
    # 2. Normalize all player names in the frame for comparison.
    player_lookup = frame["player_name"].fillna("").map(normalize_name)
    player_lookup_no_suffix = player_lookup.map(strip_name_suffix)

    # 3. Try exact match (with suffix).
    exact = frame[player_lookup == query].copy()
    if not exact.empty:
        return exact

    # 4. Try exact match (without suffix, e.g., "John Smith" vs "John Smith Jr").
    exact_no_suffix = frame[player_lookup_no_suffix == query_no_suffix].copy()
    if not exact_no_suffix.empty:
        return exact_no_suffix

    # 5. Try substring match (with suffix).
    contains = frame[player_lookup.str.contains(query, regex=False)].copy()
    if not contains.empty:
        return contains

    # 6. Try substring match (without suffix).
    if query_no_suffix:
        contains_no_suffix = frame[player_lookup_no_suffix.str.contains(query_no_suffix, regex=False)].copy()
        if not contains_no_suffix.empty:
            return contains_no_suffix

    # 7. No direct match found; generate close suggestions using difflib.
    available_names = sorted(frame["player_name"].dropna().unique().tolist())
    suggestions = difflib.get_close_matches(player_name, available_names, n=5, cutoff=0.6)
    # 8. If no display-name suggestions, try matching on normalized names.
    if not suggestions and query:
        normalized_to_display: dict[str, str] = {}
        for display_name in available_names:
            normalized_to_display.setdefault(normalize_name(display_name), display_name)
            normalized_to_display.setdefault(
                strip_name_suffix(normalize_name(display_name)),
                display_name,
            )
        normalized_choices = [choice for choice in normalized_to_display.keys() if choice]
        normalized_matches = difflib.get_close_matches(query, normalized_choices, n=5, cutoff=0.6)
        suggestions = [normalized_to_display[match] for match in normalized_matches if match in normalized_to_display]

    # 9. Raise error with suggestions if no rows found.
    suggestion_text = f" Closest matches: {', '.join(suggestions)}." if suggestions else ""
    raise ValueError(f"No player rows found for '{player_name}'.{suggestion_text}")


def latest_player_row(player_rows: pd.DataFrame) -> pd.Series:
    ranked = player_rows.copy()
    ranked["game_date"] = pd.to_datetime(ranked["game_date"], errors="coerce")
    ranked.sort_values(["game_date", "game_id"], inplace=True)
    return ranked.iloc[-1]


def latest_opponent_profile(frame: pd.DataFrame, opponent: str) -> pd.Series:
    profile_rows = frame[frame["opponent"].fillna("").str.upper() == opponent.upper()].copy()
    if profile_rows.empty:
        raise ValueError(
            f"No rows found with opponent '{opponent}'. Use the team abbreviation stored in the dataset."
        )

    profile_rows["game_date"] = pd.to_datetime(profile_rows["game_date"], errors="coerce")
    profile_rows.sort_values(["game_date", "game_id"], inplace=True)
    latest = profile_rows.iloc[-1]
    return latest[PROFILE_COLUMNS] if set(PROFILE_COLUMNS).intersection(profile_rows.columns) else latest


def build_next_game_row(frame: pd.DataFrame, player_row: pd.Series, opponent: str) -> pd.DataFrame:
    # 1. Copy the player's latest game row and set opponent + unknown outcome.
    row = player_row.copy()
    row["opponent"] = opponent.upper()
    row["player_made_3pt"] = np.nan

    # 2. Update game_date and as_of_date to today's date.
    if "game_date" in row.index:
        row["game_date"] = pd.Timestamp.today().date().isoformat()
    if "as_of_date" in row.index:
        row["as_of_date"] = pd.Timestamp.today().date().isoformat()
    # 3. Generate a synthetic game_id for the simulated matchup.
    if "game_id" in row.index:
        row["game_id"] = f"SIM-{row.get('player_id', 'UNKNOWN')}-{opponent.upper()}"

    # 4. Set home_game to 0 (neutral) if not already set.
    if "home_game" in row.index and pd.isna(row["home_game"]):
        row["home_game"] = 0

    # 5. Get the opponent's defensive profile (latest row for that opponent).
    opponent_profile = latest_opponent_profile(frame, opponent)
    # 6. Copy opponent's defensive metrics into the player's row.
    for column in PROFILE_COLUMNS:
        if column in opponent_profile.index:
            row[column] = opponent_profile[column]

    # 7. Add engineered features (ratios, derived stats) to prepare for model prediction.
    return add_engineered_features(pd.DataFrame([row]))


def predict_over_one_half(model_path: Path | None, feature_row: pd.DataFrame) -> float | None:
    if model_path is None:
        return None

    bundle = joblib.load(model_path)
    model = bundle["model"]
    probability = float(model.predict_proba(feature_row)[:, 1][0])
    return probability


def predict_lambda_from_count_model(model_path: Path | None, feature_row: pd.DataFrame) -> float | None:
    if model_path is None:
        return None

    bundle = joblib.load(model_path)
    model = bundle["model"]
    prediction = float(model.predict(feature_row)[0])
    return max(0.0, prediction)


def probability_over_one_half_from_lambda(lam: float) -> float:
    return 1.0 - math.exp(-lam) * (1.0 + lam)


def solve_lambda_for_over_one_half(target_probability: float) -> float:
    if target_probability <= 0:
        return 0.0
    if target_probability >= 0.999:
        return 12.0

    low = 0.0
    high = 12.0
    for _ in range(60):
        mid = (low + high) / 2.0
        if probability_over_one_half_from_lambda(mid) < target_probability:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def estimate_heuristic_lambda(frame: pd.DataFrame, player_rows: pd.DataFrame, player_row: pd.Series, opponent: str) -> float:
    # 1. Extract last 5 games and season average 3PM from player's current row.
    last_5 = float(pd.to_numeric(pd.Series([player_row.get("last_5_games_3pm_avg")]), errors="coerce").iloc[0])
    season_avg = float(pd.to_numeric(pd.Series([player_row.get("season_avg_3pm")]), errors="coerce").iloc[0])

    # 2. Compute mean 3PM over last 10 games.
    recent_sample = player_rows.copy()
    recent_sample["game_date"] = pd.to_datetime(recent_sample["game_date"], errors="coerce")
    recent_sample.sort_values(["game_date", "game_id"], inplace=True)
    last_10_mean = float(
        pd.to_numeric(recent_sample["player_made_3pt"], errors="coerce").tail(10).mean()
    )

    # 3. Gather valid (non-NaN) components from the three averages.
    components = [value for value in [last_5, season_avg, last_10_mean] if not np.isnan(value)]
    if not components:
        raise ValueError("Not enough player history to estimate a projection.")

    # 4. Backfill missing components with the mean of available ones.
    if np.isnan(last_5):
        last_5 = float(np.mean(components))
    if np.isnan(season_avg):
        season_avg = float(np.mean(components))
    if np.isnan(last_10_mean):
        last_10_mean = float(np.mean(components))

    # 5. Compute base lambda: weighted blend of last 5, season, and last 10 averages.
    base_lambda = (0.5 * last_5) + (0.3 * season_avg) + (0.2 * last_10_mean)

    # 6. Get opponent defensive rating: 3PM allowed against this opponent.
    opponent_rows = frame[frame["opponent"].fillna("").str.upper() == opponent.upper()].copy()
    if "opp_3pm_allowed" in opponent_rows.columns:
        opponent_allowance = pd.to_numeric(opponent_rows["opp_3pm_allowed"], errors="coerce").dropna()
    else:
        opponent_allowance = pd.Series(dtype=float)

    # 7. Get league average 3PM allowed (baseline for opponent strength).
    if "opp_3pm_allowed" in frame.columns:
        league_allowance = pd.to_numeric(frame["opp_3pm_allowed"], errors="coerce").dropna()
    else:
        league_allowance = pd.Series(dtype=float)

    # 8. Compute opponent adjustment factor: ratio of opponent's defense to league average.
    if not opponent_allowance.empty and not league_allowance.empty and league_allowance.mean() > 0:
        opponent_factor = float(opponent_allowance.iloc[-1] / league_allowance.mean())
        # 9. Clip opponent factor to reasonable bounds (0.8 to 1.25).
        opponent_factor = float(np.clip(opponent_factor, 0.8, 1.25))
    else:
        opponent_factor = 1.0

    # 10. Return final lambda: base estimate adjusted by opponent strength.
    return max(0.0, base_lambda * opponent_factor)


def poisson_probability(lam: float, made: int) -> float:
    return math.exp(-lam) * (lam ** made) / math.factorial(made)


def exact_distribution(lam: float, max_made: int) -> list[tuple[str, float]]:
    results: list[tuple[str, float]] = []
    cumulative = 0.0

    for made in range(max_made):
        probability = poisson_probability(lam, made)
        cumulative += probability
        results.append((str(made), probability))

    plus_probability = max(0.0, 1.0 - cumulative)
    results.append((f"{max_made}+", plus_probability))
    return results


def threshold_probabilities(lam: float, max_line: int = 4) -> list[tuple[str, float, float]]:
    probabilities: list[tuple[str, float, float]] = []

    for line in range(max_line):
        under = sum(poisson_probability(lam, made) for made in range(line + 1))
        over = max(0.0, 1.0 - under)
        probabilities.append((f"{line}.5", under, over))

    return probabilities


def print_report(
    player_row: pd.Series,
    opponent: str,
    lam: float,
    model_probability: float | None,
    distribution: list[tuple[str, float]],
    thresholds: list[tuple[str, float, float]],
) -> None:
    print(f"\nPlayer: {player_row['player_name']}")
    print(f"Team: {player_row.get('team', 'UNKNOWN')}")
    print(f"Opponent: {opponent.upper()}")
    print(f"Estimated 3PM mean: {lam:.2f}")

    if model_probability is not None:
        print(f"Model probability over 1.5: {model_probability:.3f}")

    print("\nExact made threes")
    for label, probability in distribution:
        print(f"  {label}: {probability:.3%}")

    print("\nOver / under lines")
    for line, under, over in thresholds:
        print(f"  Under {line}: {under:.3%} | Over {line}: {over:.3%}")


def run_projection(
    frame: pd.DataFrame,
    player_name: str,
    opponent: str,
    model_path: Path | None,
    count_model_path: Path | None,
    max_made: int,
) -> None:
    # 1. Find all game rows for the requested player.
    player_rows = select_player_rows(frame, player_name)
    # 2. Get the player's most recent game as baseline.
    player_row = latest_player_row(player_rows)
    # 3. Build feature row for next game against specified opponent.
    feature_row = build_next_game_row(frame, player_row, opponent)

    # 4. Try count model first to get direct lambda prediction.
    count_lambda = predict_lambda_from_count_model(count_model_path, feature_row)
    # 5. Fall back to binary model: use over 1.5 probability to solve for lambda.
    model_probability = predict_over_one_half(model_path, feature_row)
    # 6. Determine lambda from available models; fall back to heuristic average.
    if count_lambda is not None:
        lam = count_lambda
    elif model_probability is not None:
        lam = solve_lambda_for_over_one_half(model_probability)
    else:
        lam = estimate_heuristic_lambda(frame, player_rows, player_row, opponent)

    # 7. Generate Poisson distribution for exact makes (0, 1, 2, ..., max_made+).
    distribution = exact_distribution(lam, max_made)
    # 8. Compute over/under probabilities for key lines.
    thresholds = threshold_probabilities(lam, max_line=min(max_made, 5))
    # 9. Print final report with all projections and probabilities.
    print_report(player_row, opponent, lam, model_probability, distribution, thresholds)


def main() -> None:
    # 1. Parse command-line arguments (player, opponent, data path, models, options).
    args = parse_args()

    # 2. Verify the input dataset exists before proceeding.
    if not args.data.exists():
        raise FileNotFoundError(
            f"Dataset not found: {args.data}. Run update_features.py first to build it."
        )

    # 3. Load player game history from CSV.
    frame = pd.read_csv(args.data)
    # 4. Validate required columns are present in the dataset.
    require_columns(frame, ["player_name", "opponent", "game_date", "player_made_3pt"])

    # 5. Determine if running in interactive or batch mode.
    interactive_mode = args.interactive or not (args.player and args.opponent)
    if interactive_mode:
        # 6. Interactive mode: loop until user exits.
        print("Interactive 3PT projection mode. Press Enter on player name to exit.")
        while True:
            # 7. Prompt for player name and opponent.
            player_name, opponent = prompt_query()
            if not player_name:
                break
            if not opponent:
                print("Opponent is required.\n")
                continue
            # 8. Run projection with error handling for invalid inputs.
            try:
                run_projection(
                    frame,
                    player_name,
                    opponent,
                    args.model,
                    args.count_model,
                    args.max_made,
                )
            except Exception as exc:
                print(f"\nError: {exc}\n")
        return

    # 9. Batch mode: run single projection with provided args and exit.
    run_projection(frame, args.player, args.opponent, args.model, args.count_model, args.max_made)


if __name__ == "__main__":
    main()
