"""
Module: rank_props

Purpose:
  Ranks NBA player 3-point props by model edge versus market price. Loads player game
  history and current sportsbook odds, runs ML models (binary classifier + count model),
  computes Poisson probabilities, calculates edge vs sportsbook lines, and produces
  ranked outputs sorted by expected value.

Reads:
  - data/player_game_3pt_features.csv (historical player-game feature dataset)
  - data/player_threes_props.csv (player threes props from The Odds API)
  - models/three_pt_model.joblib (binary over-1.5 classifier)
  - models/three_pt_count_model.joblib (direct count prediction model)

Writes:
  - data/ranked_value_3pt_props.csv (ranked picks by edge, with tracking columns)
  - data/most_confident_3pt_props.csv (confidence board: main-line O/U picks only)
  - data/historical_recommendations_3pt.csv (cumulative archive of daily recommendations)
  - data/tracked_bets_3pt.csv (tracked bet log with result fields)
  - data/todays_playable_3pt_props.csv (shortlist of playable picks for today)
  - data/historical_confidence_3pt.csv (cumulative archive of daily top-10 confidence picks)
  - data/tracked_confidence_3pt.csv (tracked confidence picks with result fields)
  - data/confidence_top3_parlays.csv (daily top-3 confidence parlay tracking)
  - data/confidence_top5_parlays.csv (daily top-5 confidence parlay tracking)
  - data/player_predictability_summary.csv (holdout-based predictability metrics)
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import argparse
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

from predict_player import (
    estimate_heuristic_lambda,
    latest_player_row,
    poisson_probability,
    predict_lambda_from_count_model,
    predict_over_one_half,
    select_player_rows,
    solve_lambda_for_over_one_half,
)
from predictability import (
    DEFAULT_PREDICTABILITY_OUTPUT,
    annotate_predictability,
    build_player_predictability_table,
)
from scoring import add_engineered_features
from tracking import (
    DEFAULT_DAILY_TRACK_COUNT,
    TRACKING_START_DATE_STR,
    add_tracking_columns,
    append_archive,
    append_parlay_log,
    append_tracking_log,
    build_top5_parlay_rows,
    build_top3_parlay_rows,
    daily_top_rows,
)


# ---------------------------------------------------------------------------
# Constants and Configuration
# ---------------------------------------------------------------------------

TEAM_NAME_TO_ABBR = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "LA Clippers": "LAC",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}

TEAM_ABBR_TO_NAME = {abbr: name for name, abbr in TEAM_NAME_TO_ABBR.items()}
# Production defaults used by daily ranking unless explicitly overridden by CLI flags.
DEFAULT_BINARY_MODEL_PATH = Path("models/three_pt_model.joblib")
DEFAULT_COUNT_MODEL_PATH = Path("models/three_pt_count_model.joblib")
DEFAULT_TIMEZONE = "America/New_York"
DEFENSE_SNAPSHOT_COLUMNS = [
    "opp_3pa_allowed",
    "opp_3pm_allowed",
    "opp_def_rating",
    "opp_pace",
    "opp_pace_differential",
    "defensive_scheme_tendency",
]
DEFAULT_MAIN_LINE_MAX_DISTANCE = 0.12
DEFAULT_CONFIDENCE_MIN_PROB = 0.65
DEFAULT_CONFIDENCE_TARGET_COUNT = 25
DEFAULT_CONFIDENCE_HISTORY_COUNT = 10
DEFAULT_MAX_NEGATIVE_AMERICAN_ODDS = -225.0
DEFAULT_CONFIDENCE_BALANCE_TOP_N = 10
DEFAULT_CONFIDENCE_MIN_PER_SIDE = 2
DEFAULT_CONFIDENCE_BALANCE_MAX_PROB_DROP = 0.08
CONFIDENCE_SORT_COLUMNS = ["model_prob", "edge", "expected_value_per_unit", "main_line_distance"]
CONFIDENCE_SORT_ASCENDING = [False, False, False, True]


# ---------------------------------------------------------------------------
# CLI / Argument Parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank NBA player 3-point props by model edge versus market price."
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("data/player_game_3pt_features.csv"),
        help="Historical player-game feature dataset.",
    )
    parser.add_argument(
        "--props",
        type=Path,
        default=Path("data/player_threes_props.csv"),
        help="Player threes props fetched from The Odds API.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Optional binary over-1.5 model bundle.",
    )
    parser.add_argument(
        "--count-model",
        type=Path,
        help="Optional direct count model bundle.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/ranked_value_3pt_props.csv"),
        help="Where to write ranked value bets.",
    )
    parser.add_argument(
        "--confidence-output",
        type=Path,
        default=Path("data/most_confident_3pt_props.csv"),
        help="Where to write the most-confident board.",
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=0.03,
        help="Minimum edge required to include a side.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of rows to print.",
    )
    parser.add_argument(
        "--history-output",
        type=Path,
        default=Path("data/historical_recommendations_3pt.csv"),
        help="Cumulative file of tracked recommendations for performance review.",
    )
    parser.add_argument(
        "--playable-output",
        type=Path,
        default=Path("data/todays_playable_3pt_props.csv"),
        help="Optional shortlist after applying the tracking threshold.",
    )
    parser.add_argument(
        "--predictability-output",
        type=Path,
        default=DEFAULT_PREDICTABILITY_OUTPUT,
        help="Where to write the holdout-based player predictability summary.",
    )
    parser.add_argument(
        "--confidence-history-output",
        type=Path,
        default=Path("data/historical_confidence_3pt.csv"),
        help="Cumulative archive of the daily top 10 confidence-board rows.",
    )
    parser.add_argument(
        "--confidence-tracking-output",
        type=Path,
        default=Path("data/tracked_confidence_3pt.csv"),
        help="Tracked confidence-board rows with result fields.",
    )
    parser.add_argument(
        "--parlay-output",
        type=Path,
        default=Path("data/confidence_top3_parlays.csv"),
        help="Daily top-3 confidence parlay tracking file.",
    )
    parser.add_argument(
        "--parlay-top5-output",
        type=Path,
        default=Path("data/confidence_top5_parlays.csv"),
        help="Daily top-5 confidence parlay tracking file.",
    )
    parser.add_argument(
        "--daily-cap",
        type=int,
        default=DEFAULT_DAILY_TRACK_COUNT,
        help="Number of top value-board rows to archive each day. Use 0 to keep all rows.",
    )
    parser.add_argument(
        "--slate-date",
        help="Only include props whose local commence date matches YYYY-MM-DD.",
    )
    parser.add_argument(
        "--timezone",
        default=DEFAULT_TIMEZONE,
        help="Timezone used to map commence_time into a local slate date.",
    )
    parser.add_argument(
        "--confidence-main-line-max-distance",
        type=float,
        default=DEFAULT_MAIN_LINE_MAX_DISTANCE,
        help=(
            "Max distance from a 50/50 market line used for the confidence board. "
            "Lower values are stricter and avoid alt lines."
        ),
    )
    parser.add_argument(
        "--confidence-min-prob",
        type=float,
        default=DEFAULT_CONFIDENCE_MIN_PROB,
        help="Minimum model probability required on the confidence board.",
    )
    parser.add_argument(
        "--max-negative-american-odds",
        type=float,
        default=DEFAULT_MAX_NEGATIVE_AMERICAN_ODDS,
        help=(
            "Exclude picks priced worse than this American odds value. "
            "For example, -225 drops -230/-250 picks."
        ),
    )
    parser.add_argument(
        "--confidence-balance-top-n",
        type=int,
        default=DEFAULT_CONFIDENCE_BALANCE_TOP_N,
        help=(
            "Apply side-balance to the first N confidence rows so daily boards do not collapse "
            "to one side."
        ),
    )
    parser.add_argument(
        "--confidence-min-per-side",
        type=int,
        default=DEFAULT_CONFIDENCE_MIN_PER_SIDE,
        help="Minimum OVER and UNDER picks each required inside the confidence balance window.",
    )
    parser.add_argument(
        "--confidence-balance-max-prob-drop",
        type=float,
        default=DEFAULT_CONFIDENCE_BALANCE_MAX_PROB_DROP,
        help=(
            "Maximum model probability drop allowed when inserting minority-side picks during "
            "confidence balancing."
        ),
    )
    parser.add_argument(
        "--strict-player-lookup",
        action="store_true",
        help="Fail fast when a prop player cannot be matched to history rows.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helper Functions: Team Name Normalization
# ---------------------------------------------------------------------------

def normalize_team_to_abbr(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    upper = text.upper()
    if upper in TEAM_ABBR_TO_NAME:
        return upper
    return TEAM_NAME_TO_ABBR.get(text, upper)


def display_team_name(value: str) -> str:
    return TEAM_ABBR_TO_NAME.get(value, value)


def infer_player_team_and_opponent(player_row: pd.Series, prop_row: pd.Series) -> tuple[str, str]:
    event_team_a = normalize_team_to_abbr(prop_row.get("event_home_team", prop_row.get("team")))
    event_team_b = normalize_team_to_abbr(prop_row.get("event_away_team", prop_row.get("opponent")))
    player_team = normalize_team_to_abbr(player_row.get("team"))

    if player_team and event_team_a and event_team_b:
        if player_team == event_team_a:
            return player_team, event_team_b
        if player_team == event_team_b:
            return player_team, event_team_a
        # If player history is stale (trade/injury gap), still keep player team authoritative.
        return player_team, event_team_b

    if player_team:
        if event_team_a and event_team_a != player_team:
            return player_team, event_team_a
        if event_team_b and event_team_b != player_team:
            return player_team, event_team_b

    if event_team_a and event_team_b:
        return event_team_a, event_team_b
    return player_team, event_team_b or event_team_a


# ---------------------------------------------------------------------------
# Helper Functions: Probability Calculations
# ---------------------------------------------------------------------------

def probability_over_line(lam: float, line: float) -> float:
    threshold = int(math.floor(line))
    under_or_push = sum(poisson_probability(lam, made) for made in range(threshold + 1))
    return max(0.0, 1.0 - under_or_push)


def probability_under_line(lam: float, line: float) -> float:
    threshold = int(math.floor(line))
    return min(1.0, sum(poisson_probability(lam, made) for made in range(threshold + 1)))


# ---------------------------------------------------------------------------
# Helper Functions: Model Loading and Utilities
# ---------------------------------------------------------------------------

def resolve_model_path(path: Path | None, default_path: Path) -> Path | None:
    if path is not None:
        return path
    return default_path if default_path.exists() else None


def model_label(path: Path | None) -> str:
    if path is None:
        return "disabled"
    text = str(path)
    if "last30" in path.name:
        return f"{text} (experimental)"
    return f"{text} (production)"


def local_slate_date(commence_time: object, timezone_name: str) -> str:
    if pd.isna(commence_time):
        return ""
    timestamp = pd.to_datetime(commence_time, errors="coerce", utc=True)
    if pd.isna(timestamp):
        return ""
    return timestamp.tz_convert(ZoneInfo(timezone_name)).date().isoformat()


def projection_date_from_prop(prop_row: pd.Series) -> pd.Timestamp:
    timestamp = pd.to_datetime(prop_row.get("commence_time"), errors="coerce", utc=True)
    if pd.isna(timestamp):
        return pd.Timestamp.today().normalize()
    return timestamp.tz_convert("UTC").normalize()


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def build_feature_row(
    history: pd.DataFrame,
    prop_row: pd.Series,
    player_row: pd.Series,
    actual_team: str,
    actual_opponent: str,
) -> pd.DataFrame:
    row = player_row.copy()
    projection_date = projection_date_from_prop(prop_row)
    row["opponent"] = actual_opponent
    row["team"] = actual_team
    row["game_id"] = f"ODDS-{prop_row.get('odds_event_id', 'UNKNOWN')}-{player_row.get('player_id', 'UNKNOWN')}"
    row["game_date"] = projection_date.date().isoformat()
    row["as_of_date"] = projection_date.date().isoformat()
    row["player_made_3pt"] = np.nan
    for column in [
        "spread",
        "game_total",
        "injury_usage_bump",
        "teammate_minutes_vacated",
        "teammate_usage_vacated",
        "teammate_on_off_3pa_delta",
    ]:
        if column in row.index:
            row[column] = np.nan

    last_game_date = pd.to_datetime(player_row.get("game_date"), errors="coerce")
    if not pd.isna(last_game_date):
        days_since_last = int((projection_date.date() - last_game_date.date()).days)
        rest_days = max(days_since_last - 1, 0)
        row["rest_days"] = float(rest_days)
        row["back_to_back"] = int(rest_days == 0)

    event_home = normalize_team_to_abbr(prop_row.get("event_home_team", prop_row.get("team")))
    if event_home:
        row["home_game"] = int(normalize_team_to_abbr(actual_team) == event_home)

    latest_opp = history[
        history["opponent"].fillna("").astype(str).str.upper() == actual_opponent.upper()
    ].copy()
    if not latest_opp.empty:
        latest_opp["game_date"] = pd.to_datetime(latest_opp["game_date"], errors="coerce")
        latest_opp.sort_values(["game_date", "game_id"], inplace=True)
        latest_profile = latest_opp.iloc[-1]
        for column in [
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
        ]:
            if column in latest_profile.index:
                row[column] = latest_profile[column]

    if "spread" in prop_row.index:
        row["spread"] = prop_row["spread"]
    if "game_total" in prop_row.index:
        row["game_total"] = prop_row["game_total"]

    return add_engineered_features(pd.DataFrame([row]))


# ---------------------------------------------------------------------------
# Model Scoring and Lambda Determination
# ---------------------------------------------------------------------------

def determine_lambda(
    history: pd.DataFrame,
    player_rows: pd.DataFrame,
    player_row: pd.Series,
    feature_row: pd.DataFrame,
    opponent: str,
    model_path: Path | None,
    count_model_path: Path | None,
) -> tuple[float, float | None]:
    count_lambda = predict_lambda_from_count_model(count_model_path, feature_row)
    binary_prob = predict_over_one_half(model_path, feature_row)

    if count_lambda is not None:
        return count_lambda, binary_prob
    if binary_prob is not None:
        return solve_lambda_for_over_one_half(binary_prob), binary_prob
    return estimate_heuristic_lambda(history, player_rows, player_row, opponent), None


# ---------------------------------------------------------------------------
# Scoring and Edge Calculation
# ---------------------------------------------------------------------------

def best_side_rows(prop_row: pd.Series, lam: float, model_prob_over_1_5: float | None) -> list[dict]:
    line = float(prop_row["line"])
    over_model_prob = probability_over_line(lam, line)
    under_model_prob = probability_under_line(lam, line)
    output: list[dict] = []

    sides = [
        (
            "OVER",
            prop_row.get("over_american_odds"),
            prop_row.get("over_decimal_odds"),
            prop_row.get("over_implied_prob"),
            prop_row.get("no_vig_over_prob"),
            over_model_prob,
        ),
        (
            "UNDER",
            prop_row.get("under_american_odds"),
            prop_row.get("under_decimal_odds"),
            prop_row.get("under_implied_prob"),
            prop_row.get("no_vig_under_prob"),
            under_model_prob,
        ),
    ]

    for side, american_odds, decimal_odds, implied_prob, no_vig_prob, model_prob in sides:
        if pd.isna(decimal_odds) or pd.isna(implied_prob):
            continue
        decimal_value = float(decimal_odds)
        implied_value = float(implied_prob)
        edge = model_prob - implied_value
        expected_value = (model_prob * (decimal_value - 1.0)) - (1.0 - model_prob)
        output.append(
            {
                "side": side,
                "line": line,
                "american_odds": None if pd.isna(american_odds) else float(american_odds),
                "decimal_odds": decimal_value,
                "implied_prob": implied_value,
                "no_vig_prob": None if no_vig_prob is None else float(no_vig_prob),
                "model_prob": model_prob,
                "edge": edge,
                "expected_value_per_unit": expected_value,
                "model_prob_over_1_5_anchor": model_prob_over_1_5,
                "estimated_3pm_mean": lam,
            }
        )

    return output


def defense_snapshot(feature_row: pd.DataFrame) -> dict[str, object]:
    if feature_row.empty:
        return {}
    row = feature_row.iloc[0]
    snapshot: dict[str, object] = {}
    for column in DEFENSE_SNAPSHOT_COLUMNS:
        if column in row.index:
            snapshot[column] = row.get(column)
    return snapshot


# ---------------------------------------------------------------------------
# Ranking and Filtering
# ---------------------------------------------------------------------------

def sort_confidence_rows(rows: pd.DataFrame) -> pd.DataFrame:
    ordered = rows.copy()
    ordered.sort_values(
        CONFIDENCE_SORT_COLUMNS,
        ascending=CONFIDENCE_SORT_ASCENDING,
        inplace=True,
    )
    return ordered


def enforce_max_negative_american_odds(
    rows: pd.DataFrame,
    max_negative_american_odds: float,
) -> tuple[pd.DataFrame, int]:
    if rows.empty or "american_odds" not in rows.columns:
        return rows, 0
    odds = pd.to_numeric(rows["american_odds"], errors="coerce")
    too_juiced = odds.notna() & (odds < float(max_negative_american_odds))
    if not too_juiced.any():
        return rows, 0
    filtered = rows.loc[~too_juiced].copy()
    return filtered, int(too_juiced.sum())


def rebalance_confidence_top_window(
    ranked: pd.DataFrame,
    top_n: int,
    min_per_side: int,
    max_prob_drop: float,
) -> pd.DataFrame:
    if ranked.empty or top_n <= 0 or min_per_side <= 0 or max_prob_drop < 0:
        return ranked
    if "side" not in ranked.columns or "model_prob" not in ranked.columns:
        return ranked

    window_size = min(int(top_n), len(ranked))
    if window_size <= 0:
        return ranked

    top = ranked.head(window_size).copy()
    remainder = ranked.iloc[window_size:].copy()

    for target_side in ("OVER", "UNDER"):
        top_sides = top["side"].fillna("").astype(str).str.upper()
        have = int((top_sides == target_side).sum())
        need = int(min_per_side) - have
        if need <= 0:
            continue
        if remainder.empty:
            continue

        candidate_mask = remainder["side"].fillna("").astype(str).str.upper() == target_side
        candidates = remainder.loc[candidate_mask].copy()
        if candidates.empty:
            continue

        top_probs = pd.to_numeric(top["model_prob"], errors="coerce")
        if top_probs.notna().any():
            min_allowed = float(top_probs.min()) - float(max_prob_drop)
            candidate_probs = pd.to_numeric(candidates["model_prob"], errors="coerce")
            candidates = candidates.loc[candidate_probs >= min_allowed].copy()
        if candidates.empty:
            continue

        candidates = sort_confidence_rows(candidates)
        selected = candidates.head(need).copy()
        if selected.empty:
            continue

        donor_pool = top.loc[top_sides != target_side].copy()
        donor_pool = sort_confidence_rows(donor_pool)
        to_remove = donor_pool.tail(len(selected)).copy()
        if to_remove.empty:
            continue

        top = top.drop(index=to_remove.index, errors="ignore")
        top = pd.concat([top, selected], ignore_index=False, sort=False)
        top = sort_confidence_rows(top)

        remainder = remainder.drop(index=selected.index, errors="ignore")
        remainder = pd.concat([to_remove, remainder], ignore_index=False, sort=False)
        remainder = sort_confidence_rows(remainder)

    top = sort_confidence_rows(top)
    remainder = sort_confidence_rows(remainder)
    return pd.concat([top, remainder], ignore_index=True, sort=False)


def build_confidence_board(
    ranked: pd.DataFrame,
    max_main_line_distance: float,
    min_confidence_prob: float,
    target_count: int = DEFAULT_CONFIDENCE_TARGET_COUNT,
    balance_top_n: int = DEFAULT_CONFIDENCE_BALANCE_TOP_N,
    min_per_side: int = DEFAULT_CONFIDENCE_MIN_PER_SIDE,
    balance_max_prob_drop: float = DEFAULT_CONFIDENCE_BALANCE_MAX_PROB_DROP,
) -> pd.DataFrame:
    if ranked.empty:
        return ranked.copy()

    candidates = ranked.copy()
    if "market_key" in candidates.columns:
        standard = candidates[
            candidates["market_key"].fillna("").astype(str).str.lower() == "player_threes"
        ].copy()
        if not standard.empty:
            candidates = standard

    candidates["model_prob"] = pd.to_numeric(candidates.get("model_prob"), errors="coerce")
    candidates["edge"] = pd.to_numeric(candidates.get("edge"), errors="coerce")
    candidates["expected_value_per_unit"] = pd.to_numeric(
        candidates.get("expected_value_per_unit"), errors="coerce"
    )

    market_prob = (
        candidates["no_vig_prob"].copy()
        if "no_vig_prob" in candidates.columns
        else pd.Series(np.nan, index=candidates.index)
    )
    if "implied_prob" in candidates.columns:
        market_prob = market_prob.where(market_prob.notna(), candidates["implied_prob"])
    candidates["main_line_distance"] = (pd.to_numeric(market_prob, errors="coerce") - 0.5).abs()
    candidates["main_line_distance"] = candidates["main_line_distance"].fillna(1.0)
    line_pool = candidates[
        candidates["main_line_distance"] <= float(max_main_line_distance)
    ].copy()
    if line_pool.empty:
        # Fallback: if strict cutoff removes everything, use the closest-to-main alternatives.
        line_pool = candidates.copy()

    strict = line_pool[line_pool["model_prob"] >= float(min_confidence_prob)].copy()
    if strict.empty:
        strict = line_pool.copy()

    strict = sort_confidence_rows(strict)
    best = strict.drop_duplicates(subset=["player_name"], keep="first").copy()
    best["confidence_profile"] = "high_conf_pick"

    # Guarantee stable top-N coverage for board/history: if strict filters leave fewer than target_count,
    # fill remaining slots from the best remaining lines by confidence-first ordering.
    if target_count > 0 and len(best) < target_count:
        fallback = line_pool.copy()
        if "player_name" in fallback.columns and "player_name" in best.columns and not best.empty:
            fallback = fallback[~fallback["player_name"].isin(best["player_name"])].copy()
        fallback = sort_confidence_rows(fallback)
        fallback = fallback.drop_duplicates(subset=["player_name"], keep="first").copy()
        fallback["confidence_profile"] = "confidence_fill_pick"
        needed = target_count - len(best)
        if needed > 0 and not fallback.empty:
            best = pd.concat([best, fallback.head(needed)], ignore_index=True, sort=False)

    best = sort_confidence_rows(best)
    best = rebalance_confidence_top_window(
        best,
        top_n=balance_top_n,
        min_per_side=min_per_side,
        max_prob_drop=balance_max_prob_drop,
    )
    if target_count > 0:
        best = best.head(target_count).copy()
    return best


# ---------------------------------------------------------------------------
# Main Logic
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Parse arguments and validate input files
    args = parse_args()
    if not args.history.exists():
        raise FileNotFoundError(f"Missing history dataset: {args.history}")
    if not args.props.exists():
        raise FileNotFoundError(f"Missing props dataset: {args.props}")

    # 2. Resolve model paths (binary and count models)
    model_path = resolve_model_path(args.model, DEFAULT_BINARY_MODEL_PATH)
    count_model_path = resolve_model_path(args.count_model, DEFAULT_COUNT_MODEL_PATH)
    print(f"Using binary model: {model_label(model_path)}")
    print(f"Using count model: {model_label(count_model_path)}")

    # 3. Load historical player game features and current props
    history = pd.read_csv(args.history)
    props = pd.read_csv(args.props)
    # 4. Build and save player predictability summary (holdout-based metrics)
    predictability = build_player_predictability_table(
        history,
        binary_model_path=model_path,
        count_model_path=count_model_path,
    )
    if not predictability.empty:
        args.predictability_output.parent.mkdir(parents=True, exist_ok=True)
        predictability.to_csv(args.predictability_output, index=False)
        starred_players = int(
            predictability["predictability_star"].fillna(False).astype(bool).sum()
        )
        print(
            f"Saved predictability summary to: {args.predictability_output} "
            f"({starred_players} starred players)"
        )

    # 5. Validate props CSV is not empty
    if props.empty:
        raise RuntimeError("Props CSV is empty.")

    # 6. Map commence times to local slate dates and filter by date if specified
    if "commence_time" in props.columns:
        props["slate_date"] = props["commence_time"].map(
            lambda value: local_slate_date(value, args.timezone)
        )
    else:
        props["slate_date"] = ""

    if args.slate_date:
        props = props[props["slate_date"] == args.slate_date].copy()
        if props.empty:
            raise RuntimeError(
                f"No props found for slate date {args.slate_date} in timezone {args.timezone}."
            )

    # 7. Initialize output collections and error tracking
    value_rows: list[dict] = []
    confidence_rows: list[dict] = []
    skipped_team_mismatch = 0
    skipped_blank_player_name = 0
    lookup_failures: list[str] = []

    # 8. Iterate over each prop and score with model
    for _, prop_row in props.iterrows():
        player_name = str(prop_row.get("player_name", "")).strip()
        if not player_name:
            skipped_blank_player_name += 1
            continue
        try:
            # 8a. Look up player history rows by name
            player_rows = select_player_rows(history, player_name)
        except Exception as exc:
            message = f"{player_name}: {exc}"
            lookup_failures.append(message)
            if args.strict_player_lookup:
                raise RuntimeError(f"Player lookup failed for '{player_name}': {exc}") from exc
            continue

        # 8b. Get latest player row and infer team/opponent
        player_row = latest_player_row(player_rows)
        actual_team, actual_opponent = infer_player_team_and_opponent(player_row, prop_row)
        event_home = normalize_team_to_abbr(prop_row.get("event_home_team", prop_row.get("team")))
        event_away = normalize_team_to_abbr(prop_row.get("event_away_team", prop_row.get("opponent")))
        if event_home and event_away and actual_team and actual_team not in {event_home, event_away}:
            skipped_team_mismatch += 1
            continue

        # 8c. Build feature row for model input
        feature_row = build_feature_row(history, prop_row, player_row, actual_team, actual_opponent)

        # 8d. Determine lambda (expected 3PM) from count or binary model
        lam, binary_prob = determine_lambda(
            history,
            player_rows,
            player_row,
            feature_row,
            actual_opponent,
            model_path,
            count_model_path,
        )
        # 8e. Capture defensive matchup snapshot
        defense_row = defense_snapshot(feature_row)

        # 8f. Score both OVER and UNDER sides, compute edge and EV
        for side_row in best_side_rows(prop_row, lam, binary_prob):
            if pd.isna(side_row["edge"]) or pd.isna(side_row["expected_value_per_unit"]):
                continue
            row_payload = {
                "player_name": player_row["player_name"],
                "team": display_team_name(actual_team),
                "opponent": display_team_name(actual_opponent),
                "market_key": prop_row.get("market_key"),
                "bookmaker": prop_row.get("bookmaker"),
                "commence_time": prop_row.get("commence_time"),
                "slate_date": prop_row.get("slate_date", ""),
                **defense_row,
                **side_row,
            }
            confidence_rows.append(row_payload)
            if side_row["edge"] >= args.min_edge:
                value_rows.append(row_payload)

    # 9. Consolidate scored rows and annotate with predictability
    confidence_pool = pd.DataFrame(confidence_rows)
    ranked = pd.DataFrame(value_rows)
    confidence_pool = annotate_predictability(confidence_pool, predictability)
    ranked = annotate_predictability(ranked, predictability)

    # 10. Filter out picks priced worse than max-negative-american-odds threshold
    confidence_pool, dropped_confidence_juiced = enforce_max_negative_american_odds(
        confidence_pool,
        args.max_negative_american_odds,
    )
    ranked, dropped_value_juiced = enforce_max_negative_american_odds(
        ranked,
        args.max_negative_american_odds,
    )
    if dropped_confidence_juiced > 0:
        print(
            f"Dropped {dropped_confidence_juiced} confidence rows priced worse than "
            f"{args.max_negative_american_odds}."
        )
    if dropped_value_juiced > 0:
        print(
            f"Dropped {dropped_value_juiced} value rows priced worse than "
            f"{args.max_negative_american_odds}."
        )
    if ranked.empty and not confidence_pool.empty:
        ranked = pd.DataFrame(columns=confidence_pool.columns)

    # 11. Handle early exit if no confidence picks available
    if confidence_pool.empty:
        print("No props available for confidence ranking.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        ranked.to_csv(args.output, index=False)
        args.confidence_output.parent.mkdir(parents=True, exist_ok=True)
        confidence_pool.to_csv(args.confidence_output, index=False)
        return
    if ranked.empty:
        print("No props met the minimum edge threshold for value ranking.")

    # 12. Validate data integrity: drop rows where team equals opponent (malformed source rows)
    if {"team", "opponent"}.issubset(confidence_pool.columns):
        same_team_mask_confidence = (
            confidence_pool["team"].fillna("").astype(str).str.upper()
            == confidence_pool["opponent"].fillna("").astype(str).str.upper()
        )
        bad_count_confidence = int(same_team_mask_confidence.sum())
        if bad_count_confidence > 0:
            confidence_pool = confidence_pool.loc[~same_team_mask_confidence].copy()
            print(f"Dropped {bad_count_confidence} confidence rows where team matched opponent.")
            if confidence_pool.empty:
                print("No confidence props left after dropping invalid team/opponent rows.")
                args.output.parent.mkdir(parents=True, exist_ok=True)
                ranked.to_csv(args.output, index=False)
                args.confidence_output.parent.mkdir(parents=True, exist_ok=True)
                confidence_pool.to_csv(args.confidence_output, index=False)
                return

    # 13. Validate data integrity: drop value rows where team equals opponent
    if {"team", "opponent"}.issubset(ranked.columns):
        same_team_mask = (
            ranked["team"].fillna("").astype(str).str.upper()
            == ranked["opponent"].fillna("").astype(str).str.upper()
        )
        bad_count = int(same_team_mask.sum())
        if bad_count > 0:
            ranked = ranked.loc[~same_team_mask].copy()
            print(f"Dropped {bad_count} rows where team matched opponent.")
            if ranked.empty:
                print("No props left after dropping invalid team/opponent rows.")
                args.output.parent.mkdir(parents=True, exist_ok=True)
                ranked.to_csv(args.output, index=False)
                args.confidence_output.parent.mkdir(parents=True, exist_ok=True)
                confidence_pool.to_csv(args.confidence_output, index=False)
                return

    # 14. Rank value picks by edge and expected value
    ranked.sort_values(
        ["edge", "expected_value_per_unit", "model_prob"],
        ascending=False,
        inplace=True,
    )
    best_value = ranked.drop_duplicates(subset=["player_name"], keep="first").copy()
    best_value.insert(0, "rank", np.arange(1, len(best_value) + 1))
    recommendation_date = pd.Timestamp.today().date().isoformat()
    best_value.insert(1, "recommendation_date", recommendation_date)
    best_value = add_tracking_columns(best_value)

    # 15. Build confidence board (main-line O/U picks, balanced sides, filtered by probability)
    most_confident = build_confidence_board(
        confidence_pool,
        max_main_line_distance=args.confidence_main_line_max_distance,
        min_confidence_prob=args.confidence_min_prob,
        target_count=DEFAULT_CONFIDENCE_TARGET_COUNT,
        balance_top_n=args.confidence_balance_top_n,
        min_per_side=args.confidence_min_per_side,
        balance_max_prob_drop=args.confidence_balance_max_prob_drop,
    )
    most_confident.insert(0, "rank", np.arange(1, len(most_confident) + 1))
    most_confident.insert(1, "recommendation_date", recommendation_date)
    most_confident = add_tracking_columns(most_confident)

    # 16. Write main output files: ranked value board and most confident board
    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_value.to_csv(args.output, index=False)
    args.confidence_output.parent.mkdir(parents=True, exist_ok=True)
    most_confident.to_csv(args.confidence_output, index=False)

    # 17. Track value board picks: append to history archive and build playable shortlist
    tracked_candidates = best_value.copy()
    if "slate_date" in tracked_candidates.columns:
        tracked_candidates = tracked_candidates[
            tracked_candidates["slate_date"].fillna("").astype(str) >= TRACKING_START_DATE_STR
        ].copy()
    tracked_candidates = daily_top_rows(tracked_candidates, args.daily_cap)
    append_archive(args.history_output, tracked_candidates)
    tracking_path = args.history_output.parent / "tracked_bets_3pt.csv"
    append_tracking_log(tracking_path, tracked_candidates)
    playable = tracked_candidates.copy()
    playable["is_playable"] = True
    args.playable_output.parent.mkdir(parents=True, exist_ok=True)
    playable.to_csv(args.playable_output, index=False)

    # 18. Track confidence board picks: append to history archive and build parlay tracking
    confidence_candidates = most_confident.copy()
    if "slate_date" in confidence_candidates.columns:
        confidence_candidates = confidence_candidates[
            confidence_candidates["slate_date"].fillna("").astype(str) >= TRACKING_START_DATE_STR
        ].copy()
    confidence_candidates = daily_top_rows(confidence_candidates, DEFAULT_CONFIDENCE_HISTORY_COUNT)
    append_archive(args.confidence_history_output, confidence_candidates)
    append_tracking_log(args.confidence_tracking_output, confidence_candidates)
    # 18a. Build and log top-3 and top-5 parlay combinations
    top3_parlay = build_top3_parlay_rows(confidence_candidates)
    if not top3_parlay.empty:
        append_parlay_log(args.parlay_output, top3_parlay)
    top5_parlay = build_top5_parlay_rows(confidence_candidates)
    if not top5_parlay.empty:
        append_parlay_log(args.parlay_top5_output, top5_parlay)

    # 19. Format and display top value picks for console output
    value_printable = best_value[
        [
            "rank",
            "player_display_name" if "player_display_name" in best_value.columns else "player_name",
            "tracking_bucket",
            "side",
            "line",
            "team",
            "opponent",
            "bookmaker",
            "american_odds",
            "decimal_odds",
            "model_prob",
            "edge",
            "expected_value_per_unit",
            "estimated_3pm_mean",
        ]
    ].head(args.top).copy()
    value_printable.rename(
        columns={
            "player_display_name": "Player",
            "player_name": "Player",
            "tracking_bucket": "TrackBucket",
            "side": "Pick",
            "line": "Line",
            "team": "Team",
            "opponent": "Opp",
            "bookmaker": "Book",
            "american_odds": "AmerOdds",
            "decimal_odds": "DecOdds",
            "model_prob": "Confidence",
            "edge": "Edge",
            "expected_value_per_unit": "EV",
            "estimated_3pm_mean": "Proj_3PM",
        },
        inplace=True,
    )

    # 20. Format and display top confidence picks for console output
    confidence_printable = most_confident[
        [
            "rank",
            "player_display_name" if "player_display_name" in most_confident.columns else "player_name",
            "side",
            "line",
            "team",
            "opponent",
            "bookmaker",
            "american_odds",
            "decimal_odds",
            "model_prob",
            "edge",
            "expected_value_per_unit",
            "estimated_3pm_mean",
        ]
    ].head(args.top).copy()
    confidence_printable.rename(
        columns={
            "player_display_name": "Player",
            "player_name": "Player",
            "side": "Pick",
            "line": "Line",
            "team": "Team",
            "opponent": "Opp",
            "bookmaker": "Book",
            "american_odds": "AmerOdds",
            "decimal_odds": "DecOdds",
            "model_prob": "Confidence",
            "edge": "Edge",
            "expected_value_per_unit": "EV",
            "estimated_3pm_mean": "Proj_3PM",
        },
        inplace=True,
    )

    # 21. Print rankings and output file summary
    print("Best Value")
    print(value_printable.to_string(index=False))
    print("\nMost Confident (Main-Line O/U)")
    print(confidence_printable.to_string(index=False))
    print(f"\nSaved best-value list to: {args.output}")
    print(f"Saved most-confident list to: {args.confidence_output}")
    print(f"Saved tracked recommendations to: {args.history_output}")
    print(f"Saved tracked bets log to: {tracking_path}")
    print(f"Saved playable shortlist to: {args.playable_output}")
    print(f"Saved confidence history to: {args.confidence_history_output}")
    print(f"Saved confidence tracked log to: {args.confidence_tracking_output}")
    print(f"Saved top-3 parlay log to: {args.parlay_output}")
    print(f"Saved top-5 parlay log to: {args.parlay_top5_output}")

    # 22. Report any skipped props and lookup failures
    if skipped_team_mismatch > 0:
        print(
            f"Skipped {skipped_team_mismatch} props where player team did not match event home/away teams."
        )
    if skipped_blank_player_name > 0:
        print(f"Skipped {skipped_blank_player_name} props with blank player_name.")
    if lookup_failures:
        print(f"Skipped {len(lookup_failures)} props due to player lookup failures.")
        print("Sample lookup failures:")
        for sample in lookup_failures[:10]:
            print(f"  - {sample}")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
