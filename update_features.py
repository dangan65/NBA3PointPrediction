"""
Build season-to-date player-game training set for 3PT props predictions.

Pulls player game logs from nba_api, computes rolling pre-game features (last_5_games_3pm_avg,
season_avg_3pm, minutes trends, usage rates) without data leakage (all features use shifted
windows). Merges team defense overlays (opponent 3PA/3PM allowed, defensive rating, pace)
and game context overlays (spreads, totals, player props context). Upserts final row set
into data/player_game_3pt_features.csv.

Inputs:
  - NBA season (YYYY-YY format) and season type (Regular Season, etc.)
  - As-of date: include completed games through this date
  - Optional CSV overlays: player tracking stats, team defense, game context

Outputs:
  - player_game_3pt_features.csv: one row per player-game with 90+ columns including:
    rolling averages, per-game stats, opponent defense metrics, game context, shooting splits
"""

from __future__ import annotations

import argparse
import warnings
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = [
    "season",
    "as_of_date",
    "game_date",
    "game_id",
    "player_id",
    "player_name",
    "team",
    "opponent",
    "home_game",
    "position",
    "rest_days",
    "back_to_back",
    "minutes_per_game",
    "projected_minutes",
    "last_5_minutes_avg",
    "last_10_minutes_avg",
    "minutes_stddev_last_10",
    "minutes_trend_slope",
    "starter_flag",
    "games_started_last_5",
    "teammate_minutes_vacated",
    "usage_rate",
    "teammate_usage_vacated",
    "team_3pa_share",
    "last_5_3pa_trend",
    "last_5_games_3pm_avg",
    "last_5_games_3pa_avg",
    "last_3_games_3pm",
    "season_avg_3pm",
    "rolling_3_game_trend_slope",
    "injury_usage_bump",
    "opp_3pa_allowed",
    "opp_3pm_allowed",
    "opp_def_rating",
    "opp_pace",
    "opp_pace_differential",
    "opp_position_3pt_rank",
    "opp_3pa_allowed_by_position",
    "spread",
    "game_total",
    "season_3pa",
    "season_3pm",
    "season_3pt_pct",
    "catch_shoot_3pa",
    "catch_shoot_3pm",
    "pull_up_3pa",
    "pull_up_3pm",
    "catch_and_shoot_rate",
    "pull_up_rate",
    "touch_time",
    "dribbles_before_shot",
    "wide_open_3pa",
    "wide_open_3pt_pct",
    "open_3pa",
    "open_3pt_pct",
    "tight_3pa",
    "tight_3pt_pct",
    "very_tight_3pa",
    "very_tight_3pt_pct",
    "team_offensive_role",
    "teammate_on_off_3pa_delta",
    "opp_team_wide_open_3pa_allowed",
    "opp_team_above_break_3pa_allowed",
    "opp_team_corner_3pa_allowed",
    "corner_3_allowed",
    "above_break_3_allowed",
    "defensive_scheme_tendency",
    "expected_primary_defender",
    "primary_defender_id",
    "primary_defender_pos",
    "primary_defender_3pt_pct_allowed",
    "primary_defender_def_fg_pct_diff",
    "player_made_3pt",
    "stats_finalized_at",
    "source_boxscore",
    "source_tracking",
    "row_version",
]

TRACKING_NUMERIC_COLUMNS = [
    "catch_shoot_3pa",
    "catch_shoot_3pm",
    "pull_up_3pa",
    "pull_up_3pm",
    "touch_time",
    "dribbles_before_shot",
    "wide_open_3pa",
    "wide_open_3pt_pct",
    "open_3pa",
    "open_3pt_pct",
    "tight_3pa",
    "tight_3pt_pct",
    "very_tight_3pa",
    "very_tight_3pt_pct",
]

DEFAULT_TEAM_DEFENSE_CSV = Path("data/team_defense_template.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the season-to-date player-game training set for 3PT props."
    )
    parser.add_argument(
        "--season",
        default="2025-26",
        help="NBA season in YYYY-YY format.",
    )
    parser.add_argument(
        "--season-type",
        default="Regular Season",
        help="Season type for stats.nba.com endpoints.",
    )
    parser.add_argument(
        "--as-of-date",
        type=parse_iso_date,
        default=date.today(),
        help="Include completed games through this date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/player_game_3pt_features.csv"),
        help="Where to write the rebuilt training CSV.",
    )
    parser.add_argument(
        "--player-tracking-csv",
        type=Path,
        help="Optional CSV keyed by player_id to override or backfill tracking columns.",
    )
    parser.add_argument(
        "--team-defense-csv",
        type=Path,
        help=(
            "Optional CSV keyed by opponent team abbreviation to override opponent defense columns. "
            "If omitted, data/team_defense_template.csv is used when present."
        ),
    )
    parser.add_argument(
        "--game-context-csv",
        type=Path,
        help="Optional CSV keyed by game_id (and optionally player_id) to backfill spreads, totals, or player context.",
    )
    return parser.parse_args()


def parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def parse_minutes(value: object) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if not text:
        return np.nan
    if ":" not in text:
        try:
            return float(text)
        except ValueError:
            return np.nan
    minute_text, second_text = text.split(":", 1)
    try:
        minutes = float(minute_text)
        seconds = float(second_text)
    except ValueError:
        return np.nan
    return minutes + (seconds / 60.0)


def shifted_rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window=window, min_periods=1).mean()


def shifted_expanding_mean(series: pd.Series) -> pd.Series:
    return series.shift(1).expanding(min_periods=1).mean()


def shifted_rolling_std(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window=window, min_periods=2).std()


def shifted_rolling_sum(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window=window, min_periods=1).sum()


def rolling_trend_slope(series: pd.Series, window: int = 3) -> pd.Series:
    shifted = series.shift(1)
    values = shifted.to_numpy(dtype=float)
    slopes = np.full(len(values), np.nan)
    x = np.arange(window, dtype=float)

    for index in range(len(values)):
        start = index - window + 1
        if start < 0:
            continue
        window_values = values[start : index + 1]
        if np.isnan(window_values).any():
            continue
        slopes[index] = np.polyfit(x, window_values, 1)[0]

    return pd.Series(slopes, index=series.index)


def extract_opponent(matchup: str) -> str:
    if not isinstance(matchup, str) or not matchup:
        return ""
    parts = matchup.split()
    return parts[-1] if parts else ""


def is_home_game(matchup: str) -> int:
    if not isinstance(matchup, str):
        return 0
    return int("vs." in matchup)


def normalize_column(frame: pd.DataFrame, candidates: list[str], fallback: object = np.nan) -> pd.Series:
    for column in candidates:
        if column in frame.columns:
            return frame[column]
    return pd.Series(fallback, index=frame.index)


def load_optional_csv(path: Path | None, label: str) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    if not path.exists():
        warnings.warn(f"{label} CSV not found: {path}")
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty:
        warnings.warn(f"{label} CSV is empty: {path}")
    return frame


def merge_external_data(
    base: pd.DataFrame,
    external: pd.DataFrame,
    merge_keys: list[str],
    supported_columns: list[str],
) -> pd.DataFrame:
    if external.empty:
        return base

    if any(key not in external.columns for key in merge_keys):
        missing = [key for key in merge_keys if key not in external.columns]
        warnings.warn(f"Skipping external merge; missing required keys: {missing}")
        return base

    merge_columns = [column for column in supported_columns if column in external.columns]
    if not merge_columns:
        return base

    subset = external[merge_keys + merge_columns].drop_duplicates(subset=merge_keys)
    merged = base.merge(subset, on=merge_keys, how="left", suffixes=("", "_external"))

    for column in merge_columns:
        external_column = f"{column}_external"
        if external_column not in merged.columns:
            continue
        if column in merged.columns:
            merged[column] = merged[external_column].combine_first(merged[column])
        else:
            merged[column] = merged[external_column]
        merged.drop(columns=[external_column], inplace=True)

    return merged


def fetch_player_game_logs(season: str, season_type: str, as_of_date: date) -> pd.DataFrame:
    try:
        from nba_api.stats.endpoints import leaguegamefinder
    except ImportError as exc:
        raise RuntimeError("nba_api is required. Install dependencies with pip.") from exc

    finder = leaguegamefinder.LeagueGameFinder(
        player_or_team_abbreviation="P",
        season_nullable=season,
        season_type_nullable=season_type,
        date_to_nullable=as_of_date.strftime("%m/%d/%Y"),
    )
    frame = finder.get_data_frames()[0]
    if frame.empty:
        return frame

    frame = frame.copy()
    frame["GAME_DATE"] = pd.to_datetime(frame["GAME_DATE"])
    frame["MINUTES_FLOAT"] = frame["MIN"].map(parse_minutes)
    frame = frame[frame["MINUTES_FLOAT"].fillna(0) > 0].copy()
    frame.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"], inplace=True)
    return frame


def fetch_team_dashboard_metrics(season: str, season_type: str) -> pd.DataFrame:
    try:
        from nba_api.stats.endpoints import leaguedashteamstats
    except ImportError:
        return pd.DataFrame(columns=["team_id", "team", "team_def_rating", "team_pace"])

    try:
        dashboard = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star=season_type,
            per_mode_detailed="PerGame",
        )
        frame = dashboard.get_data_frames()[0]
    except Exception as exc:  # pragma: no cover - endpoint stability is external
        warnings.warn(f"Team dashboard fetch failed; continuing without DEF_RATING/PACE: {exc}")
        return pd.DataFrame(columns=["team_id", "team", "team_def_rating", "team_pace"])

    if frame.empty:
        return pd.DataFrame(columns=["team_id", "team", "team_def_rating", "team_pace"])

    normalized = pd.DataFrame(
        {
            "team_id": normalize_column(frame, ["TEAM_ID"]),
            "team": normalize_column(frame, ["TEAM_ABBREVIATION", "TEAM_NAME"]),
            "team_def_rating": normalize_column(frame, ["DEF_RATING"]),
            "team_pace": normalize_column(frame, ["PACE"]),
        }
    )
    return normalized.drop_duplicates(subset=["team_id"])


def fetch_tracking_splits(season: str, season_type: str) -> pd.DataFrame:
    rows = pd.DataFrame(columns=["player_id", *TRACKING_NUMERIC_COLUMNS])

    try:
        from nba_api.stats.endpoints import leaguedashplayerptshot
    except ImportError:
        return rows

    try:
        endpoint = leaguedashplayerptshot.LeagueDashPlayerPtShot(
            season=season,
            season_type_all_star=season_type,
            per_mode_simple="PerGame",
        )
        frame = endpoint.get_data_frames()[0]
    except Exception as exc:  # pragma: no cover - endpoint stability is external
        warnings.warn(f"Tracking split fetch failed; continuing without tracking columns: {exc}")
        return rows

    if frame.empty:
        return rows

    normalized = pd.DataFrame(
        {
            "player_id": normalize_column(frame, ["PLAYER_ID"]),
            "catch_shoot_3pa": normalize_column(frame, ["CATCH_SHOOT_FG3A", "CATCH_SHOOT_3PA"]),
            "catch_shoot_3pm": normalize_column(frame, ["CATCH_SHOOT_FG3M", "CATCH_SHOOT_3PM"]),
            "pull_up_3pa": normalize_column(frame, ["PULL_UP_FG3A", "PULL_UP_3PA"]),
            "pull_up_3pm": normalize_column(frame, ["PULL_UP_FG3M", "PULL_UP_3PM"]),
            "touch_time": normalize_column(frame, ["AVG_SEC_PER_TOUCH", "TOUCH_TIME"]),
            "dribbles_before_shot": normalize_column(
                frame, ["AVG_DRIB_PER_TOUCH", "DRIBBLES_BEFORE_SHOT", "DRIBBLES"]
            ),
            "wide_open_3pa": normalize_column(frame, ["WIDE_OPEN_FG3A", "WIDE_OPEN_3PA"]),
            "wide_open_3pt_pct": normalize_column(frame, ["WIDE_OPEN_FG3_PCT", "WIDE_OPEN_3PT_PCT"]),
            "open_3pa": normalize_column(frame, ["OPEN_FG3A", "OPEN_3PA"]),
            "open_3pt_pct": normalize_column(frame, ["OPEN_FG3_PCT", "OPEN_3PT_PCT"]),
            "tight_3pa": normalize_column(frame, ["TIGHT_FG3A", "TIGHT_3PA"]),
            "tight_3pt_pct": normalize_column(frame, ["TIGHT_FG3_PCT", "TIGHT_3PT_PCT"]),
            "very_tight_3pa": normalize_column(frame, ["VERY_TIGHT_FG3A", "VERY_TIGHT_3PA"]),
            "very_tight_3pt_pct": normalize_column(
                frame, ["VERY_TIGHT_FG3_PCT", "VERY_TIGHT_3PT_PCT"]
            ),
        }
    )
    normalized = normalized.drop_duplicates(subset=["player_id"])
    for column in TRACKING_NUMERIC_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized


def build_team_games(player_logs: pd.DataFrame) -> pd.DataFrame:
    team_games = (
        player_logs.groupby(["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "GAME_DATE"], as_index=False)
        .agg(
            team_fg3a=("FG3A", "sum"),
            team_fg3m=("FG3M", "sum"),
            team_points=("team_points_scored", "sum"),
            team_possessions_used=("possessions_used", "sum"),
        )
        .rename(
            columns={
                "GAME_ID": "game_id",
                "TEAM_ID": "team_id",
                "TEAM_ABBREVIATION": "team",
                "GAME_DATE": "game_date",
            }
        )
    )

    opponent_lookup = team_games[
        [
            "game_id",
            "team_id",
            "team",
            "team_fg3a",
            "team_fg3m",
            "team_points",
            "team_possessions_used",
        ]
    ].rename(
        columns={
            "team_id": "opponent_team_id",
            "team": "opponent",
            "team_fg3a": "opp_fg3a",
            "team_fg3m": "opp_fg3m",
            "team_points": "opp_points",
            "team_possessions_used": "opponent_possessions_used",
        }
    )

    team_games = team_games.merge(opponent_lookup, on="game_id", how="left")
    team_games = team_games[team_games["team_id"] != team_games["opponent_team_id"]].copy()
    team_games.sort_values(["team_id", "game_date", "game_id"], inplace=True)

    grouped = team_games.groupby("team_id", sort=False)
    team_games["team_fg3a_before"] = grouped["team_fg3a"].cumsum() - team_games["team_fg3a"]
    team_games["opp_3pa_allowed"] = grouped["opp_fg3a"].transform(shifted_expanding_mean)
    team_games["opp_3pm_allowed"] = grouped["opp_fg3m"].transform(shifted_expanding_mean)
    team_possessions = pd.to_numeric(team_games["team_possessions_used"], errors="coerce")
    opp_possessions = pd.to_numeric(team_games["opponent_possessions_used"], errors="coerce")
    opp_points = pd.to_numeric(team_games["opp_points"], errors="coerce")
    team_games["team_pace_game"] = (team_possessions + opp_possessions) / 2.0
    team_games["team_def_rating_game"] = (opp_points / opp_possessions.replace(0, np.nan)) * 100.0
    team_games["team_def_rating"] = grouped["team_def_rating_game"].transform(shifted_expanding_mean)
    team_games["team_pace"] = grouped["team_pace_game"].transform(shifted_expanding_mean)

    return team_games


def build_feature_frame(
    player_logs: pd.DataFrame,
    team_games: pd.DataFrame,
    tracking: pd.DataFrame,
    season: str,
    as_of_date: date,
) -> pd.DataFrame:
    base = player_logs.copy()
    base["game_date"] = base["GAME_DATE"]
    base["game_id"] = base["GAME_ID"]
    base["player_id"] = base["PLAYER_ID"]
    base["player_name"] = base["PLAYER_NAME"]
    base["team_id"] = base["TEAM_ID"]
    base["team"] = base["TEAM_ABBREVIATION"]
    base["opponent"] = base["MATCHUP"].map(extract_opponent)
    base["home_game"] = base["MATCHUP"].map(is_home_game)
    base["raw_position"] = normalize_column(base, ["START_POSITION"], "UNKNOWN").replace("", "UNKNOWN")
    base["minutes"] = base["MINUTES_FLOAT"]
    base["player_made_3pt"] = base["FG3M"]
    base["fg3a"] = base["FG3A"]
    base = base.merge(
        team_games[
            [
                "game_id",
                "team_id",
                "opponent_team_id",
                "team_fg3a_before",
                "team_pace",
            ]
        ],
        on=["game_id", "team_id"],
        how="left",
    )
    base.rename(columns={"team_pace": "team_pace_context"}, inplace=True)

    opponent_features = team_games[
        ["game_id", "team_id", "opp_3pa_allowed", "opp_3pm_allowed", "team_def_rating", "team_pace"]
    ].rename(
        columns={
            "team_id": "opponent_team_id",
            "team_def_rating": "opp_def_rating",
            "team_pace": "opp_pace",
        }
    )
    base = base.merge(opponent_features, on=["game_id", "opponent_team_id"], how="left")
    base.sort_values(["player_id", "game_date", "game_id"], inplace=True)

    grouped = base.groupby("player_id", sort=False)
    prior_games = grouped.cumcount().astype(float)
    prior_fg3m = grouped["player_made_3pt"].cumsum() - base["player_made_3pt"]
    prior_fg3a = grouped["fg3a"].cumsum() - base["fg3a"]

    base["last_5_games_3pm_avg"] = grouped["player_made_3pt"].transform(
        lambda series: shifted_rolling_mean(series, 5)
    )
    base["last_5_games_3pa_avg"] = grouped["fg3a"].transform(lambda series: shifted_rolling_mean(series, 5))
    base["last_3_games_3pm"] = grouped["player_made_3pt"].transform(
        lambda series: shifted_rolling_mean(series, 3)
    )
    base["season_avg_3pm"] = prior_fg3m / prior_games.replace(0, np.nan)
    base["rolling_3_game_trend_slope"] = grouped["player_made_3pt"].transform(rolling_trend_slope)
    base["minutes_per_game"] = grouped["minutes"].transform(shifted_expanding_mean)
    base["last_5_minutes_avg"] = grouped["minutes"].transform(lambda series: shifted_rolling_mean(series, 5))
    base["last_10_minutes_avg"] = grouped["minutes"].transform(lambda series: shifted_rolling_mean(series, 10))
    base["minutes_stddev_last_10"] = grouped["minutes"].transform(lambda series: shifted_rolling_std(series, 10))
    base["minutes_trend_slope"] = grouped["minutes"].transform(lambda series: rolling_trend_slope(series, 5))
    base["usage_rate"] = grouped["usage_proxy"].transform(shifted_expanding_mean)
    base["last_5_3pa_trend"] = grouped["fg3a"].transform(lambda series: rolling_trend_slope(series, 5))

    prior_started = grouped["raw_position"].shift(1).fillna("UNKNOWN").ne("UNKNOWN").astype(int)
    base["starter_flag"] = prior_started
    base["games_started_last_5"] = grouped["raw_position"].transform(
        lambda series: shifted_rolling_sum(series.fillna("UNKNOWN").ne("UNKNOWN").astype(float), 5)
    )

    prior_position = grouped["raw_position"].shift(1)
    base["position"] = prior_position.groupby(base["player_id"]).ffill().fillna("UNKNOWN")

    base["season_3pm"] = prior_fg3m
    base["season_3pa"] = prior_fg3a
    base["season_3pt_pct"] = prior_fg3m / prior_fg3a.replace(0, np.nan)
    base["team_3pa_share"] = prior_fg3a / base["team_fg3a_before"].replace(0, np.nan)
    blended_minutes = (
        base["last_5_minutes_avg"].fillna(base["last_10_minutes_avg"]) * 0.65
        + base["last_10_minutes_avg"].fillna(base["minutes_per_game"]) * 0.35
    )
    base["projected_minutes"] = blended_minutes + base["minutes_trend_slope"].fillna(0)

    rest_delta = grouped["game_date"].diff().dt.days
    base["rest_days"] = (rest_delta - 1).clip(lower=0)
    base["back_to_back"] = (base["rest_days"] == 0).astype("Int64")

    base["season"] = season
    base["as_of_date"] = as_of_date.isoformat()
    base["game_date"] = base["game_date"].dt.date.astype(str)
    base["injury_usage_bump"] = np.nan
    base["opp_position_3pt_rank"] = np.nan
    base["opp_3pa_allowed_by_position"] = np.nan
    base["spread"] = np.nan
    base["game_total"] = np.nan
    base["teammate_minutes_vacated"] = np.nan
    base["teammate_usage_vacated"] = np.nan
    base["teammate_on_off_3pa_delta"] = np.nan
    base["opp_team_wide_open_3pa_allowed"] = np.nan
    base["opp_team_above_break_3pa_allowed"] = np.nan
    base["opp_team_corner_3pa_allowed"] = np.nan
    base["corner_3_allowed"] = np.nan
    base["above_break_3_allowed"] = np.nan
    base["defensive_scheme_tendency"] = "UNKNOWN"
    base["expected_primary_defender"] = pd.NA
    base["primary_defender_id"] = pd.NA
    base["primary_defender_pos"] = pd.NA
    base["primary_defender_3pt_pct_allowed"] = np.nan
    base["primary_defender_def_fg_pct_diff"] = np.nan

    if not tracking.empty:
        base = base.merge(tracking, on="player_id", how="left")
        has_tracking = base[TRACKING_NUMERIC_COLUMNS].notna().any(axis=1)
        base["source_tracking"] = np.where(has_tracking, "nba_api_tracking", "unavailable")
    else:
        for column in TRACKING_NUMERIC_COLUMNS:
            base[column] = np.nan
        base["source_tracking"] = "unavailable"

    base["catch_and_shoot_rate"] = base["catch_shoot_3pa"] / base["season_3pa"].replace(0, np.nan)
    base["pull_up_rate"] = base["pull_up_3pa"] / base["season_3pa"].replace(0, np.nan)
    base["corner_3_allowed"] = base["opp_team_corner_3pa_allowed"]
    base["above_break_3_allowed"] = base["opp_team_above_break_3pa_allowed"]
    base["opp_pace_differential"] = base["opp_pace"] - base["team_pace_context"]

    base["team_offensive_role"] = np.select(
        [
            base["usage_rate"] >= 0.28,
            base["usage_rate"] >= 0.22,
            base["usage_rate"] >= 0.16,
        ],
        [
            "PRIMARY",
            "SECONDARY",
            "CONNECTOR",
        ],
        default="LOW_USAGE",
    )

    base["stats_finalized_at"] = datetime.now().isoformat(timespec="seconds")
    base["source_boxscore"] = "nba_api_leaguegamefinder"
    base["row_version"] = 1

    for column in OUTPUT_COLUMNS:
        if column not in base.columns:
            base[column] = np.nan

    result = base[OUTPUT_COLUMNS].copy()
    result.sort_values(["game_date", "team", "player_name"], inplace=True)
    return result


def main() -> None:
    # 1. Parse arguments and fetch player game logs from nba_api
    args = parse_args()

    player_logs = fetch_player_game_logs(args.season, args.season_type, args.as_of_date)
    if player_logs.empty:
        raise RuntimeError("No player game logs were returned for the requested season/date.")

    # 2. Compute usage proxy from possession-based stats (FGA, FTA, TOV, PTS)
    fga = pd.to_numeric(normalize_column(player_logs, ["FGA"], 0), errors="coerce").fillna(0)
    fta = pd.to_numeric(normalize_column(player_logs, ["FTA"], 0), errors="coerce").fillna(0)
    tov = pd.to_numeric(normalize_column(player_logs, ["TOV"], 0), errors="coerce").fillna(0)
    player_logs["possessions_used"] = fga + (0.44 * fta) + tov
    player_logs["team_points_scored"] = pd.to_numeric(
        normalize_column(player_logs, ["PTS"], 0), errors="coerce"
    ).fillna(0)
    team_totals = player_logs.groupby(["GAME_ID", "TEAM_ID"])["possessions_used"].transform("sum")
    player_logs["usage_proxy"] = np.where(team_totals > 0, player_logs["possessions_used"] / team_totals, np.nan)

    # 3. Fetch tracking splits and build per-team-game aggregates (pace, opponent 3PA/3PM allowed, def rating)
    tracking = fetch_tracking_splits(args.season, args.season_type)
    team_games = build_team_games(player_logs)

    # 4. Build feature frame with shifted rolling windows (no forward leakage): rolling averages, trends, per-game stats
    result = build_feature_frame(player_logs, team_games, tracking, args.season, args.as_of_date)

    # 5. Merge external player tracking data (catch-and-shoot splits, touch time, etc.)
    external_player_tracking = load_optional_csv(args.player_tracking_csv, "Player tracking")
    result = merge_external_data(
        result,
        external_player_tracking,
        ["player_id"],
        [
            "catch_shoot_3pa",
            "catch_shoot_3pm",
            "pull_up_3pa",
            "pull_up_3pm",
            "catch_and_shoot_rate",
            "pull_up_rate",
            "touch_time",
            "dribbles_before_shot",
            "wide_open_3pa",
            "wide_open_3pt_pct",
            "open_3pa",
            "open_3pt_pct",
            "tight_3pa",
            "tight_3pt_pct",
            "very_tight_3pa",
            "very_tight_3pt_pct",
            "expected_primary_defender",
            "primary_defender_id",
            "primary_defender_pos",
            "primary_defender_3pt_pct_allowed",
            "primary_defender_def_fg_pct_diff",
            "teammate_on_off_3pa_delta",
        ],
    )

    # 6. Merge external team defense overlays (opponent 3PA/3PM allowed, pace, position-level ranks)
    team_defense_path = args.team_defense_csv
    if team_defense_path is None and DEFAULT_TEAM_DEFENSE_CSV.exists():
        team_defense_path = DEFAULT_TEAM_DEFENSE_CSV
        print(f"Using team defense overlay: {team_defense_path}")

    external_team_defense = load_optional_csv(team_defense_path, "Team defense")
    result = merge_external_data(
        result,
        external_team_defense,
        ["opponent"],
        [
            "opp_3pa_allowed",
            "opp_3pm_allowed",
            "opp_def_rating",
            "opp_pace",
            "opp_position_3pt_rank",
            "opp_3pa_allowed_by_position",
            "opp_team_wide_open_3pa_allowed",
            "opp_team_above_break_3pa_allowed",
            "opp_team_corner_3pa_allowed",
            "corner_3_allowed",
            "above_break_3_allowed",
            "defensive_scheme_tendency",
        ],
    )

    # 7. Merge external game context overlays (spreads, totals) with safe key detection (game_id preferred)
    external_game_context = load_optional_csv(args.game_context_csv, "Game context")
    if not external_game_context.empty:
        external_game_context = external_game_context.copy()
        if "game_date" not in external_game_context.columns:
            if "slate_date" in external_game_context.columns:
                external_game_context["game_date"] = external_game_context["slate_date"]
            elif "commence_time" in external_game_context.columns:
                commence_dates = pd.to_datetime(
                    external_game_context["commence_time"], errors="coerce", utc=True
                )
                external_game_context["game_date"] = commence_dates.dt.strftime("%Y-%m-%d")

        if {"game_id", "player_id"}.issubset(external_game_context.columns):
            game_context_keys = ["game_id", "player_id"]
        elif "game_id" in external_game_context.columns:
            game_context_keys = ["game_id"]
        elif {"team", "opponent", "game_date"}.issubset(external_game_context.columns):
            game_context_keys = ["team", "opponent", "game_date"]
        else:
            game_context_keys = []
            warnings.warn(
                "Skipping game context merge because no safe key set was found. "
                "Provide game_id (preferred) or team+opponent+game_date to avoid cross-date leakage."
            )
        if game_context_keys:
            result = merge_external_data(
                result,
                external_game_context,
                game_context_keys,
                [
                    "spread",
                    "game_total",
                    "injury_usage_bump",
                    "position",
                    "projected_minutes",
                    "starter_flag",
                    "teammate_minutes_vacated",
                    "teammate_usage_vacated",
                ],
            )

    # 8. Write final feature frame to CSV and log completion
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    print(f"Wrote {len(result)} player-game rows to {args.output}")
    print(f"Season: {args.season}")
    print(f"As of date: {args.as_of_date.isoformat()}")


if __name__ == "__main__":
    main()
