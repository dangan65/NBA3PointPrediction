"""
Module for tracking, logging, and settling betting outcomes on 3PT props.

Purpose:
  - Manage bet tracking records (add columns, archive recommendations)
  - Build and settle parlay combinations from top-ranked predictions
  - Auto-settle tracked bets by looking up actual game outcomes from history CSV or NBA API
  - Maintain clean CSV logs of tracked bets, parlays, and historical data

Inputs:
  - DataFrames with recommendation data, tracking records, and player game history
  - File paths to CSV logs (tracking, parlay, archive, history)
  - Slate date strings and player/bookmaker details

Outputs:
  - Updated CSV files with tracked bets, parlay records, and archives
  - Boolean flags indicating whether settlement/changes occurred
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


TRACKING_START_DATE = date(2026, 3, 1)
TRACKING_START_DATE_STR = TRACKING_START_DATE.isoformat()
DEFAULT_DAILY_TRACK_COUNT = 10
TRACKING_MIN_EDGE = 0.05
STRONG_EDGE = 0.08
ELITE_EDGE = 0.12


def normalize_name(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def tracking_bucket(edge: float) -> str:
    if pd.isna(edge):
        return ""
    if edge >= ELITE_EDGE:
        return "12%+"
    if edge >= STRONG_EDGE:
        return "8% to <12%"
    if edge >= TRACKING_MIN_EDGE:
        return "5% to <8%"
    return "Below 5%"


def add_tracking_columns(frame: pd.DataFrame) -> pd.DataFrame:
    tracked = frame.copy()
    tracked["track_edge_ge_0_05"] = tracked["edge"] >= TRACKING_MIN_EDGE
    tracked["track_edge_ge_0_08"] = tracked["edge"] >= STRONG_EDGE
    tracked["track_edge_ge_0_12"] = tracked["edge"] >= ELITE_EDGE
    tracked["tracking_bucket"] = tracked["edge"].map(tracking_bucket)
    tracked["tracking_eligible"] = tracked["edge"] >= TRACKING_MIN_EDGE
    return tracked


def daily_top_rows(frame: pd.DataFrame, top_n: int) -> pd.DataFrame:
    daily = frame.copy()
    if "rank" in daily.columns:
        daily = daily.sort_values("rank")
    if top_n > 0:
        daily = daily.head(top_n).copy()
    daily["tracked_top_rank"] = np.arange(1, len(daily) + 1)
    return daily


def append_archive(archive_path: Path, rows: pd.DataFrame) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        existing = pd.read_csv(archive_path)
    else:
        existing = pd.DataFrame(columns=rows.columns)

    combined = pd.concat([existing, rows], ignore_index=True, sort=False)
    dedupe_keys = [
        column
        for column in [
            "recommendation_date",
            "slate_date",
            "player_name",
            "side",
            "line",
            "bookmaker",
            "commence_time",
        ]
        if column in combined.columns
    ]
    if dedupe_keys:
        combined.drop_duplicates(subset=dedupe_keys, keep="last", inplace=True)
    combined.to_csv(archive_path, index=False)


def append_tracking_log(tracking_path: Path, rows: pd.DataFrame) -> None:
    # 1. Define the manual entry columns that users fill in as bets resolve.
    manual_columns = [
        "bet_tracked",
        "bet_won",
        "bet_result",
        "profit_units",
        "actual_made_3pt",
        "settled_date",
        "verified_source",
        "notes",
    ]
    # 2. Create parent directory if needed and read existing tracking log (or create empty).
    tracking_path.parent.mkdir(parents=True, exist_ok=True)
    if tracking_path.exists():
        existing = pd.read_csv(tracking_path)
    else:
        existing = pd.DataFrame(columns=[*rows.columns, *manual_columns])

    # 3. Ensure all manual columns exist in the existing log (fill NaN if missing).
    for column in manual_columns:
        if column not in existing.columns:
            existing[column] = np.nan

    # 4. Standardize slate_date to string format in both incoming and existing data.
    incoming = rows.copy()
    if "slate_date" in incoming.columns:
        incoming["slate_date"] = incoming["slate_date"].fillna("").astype(str)
    if "slate_date" in existing.columns:
        existing["slate_date"] = existing["slate_date"].fillna("").astype(str)

    # 5. Identify which slates are in the incoming data.
    incoming_slates = set(incoming["slate_date"].dropna().astype(str).tolist()) if "slate_date" in incoming.columns else set()
    if incoming_slates and "slate_date" in existing.columns:
        # 6. Mark rows as settled if they have a result or bet_won value.
        has_result_col = existing.get("bet_result", pd.Series([""] * len(existing), index=existing.index)).fillna("").astype(str).str.strip().ne("")
        has_bet_won_col = pd.to_numeric(existing.get("bet_won", np.nan), errors="coerce").notna()
        settled_mask = has_result_col | has_bet_won_col
        settled_slates = set(existing.loc[settled_mask, "slate_date"].astype(str).tolist())

        # 7. Protect settled slates from being overwritten by new recommendations.
        protected_slates = incoming_slates.intersection(settled_slates)
        if protected_slates:
            incoming = incoming[~incoming["slate_date"].isin(protected_slates)].copy()

        # 8. Remove unresolved (unsettled) rows from the same slates as incoming data.
        unresolved_same_slate = existing["slate_date"].isin(incoming_slates) & (~settled_mask)
        existing = existing.loc[~unresolved_same_slate].copy()

    # 9. Build deduplication key from bet identifiers.
    dedupe_keys = [
        column
        for column in [
            "recommendation_date",
            "slate_date",
            "player_name",
            "side",
            "line",
            "bookmaker",
            "commence_time",
        ]
        if column in incoming.columns
    ]

    # 10. Merge incoming recommendations with existing manual entries, preserving manual data.
    if dedupe_keys:
        carried_manual = existing[dedupe_keys + manual_columns].drop_duplicates(
            subset=dedupe_keys,
            keep="last",
        )
        refreshed = incoming.merge(carried_manual, on=dedupe_keys, how="left")
        existing_keys = existing[dedupe_keys].apply(tuple, axis=1)
        refreshed_keys = set(refreshed[dedupe_keys].apply(tuple, axis=1))
        unmatched_existing = existing.loc[~existing_keys.isin(refreshed_keys)].copy()
        combined = pd.concat([refreshed, unmatched_existing], ignore_index=True, sort=False)
    else:
        combined = pd.concat([incoming, existing], ignore_index=True, sort=False)

    # 11. Enforce one row per slate/rank slot (dedup on slate_date + tracked_top_rank, keep newest).
    rank_slot_keys = [column for column in ["slate_date", "tracked_top_rank"] if column in combined.columns]
    if len(rank_slot_keys) == 2:
        combined["tracked_top_rank"] = pd.to_numeric(combined["tracked_top_rank"], errors="coerce")
        sort_columns = [col for col in ["slate_date", "tracked_top_rank", "recommendation_date"] if col in combined.columns]
        sort_ascending = [True] * len(sort_columns)
        combined.sort_values(
            sort_columns,
            ascending=sort_ascending,
            inplace=True,
        )
        combined.drop_duplicates(subset=rank_slot_keys, keep="last", inplace=True)

    # 12. Write the consolidated tracking log to disk.
    combined.to_csv(tracking_path, index=False)


def build_top_n_parlay_rows(confidence_rows: pd.DataFrame, legs: int) -> pd.DataFrame:
    if confidence_rows.empty:
        return pd.DataFrame()
    if legs <= 1:
        return pd.DataFrame()

    top_n = confidence_rows.sort_values("rank").head(legs).copy()
    if len(top_n) < legs:
        return pd.DataFrame()

    decimal_product = pd.to_numeric(top_n["decimal_odds"], errors="coerce").prod()
    if pd.isna(decimal_product):
        return pd.DataFrame()

    first = top_n.iloc[0]
    row = {
        "recommendation_date": first.get("recommendation_date"),
        "slate_date": first.get("slate_date"),
        "legs": int(legs),
        "parlay_decimal_odds": float(decimal_product),
        "parlay_american_display": f"{((decimal_product - 1.0) * 100):+.0f}" if decimal_product >= 2 else "",
        "parlay_hit": np.nan,
        "bet_result": "",
        "profit_units": np.nan,
        "settled_date": np.nan,
        "verified_source": np.nan,
    }
    for leg_idx in range(1, legs + 1):
        leg_row = top_n.iloc[leg_idx - 1]
        row[f"leg_{leg_idx}"] = f"{leg_row.get('player_name')} {leg_row.get('side')} {leg_row.get('line')}"
    return pd.DataFrame([row])


def build_top3_parlay_rows(confidence_rows: pd.DataFrame) -> pd.DataFrame:
    return build_top_n_parlay_rows(confidence_rows, legs=3)


def build_top5_parlay_rows(confidence_rows: pd.DataFrame) -> pd.DataFrame:
    return build_top_n_parlay_rows(confidence_rows, legs=5)


def append_parlay_log(parlay_path: Path, rows: pd.DataFrame) -> None:
    parlay_path.parent.mkdir(parents=True, exist_ok=True)
    if parlay_path.exists():
        existing = pd.read_csv(parlay_path)
    else:
        existing = pd.DataFrame(columns=rows.columns)

    combined = pd.concat([existing, rows], ignore_index=True, sort=False)
    dedupe_keys = [column for column in ["recommendation_date", "slate_date"] if column in combined.columns]
    if dedupe_keys:
        combined.drop_duplicates(subset=dedupe_keys, keep="last", inplace=True)
    combined.to_csv(parlay_path, index=False)


def calculate_profit_units(decimal_odds: float, result: str) -> float:
    if result == "WIN":
        return float(decimal_odds - 1.0)
    if result == "LOSS":
        return -1.0
    return 0.0


def season_from_slate_date(slate_date: str) -> str:
    year, month, _ = [int(part) for part in slate_date.split("-")]
    start_year = year if month >= 10 else year - 1
    end_year = (start_year + 1) % 100
    return f"{start_year}-{end_year:02d}"


def fetch_actual_makes_from_nba_api(player_name: str, slate_date: str) -> float | None:
    try:
        from nba_api.stats.endpoints import playergamelog
        from nba_api.stats.static import players
    except Exception:
        return None

    matches = players.find_players_by_full_name(player_name)
    if not matches:
        return None

    exact_match = next(
        (item for item in matches if normalize_name(item.get("full_name")) == normalize_name(player_name)),
        matches[0],
    )
    player_id = exact_match.get("id")
    if player_id is None:
        return None

    try:
        logs = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season_from_slate_date(slate_date),
        ).get_data_frames()
    except Exception:
        return None

    if not logs:
        return None

    frame = logs[0].copy()
    if frame.empty or "GAME_DATE" not in frame.columns or "FG3M" not in frame.columns:
        return None

    frame["GAME_DATE"] = pd.to_datetime(frame["GAME_DATE"], errors="coerce").dt.date.astype(str)
    exact_rows = frame[frame["GAME_DATE"] == slate_date]
    if exact_rows.empty:
        return None

    actual_made = pd.to_numeric(exact_rows.iloc[0]["FG3M"], errors="coerce")
    if pd.isna(actual_made):
        return None
    return float(actual_made)


def settle_tracked_bets_from_history(tracking_path: Path, history_path: Path) -> bool:
    # 1. Check that both tracking and history files exist.
    if not tracking_path.exists() or not history_path.exists():
        return False

    # 2. Load tracked bets and player game history from CSV.
    tracked = pd.read_csv(tracking_path)
    history = pd.read_csv(history_path)
    if tracked.empty or history.empty:
        return False

    # 3. Verify required columns exist in both datasets.
    required_tracked = {"player_name", "slate_date", "side", "line", "decimal_odds"}
    required_history = {"player_name", "game_date", "player_made_3pt"}
    if not required_tracked.issubset(tracked.columns) or not required_history.issubset(history.columns):
        return False

    # 4. Add missing result/settlement columns to the tracked bets dataframe.
    for column in ["bet_won", "bet_result", "profit_units", "actual_made_3pt", "settled_date", "verified_source"]:
        if column not in tracked.columns:
            tracked[column] = np.nan
    # 5. Ensure text fields accept string writes even when CSV loaded as float NaN columns.
    for text_col in ["bet_result", "settled_date", "verified_source"]:
        tracked[text_col] = tracked[text_col].astype(object)

    # 6. Build a normalized lookup table from game history: (player_key, game_date) -> made_3pt.
    history_lookup = history.copy()
    history_lookup["player_key"] = history_lookup["player_name"].map(normalize_name)
    history_lookup["game_date"] = history_lookup["game_date"].astype(str)
    history_lookup["player_made_3pt"] = pd.to_numeric(history_lookup["player_made_3pt"], errors="coerce")
    history_lookup = history_lookup.dropna(subset=["player_made_3pt"])
    history_lookup.sort_values(["game_date"], inplace=True)
    history_lookup = history_lookup.drop_duplicates(subset=["player_key", "game_date"], keep="last")
    lookup = history_lookup.set_index(["player_key", "game_date"])["player_made_3pt"].to_dict()
    api_lookup: dict[tuple[str, str], float | None] = {}

    # 7. Flag to track if any rows were modified.
    changed = False
    # 8. Iterate through each tracked bet and attempt to settle it.
    for idx, row in tracked.iterrows():
        # 9. Extract slate date and skip if before tracking start date or already settled.
        slate_date = str(row.get("slate_date", "") or "")
        if not slate_date or slate_date < TRACKING_START_DATE_STR:
            continue

        existing_raw_result = row.get("bet_result", "")
        existing_result = "" if pd.isna(existing_raw_result) else str(existing_raw_result).strip().upper()
        bet_won = row.get("bet_won")
        if existing_result or pd.notna(bet_won):
            continue

        # 10. Look up actual made 3PTs from history CSV (normalized player key).
        key = (normalize_name(row.get("player_name")), slate_date)
        actual_made = lookup.get(key)
        verified_source = "history_csv"
        # 11. If not found in history, try fetching from NBA API and cache result.
        if actual_made is None or pd.isna(actual_made):
            api_key = (str(row.get("player_name", "")), slate_date)
            if api_key not in api_lookup:
                api_lookup[api_key] = fetch_actual_makes_from_nba_api(api_key[0], api_key[1])
            actual_made = api_lookup[api_key]
            verified_source = "nba_api"
        # 12. Skip if actual made value is still unavailable.
        if actual_made is None or pd.isna(actual_made):
            continue

        # 13. Extract and validate line and decimal odds.
        line = pd.to_numeric(pd.Series([row.get("line")]), errors="coerce").iloc[0]
        decimal_odds = pd.to_numeric(pd.Series([row.get("decimal_odds")]), errors="coerce").iloc[0]
        if pd.isna(line) or pd.isna(decimal_odds):
            continue

        # 14. Determine bet result: compare actual_made against line and side (OVER/UNDER).
        side = str(row.get("side", "") or "").upper()
        if actual_made > line:
            result = "WIN" if side == "OVER" else "LOSS"
        elif actual_made < line:
            result = "WIN" if side == "UNDER" else "LOSS"
        else:
            result = "PUSH"

        # 15. Record actual outcome, result, settled date, and source.
        tracked.at[idx, "actual_made_3pt"] = float(actual_made)
        tracked.at[idx, "bet_result"] = result
        tracked.at[idx, "settled_date"] = slate_date
        tracked.at[idx, "verified_source"] = verified_source
        # 16. Calculate profit units based on result and decimal odds.
        if result == "PUSH":
            tracked.at[idx, "bet_won"] = np.nan
            tracked.at[idx, "profit_units"] = 0.0
        else:
            tracked.at[idx, "bet_won"] = 1 if result == "WIN" else 0
            tracked.at[idx, "profit_units"] = calculate_profit_units(float(decimal_odds), result)
        changed = True

    # 17. Write updated tracking log if any bets were settled.
    if changed:
        tracked.to_csv(tracking_path, index=False)
    return changed


def settle_top_n_parlays(parlay_path: Path, confidence_tracking_path: Path, legs: int) -> bool:
    # 1. Verify both parlay and confidence tracking files exist and legs > 1.
    if not parlay_path.exists() or not confidence_tracking_path.exists():
        return False
    if legs <= 1:
        return False

    # 2. Load parlay records and confidence tracking (settled individual bets).
    parlays = pd.read_csv(parlay_path)
    confidence = pd.read_csv(confidence_tracking_path)
    if parlays.empty or confidence.empty:
        return False

    # 3. Verify confidence tracking has the ranked top-N column.
    if "tracked_top_rank" not in confidence.columns:
        return False
    # 4. Ensure text columns exist and are object type for string assignment.
    for text_col in ["bet_result", "settled_date", "verified_source"]:
        if text_col not in parlays.columns:
            parlays[text_col] = np.nan
        parlays[text_col] = parlays[text_col].astype(object)

    # 5. Flag to track if any parlay rows were modified.
    changed = False
    # 6. Iterate through each parlay record and attempt to settle it.
    for idx, row in parlays.iterrows():
        # 7. Extract slate date and skip if empty or already settled.
        slate_date = str(row.get("slate_date", "") or "")
        if not slate_date:
            continue
        existing_raw_result = row.get("bet_result", "")
        existing_result = "" if pd.isna(existing_raw_result) else str(existing_raw_result).strip().upper()
        if existing_result:
            continue

        # 8. Get the individual legs (top-N ranked bets) for this slate.
        leg_rows = confidence[
            (confidence["slate_date"].fillna("").astype(str) == slate_date)
            & (pd.to_numeric(confidence["tracked_top_rank"], errors="coerce") <= legs)
        ].copy()
        # 9. Skip if insufficient legs or missing bet_result column.
        if len(leg_rows) < legs or "bet_result" not in leg_rows.columns:
            continue

        # 10. Extract individual leg results in rank order.
        leg_results = leg_rows.sort_values("tracked_top_rank")["bet_result"].fillna("").astype(str).str.upper().tolist()
        # 11. Skip if any leg is still unsettled (empty result).
        if any(result == "" for result in leg_results):
            continue

        # 12. Determine overall parlay result based on individual leg outcomes.
        if any(result == "LOSS" for result in leg_results):
            # If any leg loses, the whole parlay loses.
            result = "LOSS"
            parlay_hit = 0
            profit_units = -1.0
        elif all(result == "WIN" for result in leg_results):
            # If all legs win, the parlay wins.
            result = "WIN"
            parlay_hit = 1
            decimal_odds = pd.to_numeric(pd.Series([row.get("parlay_decimal_odds")]), errors="coerce").iloc[0]
            profit_units = calculate_profit_units(float(decimal_odds), result) if pd.notna(decimal_odds) else np.nan
        else:
            # If any leg pushes (but no losses), the parlay pushes.
            result = "PUSH"
            parlay_hit = np.nan
            profit_units = 0.0

        # 13. Record parlay result and settlement details.
        parlays.at[idx, "parlay_hit"] = parlay_hit
        parlays.at[idx, "bet_result"] = result
        parlays.at[idx, "profit_units"] = profit_units
        parlays.at[idx, "settled_date"] = slate_date
        parlays.at[idx, "verified_source"] = "confidence_tracking"
        changed = True

    # 14. Write updated parlay log if any records were settled.
    if changed:
        parlays.to_csv(parlay_path, index=False)
    return changed


def settle_top3_parlays(parlay_path: Path, confidence_tracking_path: Path) -> bool:
    return settle_top_n_parlays(parlay_path, confidence_tracking_path, legs=3)


def settle_top5_parlays(parlay_path: Path, confidence_tracking_path: Path) -> bool:
    return settle_top_n_parlays(parlay_path, confidence_tracking_path, legs=5)
