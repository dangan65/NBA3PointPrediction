"""
Build a team defense overlay CSV from NBA API for daily 3pt props data pipeline.

Module Purpose:
  Pull team defensive statistics from stats.nba.com via nba_api, construct team defense metrics,
  and output a template CSV used by update_features.py to augment player-level data.

Inputs:
  - Season code (e.g., '2025-26')
  - Season type ('Regular Season', etc.)
  - nba_api endpoint access

Outputs:
  - CSV with team abbreviation and defensive statistics (3PA allowed, 3PM allowed, pace, defense rating, scheme tendency)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = [
    "opponent",
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a team-defense overlay CSV from nba_api for update_features.py."
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
        "--output",
        type=Path,
        default=Path("data/team_defense_template.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def first_available(frame: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for col in candidates:
        if col in frame.columns:
            return to_numeric(frame[col])
    return pd.Series([np.nan] * len(frame), index=frame.index)


def call_team_stats(season: str, season_type: str, measure: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguedashteamstats

    endpoint = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star=season_type,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense=measure,
    )
    frames = endpoint.get_data_frames()
    return frames[0] if frames else pd.DataFrame()


def team_id_to_abbreviation_map() -> dict[int, str]:
    from nba_api.stats.static import teams

    mapping: dict[int, str] = {}
    for row in teams.get_teams():
        try:
            team_id = int(row.get("id"))
        except Exception:
            continue
        abbr = str(row.get("abbreviation") or "").strip().upper()
        if abbr:
            mapping[team_id] = abbr
    return mapping


def resolve_team_abbreviations(frame: pd.DataFrame) -> pd.Series:
    if "TEAM_ABBREVIATION" in frame.columns:
        return frame["TEAM_ABBREVIATION"].fillna("").astype(str).str.upper()

    if "TEAM_ID" in frame.columns:
        id_map = team_id_to_abbreviation_map()
        team_ids = pd.to_numeric(frame["TEAM_ID"], errors="coerce")
        mapped = team_ids.map(lambda value: id_map.get(int(value), "") if pd.notna(value) else "")
        if (mapped.astype(str).str.len() > 0).any():
            return mapped.astype(str).str.upper()

    # Fallback using team names if abbreviation and ID mapping both fail.
    if "TEAM_NAME" in frame.columns:
        name_to_abbr = {
            "ATLANTA HAWKS": "ATL",
            "BOSTON CELTICS": "BOS",
            "BROOKLYN NETS": "BKN",
            "CHARLOTTE HORNETS": "CHA",
            "CHICAGO BULLS": "CHI",
            "CLEVELAND CAVALIERS": "CLE",
            "DALLAS MAVERICKS": "DAL",
            "DENVER NUGGETS": "DEN",
            "DETROIT PISTONS": "DET",
            "GOLDEN STATE WARRIORS": "GSW",
            "HOUSTON ROCKETS": "HOU",
            "INDIANA PACERS": "IND",
            "LA CLIPPERS": "LAC",
            "LOS ANGELES LAKERS": "LAL",
            "MEMPHIS GRIZZLIES": "MEM",
            "MIAMI HEAT": "MIA",
            "MILWAUKEE BUCKS": "MIL",
            "MINNESOTA TIMBERWOLVES": "MIN",
            "NEW ORLEANS PELICANS": "NOP",
            "NEW YORK KNICKS": "NYK",
            "OKLAHOMA CITY THUNDER": "OKC",
            "ORLANDO MAGIC": "ORL",
            "PHILADELPHIA 76ERS": "PHI",
            "PHOENIX SUNS": "PHX",
            "PORTLAND TRAIL BLAZERS": "POR",
            "SACRAMENTO KINGS": "SAC",
            "SAN ANTONIO SPURS": "SAS",
            "TORONTO RAPTORS": "TOR",
            "UTAH JAZZ": "UTA",
            "WASHINGTON WIZARDS": "WAS",
        }
        names = frame["TEAM_NAME"].fillna("").astype(str).str.upper().str.strip()
        return names.map(lambda value: name_to_abbr.get(value, ""))

    return pd.Series([""] * len(frame), index=frame.index)


def derive_scheme(frame: pd.DataFrame) -> pd.Series:
    # Heuristic label so model gets a usable categorical defensive tendency.
    opp_3pa = to_numeric(frame.get("opp_3pa_allowed", pd.Series([np.nan] * len(frame), index=frame.index)))
    if opp_3pa.notna().sum() == 0:
        return pd.Series(["BALANCED"] * len(frame), index=frame.index)
    low = float(opp_3pa.quantile(0.33))
    high = float(opp_3pa.quantile(0.67))
    return opp_3pa.map(
        lambda v: "LOW_3PT_ALLOWED"
        if pd.notna(v) and v <= low
        else ("HIGH_3PT_ALLOWED" if pd.notna(v) and v >= high else "BALANCED")
    )


def main() -> None:
    args = parse_args()

    # 1. Pull base and advanced team defense stats from nba_api
    try:
        base = call_team_stats(args.season, args.season_type, "Base")
        adv = call_team_stats(args.season, args.season_type, "Advanced")
    except Exception as exc:
        raise RuntimeError(
            "Failed to pull team defense from stats.nba.com via nba_api. "
            "Check internet/DNS access and retry."
        ) from exc

    # 2. Validate that at least one endpoint returned data
    if base.empty and adv.empty:
        raise RuntimeError("nba_api returned no team stats rows.")

    # 3. Merge base and advanced stats on common keys
    if base.empty:
        frame = adv.copy()
    elif adv.empty:
        frame = base.copy()
    else:
        merge_keys = [k for k in ["TEAM_ID", "TEAM_ABBREVIATION"] if k in base.columns and k in adv.columns]
        if not merge_keys:
            frame = base.copy()
        else:
            frame = base.merge(adv, on=merge_keys, how="left", suffixes=("", "_adv"))

    # 4. Resolve team abbreviations from various column sources
    opponent = resolve_team_abbreviations(frame)

    # 5. Extract 3-point and pace stats, create output dataframe
    out = pd.DataFrame(
        {
            "opponent": opponent,
            "opp_3pa_allowed": first_available(frame, ["OPP_FG3A", "FG3A", "DFG3A"]),
            "opp_3pm_allowed": first_available(frame, ["OPP_FG3M", "FG3M", "DFG3M"]),
            "opp_def_rating": first_available(frame, ["DEF_RATING", "DEF_RATING_adv"]),
            "opp_pace": first_available(frame, ["PACE", "PACE_adv"]),
            "opp_position_3pt_rank": np.nan,
            "opp_3pa_allowed_by_position": np.nan,
            "opp_team_wide_open_3pa_allowed": np.nan,
            "opp_team_above_break_3pa_allowed": np.nan,
            "opp_team_corner_3pa_allowed": np.nan,
            "corner_3_allowed": np.nan,
            "above_break_3_allowed": np.nan,
            "defensive_scheme_tendency": "BALANCED",
        }
    )

    # 6. Filter, derive defensive scheme tendency, drop duplicates, sort and reindex
    out = out[out["opponent"].str.len() > 0].copy()
    out["defensive_scheme_tendency"] = derive_scheme(out)
    out = out[OUTPUT_COLUMNS].drop_duplicates(subset=["opponent"]).sort_values("opponent").reset_index(drop=True)

    # 7. Write team defense template to output CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} team-defense rows to {args.output}")
    print(f"Season: {args.season}")
    print(f"Season type: {args.season_type}")


if __name__ == "__main__":
    main()
