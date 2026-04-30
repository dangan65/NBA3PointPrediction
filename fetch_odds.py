"""
Fetch NBA game odds and player prop lines from The Odds API.

Pulls game spreads, totals, and player threes lines from The Odds API and writes
CSV overlays: game_context_odds.csv (spread/total keyed by team) and
player_threes_props.csv (player prop lines). Includes throttle/state logic to avoid
over-pulling the API.

Inputs:
  - The Odds API key (--api-key or ODDS_API_KEY environment variable)
  - Command-line flags for regions, odds-format, throttle behavior, output paths
  - Optional: max-events cap, game-time filters, bookmaker filters

Outputs:
  - game_context_odds.csv: spread, game_total, odds_event_id, bookmakers per team/opponent
  - player_threes_props.csv: player name, line, American/decimal odds, implied probs, market hold
  - odds_api_pull_state.json: tracks last successful pull time for throttle window
"""

from __future__ import annotations

import argparse
import os
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests


BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_KEY = "basketball_nba"
DEFAULT_BOOKS = {"draftkings", "fanduel", "betmgm", "caesars"}
DEFAULT_STATE_PATH = Path("data/odds_api_pull_state.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch NBA game and player-threes odds from The Odds API and write CSV overlays."
    )
    parser.add_argument(
        "--api-key",
        help="The Odds API key. Defaults to the ODDS_API_KEY environment variable.",
    )
    parser.add_argument(
        "--regions",
        default="us",
        help="Bookmaker regions to request (for example us or us,us2).",
    )
    parser.add_argument(
        "--odds-format",
        choices=["american", "decimal"],
        default="american",
        help="Requested odds format.",
    )
    parser.add_argument(
        "--bookmakers",
        help="Optional comma-separated bookmaker keys to keep. Defaults to a common US subset.",
    )
    parser.add_argument(
        "--game-context-out",
        type=Path,
        default=Path("data/game_context_odds.csv"),
        help="Where to write spreads/totals keyed by team and opponent.",
    )
    parser.add_argument(
        "--player-props-out",
        type=Path,
        default=Path("data/player_threes_props.csv"),
        help="Where to write player threes prop lines.",
    )
    parser.add_argument(
        "--only-if-game-within-hours",
        type=float,
        default=None,
        help="Skip the pull unless at least one game starts within this many hours from now.",
    )
    parser.add_argument(
        "--empty-ok",
        action="store_true",
        help="Write empty CSVs when no games qualify instead of exiting early.",
    )
    parser.add_argument(
        "--only-if-first-game-within-hours",
        type=float,
        default=None,
        help="Skip the pull unless the first game of the upcoming slate starts within this many hours.",
    )
    parser.add_argument(
        "--min-hours-between-pulls",
        type=float,
        default=1.0,
        help="Minimum time between successful pulls. Set 0 to disable throttling.",
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        default=DEFAULT_STATE_PATH,
        help="Path to the JSON file used to track the last successful pull time.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore the throttle window and pull immediately.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Optional cap on how many event-level prop requests to make.",
    )
    return parser.parse_args()


def require_api_key(explicit_key: str | None) -> str:
    api_key = explicit_key or os.getenv("ODDS_API_KEY")
    if not api_key:
        raise RuntimeError("Set ODDS_API_KEY or pass --api-key.")
    return api_key


def get_json(url: str, params: dict[str, str]) -> list[dict] | dict:
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def load_last_pull_time(path: Path) -> datetime | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    timestamp = payload.get("last_successful_pull_utc")
    if not isinstance(timestamp, str):
        return None
    return parse_commence_time(timestamp)


def save_last_pull_time(path: Path, pulled_at: datetime) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"last_successful_pull_utc": pulled_at.astimezone(timezone.utc).isoformat()}
    path.write_text(json.dumps(payload, indent=2))


def throttle_window_open(path: Path, min_hours_between_pulls: float, force: bool) -> tuple[bool, str | None]:
    if force or min_hours_between_pulls <= 0:
        return True, None

    last_pull = load_last_pull_time(path)
    if last_pull is None:
        return True, None

    now = datetime.now(timezone.utc)
    next_allowed = last_pull + timedelta(hours=min_hours_between_pulls)
    if now >= next_allowed:
        return True, None

    return (
        False,
        f"Last successful pull was at {last_pull.isoformat()}. Next allowed pull is {next_allowed.isoformat()}.",
    )


def american_to_decimal(price: float | int | None) -> float | None:
    if price is None:
        return None
    if price > 0:
        return 1.0 + (price / 100.0)
    if price < 0:
        return 1.0 + (100.0 / abs(price))
    return None


def implied_probability_from_decimal(decimal_odds: float | None) -> float | None:
    if decimal_odds is None or decimal_odds <= 0:
        return None
    return 1.0 / decimal_odds


def side_label(value: str | None) -> str:
    text = (value or "").strip().lower()
    if "over" in text:
        return "over"
    if "under" in text:
        return "under"
    return text


def choose_bookmakers(bookmakers: list[dict], requested: str | None) -> list[dict]:
    if requested:
        allowed = {item.strip() for item in requested.split(",") if item.strip()}
    else:
        allowed = DEFAULT_BOOKS

    filtered = [book for book in bookmakers if book.get("key") in allowed]
    return filtered or bookmakers


def market_by_key(bookmakers: list[dict], market_key: str) -> tuple[dict | None, dict | None]:
    best_book = None
    best_market = None

    for bookmaker in bookmakers:
        for market in bookmaker.get("markets", []):
            if market.get("key") == market_key:
                return bookmaker, market
        if best_book is None and bookmaker.get("markets"):
            best_book = bookmaker
            best_market = bookmaker["markets"][0]

    return best_book, best_market


def extract_spread_total_rows(event: dict, bookmakers: list[dict]) -> list[dict]:
    spread_book, spread_market = market_by_key(bookmakers, "spreads")
    total_book, total_market = market_by_key(bookmakers, "totals")
    commence_time = event.get("commence_time")
    commence_dt = parse_commence_time(commence_time)
    slate_date_utc = commence_dt.date().isoformat() if commence_dt is not None else ""

    spread_by_team: dict[str, float | None] = {
        event["home_team"]: None,
        event["away_team"]: None,
    }
    total_point = None

    if spread_market:
        for outcome in spread_market.get("outcomes", []):
            spread_by_team[outcome.get("name")] = outcome.get("point")

    if total_market and total_market.get("outcomes"):
        total_point = total_market["outcomes"][0].get("point")

    rows = []
    for team, opponent in [
        (event["home_team"], event["away_team"]),
        (event["away_team"], event["home_team"]),
    ]:
        rows.append(
            {
                "team": team,
                "opponent": opponent,
                "spread": spread_by_team.get(team),
                "game_total": total_point,
                "odds_event_id": event.get("id"),
                "event_home_team": event.get("home_team"),
                "event_away_team": event.get("away_team"),
                "commence_time": commence_time,
                "slate_date_utc": slate_date_utc,
                "bookmaker_spread": None if spread_book is None else spread_book.get("key"),
                "bookmaker_total": None if total_book is None else total_book.get("key"),
            }
        )
    return rows


def fetch_games(api_key: str, regions: str, odds_format: str) -> list[dict]:
    url = f"{BASE_URL}/sports/{SPORT_KEY}/odds"
    payload = get_json(
        url,
        {
            "apiKey": api_key,
            "regions": regions,
            "markets": "spreads,totals",
            "oddsFormat": odds_format,
        },
    )
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected response from odds endpoint.")
    return payload


def fetch_events(api_key: str) -> list[dict]:
    url = f"{BASE_URL}/sports/{SPORT_KEY}/events"
    payload = get_json(url, {"apiKey": api_key})
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected response from events endpoint.")
    return payload


def parse_commence_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def filter_events_within_window(events: list[dict], hours: float | None) -> list[dict]:
    if hours is None:
        return events

    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=hours)
    filtered: list[dict] = []

    for event in events:
        commence_time = parse_commence_time(event.get("commence_time"))
        if commence_time is None:
            continue
        if now <= commence_time <= cutoff:
            filtered.append(event)

    return filtered


def first_slate_events(events: list[dict]) -> list[dict]:
    timed_events: list[tuple[datetime, dict]] = []
    for event in events:
        commence_time = parse_commence_time(event.get("commence_time"))
        if commence_time is None:
            continue
        timed_events.append((commence_time, event))

    if not timed_events:
        return []

    timed_events.sort(key=lambda item: item[0])
    first_time = timed_events[0][0]
    first_date = first_time.date()
    return [event for commence_time, event in timed_events if commence_time.date() == first_date]


def first_game_window_open(events: list[dict], hours: float | None) -> bool:
    if hours is None:
        return True

    slate = first_slate_events(events)
    if not slate:
        return False

    first_time = min(
        parse_commence_time(event.get("commence_time")) for event in slate if parse_commence_time(event.get("commence_time")) is not None
    )
    if first_time is None:
        return False

    now = datetime.now(timezone.utc)
    open_time = first_time - timedelta(hours=hours)
    return open_time <= now <= first_time


def fetch_event_odds(api_key: str, event_id: str, regions: str, odds_format: str) -> dict:
    url = f"{BASE_URL}/sports/{SPORT_KEY}/events/{event_id}/odds"
    payload = get_json(
        url,
        {
            "apiKey": api_key,
            "regions": regions,
            "markets": "player_threes,player_threes_alternate",
            "oddsFormat": odds_format,
        },
    )
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected response from event-odds endpoint for {event_id}.")
    return payload


def extract_player_prop_rows(event: dict, bookmakers: list[dict], odds_format: str) -> list[dict]:
    grouped: dict[tuple[str, str, str, float | None, str], dict[str, object]] = {}
    target_markets = {"player_threes", "player_threes_alternate"}

    for bookmaker in bookmakers:
        for market in bookmaker.get("markets", []):
            if market.get("key") not in target_markets:
                continue
            for outcome in market.get("outcomes", []):
                price = outcome.get("price")
                decimal_odds = price if odds_format == "decimal" else american_to_decimal(price)
                implied_prob = implied_probability_from_decimal(decimal_odds)
                player_name = outcome.get("description") or outcome.get("name")
                side = side_label(outcome.get("name"))
                key = (
                    str(player_name),
                    str(event.get("home_team")),
                    str(event.get("away_team")),
                    outcome.get("point"),
                    str(bookmaker.get("key")),
                )

                record = grouped.setdefault(
                    key,
                    {
                        "player_name": player_name,
                        # Team/opponent are resolved later from player history + event teams.
                        "team": None,
                        "opponent": None,
                        "event_home_team": event.get("home_team"),
                        "event_away_team": event.get("away_team"),
                        "market_key": market.get("key"),
                        "line": outcome.get("point"),
                        "bookmaker": bookmaker.get("key"),
                        "odds_event_id": event.get("id"),
                        "commence_time": event.get("commence_time"),
                        "over_american_odds": None,
                        "under_american_odds": None,
                        "over_decimal_odds": None,
                        "under_decimal_odds": None,
                        "over_implied_prob": None,
                        "under_implied_prob": None,
                    },
                )

                if side == "over":
                    record["over_american_odds"] = price if odds_format == "american" else None
                    record["over_decimal_odds"] = decimal_odds
                    record["over_implied_prob"] = implied_prob
                elif side == "under":
                    record["under_american_odds"] = price if odds_format == "american" else None
                    record["under_decimal_odds"] = decimal_odds
                    record["under_implied_prob"] = implied_prob

    rows = list(grouped.values())
    for row in rows:
        over_prob = row["over_implied_prob"]
        under_prob = row["under_implied_prob"]
        if over_prob is not None and under_prob is not None and (over_prob + under_prob) > 0:
            total = over_prob + under_prob
            row["market_hold"] = total - 1.0
            row["no_vig_over_prob"] = over_prob / total
            row["no_vig_under_prob"] = under_prob / total
        else:
            row["market_hold"] = None
            row["no_vig_over_prob"] = None
            row["no_vig_under_prob"] = None
    return rows


def main() -> None:
    # 1. Parse command-line arguments and validate API key
    args = parse_args()
    api_key = require_api_key(args.api_key)

    # 2. Check throttle window to avoid over-pulling the API
    allowed, reason = throttle_window_open(args.state_path, args.min_hours_between_pulls, args.force)
    if not allowed:
        print(f"Skipping odds pull. {reason}")
        return

    # 3. Fetch upcoming games and validate against first-game window
    games = fetch_games(api_key, args.regions, args.odds_format)
    if not first_game_window_open(games, args.only_if_first_game_within_hours):
        if args.empty_ok:
            args.game_context_out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame().to_csv(args.game_context_out, index=False)
            args.player_props_out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame().to_csv(args.player_props_out, index=False)
        print("Outside the first-game window for the upcoming slate. Skipping odds pull.")
        return

    # 4. Filter games by time window (first slate or within N hours)
    if args.only_if_first_game_within_hours is not None:
        games = first_slate_events(games)
    else:
        games = filter_events_within_window(games, args.only_if_game_within_hours)

    if not games and not args.empty_ok:
        print("No NBA games start within the requested time window. Skipping odds pull.")
        return

    # 5. Extract spread and total lines, write game_context_odds.csv
    game_rows: list[dict] = []
    for event in games:
        bookmakers = choose_bookmakers(event.get("bookmakers", []), args.bookmakers)
        game_rows.extend(extract_spread_total_rows(event, bookmakers))

    game_context = pd.DataFrame(game_rows)
    args.game_context_out.parent.mkdir(parents=True, exist_ok=True)
    game_context.to_csv(args.game_context_out, index=False)

    # 6. Fetch all upcoming events and apply time/max-events filters
    events = fetch_events(api_key)
    if args.only_if_first_game_within_hours is not None:
        events = first_slate_events(events)
    else:
        events = filter_events_within_window(events, args.only_if_game_within_hours)
    if args.max_events is not None:
        events = events[: max(0, args.max_events)]

    # 7. Fetch player prop lines for each event (with rate-limit handling)
    prop_rows: list[dict] = []
    rate_limited = False
    for event in events:
        event_id = event.get("id")
        if not event_id:
            continue
        try:
            detailed_event = fetch_event_odds(api_key, event_id, args.regions, args.odds_format)
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code == 429:
                rate_limited = True
                print(
                    "Hit The Odds API rate limit while fetching player props. "
                    "Saving the partial results collected so far."
                )
                break
            raise
        bookmakers = choose_bookmakers(detailed_event.get("bookmakers", []), args.bookmakers)
        prop_rows.extend(extract_player_prop_rows(detailed_event, bookmakers, args.odds_format))

    # 8. Write player_threes_props.csv and update throttle state (if pull was complete)
    player_props = pd.DataFrame(prop_rows)
    args.player_props_out.parent.mkdir(parents=True, exist_ok=True)
    player_props.to_csv(args.player_props_out, index=False)

    if not rate_limited:
        save_last_pull_time(args.state_path, datetime.now(timezone.utc))

    print(f"Wrote {len(game_context)} game-context rows to {args.game_context_out}")
    print(f"Wrote {len(player_props)} player-prop rows to {args.player_props_out}")
    if rate_limited:
        print("Player props were truncated due to API rate limiting.")
        print("Throttle state was not updated because the pull was incomplete.")


if __name__ == "__main__":
    main()
