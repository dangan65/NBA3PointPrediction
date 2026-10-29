#!/bin/bash
# NBA 3PT daily pipeline — runs locally because stats.nba.com blocks
# GitHub-hosted runners. Scheduled by launchd via install_schedule.sh.
#
# The Odds API key is read from ODDS_API_KEY, or from the macOS Keychain:
#   security add-generic-password -a "$USER" -s ODDS_API_KEY -w

set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
PY="$DIR/.venv/bin/python"

echo "=== $(date) ==="

# ---------------------------------------------------------------
# 1. Sync with GitHub
# ---------------------------------------------------------------
git pull --rebase --autostash origin main

# ---------------------------------------------------------------
# 2. Slate date, season and season type (US Eastern)
# ---------------------------------------------------------------
SLATE="${1:-$(TZ='America/New_York' date +%Y-%m-%d)}"
year=${SLATE:0:4}
month=$((10#${SLATE:5:2}))
day=$((10#${SLATE:8:2}))

# A season starts in October: Oct-Dec belong to YYYY-(YY+1),
# Jan-Sep belong to (YYYY-1)-YY.
if [ "$month" -ge 10 ]; then
  start=$year
else
  start=$((year - 1))
fi
SEASON="${start}-$(printf '%02d' $(( (start + 1) % 100 )))"

# April 19 through the Finals is playoffs; Jul-Sep has no games.
if [ "$month" -ge 5 ] && [ "$month" -le 9 ] || ([ "$month" -eq 4 ] && [ "$day" -ge 19 ]); then
  SEASON_TYPE="Playoffs"
else
  SEASON_TYPE="Regular Season"
fi
echo "Slate: $SLATE  Season: $SEASON  Type: $SEASON_TYPE"

# ---------------------------------------------------------------
# 3. Fetch today's odds from The Odds API
# ---------------------------------------------------------------
if [ -z "${ODDS_API_KEY:-}" ]; then
  ODDS_API_KEY="$(security find-generic-password -a "$USER" -s ODDS_API_KEY -w 2>/dev/null || true)"
fi
if [ -z "$ODDS_API_KEY" ]; then
  echo "ODDS_API_KEY not set and not found in Keychain." >&2
  exit 1
fi
export ODDS_API_KEY

"$PY" fetch_odds.py --odds-format american --regions us --empty-ok --force

# ---------------------------------------------------------------
# 4. Update player features, rank props, refresh catalog
# ---------------------------------------------------------------
"$PY" update_features.py --season "$SEASON" --season-type "$SEASON_TYPE"

"$PY" rank_props.py \
  --history data/player_game_3pt_features.csv \
  --model   models/three_pt_model.joblib \
  --count-model models/three_pt_count_model.joblib \
  --slate-date "$SLATE"

"$PY" backfill_catalog.py \
  --features data/player_game_3pt_features.csv \
  --output   data/retroactive_prediction_catalog.csv

# ---------------------------------------------------------------
# 5. Commit updated outputs back to the repo
# ---------------------------------------------------------------
git add data/*.csv data/odds_api_pull_state.json
if git diff --cached --quiet; then
  echo "No data changes — nothing to commit."
else
  git commit -m "chore: daily pipeline update $SLATE"
  git push origin main
fi
