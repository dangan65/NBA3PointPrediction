# NBA 3PT Prop Predictor

A machine learning pipeline that predicts NBA player 3-point prop outcomes and surfaces the highest-confidence picks each day through an interactive Streamlit dashboard.

Pulls live odds from The Odds API, scores every player using a trained binary classifier (P over 1.5 threes) and a Poisson count model (expected 3PM), and ranks picks by model confidence and edge vs. sportsbook lines. Season-long prediction history, parlay hit rates, and a confidence board are all tracked automatically.

---

## Quick start

You only need Python 3.11+ installed. Everything else is handled automatically.

```bash
git clone https://github.com/YOUR_USER/nba-3pt-props.git
cd nba-3pt-props
./run.sh
```

`run.sh` will create a local virtual environment, install all dependencies from `requirements.txt`, and launch the dashboard — all in one step. On subsequent runs it just launches the app.

> **Windows users:** run these manually instead:
> ```
> python -m venv .venv
> .venv\Scripts\pip install -r requirements.txt
> .venv\Scripts\streamlit run app.py
> ```

---

## Daily pipeline

Run these in order each morning to refresh today's predictions:

```bash
# 1. Fetch today's odds
python fetch_odds.py --api-key YOUR_ODDS_API_KEY --odds-format american --regions us --empty-ok

# 2. Update player features with yesterday's results
python update_features.py --season 2025-26 --season-type "Playoffs"

# 3. Score & rank today's props
python rank_props.py \
  --history data/player_game_3pt_features.csv \
  --model   models/three_pt_model.joblib \
  --count-model models/three_pt_count_model.joblib

# 4. Refresh the season-long prediction catalog
python backfill_catalog.py
```

GitHub Actions can run this automatically at noon Eastern every day — see `.github/workflows/daily-update.yml`.

---

## Models

| Model | Purpose | Accuracy |
|-------|---------|----------|
| `models/three_pt_model.joblib` | Binary classifier — P(player hits 2+ threes) | ~74.9% |
| `models/three_pt_count_model.joblib` | Count model — expected 3PM (Poisson lambda) | — |

Training cutoff: **February 2, 2026** · Season scope: **2025-26**

Key features: `last_5_games_3pm_avg`, `season_avg_3pm`, `last_3_games_3pm`, `opp_3pa_allowed`, `opp_def_rating`, `rest_days`, `home_game`

---

## Project structure

```
├── app.py                                  # Streamlit dashboard
├── run.sh                                  # One-command launcher
├── requirements.txt
│
├── fetch_odds.py                   # Pull odds from The Odds API
├── update_features.py                # Append yesterday's results to features
├── rank_props.py                 # Score & rank today's props
├── backfill_catalog.py     # Rebuild season-long catalog
├── scoring.py                      # Core scoring logic
├── train_binary.py                      # Binary classifier trainer
├── train_count.py         # Poisson count model trainer
│
├── models/
│   ├── three_pt_model.joblib
│   └── three_pt_count_model.joblib
│
└── data/
    ├── player_game_3pt_features.csv        # Historical feature store (2025-26)
    ├── retroactive_prediction_catalog.csv  # Season-long predictions + actuals
    ├── confidence_top10_model_backfill.csv # Confidence board history
    └── SCHEMA.md                           # Full column definitions
```

---

## Odds API

Sign up at [the-odds-api.com](https://the-odds-api.com) for a free key. Pass it via `--api-key` or set the `ODDS_API_KEY` environment variable. Historical odds are not available on the free tier — dates without real odds show model-implied probabilities (marked with ★ in the dashboard).

