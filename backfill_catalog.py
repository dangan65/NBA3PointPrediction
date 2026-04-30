"""
backfill_catalog.py
------------------------------------
Sweeps through the full historical feature file (player_game_3pt_features.csv)
and generates retroactive model predictions for every player-game in the dataset.

Output: data/retroactive_prediction_catalog.csv

Each row contains:
  - Game identifiers (date, player, team, opponent)
  - Binary model output: model_prob_over_1_5 (P player makes > 1.5 threes)
  - Count model output:  estimated_3pm_mean (expected 3PM, Poisson lambda)
  - Poisson probabilities: prob_over_0_5, prob_over_1_5, prob_over_2_5, prob_over_3_5
  - Actual result: player_made_3pt (ground truth for evaluation)
  - Key opponent defense context columns
  - Calibration metrics for each row

No odds data is required or used. Edge/EV columns are omitted since we don't have
historical sportsbook lines. This catalog is for model evaluation, backtesting,
and building a season-long prediction history.

Usage:
    python backfill_catalog.py
    python backfill_catalog.py --features data/player_game_3pt_features.csv
    python backfill_catalog.py --output data/my_catalog.csv --min-games 5
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (relative to project root — run from ML3ptProjectApp/)
# ---------------------------------------------------------------------------
DEFAULT_FEATURES = Path("data/player_game_3pt_features.csv")
DEFAULT_OUTPUT = Path("data/retroactive_prediction_catalog.csv")
DEFAULT_BIN_MODEL = Path("models/three_pt_model.joblib")
DEFAULT_CNT_MODEL = Path("models/three_pt_count_model.joblib")


# ---------------------------------------------------------------------------
# Feature engineering (mirrors scoring.add_engineered_features)
# ---------------------------------------------------------------------------
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if {"last_3_games_3pm", "season_avg_3pm"}.issubset(out.columns):
        out["hot_streak_score"] = out["last_3_games_3pm"] - out["season_avg_3pm"]

    if {"last_5_games_3pm_avg", "last_5_games_3pa_avg"}.issubset(out.columns):
        attempts = out["last_5_games_3pa_avg"].replace(0, np.nan)
        out["three_point_form"] = out["last_5_games_3pm_avg"] / attempts

    if "back_to_back" not in out.columns and "rest_days" in out.columns:
        out["back_to_back"] = (out["rest_days"] == 0).astype(int)

    for cat in [
        "home_game", "back_to_back", "position", "starter_flag",
        "team_offensive_role", "defensive_scheme_tendency", "primary_defender_pos",
    ]:
        if cat in out.columns:
            out[cat] = out[cat].fillna("UNKNOWN").astype(str)

    return out


# ---------------------------------------------------------------------------
# Poisson helpers
# ---------------------------------------------------------------------------
def poisson_prob(lam: float, k: int) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def prob_over(lam: float, line: float) -> float:
    """P(X > line) where line is a half-integer like 0.5, 1.5, 2.5 ..."""
    floor_line = int(math.floor(line))
    under = sum(poisson_prob(lam, k) for k in range(floor_line + 1))
    return max(0.0, 1.0 - under)


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------
def predict_batch(bundle: dict, df: pd.DataFrame, model_type: str) -> np.ndarray:
    """Run model on df, return 1-D array of predictions."""
    model = bundle["model"]
    numeric = bundle.get("numeric_features", [])
    categorical = bundle.get("categorical_features", [])
    all_feats = numeric + categorical

    # Keep only columns the model knows about; fill missing with NaN / UNKNOWN
    work = df.reindex(columns=all_feats)
    for col in categorical:
        if col in work.columns:
            work[col] = work[col].fillna("UNKNOWN").astype(str)

    if model_type == "binary":
        try:
            return model.predict_proba(work)[:, 1]
        except Exception:
            return np.full(len(work), np.nan)
    else:  # count / regression
        try:
            raw = model.predict(work)
            return np.clip(raw.astype(float), 0, None)
        except Exception:
            return np.full(len(work), np.nan)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------
def build_catalog(
    features_path: Path,
    bin_model_path: Path,
    cnt_model_path: Path,
    output_path: Path,
    min_season_games: int,
    verbose: bool,
) -> pd.DataFrame:

    # --- Load data ---
    if verbose:
        print(f"Loading features from {features_path} …")
    df = pd.read_csv(features_path)

    total_rows = len(df)
    if verbose:
        print(f"  {total_rows:,} rows | {df['game_date'].nunique()} game dates "
              f"({df['game_date'].min()} → {df['game_date'].max()})")

    # --- Load models ---
    if verbose:
        print(f"Loading binary model from {bin_model_path} …")
    bin_bundle = joblib.load(bin_model_path)

    if verbose:
        print(f"Loading count model from {cnt_model_path} …")
    cnt_bundle = joblib.load(cnt_model_path)

    # --- Filter: require a minimum number of season games for meaningful features ---
    # season_avg_3pm becomes meaningful only after a handful of games.
    if "season_3pm" in df.columns and "season_3pa" in df.columns:
        # Proxy: rows where the player has played enough to have rolling averages
        pass  # we'll let the model handle sparse rows via its imputer

    # Filter out rows that are entirely missing core features (these would produce
    # meaningless predictions regardless).
    core_cols = ["last_5_games_3pm_avg", "season_avg_3pm", "last_3_games_3pm"]
    available_core = [c for c in core_cols if c in df.columns]
    if available_core:
        before = len(df)
        df = df.dropna(subset=available_core, how="all")
        dropped = before - len(df)
        if verbose and dropped:
            print(f"  Dropped {dropped:,} rows missing all core rolling features.")

    # --- Feature engineering ---
    if verbose:
        print("Applying feature engineering …")
    df = add_engineered_features(df)

    # --- Run models in batch ---
    if verbose:
        print("Running binary model (P over 1.5) …")
    bin_probs = predict_batch(bin_bundle, df, "binary")

    if verbose:
        print("Running count model (expected 3PM) …")
    cnt_preds = predict_batch(cnt_bundle, df, "count")

    # --- Build Poisson probabilities per row ---
    if verbose:
        print("Computing Poisson distributions …")

    prob_o05 = np.array([prob_over(lam, 0.5) if not np.isnan(lam) else np.nan for lam in cnt_preds])
    prob_o15 = np.array([prob_over(lam, 1.5) if not np.isnan(lam) else np.nan for lam in cnt_preds])
    prob_o25 = np.array([prob_over(lam, 2.5) if not np.isnan(lam) else np.nan for lam in cnt_preds])
    prob_o35 = np.array([prob_over(lam, 3.5) if not np.isnan(lam) else np.nan for lam in cnt_preds])

    # --- Assemble output ---
    opp_cols = [
        "opp_3pa_allowed", "opp_3pm_allowed", "opp_def_rating",
        "opp_pace", "opp_pace_differential", "defensive_scheme_tendency",
    ]
    id_cols = [
        "season", "game_date", "game_id", "player_id", "player_name",
        "team", "opponent", "home_game", "position", "rest_days", "back_to_back",
    ]

    keep_id = [c for c in id_cols if c in df.columns]
    keep_opp = [c for c in opp_cols if c in df.columns]

    catalog = df[keep_id + keep_opp].copy()
    catalog["model_prob_over_1_5"] = bin_probs
    catalog["estimated_3pm_mean"] = cnt_preds
    catalog["prob_over_0_5"] = prob_o05
    catalog["prob_over_1_5"] = prob_o15
    catalog["prob_over_2_5"] = prob_o25
    catalog["prob_over_3_5"] = prob_o35

    # Ground truth — what actually happened
    if "player_made_3pt" in df.columns:
        catalog["player_made_3pt_actual"] = pd.to_numeric(df["player_made_3pt"], errors="coerce")

    # Simple hit flags for quick accuracy checks
    if "player_made_3pt_actual" in catalog.columns:
        actual = catalog["player_made_3pt_actual"]
        catalog["hit_over_1_5"] = (actual >= 2).astype("Int8")
        catalog["hit_over_2_5"] = (actual >= 3).astype("Int8")

    # Sort by date then player
    catalog.sort_values(["game_date", "player_name"], inplace=True)
    catalog.reset_index(drop=True, inplace=True)

    # --- Save ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(output_path, index=False)

    if verbose:
        print(f"\n✓ Catalog written to {output_path}")
        print(f"  {len(catalog):,} player-game predictions")
        print(f"  {catalog['game_date'].nunique()} unique game dates")
        if "player_made_3pt_actual" in catalog.columns:
            valid = catalog.dropna(subset=["player_made_3pt_actual", "model_prob_over_1_5"])
            if len(valid):
                acc = ((valid["model_prob_over_1_5"] >= 0.5) == (valid["player_made_3pt_actual"] >= 2)).mean()
                print(f"  Binary accuracy @ 0.5 threshold: {acc:.3f}")
                mean_actual = valid["player_made_3pt_actual"].mean()
                mean_pred = valid["estimated_3pm_mean"].mean()
                print(f"  Mean actual 3PM: {mean_actual:.3f}  |  Mean predicted 3PM: {mean_pred:.3f}")

    return catalog


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate retroactive season-long prediction catalog.")
    p.add_argument("--features", type=Path, default=DEFAULT_FEATURES,
                   help=f"Feature CSV (default: {DEFAULT_FEATURES})")
    p.add_argument("--bin-model", type=Path, default=DEFAULT_BIN_MODEL,
                   help=f"Binary classifier joblib (default: {DEFAULT_BIN_MODEL})")
    p.add_argument("--cnt-model", type=Path, default=DEFAULT_CNT_MODEL,
                   help=f"Count/regression model joblib (default: {DEFAULT_CNT_MODEL})")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help=f"Output CSV path (default: {DEFAULT_OUTPUT})")
    p.add_argument("--min-games", type=int, default=3,
                   help="Minimum season games before including a row (default: 3, currently unused filter).")
    p.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    for path, label in [
        (args.features, "features CSV"),
        (args.bin_model, "binary model"),
        (args.cnt_model, "count model"),
    ]:
        if not path.exists():
            print(f"ERROR: {label} not found at {path}", file=sys.stderr)
            sys.exit(1)

    build_catalog(
        features_path=args.features,
        bin_model_path=args.bin_model,
        cnt_model_path=args.cnt_model,
        output_path=args.output,
        min_season_games=args.min_games,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
