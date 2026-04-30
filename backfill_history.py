"""
Module: backfill_history
Purpose: Sweep historical feature data and generate top-10 model confidence history.
Inputs: Historical player-game features CSV, trained binary model bundle, optional count model bundle.
Outputs: value_top10_model_backfill.csv (top-10 by model value), value_top10_model_backfill_summary.csv,
         confidence_top10_model_backfill.csv (top-10 by confidence), confidence_top10_model_backfill_summary.csv.
         Each CSV includes rank, date, player, team, opponent, bet odds, model prediction, actual outcome, and win/loss.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from train_binary import add_engineered_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill local top-10 model history by slate date from historical game rows."
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("data/player_game_3pt_features.csv"),
        help="Historical player-game dataset.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/three_pt_model.joblib"),
        help="Trained binary model bundle.",
    )
    parser.add_argument(
        "--count-model",
        type=Path,
        default=Path("models/three_pt_count_model.joblib"),
        help="Optional count model bundle.",
    )
    parser.add_argument(
        "--start-date",
        default="2026-03-01",
        help="Backfill start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Backfill end date (YYYY-MM-DD). Defaults to latest game_date in history.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Rows per day to keep.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/value_top10_model_backfill.csv"),
        help="Output CSV with daily top-10 model picks.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("data/value_top10_model_backfill_summary.csv"),
        help="Output CSV with daily aggregate hit-rate summary.",
    )
    parser.add_argument(
        "--confidence-output",
        type=Path,
        default=Path("data/confidence_top10_model_backfill.csv"),
        help="Output CSV with daily top-10 most-confident model picks.",
    )
    parser.add_argument(
        "--confidence-summary-output",
        type=Path,
        default=Path("data/confidence_top10_model_backfill_summary.csv"),
        help="Output CSV with daily aggregate hit-rate summary for confidence picks.",
    )
    parser.add_argument(
        "--evaluation-mode",
        choices=["out_of_sample", "in_sample"],
        default="out_of_sample",
        help=(
            "Backfill evaluation mode. 'out_of_sample' is default and only keeps rows strictly "
            "after the training cutoff date."
        ),
    )
    parser.add_argument(
        "--oos-cutoff-date",
        default=None,
        help=(
            "Optional explicit OOS cutoff (YYYY-MM-DD). Rows on or before this date are excluded "
            "when evaluation-mode is out_of_sample."
        ),
    )
    return parser.parse_args()


def prepare_features(frame: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]) -> pd.DataFrame:
    feature_frame = frame.copy()
    for column in [*numeric_features, *categorical_features]:
        if column not in feature_frame.columns:
            feature_frame[column] = np.nan
    return feature_frame[numeric_features + categorical_features]


def fair_american_odds(prob: float | int | None, favorite_cap: int = -500) -> int | None:
    value = pd.to_numeric(pd.Series([prob]), errors="coerce").iloc[0]
    if pd.isna(value):
        return None
    p = float(value)
    if p <= 0.0 or p >= 1.0:
        return None
    if p >= 0.5:
        odds = -100.0 * p / (1.0 - p)
    else:
        odds = 100.0 * (1.0 - p) / p
    american = int(round(odds))
    if american < favorite_cap:
        return int(favorite_cap)
    return american


def model_label(path: Path, experimental_hint: str = "last30") -> str:
    text = str(path)
    if experimental_hint in path.name:
        return f"{text} (experimental)"
    return f"{text} (production)"


def resolve_oos_cutoff(args: argparse.Namespace, model_bundle: dict[str, object]) -> pd.Timestamp | None:
    if args.oos_cutoff_date:
        parsed = pd.to_datetime(args.oos_cutoff_date, errors="coerce")
        if pd.isna(parsed):
            raise ValueError(f"Invalid --oos-cutoff-date value: {args.oos_cutoff_date}")
        return pd.Timestamp(parsed).normalize()

    split_info = model_bundle.get("split_info", {})
    if not isinstance(split_info, dict):
        return None
    if str(split_info.get("method", "")) != "time":
        return None

    train_end_date = split_info.get("train_end_date")
    if train_end_date is None:
        return None
    parsed = pd.to_datetime(train_end_date, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).normalize()


def main() -> None:
    # 1. Load binary model bundle and optional count model.
    args = parse_args()
    if not args.history.exists():
        raise FileNotFoundError(f"Missing history dataset: {args.history}")
    if not args.model.exists():
        raise FileNotFoundError(f"Missing model bundle: {args.model}")
    model_bundle = joblib.load(args.model)
    model = model_bundle["model"]
    print(f"Using binary model for backfill: {model_label(args.model)}")
    if args.count_model.exists():
        print(f"Using count model for backfill: {model_label(args.count_model)}")
    else:
        print("Using count model for backfill: disabled")

    # 2. Load and filter historical features by date range.
    history = pd.read_csv(args.history)
    history = add_engineered_features(history)
    history["game_date"] = pd.to_datetime(history["game_date"], errors="coerce")
    history = history[history["game_date"].notna()].copy()
    history["slate_date"] = history["game_date"].dt.date.astype(str)
    history = history[history["slate_date"] >= args.start_date].copy()
    if args.end_date:
        history = history[history["slate_date"] <= args.end_date].copy()
    if history.empty:
        raise RuntimeError("No rows available in the requested date range.")

    # 3. Apply out-of-sample cutoff if requested (exclude training data).
    if args.evaluation_mode == "out_of_sample":
        cutoff = resolve_oos_cutoff(args, model_bundle)
        if cutoff is None:
            raise RuntimeError(
                "Out-of-sample mode requires a cutoff date, but none was found. "
                "Pass --oos-cutoff-date YYYY-MM-DD or use a model bundle with time-split metadata."
            )
        before_rows = len(history)
        history = history[history["game_date"] > cutoff].copy()
        removed = before_rows - len(history)
        print(f"Applied out-of-sample cutoff: > {cutoff.date().isoformat()} (removed {removed} rows).")
        if history.empty:
            raise RuntimeError("No rows remain after applying the out-of-sample cutoff.")
    else:
        print("Evaluation mode: in_sample (includes rows that may have been seen during training).")

    # 4. Generate model predictions (binary probability and optional count mean).
    numeric_features = list(model_bundle.get("numeric_features", []))
    categorical_features = list(model_bundle.get("categorical_features", []))
    x = prepare_features(history, numeric_features, categorical_features)
    history["model_prob"] = model.predict_proba(x)[:, 1]

    if args.count_model.exists():
        count_bundle = joblib.load(args.count_model)
        count_model = count_bundle["model"]
        count_numeric = list(count_bundle.get("numeric_features", []))
        count_categorical = list(count_bundle.get("categorical_features", []))
        cx = prepare_features(history, count_numeric, count_categorical)
        history["estimated_3pm_mean"] = count_model.predict(cx)
    else:
        history["estimated_3pm_mean"] = np.nan

    # 5. Compute bet outcomes and odds; rank by model probability per day.
    history["side"] = "OVER"
    history["line"] = 1.5
    history["bet_won"] = (pd.to_numeric(history["player_made_3pt"], errors="coerce") >= 2).astype(int)
    history["bet_result"] = history["bet_won"].map({1: "WIN", 0: "LOSS"})
    history["american_odds"] = history["model_prob"].map(fair_american_odds)
    history["odds_source"] = "model_implied_capped_-500"
    history["rank"] = (
        history.groupby("slate_date")["model_prob"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    history = history[history["rank"] <= args.top_n].copy()
    history.sort_values(["slate_date", "rank"], ascending=[True, True], inplace=True)

    # 6. Prepare value backfill output (top-10 by model probability).
    output = history[
        [
            "rank",
            "slate_date",
            "player_name",
            "team",
            "opponent",
            "side",
            "line",
            "american_odds",
            "odds_source",
            "model_prob",
            "estimated_3pm_mean",
            "player_made_3pt",
            "bet_won",
            "bet_result",
        ]
    ].copy()
    output["source"] = "local_history_backfill"

    # 7. Compute daily summary stats (bets, wins, win rate) for value backfill.
    summary = (
        output.groupby("slate_date")
        .agg(
            bets=("bet_won", "size"),
            wins=("bet_won", "sum"),
            win_rate=("bet_won", "mean"),
        )
        .reset_index()
        .sort_values("slate_date", ascending=False)
    )

    # 8. Prepare confidence backfill output (top-10 by model probability, same as value).
    confidence = history.copy()
    confidence["rank"] = (
        confidence.groupby("slate_date")["model_prob"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    confidence = confidence[confidence["rank"] <= args.top_n].copy()
    confidence.sort_values(["slate_date", "rank"], ascending=[True, True], inplace=True)
    confidence_output = confidence[
        [
            "rank",
            "slate_date",
            "player_name",
            "team",
            "opponent",
            "side",
            "line",
            "american_odds",
            "odds_source",
            "model_prob",
            "estimated_3pm_mean",
            "player_made_3pt",
            "bet_won",
            "bet_result",
        ]
    ].copy()
    confidence_output["source"] = "local_confidence_backfill"
    # 9. Compute daily summary stats for confidence backfill.
    confidence_summary = (
        confidence_output.groupby("slate_date")
        .agg(
            bets=("bet_won", "size"),
            wins=("bet_won", "sum"),
            win_rate=("bet_won", "mean"),
        )
        .reset_index()
        .sort_values("slate_date", ascending=False)
    )

    # 10. Write all CSV outputs and print summary.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False)
    args.confidence_output.parent.mkdir(parents=True, exist_ok=True)
    confidence_output.to_csv(args.confidence_output, index=False)
    args.confidence_summary_output.parent.mkdir(parents=True, exist_ok=True)
    confidence_summary.to_csv(args.confidence_summary_output, index=False)

    print(f"Saved backfilled top-10 model rows to: {args.output}")
    print(f"Saved backfilled daily summary to: {args.summary_output}")
    print(f"Saved confidence backfilled top-10 model rows to: {args.confidence_output}")
    print(f"Saved confidence backfilled daily summary to: {args.confidence_summary_output}")
    print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
