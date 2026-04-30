"""
Module: backtest
Purpose: Backtest Poisson count model against historical data and report calibration hit rates.
Inputs: Historical player-game features CSV and trained Poisson count model bundle.
Outputs: Calibration diagnostics including predicted lambda vs actual 3PM, hit rates for over 0.5/1.5/2.5/3.5 lines,
         variance analysis (Poisson expected vs actual), and sample prediction misses.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scoring import add_engineered_features
from train_binary import split_dataset


DEFAULT_LINES = [0.5, 1.5, 2.5, 3.5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest Poisson 3PT assumptions against historical player-game results."
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("data/player_game_3pt_features.csv"),
        help="Historical player-game dataset.",
    )
    parser.add_argument(
        "--count-model",
        type=Path,
        default=Path("models/three_pt_count_model.joblib"),
        help="Trained count model bundle used to predict lambda.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="How many sample rows to print for lambda vs actual.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of rows reserved for chronological holdout.",
    )
    return parser.parse_args()


def poisson_over_probability(lam: float, line: float) -> float:
    threshold = int(math.floor(line))
    under_or_push = 0.0
    for made in range(threshold + 1):
        under_or_push += math.exp(-lam) * (lam**made) / math.factorial(made)
    return max(0.0, 1.0 - under_or_push)


def load_predictions(history_path: Path, model_path: Path, test_size: float) -> pd.DataFrame:
    if not history_path.exists():
        raise FileNotFoundError(f"Missing history dataset: {history_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing count model bundle: {model_path}")

    df = pd.read_csv(history_path)
    if "player_made_3pt" not in df.columns:
        raise ValueError("History CSV must include 'player_made_3pt'.")

    bundle = joblib.load(model_path)
    model = bundle["model"]

    y = pd.to_numeric(df["player_made_3pt"], errors="coerce").fillna(0.0)
    x = add_engineered_features(df.drop(columns=["player_made_3pt"]))
    x_train, x_valid, y_train, y_valid, split_info = split_dataset(
        x,
        y,
        test_size=test_size,
        random_state=42,
        split_method="time",
        stratify_target=False,
    )

    valid_predictions = np.clip(model.predict(x_valid), 1e-6, None)
    result = x_valid.copy()
    result["predicted_lambda"] = valid_predictions
    result["actual_3pm"] = y_valid.to_numpy(dtype=float)
    result["split_method"] = split_info["method"]
    result["valid_start_date"] = split_info.get("valid_start_date")
    return result


def summarize_lambda_vs_actual(frame: pd.DataFrame) -> dict[str, float]:
    actual = frame["actual_3pm"].to_numpy(dtype=float)
    pred = frame["predicted_lambda"].to_numpy(dtype=float)
    mae = float(np.mean(np.abs(pred - actual)))
    rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
    corr = float(np.corrcoef(pred, actual)[0, 1]) if len(frame) > 1 else float("nan")
    return {
        "avg_predicted_lambda": float(np.mean(pred)),
        "avg_actual_3pm": float(np.mean(actual)),
        "mae": mae,
        "rmse": rmse,
        "correlation": corr,
    }


def line_hit_rates(frame: pd.DataFrame, lines: list[float]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for line in lines:
        predicted = frame["predicted_lambda"].map(lambda lam: poisson_over_probability(float(lam), line))
        actual = (frame["actual_3pm"] > line).astype(float)

        rows.append(
            {
                "line": f"over {line}",
                "avg_model_prob": float(predicted.mean()),
                "actual_hit_rate": float(actual.mean()),
                "gap": float(predicted.mean() - actual.mean()),
                "brier_score": float(np.mean((predicted - actual) ** 2)),
            }
        )
    return pd.DataFrame(rows)


def variance_diagnostics(frame: pd.DataFrame) -> dict[str, float | str]:
    actual = frame["actual_3pm"].to_numpy(dtype=float)
    pred = np.clip(frame["predicted_lambda"].to_numpy(dtype=float), 1e-6, None)

    actual_variance = float(np.var(actual))
    poisson_expected_variance = float(np.mean(pred))
    dispersion_ratio = float(actual_variance / poisson_expected_variance) if poisson_expected_variance > 0 else float("nan")
    pearson_dispersion = float(np.mean(((actual - pred) ** 2) / pred))

    if pearson_dispersion > 1.1:
        assessment = "Poisson is underestimating variance"
    elif pearson_dispersion < 0.9:
        assessment = "Poisson is overestimating variance"
    else:
        assessment = "Poisson variance is roughly in line"

    return {
        "actual_variance": actual_variance,
        "poisson_expected_variance": poisson_expected_variance,
        "variance_ratio": dispersion_ratio,
        "pearson_dispersion": pearson_dispersion,
        "assessment": assessment,
    }


def print_sample_rows(frame: pd.DataFrame, top: int) -> None:
    sample = frame[
        [
            "player_name",
            "team",
            "opponent",
            "game_date",
            "predicted_lambda",
            "actual_3pm",
        ]
    ].copy()
    sample["abs_error"] = (sample["predicted_lambda"] - sample["actual_3pm"]).abs()
    sample.sort_values(["abs_error", "predicted_lambda"], ascending=[False, False], inplace=True)
    sample.rename(
        columns={
            "player_name": "Player",
            "team": "Team",
            "opponent": "Opp",
            "game_date": "Date",
            "predicted_lambda": "Predicted_Lambda",
            "actual_3pm": "Actual_3PM",
            "abs_error": "Abs_Error",
        },
        inplace=True,
    )
    print("\nLargest prediction misses")
    print(sample.head(top).to_string(index=False))


def main() -> None:
    # 1. Parse arguments and load historical predictions from count model.
    args = parse_args()
    frame = load_predictions(args.history, args.count_model, args.test_size)

    # 2. Summarize lambda (predicted mean) vs actual 3PM count.
    lambda_summary = summarize_lambda_vs_actual(frame)
    # 3. Calculate hit rates for standard prop lines (0.5, 1.5, 2.5, 3.5).
    hit_rate_summary = line_hit_rates(frame, DEFAULT_LINES)
    # 4. Run variance diagnostics to check if Poisson assumptions hold.
    variance_summary = variance_diagnostics(frame)

    # 5. Print lambda summary (mean prediction accuracy).
    print("Predicted lambda vs actual 3PM")
    if "valid_start_date" in frame.columns and not frame["valid_start_date"].isna().all():
        print(f"  Holdout start date:     {frame['valid_start_date'].iloc[0]}")
    print(f"  Avg predicted lambda: {lambda_summary['avg_predicted_lambda']:.3f}")
    print(f"  Avg actual 3PM:       {lambda_summary['avg_actual_3pm']:.3f}")
    print(f"  MAE:                  {lambda_summary['mae']:.3f}")
    print(f"  RMSE:                 {lambda_summary['rmse']:.3f}")
    print(f"  Correlation:          {lambda_summary['correlation']:.3f}")

    # 6. Print hit rates for each prop line.
    print("\nHit rates")
    printable_hit_rates = hit_rate_summary.rename(
        columns={
            "line": "Line",
            "avg_model_prob": "Avg_Model_Prob",
            "actual_hit_rate": "Actual_Hit_Rate",
            "gap": "Gap",
            "brier_score": "Brier",
        }
    )
    print(printable_hit_rates.to_string(index=False))

    # 7. Print variance diagnostics (Poisson over/under-dispersion).
    print("\nVariance check")
    print(f"  Actual variance:            {variance_summary['actual_variance']:.3f}")
    print(f"  Poisson expected variance:  {variance_summary['poisson_expected_variance']:.3f}")
    print(f"  Variance ratio:             {variance_summary['variance_ratio']:.3f}")
    print(f"  Pearson dispersion:         {variance_summary['pearson_dispersion']:.3f}")
    print(f"  Assessment:                {variance_summary['assessment']}")

    # 8. Print sample rows with largest prediction misses.
    print_sample_rows(frame, args.top)


if __name__ == "__main__":
    main()
