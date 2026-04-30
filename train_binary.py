"""
Module: train_binary
Purpose: Train a binary GradientBoosting classifier to predict P(player makes 2+ threes).
Inputs: CSV file with player-game features and target (player_made_3pt column).
Outputs: Trained model bundle saved to models/three_pt_model.joblib, including preprocessor,
         calibration settings, split metadata (time-based train/val split), and validation metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_NUMERIC_FEATURES = [
    "last_5_games_3pm_avg",
    "last_5_games_3pa_avg",
    "last_3_games_3pm",
    "season_avg_3pm",
    "rolling_3_game_trend_slope",
    "minutes_per_game",
    "projected_minutes",
    "last_5_minutes_avg",
    "last_10_minutes_avg",
    "minutes_stddev_last_10",
    "minutes_trend_slope",
    "usage_rate",
    "team_3pa_share",
    "last_5_3pa_trend",
    "rest_days",
    "spread",
    "game_total",
    "injury_usage_bump",
    "teammate_minutes_vacated",
    "teammate_usage_vacated",
    "teammate_on_off_3pa_delta",
    "opp_3pa_allowed",
    "opp_3pm_allowed",
    "opp_def_rating",
    "opp_pace",
    "opp_pace_differential",
    "opp_position_3pt_rank",
    "opp_3pa_allowed_by_position",
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
    "opp_team_wide_open_3pa_allowed",
    "opp_team_above_break_3pa_allowed",
    "opp_team_corner_3pa_allowed",
    "corner_3_allowed",
    "above_break_3_allowed",
    "primary_defender_3pt_pct_allowed",
    "primary_defender_def_fg_pct_diff",
]

BASE_CATEGORICAL_FEATURES = [
    "home_game",
    "back_to_back",
    "position",
    "starter_flag",
    "team_offensive_role",
    "defensive_scheme_tendency",
    "primary_defender_pos",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a binary classifier for player over 1.5 three-pointers made."
    )
    parser.add_argument("csv_path", type=Path, help="Training CSV with one row per player-game.")
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("models/three_pt_model.joblib"),
        help="Where to save the trained model bundle.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Validation split fraction.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--split-method",
        choices=["time", "random"],
        default="time",
        help="Use a chronological holdout when game_date is available.",
    )
    parser.add_argument(
        "--calibration-method",
        choices=["none", "sigmoid", "isotonic"],
        default="sigmoid",
        help="Optional probability calibration method applied to the best base model.",
    )
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=10,
        help="Number of bins for reliability summaries.",
    )
    return parser.parse_args()


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()

    required_for_hot = {"last_3_games_3pm", "season_avg_3pm"}
    if required_for_hot.issubset(df.columns):
        df["hot_streak_score"] = df["last_3_games_3pm"] - df["season_avg_3pm"]

    required_for_form = {"last_5_games_3pm_avg", "last_5_games_3pa_avg"}
    if required_for_form.issubset(df.columns):
        attempts = df["last_5_games_3pa_avg"].replace(0, np.nan)
        df["three_point_form"] = df["last_5_games_3pm_avg"] / attempts

    if "back_to_back" not in df.columns and "rest_days" in df.columns:
        df["back_to_back"] = (df["rest_days"] == 0).astype(int)

    for categorical in [
        "home_game",
        "back_to_back",
        "position",
        "starter_flag",
        "team_offensive_role",
        "defensive_scheme_tendency",
        "primary_defender_pos",
    ]:
        if categorical in df.columns:
            df[categorical] = df[categorical].fillna("UNKNOWN").astype(str)

    return df


def get_feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_features = list(BASE_NUMERIC_FEATURES)
    categorical_features = list(BASE_CATEGORICAL_FEATURES)

    for engineered in ("hot_streak_score", "three_point_form"):
        if engineered in df.columns:
            numeric_features.append(engineered)

    available_numeric = [
        col for col in numeric_features if col in df.columns and not df[col].isna().all()
    ]
    available_categorical = [
        col for col in categorical_features if col in df.columns and not df[col].isna().all()
    ]
    return available_numeric, available_categorical


def build_preprocessor(
    numeric_features: list[str], categorical_features: list[str], scale_numeric: bool
) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=numeric_steps),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ],
    )


def build_logistic_pipeline(
    numeric_features: list[str], categorical_features: list[str]
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_features, categorical_features, scale_numeric=True)),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )


def build_boosting_pipeline(
    numeric_features: list[str], categorical_features: list[str]
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_features, categorical_features, scale_numeric=False)),
            ("model", GradientBoostingClassifier(random_state=42)),
        ]
    )


def split_dataset(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
    split_method: str,
    stratify_target: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict[str, object]]:
    if split_method == "time" and "game_date" in x.columns:
        ordered = x.copy()
        ordered["_game_date"] = pd.to_datetime(ordered["game_date"], errors="coerce")
        ordered["_target"] = y.to_numpy()
        ordered.sort_values(["_game_date", "game_id", "player_id"], inplace=True, na_position="last")

        valid_size = max(1, int(np.ceil(len(ordered) * test_size)))
        split_index = max(1, len(ordered) - valid_size)

        train_frame = ordered.iloc[:split_index].copy()
        valid_frame = ordered.iloc[split_index:].copy()

        if train_frame.empty or valid_frame.empty:
            raise ValueError("Time split produced an empty train or validation set.")

        y_train = train_frame.pop("_target").astype(int)
        y_valid = valid_frame.pop("_target").astype(int)
        train_end = train_frame["_game_date"].max()
        valid_start = valid_frame["_game_date"].min()
        train_frame.drop(columns=["_game_date"], inplace=True)
        valid_frame.drop(columns=["_game_date"], inplace=True)

        return train_frame, valid_frame, y_train, y_valid, {
            "method": "time",
            "train_rows": int(len(train_frame)),
            "valid_rows": int(len(valid_frame)),
            "train_end_date": None if pd.isna(train_end) else str(train_end.date()),
            "valid_start_date": None if pd.isna(valid_start) else str(valid_start.date()),
        }

    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify_target else None,
    )
    return x_train, x_valid, y_train, y_valid, {
        "method": "random",
        "train_rows": int(len(x_train)),
        "valid_rows": int(len(x_valid)),
    }


def calibration_summary(y_true: pd.Series, probs: np.ndarray, bins: int) -> list[dict[str, float | int | None]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    labels = np.digitize(probs, edges[1:-1], right=False)
    summary: list[dict[str, float | int | None]] = []

    for index in range(bins):
        mask = labels == index
        count = int(mask.sum())
        if count == 0:
            summary.append(
                {
                    "bin": index,
                    "count": 0,
                    "prob_min": float(edges[index]),
                    "prob_max": float(edges[index + 1]),
                    "avg_pred": None,
                    "actual_rate": None,
                    "bin_brier": None,
                }
            )
            continue

        bin_probs = probs[mask]
        bin_true = y_true.to_numpy()[mask]
        summary.append(
            {
                "bin": index,
                "count": count,
                "prob_min": float(edges[index]),
                "prob_max": float(edges[index + 1]),
                "avg_pred": float(bin_probs.mean()),
                "actual_rate": float(bin_true.mean()),
                "bin_brier": float(np.mean((bin_probs - bin_true) ** 2)),
            }
        )
    return summary


def evaluate_model(
    model: Pipeline,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    calibration_bins: int,
) -> dict[str, object]:
    probs = model.predict_proba(x_valid)[:, 1]
    preds = (probs >= 0.5).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_valid, probs)),
        "accuracy": float(accuracy_score(y_valid, preds)),
        "brier_score": float(brier_score_loss(y_valid, probs)),
        "log_loss": float(log_loss(y_valid, probs)),
        "calibration": calibration_summary(y_valid, probs, calibration_bins),
    }


def main() -> None:
    # 1. Parse command-line arguments and load training data.
    args = parse_args()
    df = pd.read_csv(args.csv_path)
    df = add_engineered_features(df)

    if "player_made_3pt" not in df.columns:
        raise ValueError("Training CSV must include 'player_made_3pt'.")

    # 2. Create binary target (1 if made 2+ threes, else 0) and drop target from features.
    y = (df["player_made_3pt"] >= 2).astype(int)
    x = df.drop(columns=["player_made_3pt"])

    # 3. Identify available numeric and categorical features.
    numeric_features, categorical_features = get_feature_lists(x)
    if not numeric_features and not categorical_features:
        raise ValueError("No supported feature columns were found in the training CSV.")

    dropped_all_missing = [
        column
        for column in [*BASE_NUMERIC_FEATURES, *BASE_CATEGORICAL_FEATURES]
        if column in x.columns and column not in numeric_features and column not in categorical_features
    ]

    # 4. Split data into train/validation sets using time-based or random split.
    x_train, x_valid, y_train, y_valid, split_info = split_dataset(
        x, y, args.test_size, args.random_state, args.split_method
    )

    # 5. Build and train candidate models (Logistic Regression and Gradient Boosting).
    candidate_builders = {
        "logistic_regression": lambda: build_logistic_pipeline(numeric_features, categorical_features),
        "gradient_boosting": lambda: build_boosting_pipeline(numeric_features, categorical_features),
    }

    results: dict[str, dict[str, object]] = {}
    fitted_models: dict[str, Pipeline] = {}

    for name, builder in candidate_builders.items():
        model = builder()
        model.fit(x_train, y_train)
        results[name] = evaluate_model(model, x_valid, y_valid, args.calibration_bins)
        fitted_models[name] = model

    # 6. Select best model by log loss.
    best_name = min(results, key=lambda model_name: float(results[model_name]["log_loss"]))
    selected_name = best_name
    best_model: object = fitted_models[best_name]

    # 7. Apply optional probability calibration and select final model.
    if args.calibration_method != "none":
        calibrated_name = f"{best_name}_{args.calibration_method}_calibrated"
        calibrated_model = CalibratedClassifierCV(
            estimator=candidate_builders[best_name](),
            method=args.calibration_method,
            cv=3,
        )
        calibrated_model.fit(x_train, y_train)
        calibrated_metrics = evaluate_model(calibrated_model, x_valid, y_valid, args.calibration_bins)
        results[calibrated_name] = calibrated_metrics

        if float(calibrated_metrics["log_loss"]) <= float(results[best_name]["log_loss"]):
            selected_name = calibrated_name
            best_model = calibrated_model

    # 8. Save trained model bundle with metrics, features, and split metadata.
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_name": selected_name,
            "model": best_model,
            "metrics": results,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "split_info": split_info,
        },
        args.model_out,
    )

    # 9. Print summary of training results and model selection.
    print(f"Selected model: {selected_name}")
    print(f"Split method: {split_info['method']}")
    if dropped_all_missing:
        print(f"Skipped all-missing features: {', '.join(dropped_all_missing)}")
    if split_info["method"] == "time":
        print(f"Train through: {split_info['train_end_date']}")
        print(f"Validate from: {split_info['valid_start_date']}")

    for name, metrics in results.items():
        print(f"\n{name}")
        for metric_name in ("roc_auc", "accuracy", "brier_score", "log_loss"):
            print(f"  {metric_name}: {float(metrics[metric_name]):.4f}")
        first_bins = metrics["calibration"][: min(3, len(metrics["calibration"]))]
        for bucket in first_bins:
            if bucket["count"] == 0:
                continue
            print(
                "  calibration"
                f" [{bucket['prob_min']:.1f}, {bucket['prob_max']:.1f}):"
                f" count={bucket['count']}"
                f" avg_pred={bucket['avg_pred']:.3f}"
                f" actual={bucket['actual_rate']:.3f}"
            )

    print(f"\nSaved model bundle to: {args.model_out}")


if __name__ == "__main__":
    main()
