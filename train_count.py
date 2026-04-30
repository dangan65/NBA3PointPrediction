"""
Train a Poisson/count regression model predicting expected 3-pointers made per player-game.

Module Purpose:
  Build and train regression models (Poisson and gradient boosting) to predict player made 3-pointers,
  evaluate both models, select the best based on mean Poisson deviance, and save to joblib bundle.

Inputs:
  - Training CSV with one row per player-game including player_made_3pt target and feature columns
  - Helper functions from train_binary.py for feature engineering and dataset splitting

Outputs:
  - Joblib bundle saved to models/three_pt_count_model.joblib containing:
    model, model_name, metrics, numeric_features, categorical_features, split_info
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from train_binary import add_engineered_features, get_feature_lists, split_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a direct count model for player made threes."
    )
    parser.add_argument("csv_path", type=Path, help="Training CSV with one row per player-game.")
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("models/three_pt_count_model.joblib"),
        help="Where to save the trained count model.",
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
    return parser.parse_args()


def build_poisson_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
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

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", PoissonRegressor(alpha=0.05, max_iter=2000)),
        ]
    )


def build_boosting_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            ),
        ],
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="poisson",
                    learning_rate=0.05,
                    max_depth=4,
                    max_iter=250,
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate_model(model: Pipeline, x_valid: pd.DataFrame, y_valid: pd.Series) -> dict[str, float]:
    preds = np.clip(model.predict(x_valid), 1e-6, None)
    return {
        "mae": float(mean_absolute_error(y_valid, preds)),
        "rmse": float(root_mean_squared_error(y_valid, preds)),
        "mean_poisson_deviance": float(mean_poisson_deviance(y_valid, preds)),
    }


def main() -> None:
    # 1. Parse arguments and load training data
    args = parse_args()
    df = pd.read_csv(args.csv_path)
    df = add_engineered_features(df)

    # 2. Validate target column and extract X, y
    if "player_made_3pt" not in df.columns:
        raise ValueError("Training CSV must include 'player_made_3pt'.")

    y = pd.to_numeric(df["player_made_3pt"], errors="coerce").fillna(0).clip(lower=0)
    x = df.drop(columns=["player_made_3pt"])

    # 3. Extract numeric and categorical feature lists from training set
    numeric_features, categorical_features = get_feature_lists(x)
    if not numeric_features and not categorical_features:
        raise ValueError("No supported feature columns were found in the training CSV.")

    # 4. Split dataset into train and validation sets
    x_train, x_valid, y_train, y_valid, split_info = split_dataset(
        x, y, args.test_size, args.random_state, args.split_method, stratify_target=False
    )

    # 5. Build candidate models: Poisson regression and gradient boosting
    candidates = {
        "poisson_regression": build_poisson_pipeline(numeric_features, categorical_features),
        "poisson_gradient_boosting": build_boosting_pipeline(numeric_features, categorical_features),
    }

    # 6. Train all candidate models and evaluate on validation set
    results: dict[str, dict[str, float]] = {}
    fitted_models: dict[str, Pipeline] = {}

    for name, model in candidates.items():
        model.fit(x_train, y_train)
        results[name] = evaluate_model(model, x_valid, y_valid)
        fitted_models[name] = model

    # 7. Select best model based on mean Poisson deviance
    best_name = min(results, key=lambda model_name: results[model_name]["mean_poisson_deviance"])
    best_model = fitted_models[best_name]

    # 8. Save best model and metadata to joblib bundle
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_name": best_name,
            "model": best_model,
            "metrics": results,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "split_info": split_info,
        },
        args.model_out,
    )

    # 9. Print results and save location
    print(f"Selected count model: {best_name}")
    print(f"Split method: {split_info['method']}")
    for name, metrics in results.items():
        print(f"\n{name}")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    print(f"\nSaved count model bundle to: {args.model_out}")


if __name__ == "__main__":
    main()
