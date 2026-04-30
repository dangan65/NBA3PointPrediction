"""
Module for computing player predictability scores based on model accuracy.

Purpose:
  - Score each player's historical predictability (how consistently correct the model is)
  - Build predictability tables from game history and trained models
  - Identify "star" players with top-tier predictability within their attempt bucket
  - Annotate recommendations with predictability metrics and display names
  - Output player_predictability_scores.csv for tracking model calibration per player

Inputs:
  - Player game history (player_name, game_date, player_made_3pt)
  - Trained model bundles (binary classifier for 2+ threshold, count model for exact prediction)
  - Model feature columns and valid start dates
  - Attempt bucket bins and star percentile threshold

Outputs:
  - player_predictability_scores.csv with predictability metrics, percentiles, and star flags
  - Annotated DataFrames with player_display_name (asterisk for stars)
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scoring import add_engineered_features


DEFAULT_PREDICTABILITY_OUTPUT = Path("data/player_predictability_scores.csv")
DEFAULT_PREDICTABILITY_MIN_GAMES = 8
DEFAULT_PREDICTABILITY_STAR_PERCENTILE = 0.80
PREDICTABILITY_BUCKET_LABELS = ["low<=4", "mid 4-6", "high 6-8", "elite 8+"]
PREDICTABILITY_BUCKET_BINS = [-np.inf, 4.0, 6.0, 8.0, np.inf]
PREDICTABILITY_COLUMNS = [
    "player_display_name",
    "predictability_games",
    "predictability_score",
    "predictability_percentile",
    "predictability_bucket_percentile",
    "predictability_bucket_size",
    "predictability_attempt_bucket",
    "predictability_avg_actual_3pm",
    "predictability_avg_pred_3pm",
    "predictability_mae_3pm",
    "predictability_brier_2plus",
    "predictability_acc_2plus",
    "predictability_valid_start_date",
    "predictability_star",
]


def load_model_bundle(model_path: Path | None) -> dict[str, object] | None:
    if model_path is None or not Path(model_path).exists():
        return None
    return joblib.load(model_path)


def truthy_flag(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "*"}


def player_display_name(player_name: object, star_value: object) -> str:
    if pd.isna(player_name):
        return ""
    name = str(player_name).strip()
    if not name:
        return ""
    return f"{name} *" if truthy_flag(star_value) else name


def _safe_feature_frame(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    prepared = frame.copy()
    for column in feature_columns:
        if column not in prepared.columns:
            prepared[column] = np.nan
    return prepared[feature_columns].copy()


def _valid_start_date(*bundles: dict[str, object] | None) -> pd.Timestamp | None:
    parsed_dates: list[pd.Timestamp] = []
    for bundle in bundles:
        if not bundle:
            continue
        split_info = bundle.get("split_info", {})
        if not isinstance(split_info, dict):
            continue
        parsed = pd.to_datetime(split_info.get("valid_start_date"), errors="coerce")
        if pd.notna(parsed):
            parsed_dates.append(parsed.normalize())
    if not parsed_dates:
        return None
    return max(parsed_dates)


def _quality_percentile(metric: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(metric, errors="coerce")
    if numeric.notna().sum() <= 1:
        return pd.Series(np.nan, index=metric.index, dtype=float)
    return 1.0 - numeric.rank(pct=True, method="average", ascending=True)


def build_player_predictability_table(
    history: pd.DataFrame,
    binary_model_path: Path | None,
    count_model_path: Path | None,
    min_games: int = DEFAULT_PREDICTABILITY_MIN_GAMES,
    star_percentile: float = DEFAULT_PREDICTABILITY_STAR_PERCENTILE,
) -> pd.DataFrame:
    # 1. Validate input history has required columns.
    if history.empty or "player_name" not in history.columns or "player_made_3pt" not in history.columns:
        return pd.DataFrame(columns=["player_name", *PREDICTABILITY_COLUMNS])

    # 2. Load binary and count model bundles; return empty if neither exists.
    binary_bundle = load_model_bundle(binary_model_path)
    count_bundle = load_model_bundle(count_model_path)
    if binary_bundle is None and count_bundle is None:
        return pd.DataFrame(columns=["player_name", *PREDICTABILITY_COLUMNS])

    # 3. Add engineered features and filter to valid date range.
    valid = add_engineered_features(history.copy())
    valid["game_date"] = pd.to_datetime(valid.get("game_date"), errors="coerce")
    valid_start = _valid_start_date(binary_bundle, count_bundle)
    if valid_start is not None:
        valid = valid[valid["game_date"] >= valid_start].copy()
    if valid.empty:
        return pd.DataFrame(columns=["player_name", *PREDICTABILITY_COLUMNS])

    # 4. Prepare actual outcomes: actual 3PM count and binary over 1.5 indicator.
    valid["actual_3pm"] = pd.to_numeric(valid["player_made_3pt"], errors="coerce").fillna(0).clip(lower=0)
    valid["actual_over_1_5"] = (valid["actual_3pm"] >= 2).astype(int)

    # 5. If binary model exists, score probability of over 1.5 and compute Brier error & accuracy.
    if binary_bundle is not None:
        binary_features = list(binary_bundle.get("numeric_features", [])) + list(
            binary_bundle.get("categorical_features", [])
        )
        binary_input = _safe_feature_frame(valid, binary_features)
        valid["predictability_prob_over_1_5"] = binary_bundle["model"].predict_proba(binary_input)[:, 1]
        valid["predictability_brier_2plus"] = (
            valid["predictability_prob_over_1_5"] - valid["actual_over_1_5"]
        ) ** 2
        valid["predictability_correct_2plus"] = (
            (valid["predictability_prob_over_1_5"] >= 0.5).astype(int) == valid["actual_over_1_5"]
        ).astype(int)
    else:
        valid["predictability_prob_over_1_5"] = np.nan
        valid["predictability_brier_2plus"] = np.nan
        valid["predictability_correct_2plus"] = np.nan

    # 6. If count model exists, predict exact 3PM and compute mean absolute error.
    if count_bundle is not None:
        count_features = list(count_bundle.get("numeric_features", [])) + list(
            count_bundle.get("categorical_features", [])
        )
        count_input = _safe_feature_frame(valid, count_features)
        valid["predictability_pred_3pm"] = np.clip(count_bundle["model"].predict(count_input), 1e-6, None)
        valid["predictability_abs_err_3pm"] = (valid["predictability_pred_3pm"] - valid["actual_3pm"]).abs()
    else:
        valid["predictability_pred_3pm"] = np.nan
        valid["predictability_abs_err_3pm"] = np.nan

    # 7. Group by player and aggregate accuracy metrics and game counts.
    grouped = (
        valid.groupby("player_name")
        .agg(
            predictability_games=("player_name", "size"),
            predictability_avg_actual_3pm=("actual_3pm", "mean"),
            predictability_avg_pred_3pm=("predictability_pred_3pm", "mean"),
            predictability_mae_3pm=("predictability_abs_err_3pm", "mean"),
            predictability_brier_2plus=("predictability_brier_2plus", "mean"),
            predictability_acc_2plus=("predictability_correct_2plus", "mean"),
            predictability_avg_last5_3pa=("last_5_games_3pa_avg", "mean"),
        )
        .reset_index()
    )
    # 8. Filter to players with minimum games played.
    grouped = grouped[grouped["predictability_games"] >= int(min_games)].copy()
    if grouped.empty:
        return pd.DataFrame(columns=["player_name", *PREDICTABILITY_COLUMNS])

    # 9. Build composite predictability score from MAE and Brier quality percentiles.
    score_parts: list[tuple[float, pd.Series]] = []
    mae_quality = _quality_percentile(grouped["predictability_mae_3pm"])
    if mae_quality.notna().any():
        score_parts.append((0.6, mae_quality))
    brier_quality = _quality_percentile(grouped["predictability_brier_2plus"])
    if brier_quality.notna().any():
        score_parts.append((0.4, brier_quality))

    # 10. Compute weighted average score; use NaN if no quality metrics available.
    if score_parts:
        total_weight = sum(weight for weight, _ in score_parts)
        score = pd.Series(0.0, index=grouped.index, dtype=float)
        for weight, values in score_parts:
            score = score.add(values.fillna(0.0) * weight, fill_value=0.0)
        grouped["predictability_score"] = score / total_weight
    else:
        grouped["predictability_score"] = np.nan

    # 11. Compute overall percentile rank based on predictability_score.
    grouped["predictability_percentile"] = grouped["predictability_score"].rank(
        pct=True,
        method="average",
        ascending=True,
    )
    # 12. Bucket players by their last-5-games 3PA attempt volume.
    grouped["predictability_attempt_bucket"] = pd.cut(
        grouped["predictability_avg_last5_3pa"],
        bins=PREDICTABILITY_BUCKET_BINS,
        labels=PREDICTABILITY_BUCKET_LABELS,
    )
    grouped["predictability_attempt_bucket"] = (
        grouped["predictability_attempt_bucket"].astype("object").fillna("unknown")
    )
    # 13. Compute bucket size and rank within each bucket.
    grouped["predictability_bucket_size"] = grouped.groupby("predictability_attempt_bucket")[
        "player_name"
    ].transform("size")
    grouped["predictability_bucket_percentile"] = grouped.groupby("predictability_attempt_bucket")[
        "predictability_score"
    ].rank(
        pct=True,
        method="average",
        ascending=True,
    )
    # 14. Mark as star if top percentile within bucket (if bucket >= 5) or overall (if bucket < 5).
    grouped["predictability_star"] = np.where(
        grouped["predictability_bucket_size"] >= 5,
        grouped["predictability_bucket_percentile"] >= float(star_percentile),
        grouped["predictability_percentile"] >= float(star_percentile),
    )
    # 15. Record the valid start date from model bundles (or empty if no date found).
    grouped["predictability_valid_start_date"] = (
        valid_start.date().isoformat() if valid_start is not None else ""
    )
    # 16. Generate player display name with asterisk if star.
    grouped["player_display_name"] = grouped.apply(
        lambda row: player_display_name(row.get("player_name"), row.get("predictability_star")),
        axis=1,
    )

    # 17. Reorder and fill missing columns; sort by star status and score.
    ordered_columns = ["player_name", *PREDICTABILITY_COLUMNS]
    for column in ordered_columns:
        if column not in grouped.columns:
            grouped[column] = np.nan
    grouped = grouped[ordered_columns].copy()
    grouped.sort_values(
        ["predictability_star", "predictability_score", "predictability_games", "player_name"],
        ascending=[False, False, False, True],
        inplace=True,
    )
    return grouped


def annotate_predictability(frame: pd.DataFrame, predictability: pd.DataFrame) -> pd.DataFrame:
    # 1. Create working copy of the input frame.
    annotated = frame.copy()
    # 2. Return unchanged if no player_name column exists.
    if "player_name" not in annotated.columns:
        return annotated
    # 3. If predictability table is empty, add basic display names without stars.
    if predictability.empty or "player_name" not in predictability.columns:
        annotated["player_display_name"] = annotated["player_name"].map(
            lambda value: player_display_name(value, False)
        )
        return annotated

    # 4. Identify columns to merge from predictability (player_name + available PREDICTABILITY_COLUMNS).
    merge_columns = ["player_name", *[column for column in PREDICTABILITY_COLUMNS if column in predictability.columns]]
    # 5. Left-merge predictability metrics onto the frame, keeping latest row per player.
    annotated = annotated.merge(
        predictability[merge_columns].drop_duplicates(subset=["player_name"], keep="last"),
        on="player_name",
        how="left",
    )
    # 6. Set player_display_name using star status from merged predictability data.
    annotated["player_display_name"] = annotated.apply(
        lambda row: row.get("player_display_name")
        if pd.notna(row.get("player_display_name")) and str(row.get("player_display_name", "")).strip()
        else player_display_name(row.get("player_name"), row.get("predictability_star")),
        axis=1,
    )
    return annotated
