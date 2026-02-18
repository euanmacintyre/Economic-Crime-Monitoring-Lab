from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class ModelArtifacts:
    model: Pipeline
    scored_transactions: pd.DataFrame
    metrics: dict[str, Any]
    top_features: pd.DataFrame
    model_metrics_weekly: pd.DataFrame
    feature_drift_overall: pd.DataFrame
    feature_drift_weekly: pd.DataFrame


NUMERIC_FEATURES = [
    "amount",
    "amount_log",
    "velocity_1h",
    "small_txn_24h",
    "customer_prev_mean_amount",
    "amount_to_customer_mean",
    "is_new_device",
    "is_new_country",
    "is_unusual_country",
    "rapid_in_then_out",
    "merchant_cash_like",
    "time_since_prev_customer_min",
    "hour",
    "day_of_week",
    "is_weekend",
    "rule_hit_count",
    "rules_risk_score",
]

CATEGORICAL_FEATURES = [
    "channel",
    "country",
    "currency",
    "direction",
]

DRIFT_FEATURES = [
    "amount",
    "velocity_1h",
    "amount_to_customer_mean",
    "is_new_device",
    "rapid_in_then_out",
]


def _build_model_pipeline() -> Pipeline:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )

    classifier = LogisticRegression(max_iter=500, class_weight="balanced", solver="lbfgs")

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def _extract_top_features(model: Pipeline, limit: int = 12) -> pd.DataFrame:
    preprocessor: ColumnTransformer = model.named_steps["preprocessor"]
    classifier: LogisticRegression = model.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    coefs = classifier.coef_[0]

    importance = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
    importance["abs_coefficient"] = importance["coefficient"].abs()
    importance = importance.sort_values("abs_coefficient", ascending=False).head(limit)
    importance["direction"] = np.where(
        importance["coefficient"] >= 0,
        "higher risk",
        "lower risk",
    )
    return importance[["feature", "coefficient", "direction"]]


def _compute_feature_drift_overall(df: pd.DataFrame) -> pd.DataFrame:
    train_df = df[df["data_split"] == "train"]
    test_df = df[df["data_split"] == "test"]

    rows: list[dict[str, Any]] = []
    for feature in DRIFT_FEATURES:
        train_mean = float(train_df[feature].mean()) if not train_df.empty else np.nan
        test_mean = float(test_df[feature].mean()) if not test_df.empty else np.nan
        mean_shift = test_mean - train_mean
        relative_shift = (
            np.nan
            if np.isnan(train_mean) or train_mean == 0.0
            else mean_shift / abs(train_mean)
        )
        rows.append(
            {
                "feature": feature,
                "train_mean": train_mean,
                "test_mean": test_mean,
                "mean_shift": mean_shift,
                "relative_shift": relative_shift,
            }
        )

    return pd.DataFrame(rows)


def _compute_model_metrics_weekly(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["week_start"] = work["timestamp"].dt.to_period("W").apply(lambda x: x.start_time)

    rows: list[dict[str, Any]] = []
    for week_start, group in work.groupby("week_start"):
        if group["is_suspicious"].nunique() < 2:
            auc = np.nan
        else:
            auc = float(roc_auc_score(group["is_suspicious"], group["ml_score"]))

        rows.append(
            {
                "week_start": pd.Timestamp(week_start),
                "auc": auc,
                "transactions": int(len(group)),
                "positive_rate": float(group["is_suspicious"].mean()),
            }
        )

    return pd.DataFrame(rows).sort_values("week_start")


def _compute_feature_drift_weekly(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["week_start"] = work["timestamp"].dt.to_period("W").apply(lambda x: x.start_time)
    train_df = work[work["data_split"] == "train"]

    baseline_means = {
        feature: float(train_df[feature].mean()) if not train_df.empty else np.nan
        for feature in DRIFT_FEATURES
    }

    rows: list[dict[str, Any]] = []
    for week_start, group in work.groupby("week_start"):
        for feature in DRIFT_FEATURES:
            train_mean = baseline_means[feature]
            week_mean = float(group[feature].mean())
            mean_shift = week_mean - train_mean
            relative_shift = (
                np.nan
                if np.isnan(train_mean) or train_mean == 0.0
                else mean_shift / abs(train_mean)
            )
            rows.append(
                {
                    "week_start": pd.Timestamp(week_start),
                    "feature": feature,
                    "train_mean": train_mean,
                    "week_mean": week_mean,
                    "mean_shift": mean_shift,
                    "relative_shift": relative_shift,
                }
            )

    return pd.DataFrame(rows).sort_values(["week_start", "feature"])


def _build_local_feature_explanations(model: Pipeline, x_all: pd.DataFrame) -> list[str]:
    preprocessor: ColumnTransformer = model.named_steps["preprocessor"]
    classifier: LogisticRegression = model.named_steps["classifier"]

    transformed = preprocessor.transform(x_all)
    feature_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]

    if hasattr(transformed, "multiply"):
        contributions = transformed.multiply(coefficients)
        is_sparse = True
    else:
        contributions = transformed * coefficients
        is_sparse = False

    explanations: list[str] = []
    for row_idx in range(len(x_all)):
        if is_sparse:
            row = contributions.getrow(row_idx)
            row_values = row.data
            row_indices = row.indices
        else:
            dense_row = np.asarray(contributions[row_idx]).ravel()
            non_zero = np.nonzero(dense_row)[0]
            row_indices = non_zero
            row_values = dense_row[non_zero]

        if len(row_values) == 0:
            explanations.append("[]")
            continue

        top_order = np.argsort(np.abs(row_values))[::-1][:5]
        payload = [
            {
                "feature": str(feature_names[int(row_indices[pos])]),
                "contribution": float(row_values[pos]),
                "direction": "up" if float(row_values[pos]) >= 0 else "down",
            }
            for pos in top_order
        ]
        explanations.append(json.dumps(payload))

    return explanations


def train_and_score_model(df: pd.DataFrame, train_ratio: float = 0.6) -> ModelArtifacts:
    model_df = df.copy()
    model_df["timestamp"] = pd.to_datetime(model_df["timestamp"], utc=True)
    model_df = model_df.sort_values("timestamp").reset_index(drop=True)

    split_idx = int(len(model_df) * train_ratio)
    split_ts = model_df.iloc[max(split_idx - 1, 0)]["timestamp"]
    model_df["data_split"] = np.where(model_df["timestamp"] <= split_ts, "train", "test")

    train_df = model_df[model_df["data_split"] == "train"]
    if train_df["is_suspicious"].nunique() < 2:
        raise ValueError(
            "Training data has only one class; adjust synthetic generation parameters."
        )

    x_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train_df["is_suspicious"]
    x_all = model_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    model = _build_model_pipeline()
    model.fit(x_train, y_train)

    model_df["ml_score"] = model.predict_proba(x_all)[:, 1]
    model_df["ml_pred"] = (model_df["ml_score"] >= 0.5).astype(int)
    model_df["ml_top_features"] = _build_local_feature_explanations(model, x_all)

    test_df = model_df[model_df["data_split"] == "test"]
    y_test = test_df["is_suspicious"]
    auc = float(roc_auc_score(y_test, test_df["ml_score"])) if y_test.nunique() > 1 else np.nan
    precision = float(precision_score(y_test, test_df["ml_pred"], zero_division=0))
    recall = float(recall_score(y_test, test_df["ml_pred"], zero_division=0))
    cm = confusion_matrix(y_test, test_df["ml_pred"], labels=[0, 1])

    metrics: dict[str, Any] = {
        "train_period_end": str(split_ts),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "auc": float(auc) if not np.isnan(auc) else np.nan,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
    }

    top_features = _extract_top_features(model)
    model_metrics_weekly = _compute_model_metrics_weekly(model_df)
    feature_drift_overall = _compute_feature_drift_overall(model_df)
    feature_drift_weekly = _compute_feature_drift_weekly(model_df)

    return ModelArtifacts(
        model=model,
        scored_transactions=model_df,
        metrics=metrics,
        top_features=top_features,
        model_metrics_weekly=model_metrics_weekly,
        feature_drift_overall=feature_drift_overall,
        feature_drift_weekly=feature_drift_weekly,
    )

