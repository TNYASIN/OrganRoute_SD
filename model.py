
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = "flights_san.csv"
MODEL_PATH = "flight_delay_model.pkl"

FEATURES = [
    "Origin",
    "Reporting_Airline",
    "departure_hour",
    "DayOfWeek",
    "Month",
]

CATEGORICAL = ["Origin", "Reporting_Airline"]
NUMERIC = ["departure_hour", "DayOfWeek", "Month"]

THRESHOLDS = (0, 15, 30, 45, 60, 90, 120, 180, 240)


# ============================================================
# 2. CONSTANT PROBABILITY MODEL
# ============================================================

class ConstantProbability:
    """Fallback when training data contains only one outcome."""

    def __init__(self, probability):
        self.probability = float(probability)

    def predict_proba(self, X):
        n = len(X)
        p = self.probability
        return np.column_stack([
            np.full(n, 1.0 - p),
            np.full(n, p),
        ])


# ============================================================
# 3. FEATURE PREPARATION
# ============================================================

def _make_features(df):
    X = df[FEATURES].copy()

    for col in CATEGORICAL:
        X[col] = X[col].fillna("UNKNOWN").astype(str)

    for col in NUMERIC:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    return X


def _prepare(df):
    required = [
        "FlightDate",
        "Origin",
        "Reporting_Airline",
        "CRSDepTime",
        "DayOfWeek",
        "Month",
        "Cancelled",
        "Diverted",
        "ArrDelay",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required dataset columns: {missing}"
        )

    df = df.copy()

    df["FlightDate"] = pd.to_datetime(
        df["FlightDate"], errors="coerce"
    )

    numeric_cols = [
        "CRSDepTime",
        "DayOfWeek",
        "Month",
        "Cancelled",
        "Diverted",
        "ArrDelay",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[
        "FlightDate",
        "Origin",
        "Reporting_Airline",
        "CRSDepTime",
        "DayOfWeek",
        "Month",
        "Cancelled",
        "Diverted",
    ]).copy()

    df = df[
        df["Cancelled"].isin([0, 1])
        & df["Diverted"].isin([0, 1])
        & df["DayOfWeek"].between(1, 7)
        & df["Month"].between(1, 12)
        & df["CRSDepTime"].between(0, 2359)
    ].copy()

    hours = (df["CRSDepTime"] // 100).astype(int)
    minutes = (df["CRSDepTime"] % 100).astype(int)

    df = df[minutes < 60].copy()

    df["departure_hour"] = hours.loc[df.index]

    df["cancelled"] = df["Cancelled"].astype(int)
    df["diverted"] = df["Diverted"].astype(int)

    df["completed"] = (
        (df["cancelled"] == 0)
        & (df["diverted"] == 0)
        & df["ArrDelay"].notna()
    )

    df = df.sort_values("FlightDate").reset_index(drop=True)

    if len(df) < 100:
        raise ValueError(
            "Not enough usable flight records for training."
        )

    return df


# ============================================================
# 4. CHRONOLOGICAL DATA SPLIT
# ============================================================

def _time_splits(df):
    """Split by date: 60% train, 20% validation, 20% test."""

    dates = np.sort(df["FlightDate"].dt.normalize().unique())

    if len(dates) < 5:
        raise ValueError(
            "At least five distinct flight dates are required."
        )

    train_end = max(1, int(len(dates) * 0.60))
    val_end = max(train_end + 1, int(len(dates) * 0.80))

    if val_end >= len(dates):
        raise ValueError("Insufficient dates for test split.")

    train_cutoff = dates[train_end]
    test_cutoff = dates[val_end]

    train = df[df["FlightDate"] < train_cutoff].copy()

    val = df[
        (df["FlightDate"] >= train_cutoff)
        & (df["FlightDate"] < test_cutoff)
    ].copy()

    test = df[df["FlightDate"] >= test_cutoff].copy()

    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("One of the data splits is empty.")

    return train, val, test


# ============================================================
# 5. MACHINE LEARNING PIPELINES
# ============================================================

def _pipeline(kind):
    if kind == "logistic":
        preprocessor = ColumnTransformer([
            (
                "categorical",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent"),
                    ),
                    (
                        "encoder",
                        OneHotEncoder(handle_unknown="ignore"),
                    ),
                ]),
                CATEGORICAL,
            ),
            (
                "numeric",
                SimpleImputer(strategy="median"),
                NUMERIC,
            ),
        ])

        classifier = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
        )

    elif kind == "boosting":
        preprocessor = ColumnTransformer([
            (
                "categorical",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                CATEGORICAL,
            ),
            (
                "numeric",
                SimpleImputer(strategy="median"),
                NUMERIC,
            ),
        ])

        classifier = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        )

    else:
        raise ValueError(f"Unknown model type: {kind}")

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


# ============================================================
# 6. EVALUATION METRICS
# ============================================================

def _scores(y_true, probabilities):
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probabilities), 0.0, 1.0)

    result = {
        "brier": round(float(brier_score_loss(y, p)), 5),
        "positive_rate": round(float(np.mean(y)), 5),
        "samples": len(y),
    }

    if len(np.unique(y)) == 2:
        result["roc_auc"] = round(
            float(roc_auc_score(y, p)), 5
        )
    else:
        result["roc_auc"] = None

    return result


# ============================================================
# 7. TRAIN ONE PREDICTION TASK
# ============================================================

def _train_task(train, val, test, target):
    X_train = _make_features(train)
    X_val = _make_features(val)
    X_test = _make_features(test)

    y_train = train[target].astype(int)
    y_val = val[target].astype(int)
    y_test = test[target].astype(int)

    if y_train.nunique() < 2:
        model = ConstantProbability(y_train.mean())
        selected = "constant"

    else:
        candidates = {}

        for kind in ("logistic", "boosting"):
            model_candidate = _pipeline(kind)

            model_candidate.fit(X_train, y_train)

            predictions = model_candidate.predict_proba(
                X_val
            )[:, 1]

            score = brier_score_loss(y_val, predictions)

            candidates[kind] = (
                score,
                model_candidate,
            )

        selected = min(
            candidates,
            key=lambda name: candidates[name][0],
        )

        model = candidates[selected][1]

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = _scores(y_test, probabilities)
    metrics["selected_model"] = selected

    return model, metrics


# ============================================================
# 8. TRAIN ALL MODELS
# ============================================================

def train_model(
    data_path=DATA_PATH,
    model_path=MODEL_PATH,
):
    print("Loading flight dataset...")

    df = _prepare(pd.read_csv(data_path))

    train, val, test = _time_splits(df)

    print("Training flights:", len(train))
    print("Validation flights:", len(val))
    print("Testing flights:", len(test))

    models = {}
    metrics = {}

    # Cancellation model: all flights
    print("\nTraining cancellation model...")

    models["cancelled"], metrics["cancelled"] = _train_task(
        train, val, test, "cancelled"
    )

    # Diversion model: flights that were not cancelled
    print("Training diversion model...")

    operating = tuple(
        part[part["cancelled"] == 0].copy()
        for part in (train, val, test)
    )

    if min(len(part) for part in operating) == 0:
        raise ValueError(
            "Insufficient operating flights for diversion model."
        )

    models["diverted"], metrics["diverted"] = _train_task(
        *operating, "diverted"
    )

    # Arrival delay models: completed flights only
    print("Training arrival delay models...")

    complete = tuple(
        part[part["completed"]].copy()
        for part in (train, val, test)
    )

    if min(len(part) for part in complete) == 0:
        raise ValueError(
            "Insufficient completed flights for delay models."
        )

    for threshold in THRESHOLDS:
        target = f"delay_gt_{threshold}"

        for part in complete:
            part[target] = (
                part["ArrDelay"] > threshold
            ).astype(int)

        models[target], metrics[target] = _train_task(
            *complete, target
        )

        print(f"Trained delay > {threshold} minutes")

    bundle = {
        "version": 2,
        "features": FEATURES,
        "thresholds": THRESHOLDS,
        "models": models,
        "metrics": metrics,
        "train_end": str(train["FlightDate"].max().date()),
        "validation_end": str(val["FlightDate"].max().date()),
        "test_end": str(test["FlightDate"].max().date()),
    }

    joblib.dump(bundle, model_path)

    print("\nMODEL EVALUATION")
    print(json.dumps(metrics, indent=2))

    print("\nModel saved:", model_path)

    return bundle


# ============================================================
# 9. CONSISTENT DELAY PROBABILITIES
# ============================================================

def _monotonic_delay_probabilities(models, X):
    probabilities = np.array([
        models[f"delay_gt_{threshold}"].predict_proba(X)[:, 1]
        for threshold in THRESHOLDS
    ]).T

    # More allowed time cannot increase the predicted
    # probability of exceeding that delay threshold.
    return np.minimum.accumulate(probabilities, axis=1)


# ============================================================
# 10. PREDICT FLIGHT RISK
# ============================================================

def predict_risk(
    origin,
    airline,
    departure_hour,
    day_of_week,
    month,
    slack_minutes=60,
    model_path=MODEL_PATH,
):
    """
    Estimate flight outcome probabilities.

    slack_minutes is the time available after the
    flight's scheduled arrival.

    Supported range: 0 to 240 minutes.
    """

    hour = float(departure_hour)
    day = int(day_of_week)
    month = int(month)
    slack = float(slack_minutes)

    if not 0 <= hour <= 23:
        raise ValueError("departure_hour must be 0..23")

    if not 1 <= day <= 7:
        raise ValueError("day_of_week must be 1..7")

    if not 1 <= month <= 12:
        raise ValueError("month must be 1..12")

    if not 0 <= slack <= max(THRESHOLDS):
        raise ValueError(
            f"slack_minutes must be between "
            f"0 and {max(THRESHOLDS)}"
        )

    # Only load trusted model files.
    bundle = joblib.load(model_path)

    if (
        bundle.get("version") != 2
        or tuple(bundle.get("thresholds", ())) != THRESHOLDS
        or bundle.get("features") != FEATURES
    ):
        raise ValueError(
            "Incompatible saved model. Retrain with this model.py."
        )

    X = _make_features(pd.DataFrame([{
        "Origin": origin,
        "Reporting_Airline": airline,
        "departure_hour": hour,
        "DayOfWeek": day,
        "Month": month,
    }]))

    models = bundle["models"]

    # Probability of cancellation
    p_cancel = float(
        models["cancelled"].predict_proba(X)[0, 1]
    )

    # Probability of diversion given no cancellation
    p_divert_if_operated = float(
        models["diverted"].predict_proba(X)[0, 1]
    )

    # Probability of exceeding the allowed delay
    delay_curve = _monotonic_delay_probabilities(
        models, X
    )[0]

    p_delay = float(
        np.interp(slack, THRESHOLDS, delay_curve)
    )

    # Four mutually exclusive flight outcomes
    p_divert = (
        (1.0 - p_cancel) * p_divert_if_operated
    )

    p_completed = (
        (1.0 - p_cancel)
        * (1.0 - p_divert_if_operated)
    )

    success = p_completed * (1.0 - p_delay)

    p_late_but_completed = p_completed * p_delay

    # Verify that all outcomes sum to 100%
    total = (
        p_cancel
        + p_divert
        + success
        + p_late_but_completed
    )

    if not np.isclose(total, 1.0, atol=1e-8):
        raise RuntimeError(
            f"Outcome probabilities do not sum to 1: {total}"
        )

    return {
        "probability_cancelled": float(p_cancel),
        "probability_diverted": float(p_divert),
        "probability_late_but_completed": float(
            p_late_but_completed
        ),
        "probability_delay_over_slack_if_completed": float(
            p_delay
        ),
        "probability_arrive_within_slack": float(success),
        "slack_minutes": slack,
        "delay_curve_conditional_on_completion": {
            str(t): float(p)
            for t, p in zip(THRESHOLDS, delay_curve)
        },
        "note": (
            "Historical estimate only. Not a guarantee "
            "or medical transport authorization."
        ),
    }


# ============================================================
# 11. RUN TRAINING
# ============================================================

if __name__ == "__main__":
    train_model()
