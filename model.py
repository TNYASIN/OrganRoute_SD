
"""
OrganRoute SD - Flight Risk Prediction Model

Predicts:
1. Cancellation probability
2. Diversion probability
3. Probability of arriving later than allowed slack
4. Probability of arriving within allowed slack

Uses historical flight data.
Not validated for real-world medical transport decisions.
"""

import os
import __main__

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = "data/san_dataset_out/flights_clean.csv"

TRAINED_MODEL_PATH = "flight_delay_model.pkl"
CALIBRATED_MODEL_PATH = "flight_delay_model_calibrated.pkl"

# Default predictions use the calibrated model.
MODEL_PATH = CALIBRATED_MODEL_PATH

FEATURES = [
    "Origin",
    "Reporting_Airline",
    "Dest",
    "departure_hour",
    "DayOfWeek",
    "Month",
]

CATEGORICAL = [
    "Origin",
    "Reporting_Airline",
    "Dest",
]

NUMERIC = [
    "departure_hour",
    "DayOfWeek",
    "Month",
]

THRESHOLDS = (0, 15, 30, 45, 60, 90, 120, 180, 240)


# ============================================================
# 2. DATA PREPARATION
# ============================================================

def _prepare(df):
    df = df.copy()

    rename_map = {
        "flight_date": "FlightDate",
        "carrier": "Reporting_Airline",
        "origin": "Origin",
        "dest": "Dest",
        "dep_hour": "departure_hour",
        "day_of_week": "DayOfWeek",
        "month": "Month",
        "arr_delay_min": "ArrDelay",
    }

    df = df.rename(columns=rename_map)

    required = [
        "FlightDate",
        "Reporting_Airline",
        "Origin",
        "Dest",
        "departure_hour",
        "DayOfWeek",
        "Month",
        "ArrDelay",
        "cancelled",
        "diverted",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing dataset columns: {missing}"
        )

    df["FlightDate"] = pd.to_datetime(
        df["FlightDate"],
        errors="coerce",
    )

    weekday_map = {
        "Monday": 1,
        "Tuesday": 2,
        "Wednesday": 3,
        "Thursday": 4,
        "Friday": 5,
        "Saturday": 6,
        "Sunday": 7,
    }

    df["DayOfWeek"] = df["DayOfWeek"].map(
        lambda x: weekday_map.get(x, x)
    )

    for column in [
        "departure_hour",
        "DayOfWeek",
        "Month",
        "ArrDelay",
        "cancelled",
        "diverted",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    for column in CATEGORICAL:
        df[column] = df[column].astype("string")

    df = df.dropna(
        subset=[
            "FlightDate",
            "Origin",
            "Reporting_Airline",
            "Dest",
            "departure_hour",
            "DayOfWeek",
            "Month",
            "cancelled",
            "diverted",
        ]
    ).copy()

    df = df[
        df["departure_hour"].between(0, 23)
        & df["DayOfWeek"].between(1, 7)
        & df["Month"].between(1, 12)
    ].copy()

    df["cancelled"] = df["cancelled"].astype(int)
    df["diverted"] = df["diverted"].astype(int)

    df["completed"] = (
        (df["cancelled"] == 0)
        & (df["diverted"] == 0)
        & df["ArrDelay"].notna()
    )

    return df.sort_values(
        "FlightDate"
    ).reset_index(drop=True)


def _make_features(df):
    X = df[FEATURES].copy()

    for column in CATEGORICAL:
        X[column] = X[column].astype(str)

    for column in NUMERIC:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    return X


# ============================================================
# 3. CHRONOLOGICAL DATA SPLIT
# ============================================================

def _time_splits(df):
    dates = np.sort(
        df["FlightDate"].dropna().unique()
    )

    if len(dates) < 3:
        raise ValueError(
            "Not enough dates for a time split."
        )

    first = max(1, int(len(dates) * 0.60))
    second = max(first + 1, int(len(dates) * 0.80))

    first = min(first, len(dates) - 2)
    second = min(second, len(dates) - 1)

    train = df[
        df["FlightDate"].isin(dates[:first])
    ].copy()

    val = df[
        df["FlightDate"].isin(
            dates[first:second]
        )
    ].copy()

    test = df[
        df["FlightDate"].isin(
            dates[second:]
        )
    ].copy()

    return train, val, test


# ============================================================
# 4. MODEL PIPELINES
# ============================================================

def _pipeline(kind):
    if kind == "logistic":

        preprocessor = ColumnTransformer([
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                CATEGORICAL,
            ),
            (
                "numeric",
                SimpleImputer(
                    strategy="median"
                ),
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
                SimpleImputer(
                    strategy="median"
                ),
                NUMERIC,
            ),
        ])

        classifier = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.05,
            max_leaf_nodes=15,
            random_state=42,
        )

    else:
        raise ValueError(
            f"Unknown model type: {kind}"
        )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def _scores(y, p):
    result = {
        "brier": float(
            brier_score_loss(y, p)
        ),
        "positive_rate": float(
            np.mean(y)
        ),
        "samples": int(len(y)),
    }

    result["roc_auc"] = (
        float(roc_auc_score(y, p))
        if len(np.unique(y)) == 2
        else None
    )

    return result


# ============================================================
# 5. TRAIN ONE TASK
# ============================================================

def _train_task(
    name,
    train,
    val,
    test,
    target,
):
    X_train = _make_features(train)
    X_val = _make_features(val)
    X_test = _make_features(test)

    y_train = np.asarray(
        target(train),
        dtype=int,
    )

    y_val = np.asarray(
        target(val),
        dtype=int,
    )

    y_test = np.asarray(
        target(test),
        dtype=int,
    )

    if len(np.unique(y_train)) < 2:
        raise ValueError(
            f"Only one class in {name}"
        )

    best_model = None
    best_kind = None
    best_brier = float("inf")

    for kind in ("logistic", "boosting"):
        model = _pipeline(kind)

        model.fit(X_train, y_train)

        p_val = model.predict_proba(
            X_val
        )[:, 1]

        score = brier_score_loss(
            y_val,
            p_val,
        )

        if score < best_brier:
            best_brier = score
            best_model = model
            best_kind = kind

    p_test = best_model.predict_proba(
        X_test
    )[:, 1]

    metrics = _scores(y_test, p_test)

    metrics["model_type"] = best_kind
    metrics["validation_brier"] = float(
        best_brier
    )

    print(
        f"{name:15s} "
        f"Brier={metrics['brier']:.5f} "
        f"Model={best_kind}",
        flush=True,
    )

    return best_model, metrics


# ============================================================
# 6. TRAIN ALL MODELS
# ============================================================

def train_model(
    data_path=DATA_PATH,
    model_path=TRAINED_MODEL_PATH,
):
    print(
        "Loading historical flights...",
        flush=True,
    )

    df = _prepare(
        pd.read_csv(data_path)
    )

    train, val, test = _time_splits(df)

    print("Train:", len(train))
    print("Validation:", len(val))
    print("Test:", len(test))

    models = {}
    metrics = {}

    model, score = _train_task(
        "cancelled",
        train,
        val,
        test,
        lambda d: d["cancelled"],
    )

    models["cancelled"] = model
    metrics["cancelled"] = score

    operating_train = train[
        train["cancelled"] == 0
    ]

    operating_val = val[
        val["cancelled"] == 0
    ]

    operating_test = test[
        test["cancelled"] == 0
    ]

    model, score = _train_task(
        "diverted",
        operating_train,
        operating_val,
        operating_test,
        lambda d: d["diverted"],
    )

    models["diverted"] = model
    metrics["diverted"] = score

    completed_train = train[
        train["completed"]
    ]

    completed_val = val[
        val["completed"]
    ]

    completed_test = test[
        test["completed"]
    ]

    for threshold in THRESHOLDS:
        name = f"delay_gt_{threshold}"

        model, score = _train_task(
            name,
            completed_train,
            completed_val,
            completed_test,
            lambda d, t=threshold: (
                d["ArrDelay"] > t
            ).astype(int),
        )

        models[name] = model
        metrics[name] = score

    bundle = {
        "version": 3,
        "models": models,
        "metrics": metrics,
        "features": FEATURES,
        "thresholds": THRESHOLDS,
    }

    joblib.dump(
        bundle,
        model_path,
    )

    print(
        f"\nSaved trained model: {model_path}"
    )

    return bundle


# ============================================================
# 7. LOAD MODEL - FIX SIGMOID CALIBRATOR ERROR
# ============================================================

def _load_model(model_path):
    """
    Load an existing model, including calibrated models
    saved when calibrate_model.py ran as __main__.
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    # The calibrated pickle references
    # __main__.SigmoidCalibrator.
    #
    # Register the same class before loading.
    # This avoids the AttributeError.
    from calibrate_model import SigmoidCalibrator

    __main__.SigmoidCalibrator = SigmoidCalibrator

    # Only load trusted local model files.
    bundle = joblib.load(model_path)

    if (
        bundle.get("version") not in (3, 4)
        or tuple(
            bundle.get("thresholds", ())
        ) != THRESHOLDS
        or list(
            bundle.get("features", ())
        ) != FEATURES
    ):
        raise ValueError(
            "Incompatible saved model."
        )

    if bundle["version"] == 4:
        required = {
            "cancelled",
            "diverted",
            *[
                f"delay_gt_{t}"
                for t in THRESHOLDS
            ],
        }

        if not required.issubset(
            bundle.get("calibrators", {})
        ):
            raise ValueError(
                "Calibrated model is missing calibrators."
            )

    return bundle


# ============================================================
# 8. APPLY CALIBRATION
# ============================================================

def _predict_probability(
    bundle,
    name,
    X,
):
    model = bundle["models"][name]

    raw_probability = model.predict_proba(
        X
    )[:, 1]

    calibrators = bundle.get(
        "calibrators",
        {},
    )

    if name in calibrators:
        probability = calibrators[
            name
        ].predict(raw_probability)
    else:
        probability = raw_probability

    return np.clip(
        np.asarray(
            probability,
            dtype=float,
        ),
        0.0,
        1.0,
    )


# ============================================================
# 9. DELAY PROBABILITY CURVE
# ============================================================

def _monotonic_delay_probabilities(
    bundle,
    X,
):
    thresholds = tuple(
        bundle["thresholds"]
    )

    probabilities = np.array([
        _predict_probability(
            bundle,
            f"delay_gt_{threshold}",
            X,
        )
        for threshold in thresholds
    ]).T

    return np.minimum.accumulate(
        probabilities,
        axis=1,
    )


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
    dest="SAN",
    loaded_bundle=None,
):
    """
    Predict flight outcomes using calibrated models.

    day_of_week:
        Monday=1 through Sunday=7

    slack_minutes:
        Allowed delay after scheduled arrival.
    """

    hour = float(departure_hour)
    day = int(day_of_week)
    month = int(month)
    slack = float(slack_minutes)

    if not 0 <= hour <= 23:
        raise ValueError(
            "departure_hour must be 0..23"
        )

    if not 1 <= day <= 7:
        raise ValueError(
            "day_of_week must be 1..7"
        )

    if not 1 <= month <= 12:
        raise ValueError(
            "month must be 1..12"
        )

    if not 0 <= slack <= max(THRESHOLDS):
        raise ValueError(
            f"slack_minutes must be "
            f"0..{max(THRESHOLDS)}"
        )

    # Load model with compatibility fix.
    bundle = loaded_bundle if loaded_bundle is not None else _load_model(model_path)
    bundle = _load_model(model_path)

    X = _make_features(
        pd.DataFrame([{
            "Origin": str(origin).upper(),
            "Reporting_Airline": str(
                airline
            ).upper(),
            "Dest": str(dest).upper(),
            "departure_hour": hour,
            "DayOfWeek": day,
            "Month": month,
        }])
    )

    # Cancellation probability
    p_cancel = float(
        _predict_probability(
            bundle,
            "cancelled",
            X,
        )[0]
    )

    # Diversion probability conditional on operation
    p_divert_if_operated = float(
        _predict_probability(
            bundle,
            "diverted",
            X,
        )[0]
    )

    # Delay probabilities conditional on completion
    delay_curve = (
        _monotonic_delay_probabilities(
            bundle,
            X,
        )[0]
    )

    p_delay_if_completed = float(
        np.interp(
            slack,
            THRESHOLDS,
            delay_curve,
        )
    )

    # Four mutually exclusive outcomes
    p_divert = (
        (1.0 - p_cancel)
        * p_divert_if_operated
    )

    p_completed = (
        (1.0 - p_cancel)
        * (1.0 - p_divert_if_operated)
    )

    p_late = (
        p_completed
        * p_delay_if_completed
    )

    p_success = (
        p_completed
        * (1.0 - p_delay_if_completed)
    )

    total = (
        p_cancel
        + p_divert
        + p_late
        + p_success
    )

    if not np.isclose(
        total,
        1.0,
        atol=1e-8,
    ):
        raise RuntimeError(
            f"Probabilities sum to {total}"
        )

    return {
        "probability_cancelled": p_cancel,

        "probability_diverted": p_divert,

        "probability_late_but_completed": (
            p_late
        ),

        "probability_delay_over_slack_if_completed": (
            p_delay_if_completed
        ),

        "probability_arrive_within_slack": (
            p_success
        ),

        "slack_minutes": slack,

        "delay_curve_conditional_on_completion": {
            str(t): float(p)
            for t, p in zip(
                THRESHOLDS,
                delay_curve,
            )
        },

        "calibrated": (
            bundle["version"] == 4
        ),

        "note": (
            "Historical estimate only. "
            "Not a guarantee or medical transport "
            "authorization."
        ),
    }


# ============================================================
# 11. TRAIN ONLY WHEN RUN DIRECTLY
# ============================================================

if __name__ == "__main__":
    train_model()
