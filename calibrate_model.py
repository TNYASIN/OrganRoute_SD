
import copy
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from model import (
    DATA_PATH,
    MODEL_PATH,
    THRESHOLDS,
    _prepare,
    _time_splits,
    _make_features,
)

CALIBRATED_PATH = "flight_delay_model_calibrated.pkl"


class SigmoidCalibrator:
    """Calibrate predicted probabilities using logistic regression."""

    def __init__(self):
        self.model = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
        )
        self.constant = None

    def fit(self, probabilities, outcomes):
        p = np.asarray(probabilities, dtype=float)
        y = np.asarray(outcomes, dtype=int)

        if len(np.unique(y)) < 2:
            self.constant = float(y.mean())
            return self

        # Work on the log-odds of the original probabilities.
        p = np.clip(p, 1e-6, 1 - 1e-6)
        logits = np.log(p / (1 - p)).reshape(-1, 1)

        self.model.fit(logits, y)
        return self

    def predict(self, probabilities):
        p = np.asarray(probabilities, dtype=float)

        if self.constant is not None:
            return np.full(len(p), self.constant)

        p = np.clip(p, 1e-6, 1 - 1e-6)
        logits = np.log(p / (1 - p)).reshape(-1, 1)

        return self.model.predict_proba(logits)[:, 1]


def get_target_data(part, name):
    if name == "cancelled":
        subset = part
        y = subset["cancelled"].astype(int)

    elif name == "diverted":
        subset = part[part["cancelled"] == 0].copy()
        y = subset["diverted"].astype(int)

    else:
        threshold = int(name.split("_")[-1])
        subset = part[part["completed"]].copy()
        y = (subset["ArrDelay"] > threshold).astype(int)

    return subset, y


def evaluate(y, probabilities):
    y = np.asarray(y, dtype=int)
    p = np.asarray(probabilities, dtype=float)

    return {
        "brier": round(float(brier_score_loss(y, p)), 5),
        "predicted_rate": round(float(p.mean()), 5),
        "actual_rate": round(float(y.mean()), 5),
        "roc_auc": (
            round(float(roc_auc_score(y, p)), 5)
            if len(np.unique(y)) == 2
            else None
        ),
    }


def main():
    print("Loading dataset and trained models...", flush=True)

    df = _prepare(pd.read_csv(DATA_PATH))
    train, val, test = _time_splits(df)

    original = joblib.load(MODEL_PATH)

    calibrated = copy.copy(original)
    calibrated["calibrators"] = {}
    calibrated["calibration_method"] = "sigmoid"
    calibrated["version"] = 4

    san_test = test[test["Dest"] == "SAN"].copy()

    print("Validation flights:", len(val), flush=True)
    print("SAN test flights:", len(san_test), flush=True)

    if san_test.empty:
        raise ValueError("No SAN flights in the test set.")

    names = [
        "cancelled",
        "diverted",
        *[f"delay_gt_{t}" for t in THRESHOLDS],
    ]

    results = {}

    for name in names:
        val_subset, y_val = get_target_data(val, name)
        test_subset, y_test = get_target_data(san_test, name)

        if val_subset.empty or test_subset.empty:
            print(f"Skipping {name}: no eligible flights")
            continue

        # Fit calibration using validation data ONLY.
        X_val = _make_features(val_subset)

        val_predictions = (
            original["models"][name]
            .predict_proba(X_val)[:, 1]
        )

        calibrator = SigmoidCalibrator()
        calibrator.fit(val_predictions, y_val)

        calibrated["calibrators"][name] = calibrator

        # Evaluate on held-out SAN test flights.
        X_test = _make_features(test_subset)

        original_predictions = (
            original["models"][name]
            .predict_proba(X_test)[:, 1]
        )

        calibrated_predictions = calibrator.predict(
            original_predictions
        )

        original_scores = evaluate(
            y_test, original_predictions
        )

        calibrated_scores = evaluate(
            y_test, calibrated_predictions
        )

        results[name] = {
            "original": original_scores,
            "calibrated": calibrated_scores,
        }

        before = original_scores["brier"]
        after = calibrated_scores["brier"]

        print(
            f"{name:15s} "
            f"Brier {before:.5f} -> {after:.5f} "
            f"{'BETTER' if after < before else 'NOT BETTER'}",
            flush=True,
        )

    calibrated["calibration_results_san"] = results

    joblib.dump(calibrated, CALIBRATED_PATH)

    with open("calibration_results_san.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved:", CALIBRATED_PATH)
    print("Saved: calibration_results_san.json")
    print("\nCalibration complete.")


if __name__ == "__main__":
    main()
