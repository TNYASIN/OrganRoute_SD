
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, brier_score_loss
from model import (
    DATA_PATH,
    MODEL_PATH,
    THRESHOLDS,
    _prepare,
    _time_splits,
    _make_features,
)

# Load the same data and saved models
df = _prepare(pd.read_csv(DATA_PATH))
train, val, test = _time_splits(df)
bundle = joblib.load(MODEL_PATH)

# Only evaluate SAN flights in the held-out test period
san = test[test["Dest"] == "SAN"].copy()

print("SAN TEST SET")
print("Flights:", len(san))
print("Date range:", san["FlightDate"].min(), "to",
      san["FlightDate"].max())

if san.empty:
    raise ValueError("No SAN flights in the test set.")

def evaluate(name, subset, target):
    if subset.empty:
        print(name, ": No eligible flights")
        return

    X = _make_features(subset)
    y = subset[target].astype(int).to_numpy()

    p = bundle["models"][name].predict_proba(X)[:, 1]

    # Baseline: constant event rate learned from training data
    if name == "cancelled":
        baseline_rate = train["cancelled"].mean()
    elif name == "diverted":
        baseline_rate = train.loc[
            train["cancelled"] == 0, "diverted"
        ].mean()
    else:
        threshold = int(name.split("_")[-1])
        completed_train = train[train["completed"]]
        baseline_rate = (
            completed_train["ArrDelay"] > threshold
        ).mean()

    baseline_p = np.full(len(y), baseline_rate)

    model_brier = brier_score_loss(y, p)
    baseline_brier = brier_score_loss(y, baseline_p)

    auc = (
        roc_auc_score(y, p)
        if len(np.unique(y)) == 2
        else float("nan")
    )

    print(f"\n{name}")
    print(f"  Test flights: {len(y):,}")
    print(f"  Actual event rate: {y.mean():.2%}")
    print(f"  Average predicted risk: {p.mean():.2%}")
    print(f"  ROC-AUC: {auc:.4f}")
    print(f"  Model Brier: {model_brier:.5f}")
    print(f"  Baseline Brier: {baseline_brier:.5f}")
    print(
        "  Beats baseline:",
        "YES" if model_brier < baseline_brier else "NO"
    )


# 1. Cancellation risk
evaluate("cancelled", san, "cancelled")

# 2. Diversion risk among non-cancelled flights
operating = san[san["cancelled"] == 0].copy()
evaluate("diverted", operating, "diverted")

# 3. Arrival delay risk among completed flights
completed = san[san["completed"]].copy()

for threshold in THRESHOLDS:
    name = f"delay_gt_{threshold}"
    completed[name] = (
        completed["ArrDelay"] > threshold
    ).astype(int)

    evaluate(name, completed, name)

print("\nSAN evaluation complete.")
