
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

DATA_PATH = "flights_san.csv"
MODEL_PATH = "flight_delay_model.pkl"

FEATURES = [
    "Origin",
    "Reporting_Airline",
    "departure_hour",
    "DayOfWeek",
    "Month"
]

def train_model():
    df = pd.read_csv(DATA_PATH)

    # Keep completed, non-diverted flights
    df = df[
        (df["Cancelled"] == 0) &
        (df["Diverted"] == 0)
    ].copy()

    df = df.dropna(subset=[
        "Origin", "Reporting_Airline",
        "CRSDepTime", "ArrDelay",
        "DayOfWeek", "Month", "FlightDate"
    ])

    df["departure_hour"] = (
        df["CRSDepTime"].astype(int) // 100
    ).clip(0, 23)

    # Target: arrival more than 60 minutes late
    df["delayed_60"] = (df["ArrDelay"] > 60).astype(int)

    # Train on earlier flights, test on later flights
    df["FlightDate"] = pd.to_datetime(df["FlightDate"])
    df = df.sort_values("FlightDate")

    split = int(len(df) * 0.8)

    train = df.iloc[:split]
    test = df.iloc[split:]

    if train["delayed_60"].nunique() < 2:
        raise ValueError("Training data needs both delay classes.")
    if test["delayed_60"].nunique() < 2:
        raise ValueError("Test data needs both delay classes.")

    preprocessor = ColumnTransformer([
        (
            "airports_airlines",
            OneHotEncoder(handle_unknown="ignore"),
            ["Origin", "Reporting_Airline"]
        ),
        (
            "numeric",
            "passthrough",
            ["departure_hour", "DayOfWeek", "Month"]
        )
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(train[FEATURES], train["delayed_60"])

    probabilities = model.predict_proba(test[FEATURES])[:, 1]

    print("Training flights:", len(train))
    print("Testing flights:", len(test))
    print("ROC AUC:", round(
        roc_auc_score(test["delayed_60"], probabilities), 3
    ))
    print("Brier score:", round(
        brier_score_loss(test["delayed_60"], probabilities), 3
    ))

    joblib.dump(model, MODEL_PATH)
    print("Model saved:", MODEL_PATH)

def predict_risk(origin, airline, departure_hour,
                 day_of_week, month):

    model = joblib.load(MODEL_PATH)

    flight = pd.DataFrame([{
        "Origin": origin,
        "Reporting_Airline": airline,
        "departure_hour": departure_hour,
        "DayOfWeek": day_of_week,
        "Month": month
    }])

    probability = model.predict_proba(flight)[0][1]

    return float(probability)

if __name__ == "__main__":
    train_model()
