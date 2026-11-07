"""Convert our cleaned dataset into the flights_san.csv that core.load_flights() expects.

Run once from the repo folder:  python data_prep.py
Reads  data/san_dataset_out/flights_into_SAN.csv (our dataset)
Writes flights_san.csv (BTS column names, direct flights into SAN only)
"""
from pathlib import Path
import sys
import pandas as pd

HERE = Path(__file__).parent
candidates = [HERE / "data/san_dataset_out/flights_into_SAN.csv", HERE / "san_dataset_out/flights_into_SAN.csv",
              HERE / "flights_into_SAN.csv"]
src = next((p for p in candidates if p.exists()), None)
if src is None:
    sys.exit("Could not find flights_into_SAN.csv. Put it in data/san_dataset_out/ or next to this script.")

f = pd.read_csv(src, dtype={"crs_dep_time": str})
f = f[f["dest"] == "SAN"]                      # core.py assumes every row lands at SAN
out = pd.DataFrame({
    "Reporting_Airline": f["carrier"],
    "Origin": f["origin"],
    "Dest": f["dest"],
    "CRSDepTime": f["crs_dep_time"].astype(int),
    "CRSElapsedTime": f["crs_elapsed_min"],
    "ArrDelay": f["arr_delay_min"],
    # A diverted flight never reaches SAN on time; count it as a failure like a cancellation.
    # (core.py would otherwise fill its blank delay with 0 = "on time".)
    "Cancelled": ((f["cancelled"] == 1) | (f["diverted"] == 1)).astype(int),
    "DayOfWeek": pd.to_datetime(f["flight_date"]).dt.dayofweek + 1,  # BTS: 1 = Monday
    "Month": f["month"],
})
out.to_csv(HERE / "flights_san.csv", index=False)
print(f"Wrote flights_san.csv: {len(out):,} flights into SAN from {out['Origin'].nunique()} airports "
      f"({src.relative_to(HERE)})")