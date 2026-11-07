"""
Turn raw BTS on-time CSVs into flights_san.csv.

1. Go to https://www.transtats.bts.gov/  -> "Reporting Carrier On-Time Performance (1987-present)"
   (or the pre-zipped monthly files). Download 1-2 recent months.
2. Unzip, then:   python data_prep.py path/to/month1.csv path/to/month2.csv
"""
import sys

import pandas as pd

KEEP = ["FlightDate", "Reporting_Airline", "Origin", "Dest", "CRSDepTime", "CRSArrTime",
        "CRSElapsedTime", "ArrDelay", "Cancelled", "Diverted", "DayOfWeek", "Month", "Distance"]


def main(paths):
    parts = []
    for p in paths:
        head = pd.read_csv(p, nrows=0).columns
        use = [c for c in KEEP if c in head]
        for chunk in pd.read_csv(p, usecols=use, chunksize=500_000, low_memory=False):
            parts.append(chunk[chunk["Dest"] == "SAN"])
        print(f"read {p}")
    df = pd.concat(parts, ignore_index=True)
    df.to_csv("flights_san.csv", index=False)
    print(f"saved flights_san.csv: {len(df):,} flights into SAN from {df.Origin.nunique()} origins")
    print(df.groupby("Origin").size().sort_values(ascending=False).head(15))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
