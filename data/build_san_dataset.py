"""Build the San Diego organ-transport flight dataset from BTS monthly files.

Usage (put the downloaded BTS zips in the same folder as this script):
    pip install pandas
    python build_san_dataset.py *.zip

Outputs (in ./san_dataset_out/):
    flights_into_SAN.csv   one row per flight arriving at SAN (+ hub arrivals if --hubs)
    san_routes.csv         per airline + origin: flights/day, scheduled flight time, typical departure times, delay rates
    organ_windows.csv      viability windows
    san_hospitals.csv      transplant centers + blank drive-time columns to fill from Google Maps
    data_dictionary.md     column definitions, date range, row counts, limitations (auto-generated)
"""
import argparse, io, sys, zipfile
from pathlib import Path
import pandas as pd

DEST = "SAN"
COLS = {  # standard name: possible BTS column names (PREZIP files vs. Download page)
    "flight_date": ["FlightDate", "FL_DATE"],
    "carrier": ["Reporting_Airline", "OP_UNIQUE_CARRIER", "OP_CARRIER"],
    "flight_number": ["Flight_Number_Reporting_Airline", "OP_CARRIER_FL_NUM"],
    "origin": ["Origin", "ORIGIN"],
    "dest": ["Dest", "DEST"],
    "crs_dep_time": ["CRSDepTime", "CRS_DEP_TIME"],
    "crs_arr_time": ["CRSArrTime", "CRS_ARR_TIME"],
    "crs_elapsed_min": ["CRSElapsedTime", "CRS_ELAPSED_TIME"],
    "distance_mi": ["Distance", "DISTANCE"],
    "arr_delay_min": ["ArrDelay", "ARR_DELAY"],
    "cancelled": ["Cancelled", "CANCELLED"],
    "cancellation_code": ["CancellationCode", "CANCELLATION_CODE"],
    "diverted": ["Diverted", "DIVERTED"],
    "carrier_delay_min": ["CarrierDelay", "CARRIER_DELAY"],
    "weather_delay_min": ["WeatherDelay", "WEATHER_DELAY"],
    "nas_delay_min": ["NASDelay", "NAS_DELAY"],
    "security_delay_min": ["SecurityDelay", "SECURITY_DELAY"],
    "late_aircraft_delay_min": ["LateAircraftDelay", "LATE_AIRCRAFT_DELAY"],
}
REQUIRED = ["flight_date", "carrier", "origin", "dest", "crs_dep_time", "arr_delay_min", "cancelled", "diverted"]

HOSPITALS = [
    ("UC San Diego Health - Jacobs Medical Center", "La Jolla", "heart; kidney; pancreas; lung; liver", "Confirm which UCSD campus receives organs"),
    ("Sharp Memorial Hospital", "Kearny Mesa", "heart; kidney; pancreas; liver (joint program with UCSD)", ""),
    ("Scripps Green Hospital", "La Jolla", "kidney; liver; pancreas", ""),
    ("Rady Children's Hospital", "Kearny Mesa area", "heart; kidney; liver (pediatric)", ""),
]
ORGANS = [  # organ, min h, max h, usually commercial air
    ("kidney", 24, 36, "yes"), ("pancreas", 12, 18, "sometimes"), ("liver", 8, 12, "rarely"),
    ("intestine", 8, 16, "rarely"), ("lung", 4, 8, "no"), ("heart", 4, 6, "no"),
]
WINDOW_SRC = "https://www.life-source.org/latest/how-long-can-an-organ-be-outside-the-body-before-transplant/"
HOSP_SRC = "https://www.lifesharing.org/about-us/partners"


WANTED = {n for names in COLS.values() for n in names}


def read_file(path):
    """Read one BTS file, keeping only the columns we use (much faster, less memory)."""
    p = Path(path)
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as z:
            name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
            with z.open(name) as f:
                return pd.read_csv(io.TextIOWrapper(f, encoding="latin-1"), usecols=lambda c: c in WANTED, dtype=str)
    return pd.read_csv(p, usecols=lambda c: c in WANTED, dtype=str, encoding="latin-1")


def normalize(raw):
    out = {}
    for std, names in COLS.items():
        hit = next((n for n in names if n in raw.columns), None)
        out[std] = raw[hit] if hit else pd.Series([None] * len(raw), index=raw.index)
    missing = [c for c in REQUIRED if not any(n in raw.columns for n in COLS[c])]
    if missing:
        sys.exit(f"File is missing columns {missing}. Is it the 'Reporting Carrier On-Time Performance' dataset?")
    return pd.DataFrame(out)


def clean(d):
    d = d.copy()
    for c in ("carrier", "origin", "dest"):
        d[c] = d[c].str.strip().str.upper()
    d["flight_date"] = pd.to_datetime(d["flight_date"].str[:10])
    num = ["crs_elapsed_min", "distance_mi", "arr_delay_min", "cancelled", "diverted", "carrier_delay_min",
           "weather_delay_min", "nas_delay_min", "security_delay_min", "late_aircraft_delay_min"]
    for c in num:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["cancelled"] = d["cancelled"].fillna(0).astype(int)
    d["diverted"] = d["diverted"].fillna(0).astype(int)
    for c in ("crs_dep_time", "crs_arr_time"):  # keep as 4-digit text, e.g. "0630"; BTS uses 2400 for midnight
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).astype(int).replace(2400, 0).astype(str).str.zfill(4)
    d["dep_hour"] = d["crs_dep_time"].str[:2].astype(int)
    d["arr_hour"] = d["crs_arr_time"].str[:2].astype(int)
    d["month"] = d["flight_date"].dt.month
    d["day_of_week"] = d["flight_date"].dt.day_name()
    for m in (15, 60, 120, 240):
        d[f"late_{m}"] = ((d["arr_delay_min"] > m) | (d["cancelled"] == 1) | (d["diverted"] == 1)).astype(int)
    d["data_issue"] = (d["arr_delay_min"].isna() & (d["cancelled"] == 0) & (d["diverted"] == 0)).astype(int)
    d["flight_date"] = d["flight_date"].dt.strftime("%Y-%m-%d")
    order = ["flight_date", "month", "day_of_week", "carrier", "flight_number", "origin", "dest",
             "crs_dep_time", "dep_hour", "crs_arr_time", "arr_hour", "crs_elapsed_min", "distance_mi",
             "arr_delay_min", "cancelled", "cancellation_code", "diverted",
             "late_15", "late_60", "late_120", "late_240",
             "carrier_delay_min", "weather_delay_min", "nas_delay_min", "security_delay_min", "late_aircraft_delay_min",
             "data_issue"]
    return d[order]


def routes_table(f):
    keys = ["carrier", "origin", "dest"]
    days = f.groupby(keys)["flight_date"].nunique()
    g = f.groupby(keys).agg(
        flights=("flight_date", "size"),
        median_sched_flight_min=("crs_elapsed_min", "median"),
        pct_late_60=("late_60", "mean"), pct_late_120=("late_120", "mean"),
        pct_cancelled=("cancelled", "mean"), median_arr_delay_min=("arr_delay_min", "median"))
    g["days_operated"] = days
    g["flights_per_day"] = (g["flights"] / days).round(2)
    g["low_sample"] = (days < 20).astype(int)  # too few days to trust the percentages
    # typical timetable, per month (schedules change by season): departure times, rounded to
    # 15 min, that run on >=40% of that month's operating days. Format "1: 06:30 09:00 | 4: ..."
    t = f.assign(slot=(f["crs_dep_time"].str[:2].astype(int) * 60 + f["crs_dep_time"].str[2:].astype(int)) // 15 * 15)
    mdays = t.groupby(keys + ["month"])["flight_date"].nunique().rename("mdays")
    sl = t.groupby(keys + ["month", "slot"])["flight_date"].nunique().rename("n").reset_index()
    sl = sl.merge(mdays.reset_index(), on=keys + ["month"])
    sl = sl[sl["n"] / sl["mdays"] >= 0.4]
    per_month = sl.groupby(keys + ["month"])["slot"].apply(
        lambda s: " ".join(f"{x // 60:02d}:{x % 60:02d}" for x in sorted(s)))
    pm = per_month.reset_index()
    pm["txt"] = pm["month"].astype(str) + ": " + pm["slot"]
    g["typical_departures_local"] = pm.groupby(keys)["txt"].agg(" | ".join)
    for c in ("pct_late_60", "pct_late_120", "pct_cancelled"):
        g[c] = (g[c] * 100).round(1)
    return g.reset_index().sort_values("flights", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--hubs", type=int, default=0, help="also keep arrivals into the N busiest origin airports feeding SAN")
    ap.add_argument("--out", default="san_dataset_out")
    a = ap.parse_args()
    here = Path(__file__).parent
    if not a.files:  # no files given (e.g. VS Code Run button): use every zip/csv next to this script
        a.files = [str(p) for p in sorted(here.glob("*.zip")) + sorted(here.glob("*.csv"))]
        a.hubs = a.hubs or 5
    if not a.files:
        sys.exit(f"No .zip or .csv files found in {here}")
    out = here / a.out; out.mkdir(exist_ok=True)

    parts, monthly, hub_parts = [], [], []
    for fp in sorted(a.files):
        d = normalize(read_file(fp))
        d["dest"] = d["dest"].str.strip().str.upper()
        total = len(d)
        san = d[d["dest"] == DEST]
        parts.append(san)
        if a.hubs:
            hub_parts.append(d)  # filtered after we know which hubs
        monthly.append((Path(fp).name, total, len(san)))
        print(f"{Path(fp).name}: {total:,} US flights -> {len(san):,} into SAN", file=sys.stderr)

    f = clean(pd.concat(parts, ignore_index=True))
    hubs = []
    if a.hubs:
        hubs = f["origin"].value_counts().head(a.hubs).index.tolist()
        allf = pd.concat(hub_parts, ignore_index=True)
        allf = allf[allf["dest"].isin(hubs)]
        f = pd.concat([f, clean(allf)], ignore_index=True)
        print(f"Added arrivals into hubs: {', '.join(hubs)}", file=sys.stderr)
    key = ["flight_date", "carrier", "flight_number", "origin", "dest", "crs_dep_time"]
    dupes = int(f.duplicated(key).sum())
    f = f.drop_duplicates(key)
    f = f.sort_values(["flight_date", "dest", "crs_arr_time"])
    f.to_csv(out / "flights_into_SAN.csv", index=False)
    r = routes_table(f)
    r.to_csv(out / "san_routes.csv", index=False)
    pd.DataFrame(ORGANS, columns=["organ", "window_min_hours", "window_max_hours", "usually_commercial_air"]).assign(
        source=WINDOW_SRC).to_csv(out / "organ_windows.csv", index=False)
    hosp = out / "san_hospitals.csv"
    if hosp.exists():  # never overwrite drive times you've already filled in
        print(f"Kept your existing {hosp.name} (not overwritten).", file=sys.stderr)
    else:
        pd.DataFrame([dict(hospital=h, area=ar, organs=o, address="", drive_min_from_SAN_offpeak_2am="",
                           drive_min_from_SAN_peak_5pm_weekday="", notes=n, source=HOSP_SRC) for h, ar, o, n in HOSPITALS]
                     ).to_csv(hosp, index=False)

    san = f[f["dest"] == DEST]
    top = r[r["dest"] == DEST].head(10)
    issues = int(f["data_issue"].sum())
    dd = f"""# San Diego organ-transport flight dataset: data dictionary

**Source:** U.S. Bureau of Transportation Statistics, Reporting Carrier On-Time Performance (transtats.bts.gov).
**Built:** {pd.Timestamp.today().date()} with build_san_dataset.py
**Date range:** {f['flight_date'].min()} to {f['flight_date'].max()}
**Rows:** {len(f):,} total; {len(san):,} arriving at SAN{'; plus arrivals into hubs ' + ', '.join(hubs) if hubs else ''}
**Duplicates removed:** {dupes} | **Rows with missing delay but not cancelled/diverted (data_issue=1):** {issues}

## Files
| File | What it is |
|---|---|
| flights_into_SAN.csv | One row per scheduled flight arriving at SAN{' (and at the hub airports)' if hubs else ''} |
| san_routes.csv | One row per airline + origin + destination: volume, typical timetable, delay and cancellation rates |
| organ_windows.csv | Organ viability windows (hours) and whether each usually travels on commercial flights |
| san_hospitals.csv | San Diego transplant centers and organs; drive times from SAN to be filled from Google Maps |

## flights_into_SAN.csv columns
| Column | Meaning |
|---|---|
| flight_date | Scheduled flight date (YYYY-MM-DD, origin local date) |
| month, day_of_week | Derived from flight_date |
| carrier | Reporting airline code (WN=Southwest, AA=American, DL=Delta, UA=United, AS=Alaska, B6=JetBlue, NK=Spirit, F9=Frontier, HA=Hawaiian, OO=SkyWest...) |
| flight_number | Airline flight number |
| origin, dest | Airport codes |
| crs_dep_time / crs_arr_time | Scheduled departure / arrival, **local time** at each airport, 4-digit text "HHMM" |
| dep_hour / arr_hour | Hour of scheduled departure / arrival (0-23) |
| crs_elapsed_min | Scheduled gate-to-gate minutes |
| distance_mi | Route distance in miles |
| arr_delay_min | Minutes late at arrival (negative = early). Blank if cancelled or diverted |
| cancelled / diverted | 1 = yes |
| cancellation_code | A = airline, B = weather, C = National Air System, D = security |
| late_15 / late_60 / late_120 / late_240 | 1 if arrived more than N minutes late OR was cancelled/diverted |
| *_delay_min (5 columns) | Minutes of delay by cause; only reported when arrival delay is 15+ min |
| data_issue | 1 = delay missing without a cancellation/diversion (exclude from analysis) |

## Busiest routes into SAN in this data
| Airline | From | Flights/day | % late >60 min or cancelled | % cancelled |
|---|---|---|---|---|
""" + "\n".join(f"| {x.carrier} | {x.origin} | {x.flights_per_day} | {x.pct_late_60}% | {x.pct_cancelled}% |" for x in top.itertuples()) + f"""

## Rows per input file
| File | US flights | Into SAN |
|---|---|---|
""" + "\n".join(f"| {n} | {t:,} | {s:,} |" for n, t, s in monthly) + """

## Limitations
- Commercial flights only; charter flights (how hearts, lungs and most livers travel) are not in BTS.
- Direct legs into SAN only, unless hub arrivals were added. A connecting organ's first-leg delay is not captured otherwise.
- BTS measures aircraft delays, not organ-handling failures (organs not loaded, left on board, unclaimed at cargo).
- Carriers below the BTS reporting threshold are not included; some regional flights report under their partner airline.
- Times are local to each airport.
- Drive times in san_hospitals.csv must be filled from Google Maps and dated.
"""
    (out / "data_dictionary.md").write_text(dd)
    print(f"\nDone: {len(f):,} rows -> {out}/", file=sys.stderr)


if __name__ == "__main__":
    main()