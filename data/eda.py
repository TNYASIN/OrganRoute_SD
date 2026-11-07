"""EDA + cleaning for the San Diego organ-transport flight dataset.
Reads:   san_dataset_out/flights_into_SAN.csv, san_routes.csv
Writes:  san_dataset_out/flights_clean.csv     cleaned flight table (adds arrival date/time, flags)
         san_dataset_out/eda/                  charts (PNG) + eda_summary.md
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save charts to files, no pop-up windows
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
DATA = HERE / "san_dataset_out"
OUT = DATA / "eda"; OUT.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
BLUE, RED, GREY = "#2F6DB5", "#C0392B", "#8A8A8A"
lines = ["# EDA summary: flights into San Diego (SAN)\n"]


def say(s=""):
    print(s); lines.append(s)


# ---------------------------------------------------------------- load
f = pd.read_csv(DATA / "flights_into_SAN.csv", dtype={"crs_dep_time": str, "crs_arr_time": str, "flight_number": str})
routes = pd.read_csv(DATA / "san_routes.csv")
say(f"Loaded {len(f):,} flights, {f.shape[1]} columns.")

# ---------------------------------------------------------------- CLEANING
say("\n## Cleaning steps")
n0 = len(f)

# 1. drop rows flagged as bad data (delay missing without a cancellation/diversion)
bad = f["data_issue"] == 1
f = f[~bad]
say(f"1. Dropped {bad.sum():,} rows with data_issue = 1.")

# 2. exact duplicates
d = f.duplicated(["flight_date", "carrier", "flight_number", "origin", "dest", "crs_dep_time"])
f = f[~d]
say(f"2. Dropped {d.sum():,} duplicate flights.")

# 3. impossible values
imp = (f["crs_elapsed_min"] <= 0) | (f["distance_mi"] <= 0) | ~f["dep_hour"].between(0, 23)
f = f[~imp]
say(f"3. Dropped {imp.sum():,} rows with impossible flight time, distance or hour.")

# 4. scheduled arrival as a real date-time (handles flights landing after midnight)
dep_min = f["crs_dep_time"].str[:2].astype(int) * 60 + f["crs_dep_time"].str[2:].astype(int)
arr_min = f["crs_arr_time"].str[:2].astype(int) * 60 + f["crs_arr_time"].str[2:].astype(int)
next_day = (arr_min < dep_min).astype(int)
f["arrives_next_day"] = next_day
f["sched_arr_local"] = (pd.to_datetime(f["flight_date"]) + pd.to_timedelta(next_day, unit="D")
                        + pd.to_timedelta(arr_min, unit="m")).dt.strftime("%Y-%m-%d %H:%M")
say(f"4. Added sched_arr_local (local arrival date-time); {next_day.sum():,} flights land the next calendar day.")

# 5. helpful flags (no rows removed)
f["is_direct_to_SAN"] = (f["dest"] == "SAN").astype(int)
f["outcome"] = "on_time_or_<15"
f.loc[f["arr_delay_min"] > 15, "outcome"] = "late_15_60"
f.loc[f["arr_delay_min"] > 60, "outcome"] = "late_60_plus"
f.loc[f["diverted"] == 1, "outcome"] = "diverted"
f.loc[f["cancelled"] == 1, "outcome"] = "cancelled"
f["dep_time_block"] = pd.cut(f["dep_hour"], [-1, 5, 11, 16, 20, 23],
                             labels=["overnight 0-5", "morning 6-11", "afternoon 12-16", "evening 17-20", "late 21-23"])
low = routes.loc[routes["low_sample"] == 1, ["carrier", "origin", "dest"]].assign(route_low_sample=1)
f = f.merge(low, on=["carrier", "origin", "dest"], how="left")
f["route_low_sample"] = f["route_low_sample"].fillna(0).astype(int)
say("5. Added flags: is_direct_to_SAN, outcome, dep_time_block, route_low_sample.")
say("   Long delays were kept on purpose: extreme delays are the events that threaten an organ, not noise.")
say(f"Rows: {n0:,} -> {len(f):,}")
f.to_csv(DATA / "flights_clean.csv", index=False)

# ---------------------------------------------------------------- EDA (direct flights into SAN)
s = f[f["is_direct_to_SAN"] == 1].copy()
say(f"\n## Flights into SAN: {len(s):,} flights, {s['origin'].nunique()} origin airports, {s['carrier'].nunique()} airlines")
say(f"- Late >60 min, cancelled or diverted: {s['late_60'].mean():.1%}")
say(f"- Late >120 min, cancelled or diverted: {s['late_120'].mean():.1%}")
say(f"- Cancelled: {s['cancelled'].mean():.2%} | Diverted: {s['diverted'].mean():.2%}")
q = s["arr_delay_min"].quantile([.5, .9, .95, .99])
say(f"- Arrival delay (minutes): median {q[.5]:.0f}, 90th pct {q[.9]:.0f}, 95th {q[.95]:.0f}, 99th {q[.99]:.0f}, max {s['arr_delay_min'].max():.0f}")

# chart 1: delay distribution
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.hist(s["arr_delay_min"].dropna().clip(-60, 300), bins=72, color=BLUE)
ax.axvline(60, color=RED, ls="--", lw=1); ax.text(63, ax.get_ylim()[1] * .9, "1 hour late", color=RED)
ax.set(title="Arrival delay, flights into SAN (clipped at 300 min)", xlabel="minutes late (negative = early)", ylabel="flights")
fig.tight_layout(); fig.savefig(OUT / "1_delay_distribution.png"); plt.close(fig)

# chart 2: risk by departure hour
by_h = s.groupby("dep_hour").agg(rate=("late_60", "mean"), n=("late_60", "size"))
by_h = by_h[by_h["n"] >= 100]
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.bar(by_h.index, by_h["rate"] * 100, color=BLUE)
ax.set_xticks(by_h.index)
ax.set(title="Share of flights >1 h late, cancelled or diverted, by scheduled departure hour",
       xlabel="scheduled departure hour (origin local time)", ylabel="%")
fig.tight_layout(); fig.savefig(OUT / "2_risk_by_departure_hour.png"); plt.close(fig)
say(f"\n**By departure hour:** lowest risk at {by_h['rate'].idxmin()}:00 ({by_h['rate'].min():.1%}), "
    f"highest at {by_h['rate'].idxmax()}:00 ({by_h['rate'].max():.1%}). Delays build up through the day.")

# chart 3: by month
by_m = s.groupby("month")[["late_60", "cancelled"]].mean() * 100
names = {1: "Jan", 4: "Apr", 7: "Jul", 10: "Oct"}
fig, ax = plt.subplots(figsize=(5.5, 3.2))
ax.bar([names.get(m, m) for m in by_m.index], by_m["late_60"], color=BLUE)
ax.set(title="Share >1 h late, cancelled or diverted, by month", ylabel="%")
fig.tight_layout(); fig.savefig(OUT / "3_risk_by_month.png"); plt.close(fig)
say("**By month:** " + ", ".join(f"{names.get(m, m)} {r:.1f}%" for m, r in by_m["late_60"].items()))

# chart 4: by airline (min 500 flights)
by_c = s.groupby("carrier").agg(rate=("late_60", "mean"), n=("late_60", "size"), canc=("cancelled", "mean"))
by_c = by_c[by_c["n"] >= 500].sort_values("rate")
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.barh(by_c.index, by_c["rate"] * 100, color=BLUE)
ax.set(title="Share >1 h late, cancelled or diverted, by airline (500+ flights into SAN)", xlabel="%")
fig.tight_layout(); fig.savefig(OUT / "4_risk_by_airline.png"); plt.close(fig)
say("**By airline (500+ flights):** " + ", ".join(f"{c} {r:.1%} (n={n:,})" for c, r, n in zip(by_c.index, by_c['rate'], by_c['n'])))

# chart 5: riskiest busy routes into SAN
r = routes[(routes["dest"] == "SAN") & (routes["low_sample"] == 0) & (routes["flights_per_day"] >= 1)]
top = r.sort_values("pct_late_60", ascending=False).head(12)
fig, ax = plt.subplots(figsize=(7, 4))
lab = top["carrier"] + " " + top["origin"] + "→SAN"
ax.barh(lab[::-1], top["pct_late_60"][::-1], color=RED)
ax.set(title="Riskiest routes into SAN (1+ flight/day, 20+ days of data)", xlabel="% >1 h late, cancelled or diverted")
fig.tight_layout(); fig.savefig(OUT / "5_riskiest_routes.png"); plt.close(fig)
say("**Riskiest regular routes into SAN:** " + ", ".join(f"{c} {o} {p}%" for c, o, p in zip(top['carrier'][:5], top['origin'][:5], top['pct_late_60'][:5])))

# chart 6: what causes the long delays (minutes, for flights >60 min late)
causes = ["carrier_delay_min", "weather_delay_min", "nas_delay_min", "security_delay_min", "late_aircraft_delay_min"]
long = s[s["arr_delay_min"] > 60]
share = long[causes].sum() / long[causes].sum().sum() * 100
pretty = {"carrier_delay_min": "Airline", "weather_delay_min": "Weather", "nas_delay_min": "Air traffic system",
          "security_delay_min": "Security", "late_aircraft_delay_min": "Late incoming aircraft"}
share.index = [pretty[c] for c in share.index]
share = share.sort_values()
fig, ax = plt.subplots(figsize=(6, 3))
ax.barh(share.index, share.values, color=BLUE)
ax.set(title="Causes of delay minutes, flights into SAN >1 h late", xlabel="% of delay minutes")
fig.tight_layout(); fig.savefig(OUT / "6_delay_causes.png"); plt.close(fig)
say("**Causes of long delays (share of minutes):** " + ", ".join(f"{k} {v:.0f}%" for k, v in share.sort_values(ascending=False).items()))
cc = s.loc[s["cancelled"] == 1, "cancellation_code"].map({"A": "airline", "B": "weather", "C": "air traffic system", "D": "security"}).value_counts()
say("**Cancellation reasons:** " + ", ".join(f"{k} {v}" for k, v in cc.items()))

# chart 7: tail risk -- what share of flights exceed X minutes late (incl. cancel/divert)
thr = list(range(0, 361, 15))
fail = s["cancelled"].eq(1) | s["diverted"].eq(1)
exceed = [((s["arr_delay_min"] > t) | fail).mean() * 100 for t in thr]
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(thr, exceed, color=BLUE, lw=2)
ax.set(title="Chance a flight into SAN is more than X minutes late (or cancelled/diverted)",
       xlabel="X (minutes)", ylabel="% of flights", yscale="log")
fig.tight_layout(); fig.savefig(OUT / "7_tail_risk_curve.png"); plt.close(fig)
say("**Tail risk:** " + ", ".join(f">{t} min: {e:.1f}%" for t, e in zip(thr, exceed) if t in (30, 60, 120, 180, 240, 360)))

say("\n## Data quality notes")
say("- Times are local to each airport; arrivals after midnight are handled in sched_arr_local.")
say("- route_low_sample = 1 marks routes with <20 days of data; treat their rates as unreliable.")
say("- BTS records aircraft delays only, not organ-handling failures, so these rates are a lower bound on risk.")
(OUT / "eda_summary.md").write_text("\n".join(lines) + "\n")
print(f"\nSaved flights_clean.csv and charts to {OUT}")