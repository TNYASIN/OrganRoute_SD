"""
OrganRoute SD - core logic (no UI here).

Owners:
  - load_flights / synthetic data ...... Data person (DS #1)
  - p_on_time (delay model) ............ Model person (DS #2 / Math-CS)
  - build_options / replan ............. Math-CS
  - llm_plan (Claude/OpenAI/Gemini) ..... App person (CS/CompE)

All times are handled as San Diego local time (datetime objects).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

try:  # load API keys / AI_PROVIDER from a local .env file if present
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Constants / assumptions (state these in the demo!)
# ---------------------------------------------------------------------------

# Max cold ischemic time in hours (approximate clinical ranges, conservative upper end).
VIABILITY_HOURS = {
    "Heart": 6,
    "Lung": 8,
    "Liver": 12,
    "Pancreas": 18,
    "Kidney": 36,
}

# Time needed at the receiving hospital before the organ must be in the OR (hours).
RECIPIENT_BUFFER_H = 1.0

# Origin airports: (lat, lon, hours offset vs San Diego in November / standard time)
AIRPORTS = {
    "LAX": (33.94, -118.41, 0), "SFO": (37.62, -122.38, 0), "SJC": (37.36, -121.93, 0),
    "OAK": (37.72, -122.22, 0), "SMF": (38.70, -121.59, 0), "LAS": (36.08, -115.15, 0),
    "SEA": (47.45, -122.31, 0), "PDX": (45.59, -122.60, 0), "PHX": (33.43, -112.01, 1),
    "DEN": (39.86, -104.67, 1), "SLC": (40.79, -111.98, 1), "DFW": (32.90, -97.04, 2),
    "ORD": (41.98, -87.90, 2), "ATL": (33.64, -84.43, 3), "JFK": (40.64, -73.78, 3),
}
SAN = (32.73, -117.19)
MYF = (32.82, -117.14)  # Montgomery-Gibbs Executive Airport (charter)

# San Diego transplant hospitals: minutes of ground time from SAN / from MYF
HOSPITALS = {
    "UC San Diego Health - La Jolla (Jacobs)": (25, 15),
    "UC San Diego Health - Hillcrest": (12, 15),
    "Scripps Memorial La Jolla": (22, 12),
    "Sharp Memorial": (18, 8),
}

ORIGIN_HOSPITAL_TO_AIRPORT_MIN = 45   # donor hospital -> origin airport
COMMERCIAL_CHECKIN_MIN = 45           # must be at airport this long before departure
CHARTER_CALLOUT_H = 1.5               # crew + aircraft ready time
CHARTER_MPH = 400
CHARTER_RELIABILITY = 0.95            # assumption: weather/mechanical
GROUND_MPH = 55
GROUND_ROAD_FACTOR = 1.25             # road distance vs straight line
GROUND_RELIABILITY = 0.97
GROUND_MAX_MILES = 400

CARRIER_NAMES = {"AA": "American", "AS": "Alaska", "DL": "Delta", "UA": "United",
                 "WN": "Southwest", "B6": "JetBlue", "F9": "Frontier", "NK": "Spirit",
                 "HA": "Hawaiian", "OO": "SkyWest", "G4": "Allegiant"}


def recovery_reference(organ: str, airport_codes: set[str], hospital: str | None = None) -> dict:
    """Expose legacy assumptions to the recovery agent without generating schedules.

    These constants are background context. The recovery mission's user-specified
    deadline remains authoritative, and historical delay data is not a live feed.
    """
    return {
        "organ_window_reference_hours": VIABILITY_HOURS.get(organ),
        "recipient_buffer_reference_hours": RECIPIENT_BUFFER_H,
        "airport_coordinates": {code: AIRPORTS[code][:2] for code in airport_codes
                                if code in AIRPORTS},
        "san_to_hospital_ground_minutes_reference": HOSPITALS[hospital][0]
        if hospital in HOSPITALS else None,
        "delay_estimator": "p_on_time: historical BTS or synthetic data; not live flight status",
    }


def haversine_miles(a, b) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    d = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * math.asin(math.sqrt(d))


# ---------------------------------------------------------------------------
# Data  (DS #1)
# ---------------------------------------------------------------------------

COLS = ["Reporting_Airline", "Origin", "Dest", "CRSDepTime", "CRSElapsedTime",
        "ArrDelay", "Cancelled", "DayOfWeek", "Month"]


def load_flights(path: str = "flights_san.csv") -> tuple[pd.DataFrame, bool]:
    """Returns (df, is_real). Falls back to synthetic data so the app always runs."""
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df[[c for c in COLS if c in df.columns]].dropna(subset=["CRSDepTime", "CRSElapsedTime"])
        df["ArrDelay"] = df["ArrDelay"].fillna(0)
        df["Cancelled"] = df["Cancelled"].fillna(0)
        df["dep_hour"] = (df["CRSDepTime"].astype(int) // 100).clip(0, 23)
        return df, True
    return synthetic_flights(), False


def synthetic_flights(days: int = 60, seed: int = 7) -> pd.DataFrame:
    """Placeholder data with BTS-like columns. REPLACE with real BTS data."""
    rng = np.random.default_rng(seed)
    airlines = ["WN", "AS", "AA", "UA", "DL"]
    rows = []
    for origin, (lat, lon, _) in AIRPORTS.items():
        miles = haversine_miles((lat, lon), SAN)
        elapsed = int(miles / 450 * 60 + 35)
        n_daily = 8 if miles < 600 else 4
        sched = sorted(rng.choice(range(6, 22), size=n_daily, replace=False))
        for h in sched:
            al = rng.choice(airlines)
            minute = int(rng.choice([0, 15, 30, 45]))
            for d in range(days):
                late_prob = 0.12 + 0.015 * max(0, h - 12)  # evenings worse
                if rng.random() < late_prob:
                    delay = rng.exponential(55)
                else:
                    delay = rng.normal(-6, 11)
                rows.append({
                    "Reporting_Airline": al, "Origin": origin, "Dest": "SAN",
                    "CRSDepTime": h * 100 + minute, "CRSElapsedTime": elapsed,
                    "ArrDelay": round(delay), "Cancelled": int(rng.random() < 0.015),
                    "DayOfWeek": (d % 7) + 1, "Month": 10 + d // 31,
                })
    df = pd.DataFrame(rows)
    df["dep_hour"] = df["CRSDepTime"] // 100
    return df


# ---------------------------------------------------------------------------
# Delay model  (DS #2 / Math-CS)
# P(arrive within `slack_min` of schedule) from the empirical delay distribution,
# using the most specific slice that has enough data.
# ---------------------------------------------------------------------------

MIN_ROWS = 30


def delay_slice(df: pd.DataFrame, origin: str, airline: str, dep_hour: int) -> tuple[pd.DataFrame, str]:
    candidates = [
        (df[(df.Origin == origin) & (df.Reporting_Airline == airline) & (df.dep_hour.between(dep_hour - 1, dep_hour + 1))],
         f"{origin} {airline} ~{dep_hour}:00"),
        (df[(df.Origin == origin) & (df.Reporting_Airline == airline)], f"{origin} {airline} all day"),
        (df[df.Origin == origin], f"all {origin}->SAN"),
        (df, "all flights to SAN"),
    ]
    for sub, label in candidates:
        if len(sub) >= MIN_ROWS:
            return sub, label
    return df, "all flights to SAN"


def p_on_time(df: pd.DataFrame, origin: str, airline: str, dep_hour: int,
              slack_min: float, already_delayed_min: float = 0) -> dict:
    """Probability the flight arrives with ArrDelay <= slack_min and isn't cancelled.
    If already_delayed_min > 0, condition on delay >= that (known delay)."""
    sub, label = delay_slice(df, origin, airline, dep_hour)
    flown = sub[sub.Cancelled == 0]
    cancel_rate = sub.Cancelled.mean() if len(sub) else 0.02
    delays = flown.ArrDelay.to_numpy()
    if already_delayed_min > 0:
        # Known delay: assume remaining uncertainty is the upper tail beyond the announced delay
        tail = delays[delays >= already_delayed_min]
        delays = tail if len(tail) >= 10 else np.maximum(delays, already_delayed_min)
    p = float((delays <= slack_min).mean()) if len(delays) else 0.0
    p *= (1 - cancel_rate)
    return {"p": round(p, 3), "n": int(len(sub)), "basis": label,
            "p_late_60": round(float((flown.ArrDelay > 60).mean()) if len(flown) else 0, 3),
            "cancel_rate": round(float(cancel_rate), 3)}


# ---------------------------------------------------------------------------
# Options  (Math-CS)
# ---------------------------------------------------------------------------

@dataclass
class Option:
    mode: str            # Commercial / Charter / Ground
    label: str
    depart: str
    arrive_hospital: str
    minutes_to_spare: int
    p_success: float
    cost: str
    basis: str


def deadline(organ: str, crossclamp: datetime) -> datetime:
    return crossclamp + timedelta(hours=VIABILITY_HOURS[organ] - RECIPIENT_BUFFER_H)


def _fmt(t: datetime) -> str:
    return t.strftime("%a %I:%M %p")


def commercial_options(df, origin, hospital, ready_at_airport: datetime, dl: datetime,
                       n: int = 4, known_delay_min: float = 0) -> list[Option]:
    """Next scheduled flights origin->SAN departing after ready_at_airport (SD time)."""
    offset_h = AIRPORTS[origin][2]
    sched = (df[df.Origin == origin]
             .groupby(["Reporting_Airline", "CRSDepTime"], as_index=False)["CRSElapsedTime"].median())
    to_hosp = HOSPITALS[hospital][0]
    out = []
    for day in (0, 1):
        base = ready_at_airport.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=day)
        for _, r in sched.iterrows():
            hh, mm = divmod(int(r.CRSDepTime), 100)
            dep_sd = base + timedelta(hours=hh - offset_h, minutes=mm)
            if dep_sd < ready_at_airport + timedelta(minutes=COMMERCIAL_CHECKIN_MIN):
                continue
            arr_sd = dep_sd + timedelta(minutes=float(r.CRSElapsedTime))
            at_hosp = arr_sd + timedelta(minutes=to_hosp)
            slack = (dl - at_hosp).total_seconds() / 60  # minutes of delay we can absorb
            m = p_on_time(df, origin, r.Reporting_Airline, hh, slack, known_delay_min)
            out.append((dep_sd, Option(
                "Commercial",
                f"{CARRIER_NAMES.get(r.Reporting_Airline, r.Reporting_Airline)} {origin}->SAN dep {hh:02d}:{mm:02d} local",
                _fmt(dep_sd), _fmt(at_hosp), int(slack), m["p"], "$", f"{m['basis']} (n={m['n']})")))
    out.sort(key=lambda t: t[0])  # chronological: next departures first
    return [o for _, o in out[:n]]


def charter_option(origin, hospital, ready_at_origin_hosp: datetime, dl: datetime) -> Option:
    miles = haversine_miles(AIRPORTS[origin][:2], MYF)
    dep = ready_at_origin_hosp + timedelta(hours=CHARTER_CALLOUT_H)
    arr = dep + timedelta(hours=miles / CHARTER_MPH + 0.3)
    at_hosp = arr + timedelta(minutes=HOSPITALS[hospital][1])
    slack = (dl - at_hosp).total_seconds() / 60
    p = CHARTER_RELIABILITY if slack >= 0 else 0.0
    return Option("Charter", f"Charter jet {origin} -> Montgomery-Gibbs (MYF)", _fmt(dep), _fmt(at_hosp),
                  int(slack), round(p, 3), "$$$$", f"{miles:.0f} mi @ {CHARTER_MPH} mph + {CHARTER_CALLOUT_H}h callout (assumption)")


def ground_option(origin, hospital, ready_at_origin_hosp: datetime, dl: datetime) -> Option | None:
    miles = haversine_miles(AIRPORTS[origin][:2], SAN) * GROUND_ROAD_FACTOR
    if miles > GROUND_MAX_MILES:
        return None
    at_hosp = ready_at_origin_hosp + timedelta(hours=miles / GROUND_MPH)
    slack = (dl - at_hosp).total_seconds() / 60
    p = GROUND_RELIABILITY if slack >= 0 else 0.0
    return Option("Ground", f"Medical courier by road from {origin} area", _fmt(ready_at_origin_hosp),
                  _fmt(at_hosp), int(slack), p, "$$", f"~{miles:.0f} road mi @ {GROUND_MPH} mph (assumption)")


MODES = ["Compare all", "Commercial", "Charter", "Ground"]


def build_options(df, organ, origin, hospital, crossclamp: datetime,
                  mode: str = "Compare all") -> tuple[datetime, list[Option]]:
    """mode limits the options to one transport type (e.g. when the team always charters)."""
    dl = deadline(organ, crossclamp)
    ready_hosp = crossclamp  # organ leaves donor hospital at cross-clamp (simplification)
    ready_airport = ready_hosp + timedelta(minutes=ORIGIN_HOSPITAL_TO_AIRPORT_MIN)
    opts = commercial_options(df, origin, hospital, ready_airport, dl)
    opts.append(charter_option(origin, hospital, ready_hosp, dl))
    g = ground_option(origin, hospital, ready_hosp, dl)
    if g:
        opts.append(g)
    if mode != "Compare all":
        opts = [o for o in opts if o.mode == mode]
    return dl, rank(opts)


def rank(opts: list[Option]) -> list[Option]:
    cost_rank = {"$": 0, "$$": 1, "$$$$": 2}
    # Prefer anything >= 90% likely, then cheapest, then most likely
    return sorted(opts, key=lambda o: (o.p_success < 0.9, cost_rank.get(o.cost, 3), -o.p_success))


def replan(df, organ, origin, hospital, crossclamp: datetime, chosen: Option,
           delay_min: int, now: datetime, departed: bool) -> dict:
    """Recompute after a delay alert on the booked option (commercial, charter, or ground).
    Backups always include every transport type: in a delay, any mode is on the table."""
    dl = deadline(organ, crossclamp)
    new_arrival = datetime.strptime(chosen.arrive_hospital, "%a %I:%M %p")
    # re-anchor the parsed time to the right date
    new_arrival = now.replace(hour=new_arrival.hour, minute=new_arrival.minute, second=0, microsecond=0)
    if new_arrival < now - timedelta(hours=1):
        new_arrival += timedelta(days=1)
    new_arrival += timedelta(minutes=delay_min)
    spare = int((dl - new_arrival).total_seconds() / 60)
    backups: list[Option] = []
    if not departed:
        backups += [o for o in commercial_options(df, origin, hospital, now, dl, n=4) if o.label != chosen.label][:3]
        if chosen.mode == "Charter":
            # booked charter is stuck: a second operator needs a full callout from now
            alt = charter_option(origin, hospital, now, dl)
            alt.label = f"Backup charter (different operator) {origin} -> MYF"
        else:
            alt = charter_option(origin, hospital, now - timedelta(minutes=ORIGIN_HOSPITAL_TO_AIRPORT_MIN), dl)
        backups.append(alt)
        if chosen.mode != "Ground":
            g = ground_option(origin, hospital, now, dl)
            if g:
                backups.append(g)
    return {
        "deadline": _fmt(dl),
        "current_flight_new_arrival": _fmt(new_arrival),
        "current_flight_minutes_to_spare": spare,
        "departed": departed,
        "backups": [asdict(b) for b in rank(backups)],
    }


def status_color(minutes_to_spare: int, p: float) -> str:
    if minutes_to_spare < 0 or p < 0.6:
        return "red"
    if minutes_to_spare < 60 or p < 0.9:
        return "yellow"
    return "green"


# ---------------------------------------------------------------------------
# LLM planner, provider-pluggable  (CS/CompE)
# ---------------------------------------------------------------------------

SYSTEM = """You are a transport decision-support assistant for an organ procurement coordinator in San Diego.
You are NOT a medical device; a human makes the final call. Use ONLY the numbers provided.
Return JSON with keys:
  status: "green" | "yellow" | "red"
  recommendation: one sentence naming the option to use
  reasoning: 2-3 short sentences citing minutes to spare and probabilities
  notify: list of roles to notify (e.g. "Recipient transplant surgeon", "OR charge nurse", "Life Sharing coordinator", "Charter dispatch")
  message_draft: a short, professional text message to the transplant team
"""


PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["green", "yellow", "red"]},
        "recommendation": {"type": "string"},
        "reasoning": {"type": "string"},
        "notify": {"type": "array", "items": {"type": "string"}},
        "message_draft": {"type": "string"},
    },
    "required": ["status", "recommendation", "reasoning", "notify", "message_draft"],
    "additionalProperties": False,
}

# Switch providers with one env var:  AI_PROVIDER=claude | openai | gemini
# Override the model with AI_MODEL (otherwise the default below is used).
PROVIDERS = {
    #  name      default model        API key env var(s)
    "claude": ("claude-opus-5-5",  ["ANTHROPIC_API_KEY"]),
    "openai": ("gpt-5-mini",       ["OPENAI_API_KEY"]),
    "gemini": ("gemini-2.5-flash", ["GEMINI_API_KEY", "GOOGLE_API_KEY"]),
}
PROVIDER_LABELS = {"claude": "Claude", "openai": "OpenAI", "gemini": "Gemini"}


def _situation(context: dict) -> str:
    return "SITUATION (JSON):\n" + json.dumps(context, default=str, indent=1)


def _call_claude(context: dict, model: str) -> dict:
    import anthropic
    client = anthropic.Anthropic()
    resp = client.beta.messages.create(
        model=model,
        max_tokens=4000,
        system=SYSTEM,
        messages=[{"role": "user", "content": _situation(context)}],
        output_config={"effort": "low",  # fast enough for a live demo
                       "format": {"type": "json_schema", "schema": PLAN_SCHEMA}},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",  # if a safety filter declines, the API retries on another model
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("Claude declined the request")
    return json.loads(next(b.text for b in resp.content if b.type == "text"))


def _call_openai(context: dict, model: str) -> dict:
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": _situation(context)}],
        response_format={"type": "json_schema",
                         "json_schema": {"name": "plan", "strict": True, "schema": PLAN_SCHEMA}},
    )
    return json.loads(resp.choices[0].message.content)


def _call_gemini(context: dict, model: str) -> dict:
    from google import genai
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    client = genai.Client(api_key=key)
    resp = client.models.generate_content(
        model=model,
        contents=SYSTEM + "\n\n" + _situation(context),
        config={"response_mime_type": "application/json", "temperature": 0.2},
    )
    return json.loads(resp.text)


_CALLERS = {"claude": _call_claude, "openai": _call_openai, "gemini": _call_gemini}


def llm_plan(context: dict) -> tuple[dict, str]:
    """Returns (plan_json, source). source is the provider name or "fallback".
    Falls back to rules if no key/SDK/error so the demo never breaks."""
    provider = os.environ.get("AI_PROVIDER", "claude").lower()
    if provider not in PROVIDERS:
        fb = rule_plan(context)
        fb["reasoning"] += f" (Unknown AI_PROVIDER '{provider}', rule fallback.)"
        return fb, "fallback"
    default_model, key_vars = PROVIDERS[provider]
    if not any(os.environ.get(k) for k in key_vars):
        fb = rule_plan(context)
        fb["reasoning"] += f" (No {' / '.join(key_vars)} found; check .env and that python-dotenv is installed. Rule fallback.)"
        return fb, "fallback"
    try:
        model = os.environ.get("AI_MODEL", default_model)
        return _CALLERS[provider](context, model), provider
    except Exception as e:  # show error in UI but keep going
        fb = rule_plan(context)
        fb["reasoning"] += f" ({PROVIDER_LABELS[provider]} error, rule fallback: {e})"
        return fb, "fallback"


gemini_plan = llm_plan  # backwards-compatible alias


def rule_plan(context: dict) -> dict:
    opts = context.get("options") or context.get("replan", {}).get("backups") or []
    best = opts[0] if opts else None
    if "replan" in context:
        rp = context["replan"]
        spare = rp["current_flight_minutes_to_spare"]
        what = {"Charter": "charter", "Ground": "courier"}.get(context.get("booked_transport", {}).get("mode"), "flight")
        if spare >= 60:
            return {"status": "green", "recommendation": f"Stay on the current {what}.",
                    "reasoning": f"Still {spare} min to spare after the delay.",
                    "notify": ["Recipient transplant surgeon"],
                    "message_draft": f"Organ {what} delayed; new ETA {rp['current_flight_new_arrival']}. Still within window."}
        rec = f"Switch to: {best['label']}" if best and best["minutes_to_spare"] > 0 else f"Keep {what}; prepare backup recipient / adjust OR start."
        notify = ["Recipient transplant surgeon", "OR charge nurse", "Life Sharing coordinator"]
        if what == "charter" or (best and best["mode"] == "Charter"):
            notify.append("Charter dispatch")
        return {"status": "red" if spare < 0 else "yellow", "recommendation": rec,
                "reasoning": f"Current {what} leaves {spare} min to spare.",
                "notify": notify,
                "message_draft": f"URGENT: organ {what} delayed, {spare} min margin. Plan: {rec}"}
    if not best:
        return {"status": "red", "recommendation": "No feasible option found.", "reasoning": "",
                "notify": [], "message_draft": ""}
    return {"status": status_color(best["minutes_to_spare"], best["p_success"]),
            "recommendation": f"Use {best['label']}.",
            "reasoning": f"{best['p_success']:.0%} chance on time with {best['minutes_to_spare']} min to spare.",
            "notify": ["Recipient transplant surgeon", "Life Sharing coordinator"],
            "message_draft": f"Organ transport booked: {best['label']}, ETA {best['arrive_hospital']}."}
