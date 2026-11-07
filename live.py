"""
OrganRoute SD - live delay feeds (no UI here).

  - faa_events / faa_impact ... FAA airport status (free, no key): ground stops, delay programs, closures.
                                Airport-level: says "flights into SAN are held ~45 min", not "WN1234 is late".
  - flight_status ............. FlightAware AeroAPI per-flight status (optional, needs AEROAPI_KEY in .env).

All times returned are San Diego local (naive datetimes), matching core.py.
"""
from __future__ import annotations

import json
import os
import re
import ssl
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from zoneinfo import ZoneInfo

SD_TZ = ZoneInfo("America/Los_Angeles")
try:  # python.org macOS builds often lack system CA certs
    import certifi
    _SSL = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL = ssl.create_default_context()


def _http(url: str, headers: dict | None = None, timeout: float = 8) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "OrganRoute-SD", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout, context=_SSL) as r:
        return r.read()


# ---------------------------------------------------------------------------
# FAA airport status (nasstatus.faa.gov)
# ---------------------------------------------------------------------------

FAA_URL = "https://nasstatus.faa.gov/api/airport-status-information"
GROUND_STOP_ASSUMED_MIN = 60   # FAA gives an end time, not a delay, for ground stops (assumption)
_faa_cache: tuple[float, list[dict], str] | None = None


def _minutes(text: str | None) -> int | None:
    """'2 hours and 28 minutes' -> 148, '45 minutes' -> 45, '1 hour' -> 60."""
    if not text:
        return None
    h = re.search(r"(\d+)\s*hour", text)
    m = re.search(r"(\d+)\s*min", text)
    if not h and not m:
        return None
    return (int(h.group(1)) * 60 if h else 0) + (int(m.group(1)) if m else 0)


def parse_faa(xml_bytes: bytes) -> tuple[list[dict], str]:
    """FAA XML -> list of events: airport, kind, direction, reason, avg_min, max_min, until."""
    root = ET.fromstring(xml_bytes)
    events = []
    for dt in root.findall("Delay_type"):
        name = (dt.findtext("Name") or "").lower()
        kind = ("ground_stop" if "ground stop" in name else "ground_delay" if "ground delay" in name
                else "closure" if "closure" in name else "delay" if "arrival/departure" in name else None)
        if not kind:
            continue  # e.g. airspace flow programs: not airport-specific
        for el in dt.iter():
            arpt = el.findtext("ARPT")
            if not arpt:
                continue
            ad = el.find(".//Arrival_Departure")
            direction = (ad.get("Type") or "").lower() or None if ad is not None else None
            if kind in ("ground_stop", "ground_delay"):
                direction = "arrival"  # programs hold flights headed TO this airport
            avg, mx, mn = (_minutes(el.findtext(f".//{t}")) for t in ("Avg", "Max", "Min"))
            if avg is None:
                avg = (mn + mx) // 2 if mn is not None and mx is not None else mx if mx is not None else mn
            if kind == "ground_stop" and avg is None:
                avg = GROUND_STOP_ASSUMED_MIN
            events.append({"airport": arpt, "kind": kind, "direction": direction,
                           "reason": (el.findtext("Reason") or "").strip(), "avg_min": avg, "max_min": mx,
                           "until": el.findtext("End_Time") or el.findtext("Reopen")})
    return events, f"FAA status as of {root.findtext('Update_Time') or 'unknown'}"


def faa_events() -> tuple[list[dict], str]:
    """Current FAA events, cached 60 s (the FAA refreshes about every minute)."""
    global _faa_cache
    if _faa_cache and time.time() - _faa_cache[0] < 60:
        return _faa_cache[1], _faa_cache[2]
    try:
        events, note = parse_faa(_http(FAA_URL))
    except Exception as e:
        return [], f"FAA feed unavailable ({type(e).__name__})"
    _faa_cache = (time.time(), events, note)
    return events, note


KIND_LABEL = {"ground_stop": "Ground stop", "ground_delay": "Ground delay program", "delay": "Delays"}


def faa_impact(origin: str, mode: str, arrive_airport: str = "SAN") -> dict:
    """Which FAA events affect this booking, and the delay (minutes) to re-plan with.
    Commercial: arrival programs/delays at SAN + departure delays at the origin.
    Charter: departure delays and ground stops at the origin (a charter to MYF isn't in SAN's arrival programs).
    Closures are notes only: most are standing restrictions, not delays."""
    events, source = faa_events()
    hits, notes = [], []
    for e in events:
        if e["kind"] == "closure":
            if e["airport"] in (origin, arrive_airport):
                txt = e["reason"].lstrip("!")
                if "NON SKED TRANSIENT GA" in txt:
                    if mode == "Charter":
                        notes.append(f"{e['airport']} is closed to unscheduled private flights without prior "
                                     "permission; charters should land at Montgomery-Gibbs (MYF).")
                else:
                    notes.append(f"{e['airport']} closure notice: {txt[:120]}")
            continue
        origin_dep = e["airport"] == origin and e["direction"] in ("departure", None)
        dest_arr = e["airport"] == arrive_airport and e["direction"] in ("arrival", None)
        if mode == "Commercial" and (origin_dep or dest_arr):
            hits.append(e)
        elif mode == "Charter" and (dest_arr or (e["airport"] == origin and (origin_dep or e["kind"] == "ground_stop"))):
            hits.append(e)  # dest_arr only matches when the charter lands at SAN (MYF isn't in the feed)
    for e in hits:
        direction = f" ({e['direction']}s)" if e["kind"] == "delay" and e["direction"] else ""
        e["summary"] = (f"{KIND_LABEL[e['kind']]} at {e['airport']}{direction}: ~{e['avg_min'] or '?'} min"
                        + (f", up to {e['max_min']}" if e["max_min"] else "")
                        + (f" ({e['reason']})" if e["reason"] else ""))
    return {"events": hits, "delay_min": max((e["avg_min"] or 0 for e in hits), default=0),
            "worst_min": max((e["max_min"] or e["avg_min"] or 0 for e in hits), default=0),
            "notes": notes, "source": source}


# ---------------------------------------------------------------------------
# FlightAware AeroAPI - per-flight status (optional; billed beyond the $5/month free credit).
# Guards: called only while a watch is running, at most once per 2 min per flight,
# and a hard monthly cap counted in a local file.
# ---------------------------------------------------------------------------

AEROAPI_URL = "https://aeroapi.flightaware.com/aeroapi/flights/"
AEROAPI_MIN_INTERVAL_S = 120
USAGE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".aeroapi_usage.json")
_flight_cache: dict[str, tuple[float, dict]] = {}


def aeroapi_enabled() -> bool:
    return bool(os.environ.get("AEROAPI_KEY"))


def aeroapi_usage() -> tuple[int, int]:
    """(calls this month, monthly cap). Cap defaults to 300 calls (~$1.50 at ~$0.005/call)."""
    cap = int(os.environ.get("AEROAPI_MONTHLY_CAP", "300"))
    try:
        with open(USAGE_FILE) as f:
            u = json.load(f)
        calls = u.get("calls", 0) if u.get("month") == datetime.now(SD_TZ).strftime("%Y-%m") else 0
    except (OSError, ValueError):
        calls = 0
    return calls, cap


def _count_call() -> None:
    calls, _ = aeroapi_usage()
    try:
        with open(USAGE_FILE, "w") as f:
            json.dump({"month": datetime.now(SD_TZ).strftime("%Y-%m"), "calls": calls + 1}, f)
    except OSError:
        pass


def _sd(ts: str | None) -> datetime | None:
    if not ts:
        return None
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(SD_TZ).replace(tzinfo=None)


def pick_flight(flights: list[dict], near: datetime) -> dict:
    """The SAN-bound flight whose scheduled departure is closest to `near`."""
    def dest(f):
        return ((f.get("destination") or {}).get("code_iata") or "").upper()
    pool = [f for f in flights if dest(f) == "SAN"] or flights
    return min(pool, key=lambda f: abs(((_sd(f.get("scheduled_out")) or near) - near).total_seconds()))


def summarize_flight(f: dict, ident: str) -> dict:
    sched_in = _sd(f.get("scheduled_in"))
    est_in = _sd(f.get("actual_in") or f.get("estimated_in"))
    if sched_in and est_in:
        delay = int((est_in - sched_in).total_seconds() // 60)
    else:
        delay = int(max(f.get("arrival_delay") or 0, f.get("departure_delay") or 0) // 60)
    return {"ok": True, "ident": f.get("ident_iata") or ident, "status": f.get("status") or "",
            "cancelled": bool(f.get("cancelled")), "departed": bool(f.get("actual_out") or f.get("actual_off")),
            "delay_min": max(delay, 0), "scheduled_in": sched_in, "estimated_in": est_in,
            "destination": ((f.get("destination") or {}).get("code_iata") or "")}


def flight_status(ident: str, near: datetime) -> dict:
    """Status of flight `ident` (e.g. 'WN1234') nearest `near` (SD local). {"ok": False, "note": ...} on any problem."""
    ident = ident.replace(" ", "").upper()
    if not aeroapi_enabled():
        return {"ok": False, "note": "Per-flight status off (no AEROAPI_KEY in .env)."}
    hit = _flight_cache.get(ident)
    if hit and time.time() - hit[0] < AEROAPI_MIN_INTERVAL_S:
        return hit[1]
    calls, cap = aeroapi_usage()
    if calls >= cap:
        return {"ok": False, "note": f"FlightAware monthly cap reached ({calls}/{cap} calls); FAA only."}
    _count_call()  # count before the request so failures can't loop uncounted
    try:
        raw = _http(AEROAPI_URL + urllib.parse.quote(ident) + "?ident_type=designator",
                    headers={"x-apikey": os.environ["AEROAPI_KEY"], "Accept": "application/json"})
        flights = json.loads(raw).get("flights") or []
        res = summarize_flight(pick_flight(flights, near), ident) if flights else \
            {"ok": False, "note": f"FlightAware has no flights for {ident}."}
    except Exception as e:
        res = {"ok": False, "note": f"FlightAware error: {type(e).__name__} {str(e)[:80]}"}
    res.setdefault("note", f"FlightAware, checked {datetime.now(SD_TZ).strftime('%I:%M %p')} "
                           f"({aeroapi_usage()[0]}/{cap} calls this month)")
    _flight_cache[ident] = (time.time(), res)
    return res
