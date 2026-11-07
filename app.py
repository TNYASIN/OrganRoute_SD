"""OrganRoute SD - Streamlit UI.  Run:  streamlit run app.py"""
from dataclasses import asdict
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

import core
import live

try:  # team texts come from the notifications branch; the app works without it
    import notify
except ImportError:
    notify = None

st.set_page_config(page_title="OrganRoute SD", page_icon="🫀", layout="wide")
COLORS = {"green": "#1e8e3e", "yellow": "#f9ab00", "red": "#d93025"}


@st.cache_data
def get_data():
    return core.load_flights("flights_san.csv")


df, is_real = get_data()

st.title("🫀 OrganRoute SD")
st.caption("Transport decision support for San Diego organ transport. "
           "Tracking tells you it's late — this tells you what to do. Not a medical device.")
if not is_real:
    st.warning("Using SYNTHETIC flight data. Put real BTS data in flights_san.csv (run data_prep.py).")
else:
    st.caption(f"Delay model built on {len(df):,} real BTS flights into SAN.")


def plan_card(plan: dict, source: str):
    c = COLORS.get(plan.get("status", "yellow"), "#888")
    st.markdown(
        f"""<div style="border-left:10px solid {c};padding:14px 18px;border-radius:8px;background:rgba(127,127,127,.08)">
        <div style="font-size:1.3rem;font-weight:700;color:{c}">{plan.get('status','').upper()}</div>
        <div style="font-size:1.1rem;font-weight:600;margin:6px 0">{plan.get('recommendation','')}</div>
        <div>{plan.get('reasoning','')}</div></div>""", unsafe_allow_html=True)
    st.write("**Notify:** " + ", ".join(plan.get("notify", [])))
    st.text_area("Draft message to team", plan.get("message_draft", ""), height=90)
    st.caption(f"Plan source: {core.PROVIDER_LABELS.get(source, 'rule-based fallback')}")


def options_table(opts):
    t = pd.DataFrame([asdict(o) if not isinstance(o, dict) else o for o in opts])
    if t.empty:
        st.info("No options.")
        return
    t["status"] = [core.status_color(m, p) for m, p in zip(t.minutes_to_spare, t.p_success)]
    t["p_success"] = (t.p_success * 100).round(0).astype(int).astype(str) + "%"
    st.dataframe(t[["status", "mode", "label", "depart", "arrive_hospital", "minutes_to_spare",
                    "p_success", "cost", "basis"]], use_container_width=True, hide_index=True)


# ---------------- Sidebar inputs ----------------
with st.sidebar:
    st.header("Organ offer")
    organ = st.selectbox("Organ", list(core.VIABILITY_HOURS), index=0)
    origin = st.selectbox("Donor location (nearest airport)", list(core.AIRPORTS), index=list(core.AIRPORTS).index("PHX"))
    hospital = st.selectbox("Receiving hospital", list(core.HOSPITALS))
    now = datetime.now().replace(second=0, microsecond=0)
    d = st.date_input("Cross-clamp date", now.date())
    t = st.time_input("Cross-clamp time (San Diego time)", (now + timedelta(hours=1)).time())
    crossclamp = datetime.combine(d, t)
    st.caption(f"Viability window: ~{core.VIABILITY_HOURS[organ]} h")
    mode = st.radio("Transport type", core.MODES, index=0,
                    help="Pick one type if your team already knows how this organ will travel "
                         "(e.g. a recovery team that always flies charter). 'Compare all' ranks every option.")

CHARTER_CAVEAT = ("**Charter delays aren't in any public dataset**, so the charter's on-time chance starts from an "
                  f"assumed {core.CHARTER_RELIABILITY:.0%} and is adjusted by the live aviation forecast at departure "
                  "and arrival (stated rules, not a trained model). Enter the booking under *Charter details*; "
                  "use tab 2 for crew or mechanical delays.")
MODE_NOTES = {
    "Compare all": "Commercial on-time chances come from real BTS delay history; ground and charter are estimates. "
                   + CHARTER_CAVEAT,
    "Commercial": "Showing scheduled flights into SAN only. On-time chances come from real BTS delay history.",
    "Charter": "Showing charter only. " + CHARTER_CAVEAT,
    "Ground": "Showing ground courier only (within ~400 road miles). Travel time is a distance-based estimate.",
}

tab1, tab2 = st.tabs(["1 · Plan transport", "2 · Delay alert → re-plan"])

# ---------------- Tab 1: Planning ----------------
with tab1:
    dl = core.deadline(organ, crossclamp)
    c1, c2, c3 = st.columns(3)
    c1.metric("Cross-clamp", crossclamp.strftime("%I:%M %p"))
    c2.metric("Must arrive by", dl.strftime("%a %I:%M %p"))
    c3.metric("Window", f"{core.VIABILITY_HOURS[organ]} h")
    st.subheader("Options")
    if mode in MODE_NOTES:
        st.info(MODE_NOTES[mode])

    charter = None
    if mode in ("Compare all", "Charter"):
        with st.expander("✈️ Charter details", expanded=(mode == "Charter")):
            booked = st.toggle("I have a charter booking",
                               help="Without one, the app estimates a charter leaving "
                                    f"{core.CHARTER_CALLOUT_H} h after cross-clamp.")
            a1, a2 = st.columns(2)
            aircraft = a1.selectbox("Aircraft", list(core.CHARTER_AIRCRAFT_MPH),
                                    format_func=lambda a: f"{a} (~{core.CHARTER_AIRCRAFT_MPH[a]} mph)")
            arrive = a2.selectbox("Arrival airport", list(core.CHARTER_ARRIVAL),
                                  format_func=lambda a: core.CHARTER_ARRIVAL[a][0])
            wheels_up, operator = None, ""
            if booked:
                b1, b2, b3 = st.columns(3)
                operator = b1.text_input("Operator (optional)")
                default_wu = crossclamp + timedelta(hours=core.CHARTER_CALLOUT_H)
                wu_date = b2.date_input("Wheels-up date", default_wu.date())
                wu_time = b3.time_input("Wheels-up time (San Diego time)", default_wu.time())
                wheels_up = datetime.combine(wu_date, wu_time)
            sd_weather = st.selectbox(
                "San Diego weather", ["Live forecast"] + [k for k in core.WX_IMPACT if k != "VFR"],
                format_func=lambda k: k if k == "Live forecast" else f"Simulate: {core.WX_LABELS[k]}",
                help="Simulate bad arrival weather for a demo. San Diego's marine-layer fog is a real risk.")
            charter = core.CharterDetails(wheels_up, aircraft, arrive, operator.strip(), sd_weather)
            rules = ", ".join(f"{k} −{1 - f:.0%} / +{m} min" for k, (f, m) in core.WX_IMPACT.items() if k != "VFR")
            st.caption(f"Forecasts from aviationweather.gov (MYF uses SAN's, ~8 mi away). "
                       f"Weather rules per airport (assumptions): {rules}.")
    st.session_state["charter"] = charter

    dl, opts = core.build_options(df, organ, origin, hospital, crossclamp, mode, charter)
    st.session_state["opts"] = opts
    if not opts and mode == "Ground":
        st.warning(f"{origin} is too far for ground transport. Try Charter or Compare all.")
    options_table(opts)
    if st.button("Get recommendation", type="primary"):
        ctx = {"phase": "planning", "organ": organ, "origin": origin, "hospital": hospital,
               "crossclamp": crossclamp, "deadline": dl, "transport_type_requested": mode,
               "options": [asdict(o) for o in opts]}
        with st.spinner("Thinking..."):
            st.session_state["plan1"] = core.llm_plan(ctx)
    if "plan1" in st.session_state:
        plan_card(*st.session_state["plan1"])

# ---------------- Tab 2: Re-planning ----------------
LIVE_TRIGGER_MIN = 15   # live delays below this don't trigger a re-plan


def booked_departure(chosen) -> datetime:
    """The booked option's departure as a datetime on/after the cross-clamp date."""
    t = datetime.strptime(chosen.depart, "%a %I:%M %p")
    dep = crossclamp.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
    return dep + timedelta(days=1) if dep < crossclamp - timedelta(hours=1) else dep


def run_replan(chosen, delay, cause, departed, alert_time, source):
    rp = core.replan(df, organ, origin, hospital, crossclamp, chosen, delay, alert_time, departed,
                     st.session_state.get("charter"))
    ctx = {"phase": "replan", "organ": organ, "origin": origin, "hospital": hospital,
           "booked_transport": asdict(chosen), "delay_minutes": delay, "delay_cause": cause,
           "alert_source": source, "replan": rp}
    with st.spinner("Re-planning..."):
        st.session_state["rp"] = (rp, core.llm_plan(ctx))
    st.session_state["rp_source"] = f"{source} at {datetime.now().strftime('%I:%M %p')}"
    return st.session_state["rp"]


@st.fragment(run_every=60)
def live_panel(chosen, ident):
    """Re-runs every 60 s while watching. Triggers a re-plan (and team text) when a new delay appears."""
    arrive = (st.session_state.get("charter") or core.CharterDetails()).arrive if chosen.mode == "Charter" else "SAN"
    imp = live.faa_impact(origin, chosen.mode, arrive)
    fs = live.flight_status(ident, booked_departure(chosen)) if ident and live.aeroapi_enabled() else None

    st.write(f"**FAA** · {origin} → {arrive}: " +
             ("; ".join(e["summary"] for e in imp["events"]) if imp["events"] else "✅ no delay programs affecting this trip"))
    st.caption(imp["source"])
    for n in imp["notes"]:
        st.info(n)
    delay, cause, departed, source = imp["delay_min"], "", False, "live FAA airport status"
    if imp["events"]:
        cause = imp["events"][0]["summary"]
    if ident and not live.aeroapi_enabled():
        st.caption(f"{ident}: per-flight status off (add AEROAPI_KEY to .env).")
    elif fs is not None:
        if fs["ok"]:
            icon = "❌" if fs["cancelled"] else "⏱" if fs["delay_min"] >= LIVE_TRIGGER_MIN else "✅"
            eta = fs["estimated_in"].strftime("%I:%M %p") if fs["estimated_in"] else "?"
            st.write(f"**{fs['ident']}** {icon} {fs['status'] or 'scheduled'} · "
                     f"{'+' + str(fs['delay_min']) + ' min · ' if fs['delay_min'] else ''}ETA SAN {eta}")
            departed = fs["departed"]
            if fs["cancelled"]:
                delay, cause, source = 240, f"{fs['ident']} cancelled", "FlightAware (cancellation)"
            elif fs["delay_min"] >= delay:
                delay, cause, source = fs["delay_min"], f"{fs['ident']} airline-reported delay", "FlightAware flight status"
        st.caption(fs["note"])
    st.caption(f"Checked {datetime.now().strftime('%I:%M:%S %p')} · refreshes every 60 s")

    key = f"{chosen.label}|{delay // 15}|{cause[:40]}"
    if delay >= LIVE_TRIGGER_MIN and key != st.session_state.get("last_live_key"):
        st.session_state["last_live_key"] = key
        _, plan = run_replan(chosen, delay, cause, departed, datetime.now().replace(second=0, microsecond=0), source)
        if notify is not None and notify.enabled() and st.session_state.get("auto_notify", True):
            p = plan[0]
            ok, note = notify.send_alert(f"OrganRoute {p.get('status', '').upper()}: {p.get('recommendation', '')}"[:120],
                                         p.get("message_draft", ""), p.get("status", "yellow"))
            st.session_state["rp_source"] += f" · team {'texted' if ok else 'NOT texted'} ({note[:80]})"
        st.rerun()  # full rerun so the re-plan shows below


with tab2:
    bookable = st.session_state.get("opts", [])
    if not bookable:
        st.info("No transport options to simulate. Pick a different origin or transport type in the sidebar.")
    else:
        st.caption("Pick what was booked in tab 1, then simulate an alert. Backups include every transport type, "
                   "even if tab 1 was limited to one.")
        chosen = st.selectbox("Booked transport", bookable,
                              format_func=lambda o: f"[{o.mode}] {o.label} → hospital {o.arrive_hospital}")
        causes = {"Commercial": ["Airline delay", "Weather", "Air traffic control", "Cancellation risk"],
                  "Charter": ["Weather", "Crew availability", "Mechanical", "Air traffic control"],
                  "Ground": ["Traffic", "Road closure", "Vehicle issue"]}[chosen.mode]
        en_route = {"Commercial": ("At gate (not departed)", "In the air"),
                    "Charter": ("Not departed yet", "In the air"),
                    "Ground": ("Not departed yet", "On the road")}[chosen.mode]
        c1, c2, c3 = st.columns(3)
        delay = c1.slider("Delay alert (minutes)", 15, 240, 75, step=15)
        cause = c2.selectbox("Cause", causes)
        departed = c3.radio("Status", en_route) == en_route[1]
        if chosen.mode == "Charter":
            st.caption("There's no public charter delay history, so you set the delay; the app recomputes the margin and backups.")
        if st.button("🚨 Simulate delay alert", type="primary"):
            alert_time = booked_departure(chosen) - timedelta(minutes=30)  # alert ~30 min before departure
            run_replan(chosen, delay, cause, departed, alert_time, "simulated")

        # ---- Live watch: real FAA airport delays (+ FlightAware per-flight status if configured) ----
        st.markdown("#### 📡 Live watch")
        if chosen.mode == "Ground":
            st.caption("Live feeds cover flights only; use the simulator above for road delays.")
        else:
            w1, w2 = st.columns([3, 1])
            ident = ""
            if chosen.mode == "Commercial":
                ident = w1.text_input(
                    "Booked flight number (optional, e.g. WN1234)", key="ident",
                    help="With AEROAPI_KEY in .env, the app checks this exact flight on FlightAware "
                         "(capped monthly). Without it, it watches FAA airport-level delays only."
                ).replace(" ", "").upper()
            else:
                w1.caption("Charters usually aren't trackable by flight number; watching FAA delays at the origin.")
            watching = st.session_state.get("watching", False)
            if w2.button("⏹ Stop watching" if watching else "▶️ Start watching", key="watch_btn"):
                st.session_state["watching"] = not watching
                st.session_state.pop("last_live_key", None)
                st.rerun()
            if notify is not None:
                if notify.enabled():
                    st.toggle("Text the team automatically on live alerts", value=True, key="auto_notify")
                else:
                    st.caption("Add Twilio settings to .env to text the team automatically on live alerts.")
            if st.session_state.get("watching"):
                live_panel(chosen, ident)

        if "rp" in st.session_state:
            rp, plan = st.session_state["rp"]
            if st.session_state.get("rp_source"):
                st.caption(f"Re-plan triggered by: {st.session_state['rp_source']}")
            c1, c2, c3 = st.columns(3)
            c1.metric("New hospital ETA", rp["current_flight_new_arrival"])
            c2.metric("Deadline", rp["deadline"])
            c3.metric("Minutes to spare", rp["current_flight_minutes_to_spare"])
            plan_card(*plan)
            if rp["backups"]:
                st.subheader("Backup options")
                options_table(rp["backups"])
