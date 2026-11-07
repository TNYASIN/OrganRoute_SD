"""OrganRoute SD - Streamlit UI.  Run:  streamlit run app.py"""
from dataclasses import asdict
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

import core
import notify

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


_cards = []  # reset every rerun; gives each plan card's widgets a unique key


def plan_card(plan: dict, source: str):
    key = f"card{len(_cards)}"
    _cards.append(key)
    c = COLORS.get(plan.get("status", "yellow"), "#888")
    st.markdown(
        f"""<div style="border-left:10px solid {c};padding:14px 18px;border-radius:8px;background:rgba(127,127,127,.08)">
        <div style="font-size:1.3rem;font-weight:700;color:{c}">{plan.get('status','').upper()}</div>
        <div style="font-size:1.1rem;font-weight:600;margin:6px 0">{plan.get('recommendation','')}</div>
        <div>{plan.get('reasoning','')}</div></div>""", unsafe_allow_html=True)
    st.write("**Notify:** " + ", ".join(plan.get("notify", [])))
    msg = st.text_area("Draft message to team", plan.get("message_draft", ""), height=90, key=f"{key}_msg")
    st.caption(f"Plan source: {core.PROVIDER_LABELS.get(source, 'rule-based fallback')}")
    if notify.enabled():
        n = len(notify.recipients())
        if st.button(f"📲 Send to team ({notify.channel()}, {n} recipient{'s' * (n != 1)})", key=f"{key}_send"):
            title = f"OrganRoute {plan.get('status', '').upper()}: {plan.get('recommendation', '')}"[:120]
            with st.spinner("Sending..."):
                ok, note = notify.send_alert(title, msg, plan.get("status", "yellow"))
            (st.success if ok else st.error)(note)
    else:
        st.caption("📲 Team alerts are off. Add Twilio settings to .env to send this message (see .env.example).")


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

MODE_NOTES = {
    "Commercial": "Showing scheduled flights into SAN only. On-time chances come from real BTS delay history.",
    "Charter": "Showing charter only. **Charter delays aren't in any public dataset**, so its on-time chance is an "
               f"assumption ({core.CHARTER_RELIABILITY:.0%} reliability, {core.CHARTER_CALLOUT_H} h callout), not a "
               "prediction. Main risks are weather, crew availability, and mechanical issues; use tab 2 to plan for a delay.",
    "Ground": "Showing ground courier only (within ~400 road miles). Travel time is a distance-based estimate.",
}

tab1, tab2 = st.tabs(["1 · Plan transport", "2 · Delay alert → re-plan"])

# ---------------- Tab 1: Planning ----------------
with tab1:
    dl, opts = core.build_options(df, organ, origin, hospital, crossclamp, mode)
    st.session_state["opts"] = opts
    c1, c2, c3 = st.columns(3)
    c1.metric("Cross-clamp", crossclamp.strftime("%I:%M %p"))
    c2.metric("Must arrive by", dl.strftime("%a %I:%M %p"))
    c3.metric("Window", f"{core.VIABILITY_HOURS[organ]} h")
    st.subheader("Options")
    if mode in MODE_NOTES:
        st.info(MODE_NOTES[mode])
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
            alert_time = datetime.strptime(chosen.depart, "%a %I:%M %p")
            alert_time = crossclamp.replace(hour=alert_time.hour, minute=alert_time.minute)
            if alert_time < crossclamp:
                alert_time += timedelta(days=1)
            alert_time -= timedelta(minutes=30)  # alert arrives ~30 min before scheduled departure
            rp = core.replan(df, organ, origin, hospital, crossclamp, chosen, delay, alert_time, departed)
            ctx = {"phase": "replan", "organ": organ, "origin": origin, "hospital": hospital,
                   "booked_transport": asdict(chosen), "delay_minutes": delay, "delay_cause": cause,
                   "replan": rp}
            with st.spinner("Re-planning..."):
                st.session_state["rp"] = (rp, core.llm_plan(ctx))
        if "rp" in st.session_state:
            rp, plan = st.session_state["rp"]
            c1, c2, c3 = st.columns(3)
            c1.metric("New hospital ETA", rp["current_flight_new_arrival"])
            c2.metric("Deadline", rp["deadline"])
            c3.metric("Minutes to spare", rp["current_flight_minutes_to_spare"])
            plan_card(*plan)
            if rp["backups"]:
                st.subheader("Backup options")
                options_table(rp["backups"])
