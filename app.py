"""OrganRoute SD - Streamlit UI.  Run:  streamlit run app.py"""
from dataclasses import asdict
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

import core

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
    st.caption(f"Plan source: {'Gemini' if source == 'gemini' else 'rule-based fallback'}")


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

tab1, tab2 = st.tabs(["1 · Plan transport", "2 · Delay alert → re-plan"])

# ---------------- Tab 1: Planning ----------------
with tab1:
    dl, opts = core.build_options(df, organ, origin, hospital, crossclamp)
    st.session_state["opts"] = opts
    c1, c2, c3 = st.columns(3)
    c1.metric("Cross-clamp", crossclamp.strftime("%I:%M %p"))
    c2.metric("Must arrive by", dl.strftime("%a %I:%M %p"))
    c3.metric("Window", f"{core.VIABILITY_HOURS[organ]} h")
    st.subheader("Options")
    options_table(opts)
    if st.button("Get recommendation", type="primary"):
        ctx = {"phase": "planning", "organ": organ, "origin": origin, "hospital": hospital,
               "crossclamp": crossclamp, "deadline": dl, "options": [asdict(o) for o in opts]}
        with st.spinner("Thinking..."):
            st.session_state["plan1"] = core.gemini_plan(ctx)
    if "plan1" in st.session_state:
        plan_card(*st.session_state["plan1"])

# ---------------- Tab 2: Re-planning ----------------
with tab2:
    commercial = [o for o in st.session_state.get("opts", []) if o.mode == "Commercial"]
    if not commercial:
        st.info("No commercial flight to simulate. Pick a different origin/organ.")
    else:
        chosen = st.selectbox("Booked flight", commercial, format_func=lambda o: f"{o.label} → hospital {o.arrive_hospital}")
        c1, c2 = st.columns(2)
        delay = c1.slider("Delay alert (minutes)", 15, 240, 75, step=15)
        departed = c2.radio("Flight status", ["At gate (not departed)", "In the air"]) == "In the air"
        if st.button("🚨 Simulate delay alert", type="primary"):
            alert_time = datetime.strptime(chosen.depart, "%a %I:%M %p")
            alert_time = crossclamp.replace(hour=alert_time.hour, minute=alert_time.minute)
            if alert_time < crossclamp:
                alert_time += timedelta(days=1)
            alert_time -= timedelta(minutes=30)  # alert arrives ~30 min before scheduled departure
            rp = core.replan(df, organ, origin, hospital, crossclamp, chosen, delay, alert_time, departed)
            ctx = {"phase": "replan", "organ": organ, "origin": origin, "hospital": hospital,
                   "booked_flight": asdict(chosen), "delay_minutes": delay, "replan": rp}
            with st.spinner("Re-planning..."):
                st.session_state["rp"] = (rp, core.gemini_plan(ctx))
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
