"""OrganRoute SD - Streamlit UI.  Run:  streamlit run app.py"""
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st

import core
from recovery import DisruptionEvent, EventType, TransportStage
from recovery_agent import EventEnvelope, RecoveryAgent
from recovery_demo import demo_mission

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

tab1, tab2, tab3 = st.tabs(["1 · Plan transport", "2 · Delay alert → re-plan",
                           "3 · Recovery agent demo"])

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
            st.session_state["plan1"] = core.llm_plan(ctx)
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


# ---------------- Tab 3: deterministic, stateful recovery agent simulation ----------------
with tab3:
    st.subheader("Recovery agent · simulation")
    st.warning("All flights, airport transfers, and ground durations in this tab are hypothetical. "
               "Nothing here is a live schedule, carrier acceptance, or booking.")
    agent = RecoveryAgent(Path(__file__).with_name("recovery_demo.sqlite3"))
    with st.form("new_recovery_mission"):
        demo_organ = st.selectbox("Organ for simulated mission", list(core.VIABILITY_HOURS),
                                  key="demo_organ")
        deadline_hours = st.number_input("User-specified deadline: hours from mission start",
                                         min_value=1, max_value=48, value=6)
        create_demo = st.form_submit_button("Create simulated PHX → San Diego mission")
    if create_demo:
        started = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        mission_id = f"demo-{uuid4().hex[:12]}"
        agent.create_mission(demo_mission(mission_id, started,
                                          started + timedelta(hours=int(deadline_hours)), demo_organ))
        st.session_state["recovery_mission_id"] = mission_id

    mission_id = st.session_state.get("recovery_mission_id")
    if mission_id:
        mission = agent.get_mission(mission_id)
        st.caption(f"Mission {mission_id} · Stage: {mission.stage.value} · "
                   f"Deadline: {mission.deadline.isoformat()}")
        if mission.stage == TransportStage.ORIGIN and st.button("Simulate organ boarded current flight"):
            agent.set_stage(mission_id, TransportStage.ONBOARD, "ONBOARD_DEMO_CURRENT")
            mission = agent.get_mission(mission_id)
        if mission.stage == TransportStage.ONBOARD and mission.diverted_to and st.button(
                "Simulate landing at diversion airport"):
            agent.set_stage(mission_id, TransportStage.TRANSFER, mission.diverted_to)
            mission = agent.get_mission(mission_id)

        with st.form("recovery_event"):
            event_type = st.selectbox("Simulated disruption", [e for e in EventType
                                      if e != EventType.STAGE_UPDATED],
                                      format_func=lambda e: e.value)
            delay_minutes = st.number_input("Delay / ETA change (minutes)",
                                            min_value=0, max_value=480, value=60)
            diversion_airport = st.selectbox("Diversion airport", ["LAX", "AZA"])
            submit_event = st.form_submit_button("Apply event and re-plan")
        if submit_event:
            occurred = datetime.now(timezone.utc).replace(second=0, microsecond=0)
            try:
                if event_type == EventType.FLIGHT_DELAYED:
                    event = DisruptionEvent(event_type, occurred, delay_minutes=int(delay_minutes))
                elif event_type == EventType.ETA_UPDATED:
                    eta = occurred + timedelta(minutes=int(delay_minutes))
                    event = DisruptionEvent(event_type, occurred, updated_eta=eta)
                elif event_type == EventType.FLIGHT_DIVERTED:
                    event = DisruptionEvent(event_type, occurred,
                                            updated_eta=occurred + timedelta(minutes=90),
                                            diversion_airport=diversion_airport)
                else:
                    event = DisruptionEvent(event_type, occurred)
                agent.handle_event(EventEnvelope(mission_id, uuid4().hex, event))
            except ValueError as exc:
                st.error(str(exc))

        st.button("Refresh recovery report")
        report = agent.latest_report(mission_id)
        if report:
            st.caption(f"Plan version {report['mission_version']} · Source: {report['data_source']} · "
                       "Human approval required")
            airports = report["departure_airports_considered"]
            st.write("**Departure airports examined:** " +
                     (", ".join(f"{a['airport']} ({a['transfer_minutes']} min transfer)" for a in airports)
                      if airports else "None reachable from the current stage"))
            candidates = [report["original_plan_assessment"], *report["alternative_plans"]]
            summary = pd.DataFrame([{
                "Rank": c["rank"], "Strategy": c["strategy"],
                "Flight": c["flight"]["flight_id"] if c["flight"] else "—",
                "Delivery ETA": c["estimated_delivery"] or "Unknown",
                "Margin (min)": c["feasibility"]["deadline_margin_minutes"],
                "Feasibility": c["feasibility"]["status"],
                "Unverified steps": len(c["feasibility"]["verification_requirements"]),
                "Flight risk": (c["flight_disruption_risk"]["probability"]
                                if c["flight_disruption_risk"]["probability"] is not None
                                else c["flight_disruption_risk"]["source"]),
            } for c in sorted(candidates, key=lambda c: c["rank"])])
            st.dataframe(summary, use_container_width=True, hide_index=True)
            for candidate in sorted(candidates, key=lambda c: c["rank"]):
                with st.expander(f"#{candidate['rank']} {candidate['strategy']} · "
                                 f"{candidate['feasibility']['status']}"):
                    st.write("Timeline", candidate["timeline"])
                    st.write("Verification required", candidate["feasibility"]["verification_requirements"])
                    if candidate["feasibility"]["reasons"]:
                        st.write("Infeasibility reasons", candidate["feasibility"]["reasons"])
            eligible = report["ranked_recommendation_candidates"]
            if eligible:
                review_id = st.selectbox("Candidate for human review",
                                         [c["candidate_id"] for c in eligible],
                                         key=f"review-{mission_id}-{report['mission_version']}")
                if st.button("Record review (no booking)"):
                    agent.review_candidate(mission_id, report["mission_version"], review_id)
                    st.success("Review recorded. Transport and handoffs still need confirmation.")
            with st.expander("Agent tool trace"):
                st.json(report["agent_trace"])
            st.caption(report["uncertainty"])
