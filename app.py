"""OrganRoute SD - Streamlit UI.  Run:  streamlit run app.py"""
import os
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st

import core
from claude_recovery_agent import ClaudeRecoveryAgent
from recovery import DisruptionEvent, EventType, TransportStage
from recovery_agent import EventEnvelope, RecoveryAgent
from recovery_demo import demo_mission
import live
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
        if st.button(f"📲 Send to team ({notify.describe()})", key=f"{key}_send"):
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
    now = core.sd_now()
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

tab1, tab2, tab3 = st.tabs(["1 · Plan transport", "2 · Delay alert → re-plan",
                           "3 · Recovery agent demo"])

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
    st.session_state["rp_source"] = f"{source} at {core.sd_now().strftime('%I:%M %p')}"
    st.session_state.pop("text_result", None)
    auto_text(st.session_state["rp"][1][0])
    return st.session_state["rp"]


def auto_text(plan: dict):
    """Text the team about a new re-plan (live or simulated) if the toggle is on and Twilio is set up."""
    if notify is None or not notify.enabled() or not st.session_state.get("auto_notify", True):
        return
    ok, note = notify.send_alert(f"OrganRoute {plan.get('status', '').upper()}: {plan.get('recommendation', '')}"[:120],
                                 plan.get("message_draft", ""), plan.get("status", "yellow"))
    st.session_state["rp_source"] += f" · team {'alerted' if ok else 'NOT alerted'}"
    st.session_state["text_result"] = (ok, note)


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
    st.caption(f"Checked {core.sd_now().strftime('%I:%M %p')} · refreshes every 60 s")

    key = f"{chosen.label}|{delay // 15}|{cause[:40]}"
    if delay >= LIVE_TRIGGER_MIN and key != st.session_state.get("last_live_key"):
        st.session_state["last_live_key"] = key
        run_replan(chosen, delay, cause, departed, core.sd_now(), source)
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
        if notify is not None:
            if notify.enabled():
                st.toggle(f"📲 Alert the team automatically on delay alerts, live or simulated ({notify.describe()})",
                          value=True, key="auto_notify")
            else:
                st.caption(f"📲 Team alerts are off: {notify.setup_hint()}, then restart the app.")
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
            if st.session_state.get("watching"):
                live_panel(chosen, ident)

        if "rp" in st.session_state:
            rp, plan = st.session_state["rp"]
            if st.session_state.get("rp_source"):
                st.caption(f"Re-plan triggered by: {st.session_state['rp_source']}")
            if st.session_state.get("text_result"):
                ok, note = st.session_state["text_result"]
                (st.success if ok else st.error)(f"📲 {note}")
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

    def run_claude_recovery(mission_id: str):
        if not os.getenv("ANTHROPIC_API_KEY"):
            return
        try:
            with st.spinner("Claude is investigating recovery options with tools..."):
                ClaudeRecoveryAgent(agent).analyze(mission_id)
        except Exception as exc:
            st.error(f"Claude assessment failed: {exc}")

    with st.form("new_recovery_mission"):
        demo_organ = st.selectbox("Organ for simulated mission", list(core.VIABILITY_HOURS),
                                  key="demo_organ")
        demo_hospital = st.selectbox("Receiving hospital", list(core.HOSPITALS),
                                     key="demo_hospital")
        deadline_hours = st.number_input("User-specified deadline: hours from mission start",
                                         min_value=1, max_value=48, value=6)
        demo_risk_threshold = st.number_input("Demo flight risk flag threshold (%)",
                                              min_value=0, max_value=100, value=25)
        create_demo = st.form_submit_button("Create simulated PHX → San Diego mission")
    if create_demo:
        started = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        mission_id = f"demo-{uuid4().hex[:12]}"
        agent = RecoveryAgent(Path(__file__).with_name("recovery_demo.sqlite3"),
                              risk_flag_threshold=demo_risk_threshold / 100)
        agent.create_mission(demo_mission(mission_id, started,
                                          started + timedelta(hours=int(deadline_hours)), demo_organ,
                                          demo_hospital, core.HOSPITALS[demo_hospital][0]))
        st.session_state["recovery_mission_id"] = mission_id

    mission_id = st.session_state.get("recovery_mission_id")
    if mission_id:
        mission = agent.get_mission(mission_id)
        initial = agent.initial_assessment(mission_id) or {
            "flight_id": mission.current_flight.flight_id if mission.current_flight else None,
            "scheduled_delivery_eta": None, "deadline_margin_minutes": None,
            "flight_disruption_risk": {"probability": None, "source": "legacy_mission_no_initial_assessment"},
            "elevated_flight_risk_demo_flag": False, "risk_flag_threshold": agent.risk_flag_threshold}
        st.write("**Before disruption: scheduled plan (simulated flight)**")
        st.write({"Flight": initial["flight_id"],
                  "Scheduled delivery ETA": initial["scheduled_delivery_eta"],
                  "Deadline margin (min)": initial["deadline_margin_minutes"],
                  "Historical flight prediction target": initial["flight_disruption_risk"].get("prediction_type") or "Unknown",
                  "Historical flight probability": initial["flight_disruption_risk"]["probability"]})
        if initial["elevated_flight_risk_demo_flag"]:
            st.info(f"Demo flight risk flag: probability meets the configurable {initial['risk_flag_threshold']:.0%} threshold. Verify options before acting.")
        with st.expander("Initial model coverage and warnings"):
            st.json(initial["flight_disruption_risk"])
        if agent.latest_report(mission_id) is None and st.button("Prepare recovery alternatives now"):
            try:
                prepared_at = datetime.now(timezone.utc).replace(second=0, microsecond=0)
                agent.handle_event(EventEnvelope(mission_id, uuid4().hex,
                    DisruptionEvent(EventType.ETA_UPDATED, prepared_at,
                                    updated_eta=mission.current_expected_arrival)))
            except ValueError as exc:
                st.error(str(exc))
        st.caption(f"Mission {mission_id} · Stage: {mission.stage.value} · "
                   f"Deadline: {mission.deadline.isoformat()}")
        if mission.stage == TransportStage.ORIGIN and st.button("Simulate organ boarded current flight"):
            agent.set_stage(mission_id, TransportStage.ONBOARD, "ONBOARD_DEMO_CURRENT")
            run_claude_recovery(mission_id)
            mission = agent.get_mission(mission_id)
        if mission.stage == TransportStage.ONBOARD and mission.diverted_to and st.button(
                "Simulate landing at diversion airport"):
            agent.set_stage(mission_id, TransportStage.TRANSFER, mission.diverted_to)
            run_claude_recovery(mission_id)
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
                run_claude_recovery(mission_id)
            except ValueError as exc:
                st.error(str(exc))

        st.button("Refresh recovery report")
        report = agent.latest_report(mission_id)
        if report:
            st.write("Observed simulation event", report["event"])
            st.write("Updated current-flight ETA",
                     report["updated_mission_status"]["current_expected_arrival"] or "Unavailable")
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
                "Historical P(flight cancelled, diverted, or >60 min late)": (f"{c['flight_disruption_risk']['probability']:.1%}"
                                if c["flight_disruption_risk"]["probability"] is not None
                                else "Unknown / " + c["flight_disruption_risk"]["source"]),
            } for c in sorted(candidates, key=lambda c: c["rank"])])
            st.dataframe(summary, use_container_width=True, hide_index=True)
            st.caption(report.get("ranking_basis", "Review deadline, verification requirements, and unknown risk before acting."))
            for candidate in sorted(candidates, key=lambda c: c["rank"]):
                with st.expander(f"#{candidate['rank']} {candidate['strategy']} · "
                                 f"{candidate['feasibility']['status']}"):
                    st.write("Timeline", candidate["timeline"])
                    st.write("Historical flight risk and model coverage", candidate["flight_disruption_risk"])
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
            if not os.getenv("ANTHROPIC_API_KEY"):
                st.info("Claude recovery agent is not configured. Set ANTHROPIC_API_KEY, "
                        "then run a new event or click below.")
            if st.button("Run Claude tool agent for this plan"):
                run_claude_recovery(mission_id)
            assessment = agent.get_ai_assessment(mission_id, report["mission_version"])
            if assessment:
                st.subheader("Claude tool agent assessment")
                st.caption(f"Model: {assessment['model']} · Plan version: {assessment['plan_version']} · "
                           "Human approval required")
                st.write(assessment["agent_reasoning"])
                st.write("Selected computed candidates", assessment["selected_options"])
                st.write("Questions to verify", assessment["verification_questions"])
                with st.expander("Claude tool calls"):
                    st.json(assessment["tool_trace"])
            st.caption(report["uncertainty"])
