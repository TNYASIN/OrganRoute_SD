"""Persistence, airport discovery, event loop, and provenance tests."""

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from recovery import DisruptionEvent, EventType, TransportStage
from recovery_agent import EventEnvelope, JsonlEventSource, RecoveryAgent
from recovery_demo import DONOR_SITE, demo_mission


NOW = datetime(2030, 1, 1, 12, tzinfo=timezone.utc)


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "agent.sqlite3"
        self.agent = RecoveryAgent(self.db)
        self.agent.create_mission(demo_mission("M1", NOW, NOW + timedelta(hours=6)))

    def tearDown(self):
        self.tmp.cleanup()

    def test_discovers_multiple_airports_and_complete_timelines(self):
        report = self.agent.handle_event(EventEnvelope(
            "M1", "delay-1", DisruptionEvent(EventType.FLIGHT_DELAYED, NOW, delay_minutes=60)))
        self.assertEqual({a["airport"] for a in report["departure_airports_considered"]},
                         {"PHX", "AZA"})
        options = [report["original_plan_assessment"], *report["alternative_plans"]]
        self.assertEqual(report["original_plan_assessment"]["feasibility"]["status"], "CONDITIONAL")
        self.assertEqual(report["original_plan_assessment"]["timeline"][-1]["location"],
                         report["updated_mission_status"]["destination"])
        self.assertEqual(sum(o["strategy"] == "ALTERNATIVE_DIRECT_FLIGHT" for o in options), 2)
        self.assertIn("NEARBY_AIRPORT_GROUND_TRANSFER", [o["strategy"] for o in options])
        aza = next(o for o in options if o["flight"] and o["flight"]["origin"] == "AZA")
        self.assertEqual(aza["timeline"][1]["step"], "departure_airport_transfer")
        self.assertEqual(aza["timeline"][-1]["location"], report["updated_mission_status"]["destination"])
        self.assertEqual(aza["access_ground"]["source"], "hypothetical_demo")
        self.assertEqual(aza["feasibility"]["status"], "CONDITIONAL")
        self.assertTrue(report["human_approval_required"])
        json.dumps(report)

    def test_event_idempotency_and_restart(self):
        envelope = EventEnvelope("M1", "delay-1", DisruptionEvent(EventType.FLIGHT_DELAYED, NOW,
                                                                  delay_minutes=60))
        first = self.agent.handle_event(envelope)
        self.assertEqual(self.agent.handle_event(envelope), first)
        restarted = RecoveryAgent(self.db)
        self.assertEqual(restarted.latest_report("M1"), first)
        self.assertEqual(len(restarted.get_mission("M1").event_history), 1)
        self.assertEqual(restarted.get_mission("M1").current_expected_arrival,
                         NOW + timedelta(minutes=230))

    def test_polling_source_triggers_replanning(self):
        class Source:
            def poll(self):
                return [EventEnvelope("M1", "cancel-1",
                                      DisruptionEvent(EventType.FLIGHT_CANCELLED, NOW))]
        reports = self.agent.run_once(Source())
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0]["original_plan_assessment"]["feasibility"]["status"], "INFEASIBLE")
        self.assertTrue(self.agent.get_mission("M1").current_flight_cancelled)

    def test_jsonl_event_source_is_replay_safe(self):
        path = Path(self.tmp.name) / "events.jsonl"
        path.write_text(json.dumps({"mission_id": "M1", "event_id": "delay-file-1",
                                    "event": {"type": "FLIGHT_DELAYED",
                                              "occurred_at": NOW.isoformat(), "delay_minutes": 45}}) + "\n",
                        encoding="utf-8")
        source = JsonlEventSource(path)
        first = self.agent.run_once(source)
        second = self.agent.run_once(source)
        self.assertEqual(first, second)
        self.assertEqual(len(self.agent.get_mission("M1").event_history), 1)

    def test_review_records_only_current_feasible_candidate(self):
        report = self.agent.handle_event(EventEnvelope("M1", "delay-review-1",
                       DisruptionEvent(EventType.FLIGHT_DELAYED, NOW, delay_minutes=45)))
        choice = report["ranked_recommendation_candidates"][0]["candidate_id"]
        review = self.agent.review_candidate("M1", report["mission_version"], choice)
        self.assertEqual(review["effect"], "review_only_no_booking")
        self.assertEqual(self.agent.review_candidate("M1", report["mission_version"], choice), review)
        with self.assertRaises(ValueError):
            self.agent.review_candidate("M1", report["mission_version"], "missing")
        newer = self.agent.handle_event(EventEnvelope("M1", "eta-review-2",
                DisruptionEvent(EventType.ETA_UPDATED, NOW + timedelta(minutes=1),
                                updated_eta=NOW + timedelta(hours=4))))
        self.assertGreater(newer["mission_version"], report["mission_version"])
        with self.assertRaises(ValueError):
            self.agent.review_candidate("M1", report["mission_version"], choice)

    def test_stage_update_prevents_airborne_switch(self):
        stage_report = self.agent.set_stage("M1", TransportStage.ONBOARD, "ONBOARD_DEMO_CURRENT", at=NOW)
        self.assertEqual(stage_report["event"]["type"], "STAGE_UPDATED")
        report = self.agent.handle_event(EventEnvelope(
            "M1", "eta-1", DisruptionEvent(EventType.ETA_UPDATED, NOW,
                                            updated_eta=NOW + timedelta(hours=3))))
        self.assertEqual(report["updated_mission_status"]["transport_stage"], "ONBOARD_AIRCRAFT")
        self.assertEqual([p["strategy"] for p in report["alternative_plans"]],
                         ["SPECIALIZED_TRANSPORT_ESCALATION"])

    def test_landing_after_diversion_replans_from_diversion_airport(self):
        self.agent.set_stage("M1", TransportStage.ONBOARD, "ONBOARD_DEMO_CURRENT", at=NOW)
        self.agent.handle_event(EventEnvelope("M1", "divert-1",
            DisruptionEvent(EventType.FLIGHT_DIVERTED, NOW + timedelta(minutes=5),
                            updated_eta=NOW + timedelta(minutes=95), diversion_airport="LAX")))
        report = self.agent.set_stage("M1", TransportStage.TRANSFER, "LAX",
                                      at=NOW + timedelta(minutes=100))
        self.assertEqual(report["updated_mission_status"]["current_location"], "LAX")
        ground = next(p for p in report["alternative_plans"] if p["strategy"] == "DIRECT_GROUND")
        self.assertEqual(ground["feasibility"]["status"], "CONDITIONAL")
        self.assertEqual(ground["timeline"][0]["location"], "LAX")

    def test_late_event_does_not_corrupt_state(self):
        self.agent.handle_event(EventEnvelope("M1", "cancel-1",
                          DisruptionEvent(EventType.FLIGHT_CANCELLED, NOW)))
        before = self.agent.get_mission("M1")
        with self.assertRaises(ValueError):
            self.agent.handle_event(EventEnvelope("M1", "old-1",
                DisruptionEvent(EventType.ETA_UPDATED, NOW - timedelta(minutes=1),
                                updated_eta=NOW + timedelta(hours=3))))
        after = self.agent.get_mission("M1")
        self.assertEqual(after.event_history, before.event_history)

    def test_demo_location_is_not_presented_as_real_address(self):
        state = self.agent.get_mission("M1")
        self.assertEqual(state.current_location, DONOR_SITE)
        self.assertTrue(state.simulation_mode)


if __name__ == "__main__":
    unittest.main()
