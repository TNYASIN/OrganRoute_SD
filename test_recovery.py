"""Focused tests for the deterministic recovery planning backend."""

import json
import unittest
from datetime import datetime, timedelta, timezone

from recovery import (DisruptionEvent, EventType, Flight, MissionState, TransportCatalog,
                      TransportStage, calculate_remaining_time, hypothetical_demo_catalog,
                      process_disruption)


START = datetime(2030, 1, 1, 12, tzinfo=timezone.utc)


def mission(*, deadline=None, stage=TransportStage.AIRPORT, eta=None):
    current = Flight("CURRENT", "PHX", "SAN", START + timedelta(hours=1),
                     START + timedelta(hours=2))
    return MissionState("M1", "Heart", "PHX", "SAN", "PHX", stage,
                        deadline or START + timedelta(hours=6), current,
                        eta or current.arrival, simulation_mode=True)


class RecoveryTests(unittest.TestCase):
    def test_delayed_flight_with_alternatives_and_state_history(self):
        state = mission()
        event = DisruptionEvent(EventType.FLIGHT_DELAYED, START, delay_minutes=90)
        report = process_disruption(state, event, hypothetical_demo_catalog(START))
        self.assertEqual(state.current_expected_arrival, START + timedelta(hours=3, minutes=30))
        self.assertEqual(len(state.event_history), 1)
        self.assertEqual(report["original_plan_assessment"]["feasibility"]["status"], "CONDITIONAL")
        self.assertIn("ALTERNATIVE_DIRECT_FLIGHT", [p["strategy"] for p in report["alternative_plans"]])
        self.assertIn("NEARBY_AIRPORT_GROUND_TRANSFER", [p["strategy"] for p in report["alternative_plans"]])
        self.assertTrue(report["ranked_recommendation_candidates"])
        self.assertTrue(report["human_approval_required"])
        self.assertIsNone(report["original_plan_assessment"]["flight_disruption_risk"]["probability"])
        json.dumps(report)

    def test_cancelled_flight_with_alternatives(self):
        state = mission()
        report = process_disruption(state, DisruptionEvent(EventType.FLIGHT_CANCELLED, START),
                                    hypothetical_demo_catalog(START))
        self.assertTrue(state.current_flight_cancelled)
        self.assertEqual(report["original_plan_assessment"]["feasibility"]["status"], "INFEASIBLE")
        self.assertIn("DIRECT_GROUND", [p["strategy"] for p in report["alternative_plans"]])
        self.assertTrue(report["ranked_recommendation_candidates"])

    def test_no_feasible_alternatives(self):
        state = mission(deadline=START + timedelta(minutes=20))
        report = process_disruption(state, DisruptionEvent(EventType.FLIGHT_CANCELLED, START),
                                    hypothetical_demo_catalog(START))
        self.assertEqual(report["ranked_recommendation_candidates"], [])
        self.assertTrue(all(p["feasibility"]["status"] == "INFEASIBLE"
                            for p in report["alternative_plans"]))

    def test_onboard_cannot_switch(self):
        state = mission(stage=TransportStage.ONBOARD)
        report = process_disruption(state, DisruptionEvent(EventType.ETA_UPDATED, START,
                                    updated_eta=START + timedelta(hours=3)),
                                    hypothetical_demo_catalog(START))
        self.assertEqual([p["strategy"] for p in report["alternative_plans"]],
                         ["SPECIALIZED_TRANSPORT_ESCALATION"])
        self.assertEqual(report["alternative_plans"][0]["feasibility"]["status"], "INFEASIBLE")
        self.assertEqual(report["ranked_recommendation_candidates"],
                         [{"rank": report["original_plan_assessment"]["rank"],
                           "candidate_id": report["original_plan_assessment"]["candidate_id"],
                           "strategy": "CONTINUE_CURRENT_FLIGHT",
                           "feasibility_status": "CONDITIONAL",
                           "deadline_margin_minutes": report["original_plan_assessment"]["feasibility"]["deadline_margin_minutes"],
                           "flight_disruption_risk": report["original_plan_assessment"]["flight_disruption_risk"],
                           "verification_requirements": report["original_plan_assessment"]["feasibility"]["verification_requirements"]}])

    def test_diversion_does_not_offer_airborne_switch(self):
        state = mission(stage=TransportStage.ONBOARD)
        report = process_disruption(state, DisruptionEvent(EventType.FLIGHT_DIVERTED, START,
                                    diversion_airport="LAX", updated_eta=START + timedelta(hours=2)),
                                    hypothetical_demo_catalog(START))
        self.assertEqual(report["updated_mission_status"]["diverted_to"], "LAX")
        self.assertEqual(report["original_plan_assessment"]["feasibility"]["status"], "INFEASIBLE")
        self.assertEqual(report["ranked_recommendation_candidates"], [])

    def test_injected_flight_risk_is_flight_only(self):
        report = process_disruption(mission(), DisruptionEvent(EventType.FLIGHT_DELAYED, START,
                                    delay_minutes=30), hypothetical_demo_catalog(START),
                                    flight_risk_predictor=lambda flight: 0.25)
        self.assertIsNone(report["original_plan_assessment"]["flight_disruption_risk"]["probability"])
        self.assertEqual(report["original_plan_assessment"]["flight_disruption_risk"]["source"],
                         "observed_event_not_modelled")
        alternative = next(p for p in report["alternative_plans"] if p["flight"])
        self.assertEqual(alternative["flight_disruption_risk"]["probability"], 0.25)
        direct_ground = next(p for p in report["alternative_plans"] if p["strategy"] == "DIRECT_GROUND")
        self.assertEqual(direct_ground["flight_disruption_risk"]["source"], "not_applicable")

    def test_missing_and_unsupported_flight_data(self):
        state = mission()
        report = process_disruption(state, DisruptionEvent(EventType.FLIGHT_DELAYED, START,
                                    delay_minutes=30), TransportCatalog())
        self.assertIsNone(report["original_plan_assessment"]["flight_disruption_risk"]["probability"])
        self.assertEqual(report["data_source"], "hypothetical_demo")
        with self.assertRaises(ValueError):
            hypothetical_demo_catalog(START, "JFK", "SAN")
        unsupported = process_disruption(mission(), DisruptionEvent(EventType.FLIGHT_DELAYED, START,
                                         delay_minutes=30), flight_risk_predictor=lambda flight: None)
        self.assertIsNone(unsupported["original_plan_assessment"]["flight_disruption_risk"]["probability"])
        with self.assertRaises(ValueError):
            process_disruption(mission(), DisruptionEvent(EventType.FLIGHT_DELAYED, START,
                               delay_minutes=30), TransportCatalog(source="unconfigured_provider"))

    def test_negative_remaining_time(self):
        state = mission(deadline=START - timedelta(minutes=1))
        self.assertEqual(calculate_remaining_time(state, START), -1)
        report = process_disruption(state, DisruptionEvent(EventType.FLIGHT_CANCELLED, START))
        self.assertEqual(report["updated_mission_status"]["remaining_minutes"], -1)
        self.assertEqual(report["ranked_recommendation_candidates"], [])

    def test_invalid_eta_inputs(self):
        with self.assertRaises(ValueError):
            DisruptionEvent(EventType.ETA_UPDATED, START)
        with self.assertRaises(ValueError):
            DisruptionEvent(EventType.ETA_UPDATED, START, updated_eta=START - timedelta(minutes=1))
        with self.assertRaises(ValueError):
            DisruptionEvent(EventType.ETA_UPDATED, START, updated_eta=datetime(2030, 1, 1))
        with self.assertRaises(ValueError):
            Flight("X", "PHX", "SAN", START, START - timedelta(minutes=1))

    def test_simulation_is_explicit_and_events_are_ordered(self):
        state = mission()
        state.simulation_mode = False
        with self.assertRaises(ValueError):
            process_disruption(state, DisruptionEvent(EventType.FLIGHT_CANCELLED, START))
        self.assertEqual(state.event_history, [])
        state.simulation_mode = True
        process_disruption(state, DisruptionEvent(EventType.FLIGHT_DELAYED, START,
                                                  delay_minutes=10))
        with self.assertRaises(ValueError):
            process_disruption(state, DisruptionEvent(EventType.ETA_UPDATED,
                               START - timedelta(minutes=1), updated_eta=START + timedelta(hours=3)))
        self.assertEqual(len(state.event_history), 1)


if __name__ == "__main__":
    unittest.main()
