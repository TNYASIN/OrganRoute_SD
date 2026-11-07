"""Integration checks against the teammate's actual calibrated model artifact."""

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from flight_model_adapter import FlightModelAdapter, _load_bundle
from recovery import (DisruptionEvent, EventType, Flight, rank_recovery_options)
from recovery_agent import EventEnvelope, RecoveryAgent
from recovery_demo import demo_mission


NOW = datetime(2030, 1, 1, 12, tzinfo=timezone.utc)


class TrainedModelIntegrationTests(unittest.TestCase):
    def test_actual_artifact_loads_once_and_maps_local_features(self):
        _load_bundle.cache_clear()
        flight = Flight("TEST", "PHX", "SAN", NOW + timedelta(minutes=90),
                        NOW + timedelta(minutes=170), carrier="WN")
        adapter = FlightModelAdapter()
        first = adapter(flight)
        second = adapter(flight)
        self.assertEqual(first, second)
        self.assertEqual(_load_bundle.cache_info().misses, 1)
        self.assertEqual(first["source"], "trained_flight_model")
        self.assertEqual(first["model_version"], 4)
        self.assertEqual(first["prediction_threshold_minutes"], 60)
        self.assertEqual(first["feature_values"], {
            "Origin": "PHX", "Reporting_Airline": "WN", "Dest": "SAN",
            "departure_hour": 6, "DayOfWeek": 2, "Month": 1})
        self.assertGreaterEqual(first["prediction_value"], 0)
        self.assertLessEqual(first["prediction_value"], 1)
        self.assertEqual(first["probability"], first["prediction_value"])

    def test_unsupported_route_and_missing_artifact_remain_unknown(self):
        flight = Flight("UNSUPPORTED", "AZA", "SAN", NOW + timedelta(hours=2),
                        NOW + timedelta(hours=3), carrier="WN")
        unsupported = FlightModelAdapter()(flight)
        self.assertFalse(unsupported["supported_route"])
        self.assertIsNone(unsupported["probability"])
        with tempfile.TemporaryDirectory() as tmp:
            missing = FlightModelAdapter(model_path=Path(tmp) / "missing.pkl")(flight)
        self.assertEqual(missing["source"], "missing_model_artifact")
        self.assertIsNone(missing["probability"])

    def test_initial_and_post_disruption_metadata_and_margins(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = RecoveryAgent(Path(tmp) / "agent.sqlite3")
            state = demo_mission("M", NOW, NOW + timedelta(hours=6))
            agent.create_mission(state)
            initial = agent.initial_assessment("M")
            self.assertEqual(initial["flight_disruption_risk"]["source"], "trained_flight_model")
            self.assertEqual(initial["deadline_margin_minutes"], 360 - 170 - 30 - 25)
            report = agent.handle_event(EventEnvelope(
                "M", "delay", DisruptionEvent(EventType.FLIGHT_DELAYED, NOW, delay_minutes=90)))
            current = report["original_plan_assessment"]
            self.assertEqual(current["flight_disruption_risk"]["source"],
                             "observed_event_not_modelled")
            self.assertIsNone(current["flight_disruption_risk"]["probability"])
            self.assertEqual(agent.get_mission("M").current_expected_arrival,
                             NOW + timedelta(minutes=260))
            self.assertEqual(current["feasibility"]["deadline_margin_minutes"],
                             360 - 260 - 30 - 25)
            alternative = next(c for c in report["alternative_plans"]
                               if c["flight"] and c["flight"]["origin"] == "PHX"
                               and c["flight"]["destination_airport"] == "SAN")
            self.assertEqual(alternative["flight_disruption_risk"]["source"],
                             "trained_flight_model")
            self.assertEqual(alternative["feasibility"]["deadline_margin_minutes"],
                             360 - 200 - 30 - 25)
            unknown = next(c for c in report["alternative_plans"]
                           if c["flight"] and c["flight"]["origin"] == "AZA")
            self.assertIsNone(unknown["flight_disruption_risk"]["probability"])
            self.assertEqual(unknown["flight_disruption_risk"]["source"], "unsupported_route")
            cancelled = agent.handle_event(EventEnvelope(
                "M", "cancel", DisruptionEvent(EventType.FLIGHT_CANCELLED,
                                                NOW + timedelta(minutes=1))))
            self.assertEqual(cancelled["original_plan_assessment"]["feasibility"]["status"],
                             "INFEASIBLE")
            self.assertTrue(cancelled["alternative_plans"])

    def test_unknown_flight_risk_is_not_ranked_as_zero(self):
        def option(identifier, risk):
            return {"candidate_id": identifier, "flight": {"flight_id": identifier},
                    "flight_disruption_risk": {"probability": risk},
                    "feasibility": {"status": "CONDITIONAL", "verification_requirements": [],
                                    "deadline_margin_minutes": 30}, "handoff_count": 1}
        ranked = rank_recovery_options([option("unknown", None), option("known", 0.3)])
        self.assertEqual([c["candidate_id"] for c in ranked], ["known", "unknown"])


if __name__ == "__main__":
    unittest.main()
