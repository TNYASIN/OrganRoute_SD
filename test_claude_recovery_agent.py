"""Tool-loop tests with a fake Claude client; no API key or network required."""

import tempfile
import unittest
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from claude_recovery_agent import ClaudeRecoveryAgent
from recovery import DisruptionEvent, EventType
from recovery_agent import EventEnvelope, RecoveryAgent
from recovery_demo import demo_mission


NOW = datetime(2030, 1, 1, 12, tzinfo=timezone.utc)


def tool(name, args=None, number=1):
    return SimpleNamespace(type="tool_use", name=name, input=args or {}, id=f"tool-{number}")


def response(*blocks, stop="tool_use"):
    return SimpleNamespace(stop_reason=stop, content=list(blocks))


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        return self.responses.pop(0)


class ClaudeAgentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.recovery = RecoveryAgent(Path(self.tmp.name) / "test.sqlite3")
        self.recovery.create_mission(demo_mission("M1", NOW, NOW + timedelta(hours=6)))
        self.report = self.recovery.handle_event(EventEnvelope(
            "M1", "delay-1", DisruptionEvent(EventType.FLIGHT_DELAYED, NOW, delay_minutes=60)))

    def tearDown(self):
        self.tmp.cleanup()

    def test_real_tool_loop_selects_only_computed_candidate_and_caches(self):
        cid = self.report["ranked_recommendation_candidates"][0]["candidate_id"]
        fake = SimpleNamespace(messages=FakeMessages([
            response(tool("get_mission_state", number=1)),
            response(tool("list_recovery_candidates", number=2),
                     tool("get_core_reference", number=3)),
            response(tool("inspect_candidate", {"candidate_id": cid}, number=4)),
            response(tool("submit_recovery_assessment", {
                "status": "RECOVERY_CANDIDATES", "candidate_ids": [cid],
                "reasoning": "This candidate has the strongest documented margin; verify all handoffs.",
                "verification_questions": ["Can the carrier accept the shipment?"]}, number=5)),
        ]))
        agent = ClaudeRecoveryAgent(self.recovery, client=fake, model="fake-test-model")
        assessment = agent.analyze("M1")
        self.assertEqual(assessment["selected_candidate_ids"], [cid])
        self.assertEqual(assessment["selected_options"][0]["estimated_delivery"],
                         next(c for c in [self.report["original_plan_assessment"],
                                        *self.report["alternative_plans"]]
                              if c["candidate_id"] == cid)["estimated_delivery"])
        self.assertTrue(assessment["human_approval_required"])
        self.assertFalse(assessment["booking_executed"])
        self.assertEqual(fake.messages.calls[0]["tool_choice"]["name"], "get_mission_state")
        self.assertEqual(fake.messages.calls[1]["messages"][-1]["content"][0]["type"], "tool_result")
        self.assertEqual(agent.analyze("M1"), assessment)
        self.assertEqual(len(fake.messages.calls), 4)

    def test_fabricated_candidate_is_rejected(self):
        fake = SimpleNamespace(messages=FakeMessages([
            response(tool("get_mission_state", number=1)),
            response(tool("list_recovery_candidates", number=2)),
            response(tool("submit_recovery_assessment", {
                "status": "RECOVERY_CANDIDATES", "candidate_ids": ["FAKE-FLIGHT"],
                "reasoning": "Take a made-up flight", "verification_questions": []}, number=3)),
            response(stop="end_turn"),
        ]))
        with self.assertRaises(RuntimeError):
            ClaudeRecoveryAgent(self.recovery, client=fake).analyze("M1")
        self.assertIsNone(self.recovery.get_ai_assessment("M1", self.report["mission_version"]))
        self.assertTrue(fake.messages.calls[-1]["messages"][-1]["content"][0]["is_error"])

    def test_missing_key_is_explicit_not_a_fake_ai_fallback(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}):
            with self.assertRaisesRegex(RuntimeError, "ANTHROPIC_API_KEY"):
                ClaudeRecoveryAgent(self.recovery).analyze("M1")

    def test_no_feasible_options_can_be_submitted_without_inventing_route(self):
        self.recovery.create_mission(demo_mission("M2", NOW, NOW + timedelta(minutes=20)))
        report = self.recovery.handle_event(EventEnvelope("M2", "cancel-2",
                       DisruptionEvent(EventType.FLIGHT_CANCELLED, NOW)))
        self.assertEqual(report["ranked_recommendation_candidates"], [])
        fake = SimpleNamespace(messages=FakeMessages([
            response(tool("get_mission_state", number=1)),
            response(tool("list_recovery_candidates", number=2)),
            response(tool("submit_recovery_assessment", {
                "status": "NO_FEASIBLE_OPTIONS", "candidate_ids": [],
                "reasoning": "No documented route reaches the user deadline.",
                "verification_questions": ["Can specialized dispatch provide an ETA?"]}, number=3)),
        ]))
        assessment = ClaudeRecoveryAgent(self.recovery, client=fake).analyze("M2")
        self.assertEqual(assessment["status"], "NO_FEASIBLE_OPTIONS")
        self.assertEqual(assessment["selected_options"], [])


if __name__ == "__main__":
    unittest.main()
