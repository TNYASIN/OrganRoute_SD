"""Claude tool-use agent over the deterministic OrganRoute recovery planner.

Claude chooses which read-only evidence tools to call. It can select only
candidate IDs produced by recovery.py; it cannot alter route calculations,
confirm transport availability, book travel, or make a medical decision.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from recovery_agent import RecoveryAgent


SYSTEM = """You are OrganRoute's transport recovery investigation agent.
Use tools to inspect the mission and candidate routes before submitting an assessment.
You MUST call list_recovery_candidates, and inspect each candidate you select.
Select only existing, non-infeasible candidate IDs. Never invent a flight, route,
carrier acceptance, confirmed availability, or medical deadline. The user-provided
mission deadline is authoritative. Core reference constants are context only.
All demo transport data is hypothetical. Preserve unknown risk and unverified steps.
If no candidate meets the deadline, submit NO_FEASIBLE_OPTIONS with no IDs.
Your submit_recovery_assessment tool call must be the sole tool call in its turn.
Your output is decision support; a human must verify logistics and approve action.
"""


TOOLS = [
    {"name": "get_mission_state", "description": "Read the current saved mission, latest event, and deadline.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "get_core_reference", "description": "Read existing core.py organ and airport assumptions; historical BTS is not live availability.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "list_recovery_candidates", "description": "List deterministic candidate IDs, feasibility, margin, risk, and verification needs.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "inspect_candidate", "description": "Read the full computed timeline and source data for one candidate ID.",
     "input_schema": {"type": "object", "properties": {
         "candidate_id": {"type": "string"}}, "required": ["candidate_id"],
         "additionalProperties": False}},
    {"name": "submit_recovery_assessment", "description": "Submit a validated ranked shortlist and questions for human verification; no booking.",
     "input_schema": {"type": "object", "properties": {
         "status": {"type": "string", "enum": ["RECOVERY_CANDIDATES", "NO_FEASIBLE_OPTIONS"]},
         "candidate_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
         "reasoning": {"type": "string"},
         "verification_questions": {"type": "array", "items": {"type": "string"}, "maxItems": 5}},
         "required": ["status", "candidate_ids", "reasoning", "verification_questions"],
         "additionalProperties": False}},
]


def _block_dict(block) -> dict:
    if isinstance(block, dict):
        return block
    if block.type == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    if block.type == "text":
        return {"type": "text", "text": block.text}
    return block.model_dump(exclude_none=True)


class ClaudeRecoveryAgent:
    def __init__(self, recovery: RecoveryAgent, client=None, model: str | None = None,
                 max_turns: int = 10):
        self.recovery = recovery
        self.client = client
        self.model = model or os.getenv("RECOVERY_CLAUDE_MODEL") or os.getenv("AI_MODEL") or "claude-opus-5-5"
        self.max_turns = max_turns
        if max_turns < 2:
            raise ValueError("max_turns must be at least 2")

    def _client(self):
        if self.client is not None:
            return self.client
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("Set ANTHROPIC_API_KEY to run the Claude recovery agent")
        import anthropic
        self.client = anthropic.Anthropic(api_key=key, timeout=30.0)
        return self.client

    def _core_reference(self, report: dict) -> dict:
        try:
            import core
        except ImportError as exc:
            return {"source": "core.py", "available": False, "reason": str(exc)}
        organ = report["updated_mission_status"]["organ_type"]
        airport_codes = {c["flight"]["origin"] for c in
                         [report["original_plan_assessment"], *report["alternative_plans"]]
                         if c["flight"]}
        return {"source": "core.py", "available": True,
                **core.recovery_reference(organ, airport_codes,
                                          report["updated_mission_status"]["destination"]),
                "historical_bts_file_present": Path(__file__).with_name("flights_san.csv").exists(),
                "note": "These are legacy assumptions, not live schedules or a replacement for the mission deadline."}

    @staticmethod
    def _candidates(report: dict) -> list[dict]:
        return [report["original_plan_assessment"], *report["alternative_plans"]]

    def _submit(self, mission_id: str, report: dict, arguments: dict,
                listed: bool, inspected: set[str], trace: list[dict]) -> dict:
        if not listed:
            raise ValueError("list_recovery_candidates must be called before submission")
        if self.recovery.latest_report(mission_id)["mission_version"] != report["mission_version"]:
            raise ValueError("mission plan changed during the agent run")
        ids = arguments.get("candidate_ids")
        status = arguments.get("status")
        reasoning = arguments.get("reasoning")
        questions = arguments.get("verification_questions")
        if not isinstance(ids, list) or len(ids) > 3 or len(ids) != len(set(ids)):
            raise ValueError("submit one to three unique candidate IDs, or none if no route is feasible")
        if not isinstance(reasoning, str) or not reasoning.strip() or len(reasoning) > 1200:
            raise ValueError("reasoning must be a nonempty string of at most 1200 characters")
        if not isinstance(questions, list) or len(questions) > 5 or any(
                not isinstance(q, str) or not q.strip() for q in questions):
            raise ValueError("verification_questions must be at most five nonempty strings")
        eligible = {c["candidate_id"] for c in self._candidates(report)
                    if c["feasibility"]["status"] != "INFEASIBLE"}
        if ids and (status != "RECOVERY_CANDIDATES" or not set(ids) <= eligible):
            raise ValueError("all selected IDs must be feasible candidates in this plan")
        if any(candidate_id not in inspected for candidate_id in ids):
            raise ValueError("inspect every selected candidate before submission")
        if not ids and (status != "NO_FEASIBLE_OPTIONS" or eligible):
            raise ValueError("empty shortlist is allowed only when no feasible candidates exist")
        if ids and status != "RECOVERY_CANDIDATES":
            raise ValueError("status does not match selected candidates")
        selected = [next(c for c in self._candidates(report) if c["candidate_id"] == cid)
                    for cid in ids]
        return {"mission_id": mission_id, "plan_version": report["mission_version"],
                "source": "claude_tool_agent", "model": self.model, "status": status,
                "selected_candidate_ids": ids,
                "selected_options": [{"candidate_id": c["candidate_id"], "strategy": c["strategy"],
                                      "rank": c["rank"], "estimated_delivery": c["estimated_delivery"],
                                      "deadline_margin_minutes": c["feasibility"]["deadline_margin_minutes"],
                                      "feasibility_status": c["feasibility"]["status"],
                                      "verification_requirements": c["feasibility"]["verification_requirements"],
                                      "flight_disruption_risk": c["flight_disruption_risk"]}
                                     for c in selected],
                "agent_reasoning": reasoning.strip(), "verification_questions": questions,
                "data_source": report["data_source"], "human_approval_required": True,
                "booking_executed": False, "tool_trace": trace}

    def analyze(self, mission_id: str, force: bool = False) -> dict:
        report = self.recovery.latest_report(mission_id)
        if report is None:
            raise ValueError("create a recovery plan before running the Claude agent")
        if not force:
            cached = self.recovery.get_ai_assessment(mission_id, report["mission_version"])
            if cached is not None:
                return cached
        client = self._client()
        messages = [{"role": "user", "content":
                     f"Investigate mission {mission_id}, plan version {report['mission_version']}. "
                     "Use the tools to select the safest documented recovery candidates."}]
        listed = False
        inspected: set[str] = set()
        trace: list[dict] = []
        for turn in range(self.max_turns):
            response = client.messages.create(
                model=self.model, max_tokens=1600, system=SYSTEM, tools=TOOLS,
                tool_choice=({"type": "tool", "name": "get_mission_state"}
                             if turn == 0 else {"type": "auto"}), messages=messages)
            if response.stop_reason != "tool_use":
                raise RuntimeError(f"Claude stopped before submitting an assessment: {response.stop_reason}")
            blocks = [_block_dict(block) for block in response.content]
            calls = [block for block in blocks if block["type"] == "tool_use"]
            if not calls:
                raise RuntimeError("Claude returned tool_use without a tool call")
            messages.append({"role": "assistant", "content": blocks})
            results = []
            for call in calls:
                name = call["name"]
                arguments = call["input"]
                try:
                    if name == "get_mission_state":
                        output = {"mission": report["updated_mission_status"],
                                  "event": report["event"], "data_source": report["data_source"]}
                    elif name == "get_core_reference":
                        output = self._core_reference(report)
                    elif name == "list_recovery_candidates":
                        listed = True
                        output = report["ranked_recommendation_candidates"]
                        output = {"recommended_candidates": output,
                                  "all_candidates": [{"candidate_id": c["candidate_id"],
                                                      "strategy": c["strategy"],
                                                      "status": c["feasibility"]["status"]}
                                                     for c in self._candidates(report)]}
                    elif name == "inspect_candidate":
                        cid = arguments.get("candidate_id")
                        output = next((c for c in self._candidates(report)
                                       if c["candidate_id"] == cid), None)
                        if output is None:
                            raise ValueError("unknown candidate ID")
                        inspected.add(cid)
                    elif name == "submit_recovery_assessment":
                        if len(calls) != 1:
                            raise ValueError("submit_recovery_assessment must be the sole tool call")
                        assessment = self._submit(mission_id, report, arguments, listed, inspected,
                                                  [*trace, {"tool": name, "result": "accepted"}])
                        self.recovery.save_ai_assessment(mission_id, report["mission_version"], assessment)
                        return assessment
                    else:
                        raise ValueError("unknown or unauthorized tool")
                    trace.append({"tool": name, "result": "ok"})
                    results.append({"type": "tool_result", "tool_use_id": call["id"],
                                    "content": json.dumps(output)})
                except (KeyError, TypeError, ValueError) as exc:
                    trace.append({"tool": name, "result": "error"})
                    results.append({"type": "tool_result", "tool_use_id": call["id"],
                                    "content": str(exc), "is_error": True})
            messages.append({"role": "user", "content": results})
        raise RuntimeError("Claude exceeded the configured tool-use turn limit")
