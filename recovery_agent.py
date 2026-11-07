"""Durable, event-driven recovery orchestrator. No LLM or external provider calls."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Protocol
from uuid import uuid4

from recovery import (DisruptionEvent, EventType, Flight, FlightRiskPredictor, MissionState,
                      TransportCatalog, TransportStage, process_disruption)
from recovery_demo import demo_catalog


class EventSource(Protocol):
    def poll(self) -> list["EventEnvelope"]: ...


@dataclass(frozen=True)
class EventEnvelope:
    mission_id: str
    event_id: str
    event: DisruptionEvent


class JsonlEventSource:
    """Read simulation events appended by another process; event IDs make replay safe."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def poll(self) -> list[EventEnvelope]:
        if not self.path.exists():
            return []
        envelopes = []
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                event_data = raw["event"]
                event_data["type"] = EventType(event_data["type"])
                event_data["occurred_at"] = datetime.fromisoformat(event_data["occurred_at"])
                if event_data.get("updated_eta"):
                    event_data["updated_eta"] = datetime.fromisoformat(event_data["updated_eta"])
                if event_data.get("new_stage"):
                    event_data["new_stage"] = TransportStage(event_data["new_stage"])
                envelopes.append(EventEnvelope(raw["mission_id"], raw["event_id"],
                                               DisruptionEvent(**event_data)))
            except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid simulation event on line {line_number}: {exc}") from exc
        return envelopes


def _json_default(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _restore_state(raw: str) -> MissionState:
    data = json.loads(raw)
    data["deadline"] = datetime.fromisoformat(data["deadline"])
    data["stage"] = TransportStage(data["stage"])
    if data["current_expected_arrival"]:
        data["current_expected_arrival"] = datetime.fromisoformat(data["current_expected_arrival"])
    if data["current_flight"]:
        flight = data["current_flight"]
        for key in ("departure", "arrival", "observed_at", "expires_at"):
            if flight.get(key):
                flight[key] = datetime.fromisoformat(flight[key])
        data["current_flight"] = Flight(**flight)
    return MissionState(**data)


def discover_departure_airports(state: MissionState, catalog: TransportCatalog,
                                at: datetime) -> list[dict]:
    """Find airports reachable from the mission's current location in this catalog."""
    result = []
    for airport in sorted({f.origin for f in catalog.flights} |
                          ({state.current_flight.origin} if state.current_flight else set())):
        if airport == state.current_location:
            result.append({"airport": airport, "transfer_minutes": 0, "source": "current_location"})
        elif state.stage == TransportStage.ORIGIN:
            transfer = next((g for g in catalog.ground
                             if g.origin == state.current_location and g.destination == airport), None)
            if transfer:
                result.append({"airport": airport, "transfer_minutes": transfer.duration_minutes,
                               "source": transfer.source, "observed_at":
                               transfer.observed_at.isoformat() if transfer.observed_at else None,
                               "expires_at": transfer.expires_at.isoformat() if transfer.expires_at else None})
    return result


class RecoveryAgent:
    """Accept events, select read-only planning tools, persist state and reports."""

    def __init__(self, database: str | Path,
                 catalog_provider: Callable[[MissionState, datetime], TransportCatalog] = demo_catalog,
                 flight_risk_predictor: FlightRiskPredictor | None = None,
                 provider_configured: bool = False):
        self.database = str(database)
        self.catalog_provider = catalog_provider
        self.flight_risk_predictor = flight_risk_predictor
        self.provider_configured = provider_configured
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS missions (
                mission_id TEXT PRIMARY KEY, state_json TEXT NOT NULL,
                version INTEGER NOT NULL, latest_report_json TEXT)""")
            db.execute("""CREATE TABLE IF NOT EXISTS events (
                mission_id TEXT NOT NULL, event_id TEXT NOT NULL,
                event_json TEXT NOT NULL, report_json TEXT NOT NULL,
                PRIMARY KEY (mission_id, event_id))""")
            db.execute("""CREATE TABLE IF NOT EXISTS reviews (
                mission_id TEXT NOT NULL, plan_version INTEGER NOT NULL,
                candidate_id TEXT NOT NULL, reviewed_at TEXT NOT NULL,
                PRIMARY KEY (mission_id, plan_version, candidate_id))""")
            db.execute("""CREATE TABLE IF NOT EXISTS ai_assessments (
                mission_id TEXT NOT NULL, plan_version INTEGER NOT NULL,
                assessment_json TEXT NOT NULL,
                PRIMARY KEY (mission_id, plan_version))""")

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.database, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def create_mission(self, state: MissionState) -> None:
        if not state.simulation_mode:
            raise ValueError("this agent currently accepts simulation missions only")
        with self._connect() as db:
            db.execute("INSERT INTO missions VALUES (?, ?, 0, NULL)",
                       (state.mission_id, json.dumps(asdict(state), default=_json_default)))

    def get_mission(self, mission_id: str) -> MissionState:
        with self._connect() as db:
            row = db.execute("SELECT state_json FROM missions WHERE mission_id=?", (mission_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown mission {mission_id}")
        return _restore_state(row["state_json"])

    def latest_report(self, mission_id: str) -> dict | None:
        with self._connect() as db:
            row = db.execute("SELECT latest_report_json FROM missions WHERE mission_id=?",
                             (mission_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown mission {mission_id}")
        return json.loads(row["latest_report_json"]) if row["latest_report_json"] else None

    def get_ai_assessment(self, mission_id: str, plan_version: int) -> dict | None:
        with self._connect() as db:
            row = db.execute("SELECT assessment_json FROM ai_assessments WHERE mission_id=? "
                             "AND plan_version=?", (mission_id, plan_version)).fetchone()
        return json.loads(row["assessment_json"]) if row else None

    def save_ai_assessment(self, mission_id: str, plan_version: int, assessment: dict) -> None:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT latest_report_json FROM missions WHERE mission_id=?",
                             (mission_id,)).fetchone()
            if row is None or row["latest_report_json"] is None:
                raise ValueError("mission has no current plan")
            if json.loads(row["latest_report_json"])["mission_version"] != plan_version:
                raise ValueError("plan changed during AI assessment")
            db.execute("INSERT OR REPLACE INTO ai_assessments VALUES (?, ?, ?)",
                       (mission_id, plan_version, json.dumps(assessment)))

    def review_candidate(self, mission_id: str, plan_version: int, candidate_id: str) -> dict:
        """Record human review only; no availability confirmation or booking occurs."""
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT latest_report_json FROM missions WHERE mission_id=?",
                             (mission_id,)).fetchone()
            if row is None or row["latest_report_json"] is None:
                raise ValueError("mission has no current plan")
            report = json.loads(row["latest_report_json"])
            if report["mission_version"] != plan_version:
                raise ValueError("plan is stale; review the latest version")
            eligible = {c["candidate_id"] for c in report["ranked_recommendation_candidates"]}
            if candidate_id not in eligible:
                raise ValueError("candidate is infeasible or absent from the current plan")
            reviewed_at = datetime.now(timezone.utc).isoformat()
            db.execute("INSERT OR IGNORE INTO reviews VALUES (?, ?, ?, ?)",
                       (mission_id, plan_version, candidate_id, reviewed_at))
            saved = db.execute("SELECT reviewed_at FROM reviews WHERE mission_id=? AND plan_version=? "
                               "AND candidate_id=?", (mission_id, plan_version, candidate_id)).fetchone()
            return {"mission_id": mission_id, "plan_version": plan_version,
                    "candidate_id": candidate_id, "reviewed_at": saved["reviewed_at"],
                    "effect": "review_only_no_booking"}

    def set_stage(self, mission_id: str, stage: TransportStage, location: str,
                  at: datetime | None = None) -> dict:
        """Record a simulated stage event and immediately replan."""
        if not location or not isinstance(stage, TransportStage):
            raise ValueError("stage and location are required")
        event = DisruptionEvent(EventType.STAGE_UPDATED, at or datetime.now(timezone.utc),
                                new_stage=stage, new_location=location)
        return self.handle_event(EventEnvelope(mission_id, uuid4().hex, event))

    def handle_event(self, envelope: EventEnvelope) -> dict:
        if not envelope.event_id:
            raise ValueError("event_id is required for deduplication")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            duplicate = db.execute("SELECT report_json FROM events WHERE mission_id=? AND event_id=?",
                                   (envelope.mission_id, envelope.event_id)).fetchone()
            if duplicate:
                return json.loads(duplicate["report_json"])
            row = db.execute("SELECT state_json, version FROM missions WHERE mission_id=?",
                             (envelope.mission_id,)).fetchone()
            if row is None:
                raise KeyError(f"unknown mission {envelope.mission_id}")
            state = _restore_state(row["state_json"])
            catalog = self.catalog_provider(state, envelope.event.occurred_at)
            airports = discover_departure_airports(state, catalog, envelope.event.occurred_at)
            report = process_disruption(state, envelope.event, catalog,
                                        self.flight_risk_predictor, self.provider_configured)
            report["mission_version"] = row["version"] + 1
            report["departure_airports_considered"] = airports
            report["agent_trace"] = [
                {"tool": "load_mission", "result": "persisted_state"},
                {"tool": "load_transport_catalog", "result": catalog.source},
                {"tool": "discover_departure_airports", "result": [a["airport"] for a in airports]},
                {"tool": "generate_alternative_routes", "result":
                 1 + len(report["alternative_plans"])},
                {"tool": "evaluate_and_rank", "result":
                 len(report["ranked_recommendation_candidates"])}]
            state_json = json.dumps(asdict(state), default=_json_default)
            report_json = json.dumps(report)
            db.execute("UPDATE missions SET state_json=?, version=?, latest_report_json=? WHERE mission_id=?",
                       (state_json, row["version"] + 1, report_json, envelope.mission_id))
            db.execute("INSERT INTO events VALUES (?, ?, ?, ?)",
                       (envelope.mission_id, envelope.event_id,
                        json.dumps(asdict(envelope.event), default=_json_default), report_json))
            return report

    def run_once(self, source: EventSource) -> list[dict]:
        """Poll a source and automatically replan every new event."""
        return [self.handle_event(envelope) for envelope in source.poll()]

    def run_forever(self, source: EventSource, interval_seconds: float = 2,
                    on_report: Callable[[dict], None] | None = None) -> None:
        """Run the read-only event poll and planning loop until interrupted."""
        if interval_seconds <= 0:
            raise ValueError("poll interval must be positive")
        while True:
            for report in self.run_once(source):
                if on_report is not None:
                    on_report(report)
            time.sleep(interval_seconds)
