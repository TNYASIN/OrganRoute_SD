"""Stateful, deterministic organ transport recovery planning backend.

All transport options supplied here are hypothetical. A caller must supply and
verify actual schedules, acceptance, and handoffs before acting on a plan.
This module does not book transport or make medical decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable


class TransportStage(str, Enum):
    ORIGIN = "AT_ORIGIN"
    AIRPORT = "AT_AIRPORT"
    ONBOARD = "ONBOARD_AIRCRAFT"
    TRANSFER = "AT_TRANSFER_AIRPORT"
    GROUND = "IN_GROUND_TRANSPORT"


class EventType(str, Enum):
    FLIGHT_DELAYED = "FLIGHT_DELAYED"
    FLIGHT_CANCELLED = "FLIGHT_CANCELLED"
    FLIGHT_DIVERTED = "FLIGHT_DIVERTED"
    ETA_UPDATED = "ETA_UPDATED"
    STAGE_UPDATED = "STAGE_UPDATED"


def _aware(value: datetime, name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")


@dataclass(frozen=True)
class Flight:
    flight_id: str
    origin: str
    destination_airport: str
    departure: datetime
    arrival: datetime
    carrier: str = "unverified carrier"
    source: str = "hypothetical_demo"
    verified: bool = False
    observed_at: datetime | None = None
    expires_at: datetime | None = None

    def __post_init__(self) -> None:
        _aware(self.departure, "flight departure")
        _aware(self.arrival, "flight arrival")
        if self.arrival <= self.departure:
            raise ValueError("flight arrival must follow departure")
        if not self.flight_id or not self.origin or not self.destination_airport:
            raise ValueError("flight identity and airports are required")
        if self.observed_at is not None:
            _aware(self.observed_at, "flight observation time")
        if self.expires_at is not None:
            _aware(self.expires_at, "flight expiry time")


@dataclass(frozen=True)
class GroundOption:
    origin: str
    destination: str
    duration_minutes: int
    source: str = "hypothetical_demo"
    verified: bool = False
    observed_at: datetime | None = None
    expires_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.duration_minutes <= 0:
            raise ValueError("ground duration must be positive")
        if self.observed_at is not None:
            _aware(self.observed_at, "ground observation time")
        if self.expires_at is not None:
            _aware(self.expires_at, "ground expiry time")


@dataclass(frozen=True)
class TransportCatalog:
    flights: tuple[Flight, ...] = ()
    ground: tuple[GroundOption, ...] = ()
    source: str = "hypothetical_demo"


@dataclass(frozen=True)
class DisruptionEvent:
    type: EventType
    occurred_at: datetime
    updated_eta: datetime | None = None
    delay_minutes: int | None = None
    diversion_airport: str | None = None
    new_stage: TransportStage | None = None
    new_location: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.type, EventType):
            raise ValueError("unsupported event type")
        _aware(self.occurred_at, "event time")
        if self.updated_eta is not None:
            _aware(self.updated_eta, "updated ETA")
            if self.updated_eta < self.occurred_at:
                raise ValueError("updated ETA cannot precede the event")
        if self.type == EventType.ETA_UPDATED and self.updated_eta is None:
            raise ValueError("ETA_UPDATED requires updated_eta")
        if self.type == EventType.FLIGHT_DELAYED and self.delay_minutes is None and self.updated_eta is None:
            raise ValueError("FLIGHT_DELAYED requires delay_minutes or updated_eta")
        if self.delay_minutes is not None and self.delay_minutes < 0:
            raise ValueError("delay_minutes cannot be negative")
        if self.type == EventType.FLIGHT_DIVERTED and not self.diversion_airport:
            raise ValueError("FLIGHT_DIVERTED requires diversion_airport")
        if self.type == EventType.STAGE_UPDATED and (
                not isinstance(self.new_stage, TransportStage) or not self.new_location):
            raise ValueError("STAGE_UPDATED requires new_stage and new_location")


@dataclass
class MissionState:
    mission_id: str
    organ_type: str
    origin: str
    destination: str
    current_location: str
    stage: TransportStage
    deadline: datetime
    current_flight: Flight | None
    current_expected_arrival: datetime | None
    handling_minutes: int = 30
    ground_transfer_minutes: int = 30
    simulation_mode: bool = False
    event_history: list[dict] = field(default_factory=list)
    current_flight_cancelled: bool = False
    diverted_to: str | None = None
    destination_airport: str | None = None

    def __post_init__(self) -> None:
        _aware(self.deadline, "deadline")
        if self.current_expected_arrival is not None:
            _aware(self.current_expected_arrival, "current expected arrival")
        if self.handling_minutes < 0 or self.ground_transfer_minutes < 0:
            raise ValueError("handling times cannot be negative")
        if not all((self.mission_id, self.organ_type, self.origin, self.destination, self.current_location)):
            raise ValueError("mission identity, organ, origin, destination, and location are required")
        if self.stage == TransportStage.ONBOARD and self.current_flight is None:
            raise ValueError("onboard mission requires a current flight")


FlightRiskPredictor = Callable[[Flight], float | dict | None]


def calculate_remaining_time(state: MissionState, at: datetime) -> int:
    _aware(at, "evaluation time")
    return int((state.deadline - at).total_seconds() // 60)


def predict_flight_disruption(flight: Flight, predictor: FlightRiskPredictor | None = None) -> dict:
    """Only commercial flight risk. Inject an existing trained model adapter if available."""
    if predictor is None:
        return {"probability": None, "source": "unavailable", "scope": "commercial_flight_disruption"}
    try:
        probability = predictor(flight)
    except (KeyError, ValueError):
        return {"probability": None, "source": "unavailable_unsupported_flight_data",
                "scope": "commercial_flight_disruption"}
    if probability is None:
        return {"probability": None, "source": "unavailable", "scope": "commercial_flight_disruption"}
    if isinstance(probability, dict):
        value = probability.get("probability")
        if value is not None and not 0 <= value <= 1:
            raise ValueError("flight risk predictor must return a probability in [0, 1]")
        return probability
    if not 0 <= probability <= 1:
        raise ValueError("flight risk predictor must return a probability in [0, 1]")
    return {"probability": float(probability), "source": "injected_flight_model", "scope": "commercial_flight_disruption"}


def hypothetical_demo_catalog(at: datetime, origin: str = "PHX",
                              destination: str = "SAN") -> TransportCatalog:
    """Predefined relative-time scenario; every option is explicitly hypothetical."""
    _aware(at, "scenario time")
    if (origin, destination) != ("PHX", "SAN"):
        raise ValueError("the predefined demo scenario only covers PHX to SAN")
    return TransportCatalog(
        flights=(
            Flight("DEMO-DIRECT-1", "PHX", "SAN", at + timedelta(hours=2),
                   at + timedelta(hours=3, minutes=20)),
            Flight("DEMO-NEARBY-1", "PHX", "LAX", at + timedelta(hours=1, minutes=40),
                   at + timedelta(hours=3)),
        ),
        ground=(GroundOption("LAX", "SAN", 150), GroundOption("PHX", "SAN", 390)),
    )


def _step(name: str, start: datetime, end: datetime, location: str) -> dict:
    return {"step": name, "start": start.isoformat(), "end": end.isoformat(), "location": location}


def generate_alternative_routes(state: MissionState, event: DisruptionEvent,
                                catalog: TransportCatalog) -> list[dict]:
    """Generate only routes that are represented in the supplied catalog."""
    routes: list[dict] = []
    now = event.occurred_at
    onboard = state.stage == TransportStage.ONBOARD
    flight = state.current_flight

    def access_to(airport: str) -> tuple[GroundOption | None, bool]:
        if state.stage == TransportStage.ONBOARD and flight is not None and airport == flight.origin:
            return None, True
        if airport == state.current_location:
            return None, True
        if state.stage == TransportStage.ORIGIN:
            access = next((g for g in catalog.ground
                           if g.origin == state.current_location and g.destination == airport), None)
            return access, access is not None
        return None, False

    if flight is not None:
        access, accessible = access_to(flight.origin)
        arrival_ground = next((g for g in catalog.ground
                               if g.origin == flight.destination_airport and g.destination == state.destination), None)
        routes.append({"strategy": "CONTINUE_CURRENT_FLIGHT", "flight": flight,
                       "ground": arrival_ground, "access_ground": access, "access_missing": not accessible,
                       "requires_switch": False})
    if not onboard:
        for alternative in catalog.flights:
            if alternative.flight_id == (flight.flight_id if flight else None):
                continue
            access, accessible = access_to(alternative.origin)
            if not accessible:
                continue
            ready = now + timedelta(minutes=state.handling_minutes +
                                    (access.duration_minutes if access else 0))
            if alternative.departure < ready:
                continue
            ground = next((g for g in catalog.ground
                           if g.origin == alternative.destination_airport and g.destination == state.destination), None)
            if alternative.destination_airport != state.destination and ground is None:
                continue
            strategy = ("ALTERNATIVE_DIRECT_FLIGHT" if alternative.destination_airport in
                        (state.destination, state.destination_airport)
                        else "NEARBY_AIRPORT_GROUND_TRANSFER")
            routes.append({"strategy": strategy, "flight": alternative, "ground": ground,
                           "access_ground": access, "access_missing": False,
                           "requires_switch": True})
        for ground in catalog.ground:
            if ground.origin == state.current_location and ground.destination == state.destination:
                routes.append({"strategy": "DIRECT_GROUND", "flight": None, "ground": ground,
                               "access_ground": None, "access_missing": False,
                               "requires_switch": True})
    routes.append({"strategy": "SPECIALIZED_TRANSPORT_ESCALATION", "flight": None,
                   "ground": None, "access_ground": None, "access_missing": False,
                   "requires_switch": True})
    return routes


def estimate_route_arrival(state: MissionState, event: DisruptionEvent, route: dict) -> dict:
    """Return a traceable airport-to-destination timeline; unknown legs have no ETA."""
    now = event.occurred_at
    strategy = route["strategy"]
    flight: Flight | None = route["flight"]
    ground: GroundOption | None = route["ground"]
    access: GroundOption | None = route.get("access_ground")
    timeline: list[dict] = []
    if strategy == "SPECIALIZED_TRANSPORT_ESCALATION":
        return {"estimated_delivery": None, "timeline": timeline}
    if flight is not None:
        if route.get("access_missing"):
            return {"estimated_delivery": None, "timeline": timeline}
        if access is not None:
            leave = now + timedelta(minutes=state.handling_minutes)
            airport_ready = leave + timedelta(minutes=access.duration_minutes)
            timeline.extend([_step("origin_handling", now, leave, state.current_location),
                             _step("departure_airport_transfer", leave, airport_ready, flight.origin)])
            if airport_ready > flight.departure:
                return {"estimated_delivery": None, "timeline": timeline}
        if strategy == "CONTINUE_CURRENT_FLIGHT":
            if state.current_flight_cancelled or state.diverted_to:
                return {"estimated_delivery": None, "timeline": timeline}
            arrival = state.current_expected_arrival
            if arrival is None or arrival < now:
                return {"estimated_delivery": None, "timeline": timeline}
            start = max(now, flight.departure)
        else:
            start = flight.departure
            arrival = flight.arrival
        if arrival < start:
            return {"estimated_delivery": None, "timeline": timeline}
        timeline.append(_step("flight", start, arrival, flight.destination_airport))
        handling_end = arrival + timedelta(minutes=state.handling_minutes)
        timeline.append(_step("arrival_handling", arrival, handling_end, flight.destination_airport))
        if ground is not None:
            delivered = handling_end + timedelta(minutes=ground.duration_minutes)
            timeline.append(_step("ground_transfer", handling_end, delivered, state.destination))
        elif flight.destination_airport == state.destination:
            delivered = handling_end + timedelta(minutes=state.ground_transfer_minutes)
            timeline.append(_step("destination_ground_transfer", handling_end, delivered, state.destination))
        else:
            return {"estimated_delivery": None, "timeline": timeline}
    elif ground is not None:
        start = now + timedelta(minutes=state.handling_minutes)
        delivered = start + timedelta(minutes=ground.duration_minutes)
        timeline.extend([_step("origin_handling", now, start, state.current_location),
                         _step("direct_ground", start, delivered, state.destination)])
    else:
        return {"estimated_delivery": None, "timeline": timeline}
    return {"estimated_delivery": delivered, "timeline": timeline}


def evaluate_route_feasibility(state: MissionState, event: DisruptionEvent, route: dict,
                               estimate: dict) -> dict:
    strategy = route["strategy"]
    flight: Flight | None = route["flight"]
    ground: GroundOption | None = route["ground"]
    access: GroundOption | None = route.get("access_ground")
    requirements: list[str] = []
    reasons: list[str] = []
    executable = True
    if state.stage == TransportStage.ONBOARD and route["requires_switch"]:
        executable = False
        reasons.append("Organ is already onboard; transfer cannot start from current stage")
    if strategy == "CONTINUE_CURRENT_FLIGHT" and (state.current_flight_cancelled or state.diverted_to):
        executable = False
        reasons.append("Original flight cannot complete the planned route")
    if route.get("access_missing"):
        executable = False
        reasons.append("No documented transfer from current location to departure airport")
    if strategy == "SPECIALIZED_TRANSPORT_ESCALATION":
        requirements.append("Specialized transport dispatch, route, capacity, and ETA must be confirmed")
    if flight is not None:
        if not flight.verified:
            requirements.extend([f"Verify {flight.flight_id} schedule and availability",
                                 f"Confirm {flight.carrier} accepts this organ shipment"])
        requirements.append("Confirm airport handoff and custody")
        if flight.expires_at is not None and flight.expires_at < event.occurred_at:
            requirements.append(f"Refresh stale flight information for {flight.flight_id}")
    if access is not None:
        if not access.verified:
            requirements.append(f"Confirm departure transfer from {access.origin} to {access.destination}")
        if access.expires_at is not None and access.expires_at < event.occurred_at:
            requirements.append("Refresh stale departure transfer information")
    if ground is not None:
        if not ground.verified:
            requirements.append(f"Confirm ground carrier and travel time from {ground.origin}")
        requirements.append("Confirm ground handoff and custody")
        if ground.expires_at is not None and ground.expires_at < event.occurred_at:
            requirements.append("Refresh stale ground information")
    if flight is not None and ground is None:
        requirements.append("Confirm destination ground transfer and travel time")
    eta = estimate["estimated_delivery"]
    margin = int((state.deadline - eta).total_seconds() // 60) if eta is not None else None
    if eta is None:
        executable = False
        reasons.append("Delivery ETA cannot be established")
    elif margin < 0:
        executable = False
        reasons.append("Estimated delivery misses the specified deadline")
    status = "INFEASIBLE" if not executable else ("CONDITIONAL" if requirements else "FEASIBLE")
    return {"status": status, "executable_from_stage": executable,
            "deadline_margin_minutes": margin, "reasons": reasons,
            "verification_requirements": requirements}


def rank_recovery_options(options: list[dict]) -> list[dict]:
    """Feasibility first; compare known flight risk without treating unknown as zero."""
    def key(option: dict) -> tuple:
        risk = option["flight_disruption_risk"]["probability"]
        margin = option["feasibility"]["deadline_margin_minutes"]
        return (option["feasibility"]["status"] == "INFEASIBLE",
                len(option["feasibility"]["verification_requirements"]),
                risk is None and option["flight"] is not None,
                risk if risk is not None else 1,
                -(margin if margin is not None else -10**9),
                option["handoff_count"])
    ranked = sorted(options, key=key)
    for index, option in enumerate(ranked, 1):
        option["rank"] = index
    return ranked


def _ground_json(ground: GroundOption) -> dict:
    return {**asdict(ground),
            "observed_at": ground.observed_at.isoformat() if ground.observed_at else None,
            "expires_at": ground.expires_at.isoformat() if ground.expires_at else None}


def process_disruption(state: MissionState, event: DisruptionEvent,
                       catalog: TransportCatalog | None = None,
                       flight_risk_predictor: FlightRiskPredictor | None = None,
                       provider_configured: bool = False) -> dict:
    """Apply an event to mission state and return a JSON-serializable recovery report."""
    if not state.simulation_mode:
        raise ValueError("Disruption simulation requires mission.simulation_mode=True")
    catalog = catalog or TransportCatalog()
    sources = [catalog.source, *(f.source for f in catalog.flights),
               *(g.source for g in catalog.ground)]
    if state.current_flight:
        sources.append(state.current_flight.source)
    if not provider_configured and any(source != "hypothetical_demo" for source in sources):
        raise ValueError("non-demo transport data requires an explicitly configured provider")
    if state.event_history and event.occurred_at < datetime.fromisoformat(state.event_history[-1]["occurred_at"]):
        raise ValueError("events must be applied in chronological order")
    if state.current_flight is None:
        raise ValueError("a disruption event requires a current flight")
    if event.type == EventType.FLIGHT_DIVERTED and state.stage not in (
            TransportStage.ONBOARD, TransportStage.TRANSFER):
        raise ValueError("flight diversion requires an onboard or transfer-stage mission")
    if event.type == EventType.FLIGHT_DELAYED:
        if state.current_flight_cancelled:
            raise ValueError("cannot delay a cancelled flight")
        if event.updated_eta is None and state.current_expected_arrival is None:
            raise ValueError("cannot apply relative delay without a current ETA")
        state.current_expected_arrival = (event.updated_eta if event.updated_eta is not None else
                                          state.current_expected_arrival + timedelta(minutes=event.delay_minutes))
    elif event.type == EventType.ETA_UPDATED:
        state.current_expected_arrival = event.updated_eta
    elif event.type == EventType.FLIGHT_CANCELLED:
        if state.stage == TransportStage.ONBOARD:
            raise ValueError("an onboard flight cannot be cancelled; use FLIGHT_DIVERTED")
        state.current_flight_cancelled = True
        state.current_expected_arrival = None
    elif event.type == EventType.FLIGHT_DIVERTED:
        state.diverted_to = event.diversion_airport
        state.current_expected_arrival = event.updated_eta
        if state.stage == TransportStage.TRANSFER:
            state.current_location = event.diversion_airport
    elif event.type == EventType.STAGE_UPDATED:
        if event.new_stage == TransportStage.ONBOARD and state.current_flight_cancelled:
            raise ValueError("cannot board a cancelled flight")
        if event.new_stage == TransportStage.TRANSFER and state.diverted_to != event.new_location:
            raise ValueError("transfer location must match the diversion airport")
        state.stage = event.new_stage
        state.current_location = event.new_location
    else:
        raise ValueError("unsupported event type")
    state.event_history.append({"type": event.type.value, "occurred_at": event.occurred_at.isoformat(),
                                "updated_eta": event.updated_eta.isoformat() if event.updated_eta else None,
                                "delay_minutes": event.delay_minutes,
                                "diversion_airport": event.diversion_airport,
                                "new_stage": event.new_stage.value if event.new_stage else None,
                                "new_location": event.new_location})
    options = []
    for i, route in enumerate(generate_alternative_routes(state, event, catalog), 1):
        estimate = estimate_route_arrival(state, event, route)
        feasibility = evaluate_route_feasibility(state, event, route, estimate)
        flight = route["flight"]
        if flight is None:
            risk = {"probability": None, "source": "not_applicable",
                    "scope": "commercial_flight_disruption"}
        elif route["strategy"] == "CONTINUE_CURRENT_FLIGHT":
            risk = {"probability": None, "source": "observed_event_not_modelled",
                    "scope": "commercial_flight_disruption",
                    "warnings": ["The pre-departure model does not condition on the observed flight event."]}
        else:
            risk = predict_flight_disruption(flight, flight_risk_predictor)
        options.append({"candidate_id": f"{state.mission_id}-{len(state.event_history)}-{i}",
                        "strategy": route["strategy"], "flight": ({**asdict(flight),
                            "departure": flight.departure.isoformat(), "arrival": flight.arrival.isoformat(),
                            "observed_at": flight.observed_at.isoformat() if flight.observed_at else None,
                            "expires_at": flight.expires_at.isoformat() if flight.expires_at else None}
                            if flight else None),
                        "ground": _ground_json(route["ground"]) if route["ground"] else None,
                        "access_ground": _ground_json(route["access_ground"]) if route["access_ground"] else None,
                        "estimated_delivery": (estimate["estimated_delivery"].isoformat()
                                               if estimate["estimated_delivery"] else None),
                        "timeline": estimate["timeline"], "flight_disruption_risk": risk,
                        "handoff_count": (1 if flight else 0) + (1 if route["ground"] else 0)
                                         + (1 if route["access_ground"] else 0),
                        "feasibility": feasibility})
    ranked = rank_recovery_options(options)
    original = next((o for o in ranked if o["strategy"] == "CONTINUE_CURRENT_FLIGHT"), None)
    viable = [o for o in ranked if o["feasibility"]["status"] != "INFEASIBLE"]
    recommendations = [{"rank": o["rank"], "candidate_id": o["candidate_id"],
                        "strategy": o["strategy"], "feasibility_status": o["feasibility"]["status"],
                        "deadline_margin_minutes": o["feasibility"]["deadline_margin_minutes"],
                        "flight_disruption_risk": o["flight_disruption_risk"],
                        "verification_requirements": o["feasibility"]["verification_requirements"]}
                       for o in viable]
    return {"mission_id": state.mission_id, "simulation_mode": True,
            "data_source": catalog.source, "event": state.event_history[-1],
            "ranking_basis": "Feasibility, unverified logistics count, known commercial flight risk, deadline margin, then handoffs. Unknown flight risk remains unknown; ground transport has no flight-model estimate.",
            "updated_mission_status": {"organ_type": state.organ_type, "origin": state.origin,
                                       "destination": state.destination, "current_location": state.current_location,
                                       "transport_stage": state.stage.value, "deadline": state.deadline.isoformat(),
                                       "remaining_minutes": calculate_remaining_time(state, event.occurred_at),
                                       "current_flight": ({**asdict(state.current_flight),
                                           "departure": state.current_flight.departure.isoformat(),
                                           "arrival": state.current_flight.arrival.isoformat(),
                                           "observed_at": state.current_flight.observed_at.isoformat()
                                           if state.current_flight.observed_at else None,
                                           "expires_at": state.current_flight.expires_at.isoformat()
                                           if state.current_flight.expires_at else None}),
                                       "current_expected_arrival": (state.current_expected_arrival.isoformat()
                                                                    if state.current_expected_arrival else None),
                                       "handling_minutes": state.handling_minutes,
                                       "ground_transfer_minutes": state.ground_transfer_minutes,
                                       "diverted_to": state.diverted_to,
                                       "destination_airport": state.destination_airport,
                                       "event_count": len(state.event_history)},
            "original_plan_assessment": original, "alternative_plans": [o for o in ranked if o is not original],
            "ranked_recommendation_candidates": recommendations,
            "human_approval_required": True,
            "uncertainty": "Hypothetical transport options are unconfirmed; no booking or medical decision is made."}
