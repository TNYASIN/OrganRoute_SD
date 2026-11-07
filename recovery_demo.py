"""Explicitly hypothetical mission and transport catalog for offline demonstrations."""

from datetime import datetime, timedelta

from recovery import Flight, GroundOption, MissionState, TransportCatalog, TransportStage

DONOR_SITE = "DEMO_PHOENIX_DONOR_SITE"
RECEIVING_SITE = "DEMO_SAN_DIEGO_HOSPITAL"


def demo_mission(mission_id: str, now: datetime, deadline: datetime,
                 organ_type: str = "Heart") -> MissionState:
    current = Flight("DEMO-CURRENT", "PHX", "SAN", now + timedelta(minutes=90),
                     now + timedelta(minutes=170), observed_at=now,
                     expires_at=now + timedelta(hours=6))
    return MissionState(mission_id, organ_type, DONOR_SITE, RECEIVING_SITE,
                        DONOR_SITE, TransportStage.ORIGIN, deadline, current,
                        current.arrival, simulation_mode=True, destination_airport="SAN")


def demo_catalog(state: MissionState, at: datetime) -> TransportCatalog:
    """Time offsets are scenario fixtures, never published schedules or availability."""
    if (state.origin, state.destination) != (DONOR_SITE, RECEIVING_SITE):
        raise ValueError("this hypothetical catalog only covers the built-in demo mission")
    expires = at + timedelta(hours=6)

    def flight(fid: str, origin: str, dest: str, depart_min: int, arrive_min: int) -> Flight:
        return Flight(fid, origin, dest, at + timedelta(minutes=depart_min),
                      at + timedelta(minutes=arrive_min), observed_at=at, expires_at=expires)

    def ground(origin: str, dest: str, minutes: int) -> GroundOption:
        return GroundOption(origin, dest, minutes, observed_at=at, expires_at=expires)

    return TransportCatalog(
        flights=(flight("DEMO-PHX-SAN", "PHX", "SAN", 120, 200),
                 flight("DEMO-AZA-SAN", "AZA", "SAN", 150, 235),
                 flight("DEMO-PHX-LAX", "PHX", "LAX", 110, 180)),
        ground=(ground(DONOR_SITE, "PHX", 35), ground(DONOR_SITE, "AZA", 55),
                ground("SAN", RECEIVING_SITE, 25), ground("LAX", RECEIVING_SITE, 150),
                ground(DONOR_SITE, RECEIVING_SITE, 390)),
        source="hypothetical_demo",
    )
