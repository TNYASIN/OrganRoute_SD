"""Bridge a scheduled commercial Flight to the teammate's calibrated BTS model.

The result is historical aircraft risk for a hypothetical or supplied schedule.
It is not a live itinerary, delivery probability, or medical assessment.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

from recovery import Flight


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "flight_delay_model_calibrated.pkl"
ROUTES_PATH = ROOT / "data" / "san_dataset_out" / "san_routes.csv"
RISK_THRESHOLD = 0.25  # Demo flag only; never an action or medical threshold.

AIRPORT_ZONES = {
    **dict.fromkeys("ABQ DEN ELP PHX TUS".split(), "America/Phoenix"),
    **dict.fromkeys("LAX LAS SFO SJC OAK SMF FAT MRY SBP STS RNO PDX SEA BOI EUG RDM GEG".split(), "America/Los_Angeles"),
    **dict.fromkeys("DFW DAL HOU IAH AUS SAT BNA MCI STL MSP ORD MDW MKE MSY".split(), "America/Chicago"),
    **dict.fromkeys("ATL BOS BWI CLT CMH DCA DTW EWR FLL IAD IND JAC JFK MCO MIA PHL TPA".split(), "America/New_York"),
    **dict.fromkeys("SLC PVU COS".split(), "America/Denver"),
    **dict.fromkeys("HNL KOA LIH OGG".split(), "Pacific/Honolulu"),
}
# A few airports need their own zone rather than the regional group above.
AIRPORT_ZONES.update({"ABQ": "America/Denver", "DEN": "America/Denver",
                      "ELP": "America/Denver", "LAS": "America/Los_Angeles",
                      "JAC": "America/Denver", "BOI": "America/Boise",
                      "RNO": "America/Los_Angeles", "MSP": "America/Chicago"})


@lru_cache(maxsize=4)
def _load_bundle(path: str):
    import model
    return model._load_model(path)


@lru_cache(maxsize=2)
def _supported_routes(path: str) -> dict[tuple[str, str, str], int]:
    with open(path, newline="", encoding="utf-8") as handle:
        return {(row["carrier"], row["origin"], row["dest"]): int(row["flights"])
                for row in csv.DictReader(handle) if row["low_sample"] == "0"}


class FlightModelAdapter:
    """Callable model adapter; artifacts load once on the first supported call."""

    def __init__(self, model_path: str | Path = MODEL_PATH,
                 routes_path: str | Path = ROUTES_PATH):
        self.model_path = Path(model_path)
        self.routes_path = Path(routes_path)

    @staticmethod
    def _unknown(reason: str, warnings: list[str], supported_route: bool = False) -> dict:
        return {"probability": None, "prediction_type": None,
                "prediction_value": None, "prediction_threshold_minutes": 60,
                "model_name": "calibrated_BTS_flight_outcomes", "model_version": None,
                "supported_route": supported_route, "data_source": "BTS_2025_sampled_months",
                "source": reason, "scope": "commercial_flight_disruption",
                "warnings": warnings}

    def __call__(self, flight: Flight) -> dict:
        carrier = flight.carrier.strip().upper()
        origin = flight.origin.strip().upper()
        dest = flight.destination_airport.strip().upper()
        if not self.model_path.is_file():
            return self._unknown("missing_model_artifact", ["Calibrated model artifact is unavailable."])
        if not self.routes_path.is_file():
            return self._unknown("missing_route_coverage", ["Historical route coverage file is unavailable."])
        try:
            routes = _supported_routes(str(self.routes_path.resolve()))
        except (OSError, ValueError, KeyError) as exc:
            return self._unknown("invalid_route_coverage", [f"Route coverage cannot be read: {exc}"])
        if dest != "SAN" or (carrier, origin, dest) not in routes:
            return self._unknown("unsupported_route", [
                "Carrier, origin, and destination are outside the supported SAN historical routes; risk is unknown."])
        zone_name = AIRPORT_ZONES.get(origin)
        if zone_name is None:
            return self._unknown("unsupported_departure_timezone", [
                "Origin local departure time cannot be mapped to model features."], True)
        try:
            local = flight.departure.astimezone(ZoneInfo(zone_name))
            import model
            bundle = _load_bundle(str(self.model_path.resolve()))
            result = model.predict_risk(origin, carrier, local.hour,
                                        local.isoweekday(), local.month, slack_minutes=60,
                                        dest=dest, loaded_bundle=bundle)
        except (ImportError, OSError, ValueError, RuntimeError, AttributeError, KeyError) as exc:
            return self._unknown("model_inference_error", [f"Model inference failed: {exc}"], True)
        probability = 1.0 - result["probability_arrive_within_slack"]
        warnings = ["Historical commercial flight estimate, not a live schedule or confirmed availability.",
                    "This probability does not include organ handoffs, ground transport, or medical viability."]
        if local.month not in (1, 4, 7, 10):
            warnings.append("Departure month was not sampled in the 2025 training data.")
        if local.year != 2025:
            warnings.append("Departure year differs from the 2025 historical sample.")
        if flight.source == "hypothetical_demo":
            warnings.append("Flight schedule and carrier assignment are hypothetical demo inputs.")
        return {"probability": probability,
                "prediction_type": "flight_cancelled_diverted_or_arrival_delay_over_60_minutes",
                "prediction_value": probability, "prediction_threshold_minutes": 60,
                "model_name": "calibrated_BTS_flight_outcomes", "model_version": bundle["version"],
                "supported_route": True, "historical_route_flights": routes[(carrier, origin, dest)],
                "data_source": "BTS_2025_Jan_Apr_Jul_Oct",
                "source": "trained_flight_model", "scope": "commercial_flight_disruption",
                "probability_cancelled": result["probability_cancelled"],
                "probability_diverted": result["probability_diverted"],
                "probability_delay_over_60_if_completed": result["probability_delay_over_slack_if_completed"],
                "feature_values": {"Origin": origin, "Reporting_Airline": carrier, "Dest": dest,
                                   "departure_hour": local.hour,
                                   "DayOfWeek": local.isoweekday(), "Month": local.month},
                "warnings": warnings}
