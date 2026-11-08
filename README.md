# OrganRoute SD 🫀

(Project for AIC at UC San Diego Hack for Humanity)

**Tracking tells you an organ is late. OrganRoute tells you what to do about it.**

A donor heart survives about six hours on ice. When its flight is delayed, a transplant coordinator has minutes to pick a backup. OrganRoute SD is decision support for San Diego organ transport (Life Sharing, UC San Diego Health, Scripps, Sharp): it ranks transport options against the organ's viability deadline, watches for delays, re-plans, and alerts the team.

> **Decision support only, not a medical device.** A human makes every decision. Viability windows are approximate, flight schedules are rebuilt from 2025 history (not tonight's live schedule), and charter reliability is a stated assumption.

## What it does

| Tab | What happens |
|---|---|
| **1 · Plan transport** | Enter organ, donor airport, receiving hospital, and cross-clamp time. The app computes the deadline and ranks commercial flights into SAN, a charter to Montgomery-Gibbs, and a ground courier by minutes to spare and on-time chance. Claude recommends one, explains why, lists who to notify, and drafts the team message. |
| **2 · Delay alert → re-plan** | Simulate a delay, or **live-watch** the FAA airport status feed (and the booked flight on FlightAware, if configured). A new delay recomputes the margin, re-ranks backups (another flight, a backup charter, ground), and **alerts the team's phones**. |
| **3 · Recovery agent demo** | A simulation of event-driven recovery: missions, versioned plans, and a Claude tool-use agent that can only choose among computed candidates. See [Recovery agent](#recovery-agent-demo). |

## Run

```bash
pip install -r requirements.txt   # pins scikit-learn 1.6.1 for the saved model
cp .env.example .env              # add ANTHROPIC_API_KEY (everything else is optional)
streamlit run app.py
```

On macOS with several Pythons installed, run pip and streamlit with the **same** interpreter, e.g. `python3 -m pip install -r requirements.txt` then `python3 -m streamlit run app.py`.

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Restart the app after editing `.env`; it is read at startup.

### Configuration (`.env`)

| Key | Needed for | Default / notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | Claude recommendations | Without any key, a rule-based plan is used |
| `AI_PROVIDER` | Choose the LLM | `claude` (default), `openai`, `gemini` |
| `AI_MODEL` | Override the provider's model | `claude-opus-5-5`, `gpt-5-mini`, `gemini-3.8-flash` |
| `USE_ML_MODEL` | Turn the ML model off | `1` (on); `0` falls back to historical frequencies |
| `NTFY_TOPIC` | Phone push alerts (free) | Install the ntfy app and subscribe to the same, hard-to-guess topic |
| `AEROAPI_KEY` | Live status of one booked flight (FlightAware) | Optional; capped by `AEROAPI_MONTHLY_CAP` (default 300 calls ≈ $1.50) |
| `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM`, `ALERT_TO` | SMS / WhatsApp alerts | Optional; free Twilio trials cannot send custom messages |

## How it works

### Data
- **Flights:** U.S. DOT **Bureau of Transportation Statistics** on-time records, cleaned by `data/build_san_dataset.py` into `data/san_dataset_out/` (~333k flights; **32,612 into SAN**; January, April, July and October 2025).
- **Schedule:** `_schedule()` in `core.py` rebuilds recurring departures: an (airline, departure time) slot must recur ≥ 10 times, and same-airline slots within 15 minutes are merged. The planner lists the next four departures the organ can make (45 min to the airport + 45 min check-in).
- **Viability windows** (`VIABILITY_HOURS`): heart 6 h, lung 8 h, liver 12 h, pancreas 18 h, kidney 36 h, minus 1 h of preparation at the receiving hospital. Machine perfusion is not modeled.
- All times are San Diego time, whatever timezone the computer or server uses (`core.sd_now()`).

### On-time chance
- **Commercial: calibrated ML model** (`model.py`, `calibrate_model.py`): gradient boosting on origin, airline, destination, departure hour, day of week and month. For each option it predicts the chance the flight is completed and arrives within that option's exact minutes to spare (plus cancellation and diversion risk separately). Wired in through `p_on_time()` in `core.py`; the basis column reads `ML model (calibrated)`.
- **Fallback: historical frequencies.** When the model is unavailable, or a delay is already known, the app uses the share of comparable past flights (same route, airline, ±1 hour) that arrived within the slack, times (1 − cancellation rate). Thin slices (< 30 flights) widen to the route, then all SAN arrivals.
- **Charter:** an assumed 95% base, adjusted by **live aviation forecasts** (TAFs from aviationweather.gov) at departure and arrival. Each airport's FAA flight category applies a stated rule: MVFR −3% / +10 min, IFR −10% / +25 min, thunderstorms −20% / +30 min, LIFR (fog) −25% / +45 min. Montgomery-Gibbs has no forecast, so San Diego International's is used. No public charter delay data exists, so these are labeled assumptions. Enter a real booking (wheels-up, aircraft, arrival airport) or use the estimate; "simulate San Diego weather" shows the effect.
- **Ground:** road distance (1.25 × straight line) at 55 mph, within 400 road miles.

**Ranking:** options ≥ 90% likely first, then cheapest (commercial `$`, ground `$$`, charter `$$$$`: relative tiers, not prices), then most likely.

### Model accuracy
Trained on earlier dates and scored on later held-out dates (`calibration_results_san.json`):

| Target | ROC AUC | Brier | Predicted rate | Actual rate |
|---|---|---|---|---|
| Arrives > 15 min late | 0.651 | 0.187 | 23.9% | 26.8% |
| Arrives > 45 min late | 0.640 | 0.089 | 10.2% | 10.1% |
| Arrives > 60 min late | 0.644 | 0.061 | 7.6% | 6.6% |
| Arrives > 120 min late | 0.661 | 0.020 | 2.7% | 2.0% |
| Cancelled | 0.590 | 0.009 | 0.9% | 1.0% |

The probabilities are well calibrated (predicted rates track actual rates), while ranking individual flights is modest (AUC ≈ 0.65): from schedule features alone, delays are hard to predict. The training data covers four months, so other months are out of sample.

### Recommendations and fallbacks
- **LLM planner** (`llm_plan` in `core.py`): Claude receives only numbers the code computed and must answer in a fixed JSON schema (status, recommendation, reasoning, who to notify, message draft). It never calculates times or probabilities.
- **Rule-based fallback** (`rule_plan`): with no key, a provider error, or a refusal, the top-ranked option becomes the plan and status follows fixed thresholds (red: misses the deadline or < 60%; yellow: < 60 min to spare or < 90%; green otherwise). The card shows "Plan source: rule-based fallback" and the reason.

### Live watch and alerts
- `live.py` polls the **FAA NAS airport status** feed (free, no key) every 60 s while watching: ground stops and delay programs at SAN, departure delays at the origin. A new delay ≥ 15 min triggers the re-plan once per event.
- With `AEROAPI_KEY`, the exact booked flight number is checked on **FlightAware** (only while watching, at most once per 2 minutes per flight, monthly cap).
- `notify.py` sends the plan through every configured channel: **ntfy** push (recommended) and/or Twilio. The 📲 button sends manually; a toggle sends automatically on live or simulated delay alerts. The ntfy topic is never shown in the UI. Never include patient-identifying details in alerts.

## Recovery agent demo

The **Recovery agent demo** tab is separate from the planner. Create a simulated mission, choose a receiving hospital, enter a transport deadline, and apply a delay, cancellation, diversion, or ETA update. The mission starts at a named **hypothetical donor site**; the agent examines transfers to both PHX and AZA instead of asking you to pick the nearest airport. The hospital list and SAN ground-time reference come from `core.py`; the demo flight and transfer times remain hypothetical. Mission state and plan versions are stored in `recovery_demo.sqlite3` (ignored by Git). No LLM key is required.

The demo catalog contains invented PHX/AZA/SAN/LAX flight and ground scenarios. They are **not current schedules, bookings, road forecasts, or carrier acceptance**. The planner marks unconfirmed handoffs and keeps approval with a person. **Record review** stores a human review of the current plan version; it does not confirm transport availability, book a carrier, or authorize a medical decision.

### Trained flight model in the recovery agent
`flight_model_adapter.py` maps each future flight to the model's six features (origin, reporting airline, destination, **origin-local** departure hour, day of week, month), reusing `model.py` preprocessing and calibrated prediction and loading the artifact once per process. The route list is `data/san_dataset_out/san_routes.csv`. Predictions are restricted to sufficiently represented carrier-origin routes into SAN; other flights show unknown risk, and months outside the sample carry an out-of-sample warning.

Here the prediction is the historical probability that a commercial flight is cancelled, diverted, or arrives more than **60 minutes** late. It does not estimate organ delivery, clinical viability, or ground reliability, and a probability is never added to the delivery ETA. After a disruption, the agent uses the observed event to update the flight ETA and does not apply the pre-departure model to a flight whose delay or cancellation is already observed.

### Claude tool agent
With `ANTHROPIC_API_KEY` set (optionally `RECOVERY_CLAUDE_MODEL`, default `claude-opus-5-5`), Claude runs a client-side tool-use loop: it reads the saved mission, can inspect `core.py` reference assumptions, lists and inspects computed candidates, and submits a structured shortlist. The app validates every selected candidate ID against the deterministic plan, rejects infeasible or invented routes, and caches the assessment by plan version. Without a key it reports that Claude is not configured and does not label the rule plan as an AI assessment. The recovery workflow accepts simulation missions only; user authentication is not connected, so reviews are demo reviews.

### Event-driven worker
Start the worker in a second terminal after creating a mission:

```bash
python recovery_worker.py --database recovery_demo.sqlite3 --events recovery_demo_events.jsonl
```

Append one JSON object per line to `recovery_demo_events.jsonl` using the mission ID shown in the app (`occurred_at` must be ISO 8601 with a timezone offset):

```json
{"mission_id":"demo-REPLACE","event_id":"delay-001","event":{"type":"FLIGHT_DELAYED","occurred_at":"2030-01-01T12:00:00+00:00","delay_minutes":60}}
```

The worker polls the file and replans automatically; repeated event IDs return the saved plan without applying the delay twice. Click **Refresh recovery report** to see a worker-produced plan. This file is a simulation event source, not a live flight feed.

## Tests

```bash
python -m pytest -q
# or: python -m unittest -v test_recovery.py test_recovery_agent.py test_claude_recovery_agent.py test_flight_model_adapter.py
```

## Limitations and next steps
- **Schedules are historical:** planning lists flights that recurred in 2025, not tonight's confirmed flights; next step is a live schedule feed (FlightAware support is already wired for tracking).
- **Charter and ground** use stated assumptions; operator data would allow real models.
- **Model:** modest discrimination (AUC ≈ 0.65), four months of training data; no live weather in the commercial model yet (BTS weather-delay data plus historical weather is the next feature).
- **Clinical:** windows are approximate; machine perfusion, organ-specific logistics, and UNOS/OPO systems are not integrated.

## AI tools used
- **Claude API** (`claude-opus-5-5`): in-app recommendations and re-planning (tabs 1–2) and the recovery tool-use agent (tab 3). Update if you switch `AI_PROVIDER`.
- **Claude Code**: development assistance.
