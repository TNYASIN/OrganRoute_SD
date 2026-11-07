# OrganRoute SD 🫀

**Tracking tells you an organ is late. OrganRoute tells you what to do about it.**

Decision support for San Diego organ transport (Life Sharing, UC San Diego Health, Scripps, Sharp).

1. **Plan:** organ type + donor airport + cross-clamp time → viability deadline, then compares commercial flights into SAN (delay risk from real BTS data), charter to Montgomery-Gibbs, and ground courier. An LLM (Claude by default) recommends one and explains why.
2. **Re-plan:** simulate a delay alert → recompute minutes to spare, re-rank backups. The LLM returns status, recommendation, who to notify, and a drafted message to the team.

> Decision support only, not a medical device. Live UNOS feeds aren't public, so delay events are simulated. Viability windows are approximate.

## Run
```bash
pip install -r requirements.txt
cp .env.example .env         # then paste your ANTHROPIC_API_KEY into .env
                             # (switch provider there with AI_PROVIDER=openai|gemini; app falls back to rules without a key)
python data_prep.py On_Time_*.csv   # optional: builds flights_san.csv from BTS downloads
streamlit run app.py
```

## Recovery agent demo

The **Recovery agent demo** tab is separate from the original planner. Create a
simulated mission, choose a receiving hospital, enter a transport deadline, and apply a delay, cancellation,
diversion, or ETA update. The mission starts at a named **hypothetical donor site**;
the agent examines transfers to both PHX and AZA instead of asking you to pick
the nearest airport. The hospital list and SAN ground-time reference come from
the existing `core.py`; the demo flight and transfer times remain hypothetical.
It stores mission state and plan versions in
`recovery_demo.sqlite3` (ignored by Git). It requires no LLM key; install the
packages in `requirements.txt` for the trained flight model.

The demo catalog contains invented PHX/AZA/SAN/LAX flight and ground scenarios.
They are **not current schedules, bookings, road forecasts, or carrier acceptance**.
The planner marks unconfirmed handoffs and keeps approval with a person. The
teammate's calibrated scikit-learn artifact is included; no live transport
provider is configured. Unsupported flights retain unknown risk.
The **Record review** action stores a human review of the current plan version;
it does not confirm transport availability, book a carrier, or authorize a
medical decision.

### Trained flight model integration

`flight_model_adapter.py` maps each future flight to the six exact model
features: origin, reporting airline, destination, **origin-local** departure
hour, day of week, and month. It reuses `model.py` preprocessing and calibrated
prediction logic, loading the trusted local artifact once per process. The
historical route list is `data/san_dataset_out/san_routes.csv`. Predictions
are restricted to sufficiently represented carrier-origin routes into SAN;
other flights show unknown risk. The source data samples January, April, July,
and October 2025, so other departure months carry an out-of-sample warning.
The artifact was saved with scikit-learn 1.6.1; `requirements.txt` pins that
version.

The prediction is the historical probability that a commercial flight is
cancelled, diverted, or arrives more than **60 minutes** after its scheduled
arrival. It does not estimate organ delivery, clinical viability, or ground
transport reliability. A probability is never added to the delivery ETA.

At mission creation, the app shows the initial simulated flight, scheduled
delivery ETA, deadline margin, exact model target, and a configurable demo risk
flag (default 25%). **Prepare recovery alternatives now** creates a versioned
plan. After a disruption, the agent uses the observed event to update the
current flight ETA and computes each alternative separately. It does not apply
the pre-departure model to a flight whose delay or cancellation is already
observed. Claude's tools receive model provenance, coverage, and warnings;
they cannot change the calculations or invent probabilities. Transport
schedules remain hypothetical and require human verification.

On Windows PowerShell from this repository directory:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -v
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit, select **Recovery agent demo**, create
a mission, and apply a simulated delay or cancellation. `ANTHROPIC_API_KEY` is
optional for deterministic planning; setting it enables the Claude assessment.

### Claude tool agent

Set `ANTHROPIC_API_KEY` in the process environment to enable Claude's real
client-side tool-use loop. The Anthropic SDK is already in `requirements.txt`.
On PowerShell, before starting Streamlit or the worker:

```powershell
$env:ANTHROPIC_API_KEY = "your-key-here"
$env:RECOVERY_CLAUDE_MODEL = "claude-opus-5-5"  # optional override
streamlit run app.py
```

When a new recovery event arrives, Claude reads the saved mission, can inspect
the existing `core.py` reference assumptions, lists and inspects computed
candidates, and submits a structured shortlist. The app validates every selected
candidate ID against the deterministic plan, rejects infeasible or invented
routes, and caches the assessment by plan version. The worker also runs Claude
automatically for new events when the key is set. Without a key, it reports that
Claude is not configured; it does not label the rule plan as an AI assessment.

The recovery workflow currently accepts simulation missions only. User account
authentication is not connected yet; the review button records a demo review,
not an authenticated authorization. Do not put API keys in source files or Git.

Run the deterministic backend tests:

```bash
python -m unittest -v test_recovery.py test_recovery_agent.py test_claude_recovery_agent.py
```

For event-driven operation, start the worker in a second terminal after creating
a mission in the app:

```bash
python recovery_worker.py --database recovery_demo.sqlite3 --events recovery_demo_events.jsonl
```

Append one JSON object per line to `recovery_demo_events.jsonl`, using the mission
ID shown in the app. Example (replace `occurred_at` with a current ISO 8601 time
including a timezone offset):

```json
{"mission_id":"demo-REPLACE","event_id":"delay-001","event":{"type":"FLIGHT_DELAYED","occurred_at":"2030-01-01T12:00:00+00:00","delay_minutes":60}}
```

The worker polls this local file and replans automatically. Repeated event IDs
return the saved plan without applying the delay twice. Click **Refresh recovery
report** in Streamlit to see a worker-produced plan. This file is a simulation
event source, not a live flight feed.

The original two tabs retain their existing code and may call the legacy LLM
planner when a corresponding API key is configured. The recovery tab and worker
use the separate Claude tool agent described above.

## How it works
- **Delay model:** empirical arrival-delay distribution for each origin × airline × departure hour (falling back to coarser slices when there's too little data). P(on time) = P(arrival delay ≤ slack) × (1 − cancellation rate), where slack = deadline − scheduled arrival at the hospital.
- **Charter / ground:** distance-based estimates with stated assumptions (see constants in `core.py`).
- **LLM planner** (`llm_plan` in `core.py`) reasons over the computed options and returns schema-validated JSON. Set `AI_PROVIDER` to `claude` (default, `claude-opus-5-5`), `openai` (`gpt-5-mini`), or `gemini` (`gemini-2.5-flash`); `AI_MODEL` overrides the model.

## AI tools used
- Claude API (`claude-opus-5-5`): in-app recommendation and re-planning (update this line if you switch `AI_PROVIDER`)
- Claude: starter code scaffold
- Copilot / Gemini CLI: (fill in what you used)
