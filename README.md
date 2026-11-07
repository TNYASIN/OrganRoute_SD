# OrganRoute SD 🫀

**Tracking tells you an organ is late. OrganRoute tells you what to do about it.**

Decision support for San Diego organ transport (Life Sharing, UC San Diego Health, Scripps, Sharp).

1. **Plan:** organ type + donor airport + cross-clamp time → viability deadline, then compares commercial flights into SAN (delay risk from real BTS data), charter to Montgomery-Gibbs, and ground courier. Gemini recommends one and explains why.
2. **Re-plan:** simulate a delay alert → recompute minutes to spare, re-rank backups. Gemini returns status, recommendation, who to notify, and a drafted message to the team.

> Decision support only, not a medical device. Live UNOS feeds aren't public, so delay events are simulated. Viability windows are approximate.

## Run
```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...        # free key from Google AI Studio (app falls back to rules without it)
python data_prep.py On_Time_*.csv   # optional: builds flights_san.csv from BTS downloads
streamlit run app.py
```

## How it works
- **Delay model:** empirical arrival-delay distribution for each origin × airline × departure hour (falling back to coarser slices when there's too little data). P(on time) = P(arrival delay ≤ slack) × (1 − cancellation rate), where slack = deadline − scheduled arrival at the hospital.
- **Charter / ground:** distance-based estimates with stated assumptions (see constants in `core.py`).
- **Gemini** (`gemini-2.5-flash`, set `GEMINI_MODEL` to change) reasons over the computed options and returns structured JSON.

## AI tools used
- Gemini API: in-app recommendation and re-planning
- Claude: starter code scaffold
- Copilot / Gemini CLI: (fill in what you used)
