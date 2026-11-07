# OrganRoute SD 🫀

**Tracking tells you an organ is late. OrganRoute tells you what to do about it.**

Decision support for San Diego organ transport (Life Sharing, UC San Diego Health, Scripps, Sharp).

1. **Plan:** organ type + donor airport + cross-clamp time → viability deadline, then compares commercial flights into SAN (delay risk from real BTS data), charter to Montgomery-Gibbs, and ground courier. An LLM (Claude by default) recommends one and explains why.
2. **Re-plan:** simulate a delay alert → recompute minutes to spare, re-rank backups. The LLM returns status, recommendation, who to notify, and a drafted message to the team.

> Decision support only, not a medical device. Live UNOS feeds aren't public, so delay events are simulated. Viability windows are approximate.

## Run
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...     # app falls back to rules without a key
# optional: switch provider with one line
# export AI_PROVIDER=openai  (needs OPENAI_API_KEY)   or   AI_PROVIDER=gemini  (needs GEMINI_API_KEY)
# export AI_MODEL=...        override the default model for that provider
python data_prep.py On_Time_*.csv   # optional: builds flights_san.csv from BTS downloads
streamlit run app.py
```

## How it works
- **Delay model:** empirical arrival-delay distribution for each origin × airline × departure hour (falling back to coarser slices when there's too little data). P(on time) = P(arrival delay ≤ slack) × (1 − cancellation rate), where slack = deadline − scheduled arrival at the hospital.
- **Charter / ground:** distance-based estimates with stated assumptions (see constants in `core.py`).
- **LLM planner** (`llm_plan` in `core.py`) reasons over the computed options and returns schema-validated JSON. Set `AI_PROVIDER` to `claude` (default, `claude-opus-5-5`), `openai` (`gpt-5-mini`), or `gemini` (`gemini-2.5-flash`); `AI_MODEL` overrides the model.

## AI tools used
- Claude API (`claude-opus-5-5`): in-app recommendation and re-planning (update this line if you switch `AI_PROVIDER`)
- Claude: starter code scaffold
- Copilot / Gemini CLI: (fill in what you used)
