"""Poll an explicit local simulation event file and re-plan missions automatically."""

import argparse
import os

from claude_recovery_agent import ClaudeRecoveryAgent
from recovery_agent import JsonlEventSource, RecoveryAgent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default="recovery_demo.sqlite3")
    parser.add_argument("--events", default="recovery_demo_events.jsonl")
    parser.add_argument("--interval", type=float, default=2)
    args = parser.parse_args()
    print(f"Watching {args.events}; storing plans in {args.database}. Press Ctrl+C to stop.")
    recovery = RecoveryAgent(args.database)
    claude = ClaudeRecoveryAgent(recovery) if os.getenv("ANTHROPIC_API_KEY") else None
    failed_versions = set()

    def on_report(report):
        if claude is None:
            return
        key = (report["mission_id"], report["mission_version"])
        if key in failed_versions:
            return
        try:
            claude.analyze(report["mission_id"])
        except Exception as exc:
            failed_versions.add(key)
            print(f"Claude assessment failed for {key}: {exc}")

    if claude is None:
        print("ANTHROPIC_API_KEY is not set; deterministic plans will run without Claude.")
    try:
        recovery.run_forever(JsonlEventSource(args.events), args.interval, on_report=on_report)
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()
