"""Poll an explicit local simulation event file and re-plan missions automatically."""

import argparse

from recovery_agent import JsonlEventSource, RecoveryAgent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default="recovery_demo.sqlite3")
    parser.add_argument("--events", default="recovery_demo_events.jsonl")
    parser.add_argument("--interval", type=float, default=2)
    args = parser.parse_args()
    print(f"Watching {args.events}; storing plans in {args.database}. Press Ctrl+C to stop.")
    try:
        RecoveryAgent(args.database).run_forever(JsonlEventSource(args.events), args.interval)
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()
