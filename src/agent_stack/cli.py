"""Command line entry point.

    agent-stack run --provider google --notes-file examples/nordwerk.txt
    agent-stack run "Call with Jonas Weber from AlpenTech ..."
    agent-stack providers
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent_stack.agent import run_meeting_agent
from agent_stack.llm import PROVIDERS, available_providers, is_transient_error, model_name_for


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent-stack", description="Mini Agent Stack CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the meeting-notes agent")
    run.add_argument("notes", nargs="?", help="Meeting notes as text (or use --notes-file)")
    run.add_argument("--notes-file", type=Path, help="Path to a text file with meeting notes")
    run.add_argument("--provider", choices=list(PROVIDERS), help="LLM provider (default: LLM_PROVIDER)")

    sub.add_parser("providers", help="Show providers, models and whether they are configured")

    args = parser.parse_args(argv)

    if args.command == "providers":
        ready = available_providers()
        for name in PROVIDERS:
            print(f"{name:10} {model_name_for(name):24} {'ready' if name in ready else 'missing API key'}")
        return 0

    notes = args.notes_file.read_text(encoding="utf-8") if args.notes_file else args.notes
    if not notes:
        parser.error("Provide meeting notes as text or with --notes-file")

    try:
        result = run_meeting_agent(notes, provider=args.provider)
    except Exception as exc:  # noqa: BLE001 - show a readable message instead of a long traceback
        hint = "The provider seems overloaded or rate-limited; try again in a minute or use --provider." if is_transient_error(exc) else "Check your .env (API key, model name)."
        print(f"\nRun failed: {type(exc).__name__}: {str(exc)[:400]}\n{hint}", file=sys.stderr)
        return 1
    print("\nTool calls:")
    for call in result["tool_calls"]:
        print(f"  - {call['name']}({json.dumps(call['args'], ensure_ascii=False)})")
    print(f"\nSteps: {result['steps']}   Latency: {result['latency_s']}s\n")
    print(result["report"].model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
