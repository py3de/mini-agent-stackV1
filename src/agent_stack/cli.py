"""Command line entry point.

    agent-stack run --notes-file examples/nordwerk.txt              # asks you before every CRM write
    agent-stack run --mode read_only --notes-file examples/nordwerk.txt
    agent-stack run --mode auto "Call with Jonas Weber from AlpenTech ..."
    agent-stack audit --run-id 3f2a9c1b7e44                          # show the audit trail of a run
    agent-stack providers
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent_stack.agent import run_meeting_agent
from agent_stack.governance import MODES, get_audit
from agent_stack.llm import PROVIDERS, available_providers, is_transient_error, model_name_for


def ask_in_terminal(pending: list[dict]) -> dict[str, str]:
    """Human approval in the terminal: one yes/no question per pending write action."""
    decisions = {}
    print("\nThe agent wants to make these changes:")
    for action in pending:
        args = json.dumps(action["args"], ensure_ascii=False)
        answer = input(f"  - {action['tool']}({args})\n    Approve? [y/N] ").strip().lower()
        decisions[action["id"]] = "approve" if answer in ("y", "yes") else "reject"
    return decisions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent-stack", description="Mini Agent Stack CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the meeting-notes agent")
    run.add_argument("notes", nargs="?", help="Meeting notes as text (or use --notes-file)")
    run.add_argument("--notes-file", type=Path, help="Path to a text file with meeting notes")
    run.add_argument("--provider", choices=list(PROVIDERS), help="LLM provider (default: LLM_PROVIDER)")
    run.add_argument("--mode", choices=MODES, help="Governance mode (default: GOVERNANCE_MODE or 'approve')")

    audit = sub.add_parser("audit", help="Show the audit trail (all runs, or one run)")
    audit.add_argument("--run-id", help="Only show events for this run")

    sub.add_parser("providers", help="Show providers, models and whether they are configured")

    args = parser.parse_args(argv)

    if args.command == "providers":
        ready = available_providers()
        for name in PROVIDERS:
            print(f"{name:10} {model_name_for(name):24} {'ready' if name in ready else 'missing API key'}")
        return 0

    if args.command == "audit":
        for event in get_audit().events(args.run_id):
            details = {k: v for k, v in event.items() if k not in ("ts", "run_id", "event")}
            print(f"{event['ts']}  {event['run_id']}  {event['event']:18} {json.dumps(details, ensure_ascii=False)[:160]}")
        return 0

    notes = args.notes_file.read_text(encoding="utf-8") if args.notes_file else args.notes
    if not notes:
        parser.error("Provide meeting notes as text or with --notes-file")

    try:
        result = run_meeting_agent(notes, provider=args.provider, mode=args.mode, approver=ask_in_terminal)
    except Exception as exc:  # noqa: BLE001 - show a readable message instead of a long traceback
        hint = "The provider seems overloaded or rate-limited; try again in a minute or use --provider." if is_transient_error(exc) else "Check your .env (API key, model name)."
        print(f"\nRun failed: {type(exc).__name__}: {str(exc)[:400]}\n{hint}", file=sys.stderr)
        return 1

    marks = {"allowed": "  read  ", "approved": "approved", "rejected": "REJECTED", "blocked": "BLOCKED "}
    print("\nTool calls (governance decision):")
    for event in get_audit().events(result["run_id"]):
        if event["event"] == "tool_call":
            print(f"  [{marks.get(event['decision'], event['decision'])}] {event['tool']}({json.dumps(event['args'], ensure_ascii=False)})")
    print(f"\nRun {result['run_id']}   Mode: {result['mode']}   Status: {result['status']}   "
          f"Steps: {result['steps']}   Latency: {result['latency_s']}s")
    if result["escalation"]:
        print(f"ESCALATED for human review: {result['escalation']}")
    print(f"Audit trail: agent-stack audit --run-id {result['run_id']}\n")
    print(result["report"].model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
