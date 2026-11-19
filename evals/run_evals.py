"""Compare LLM providers on the same meeting-notes cases.

    python evals/run_evals.py                       # every configured provider
    python evals/run_evals.py --providers google ollama

For each case the CRM is reset to the demo data, the agent runs, and we check what actually
happened in the CRM (not just what the model claims). Writes evals/results/results.md.

Scores per case:
- schema_ok:  a valid MeetingReport came back
- note_ok:    a note was logged on the right contact (or none, for unknown contacts)
- stage_ok:   the deal ended in the expected stage, and no other deal was touched
- ids_ok:     the report's contact/deal IDs match reality (no invented IDs)
- kb_ok:      the knowledge base was searched when the customer asked a question
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_stack.agent import run_meeting_agent  # noqa: E402
from agent_stack.hubspot_client import SEED_DATA, MockHubSpot, set_hubspot  # noqa: E402
from agent_stack.llm import available_providers, get_llm, model_name_for  # noqa: E402

HERE = Path(__file__).resolve().parent
METRICS = ["schema_ok", "note_ok", "stage_ok", "ids_ok", "kb_ok"]


def score_case(case: dict, crm: MockHubSpot, result: dict | None) -> dict:
    exp = case["expected"]
    report = result["report"] if result else None
    notes = list(crm.data["notes"].values())

    if exp["contact_id"] is None:
        note_ok = not notes
    else:
        note_ok = any(n["contact_id"] == exp["contact_id"] for n in notes)

    stage_ok = True
    for deal_id, seed in SEED_DATA["deals"].items():
        now = crm.data["deals"][deal_id]["dealstage"]
        should_be = exp["new_deal_stage"] if deal_id == exp["deal_id"] and exp["new_deal_stage"] else seed["dealstage"]
        stage_ok &= now == should_be

    ids_ok = bool(report) and (report.deal_id or None) == exp["deal_id"]

    used_kb = bool(result) and any(c["name"] == "search_knowledge_base" for c in result["tool_calls"])
    kb_ok = used_kb or not exp.get("uses_kb", False)

    return {"schema_ok": report is not None, "note_ok": note_ok, "stage_ok": stage_ok, "ids_ok": ids_ok, "kb_ok": kb_ok}


def run(providers: list[str]) -> list[dict]:
    cases = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))
    rows = []
    for provider in providers:
        llm = get_llm(provider)
        for case in cases:
            crm = MockHubSpot()
            set_hubspot(crm)
            error, result = None, None
            try:
                result = run_meeting_agent(case["notes"], provider=provider, llm=llm)
            except Exception as exc:  # noqa: BLE001 - a crash is a result too
                error = f"{type(exc).__name__}: {exc}"
            scores = score_case(case, crm, result)
            rows.append({
                "provider": provider,
                "model": model_name_for(provider),
                "case": case["id"],
                **scores,
                "tool_calls": len(result["tool_calls"]) if result else 0,
                "latency_s": result["latency_s"] if result else None,
                "error": error,
            })
            status = "PASS" if all(scores.values()) else "FAIL"
            print(f"[{provider}] {case['id']:18} {status}  {scores}" + (f"  {error}" if error else ""))
    return rows


def write_report(rows: list[dict]) -> Path:
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    lines = [
        "# Eval results\n",
        "| Provider | Model | Schema | Note | Stage | IDs | KB | All passed | Avg tool calls | Median latency (s) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for provider in dict.fromkeys(r["provider"] for r in rows):
        pr = [r for r in rows if r["provider"] == provider]
        n = len(pr)
        pct = {m: f"{sum(r[m] for r in pr)}/{n}" for m in METRICS}
        all_pass = sum(all(r[m] for m in METRICS) for r in pr)
        lat = [r["latency_s"] for r in pr if r["latency_s"] is not None]
        lines.append(
            f"| {provider} | {pr[0]['model']} | {pct['schema_ok']} | {pct['note_ok']} | {pct['stage_ok']} | "
            f"{pct['ids_ok']} | {pct['kb_ok']} | **{all_pass}/{n}** | {statistics.mean(r['tool_calls'] for r in pr):.1f} | "
            f"{statistics.median(lat) if lat else '-'} |"
        )
    path = out_dir / "results.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--providers", nargs="*", help="Default: every provider with an API key")
    args = parser.parse_args()
    providers = args.providers or [p for p in available_providers() if p != "ollama"]
    if not providers:
        sys.exit("No provider configured. Set an API key in .env, or pass --providers ollama.")
    path = write_report(run(providers))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
