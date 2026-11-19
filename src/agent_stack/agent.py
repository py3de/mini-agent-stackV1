r"""The meeting-notes -> CRM agent, built as an explicit LangGraph StateGraph with a governance gate.

    START -> agent --(tool calls)--> gate --> agent -> ... -> report -> END
                   \--(done / step limit / escalation)-----> report

- `agent`  decides which tools to call (tool calling).
- `gate`   applies the governance layer to every tool call before it runs: read tools run freely;
           write tools are approved by a human (LangGraph interrupt), blocked (read_only mode) or
           auto-approved (auto mode). Every decision is written to the audit trail. Repeated tool
           errors escalate the run to a human instead of letting the model keep guessing.
- `report` turns the whole conversation into a validated `MeetingReport` (structured output).

Runs are checkpointed, so a run paused for approval can be resumed later by its run ID.
The graph is provider-agnostic: it only receives a LangChain chat model.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, RetryPolicy, interrupt

from agent_stack.config import settings
from agent_stack.governance import current_mode, get_audit, new_run_id, permission_for
from agent_stack.llm import get_fallback_llms, get_llm, is_transient_error
from agent_stack.schemas import MeetingReport
from agent_stack.tools import get_tools

AGENT_PROMPT = """You are a CRM assistant for a B2B consultancy. You receive raw meeting notes.
Your job:
1. Identify the customer contact and the related deal by searching the CRM (never invent IDs).
2. Log a concise meeting note on the contact (summary + next steps).
3. Only if the notes clearly show the deal moved forward or backward, update the deal stage.
   Valid stages: appointmentscheduled, qualifiedtobuy, presentationscheduled,
   decisionmakerboughtin, contractsent, closedwon, closedlost.
4. If the customer raised a question or concern (e.g. hosting, security, timelines, payment terms),
   search the knowledge base so the follow-up email can answer it with facts. Never guess company facts.
Write actions may be rejected by a human reviewer or blocked by policy. If that happens, do not retry
the same action; continue with what you can do.
When you are done with the CRM, reply briefly with what you did. Do not ask the user questions."""

REPORT_PROMPT = """Using the meeting notes and every CRM action above, produce the final MeetingReport.
List exactly the CRM changes that were actually made (use the IDs returned by the tools); do not list
actions that were rejected or blocked. In the follow-up email, answer the customer's open questions only
with facts found in the knowledge base."""

ESCALATE_AFTER_ERRORS = int(os.getenv("ESCALATE_AFTER_ERRORS", "2"))

# Transient provider errors (overload, rate limits) are retried with exponential backoff.
TRANSIENT_RETRY = RetryPolicy(max_attempts=3, initial_interval=2.0, backoff_factor=2.0, retry_on=is_transient_error)

# Paused runs live here until they are approved or rejected. Swap for a Postgres checkpointer to
# survive restarts (see roadmap).
CHECKPOINTER = InMemorySaver(
    serde=JsonPlusSerializer(allowed_msgpack_modules=[("agent_stack.schemas", "MeetingReport"), ("agent_stack.schemas", "CRMChange")])
)
_RUNS: dict[str, dict[str, Any]] = {}


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    steps: int
    report: MeetingReport | None
    run_id: str
    mode: str
    tool_errors: int
    escalation: str | None


def _normalize_decisions(answer: Any, pending: list[dict[str, Any]]) -> dict[str, str]:
    """Accept 'approve' / 'reject' for all pending actions, or {action_id: 'approve' | 'reject'}."""
    if isinstance(answer, str):
        return {c["id"]: answer.lower() for c in pending}
    if isinstance(answer, dict):
        return {c["id"]: str(answer.get(c["id"], "reject")).lower() for c in pending}
    return {c["id"]: "reject" for c in pending}


def build_agent(
    llm: BaseChatModel | None = None,
    provider: str | None = None,
    toolset: str = "crm",
    fallbacks: list[BaseChatModel] | None = None,
):
    """Compile the graph for a given model (or provider). Returns a runnable, checkpointed LangGraph app.

    Resilience, from inside out: if the primary model fails, each call falls back to the backup
    models (LLM_FALLBACKS); if everything fails with a transient error, the node is retried.
    """
    if llm is None:
        llm = get_llm(provider)
        fallbacks = get_fallback_llms(provider) if fallbacks is None else fallbacks
    fallbacks = fallbacks or []
    tools = get_tools(toolset)
    tools_by_name = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)
    report_llm = llm.with_structured_output(MeetingReport)
    if fallbacks:
        llm_with_tools = llm_with_tools.with_fallbacks([f.bind_tools(tools) for f in fallbacks])
        report_llm = report_llm.with_fallbacks([f.with_structured_output(MeetingReport) for f in fallbacks])

    def agent_node(state: AgentState) -> dict[str, Any]:
        response = llm_with_tools.invoke([SystemMessage(AGENT_PROMPT), *state["messages"]])
        return {"messages": [response], "steps": state["steps"] + 1}

    def route_after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        wants_tools = isinstance(last, AIMessage) and bool(last.tool_calls)
        return "gate" if wants_tools and state["steps"] < settings.max_agent_steps else "report"

    def gate_node(state: AgentState) -> dict[str, Any]:
        calls = state["messages"][-1].tool_calls
        mode, run_id = state["mode"], state["run_id"]

        decisions: dict[str, str] = {}
        pending: list[dict[str, Any]] = []
        for call in calls:
            if permission_for(call["name"]) == "read" or mode == "auto":
                decisions[call["id"]] = "allowed"
            elif mode == "read_only":
                decisions[call["id"]] = "blocked"
            else:
                pending.append(call)

        if pending:
            # Pauses the run here. Nothing below runs until a human answers, and nothing above has
            # side effects, so it is safe that LangGraph re-runs this node when the run resumes.
            answer = interrupt({
                "run_id": run_id,
                "pending_actions": [{"id": c["id"], "tool": c["name"], "args": c["args"]} for c in pending],
            })
            for call_id, verdict in _normalize_decisions(answer, pending).items():
                decisions[call_id] = "approved" if verdict == "approve" else "rejected"

        audit = get_audit()
        errors = state["tool_errors"]
        messages = []
        for call in calls:
            name, decision = call["name"], decisions[call["id"]]
            if decision in ("allowed", "approved"):
                tool = tools_by_name.get(name)
                try:
                    output = str(tool.invoke(call["args"])) if tool else f"Error: unknown tool '{name}'"
                except Exception as exc:  # noqa: BLE001 - a failing tool is reported, not fatal
                    output = f"Error running {name}: {exc}"
                errors = errors + 1 if output.startswith("Error") else 0
            elif decision == "blocked":
                output = f"Blocked by governance policy: '{name}' is a write action and this run is read-only. Not executed."
            else:
                output = f"Rejected by human reviewer: '{name}' was not executed. Do not retry it; continue without it."
            audit.record("tool_call", run_id, tool=name, args=call["args"], permission=permission_for(name),
                         decision=decision, result=output[:300])
            messages.append(ToolMessage(content=output, tool_call_id=call["id"], name=name))

        escalation = f"{errors} consecutive tool errors" if errors >= ESCALATE_AFTER_ERRORS else None
        return {"messages": messages, "tool_errors": errors, "escalation": escalation}

    def route_after_gate(state: AgentState) -> str:
        return "report" if state["escalation"] else "agent"

    def report_node(state: AgentState) -> dict[str, Any]:
        escalation = state["escalation"]
        last = state["messages"][-1]
        if not escalation and isinstance(last, AIMessage) and last.tool_calls:
            escalation = f"step limit of {settings.max_agent_steps} reached"
        prompt = REPORT_PROMPT
        if escalation:
            prompt += f"\nThis run was escalated for human review ({escalation}); say so in the summary."
        # A final human turn keeps every provider happy (some reject a trailing AI/tool message).
        report = report_llm.invoke([*state["messages"], HumanMessage(prompt)])
        return {"report": report, "escalation": escalation}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node, retry_policy=TRANSIENT_RETRY)
    graph.add_node("gate", gate_node)  # no retries: it has side effects
    graph.add_node("report", report_node, retry_policy=TRANSIENT_RETRY)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent, {"gate": "gate", "report": "report"})
    graph.add_conditional_edges("gate", route_after_gate, {"agent": "agent", "report": "report"})
    graph.add_edge("report", END)
    return graph.compile(checkpointer=CHECKPOINTER)


def tool_calls_from(messages: list[AnyMessage]) -> list[dict[str, Any]]:
    """Every tool call the model made, in order (useful for logging and evals)."""
    return [
        {"name": call["name"], "args": call["args"]}
        for m in messages
        if isinstance(m, AIMessage)
        for call in m.tool_calls
    ]


def _result(run_id: str, output: dict[str, Any]) -> dict[str, Any]:
    """Turn graph output into a run result: either awaiting approval, or finished (completed / escalated)."""
    run = _RUNS[run_id]
    state = run["app"].get_state(run["config"]).values
    base = {
        "run_id": run_id,
        "mode": state["mode"],
        "tool_calls": tool_calls_from(state["messages"]),
        "steps": state["steps"],
        "latency_s": round(time.perf_counter() - run["started"], 2),
    }
    audit = get_audit()
    if output.get("__interrupt__"):
        pending = output["__interrupt__"][0].value["pending_actions"]
        audit.record("approval_requested", run_id, pending_actions=pending)
        return {**base, "status": "awaiting_approval", "pending_actions": pending, "report": None, "escalation": None}

    escalation = state.get("escalation")
    status = "escalated" if escalation else "completed"
    if escalation:
        audit.record("escalated", run_id, reason=escalation)
    audit.record("run_finished", run_id, status=status, steps=state["steps"], latency_s=base["latency_s"])
    _RUNS.pop(run_id, None)
    return {**base, "status": status, "pending_actions": [], "report": state["report"], "escalation": escalation}


def start_meeting_run(
    notes: str,
    provider: str | None = None,
    llm: BaseChatModel | None = None,
    fallbacks: list[BaseChatModel] | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    """Start a run. Returns a finished result, or status 'awaiting_approval' with the pending actions."""
    mode = current_mode(mode)
    run_id = new_run_id()
    app = build_agent(llm=llm, provider=provider, fallbacks=fallbacks)
    config = {
        "configurable": {"thread_id": run_id},
        "run_name": "meeting-notes-agent",
        "tags": [provider or settings.llm_provider, f"governance:{mode}"],
    }
    _RUNS[run_id] = {"app": app, "config": config, "started": time.perf_counter()}
    get_audit().record("run_started", run_id, mode=mode, provider=provider or settings.llm_provider)
    output = app.invoke(
        {
            "messages": [HumanMessage(f"Meeting notes:\n\n{notes}")],
            "steps": 0,
            "report": None,
            "run_id": run_id,
            "mode": mode,
            "tool_errors": 0,
            "escalation": None,
        },
        config=config,
    )
    return _result(run_id, output)


def resume_meeting_run(run_id: str, decisions: str | dict[str, str]) -> dict[str, Any]:
    """Resume a paused run with 'approve' / 'reject' (for all) or {action_id: 'approve' | 'reject'}."""
    if run_id not in _RUNS:
        raise KeyError(f"No run awaiting approval with id '{run_id}'")
    run = _RUNS[run_id]
    get_audit().record("approval_decision", run_id, decisions=decisions)
    output = run["app"].invoke(Command(resume=decisions), config=run["config"])
    return _result(run_id, output)


Approver = Callable[[list[dict[str, Any]]], "str | dict[str, str]"]


def run_meeting_agent(
    notes: str,
    provider: str | None = None,
    llm: BaseChatModel | None = None,
    fallbacks: list[BaseChatModel] | None = None,
    mode: str | None = None,
    approver: Approver | None = None,
) -> dict[str, Any]:
    """Run end to end. In approve mode, `approver` is asked for each batch of pending write actions;
    without an approver the result is returned as 'awaiting_approval' so the caller can resume later."""
    result = start_meeting_run(notes, provider=provider, llm=llm, fallbacks=fallbacks, mode=mode)
    while result["status"] == "awaiting_approval" and approver is not None:
        result = resume_meeting_run(result["run_id"], approver(result["pending_actions"]))
    return result
