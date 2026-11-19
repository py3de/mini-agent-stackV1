r"""The meeting-notes -> CRM agent, built as an explicit LangGraph StateGraph.

    START -> agent --(tool calls?)--> tools -> agent -> ... -> report -> END
                   \--(done or step limit)--------------------> report

- `agent` decides which CRM tools to call (tool calling).
- `tools` executes them (LangGraph's ToolNode).
- `report` turns the whole conversation into a validated `MeetingReport` (structured output).

The graph is provider-agnostic: it only receives a LangChain chat model.
"""

from __future__ import annotations

import time
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import RetryPolicy

from agent_stack.config import settings
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
When you are done with the CRM, reply briefly with what you did. Do not ask the user questions."""

REPORT_PROMPT = """Using the meeting notes and every CRM action above, produce the final MeetingReport.
List exactly the CRM changes that were actually made (use the IDs returned by the tools).
In the follow-up email, answer the customer's open questions only with facts found in the knowledge base."""


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    steps: int
    report: MeetingReport | None


# Transient provider errors (overload, rate limits) are retried with exponential backoff.
TRANSIENT_RETRY = RetryPolicy(max_attempts=3, initial_interval=2.0, backoff_factor=2.0, retry_on=is_transient_error)


def build_agent(
    llm: BaseChatModel | None = None,
    provider: str | None = None,
    toolset: str = "crm",
    fallbacks: list[BaseChatModel] | None = None,
):
    """Compile the graph for a given model (or provider). Returns a runnable LangGraph app.

    Resilience, from inside out: if the primary model fails, each call falls back to the backup
    models (LLM_FALLBACKS); if everything fails with a transient error, the node is retried.
    """
    if llm is None:
        llm = get_llm(provider)
        fallbacks = get_fallback_llms(provider) if fallbacks is None else fallbacks
    fallbacks = fallbacks or []
    tools = get_tools(toolset)
    llm_with_tools = llm.bind_tools(tools)
    report_llm = llm.with_structured_output(MeetingReport)
    if fallbacks:
        llm_with_tools = llm_with_tools.with_fallbacks([f.bind_tools(tools) for f in fallbacks])
        report_llm = report_llm.with_fallbacks([f.with_structured_output(MeetingReport) for f in fallbacks])

    def agent_node(state: AgentState) -> dict[str, Any]:
        response = llm_with_tools.invoke([SystemMessage(AGENT_PROMPT), *state["messages"]])
        return {"messages": [response], "steps": state["steps"] + 1}

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        wants_tools = isinstance(last, AIMessage) and bool(last.tool_calls)
        return "tools" if wants_tools and state["steps"] < settings.max_agent_steps else "report"

    def report_node(state: AgentState) -> dict[str, Any]:
        # A final human turn keeps every provider happy (some reject a trailing AI/tool message).
        report = report_llm.invoke([*state["messages"], HumanMessage(REPORT_PROMPT)])
        return {"report": report}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node, retry_policy=TRANSIENT_RETRY)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("report", report_node, retry_policy=TRANSIENT_RETRY)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", "report": "report"})
    graph.add_edge("tools", "agent")
    graph.add_edge("report", END)
    return graph.compile()


def tool_calls_from(messages: list[AnyMessage]) -> list[dict[str, Any]]:
    """Every tool call the model made, in order (useful for logging and evals)."""
    return [
        {"name": call["name"], "args": call["args"]}
        for m in messages
        if isinstance(m, AIMessage)
        for call in m.tool_calls
    ]


def run_meeting_agent(
    notes: str,
    provider: str | None = None,
    llm: BaseChatModel | None = None,
    fallbacks: list[BaseChatModel] | None = None,
) -> dict[str, Any]:
    """Process one set of meeting notes end to end and return the report plus run metadata."""
    app = build_agent(llm=llm, provider=provider, fallbacks=fallbacks)
    started = time.perf_counter()
    final = app.invoke(
        {"messages": [HumanMessage(f"Meeting notes:\n\n{notes}")], "steps": 0, "report": None},
        config={"run_name": "meeting-notes-agent", "tags": [provider or settings.llm_provider]},
    )
    return {
        "report": final["report"],
        "tool_calls": tool_calls_from(final["messages"]),
        "steps": final["steps"],
        "latency_s": round(time.perf_counter() - started, 2),
    }
