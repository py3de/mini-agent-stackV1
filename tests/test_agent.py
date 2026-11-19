from langchain_core.messages import AIMessage

from agent_stack.agent import run_meeting_agent
from agent_stack.config import Settings
from agent_stack.schemas import MeetingReport
from tests.fakes import ScriptedChatModel, tool_call

REPORT_ARGS = {
    "summary": "Budget approved; contract to be sent.",
    "contact_email": "lena.fischer@nordwerk.de",
    "deal_id": "9001",
    "new_deal_stage": "contractsent",
    "action_items": ["Send contract (Max)"],
    "crm_changes": [
        {"object_type": "note", "object_id": "50001", "action": "created note"},
        {"object_type": "deal", "object_id": "9001", "action": "moved deal stage", "details": "contractsent"},
    ],
    "follow_up_email": "Hi Lena, thanks for the call...",
    "sentiment": "positive",
}


def test_full_run_updates_crm_and_returns_structured_report(crm):
    llm = ScriptedChatModel(
        responses=[
            tool_call("search_contacts", {"query": "Nordwerk"}, "c1"),
            tool_call("search_deals", {"query": "Nordwerk"}, "c2"),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "create_note", "args": {"contact_id": "101", "body": "Budget approved."}, "id": "c3"},
                    {"name": "update_deal_stage", "args": {"deal_id": "9001", "stage": "contractsent"}, "id": "c4"},
                ],
            ),
            AIMessage(content="Logged the note and moved the deal to contractsent."),
            tool_call("MeetingReport", REPORT_ARGS, "r1"),
        ]
    )

    result = run_meeting_agent("Call with Lena from Nordwerk ...", llm=llm, mode="auto")

    assert isinstance(result["report"], MeetingReport)
    assert result["report"].new_deal_stage == "contractsent"
    assert [c["name"] for c in result["tool_calls"]] == ["search_contacts", "search_deals", "create_note", "update_deal_stage"]
    assert crm.data["deals"]["9001"]["dealstage"] == "contractsent"
    assert [n["contact_id"] for n in crm.data["notes"].values()] == ["101"]


def test_tool_errors_are_returned_to_the_model_not_raised(crm):
    llm = ScriptedChatModel(
        responses=[
            tool_call("update_deal_stage", {"deal_id": "does-not-exist", "stage": "closedwon"}, "c1"),
            AIMessage(content="Deal not found, nothing changed."),
            tool_call("MeetingReport", {**REPORT_ARGS, "deal_id": None, "new_deal_stage": None, "crm_changes": []}, "r1"),
        ]
    )

    result = run_meeting_agent("notes about an unknown deal", llm=llm, mode="auto")

    tool_message = llm.calls[1][-1]  # what the model saw after the failing tool call
    assert "Error updating deal" in tool_message.content
    assert result["report"].deal_id is None


def test_step_limit_stops_runaway_tool_loops(crm, monkeypatch):
    monkeypatch.setattr("agent_stack.agent.settings", Settings(max_agent_steps=2))
    llm = ScriptedChatModel(
        responses=[
            tool_call("search_contacts", {"query": "a"}, "c1"),
            tool_call("search_contacts", {"query": "b"}, "c2"),  # step limit reached here
            tool_call("MeetingReport", {**REPORT_ARGS, "crm_changes": []}, "r1"),
        ]
    )

    result = run_meeting_agent("notes", llm=llm, mode="auto")

    assert result["steps"] == 2
    assert result["report"] is not None


class OverloadedModel(ScriptedChatModel):
    """Always fails like an overloaded provider."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise RuntimeError("503 UNAVAILABLE: This model is currently experiencing high demand.")


def test_falls_back_to_backup_model_when_primary_is_overloaded(crm):
    backup = ScriptedChatModel(
        responses=[
            AIMessage(content="Nothing to change."),
            tool_call("MeetingReport", {**REPORT_ARGS, "crm_changes": []}, "r1"),
        ]
    )

    result = run_meeting_agent("notes", llm=OverloadedModel(responses=[]), fallbacks=[backup], mode="auto")

    assert result["report"].summary == REPORT_ARGS["summary"]


def test_transient_error_detection():
    from agent_stack.llm import is_transient_error

    assert is_transient_error(RuntimeError("503 UNAVAILABLE high demand"))
    assert is_transient_error(RuntimeError("429 RESOURCE_EXHAUSTED"))
    assert not is_transient_error(RuntimeError("404 NOT_FOUND model does not exist"))
    assert not is_transient_error(RuntimeError("401 invalid API key"))
