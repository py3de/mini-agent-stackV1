from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from agent_stack import api
from agent_stack.agent import resume_meeting_run, run_meeting_agent, start_meeting_run
from agent_stack.governance import permission_for
from tests.fakes import ScriptedChatModel, tool_call

REPORT = {"summary": "s", "follow_up_email": "e", "sentiment": "positive"}


def write_calls() -> AIMessage:
    """The model wants one read and two writes in one turn."""
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "search_contacts", "args": {"query": "Nordwerk"}, "id": "read1"},
            {"name": "create_note", "args": {"contact_id": "101", "body": "Budget approved."}, "id": "note1"},
            {"name": "update_deal_stage", "args": {"deal_id": "9001", "stage": "contractsent"}, "id": "stage1"},
        ],
    )


def scripted() -> ScriptedChatModel:
    return ScriptedChatModel(responses=[write_calls(), AIMessage(content="Done."), tool_call("MeetingReport", REPORT, "r")])


def test_write_actions_pause_for_approval_and_nothing_changes_before(crm, audit):
    result = run_meeting_agent("notes", llm=scripted(), mode="approve")

    assert result["status"] == "awaiting_approval"
    assert [a["tool"] for a in result["pending_actions"]] == ["create_note", "update_deal_stage"]
    assert crm.data["notes"] == {}  # paused before ANY tool ran, reads included
    assert crm.data["deals"]["9001"]["dealstage"] == "qualifiedtobuy"

    final = resume_meeting_run(result["run_id"], "approve")

    assert final["status"] == "completed"
    assert crm.data["deals"]["9001"]["dealstage"] == "contractsent"
    assert len(crm.data["notes"]) == 1
    events = [e["event"] for e in audit.events(result["run_id"])]
    assert events == ["run_started", "approval_requested", "approval_decision",
                      "tool_call", "tool_call", "tool_call", "run_finished"]


def test_rejected_actions_are_not_executed_and_the_model_is_told(crm, audit):
    llm = scripted()
    result = run_meeting_agent("notes", llm=llm, mode="approve", approver=lambda pending: "reject")

    assert result["status"] == "completed"
    assert crm.data["notes"] == {}
    assert crm.data["deals"]["9001"]["dealstage"] == "qualifiedtobuy"
    tool_results = [m.content for m in llm.calls[1] if m.type == "tool"]
    assert sum("Rejected by human reviewer" in t for t in tool_results) == 2
    decisions = [e["decision"] for e in audit.events(result["run_id"]) if e["event"] == "tool_call"]
    assert decisions == ["allowed", "rejected", "rejected"]


def test_decisions_can_be_given_per_action(crm):
    result = run_meeting_agent(
        "notes", llm=scripted(), mode="approve",
        approver=lambda pending: {"note1": "approve", "stage1": "reject"},
    )

    assert result["status"] == "completed"
    assert len(crm.data["notes"]) == 1
    assert crm.data["deals"]["9001"]["dealstage"] == "qualifiedtobuy"


def test_read_only_mode_blocks_writes_without_asking(crm, audit):
    asked = []
    result = run_meeting_agent("notes", llm=scripted(), mode="read_only", approver=lambda p: asked.append(p) or "approve")

    assert asked == []
    assert result["status"] == "completed"
    assert crm.data["notes"] == {}
    decisions = [e["decision"] for e in audit.events(result["run_id"]) if e["event"] == "tool_call"]
    assert decisions == ["allowed", "blocked", "blocked"]


def test_unknown_tools_are_treated_as_writes():
    assert permission_for("search_contacts") == "read"
    assert permission_for("delete_all_deals") == "write"


def test_repeated_tool_errors_escalate_to_a_human(crm, audit):
    llm = ScriptedChatModel(
        responses=[
            tool_call("update_deal_stage", {"deal_id": "nope-1", "stage": "closedwon"}, "e1"),
            tool_call("update_deal_stage", {"deal_id": "nope-2", "stage": "closedwon"}, "e2"),
            tool_call("MeetingReport", REPORT, "r"),  # goes straight to the report after escalation
        ]
    )

    result = run_meeting_agent("notes", llm=llm, mode="auto")

    assert result["status"] == "escalated"
    assert "2 consecutive tool errors" in result["escalation"]
    assert "escalated" in [e["event"] for e in audit.events(result["run_id"])]


def test_api_approval_flow_requires_token(crm, monkeypatch):
    llm = scripted()
    monkeypatch.setattr(api, "start_meeting_run", lambda notes, provider=None, mode=None: start_meeting_run(notes, llm=llm, mode=mode))
    monkeypatch.setenv("APPROVER_TOKEN", "secret-token")
    monkeypatch.setenv("GOVERNANCE_MODE", "approve")
    client = TestClient(api.app)

    started = client.post("/agent/run", json={"notes": "Call with Lena from Nordwerk", "provider": "google"}).json()
    assert started["status"] == "awaiting_approval"

    url = f"/runs/{started['run_id']}/decision"
    assert client.post(url, json={"decision": "approve"}).status_code == 401
    done = client.post(url, json={"decision": "approve"}, headers={"X-Approver-Token": "secret-token"}).json()

    assert done["status"] == "completed"
    assert crm.data["deals"]["9001"]["dealstage"] == "contractsent"
    assert client.get(f"/runs/{started['run_id']}/audit").status_code == 200
