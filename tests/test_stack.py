import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from agent_stack import api
from agent_stack.llm import ProviderNotConfigured, get_llm
from agent_stack.schemas import MeetingReport
from agent_stack.tools.crm import create_note, search_contacts, update_deal_stage


def test_tools_work_against_mock_crm(crm):
    found = json.loads(search_contacts.invoke({"query": "alpentech"}))
    assert found[0]["id"] == "102"
    assert "created" in create_note.invoke({"contact_id": "102", "body": "hi"})
    assert "updated" in update_deal_stage.invoke({"deal_id": "9002", "stage": "closedwon"})
    assert crm.data["deals"]["9002"]["dealstage"] == "closedwon"


def test_unknown_provider_and_missing_key(monkeypatch):
    with pytest.raises(ValueError):
        get_llm("not-a-provider")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ProviderNotConfigured):
        get_llm("openai")


def test_mcp_server_exposes_the_same_tools(crm):
    from agent_stack.mcp_server import server

    names = {t.name for t in asyncio.run(server.list_tools())}
    assert names == {"search_contacts", "search_deals", "create_note", "update_deal_stage", "search_knowledge_base"}
    asyncio.run(server.call_tool("update_deal_stage", {"deal_id": "9003", "stage": "closedwon"}))
    assert crm.data["deals"]["9003"]["dealstage"] == "closedwon"


def test_webhook_requires_secret_and_returns_report(monkeypatch):
    fake_report = MeetingReport(summary="s", follow_up_email="e", sentiment="neutral")
    monkeypatch.setattr(
        api, "run_meeting_agent",
        lambda notes, provider=None: {"report": fake_report, "tool_calls": [], "steps": 1, "latency_s": 0.1},
    )
    client = TestClient(api.app)
    body = {"notes": "Some meeting notes here", "provider": "google"}

    assert client.post("/webhooks/meeting-notes", json=body).status_code == 401
    ok = client.post("/webhooks/meeting-notes", json=body, headers={"X-Webhook-Secret": api.settings.webhook_secret})
    assert ok.status_code == 200
    assert ok.json()["report"]["sentiment"] == "neutral"
