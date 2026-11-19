"""CRM tools the agent can call.

Plain functions decorated with `@tool`: the docstring is what the LLM sees, so it says when
to use the tool, not just what it does. Errors are returned as text instead of raised, so the
model can recover (e.g. search again) rather than the whole run crashing.
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

from agent_stack.hubspot_client import get_hubspot
from agent_stack.schemas import DealStage


def _json(data: object) -> str:
    return json.dumps(data, ensure_ascii=False)


@tool
def search_contacts(query: str) -> str:
    """Find CRM contacts by name, email or company. Use this first to get a contact_id."""
    results = get_hubspot().search_contacts(query)
    return _json(results) if results else f"No contacts found for '{query}'. Try a shorter query, e.g. only the company or last name."


@tool
def search_deals(query: str) -> str:
    """Find deals by deal name or company. Use this to get a deal_id and the current deal stage."""
    results = get_hubspot().search_deals(query)
    return _json(results) if results else f"No deals found for '{query}'. Try only the company name."


@tool
def create_note(contact_id: str, body: str) -> str:
    """Log a meeting note on a contact. Body should be a concise summary plus next steps."""
    try:
        return _json({"status": "created", "note": get_hubspot().create_note(contact_id, body)})
    except Exception as exc:  # noqa: BLE001 - surface any backend error to the model
        return f"Error creating note: {exc}"


@tool
def update_deal_stage(deal_id: str, stage: DealStage) -> str:
    """Move a deal to a new pipeline stage. Only call this if the meeting clearly changed the deal's status."""
    try:
        return _json({"status": "updated", "deal": get_hubspot().update_deal_stage(deal_id, stage)})
    except Exception as exc:  # noqa: BLE001
        return f"Error updating deal: {exc}"


CRM_TOOLS = [search_contacts, search_deals, create_note, update_deal_stage]
