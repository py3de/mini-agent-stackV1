"""HubSpot CRM client with two interchangeable backends.

- MockHubSpot: an in-memory CRM with demo data, so the whole stack runs without an account.
- LiveHubSpot: the real HubSpot CRM v3 API (use a free developer test account + private app token).

Both expose the same four methods, so tools, the agent and the MCP server never care which
one is active. Select with HUBSPOT_MODE=mock|live.
"""

from __future__ import annotations

import copy
import itertools
import time
from typing import Any, Protocol

import httpx

from agent_stack.config import settings

SEED_DATA: dict[str, Any] = {
    "contacts": {
        "101": {"id": "101", "firstname": "Lena", "lastname": "Fischer", "email": "lena.fischer@nordwerk.de", "company": "Nordwerk GmbH"},
        "102": {"id": "102", "firstname": "Jonas", "lastname": "Weber", "email": "j.weber@alpentech.io", "company": "AlpenTech"},
        "103": {"id": "103", "firstname": "Sofia", "lastname": "Lindqvist", "email": "sofia@fjordlogistics.se", "company": "Fjord Logistics"},
    },
    "deals": {
        "9001": {"id": "9001", "dealname": "Nordwerk - CRM Agent Rollout", "dealstage": "qualifiedtobuy", "amount": "48000", "contact_id": "101"},
        "9002": {"id": "9002", "dealname": "AlpenTech - Service Hub Migration", "dealstage": "presentationscheduled", "amount": "22000", "contact_id": "102"},
        "9003": {"id": "9003", "dealname": "Fjord Logistics - Sales Copilot", "dealstage": "appointmentscheduled", "amount": "65000", "contact_id": "103"},
    },
    "notes": {},
}


class HubSpotBackend(Protocol):
    def search_contacts(self, query: str) -> list[dict[str, Any]]: ...
    def search_deals(self, query: str) -> list[dict[str, Any]]: ...
    def create_note(self, contact_id: str, body: str) -> dict[str, Any]: ...
    def update_deal_stage(self, deal_id: str, stage: str) -> dict[str, Any]: ...


class MockHubSpot:
    """In-memory CRM. `reset()` restores the demo data (used by tests and evals)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.data = copy.deepcopy(SEED_DATA)
        self._ids = itertools.count(50001)

    @staticmethod
    def _matches(record: dict[str, Any], query: str) -> bool:
        q = query.lower().strip()
        return any(q in str(v).lower() for v in record.values())

    def search_contacts(self, query: str) -> list[dict[str, Any]]:
        return [c for c in self.data["contacts"].values() if self._matches(c, query)]

    def search_deals(self, query: str) -> list[dict[str, Any]]:
        return [d for d in self.data["deals"].values() if self._matches(d, query)]

    def create_note(self, contact_id: str, body: str) -> dict[str, Any]:
        if contact_id not in self.data["contacts"]:
            raise ValueError(f"Contact {contact_id} not found")
        note = {"id": str(next(self._ids)), "contact_id": contact_id, "body": body}
        self.data["notes"][note["id"]] = note
        return note

    def update_deal_stage(self, deal_id: str, stage: str) -> dict[str, Any]:
        deal = self.data["deals"].get(deal_id)
        if deal is None:
            raise ValueError(f"Deal {deal_id} not found")
        deal["dealstage"] = stage
        return deal


class LiveHubSpot:
    """Thin wrapper around the HubSpot CRM v3 REST API."""

    BASE_URL = "https://api.hubapi.com"
    NOTE_TO_CONTACT_ASSOCIATION = 202  # HubSpot-defined association type: note -> contact

    def __init__(self, token: str) -> None:
        if not token:
            raise RuntimeError("HUBSPOT_MODE=live needs HUBSPOT_ACCESS_TOKEN (private app token).")
        self.client = httpx.Client(
            base_url=self.BASE_URL,
            headers={"Authorization": f"Bearer {token}"},
            timeout=20,
        )

    def _search(self, object_type: str, query: str, properties: list[str]) -> list[dict[str, Any]]:
        r = self.client.post(
            f"/crm/v3/objects/{object_type}/search",
            json={"query": query, "properties": properties, "limit": 5},
        )
        r.raise_for_status()
        return [{"id": row["id"], **row["properties"]} for row in r.json().get("results", [])]

    def search_contacts(self, query: str) -> list[dict[str, Any]]:
        return self._search("contacts", query, ["firstname", "lastname", "email", "company"])

    def search_deals(self, query: str) -> list[dict[str, Any]]:
        return self._search("deals", query, ["dealname", "dealstage", "amount"])

    def create_note(self, contact_id: str, body: str) -> dict[str, Any]:
        payload = {
            "properties": {"hs_note_body": body, "hs_timestamp": int(time.time() * 1000)},
            "associations": [
                {
                    "to": {"id": contact_id},
                    "types": [{"associationCategory": "HUBSPOT_DEFINED", "associationTypeId": self.NOTE_TO_CONTACT_ASSOCIATION}],
                }
            ],
        }
        r = self.client.post("/crm/v3/objects/notes", json=payload)
        r.raise_for_status()
        return {"id": r.json()["id"], "contact_id": contact_id, "body": body}

    def update_deal_stage(self, deal_id: str, stage: str) -> dict[str, Any]:
        r = self.client.patch(f"/crm/v3/objects/deals/{deal_id}", json={"properties": {"dealstage": stage}})
        r.raise_for_status()
        return {"id": deal_id, **r.json().get("properties", {})}


_backend: HubSpotBackend | None = None


def get_hubspot() -> HubSpotBackend:
    """Shared backend instance for the current process."""
    global _backend
    if _backend is None:
        _backend = LiveHubSpot(settings.hubspot_access_token) if settings.hubspot_mode == "live" else MockHubSpot()
    return _backend


def set_hubspot(backend: HubSpotBackend) -> None:
    """Swap the backend (used by tests and evals)."""
    global _backend
    _backend = backend
