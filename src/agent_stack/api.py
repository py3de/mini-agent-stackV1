"""HTTP API so other systems can use the agent, with human approval built in.

- POST /agent/run                     start a run on meeting notes. In approve mode the response can be
                                      status "awaiting_approval" with the pending write actions.
- POST /webhooks/meeting-notes        same, for external systems (e.g. a meeting recorder); needs X-Webhook-Secret
- POST /runs/{run_id}/decision        approve or reject pending actions; needs X-Approver-Token
- GET  /runs/{run_id}/audit           the audit trail of one run
- GET  /providers                     which LLM providers are configured right now
- GET  /health

Callers can make a run stricter (mode "read_only") but never looser: "auto" mode is only
available through server configuration, not per request.

Run:  uvicorn agent_stack.api:app --reload
"""

from __future__ import annotations

import os
import secrets
from typing import Literal

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from agent_stack.agent import resume_meeting_run, start_meeting_run
from agent_stack.config import settings
from agent_stack.governance import current_mode, get_audit
from agent_stack.llm import PROVIDERS, ProviderNotConfigured, available_providers, model_name_for
from agent_stack.schemas import MeetingReport

app = FastAPI(title="Mini Agent Stack", version="0.2.0")


class RunRequest(BaseModel):
    notes: str = Field(min_length=10, description="Raw meeting notes")
    provider: str | None = Field(default=None, description="google | openai | anthropic | ollama")
    mode: Literal["approve", "read_only"] | None = Field(default=None, description="Only stricter modes can be requested")


class DecisionRequest(BaseModel):
    decision: Literal["approve", "reject"] | dict[str, Literal["approve", "reject"]] = Field(
        description="'approve' / 'reject' for all pending actions, or {action_id: 'approve' | 'reject'}"
    )


class PendingAction(BaseModel):
    id: str
    tool: str
    args: dict


class RunResponse(BaseModel):
    run_id: str
    status: Literal["awaiting_approval", "completed", "escalated"]
    mode: str
    pending_actions: list[PendingAction]
    report: MeetingReport | None
    escalation: str | None
    tool_calls: list[dict]
    steps: int
    latency_s: float


def _start(req: RunRequest) -> RunResponse:
    provider = req.provider or settings.llm_provider
    if provider not in PROVIDERS:
        raise HTTPException(400, f"Unknown provider '{provider}'")
    try:
        result = start_meeting_run(req.notes, provider=provider, mode=req.mode or current_mode())
    except ProviderNotConfigured as exc:
        raise HTTPException(400, str(exc)) from exc
    return RunResponse(**result)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "hubspot_mode": settings.hubspot_mode, "governance_mode": current_mode()}


@app.get("/providers")
def providers() -> dict:
    ready = available_providers()
    return {name: {"model": model_name_for(name), "configured": name in ready} for name in PROVIDERS}


@app.post("/agent/run", response_model=RunResponse)
def agent_run(req: RunRequest) -> RunResponse:
    return _start(req)


@app.post("/webhooks/meeting-notes", response_model=RunResponse)
def meeting_notes_webhook(req: RunRequest, x_webhook_secret: str = Header(default="")) -> RunResponse:
    if not secrets.compare_digest(x_webhook_secret, settings.webhook_secret):
        raise HTTPException(401, "Invalid webhook secret")
    return _start(req)


@app.post("/runs/{run_id}/decision", response_model=RunResponse)
def decide(run_id: str, req: DecisionRequest, x_approver_token: str = Header(default="")) -> RunResponse:
    if not secrets.compare_digest(x_approver_token, os.getenv("APPROVER_TOKEN", "change-me-approver")):
        raise HTTPException(401, "Invalid approver token")
    try:
        return RunResponse(**resume_meeting_run(run_id, req.decision))
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/runs/{run_id}/audit")
def run_audit(run_id: str) -> list[dict]:
    events = get_audit().events(run_id)
    if not events:
        raise HTTPException(404, f"No audit events for run '{run_id}'")
    return events
