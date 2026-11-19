"""HTTP API so other systems can use the agent.

- POST /agent/run                 run the agent on meeting notes (choose a provider per request)
- POST /webhooks/meeting-notes    same, for external systems (e.g. a meeting recorder); needs X-Webhook-Secret
- GET  /providers                 which LLM providers are configured right now
- GET  /health

Run:  uvicorn agent_stack.api:app --reload
"""

from __future__ import annotations

import secrets

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from agent_stack.agent import run_meeting_agent
from agent_stack.config import settings
from agent_stack.llm import PROVIDERS, ProviderNotConfigured, available_providers, model_name_for
from agent_stack.schemas import MeetingReport

app = FastAPI(title="Mini Agent Stack", version="0.1.0")


class RunRequest(BaseModel):
    notes: str = Field(min_length=10, description="Raw meeting notes")
    provider: str | None = Field(default=None, description="google | openai | anthropic | ollama")


class RunResponse(BaseModel):
    provider: str
    report: MeetingReport
    tool_calls: list[dict]
    steps: int
    latency_s: float


def _run(req: RunRequest) -> RunResponse:
    provider = req.provider or settings.llm_provider
    if provider not in PROVIDERS:
        raise HTTPException(400, f"Unknown provider '{provider}'")
    try:
        result = run_meeting_agent(req.notes, provider=provider)
    except ProviderNotConfigured as exc:
        raise HTTPException(400, str(exc)) from exc
    return RunResponse(provider=provider, **result)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "hubspot_mode": settings.hubspot_mode}


@app.get("/providers")
def providers() -> dict:
    ready = available_providers()
    return {name: {"model": model_name_for(name), "configured": name in ready} for name in PROVIDERS}


@app.post("/agent/run", response_model=RunResponse)
def agent_run(req: RunRequest) -> RunResponse:
    return _run(req)


@app.post("/webhooks/meeting-notes", response_model=RunResponse)
def meeting_notes_webhook(req: RunRequest, x_webhook_secret: str = Header(default="")) -> RunResponse:
    if not secrets.compare_digest(x_webhook_secret, settings.webhook_secret):
        raise HTTPException(401, "Invalid webhook secret")
    return _run(req)
