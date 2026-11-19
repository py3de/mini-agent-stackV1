"""Central configuration, read once from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "google")
    hubspot_mode: str = os.getenv("HUBSPOT_MODE", "mock")
    hubspot_access_token: str = os.getenv("HUBSPOT_ACCESS_TOKEN", "")
    webhook_secret: str = os.getenv("WEBHOOK_SECRET", "change-me")
    max_agent_steps: int = int(os.getenv("MAX_AGENT_STEPS", "8"))


settings = Settings()
