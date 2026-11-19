"""Structured outputs.

The agent's final answer is always validated against `MeetingReport`, whichever LLM produced
it. Downstream systems (webhooks, dashboards, other agents) can rely on this shape.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DealStage = Literal[
    "appointmentscheduled",
    "qualifiedtobuy",
    "presentationscheduled",
    "decisionmakerboughtin",
    "contractsent",
    "closedwon",
    "closedlost",
]


class CRMChange(BaseModel):
    """One change the agent made (or proposed) in the CRM."""

    object_type: Literal["contact", "deal", "note"]
    object_id: str = Field(description="ID of the CRM record that was changed or created")
    action: str = Field(description="What was done, e.g. 'created note' or 'moved deal stage'")
    details: str = Field(default="", description="Short human-readable detail, e.g. the new stage")


class MeetingReport(BaseModel):
    """Final, structured result of processing one set of meeting notes."""

    summary: str = Field(description="2-3 sentence summary of the meeting")
    contact_email: str | None = Field(default=None, description="Email of the main customer contact, if identified")
    deal_id: str | None = Field(default=None, description="ID of the related deal, if identified")
    new_deal_stage: DealStage | None = Field(default=None, description="Deal stage after the meeting, if it changed")
    action_items: list[str] = Field(default_factory=list, description="Concrete next steps with owners if known")
    crm_changes: list[CRMChange] = Field(default_factory=list, description="Every CRM change the agent made")
    follow_up_email: str = Field(description="Short, friendly follow-up email to the customer")
    sentiment: Literal["positive", "neutral", "negative"] = Field(description="Customer sentiment in the meeting")
