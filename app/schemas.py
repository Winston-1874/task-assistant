from datetime import date
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class Intent(BaseModel):
    kind: Literal["new_task", "command", "query", "update_context"]
    confidence: float = Field(ge=0.0, le=1.0)
    payload: dict[str, Any]


class Classification(BaseModel):
    category: str  # nom existant ou "NEW:<nom>"
    context_id: int | None = None
    context_suggestion: str | None = None
    urgency: Literal["critique", "haute", "normale", "basse"]
    due_date: date | None = None
    needs_due_date: bool = False
    estimated_minutes: int | None = None
    tags: Annotated[list[str], Field(max_length=5)] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
