from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Intent(BaseModel):
    kind: Literal["new_task", "command", "query", "update_context"]
    confidence: float = Field(ge=0.0, le=1.0)
    payload: dict


class Classification(BaseModel):
    category: str  # nom existant ou "NEW:<nom>"
    context_id: Optional[int] = None
    context_suggestion: Optional[str] = None
    urgency: Literal["critique", "haute", "normale", "basse"]
    due_date: Optional[date] = None
    needs_due_date: bool = False
    estimated_minutes: Optional[int] = None
    tags: list[str] = Field(default_factory=list, max_length=5)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
