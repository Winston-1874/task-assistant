from datetime import date
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator

_URGENCY_ALIASES: dict[str, str] = {
    "urgent": "critique",
    "critical": "critique",
    "high": "haute",
    "medium": "normale",
    "normal": "normale",
    "low": "basse",
}

_CONFIDENCE_ALIASES: dict[str, float] = {
    "high": 0.9,
    "medium": 0.6,
    "low": 0.3,
}


class Intent(BaseModel):
    kind: Literal["new_task", "command", "query", "update_context"]
    confidence: float = Field(ge=0.0, le=1.0)
    payload: dict[str, Any]

    @field_validator("payload", mode="before")
    @classmethod
    def _normalize_payload(cls, v: Any) -> dict[str, Any]:
        if isinstance(v, dict):
            return v
        return {"raw": v}


class Classification(BaseModel):
    category: str | None = None  # nom existant, "NEW:<nom>", ou None (fallback)
    context_id: int | None = None
    context_suggestion: str | None = None
    urgency: Literal["critique", "haute", "normale", "basse"]
    due_date: date | None = None
    needs_due_date: bool = False
    estimated_minutes: int | None = None
    tags: Annotated[list[str], Field(max_length=5)] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str

    @field_validator("urgency", mode="before")
    @classmethod
    def _normalize_urgency(cls, v: Any) -> str:
        if isinstance(v, str):
            normalized = v.lower().strip()
            return _URGENCY_ALIASES.get(normalized, normalized)
        return v

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, v: Any) -> float:
        if isinstance(v, str):
            lower = v.lower().strip()
            if lower in _CONFIDENCE_ALIASES:
                return _CONFIDENCE_ALIASES[lower]
            return float(lower)
        return v


# ---------------------------------------------------------------------------
# Étage 3 — Dialogues proactifs
# ---------------------------------------------------------------------------


class ProactiveMessage(BaseModel):
    """Réponse LLM pour ask_due_date et zombie_check."""

    message: str


class SignalPriority(BaseModel):
    """Une entrée dans la liste priorisée de la colonne Signal."""

    task_title: str
    reason: str


class SignalResponse(BaseModel):
    """Réponse LLM pour generate_signal — wrappée pour json_object."""

    priorities: list[SignalPriority] = Field(default_factory=list)


class DigestContent(BaseModel):
    """Réponse LLM pour le digest matinal."""

    summary: str
    top_tasks: list[str] = Field(default_factory=list)
    alert: str | None = None
