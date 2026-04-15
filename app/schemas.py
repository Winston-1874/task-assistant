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

    # default_factory : résiste au cas où le LLM omet la clé (retourne {})
    priorities: list[SignalPriority] = Field(default_factory=list)


class DigestContent(BaseModel):
    """Réponse LLM pour le digest matinal."""

    summary: str  # 2-4 phrases résumant la journée
    top_tasks: list[str] = Field(default_factory=list)  # titres des 3-5 tâches clés
    alert: str | None = None  # surcharge capacité ou zombie critique
