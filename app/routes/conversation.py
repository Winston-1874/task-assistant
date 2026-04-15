"""
Route conversationnelle — POST /conversation/parse.

Reçoit un message libre, passe par le routeur d'intention, puis exécute l'action.
Retourne un fragment HTML inséré dans le toast de réponse.
"""

import html
import json
import logging
from datetime import date

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import require_auth
from app.db import get_db
from app.llm.classifier import classify_task
from app.llm.client import LLMTransportError
from app.llm.proactive import ask_due_date
from app.llm.router import route_intent
from app.models import Task

from app.templates_config import templates

logger = logging.getLogger(__name__)
router = APIRouter(dependencies=[Depends(require_auth)])


@router.post("/conversation/parse", response_class=HTMLResponse)
async def parse_message(
    request: Request,
    message: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Orchestre : routeur → action → fragment HTML de réponse.

    - new_task  : classifie + crée la tâche, retourne task_card
    - command / query / update_context : message informatif V1
    - Erreur réseau LLM → fragment d'erreur
    """
    try:
        intent = await route_intent(message)
    except LLMTransportError as e:
        logger.error("Réseau LLM mort lors du routage: %s", e)
        return _error_html("Service LLM indisponible. Réessaie dans quelques instants.")

    if intent.kind == "new_task":
        return await _handle_new_task(request, message, db)

    labels = {
        "command": "Commande détectée — fonctionnalité à venir.",
        "query": "Question sur le backlog — fonctionnalité à venir.",
        "update_context": "Info contexte notée — fonctionnalité à venir.",
    }
    text = html.escape(labels.get(intent.kind, "Message reçu."))
    return HTMLResponse(f'<div class="toast toast-info">{text}</div>')


async def _handle_new_task(
    request: Request,
    message: str,
    db: AsyncSession,
) -> HTMLResponse:
    try:
        classification, llm_result = await classify_task(message, None, db)
    except LLMTransportError as e:
        logger.error("Réseau LLM mort lors de la classification: %s", e)
        return _error_html("Service LLM indisponible — message non sauvegardé. Réessaie dans quelques instants.")

    resolved_category = _resolve_category(classification.category)
    task = Task(
        title=message[:200],
        category=resolved_category,
        context_id=classification.context_id,
        urgency=classification.urgency,
        due_date=classification.due_date,
        estimated_minutes=classification.estimated_minutes,
        tags=json.dumps(classification.tags) if classification.tags else None,
        needs_review=1 if (classification.confidence < 0.7 or resolved_category is None) else 0,
        llm_confidence=classification.confidence,
        llm_reasoning=classification.reasoning,
        llm_raw_response=llm_result.raw_json if llm_result else None,
    )
    db.add(task)
    await db.flush()

    pending = None
    if classification.needs_due_date and task.due_date is None:
        pending = await ask_due_date(task, db)

    await db.commit()

    return templates.TemplateResponse(
        request,
        "fragments/task_card.html",
        {"task": task, "pending": pending, "today": date.today()},
    )


def _resolve_category(category: str | None) -> str | None:
    """Retourne None si la catégorie est un préfixe NEW: (LLM hallucination) ou déjà None."""
    if category and category.startswith("NEW:"):
        return None
    return category


def _error_html(message: str) -> HTMLResponse:
    return HTMLResponse(
        f'<div class="toast toast-error" role="alert">{html.escape(message)}</div>'
    )
