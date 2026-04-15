"""
Route conversationnelle — POST /conversation/parse.

Reçoit un message libre, passe par le routeur d'intention, puis exécute l'action.
Retourne un fragment HTML inséré dans le toast de réponse.
"""

import html
import json
import logging
from datetime import date

from fastapi import APIRouter, BackgroundTasks, Depends, Form, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import require_auth
from app.db import get_db, get_db_session
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
    background_tasks: BackgroundTasks,
    message: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    try:
        intent = await route_intent(message)
    except LLMTransportError as e:
        logger.error("Réseau LLM mort lors du routage: %s", e)
        return _error_html("Service LLM indisponible. Réessaie dans quelques instants.")

    if intent.kind == "new_task":
        return await _handle_new_task(request, background_tasks, message, db)

    labels = {
        "command": "Commande détectée — fonctionnalité à venir.",
        "query": "Question sur le backlog — fonctionnalité à venir.",
        "update_context": "Info contexte notée — fonctionnalité à venir.",
    }
    text = html.escape(labels.get(intent.kind, "Message reçu."))
    return HTMLResponse(f'<div class="toast toast-info">{text}</div>')


async def _handle_new_task(
    request: Request,
    background_tasks: BackgroundTasks,
    message: str,
    db: AsyncSession,
) -> HTMLResponse:
    task = Task(
        title=message[:200],
        llm_pending=True,
        status="open",
    )
    db.add(task)
    await db.commit()
    await db.refresh(task)

    background_tasks.add_task(_classify_and_update, task.id, message, None)

    return templates.TemplateResponse(
        request,
        "fragments/task_card.html",
        {"task": task, "pending": None, "today": date.today()},
    )


async def _classify_and_update(task_id: int, title: str, description: str | None) -> None:
    """Classifie la tâche en arrière-plan et met à jour la DB."""
    async with get_db_session() as db:
        result = await db.execute(select(Task).where(Task.id == task_id))
        task = result.scalar_one_or_none()
        if task is None:
            logger.warning("_classify_and_update: tâche %d introuvable", task_id)
            return

        try:
            classification, llm_result = await classify_task(title, description, db)
        except Exception as e:
            logger.error("_classify_and_update: erreur classification tâche %d: %s", task_id, e)
            task.llm_pending = False
            task.needs_review = 1
            await db.commit()
            return

        resolved_category = _resolve_category(classification.category)
        task.category = resolved_category
        task.context_id = classification.context_id
        task.urgency = classification.urgency
        task.due_date = classification.due_date
        task.estimated_minutes = classification.estimated_minutes
        task.tags = json.dumps(classification.tags) if classification.tags else None
        task.needs_review = 1 if (classification.confidence < 0.7 or resolved_category is None) else 0
        task.llm_confidence = classification.confidence
        task.llm_reasoning = classification.reasoning
        task.llm_raw_response = llm_result.raw_json if llm_result else None
        task.llm_pending = False

        if classification.needs_due_date and task.due_date is None:
            await ask_due_date(task, db)

        await db.commit()


def _resolve_category(category: str | None) -> str | None:
    """Retourne None si la catégorie est un préfixe NEW: (LLM hallucination) ou déjà None."""
    if category and category.startswith("NEW:"):
        return None
    return category


def _error_html(message: str) -> HTMLResponse:
    return HTMLResponse(
        f'<div class="toast toast-error" role="alert">{html.escape(message)}</div>'
    )
