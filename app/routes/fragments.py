"""
Fragments HTMX — réponses HTML partielles pour les interactions sans rechargement.

Toutes ces routes nécessitent auth. Elles retournent du HTML (pas JSON).
Convention : POST crée, PUT met à jour, DELETE supprime.
"""

import json
import logging
from datetime import date, datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from sqlalchemy import case, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import require_auth
from app.config import settings
from app.db import get_db
from app.llm.proactive import generate_signal  # ask_due_date used via _classify_and_update
from app.memory import record_correction
from app.models import PendingPrompt, Task
from app.routes.conversation import _classify_and_update, _resolve_category
from app.templates_config import templates

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/fragments", dependencies=[Depends(require_auth)])


# ---------------------------------------------------------------------------
# Création de tâche
# ---------------------------------------------------------------------------


@router.post("/tasks", response_class=HTMLResponse)
async def create_task(
    request: Request,
    background_tasks: BackgroundTasks,
    title: str = Form(...),
    description: str = Form(""),
    db: AsyncSession = Depends(get_db),
):
    task = Task(
        title=title,
        description=description or None,
        llm_pending=True,
        status="open",
    )
    db.add(task)
    await db.commit()
    await db.refresh(task)

    background_tasks.add_task(_classify_and_update, task.id, title, description or None)

    return templates.TemplateResponse(
        request,
        "fragments/task_card.html",
        {"task": task, "pending": None, "today": date.today()},
    )


# ---------------------------------------------------------------------------
# Mise à jour d'une tâche
# ---------------------------------------------------------------------------


@router.put("/tasks/{task_id}", response_class=HTMLResponse)
async def update_task(
    request: Request,
    task_id: int,
    title: str = Form(...),
    description: str = Form(""),
    urgency: str = Form("normale"),
    due_date: str = Form(""),
    db: AsyncSession = Depends(get_db),
):
    _VALID_URGENCIES = {"critique", "haute", "normale", "basse"}
    if urgency not in _VALID_URGENCIES:
        raise HTTPException(status_code=422, detail=f"Urgence invalide: {urgency}")

    task = await _get_task_or_404(db, task_id)
    task.title = title
    task.description = description or None
    task.urgency = urgency
    task.due_date = date.fromisoformat(due_date) if due_date else None
    await db.commit()

    return templates.TemplateResponse(
        request,
        "fragments/task_card.html",
        {"task": task, "pending": None, "today": date.today()},
    )


# ---------------------------------------------------------------------------
# Suppression
# ---------------------------------------------------------------------------


@router.delete("/tasks/{task_id}")
async def delete_task(task_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Task).where(Task.id == task_id))
    task = result.scalar_one_or_none()
    if task is not None:
        await db.delete(task)
        await db.commit()
    return Response(status_code=200)


# ---------------------------------------------------------------------------
# Poll statut classification (optimistic UI)
# ---------------------------------------------------------------------------


@router.get("/tasks/{task_id}/status", response_class=HTMLResponse)
async def task_status(
    request: Request,
    task_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Retourne la carte en mode 'classifying' si llm_pending=True,
    ou la carte complète si la classification est terminée.
    HTMX polling s'arrête automatiquement quand la carte sans hx-trigger est retournée.
    """
    task = await _get_task_or_404(db, task_id)
    today = date.today()

    pending_prompt = None
    if not task.llm_pending:
        pp_result = await db.execute(
            select(PendingPrompt).where(
                PendingPrompt.task_id == task_id,
                PendingPrompt.resolved_at.is_(None),
            )
        )
        pending_prompt = pp_result.scalar_one_or_none()

    return templates.TemplateResponse(
        request,
        "fragments/task_card.html",
        {"task": task, "pending": pending_prompt, "today": today},
    )


# ---------------------------------------------------------------------------
# Marquer done
# ---------------------------------------------------------------------------


@router.post("/tasks/{task_id}/done", response_class=HTMLResponse)
async def mark_done(
    request: Request,
    task_id: int,
    db: AsyncSession = Depends(get_db),
):
    task = await _get_task_or_404(db, task_id)
    task.status = "done"
    task.completed_at = datetime.now(tz=timezone.utc)
    await db.commit()

    return templates.TemplateResponse(
        request,
        "fragments/task_card.html",
        {"task": task, "pending": None, "today": date.today()},
    )


# ---------------------------------------------------------------------------
# Undo
# ---------------------------------------------------------------------------


@router.post("/tasks/{task_id}/undo/done", response_class=HTMLResponse)
async def undo_done(
    request: Request,
    task_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Annule un marquage done : remet la tâche en status 'open'."""
    task = await _get_task_or_404(db, task_id)
    if task.status == "done":
        task.status = "open"
        task.completed_at = None
    await db.commit()

    return templates.TemplateResponse(
        request,
        "fragments/task_card.html",
        {"task": task, "pending": None, "today": date.today()},
    )


# ---------------------------------------------------------------------------
# Correction manuelle (mémoire apprenante)
# ---------------------------------------------------------------------------


@router.post("/tasks/{task_id}/correct", response_class=HTMLResponse)
async def correct_task(
    request: Request,
    task_id: int,
    field: str = Form(...),
    new_value: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    _ALLOWED_CORRECTION_FIELDS = {"urgency", "category", "due_date", "context_id"}
    if field not in _ALLOWED_CORRECTION_FIELDS:
        raise HTTPException(status_code=400, detail=f"Champ inconnu: {field}")

    task = await _get_task_or_404(db, task_id)
    old_value = str(getattr(task, field, ""))

    if field == "urgency":
        if new_value not in {"critique", "haute", "normale", "basse"}:
            raise HTTPException(status_code=422, detail=f"Urgence invalide: {new_value}")
        task.urgency = new_value
    elif field == "category":
        task.category = new_value
    elif field == "due_date":
        if new_value:
            try:
                task.due_date = date.fromisoformat(new_value)
            except ValueError:
                raise HTTPException(status_code=422, detail=f"Date invalide: {new_value}")
        else:
            task.due_date = None
    elif field == "context_id":
        if new_value:
            try:
                task.context_id = int(new_value)
            except ValueError:
                raise HTTPException(status_code=422, detail=f"context_id invalide: {new_value}")
        else:
            task.context_id = None

    task.was_corrected = 1
    task.needs_review = 0

    await record_correction(
        db,
        task_id=task.id,
        task_title=task.title,
        task_description=task.description,
        field=field,
        old_value=old_value,
        new_value=new_value,
    )
    await db.commit()

    return templates.TemplateResponse(
        request,
        "fragments/task_card.html",
        {"task": task, "pending": None, "today": date.today()},
    )


# ---------------------------------------------------------------------------
# Réponse à un pending prompt
# ---------------------------------------------------------------------------


@router.post("/tasks/{task_id}/prompt/{prompt_id}/answer", response_class=HTMLResponse)
async def answer_prompt(
    request: Request,
    task_id: int,
    prompt_id: int,
    answer: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    task = await _get_task_or_404(db, task_id)
    prompt_result = await db.execute(
        select(PendingPrompt).where(
            PendingPrompt.id == prompt_id,
            PendingPrompt.task_id == task_id,
        )
    )
    prompt = prompt_result.scalar_one_or_none()
    if prompt is None:
        return HTMLResponse("<div>Prompt introuvable</div>", status_code=404)

    prompt.resolved_at = datetime.now(tz=timezone.utc)
    prompt.resolution = json.dumps({"answer": answer})

    if prompt.kind == "ask_due_date" and answer not in ("no_deadline", ""):
        try:
            task.due_date = date.fromisoformat(answer)
        except ValueError:
            pass  # Valeur libre (ex. "cette semaine") — pas de parsing

    await db.commit()

    return templates.TemplateResponse(
        request,
        "fragments/task_card.html",
        {"task": task, "pending": None, "today": date.today()},
    )


# ---------------------------------------------------------------------------
# Colonne Signal
# ---------------------------------------------------------------------------


@router.get("/signal", response_class=HTMLResponse)
async def signal_column(request: Request, db: AsyncSession = Depends(get_db)):

    urgency_order = case(
        {"critique": 0, "haute": 1, "normale": 2, "basse": 3},
        value=Task.urgency,
        else_=4,
    )
    tasks_result = await db.execute(
        select(Task)
        .where(Task.status.in_(["open", "doing"]))
        .order_by(urgency_order, Task.due_date.nullslast())
        .limit(30)
    )
    tasks = list(tasks_result.scalars().all())

    try:
        priorities = await generate_signal(tasks, capacity_minutes=settings.daily_capacity_minutes)
    except Exception:
        logger.exception("generate_signal échoué")
        priorities = []

    return templates.TemplateResponse(
        request,
        "fragments/signal.html",
        {"priorities": priorities},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _get_task_or_404(db: AsyncSession, task_id: int) -> Task:
    result = await db.execute(select(Task).where(Task.id == task_id))
    task = result.scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail="Tâche introuvable")
    return task
