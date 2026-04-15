"""Pages principales — index, week, waiting, inbox."""

from datetime import date, timedelta

from fastapi import APIRouter, Depends, Request
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import require_auth
from app.db import get_db
from app.models import PendingPrompt, Task
from app.templates_config import templates

router = APIRouter(dependencies=[Depends(require_auth)])


@router.get("/")
async def index(request: Request, db: AsyncSession = Depends(get_db)):
    today = date.today()
    week_end = today + timedelta(days=7)

    sections = {
        "now": await _query(
            db,
            select(Task).where(
                Task.status == "open",
                or_(Task.due_date <= today, Task.urgency == "critique"),
            ).order_by(Task.urgency, Task.due_date.nullslast()),
        ),
        "week": await _query(
            db,
            select(Task).where(
                Task.status == "open",
                Task.due_date > today,
                Task.due_date <= week_end,
                Task.urgency != "critique",
            ).order_by(Task.due_date),
        ),
        "waiting": await _query(
            db,
            select(Task).where(Task.status == "waiting").order_by(Task.touched_at.desc()),
        ),
        "veille": await _query(
            db,
            select(Task).where(
                Task.status == "open",
                Task.urgency == "basse",
                Task.due_date.is_(None),
                Task.needs_review == 0,
                Task.llm_confidence >= 0.7,
            ).order_by(Task.created_at.desc()),
        ),
        "inbox": await _query(
            db,
            select(Task).where(
                Task.status.in_(["open", "doing"]),
                Task.was_corrected == 0,
                or_(Task.needs_review == 1, Task.llm_confidence < 0.7),
            ).order_by(Task.created_at.desc()),
        ),
    }
    pending = await _pending_prompts(db, sections)
    return templates.TemplateResponse(
        request,
        "index.html",
        {"sections": sections, "pending": pending, "today": today},
    )


@router.get("/week")
async def week_view(request: Request, db: AsyncSession = Depends(get_db)):
    today = date.today()
    week_end = today + timedelta(days=7)
    tasks = await _query(
        db,
        select(Task)
        .where(Task.status.in_(["open", "doing"]), Task.due_date <= week_end)
        .order_by(Task.due_date.nullslast(), Task.urgency),
    )
    return templates.TemplateResponse(
        request,
        "index.html",
        {"sections": {"now": [], "week": tasks, "waiting": [], "veille": [], "inbox": []}, "pending": {}, "today": today},
    )


@router.get("/waiting")
async def waiting_view(request: Request, db: AsyncSession = Depends(get_db)):
    tasks = await _query(
        db,
        select(Task).where(Task.status == "waiting").order_by(Task.touched_at.desc()),
    )
    return templates.TemplateResponse(
        request,
        "index.html",
        {"sections": {"now": [], "week": [], "waiting": tasks, "veille": [], "inbox": []}, "pending": {}, "today": date.today()},
    )


@router.get("/inbox")
async def inbox_view(request: Request, db: AsyncSession = Depends(get_db)):
    tasks = await _query(
        db,
        select(Task)
        .where(or_(Task.needs_review == 1, Task.llm_confidence < 0.7))
        .order_by(Task.created_at.desc()),
    )
    return templates.TemplateResponse(
        request,
        "index.html",
        {"sections": {"now": [], "week": [], "waiting": [], "veille": [], "inbox": tasks}, "pending": {}, "today": date.today()},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _query(db: AsyncSession, stmt) -> list[Task]:
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def _pending_prompts(
    db: AsyncSession,
    sections: dict[str, list[Task]],
) -> dict[int, PendingPrompt]:
    """Retourne les pending prompts non résolus indexés par task_id."""
    all_ids = [t.id for tasks in sections.values() for t in tasks if t.id is not None]
    if not all_ids:
        return {}
    result = await db.execute(
        select(PendingPrompt)
        .where(PendingPrompt.task_id.in_(all_ids))
        .where(PendingPrompt.resolved_at.is_(None))
    )
    return {p.task_id: p for p in result.scalars().all() if p.task_id is not None}
