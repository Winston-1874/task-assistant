"""
Étage 2 — Classification de tâche.

Orchestre : chargement DB → few-shots → prompt → LLM → fallback inbox.
"""

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.llm.client import LLMClient, LLMParseError, LLMResult, LLMTransportError, get_llm_client
from app.llm.prompts import (
    TASK_CLASSIFIER_SYSTEM,
    CategoryDict,
    ContextDict,
    task_classifier_user,
)
from app.memory import select_few_shots
from app.models import Category, CategoryNote, Context
from app.schemas import Classification

logger = logging.getLogger(__name__)

_INBOX_FALLBACK = Classification(
    category="inbox",
    urgency="normale",
    confidence=0.0,
    reasoning="Fallback automatique — erreur LLM.",
    needs_due_date=False,
    tags=[],
)


async def classify_task(
    title: str,
    description: str | None,
    db: AsyncSession,
    llm: LLMClient | None = None,
) -> tuple[Classification, LLMResult[Classification] | None]:
    """
    Classifie une tâche via LLM.

    Retourne (classification, llm_result).
    llm_result est None si on est tombé en fallback sans appel réussi
    (persistez llm_result.raw_json dans Task.llm_raw_response quand non-None).
    """
    if llm is None:
        llm = get_llm_client()

    categories = await _load_categories(db)
    contexts = await _load_contexts(db)
    few_shots = await select_few_shots(title, description, db)

    messages = [
        {"role": "system", "content": TASK_CLASSIFIER_SYSTEM},
        {
            "role": "user",
            "content": task_classifier_user(title, description, categories, contexts, few_shots),
        },
    ]

    try:
        result = await llm.structured_complete(messages=messages, schema=Classification)  # type: ignore[arg-type]
        logger.info("Classification OK — category=%s confidence=%.2f", result.data.category, result.data.confidence)
        return result.data, result

    except (LLMParseError, LLMTransportError) as e:
        logger.warning("Classification fallback inbox: %s", e)
        fallback = Classification(
            category="inbox",
            urgency="normale",
            confidence=0.0,
            reasoning=f"Fallback automatique : {type(e).__name__}",
            needs_due_date=False,
            tags=[],
        )
        return fallback, None


# ---------------------------------------------------------------------------
# Chargement DB
# ---------------------------------------------------------------------------


async def _load_categories(db: AsyncSession) -> list[CategoryDict]:
    cats_result = await db.execute(select(Category))
    categories = list(cats_result.scalars().all())

    notes_result = await db.execute(select(CategoryNote))
    notes_by_cat: dict[str | None, list[str]] = {}
    for note in notes_result.scalars().all():
        notes_by_cat.setdefault(note.category, []).append(note.note)

    return [
        CategoryDict(
            name=cat.name,
            description=cat.description,
            **({"notes": " | ".join(notes_by_cat[cat.name])} if cat.name in notes_by_cat else {}),
        )
        for cat in categories
    ]


async def _load_contexts(db: AsyncSession) -> list[ContextDict]:
    result = await db.execute(
        select(Context).where(Context.archived == 0)
    )
    return [
        ContextDict(
            id=ctx.id,
            name=ctx.name,
            **({"kind": ctx.kind} if ctx.kind else {}),
            **({"aliases": ctx.aliases} if ctx.aliases else {}),
        )
        for ctx in result.scalars().all()
    ]
