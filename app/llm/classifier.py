"""
Étage 2 — Classification de tâche.

Orchestre : chargement DB → few-shots → prompt → LLM → fallback inbox.
"""

import logging

from openai.types.chat import ChatCompletionMessageParam
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


async def classify_task(
    title: str,
    description: str | None,
    db: AsyncSession,
    llm: LLMClient | None = None,
) -> tuple[Classification, LLMResult[Classification] | None]:
    """
    Classifie une tâche via LLM.

    Retourne (classification, llm_result).
    - llm_result non-None → succès : persister llm_result.raw_json dans Task.llm_raw_response.
    - llm_result est None → fallback inbox (erreur LLM) : poser needs_review=1 sur la tâche.
    """
    if llm is None:
        llm = get_llm_client()

    categories = await _load_categories(db)
    contexts = await _load_contexts(db)
    few_shots = await select_few_shots(title, description, db)

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": TASK_CLASSIFIER_SYSTEM},
        {
            "role": "user",
            "content": task_classifier_user(title, description, categories, contexts, few_shots),
        },
    ]

    try:
        result = await llm.structured_complete(messages=messages, schema=Classification)
        logger.info(
            "Classification OK — category=%s confidence=%.2f",
            result.data.category,
            result.data.confidence,
        )
        return result.data, result

    except (LLMParseError, LLMTransportError) as e:
        logger.warning("Classification fallback: %s", e)
        fallback = Classification(
            category=None,
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
    notes_by_cat: dict[str, list[str]] = {}
    for note in notes_result.scalars().all():
        if note.category is None:
            logger.warning("CategoryNote id=%s sans catégorie — ignorée", note.id)
            continue
        notes_by_cat.setdefault(note.category, []).append(note.note)

    result: list[CategoryDict] = []
    for cat in categories:
        d: CategoryDict = {"name": cat.name, "description": cat.description}
        if cat.name in notes_by_cat:
            d["notes"] = " | ".join(notes_by_cat[cat.name])
        result.append(d)
    return result


async def _load_contexts(db: AsyncSession) -> list[ContextDict]:
    stmt = await db.execute(
        select(Context).where(Context.archived == 0)
    )
    contexts: list[ContextDict] = []
    for ctx in stmt.scalars().all():
        d: ContextDict = {"id": ctx.id, "name": ctx.name}
        if ctx.kind:
            d["kind"] = ctx.kind
        if ctx.aliases:
            d["aliases"] = ctx.aliases
        contexts.append(d)
    return contexts
