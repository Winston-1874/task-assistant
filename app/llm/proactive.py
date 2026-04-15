"""
Étage 3 — Dialogues proactifs.

Trois fonctions :
- ask_due_date : PendingPrompt 'ask_due_date' si tâche sans échéance
- check_zombie  : PendingPrompt 'zombie_check' si tâche qui traîne
- generate_signal : liste priorisée pour la colonne Signal (sans DB)

Convention transaction : ask_due_date et check_zombie font un flush() sans commit.
Le caller contrôle la transaction (même convention que record_correction et classifier).

Gestion erreurs :
- LLMParseError → fallback message texte (non critique pour les prompts) ou [] (signal)
- LLMTransportError → propagé dans les trois cas
"""

import logging
from datetime import datetime, timezone

from openai.types.chat import ChatCompletionMessageParam
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.llm.client import LLMClient, LLMParseError, LLMTransportError, get_llm_client
from app.llm.prompts import (
    PROACTIVE_SYSTEM,
    DigestTaskDict,
    TaskSignalDict,
    digest_user,
    proactive_ask_due_date_user,
    proactive_signal_user,
    proactive_zombie_user,
)
from app.models import PendingPrompt, Task
from app.schemas import DigestContent, ProactiveMessage, SignalPriority, SignalResponse

logger = logging.getLogger(__name__)


async def ask_due_date(
    task: Task,
    db: AsyncSession,
    llm: LLMClient | None = None,
) -> PendingPrompt:
    """
    Génère (ou retourne) un PendingPrompt 'ask_due_date' pour une tâche sans échéance.

    Idempotent : si un prompt non résolu du même kind existe déjà pour cette tâche,
    le retourne sans créer de doublon ni appeler le LLM.
    """
    if llm is None:
        llm = get_llm_client()

    existing = await _get_unresolved_prompt(db, task.id, "ask_due_date")
    if existing:
        return existing

    fallback = f'"{task.title}" — pour quand tu le vois bouclé ?'
    message = await _generate_message(proactive_ask_due_date_user(task.title), llm, fallback)
    return await _create_prompt(db, task.id, "ask_due_date", message)


async def check_zombie(
    task: Task,
    db: AsyncSession,
    llm: LLMClient | None = None,
) -> PendingPrompt:
    """
    Génère (ou retourne) un PendingPrompt 'zombie_check' pour une tâche qui traîne.

    Idempotent : même logique que ask_due_date.
    """
    if llm is None:
        llm = get_llm_client()

    existing = await _get_unresolved_prompt(db, task.id, "zombie_check")
    if existing:
        return existing

    days_idle = _days_since(task.touched_at)
    fallback = f'"{task.title}" — tu la fais, tu la délègues ou on l\'annule ?'
    message = await _generate_message(
        proactive_zombie_user(task.title, days_idle, task.postponed_count),
        llm,
        fallback,
    )
    return await _create_prompt(db, task.id, "zombie_check", message)


async def generate_signal(
    tasks: list[Task],
    capacity_minutes: int,
    llm: LLMClient | None = None,
) -> list[SignalPriority]:
    """
    Retourne la liste priorisée des tâches pour la colonne Signal.

    capacity_minutes : valeur lue par le caller (depuis config ou table settings).
    Passé en paramètre pour éviter que la fonction lise settings figé au démarrage.

    - Liste vide → retourne [] sans appel LLM.
    - LLMParseError → [] (non critique, UI affiche colonne vide).
    - LLMTransportError → propagé.
    """
    if not tasks:
        return []

    if llm is None:
        llm = get_llm_client()

    task_signals: list[TaskSignalDict] = [_to_signal_dict(t) for t in tasks]
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": PROACTIVE_SYSTEM},
        {
            "role": "user",
            "content": proactive_signal_user(task_signals, capacity_minutes),
        },
    ]

    try:
        result = await llm.structured_complete(messages=messages, schema=SignalResponse)
        logger.info("Signal généré — %d priorités", len(result.data.priorities))
        return result.data.priorities
    except LLMParseError as e:
        logger.warning("Signal parse error — retour liste vide: %s", e)
        return []


# ---------------------------------------------------------------------------
# Helpers privés
# ---------------------------------------------------------------------------


async def _get_unresolved_prompt(
    db: AsyncSession,
    task_id: int | None,
    kind: str,
) -> PendingPrompt | None:
    if task_id is None:
        return None
    result = await db.execute(
        select(PendingPrompt)
        .where(PendingPrompt.task_id == task_id)
        .where(PendingPrompt.kind == kind)
        .where(PendingPrompt.resolved_at.is_(None))
        .limit(1)
    )
    return result.scalar_one_or_none()


async def _generate_message(
    user_content: str,
    llm: LLMClient,
    fallback: str,
) -> str:
    """
    Appelle le LLM pour générer un message proactif.
    LLMParseError → fallback texte. LLMTransportError → propagé.
    """
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": PROACTIVE_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    try:
        result = await llm.structured_complete(messages=messages, schema=ProactiveMessage)
        return result.data.message
    except LLMParseError as e:
        logger.warning("Proactive message parse error — fallback: %s", e)
        return fallback


async def _create_prompt(
    db: AsyncSession,
    task_id: int | None,
    kind: str,
    message: str,
) -> PendingPrompt:
    prompt = PendingPrompt(task_id=task_id, kind=kind, message=message)
    db.add(prompt)
    await db.flush()
    return prompt


def _days_since(dt: datetime) -> int:
    """
    Nombre de jours entiers écoulés depuis dt (tronqué, pas arrondi).
    Ex : 4j23h → 4. Gère les datetimes naifs (UTC implicite, comme SQLite func.now()).
    Retourne 0 si dt est dans le futur.
    """
    now = datetime.now(tz=timezone.utc)
    aware = dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    return max(0, (now - aware).days)


async def generate_digest(
    today_tasks: list[Task],
    week_tasks: list[Task],
    capacity_minutes: int,
    llm: LLMClient | None = None,
) -> DigestContent:
    """
    Génère le digest matinal.

    LLMParseError → fallback minimal (liste brute). LLMTransportError → propagé.
    """
    if llm is None:
        llm = get_llm_client()

    today_iso = datetime.now(tz=timezone.utc).date().isoformat()
    user_content = digest_user(
        today_tasks=[_to_digest_dict(t) for t in today_tasks],
        week_tasks=[_to_digest_dict(t) for t in week_tasks],
        capacity_minutes=capacity_minutes,
        today_iso=today_iso,
    )
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": PROACTIVE_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    try:
        result = await llm.structured_complete(messages=messages, schema=DigestContent)
        return result.data
    except LLMParseError as e:
        logger.warning("Digest parse error — fallback liste brute: %s", e)
        titles = [t.title for t in (today_tasks + week_tasks)[:5]]
        return DigestContent(
            summary="Digest indisponible — voici les tâches du jour.",
            top_tasks=titles,
        )


def _to_digest_dict(task: Task) -> DigestTaskDict:
    d: DigestTaskDict = {
        "title": task.title,
        "urgency": task.urgency or "normale",
    }
    if task.due_date:
        d["due_date"] = task.due_date.isoformat()
    if task.estimated_minutes:
        d["estimated_minutes"] = task.estimated_minutes
    return d


def _to_signal_dict(task: Task) -> TaskSignalDict:
    if task.urgency is None:
        logger.warning("Task id=%s sans urgency — fallback 'normale' pour signal", task.id)
    d: TaskSignalDict = {
        "title": task.title,
        "urgency": task.urgency or "normale",
    }
    if task.due_date:
        d["due_date"] = task.due_date.isoformat()
    return d
