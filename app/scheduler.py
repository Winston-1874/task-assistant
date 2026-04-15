"""
APScheduler — tâches planifiées.

Trois jobs :
- job_check_zombies   : quotidien 02h00 — crée des PendingPrompts zombie_check
- job_archive_done    : hebdomadaire dimanche 03h00 — archive les tâches done > 90j
- job_generate_digest : quotidien 07h00 — génère le digest matinal (si digest_enabled)

Chaque job crée sa propre session DB via get_db_session().
Les erreurs sont attrapées individuellement pour ne pas crasher le scheduler.
"""

import json
import logging
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from markupsafe import escape
from sqlalchemy import case, select, text

from app.config import settings
from app.db import get_db_session
from app.llm.proactive import check_zombie, generate_digest
from app.models import Digest, Task

logger = logging.getLogger(__name__)

_ARCHIVE_DONE_DAYS = 90
_TZ = "Europe/Brussels"

# Colonnes communes tasks ↔ tasks_archive (sans colonnes spécifiques à chaque table)
_ARCHIVE_COLS = (
    "id, title, description, category, context_id, urgency, due_date, "
    "estimated_minutes, actual_minutes, tags, status, waiting_reason, "
    "created_at, completed_at, touched_at, llm_raw_response, llm_confidence, "
    "llm_reasoning, was_corrected, postponed_count, needs_review"
)


# ---------------------------------------------------------------------------
# Job : détection zombies
# ---------------------------------------------------------------------------


async def job_check_zombies() -> None:
    """
    Cherche les tâches ouvertes qui traînent et crée des PendingPrompts zombie_check.
    Critères : postponed_count >= 3 OU touched_at < now - zombie_threshold_days.
    Idempotent grâce à check_zombie() qui ne crée pas de doublon si prompt existant.
    """
    threshold = datetime.now(tz=timezone.utc) - timedelta(days=settings.zombie_threshold_days)

    async with get_db_session() as db:
        result = await db.execute(
            select(Task).where(
                Task.status == "open",
            )
        )
        candidates = result.scalars().all()

        zombies = [
            t for t in candidates
            if (t.postponed_count or 0) >= 3
            or (
                t.touched_at is not None
                and (
                    t.touched_at if t.touched_at.tzinfo is not None
                    else t.touched_at.replace(tzinfo=timezone.utc)
                ) < threshold
            )
        ]

        if not zombies:
            logger.info("job_check_zombies : aucun zombie détecté")
            return

        logger.info("job_check_zombies : %d zombie(s) détectés", len(zombies))
        created = 0
        for task in zombies:
            try:
                prompt = await check_zombie(task, db)
                if prompt.id is None or prompt.resolved_at is not None:
                    continue  # déjà existant ou déjà résolu
                created += 1
            except Exception:
                logger.exception("job_check_zombies : erreur sur task id=%s", task.id)

        try:
            await db.commit()
            logger.info("job_check_zombies : %d prompt(s) créés", created)
        except Exception:
            logger.exception("job_check_zombies : échec du commit — rollback implicite")
            raise


# ---------------------------------------------------------------------------
# Job : archivage des tâches done
# ---------------------------------------------------------------------------


async def job_archive_done() -> None:
    """
    Déplace les tâches done depuis plus de ARCHIVE_DONE_DAYS jours vers tasks_archive.
    Utilise du SQL brut (tasks_archive n'a pas de modèle ORM).
    """
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=_ARCHIVE_DONE_DAYS)

    async with get_db_session() as db:
        insert_sql = text(f"""
            INSERT OR IGNORE INTO tasks_archive
            ({_ARCHIVE_COLS}, archived_at)
            SELECT {_ARCHIVE_COLS}, CURRENT_TIMESTAMP
            FROM tasks
            WHERE status = 'done'
            AND completed_at < :cutoff
        """)
        delete_sql = text("""
            DELETE FROM tasks
            WHERE status = 'done'
            AND completed_at < :cutoff
        """)

        await db.execute(insert_sql, {"cutoff": cutoff.isoformat()})
        delete_result = await db.execute(delete_sql, {"cutoff": cutoff.isoformat()})
        await db.commit()

        archived = delete_result.rowcount
        logger.info("job_archive_done : %d tâche(s) archivée(s)", archived)


# ---------------------------------------------------------------------------
# Job : digest matinal
# ---------------------------------------------------------------------------


async def job_generate_digest() -> None:
    """
    Génère le digest matinal via LLM et l'enregistre dans la table digests.
    Upsert sur la date du jour (idempotent si relancé).
    Skippé si settings.digest_enabled est False.
    """
    if not settings.digest_enabled:
        logger.debug("job_generate_digest : digest désactivé — skip")
        return

    today = datetime.now(tz=timezone.utc).date()
    week_end = today + timedelta(days=7)

    _urgency_order = case(
        {"critique": 1, "haute": 2, "normale": 3, "basse": 4},
        value=Task.urgency,
        else_=5,
    )

    async with get_db_session() as db:
        today_result = await db.execute(
            select(Task).where(
                Task.status.in_(["open", "doing"]),
                Task.due_date <= today,
            ).order_by(_urgency_order)
        )
        today_tasks = list(today_result.scalars().all())

        week_result = await db.execute(
            select(Task).where(
                Task.status.in_(["open", "doing"]),
                Task.due_date > today,
                Task.due_date <= week_end,
            ).order_by(Task.due_date)
        )
        week_tasks = list(week_result.scalars().all())

        try:
            content = await generate_digest(
                today_tasks=today_tasks,
                week_tasks=week_tasks,
                capacity_minutes=settings.daily_capacity_minutes,
            )
        except Exception:
            logger.exception("job_generate_digest : erreur LLM")
            return

        content_html = _digest_to_html(content)
        content_text = _digest_to_text(content)
        task_ids = json.dumps([t.id for t in today_tasks + week_tasks])

        # Upsert : remplace si un digest existe déjà pour cette date
        existing = await db.execute(select(Digest).where(Digest.date == today))
        digest = existing.scalar_one_or_none()
        if digest is None:
            digest = Digest(date=today, content_html=content_html, content_text=content_text, task_ids=task_ids)
            db.add(digest)
        else:
            digest.content_html = content_html
            digest.content_text = content_text
            digest.task_ids = task_ids

        await db.commit()
        logger.info("job_generate_digest : digest du %s enregistré", today)


def _digest_to_html(content) -> str:
    lines = [f"<p>{escape(content.summary)}</p>"]
    if content.top_tasks:
        items = "".join(f"<li>{escape(task)}</li>" for task in content.top_tasks)
        lines.append(f"<ul>{items}</ul>")
    if content.alert:
        lines.append(f'<p class="digest-alert">{escape(content.alert)}</p>')
    return "\n".join(lines)


def _digest_to_text(content) -> str:
    parts = [content.summary]
    if content.top_tasks:
        parts.append("\nPriorités :")
        parts.extend(f"  • {t}" for t in content.top_tasks)
    if content.alert:
        parts.append(f"\n⚠️ {content.alert}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Initialisation du scheduler
# ---------------------------------------------------------------------------


def create_scheduler() -> AsyncIOScheduler:
    """Crée et configure le scheduler. Ne le démarre pas."""
    scheduler = AsyncIOScheduler(timezone=_TZ)

    scheduler.add_job(
        job_check_zombies,
        CronTrigger(hour=2, minute=0, timezone=_TZ),
        id="check_zombies",
        name="Détection zombies",
        replace_existing=True,
        misfire_grace_time=3600,  # 1h de grâce si le process était arrêté
    )
    scheduler.add_job(
        job_archive_done,
        CronTrigger(day_of_week="sun", hour=3, minute=0, timezone=_TZ),
        id="archive_done",
        name="Archivage tâches done",
        replace_existing=True,
        misfire_grace_time=3600,
    )
    scheduler.add_job(
        job_generate_digest,
        CronTrigger(hour=7, minute=0, timezone=_TZ),
        id="generate_digest",
        name="Digest matinal",
        replace_existing=True,
        misfire_grace_time=1800,  # 30min de grâce pour le digest
    )

    return scheduler
