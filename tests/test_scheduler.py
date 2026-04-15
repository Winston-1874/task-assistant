"""
Tests des jobs APScheduler.

Les jobs sont testés directement (sans démarrer le scheduler) avec :
- DB SQLite in-memory
- LLM mocké via monkeypatch / MagicMock
"""

import json
from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.db import Base, get_db_session
from app.models import Digest, PendingPrompt, Task
from app.schemas import DigestContent
from app.scheduler import (
    _ARCHIVE_DONE_DAYS,
    job_archive_done,
    job_check_zombies,
    job_generate_digest,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def db_engine():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # tasks_archive n'est pas dans les modèles ORM — créer manuellement
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS tasks_archive (
              id INTEGER NOT NULL,
              title TEXT NOT NULL,
              description TEXT,
              category TEXT,
              context_id INTEGER,
              urgency TEXT,
              due_date DATE,
              estimated_minutes INTEGER,
              actual_minutes INTEGER,
              tags TEXT,
              status TEXT NOT NULL,
              waiting_reason TEXT,
              created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
              completed_at DATETIME,
              touched_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
              llm_raw_response TEXT,
              llm_confidence REAL,
              llm_reasoning TEXT,
              was_corrected INTEGER NOT NULL DEFAULT 0,
              postponed_count INTEGER NOT NULL DEFAULT 0,
              needs_review INTEGER NOT NULL DEFAULT 0,
              archived_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY (id)
            )
        """))
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
async def session_factory(db_engine):
    return async_sessionmaker(db_engine, expire_on_commit=False)


@pytest.fixture
async def db(session_factory) -> AsyncSession:
    session = session_factory()
    try:
        yield session
    finally:
        await session.close()


async def _make_task(db: AsyncSession, **kwargs) -> Task:
    defaults = {
        "title": "Tâche test",
        "urgency": "normale",
        "status": "open",
        "touched_at": datetime.now(tz=timezone.utc),
        "postponed_count": 0,
        "was_corrected": 0,
    }
    defaults.update(kwargs)
    task = Task(**defaults)
    db.add(task)
    await db.flush()
    return task


# ---------------------------------------------------------------------------
# Tests job_check_zombies
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_zombies_postponed_count(session_factory):
    """Tâche avec postponed_count >= 3 → prompt zombie créé."""
    async with session_factory() as db:
        task = await _make_task(db, postponed_count=3)
        await db.commit()

    mock_prompt = MagicMock()
    mock_prompt.id = 1
    mock_prompt.resolved_at = None

    with (
        patch("app.scheduler.get_db_session") as mock_ctx,
        patch("app.scheduler.check_zombie", new_callable=AsyncMock) as mock_zombie,
    ):
        mock_ctx.return_value.__aenter__ = AsyncMock(return_value=session_factory())
        mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_zombie.return_value = mock_prompt

        async with session_factory() as db:
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=db)

            await job_check_zombies()

        mock_zombie.assert_called_once()


@pytest.mark.asyncio
async def test_check_zombies_idle_too_long(session_factory):
    """Tâche non touchée depuis > zombie_threshold_days → candidate zombie."""
    old_touch = datetime.now(tz=timezone.utc) - timedelta(days=22)

    async with session_factory() as db:
        task = await _make_task(db, touched_at=old_touch)
        await db.commit()
        task_id = task.id

    with (
        patch("app.scheduler.get_db_session") as mock_ctx,
        patch("app.scheduler.check_zombie", new_callable=AsyncMock) as mock_zombie,
    ):
        mock_prompt = MagicMock()
        mock_prompt.id = 1
        mock_prompt.resolved_at = None
        mock_zombie.return_value = mock_prompt

        async with session_factory() as db:
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=db)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            await job_check_zombies()

        mock_zombie.assert_called_once()


@pytest.mark.asyncio
async def test_check_zombies_recent_task_ignored(session_factory):
    """Tâche récente + postponed_count=0 → pas de prompt zombie."""
    async with session_factory() as db:
        await _make_task(db, touched_at=datetime.now(tz=timezone.utc), postponed_count=0)
        await db.commit()

    with (
        patch("app.scheduler.get_db_session") as mock_ctx,
        patch("app.scheduler.check_zombie", new_callable=AsyncMock) as mock_zombie,
    ):
        async with session_factory() as db:
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=db)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            await job_check_zombies()

        mock_zombie.assert_not_called()


@pytest.mark.asyncio
async def test_check_zombies_done_task_ignored(session_factory):
    """Tâche done avec postponed_count >= 3 → ignorée (filtre status='open')."""
    async with session_factory() as db:
        await _make_task(db, status="done", postponed_count=5)
        await db.commit()

    with (
        patch("app.scheduler.get_db_session") as mock_ctx,
        patch("app.scheduler.check_zombie", new_callable=AsyncMock) as mock_zombie,
    ):
        async with session_factory() as db:
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=db)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            await job_check_zombies()

        mock_zombie.assert_not_called()


@pytest.mark.asyncio
async def test_check_zombies_partial_llm_error_continues(session_factory):
    """Erreur LLM sur tâche 1 → tâche 2 crée quand même un prompt (boucle continue)."""
    async with session_factory() as db:
        await _make_task(db, postponed_count=3, title="Tâche A")
        await _make_task(db, postponed_count=3, title="Tâche B")
        await db.commit()

    mock_prompt = MagicMock()
    mock_prompt.id = 1
    mock_prompt.resolved_at = None

    with (
        patch("app.scheduler.get_db_session") as mock_ctx,
        patch("app.scheduler.check_zombie", new_callable=AsyncMock) as mock_zombie,
    ):
        # Première tâche lève une exception, deuxième réussit
        mock_zombie.side_effect = [Exception("LLM mort"), mock_prompt]

        async with session_factory() as db:
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=db)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            await job_check_zombies()

        assert mock_zombie.call_count == 2, "check_zombie doit être appelé pour les deux tâches"


# ---------------------------------------------------------------------------
# Tests job_archive_done
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_archive_done_moves_old_tasks(session_factory):
    """Tâche done > 90j → présente dans tasks_archive, absente de tasks."""
    old_completed = datetime.now(tz=timezone.utc) - timedelta(days=_ARCHIVE_DONE_DAYS + 1)

    async with session_factory() as db:
        task = await _make_task(db, status="done", completed_at=old_completed)
        await db.commit()
        task_id = task.id

    with patch("app.scheduler.get_db_session") as mock_ctx:
        async with session_factory() as db:
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=db)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            await job_archive_done()

        async with session_factory() as db:
            in_tasks = (await db.execute(select(Task).where(Task.id == task_id))).scalar_one_or_none()
            in_archive = (await db.execute(text("SELECT id FROM tasks_archive WHERE id = :id"), {"id": task_id})).fetchone()

        assert in_tasks is None, "Tâche doit être supprimée de tasks"
        assert in_archive is not None, "Tâche doit être dans tasks_archive"


@pytest.mark.asyncio
async def test_archive_done_keeps_recent_tasks(session_factory):
    """Tâche done < 90j → reste dans tasks."""
    recent_completed = datetime.now(tz=timezone.utc) - timedelta(days=30)

    async with session_factory() as db:
        task = await _make_task(db, status="done", completed_at=recent_completed)
        await db.commit()
        task_id = task.id

    with patch("app.scheduler.get_db_session") as mock_ctx:
        async with session_factory() as db:
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=db)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            await job_archive_done()

        async with session_factory() as db:
            in_tasks = (await db.execute(select(Task).where(Task.id == task_id))).scalar_one_or_none()

        assert in_tasks is not None, "Tâche récente doit rester dans tasks"


@pytest.mark.asyncio
async def test_archive_done_idempotent(session_factory):
    """INSERT OR IGNORE : pré-insertion dans tasks_archive → pas de doublon sur second passage."""
    old_completed = datetime.now(tz=timezone.utc) - timedelta(days=_ARCHIVE_DONE_DAYS + 1)

    async with session_factory() as db:
        task = await _make_task(db, status="done", completed_at=old_completed)
        await db.commit()
        task_id = task.id
        # Pré-insérer dans tasks_archive pour simuler une archive partielle
        await db.execute(text("""
            INSERT OR IGNORE INTO tasks_archive
            (id, title, urgency, status, created_at, touched_at, was_corrected, postponed_count, needs_review, archived_at)
            VALUES (:id, 'Tâche test', 'normale', 'done', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 0, 0, 0, CURRENT_TIMESTAMP)
        """), {"id": task_id})
        await db.commit()

    with patch("app.scheduler.get_db_session") as mock_ctx:
        async with session_factory() as db:
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=db)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            # La tâche existe dans tasks ET tasks_archive → INSERT OR IGNORE ne duplique pas
            await job_archive_done()

        async with session_factory() as db:
            rows = (await db.execute(text("SELECT COUNT(*) FROM tasks_archive WHERE id = :id"), {"id": task_id})).scalar()
        assert rows == 1


# ---------------------------------------------------------------------------
# Tests job_generate_digest
# ---------------------------------------------------------------------------


def _make_digest_content(**kwargs) -> DigestContent:
    defaults = {
        "summary": "Journée chargée avec 3 dossiers urgents.",
        "top_tasks": ["TVA AGRIWAN", "Bilan NewCo"],
        "alert": None,
    }
    defaults.update(kwargs)
    return DigestContent(**defaults)


@pytest.mark.asyncio
async def test_generate_digest_creates_record(session_factory):
    """job_generate_digest crée un Digest pour aujourd'hui."""
    today = date.today()

    with (
        patch("app.scheduler.get_db_session") as mock_ctx,
        patch("app.scheduler.generate_digest", new_callable=AsyncMock) as mock_gen,
    ):
        mock_gen.return_value = _make_digest_content()

        async with session_factory() as db:
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=db)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            await job_generate_digest()

        async with session_factory() as db:
            digest = (await db.execute(select(Digest).where(Digest.date == today))).scalar_one_or_none()

        assert digest is not None
        assert "chargée" in digest.content_html
        assert "TVA AGRIWAN" in digest.content_html


@pytest.mark.asyncio
async def test_generate_digest_upserts(session_factory):
    """Relancer job_generate_digest → met à jour le digest existant, pas de doublon."""
    today = date.today()

    with (
        patch("app.scheduler.get_db_session") as mock_ctx,
        patch("app.scheduler.generate_digest", new_callable=AsyncMock) as mock_gen,
    ):
        mock_gen.return_value = _make_digest_content(summary="Première version.")
        async with session_factory() as db:
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=db)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            await job_generate_digest()

        mock_gen.return_value = _make_digest_content(summary="Deuxième version.")
        async with session_factory() as db:
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=db)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            await job_generate_digest()

        async with session_factory() as db:
            digests = (await db.execute(select(Digest).where(Digest.date == today))).scalars().all()

        assert len(digests) == 1
        assert "Deuxième" in digests[0].content_text


@pytest.mark.asyncio
async def test_generate_digest_skipped_when_disabled():
    """digest_enabled=False → job ne crée rien et ne call pas le LLM."""
    with (
        patch("app.scheduler.settings") as mock_settings,
        patch("app.scheduler.generate_digest", new_callable=AsyncMock) as mock_gen,
        patch("app.scheduler.get_db_session") as mock_ctx,
    ):
        mock_settings.digest_enabled = False
        await job_generate_digest()
        mock_gen.assert_not_called()
        mock_ctx.assert_not_called()


@pytest.mark.asyncio
async def test_generate_digest_llm_error_does_not_crash(session_factory):
    """Erreur LLM dans generate_digest → job log et retourne sans planter."""
    with (
        patch("app.scheduler.get_db_session") as mock_ctx,
        patch("app.scheduler.generate_digest", new_callable=AsyncMock) as mock_gen,
    ):
        mock_gen.side_effect = Exception("Réseau LLM mort")

        async with session_factory() as db:
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=db)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            # Ne doit pas lever
            await job_generate_digest()


# ---------------------------------------------------------------------------
# Tests create_scheduler
# ---------------------------------------------------------------------------


def test_create_scheduler_has_three_jobs():
    from app.scheduler import create_scheduler
    scheduler = create_scheduler()
    job_ids = {j.id for j in scheduler.get_jobs()}
    assert "check_zombies" in job_ids
    assert "archive_done" in job_ids
    assert "generate_digest" in job_ids
