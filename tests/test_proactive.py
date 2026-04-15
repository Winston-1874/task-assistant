"""Tests de app/llm/proactive.py — mock LLM + DB en mémoire."""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db import Base
from app.llm.client import LLMParseError, LLMResult, LLMTransportError
from app.llm.proactive import ask_due_date, check_zombie, generate_signal, _days_since
from app.models import PendingPrompt, Task
from app.schemas import ProactiveMessage, SignalPriority, SignalResponse

_CAPACITY = 420  # minutes/jour utilisé dans tous les tests generate_signal


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def db() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    # FK enforcement : détecte les violations de contrainte que SQLite ignore par défaut
    @event.listens_for(engine.sync_engine, "connect")
    def set_fk(conn, _):  # type: ignore[misc]
        conn.execute("PRAGMA foreign_keys=ON")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    session = session_factory()
    try:
        yield session
    finally:
        try:
            await session.close()
        finally:
            await engine.dispose()


async def _persist_task(db: AsyncSession, **kwargs: object) -> Task:
    """
    Insère une Task en DB et retourne l'objet persisté.
    Tous les attributs sont fournis explicitement pour éviter les lazy-loads
    sur server_default après flush (MissingGreenlet en contexte async).
    """
    defaults: dict[str, object] = {
        "title": "TVA AGRIWAN",
        "urgency": "haute",
        "postponed_count": 0,
        "status": "open",
        "touched_at": datetime.now(tz=timezone.utc),
    }
    defaults.update(kwargs)
    task = Task(**defaults)  # type: ignore[arg-type]
    db.add(task)
    await db.flush()
    return task


def make_task(
    title: str = "TVA AGRIWAN",
    urgency: str | None = "haute",
    postponed_count: int = 0,
    touched_at: datetime | None = None,
) -> Task:
    """Tâche non persistée — pour generate_signal (pas de DB) et tests unitaires."""
    task = Task()
    task.title = title
    task.urgency = urgency
    task.postponed_count = postponed_count
    task.touched_at = touched_at or datetime.now(tz=timezone.utc)
    task.status = "open"
    task.due_date = None
    return task


# ---------------------------------------------------------------------------
# Helpers LLM mock — séparés par schema pour éviter la confusion de types
# ---------------------------------------------------------------------------


def make_message_result(message: str = "Pour quand tu vois ça bouclé ?") -> LLMResult[ProactiveMessage]:
    data = ProactiveMessage(message=message)
    return LLMResult(
        data=data,
        raw_json=json.dumps({"message": message}),
        prompt_tokens=50,
        completion_tokens=20,
        prompt_version="1.0.0",
        model="google/gemini-2.5-flash",
    )


def make_signal_result(priorities: list[dict] | None = None) -> LLMResult[SignalResponse]:
    # Utilise `is not None` et non `or` : une liste vide est un cas de test valide
    prios = priorities if priorities is not None else [
        {"task_title": "TVA AGRIWAN", "reason": "Deadline aujourd'hui"}
    ]
    data = SignalResponse(priorities=[SignalPriority(**p) for p in prios])
    return LLMResult(
        data=data,
        raw_json=json.dumps({"priorities": prios}),
        prompt_tokens=100,
        completion_tokens=60,
        prompt_version="1.0.0",
        model="google/gemini-2.5-flash",
    )


def make_mock_llm_message(
    result: LLMResult[ProactiveMessage] | None = None,
    raises: Exception | None = None,
) -> MagicMock:
    llm = MagicMock()
    if raises:
        llm.structured_complete = AsyncMock(side_effect=raises)
    else:
        llm.structured_complete = AsyncMock(return_value=result or make_message_result())
    return llm


def make_mock_llm_signal(
    result: LLMResult[SignalResponse] | None = None,
    raises: Exception | None = None,
) -> MagicMock:
    llm = MagicMock()
    if raises:
        llm.structured_complete = AsyncMock(side_effect=raises)
    else:
        llm.structured_complete = AsyncMock(return_value=result or make_signal_result())
    return llm


# ---------------------------------------------------------------------------
# Tests ask_due_date
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ask_due_date_creates_prompt(db: AsyncSession) -> None:
    task = await _persist_task(db)
    llm = make_mock_llm_message()
    prompt = await ask_due_date(task, db, llm=llm)
    await db.commit()

    assert prompt.id is not None
    assert prompt.kind == "ask_due_date"
    assert prompt.task_id == task.id
    assert prompt.message
    assert prompt.resolved_at is None


@pytest.mark.asyncio
async def test_ask_due_date_uses_llm_message(db: AsyncSession) -> None:
    task = await _persist_task(db)
    llm = make_mock_llm_message(result=make_message_result("Pour quand tu le vois bouclé ?"))
    prompt = await ask_due_date(task, db, llm=llm)

    assert prompt.message == "Pour quand tu le vois bouclé ?"


@pytest.mark.asyncio
async def test_ask_due_date_idempotent_after_commit(db: AsyncSession) -> None:
    """Deux appels séparés par un commit → même PendingPrompt, LLM appelé 1 seule fois."""
    task = await _persist_task(db)
    llm = make_mock_llm_message()

    prompt1 = await ask_due_date(task, db, llm=llm)
    await db.commit()

    prompt2 = await ask_due_date(task, db, llm=llm)

    assert prompt1.id == prompt2.id
    assert llm.structured_complete.call_count == 1


@pytest.mark.asyncio
async def test_ask_due_date_idempotent_same_session(db: AsyncSession) -> None:
    """Deux appels dans la même transaction (sans commit) → idempotent via autoflush."""
    task = await _persist_task(db)
    llm = make_mock_llm_message()

    prompt1 = await ask_due_date(task, db, llm=llm)
    # PAS de commit — autoflush SQLAlchemy rend le prompt visible au SELECT suivant
    prompt2 = await ask_due_date(task, db, llm=llm)

    assert prompt1.id == prompt2.id
    assert llm.structured_complete.call_count == 1


@pytest.mark.asyncio
async def test_ask_due_date_parse_error_uses_fallback(db: AsyncSession) -> None:
    task = await _persist_task(db, title="Bilan NewCo")
    llm = make_mock_llm_message(raises=LLMParseError("JSON invalide"))
    prompt = await ask_due_date(task, db, llm=llm)

    assert prompt.kind == "ask_due_date"
    assert "Bilan NewCo" in prompt.message


@pytest.mark.asyncio
async def test_ask_due_date_transport_error_propagates(db: AsyncSession) -> None:
    task = await _persist_task(db)
    llm = make_mock_llm_message(raises=LLMTransportError("Réseau mort"))

    with pytest.raises(LLMTransportError):
        await ask_due_date(task, db, llm=llm)


@pytest.mark.asyncio
async def test_ask_due_date_does_not_commit(db: AsyncSession) -> None:
    """flush() sans commit — rollback doit effacer le prompt."""
    task = await _persist_task(db)
    await db.commit()  # commit la tâche, pas le prompt

    llm = make_mock_llm_message()
    await ask_due_date(task, db, llm=llm)
    await db.rollback()

    from sqlalchemy import select as sa_select
    result = await db.execute(sa_select(PendingPrompt).where(PendingPrompt.kind == "ask_due_date"))
    assert result.scalar_one_or_none() is None


@pytest.mark.asyncio
async def test_ask_due_date_task_id_none_creates_orphan_prompt(db: AsyncSession) -> None:
    """task.id=None → PendingPrompt sans FK (task_id=None, autorisé par le modèle)."""
    task = make_task()
    task.id = None  # type: ignore[assignment]
    llm = make_mock_llm_message()
    prompt = await ask_due_date(task, db, llm=llm)

    assert prompt.task_id is None
    assert prompt.kind == "ask_due_date"


# ---------------------------------------------------------------------------
# Tests check_zombie
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_zombie_creates_prompt(db: AsyncSession) -> None:
    task = await _persist_task(
        db,
        postponed_count=3,
        touched_at=datetime.now(tz=timezone.utc) - timedelta(days=25),
    )
    llm = make_mock_llm_message()
    prompt = await check_zombie(task, db, llm=llm)
    await db.commit()

    assert prompt.id is not None
    assert prompt.kind == "zombie_check"
    assert prompt.task_id == task.id
    assert prompt.message


@pytest.mark.asyncio
async def test_check_zombie_idempotent(db: AsyncSession) -> None:
    task = await _persist_task(db, postponed_count=3)
    llm = make_mock_llm_message()

    prompt1 = await check_zombie(task, db, llm=llm)
    await db.commit()

    prompt2 = await check_zombie(task, db, llm=llm)

    assert prompt1.id == prompt2.id
    assert llm.structured_complete.call_count == 1


@pytest.mark.asyncio
async def test_check_zombie_parse_error_uses_fallback(db: AsyncSession) -> None:
    task = await _persist_task(db, title="Relance Dupont", postponed_count=2)
    llm = make_mock_llm_message(raises=LLMParseError("bad"))
    prompt = await check_zombie(task, db, llm=llm)

    assert "Relance Dupont" in prompt.message


@pytest.mark.asyncio
async def test_check_zombie_transport_error_propagates(db: AsyncSession) -> None:
    task = await _persist_task(db, postponed_count=3)
    llm = make_mock_llm_message(raises=LLMTransportError("Timeout"))

    with pytest.raises(LLMTransportError):
        await check_zombie(task, db, llm=llm)


@pytest.mark.asyncio
async def test_ask_due_date_and_zombie_independent(db: AsyncSession) -> None:
    """ask_due_date et zombie_check sont deux kinds distincts — pas de confusion."""
    task = await _persist_task(db)
    llm = make_mock_llm_message()

    prompt_due = await ask_due_date(task, db, llm=llm)
    await db.commit()

    prompt_zombie = await check_zombie(task, db, llm=llm)
    await db.commit()

    assert prompt_due.id != prompt_zombie.id
    assert prompt_due.kind == "ask_due_date"
    assert prompt_zombie.kind == "zombie_check"


# ---------------------------------------------------------------------------
# Tests generate_signal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_signal_returns_priorities() -> None:
    tasks = [make_task("TVA AGRIWAN", urgency="haute")]
    llm = make_mock_llm_signal()
    priorities = await generate_signal(tasks, capacity_minutes=_CAPACITY, llm=llm)

    assert len(priorities) == 1
    assert priorities[0].task_title == "TVA AGRIWAN"
    assert priorities[0].reason


@pytest.mark.asyncio
async def test_generate_signal_empty_tasks_skips_llm() -> None:
    llm = make_mock_llm_signal()
    result = await generate_signal([], capacity_minutes=_CAPACITY, llm=llm)

    assert result == []
    llm.structured_complete.assert_not_called()


@pytest.mark.asyncio
async def test_generate_signal_parse_error_returns_empty() -> None:
    tasks = [make_task()]
    llm = make_mock_llm_signal(raises=LLMParseError("bad signal"))
    result = await generate_signal(tasks, capacity_minutes=_CAPACITY, llm=llm)

    assert result == []


@pytest.mark.asyncio
async def test_generate_signal_transport_error_propagates() -> None:
    tasks = [make_task()]
    llm = make_mock_llm_signal(raises=LLMTransportError("dead"))

    with pytest.raises(LLMTransportError):
        await generate_signal(tasks, capacity_minutes=_CAPACITY, llm=llm)


@pytest.mark.asyncio
async def test_generate_signal_multiple_priorities() -> None:
    tasks = [make_task(f"Tâche {i}") for i in range(5)]
    prios = [{"task_title": f"Tâche {i}", "reason": f"Raison {i}"} for i in range(3)]
    llm = make_mock_llm_signal(result=make_signal_result(prios))
    result = await generate_signal(tasks, capacity_minutes=_CAPACITY, llm=llm)

    assert len(result) == 3


@pytest.mark.asyncio
async def test_generate_signal_null_urgency_uses_fallback_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Tâche sans urgency → fallback 'normale' loggué en warning."""
    tasks = [make_task(urgency=None)]
    llm = make_mock_llm_signal()

    with caplog.at_level("WARNING"):
        await generate_signal(tasks, capacity_minutes=_CAPACITY, llm=llm)

    assert any("urgency" in r.message.lower() for r in caplog.records)


@pytest.mark.asyncio
async def test_generate_signal_empty_priorities_from_llm() -> None:
    """LLM retourne priorities:[] (liste vide explicite) → [] retourné."""
    tasks = [make_task()]
    llm = make_mock_llm_signal(result=make_signal_result(priorities=[]))
    result = await generate_signal(tasks, capacity_minutes=_CAPACITY, llm=llm)

    assert result == []


# ---------------------------------------------------------------------------
# Tests unitaires _days_since
# ---------------------------------------------------------------------------


def test_days_since_recent() -> None:
    dt = datetime.now(tz=timezone.utc) - timedelta(days=5)
    assert _days_since(dt) == 5


def test_days_since_naive_datetime() -> None:
    """Datetime naif (UTC implicite, comme retourné par SQLite func.now())."""
    dt = datetime.now(tz=timezone.utc).replace(tzinfo=None) - timedelta(days=10)
    assert _days_since(dt) == 10


def test_days_since_now_returns_zero() -> None:
    assert _days_since(datetime.now(tz=timezone.utc)) == 0


def test_days_since_never_negative() -> None:
    """Datetime dans le futur → 0, pas négatif."""
    future = datetime.now(tz=timezone.utc) + timedelta(days=2)
    assert _days_since(future) == 0


def test_days_since_truncates_not_rounds() -> None:
    """4j23h → 4, pas 5. Comportement documenté : tronqué, pas arrondi."""
    dt = datetime.now(tz=timezone.utc) - timedelta(days=4, hours=23)
    assert _days_since(dt) == 4
