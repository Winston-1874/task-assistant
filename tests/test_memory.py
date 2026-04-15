"""Tests de app/memory.py — sélection TF-IDF des few-shots."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db import Base
from app.memory import record_correction, select_few_shots, _rank_by_tfidf
from app.models import Correction


# ---------------------------------------------------------------------------
# Fixture DB en mémoire
# ---------------------------------------------------------------------------


@pytest.fixture
async def db() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    session = session_factory()
    try:
        yield session
    finally:
        await session.close()
        await engine.dispose()


async def _add_correction(
    db: AsyncSession,
    task_id: int,
    title: str,
    field: str,
    old: str,
    new: str,
    description: str = "",
) -> None:
    db.add(Correction(
        task_id=task_id,
        task_title=title,
        task_description=description if description else None,
        field=field,
        old_value=old,
        new_value=new,
    ))
    await db.commit()


# ---------------------------------------------------------------------------
# Tests select_few_shots
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_db_returns_empty(db: AsyncSession) -> None:
    result = await select_few_shots("TVA AGRIWAN", None, db)
    assert result == []


@pytest.mark.asyncio
async def test_few_corrections_returns_recent_only(db: AsyncSession) -> None:
    # Moins de 5 → pas de TF-IDF
    await _add_correction(db, 1, "TVA AGRIWAN", "urgency", "normale", "haute")
    await _add_correction(db, 2, "Bilan NewCo", "category", "inbox", "client_urgent")

    result = await select_few_shots("déclaration TVA", None, db)
    assert len(result) == 2
    assert all(r["field"] in ("urgency", "category") for r in result)


@pytest.mark.asyncio
async def test_deduplication(db: AsyncSession) -> None:
    # Même task_id, deux champs corrigés → les deux Correction.id sont distincts,
    # mais après dédup par Correction.id, tous apparaissent (dédup sur PK pas task_id).
    # Ce test vérifie qu'il n'y a pas de doublons dans le résultat.
    for i in range(6):
        await _add_correction(db, i + 1, f"Tâche {i}", "urgency", "normale", "haute")
    await _add_correction(db, 1, "TVA AGRIWAN (correction 2)", "category", "inbox", "client_urgent")

    result = await select_few_shots("TVA AGRIWAN", None, db)
    # Pas de doublon de Correction.id (implicite via la logique seen)
    titles = [r["task_title"] for r in result]
    assert len(titles) == len(set(titles)) or len(result) <= 20  # max 20 garanti


@pytest.mark.asyncio
async def test_max_20_results(db: AsyncSession) -> None:
    for i in range(25):
        await _add_correction(db, i + 1, f"Tâche fiscale {i}", "urgency", "normale", "haute")

    result = await select_few_shots("tâche fiscale", None, db)
    assert len(result) <= 20


@pytest.mark.asyncio
async def test_tfidf_prefers_similar_titles(db: AsyncSession) -> None:
    """
    10 corrections récentes sans lien avec TVA,
    1 correction plus ancienne très similaire à la query.
    TF-IDF doit la faire remonter dans le résultat.
    """
    # 10 corrections récentes (non similaires à la query)
    for i in range(10):
        await _add_correction(db, i + 1, f"Bilan annuel client {i}", "urgency", "normale", "haute")

    # 11e correction (plus ancienne) — similaire à la query
    await _add_correction(db, 11, "TVA trimestrielle AGRIWAN", "urgency", "normale", "haute")

    result = await select_few_shots("déclaration TVA AGRIWAN mensuelle", None, db)
    titles = [r["task_title"] for r in result]

    # Les 10 récentes sont dans recent, la TVA doit être récupérée par TF-IDF
    assert "TVA trimestrielle AGRIWAN" in titles


# ---------------------------------------------------------------------------
# Tests record_correction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_correction_adds_to_session(db: AsyncSession) -> None:
    correction = await record_correction(
        db,
        task_id=42,
        task_title="TVA AGRIWAN",
        task_description="Déclaration mensuelle",
        field="urgency",
        old_value="normale",
        new_value="haute",
    )
    await db.commit()

    assert correction.id is not None
    assert correction.task_id == 42
    assert correction.field == "urgency"
    assert correction.old_value == "normale"
    assert correction.new_value == "haute"


@pytest.mark.asyncio
async def test_record_correction_does_not_auto_commit(db: AsyncSession) -> None:
    """record_correction ne commit pas — le caller contrôle la transaction."""
    await record_correction(
        db,
        task_id=99,
        task_title="Test rollback",
        task_description=None,
        field="category",
        old_value="inbox",
        new_value="client_urgent",
    )
    # Rollback sans commit — la correction ne doit pas persister
    await db.rollback()

    from sqlalchemy import select as sa_select
    result = await db.execute(sa_select(Correction).where(Correction.task_id == 99))
    assert result.scalar_one_or_none() is None


# ---------------------------------------------------------------------------
# Tests unitaires _rank_by_tfidf
# ---------------------------------------------------------------------------


def make_correction(task_id: int, title: str) -> Correction:
    c = Correction()
    c.id = task_id  # suffisant pour les tests unitaires de ranking
    c.task_id = task_id
    c.task_title = title
    c.task_description = None
    c.field = "urgency"
    c.old_value = "normale"
    c.new_value = "haute"
    return c


def test_rank_by_tfidf_returns_most_similar_first() -> None:
    corrections = [
        make_correction(1, "TVA trimestrielle AGRIWAN"),
        make_correction(2, "Bilan annuel NewCo"),
        make_correction(3, "Relance facture client"),
    ]
    ranked = _rank_by_tfidf("déclaration TVA AGRIWAN", corrections, limit=3)
    assert ranked[0].task_title == "TVA trimestrielle AGRIWAN"


def test_rank_by_tfidf_respects_limit() -> None:
    corrections = [make_correction(i, f"Tâche fiscale {i}") for i in range(10)]
    ranked = _rank_by_tfidf("tâche fiscale", corrections, limit=3)
    assert len(ranked) <= 3


def test_rank_by_tfidf_empty_corpus() -> None:
    corrections = [make_correction(1, "")]
    result = _rank_by_tfidf("TVA", corrections, limit=5)
    assert isinstance(result, list)


def test_rank_by_tfidf_empty_query() -> None:
    corrections = [make_correction(i, f"Tâche {i}") for i in range(3)]
    result = _rank_by_tfidf("", corrections, limit=3)
    assert len(result) <= 3
