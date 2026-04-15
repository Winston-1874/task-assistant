"""Tests de app/memory.py — sélection TF-IDF des few-shots."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db import Base
from app.memory import select_few_shots, _rank_by_tfidf
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
    async with session_factory() as session:
        yield session
    await engine.dispose()


async def _add_correction(
    db: AsyncSession, task_id: int, title: str, field: str, old: str, new: str, description: str = ""
) -> None:
    db.add(Correction(
        task_id=task_id,
        task_title=title,
        task_description=description or None,
        field=field,
        old_value=old,
        new_value=new,
    ))
    await db.commit()


# ---------------------------------------------------------------------------
# Tests
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
    # Même task_id, deux champs corrigés → ne doit apparaître qu'une fois
    for i in range(6):
        await _add_correction(db, i + 1, f"Tâche {i}", "urgency", "normale", "haute")
    await _add_correction(db, 1, "TVA AGRIWAN", "category", "inbox", "client_urgent")

    result = await select_few_shots("TVA AGRIWAN", None, db)
    task_ids = [r["task_title"] for r in result]
    assert len(task_ids) == len(set(task_ids))


@pytest.mark.asyncio
async def test_max_20_results(db: AsyncSession) -> None:
    for i in range(25):
        await _add_correction(db, i + 1, f"Tâche fiscale {i}", "urgency", "normale", "haute")

    result = await select_few_shots("tâche fiscale", None, db)
    assert len(result) <= 20


@pytest.mark.asyncio
async def test_tfidf_prefers_similar_titles(db: AsyncSession) -> None:
    # 5+ corrections nécessaires pour activer TF-IDF
    await _add_correction(db, 1, "TVA trimestrielle AGRIWAN", "urgency", "normale", "haute")
    await _add_correction(db, 2, "Bilan annuel NewCo", "category", "inbox", "client_standard")
    await _add_correction(db, 3, "Relance facture Dupont", "urgency", "normale", "haute")
    await _add_correction(db, 4, "RH cabinet recrutement", "category", "inbox", "admin_cabinet")
    await _add_correction(db, 5, "Déclaration IPP 2024", "urgency", "normale", "critique")

    result = await select_few_shots("déclaration TVA mensuelle", None, db)
    # La correction TVA ou IPP doit remonter
    titles = [r["task_title"] for r in result]
    assert any("TVA" in t or "IPP" in t or "Déclaration" in t for t in titles)


# ---------------------------------------------------------------------------
# Tests unitaires _rank_by_tfidf
# ---------------------------------------------------------------------------


def make_correction(task_id: int, title: str) -> Correction:
    c = Correction()
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
    corrections = [make_correction(i, f"Tâche {i}") for i in range(10)]
    ranked = _rank_by_tfidf("tâche fiscale", corrections, limit=3)
    assert len(ranked) <= 3


def test_rank_by_tfidf_empty_corpus() -> None:
    corrections = [make_correction(1, "")]
    result = _rank_by_tfidf("TVA", corrections, limit=5)
    assert isinstance(result, list)
