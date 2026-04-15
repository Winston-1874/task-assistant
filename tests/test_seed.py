"""
Tests du seed initial.

Vérifie que seed_initial_data() :
- insère les catégories et contextes attendus
- est idempotent (pas de doublon au second appel)
- ne plante pas si la base est déjà peuplée
"""

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool
from unittest.mock import AsyncMock, patch

from app.db import Base
from app.models import Category, Context
from app.seed import (
    _DEFAULT_CATEGORIES,
    _EXAMPLE_CONTEXTS,
    seed_initial_data,
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
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
async def session_factory(db_engine):
    return async_sessionmaker(db_engine, expire_on_commit=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_seed_inserts_all_categories(session_factory):
    """Toutes les catégories par défaut sont présentes après seed."""
    with patch("app.seed.get_db_session") as mock_ctx:
        async with session_factory() as db:
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=db)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            await seed_initial_data()

    async with session_factory() as db:
        result = await db.execute(select(Category))
        categories = {c.name for c in result.scalars().all()}

    expected = {cat["name"] for cat in _DEFAULT_CATEGORIES}
    assert expected == categories


@pytest.mark.asyncio
async def test_seed_inserts_all_contexts(session_factory):
    """Tous les contextes d'exemple sont présents après seed."""
    with patch("app.seed.get_db_session") as mock_ctx:
        async with session_factory() as db:
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=db)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            await seed_initial_data()

    async with session_factory() as db:
        result = await db.execute(select(Context))
        context_names = {c.name for c in result.scalars().all()}

    expected = {ctx["name"] for ctx in _EXAMPLE_CONTEXTS}
    assert expected == context_names


@pytest.mark.asyncio
async def test_seed_category_fields(session_factory):
    """Les champs description et color sont correctement insérés."""
    with patch("app.seed.get_db_session") as mock_ctx:
        async with session_factory() as db:
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=db)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            await seed_initial_data()

    async with session_factory() as db:
        cat = (await db.execute(select(Category).where(Category.name == "client_urgent"))).scalar_one()

    assert "deadline" in cat.description.lower()
    assert cat.color is not None


@pytest.mark.asyncio
async def test_seed_idempotent_categories(session_factory):
    """Deux appels successifs → toujours exactement 8 catégories, pas de doublon."""
    with patch("app.seed.get_db_session") as mock_ctx:
        async with session_factory() as db:
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=db)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            await seed_initial_data()

    with patch("app.seed.get_db_session") as mock_ctx:
        async with session_factory() as db:
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=db)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            await seed_initial_data()

    async with session_factory() as db:
        count = (await db.execute(text("SELECT COUNT(*) FROM categories"))).scalar()

    assert count == len(_DEFAULT_CATEGORIES)


@pytest.mark.asyncio
async def test_seed_idempotent_contexts(session_factory):
    """Deux appels successifs → toujours exactement N contextes, pas de doublon."""
    for _ in range(2):
        with patch("app.seed.get_db_session") as mock_ctx:
            async with session_factory() as db:
                mock_ctx.return_value.__aenter__ = AsyncMock(return_value=db)
                mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
                await seed_initial_data()

    async with session_factory() as db:
        count = (await db.execute(text("SELECT COUNT(*) FROM contexts"))).scalar()

    assert count == len(_EXAMPLE_CONTEXTS)


@pytest.mark.asyncio
async def test_seed_preserves_existing_category(session_factory):
    """Catégorie existante déjà modifiée → non écrasée par le seed."""
    async with session_factory() as db:
        db.add(Category(name="client_urgent", description="Ma description custom", color="#000000"))
        await db.commit()

    with patch("app.seed.get_db_session") as mock_ctx:
        async with session_factory() as db:
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=db)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            await seed_initial_data()

    async with session_factory() as db:
        cat = (await db.execute(select(Category).where(Category.name == "client_urgent"))).scalar_one()

    assert cat.description == "Ma description custom", "INSERT OR IGNORE doit préserver l'existant"


@pytest.mark.asyncio
async def test_seed_preserves_existing_context(session_factory):
    """Contexte existant (même name) → non dupliqué par le seed."""
    async with session_factory() as db:
        db.add(Context(name="Cabinet REBC", kind="interne", notes="Notes custom"))
        await db.commit()

    with patch("app.seed.get_db_session") as mock_ctx:
        async with session_factory() as db:
            mock_ctx.return_value.__aenter__ = AsyncMock(return_value=db)
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            await seed_initial_data()

    async with session_factory() as db:
        result = await db.execute(select(Context).where(Context.name == "Cabinet REBC"))
        rows = result.scalars().all()

    assert len(rows) == 1, "Le contexte existant ne doit pas être dupliqué"
    assert rows[0].notes == "Notes custom", "Le contenu existant doit être préservé"
