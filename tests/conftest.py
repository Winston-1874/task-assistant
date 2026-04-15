"""
Configuration pytest globale.

Les variables d'environnement doivent être posées AVANT tout import de app.*
car app.config.settings est instancié au niveau module.
"""

import os

# Valeurs minimales pour que Settings() ne lève pas à la collection
os.environ.setdefault("OPENROUTER_API_KEY", "sk-test-dummy")
os.environ.setdefault("APP_PASSWORD_HASH", "$2b$04$UCgIWmqmpBHxtAKVoP81..6Z2IkEY4GLQl4mZpaqNnzzSBj2rcPU2")
os.environ.setdefault("SECRET_KEY", "test-secret-key-not-for-production")
os.environ.setdefault("ENVIRONMENT", "dev")

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.auth import create_session_cookie
from app.db import Base, get_db
from app.main import app


@pytest.fixture
async def db_engine():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
async def db_session(db_engine):
    session_factory = async_sessionmaker(db_engine, expire_on_commit=False)
    session = session_factory()
    try:
        yield session
    finally:
        try:
            await session.close()
        finally:
            pass


@pytest.fixture
async def client(db_session: AsyncSession):
    """Client HTTP avec DB en mémoire injectée via override de dépendance."""

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def auth_cookies():
    """Cookie de session valide pour les tests authentifiés."""
    return {"session": create_session_cookie()}
