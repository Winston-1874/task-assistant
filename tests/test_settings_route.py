"""Tests de la page de configuration modèle LLM."""
import pytest
from unittest.mock import AsyncMock, patch
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
    factory = async_sessionmaker(db_engine, expire_on_commit=False)
    session = factory()
    try:
        yield session
    finally:
        await session.close()


@pytest.fixture
async def client(db_session: AsyncSession):
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def auth_cookies():
    return {"session": create_session_cookie()}


@pytest.mark.asyncio
async def test_settings_page_loads(client, auth_cookies):
    """GET /settings retourne 200 avec le modèle courant."""
    response = await client.get("/settings", cookies=auth_cookies)
    assert response.status_code == 200
    assert "llm_model" in response.text or "gemini" in response.text.lower()


@pytest.mark.asyncio
async def test_save_model_updates_settings(client, auth_cookies):
    """POST /fragments/settings/model/save persiste le modèle."""
    response = await client.post(
        "/fragments/settings/model/save",
        data={"model": "openai/gpt-4o"},
        cookies=auth_cookies,
    )
    assert response.status_code == 200
    assert "openai/gpt-4o" in response.text

    from app.config import settings
    assert settings.llm_model == "openai/gpt-4o"


@pytest.mark.asyncio
async def test_test_model_ok(client, auth_cookies):
    """POST /fragments/settings/model/test retourne succès si LLM répond."""
    with patch("app.routes.settings.get_llm_client") as mock_factory:
        mock_client = AsyncMock()
        mock_client.structured_complete = AsyncMock(return_value=None)
        mock_factory.return_value = mock_client

        response = await client.post(
            "/fragments/settings/model/test",
            data={"model": "google/gemini-2.5-flash"},
            cookies=auth_cookies,
        )
    assert response.status_code == 200
    assert "OK" in response.text or "ok" in response.text.lower()


@pytest.mark.asyncio
async def test_test_model_error(client, auth_cookies):
    """POST /fragments/settings/model/test retourne erreur si LLM échoue."""
    from app.llm.client import LLMTransportError
    with patch("app.routes.settings.get_llm_client") as mock_factory:
        mock_client = AsyncMock()
        mock_client.structured_complete = AsyncMock(side_effect=LLMTransportError("timeout"))
        mock_factory.return_value = mock_client

        response = await client.post(
            "/fragments/settings/model/test",
            data={"model": "modele/inexistant"},
            cookies=auth_cookies,
        )
    assert response.status_code == 200
    assert "erreur" in response.text.lower() or "error" in response.text.lower() or "échec" in response.text.lower()
