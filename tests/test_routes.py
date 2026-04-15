"""
Tests d'intégration des routes FastAPI.

Utilise httpx.AsyncClient avec l'app FastAPI en mode test.
DB en mémoire. LLM mocké via monkeypatch.
"""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.auth import create_session_cookie
from app.db import Base, get_db
from app.llm.client import LLMResult
from app.main import app
from app.models import Task
from app.schemas import Classification


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


def make_classification_result(
    category: str = "client_urgent",
    confidence: float = 0.9,
) -> LLMResult[Classification]:
    data = Classification(
        category=category,
        urgency="haute",
        confidence=confidence,
        reasoning="Test classification",
        needs_due_date=False,
        tags=[],
    )
    return LLMResult(
        data=data,
        raw_json=json.dumps({"category": category}),
        prompt_tokens=100,
        completion_tokens=50,
        prompt_version="1.0.0",
        model="google/gemini-2.5-flash",
    )


# ---------------------------------------------------------------------------
# Tests auth
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_redirect_unauthenticated(client: AsyncClient) -> None:
    response = await client.get("/", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/login"


@pytest.mark.asyncio
async def test_login_page_accessible(client: AsyncClient) -> None:
    response = await client.get("/login")
    assert response.status_code == 200
    assert "Mot de passe" in response.text


@pytest.mark.asyncio
async def test_login_wrong_password(client: AsyncClient) -> None:
    response = await client.post(
        "/login",
        data={"password": "mauvais"},
        follow_redirects=False,
    )
    assert response.status_code == 401
    assert "incorrect" in response.text.lower()


@pytest.mark.asyncio
async def test_logout_clears_session(client: AsyncClient, auth_cookies: dict) -> None:
    response = await client.post(
        "/logout",
        cookies=auth_cookies,
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/login"


# ---------------------------------------------------------------------------
# Tests page principale
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_index_authenticated(client: AsyncClient, auth_cookies: dict) -> None:
    response = await client.get("/", cookies=auth_cookies)
    assert response.status_code == 200
    assert "Tasks" in response.text


@pytest.mark.asyncio
async def test_week_view(client: AsyncClient, auth_cookies: dict) -> None:
    response = await client.get("/week", cookies=auth_cookies)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_inbox_view(client: AsyncClient, auth_cookies: dict) -> None:
    response = await client.get("/inbox", cookies=auth_cookies)
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Tests création de tâche (fragment)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_task_returns_card(
    client: AsyncClient,
    auth_cookies: dict,
    db_session: AsyncSession,
) -> None:
    """La création de tâche retourne une carte immédiatement (optimistic UI)."""
    with patch("app.routes.fragments._classify_and_update", new_callable=AsyncMock):
        response = await client.post(
            "/fragments/tasks",
            data={"title": "TVA AGRIWAN", "description": ""},
            cookies=auth_cookies,
        )

    assert response.status_code == 200
    assert "TVA AGRIWAN" in response.text
    assert "task-" in response.text


@pytest.mark.asyncio
async def test_create_task_optimistic_has_llm_pending(
    client: AsyncClient,
    auth_cookies: dict,
    db_session: AsyncSession,
) -> None:
    """La carte retournée immédiatement contient le polling HTMX (llm_pending=True)."""
    with patch("app.routes.fragments._classify_and_update", new_callable=AsyncMock):
        response = await client.post(
            "/fragments/tasks",
            data={"title": "Tâche optimiste", "description": ""},
            cookies=auth_cookies,
        )

    assert response.status_code == 200
    assert "hx-trigger" in response.text  # carte en mode polling


# ---------------------------------------------------------------------------
# Tests mark done
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mark_done(
    client: AsyncClient,
    auth_cookies: dict,
    db_session: AsyncSession,
) -> None:
    task = Task(
        title="TVA AGRIWAN",
        urgency="haute",
        status="open",
        touched_at=datetime.now(tz=timezone.utc),
    )
    db_session.add(task)
    await db_session.commit()

    response = await client.post(
        f"/fragments/tasks/{task.id}/done",
        cookies=auth_cookies,
    )
    assert response.status_code == 200
    # La card retournée doit afficher la tâche (pas de redirect)
    assert "TVA AGRIWAN" in response.text


# ---------------------------------------------------------------------------
# Tests delete
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_task(
    client: AsyncClient,
    auth_cookies: dict,
    db_session: AsyncSession,
) -> None:
    task = Task(
        title="À supprimer",
        urgency="basse",
        status="open",
        touched_at=datetime.now(tz=timezone.utc),
    )
    db_session.add(task)
    await db_session.commit()
    task_id = task.id

    response = await client.delete(
        f"/fragments/tasks/{task_id}",
        cookies=auth_cookies,
    )
    assert response.status_code == 200

    # Vérifie suppression en DB
    from sqlalchemy import select
    result = await db_session.execute(select(Task).where(Task.id == task_id))
    assert result.scalar_one_or_none() is None


# ---------------------------------------------------------------------------
# Tests undo done
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_undo_done(
    client: AsyncClient,
    auth_cookies: dict,
    db_session: AsyncSession,
) -> None:
    task = Task(
        title="TVA AGRIWAN",
        urgency="haute",
        status="done",
        touched_at=datetime.now(tz=timezone.utc),
        completed_at=datetime.now(tz=timezone.utc),
    )
    db_session.add(task)
    await db_session.commit()

    response = await client.post(
        f"/fragments/tasks/{task.id}/undo/done",
        cookies=auth_cookies,
    )
    assert response.status_code == 200
    assert "TVA AGRIWAN" in response.text
    # Le titre ne doit pas être barré (tâche open)
    assert "line-through" not in response.text

    from sqlalchemy import select
    result = await db_session.execute(select(Task).where(Task.id == task.id))
    restored = result.scalar_one_or_none()
    assert restored is not None
    assert restored.status == "open"
    assert restored.completed_at is None


@pytest.mark.asyncio
async def test_undo_done_on_open_task_is_noop(
    client: AsyncClient,
    auth_cookies: dict,
    db_session: AsyncSession,
) -> None:
    """undo/done sur une tâche déjà open → 200 sans erreur, status reste open."""
    task = Task(
        title="Tâche déjà open",
        urgency="normale",
        status="open",
        touched_at=datetime.now(tz=timezone.utc),
    )
    db_session.add(task)
    await db_session.commit()

    response = await client.post(
        f"/fragments/tasks/{task.id}/undo/done",
        cookies=auth_cookies,
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_correct_task_invalid_field_returns_400(
    client: AsyncClient,
    auth_cookies: dict,
    db_session: AsyncSession,
) -> None:
    task = Task(
        title="Tâche test",
        urgency="haute",
        status="open",
        touched_at=datetime.now(tz=timezone.utc),
    )
    db_session.add(task)
    await db_session.commit()

    response = await client.post(
        f"/fragments/tasks/{task.id}/correct",
        data={"field": "__class__", "new_value": "malicious"},
        cookies=auth_cookies,
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_update_task_invalid_urgency_returns_422(
    client: AsyncClient,
    auth_cookies: dict,
    db_session: AsyncSession,
) -> None:
    task = Task(
        title="Tâche test",
        urgency="haute",
        status="open",
        touched_at=datetime.now(tz=timezone.utc),
    )
    db_session.add(task)
    await db_session.commit()

    response = await client.put(
        f"/fragments/tasks/{task.id}",
        data={"title": "Tâche test", "urgency": "INVALIDE", "description": ""},
        cookies=auth_cookies,
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_correct_task_invalid_urgency_returns_422(
    client: AsyncClient,
    auth_cookies: dict,
    db_session: AsyncSession,
) -> None:
    task = Task(
        title="Tâche test",
        urgency="haute",
        status="open",
        touched_at=datetime.now(tz=timezone.utc),
    )
    db_session.add(task)
    await db_session.commit()

    response = await client.post(
        f"/fragments/tasks/{task.id}/correct",
        data={"field": "urgency", "new_value": "INVALIDE"},
        cookies=auth_cookies,
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_correct_task_invalid_due_date_returns_422(
    client: AsyncClient,
    auth_cookies: dict,
    db_session: AsyncSession,
) -> None:
    task = Task(
        title="Tâche test",
        urgency="haute",
        status="open",
        touched_at=datetime.now(tz=timezone.utc),
    )
    db_session.add(task)
    await db_session.commit()

    response = await client.post(
        f"/fragments/tasks/{task.id}/correct",
        data={"field": "due_date", "new_value": "pas une date"},
        cookies=auth_cookies,
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_delete_task_idempotent(
    client: AsyncClient,
    auth_cookies: dict,
    db_session: AsyncSession,
) -> None:
    """DELETE sur tâche inexistante → 200 (idempotent, timer deferred côté client)."""
    response = await client.delete(
        "/fragments/tasks/99999",
        cookies=auth_cookies,
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_correct_task_invalid_context_id_returns_422(
    client: AsyncClient,
    auth_cookies: dict,
    db_session: AsyncSession,
) -> None:
    """correct_task avec context_id non entier → 422 (pas 500)."""
    task = Task(
        title="Tâche test",
        urgency="normale",
        status="open",
        touched_at=datetime.now(tz=timezone.utc),
    )
    db_session.add(task)
    await db_session.commit()

    response = await client.post(
        f"/fragments/tasks/{task.id}/correct",
        data={"field": "context_id", "new_value": "pas_un_entier"},
        cookies=auth_cookies,
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_fragment_route_requires_auth(client: AsyncClient) -> None:
    """Les routes fragments renvoient 303 → /login sans cookie de session."""
    response = await client.post(
        "/fragments/tasks/1/done",
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/login"


# ---------------------------------------------------------------------------
# Tests conversation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_conversation_new_task(
    client: AsyncClient,
    auth_cookies: dict,
) -> None:
    from app.schemas import Intent

    intent = Intent(kind="new_task", confidence=0.9, payload={})
    classification_result = make_classification_result()

    with (
        patch("app.routes.conversation.route_intent", new_callable=AsyncMock) as mock_route,
        patch("app.routes.conversation.classify_task", new_callable=AsyncMock) as mock_classify,
    ):
        mock_route.return_value = intent
        mock_classify.return_value = (classification_result.data, classification_result)

        response = await client.post(
            "/conversation/parse",
            data={"message": "Faire la TVA AGRIWAN"},
            cookies=auth_cookies,
        )

    assert response.status_code == 200
    assert "Faire la TVA AGRIWAN" in response.text


@pytest.mark.asyncio
async def test_conversation_non_task_intent(
    client: AsyncClient,
    auth_cookies: dict,
) -> None:
    from app.schemas import Intent

    intent = Intent(kind="query", confidence=0.85, payload={})
    with patch("app.routes.conversation.route_intent", new_callable=AsyncMock) as mock_route:
        mock_route.return_value = intent
        response = await client.post(
            "/conversation/parse",
            data={"message": "Qu'est-ce qui traîne ?"},
            cookies=auth_cookies,
        )

    assert response.status_code == 200
    assert "venir" in response.text  # message "fonctionnalité à venir"


@pytest.mark.asyncio
async def test_conversation_parse_returns_card_immediately(client, auth_cookies, db_session, monkeypatch):
    """La route retourne une carte (llm_pending) sans attendre le LLM."""
    from app.schemas import Intent

    async def slow_classify(*args, **kwargs):
        import asyncio
        await asyncio.sleep(100)  # ne sera jamais atteint car BackgroundTask

    monkeypatch.setattr("app.routes.conversation.classify_task", slow_classify)
    monkeypatch.setattr(
        "app.routes.conversation.route_intent",
        AsyncMock(return_value=Intent(kind="new_task", confidence=0.9, payload={})),
    )

    response = await client.post(
        "/conversation/parse",
        data={"message": "TVA AGRIWAN à envoyer"},
        cookies=auth_cookies,
    )
    assert response.status_code == 200
    assert "TVA AGRIWAN" in response.text


@pytest.mark.asyncio
async def test_poll_endpoint_returns_spinner_while_pending(client, auth_cookies, db_session):
    """GET /fragments/tasks/{id}/status retourne spinner si llm_pending=True."""
    from app.models import Task
    task = Task(title="Test", llm_pending=True, status="open")
    db_session.add(task)
    await db_session.commit()
    await db_session.refresh(task)

    response = await client.get(
        f"/fragments/tasks/{task.id}/status",
        cookies=auth_cookies,
    )
    assert response.status_code == 200
    assert "hx-trigger" in response.text  # carte encore en mode polling


@pytest.mark.asyncio
async def test_poll_endpoint_returns_full_card_when_ready(client, auth_cookies, db_session):
    """GET /fragments/tasks/{id}/status retourne carte complète si llm_pending=False."""
    from app.models import Task
    task = Task(title="Test", llm_pending=False, urgency="normale", status="open")
    db_session.add(task)
    await db_session.commit()
    await db_session.refresh(task)

    response = await client.get(
        f"/fragments/tasks/{task.id}/status",
        cookies=auth_cookies,
    )
    assert response.status_code == 200
    assert 'hx-trigger="every 2s"' not in response.text
