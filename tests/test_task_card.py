"""
Tests TDD pour le fragment task_card.html.

Vérifie le rendu du spinner (llm_pending=True), de la carte complète,
du toggle reasoning et du bouton "Corriger IA".
"""

from datetime import datetime, timezone

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Task


# ---------------------------------------------------------------------------
# 1. Spinner quand llm_pending=True
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_task_card_shows_spinner_when_llm_pending(
    client: AsyncClient,
    auth_cookies: dict,
    db_session: AsyncSession,
) -> None:
    """Le poll endpoint retourne une carte avec animate-pulse et hx-trigger quand llm_pending=True."""
    task = Task(
        title="Tâche en cours de classification",
        urgency="normale",
        status="open",
        llm_pending=True,
        touched_at=datetime.now(tz=timezone.utc),
    )
    db_session.add(task)
    await db_session.commit()
    await db_session.refresh(task)

    response = await client.get(
        f"/fragments/tasks/{task.id}/status",
        cookies=auth_cookies,
    )
    assert response.status_code == 200
    assert "animate-pulse" in response.text
    assert "hx-trigger" in response.text


# ---------------------------------------------------------------------------
# 2. Carte complète quand llm_pending=False
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_task_card_shows_full_content_when_ready(
    client: AsyncClient,
    auth_cookies: dict,
    db_session: AsyncSession,
) -> None:
    """Le poll endpoint retourne la carte avec le titre quand llm_pending=False."""
    task = Task(
        title="Tâche classifiée",
        urgency="haute",
        status="open",
        llm_pending=False,
        touched_at=datetime.now(tz=timezone.utc),
    )
    db_session.add(task)
    await db_session.commit()
    await db_session.refresh(task)

    response = await client.get(
        f"/fragments/tasks/{task.id}/status",
        cookies=auth_cookies,
    )
    assert response.status_code == 200
    assert "Tâche classifiée" in response.text
    assert "animate-pulse" not in response.text
    assert 'hx-trigger="every 2s"' not in response.text


# ---------------------------------------------------------------------------
# 3. Bouton "voir plus" quand reasoning > 120 caractères
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_task_card_reasoning_show_toggle_button_when_long(
    client: AsyncClient,
    auth_cookies: dict,
    db_session: AsyncSession,
) -> None:
    """Une tâche avec un raisonnement > 120 caractères affiche un bouton 'voir plus'."""
    long_reasoning = (
        "Ceci est un raisonnement très long qui dépasse certainement les cent vingt "
        "caractères requis pour déclencher l'affichage du bouton de bascule voir plus."
    )
    assert len(long_reasoning) > 120

    task = Task(
        title="Tâche avec long raisonnement",
        urgency="normale",
        status="open",
        llm_pending=False,
        llm_reasoning=long_reasoning,
        touched_at=datetime.now(tz=timezone.utc),
    )
    db_session.add(task)
    await db_session.commit()
    await db_session.refresh(task)

    response = await client.get(
        f"/fragments/tasks/{task.id}/status",
        cookies=auth_cookies,
    )
    assert response.status_code == 200
    assert "voir plus" in response.text


# ---------------------------------------------------------------------------
# 4. Pas de bouton "voir plus" quand reasoning <= 120 caractères
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_task_card_reasoning_no_toggle_when_short(
    client: AsyncClient,
    auth_cookies: dict,
    db_session: AsyncSession,
) -> None:
    """Une tâche avec un raisonnement <= 120 caractères n'affiche pas de bouton 'voir plus'."""
    short_reasoning = "Raisonnement court."
    assert len(short_reasoning) <= 120

    task = Task(
        title="Tâche avec court raisonnement",
        urgency="normale",
        status="open",
        llm_pending=False,
        llm_reasoning=short_reasoning,
        touched_at=datetime.now(tz=timezone.utc),
    )
    db_session.add(task)
    await db_session.commit()
    await db_session.refresh(task)

    response = await client.get(
        f"/fragments/tasks/{task.id}/status",
        cookies=auth_cookies,
    )
    assert response.status_code == 200
    assert "voir plus" not in response.text


# ---------------------------------------------------------------------------
# 5. Bouton "Corriger IA" visible quand reasoning est défini
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_task_card_corriger_ia_button_visible_when_reasoning(
    client: AsyncClient,
    auth_cookies: dict,
    db_session: AsyncSession,
) -> None:
    """La carte affiche le bouton 'Corriger IA' quand llm_reasoning est défini."""
    task = Task(
        title="Tâche avec raisonnement",
        urgency="normale",
        status="open",
        llm_pending=False,
        llm_reasoning="Un raisonnement quelconque.",
        touched_at=datetime.now(tz=timezone.utc),
    )
    db_session.add(task)
    await db_session.commit()
    await db_session.refresh(task)

    response = await client.get(
        f"/fragments/tasks/{task.id}/status",
        cookies=auth_cookies,
    )
    assert response.status_code == 200
    assert "Corriger IA" in response.text
