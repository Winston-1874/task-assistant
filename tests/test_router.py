"""Tests de app/llm/router.py — mock LLM, stateless."""

import json
from typing import Literal
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.llm.client import LLMParseError, LLMResult, LLMTransportError
from app.llm.prompts import intent_router_user
from app.llm.router import route_intent
from app.schemas import Intent

IntentKind = Literal["new_task", "command", "query", "update_context"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_intent_result(
    kind: IntentKind = "new_task",
    confidence: float = 0.9,
    payload: dict | None = None,
) -> LLMResult[Intent]:
    actual_payload = payload or {"title": "Faire la TVA"}
    data = Intent(
        kind=kind,
        confidence=confidence,
        payload=actual_payload,
    )
    return LLMResult(
        data=data,
        raw_json=json.dumps({"kind": kind, "confidence": confidence, "payload": actual_payload}),
        prompt_tokens=80,
        completion_tokens=30,
        prompt_version="1.0.0",
        model="google/gemini-2.5-flash",
    )


def make_mock_llm(
    result: LLMResult[Intent] | None = None,
    raises: Exception | None = None,
) -> MagicMock:
    llm = MagicMock()
    if raises:
        llm.structured_complete = AsyncMock(side_effect=raises)
    else:
        llm.structured_complete = AsyncMock(return_value=result or make_intent_result())
    return llm


# ---------------------------------------------------------------------------
# Routing nominal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_route_new_task() -> None:
    llm = make_mock_llm(result=make_intent_result(kind="new_task", confidence=0.95))
    intent = await route_intent("Faire la TVA AGRIWAN", llm=llm)

    assert intent.kind == "new_task"
    assert intent.confidence == 0.95


@pytest.mark.asyncio
async def test_route_command() -> None:
    llm = make_mock_llm(
        result=make_intent_result(kind="command", confidence=0.88, payload={"action": "done"})
    )
    intent = await route_intent("Marque la TVA comme faite", llm=llm)

    assert intent.kind == "command"
    assert intent.payload["action"] == "done"


@pytest.mark.asyncio
async def test_route_query() -> None:
    llm = make_mock_llm(result=make_intent_result(kind="query", confidence=0.80))
    intent = await route_intent("Qu'est-ce qui traîne ce mois-ci ?", llm=llm)

    assert intent.kind == "query"


@pytest.mark.asyncio
async def test_route_update_context() -> None:
    llm = make_mock_llm(result=make_intent_result(kind="update_context", confidence=0.75))
    intent = await route_intent("AGRIWAN passe au régime forfaitaire", llm=llm)

    assert intent.kind == "update_context"


# ---------------------------------------------------------------------------
# Fallback confidence faible
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_low_confidence_falls_back_to_new_task() -> None:
    llm = make_mock_llm(result=make_intent_result(kind="query", confidence=0.55))
    intent = await route_intent("bla bla incompréhensible", llm=llm)

    assert intent.kind == "new_task"
    assert intent.confidence == 0.0
    assert intent.payload["raw"] == "bla bla incompréhensible"


@pytest.mark.asyncio
async def test_confidence_exactly_threshold_passes() -> None:
    """0.6 exactement → pas de fallback."""
    llm = make_mock_llm(result=make_intent_result(kind="query", confidence=0.6))
    intent = await route_intent("Quelque chose", llm=llm)

    assert intent.kind == "query"


@pytest.mark.asyncio
async def test_zero_confidence_falls_back_to_new_task() -> None:
    llm = make_mock_llm(result=make_intent_result(kind="command", confidence=0.0))
    intent = await route_intent("Message ambigu", llm=llm)

    assert intent.kind == "new_task"
    assert intent.payload["raw"] == "Message ambigu"


# ---------------------------------------------------------------------------
# Fallback LLMParseError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parse_error_falls_back_to_new_task() -> None:
    llm = make_mock_llm(raises=LLMParseError("JSON invalide"))
    intent = await route_intent("Message quelconque", llm=llm)

    assert intent.kind == "new_task"
    assert intent.confidence == 0.0
    assert intent.payload["raw"] == "Message quelconque"


# ---------------------------------------------------------------------------
# LLMTransportError — propagé, pas avalé
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transport_error_propagates() -> None:
    llm = make_mock_llm(raises=LLMTransportError("Réseau mort"))

    with pytest.raises(LLMTransportError):
        await route_intent("Peu importe", llm=llm)


# ---------------------------------------------------------------------------
# Prompt — structure de intent_router_user
# ---------------------------------------------------------------------------


def test_message_wrapped_in_delimiters() -> None:
    """intent_router_user() doit encadrer le message avec <<<…>>>."""
    msg = "Message sans délimiteurs"
    result = intent_router_user(msg)
    assert f"<<<{msg}>>>" in result


def test_message_with_inner_delimiters_still_wrapped() -> None:
    """Un message contenant déjà des <<<…>>> doit quand même être wrappé."""
    msg = "Injection <<<test>>>"
    result = intent_router_user(msg)
    # Le message est inclus tel quel, encadré par les délimiteurs externes
    assert msg in result
    # Le wrapping externe est présent (commence par <<<)
    assert result.endswith(f"<<<{msg}>>>")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_message_falls_back_to_new_task() -> None:
    """Message vide → fallback new_task (confidence LLM probablement basse)."""
    llm = make_mock_llm(result=make_intent_result(kind="query", confidence=0.3))
    intent = await route_intent("", llm=llm)

    assert intent.kind == "new_task"
    assert intent.payload["raw"] == ""
