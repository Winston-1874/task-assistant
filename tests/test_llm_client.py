"""Tests unitaires du wrapper LLM — sans appel réseau (mocks)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from app.llm.client import (
    LLMClient,
    LLMParseError,
    LLMResult,
    LLMTransportError,
    _extract_json,
)
from app.schemas import Classification

VALID_PAYLOAD = {
    "category": "client_urgent",
    "urgency": "haute",
    "confidence": 0.92,
    "reasoning": "Deadline fiscale imminente",
    "needs_due_date": False,
    "tags": ["tva"],
}


def make_mock_response(content: str | None, finish_reason: str = "stop") -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.choices[0].finish_reason = finish_reason
    resp.usage = MagicMock(prompt_tokens=100, completion_tokens=50)
    return resp


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------


def test_extract_json_plain():
    assert _extract_json('{"a": 1}') == '{"a": 1}'


def test_extract_json_with_prose():
    assert _extract_json('Voici le JSON : {"a": 1} fin') == '{"a": 1}'


def test_extract_json_markdown_fences():
    raw = "```json\n{\"a\": 1}\n```"
    assert _extract_json(raw) == '{"a": 1}'


def test_extract_json_nested():
    raw = '{"nested": {"b": 2}}'
    assert _extract_json(raw) == '{"nested": {"b": 2}}'


def test_extract_json_no_json():
    with pytest.raises(LLMParseError, match="Aucun objet JSON"):
        _extract_json("pas de json ici")


def test_extract_json_unclosed():
    with pytest.raises(LLMParseError, match="non fermé"):
        _extract_json('{"a": 1')


# ---------------------------------------------------------------------------
# structured_complete — cas nominaux
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_structured_complete_returns_llm_result():
    client = LLMClient()
    with patch.object(
        client._client.chat.completions,
        "create",
        new=AsyncMock(return_value=make_mock_response(json.dumps(VALID_PAYLOAD))),
    ):
        result = await client.structured_complete(
            messages=[{"role": "user", "content": "test"}],
            schema=Classification,
        )

    assert isinstance(result, LLMResult)
    assert result.data.category == "client_urgent"
    assert result.data.urgency == "haute"
    assert result.prompt_tokens == 100
    assert result.completion_tokens == 50
    assert result.prompt_version == "1.0.0"
    assert result.raw_json == json.dumps(VALID_PAYLOAD)


@pytest.mark.asyncio
async def test_structured_complete_strips_markdown_fences():
    client = LLMClient()
    raw = "```json\n" + json.dumps(VALID_PAYLOAD) + "\n```"
    with patch.object(
        client._client.chat.completions,
        "create",
        new=AsyncMock(return_value=make_mock_response(raw)),
    ):
        result = await client.structured_complete(
            messages=[{"role": "user", "content": "test"}],
            schema=Classification,
        )
    assert result.data.category == "client_urgent"


# ---------------------------------------------------------------------------
# structured_complete — erreurs de parsing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bad_json_raises_llm_parse_error():
    client = LLMClient()
    with patch.object(
        client._client.chat.completions,
        "create",
        new=AsyncMock(return_value=make_mock_response("pas du json du tout")),
    ):
        with pytest.raises(LLMParseError):
            await client.structured_complete(
                messages=[{"role": "user", "content": "test"}],
                schema=Classification,
            )


@pytest.mark.asyncio
async def test_invalid_schema_raises_llm_parse_error():
    client = LLMClient()
    bad = {**VALID_PAYLOAD, "urgency": "INVALIDE"}
    with patch.object(
        client._client.chat.completions,
        "create",
        new=AsyncMock(return_value=make_mock_response(json.dumps(bad))),
    ):
        with pytest.raises(LLMParseError):
            await client.structured_complete(
                messages=[{"role": "user", "content": "test"}],
                schema=Classification,
            )


@pytest.mark.asyncio
async def test_finish_reason_length_raises_llm_parse_error():
    client = LLMClient()
    with patch.object(
        client._client.chat.completions,
        "create",
        new=AsyncMock(return_value=make_mock_response("{}", finish_reason="length")),
    ):
        with pytest.raises(LLMParseError, match="tronquée"):
            await client.structured_complete(
                messages=[{"role": "user", "content": "test"}],
                schema=Classification,
            )


@pytest.mark.asyncio
async def test_none_content_raises_llm_parse_error():
    client = LLMClient()
    with patch.object(
        client._client.chat.completions,
        "create",
        new=AsyncMock(return_value=make_mock_response(None)),
    ):
        with pytest.raises(LLMParseError, match="no content"):
            await client.structured_complete(
                messages=[{"role": "user", "content": "test"}],
                schema=Classification,
            )


@pytest.mark.asyncio
async def test_empty_choices_raises_llm_parse_error():
    client = LLMClient()
    resp = MagicMock()
    resp.choices = []
    with patch.object(
        client._client.chat.completions,
        "create",
        new=AsyncMock(return_value=resp),
    ):
        with pytest.raises(LLMParseError, match="empty choices"):
            await client.structured_complete(
                messages=[{"role": "user", "content": "test"}],
                schema=Classification,
            )
