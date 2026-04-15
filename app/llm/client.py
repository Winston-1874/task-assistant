"""
Wrapper LLM sur OpenRouter (API compatible OpenAI).

Fonctionnalités :
- structured_complete() → LLMResult[T] avec raw_json pour persistance dans llm_raw_response
- Extraction JSON défensive (strips markdown fences, trouve le premier {...})
- Timeout, retry 1×, rate limit depuis Settings
- Vérification finish_reason, choices non vides, content non None
- Token logging garanti via finally
- Hiérarchie d'exceptions LLMError pour contrat clair côté appelant
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Generic, TypeVar

from openai import AsyncOpenAI, APIStatusError, APITimeoutError, RateLimitError
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ValidationError

from app.config import settings
from app.llm.prompts import PROMPT_VERSION

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class LLMError(Exception):
    """Base pour toutes les erreurs LLM."""


class LLMTransportError(LLMError):
    """Timeout, rate limit, erreur serveur après tous les retries."""


class LLMParseError(LLMError):
    """JSON invalide ou validation Pydantic échouée."""


# ---------------------------------------------------------------------------
# Résultat typé
# ---------------------------------------------------------------------------


@dataclass
class LLMResult(Generic[T]):
    """
    Résultat d'un appel structuré.
    raw_json doit être stocké dans Task.llm_raw_response pour audit et stats coût.
    """

    data: T
    raw_json: str
    prompt_tokens: int
    completion_tokens: int
    prompt_version: str
    model: str


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class LLMClient:
    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
        )

    async def structured_complete(
        self,
        messages: list[ChatCompletionMessageParam],
        schema: type[T],
        model: str | None = None,
    ) -> LLMResult[T]:
        """
        Appelle le LLM et retourne un LLMResult[T] validé.

        Lève :
        - LLMTransportError : réseau, timeout, rate limit épuisés
        - LLMParseError     : JSON invalide ou validation Pydantic échouée
        """
        resolved_model = model or settings.llm_model
        raw, usage = await self._call_with_retry(messages, resolved_model)

        try:
            json_str = _extract_json(raw)
            data = json.loads(json_str)
            result = schema.model_validate(data)
        except (LLMParseError, json.JSONDecodeError, ValidationError) as e:
            logger.error(
                "LLM parse error (%s) — raw[:500]: %s",
                schema.__name__,
                raw[:500],
            )
            raise LLMParseError(str(e)) from e
        finally:
            _log_tokens(usage, schema.__name__)

        return LLMResult(
            data=result,
            raw_json=json_str,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            prompt_version=usage["prompt_version"],
            model=usage["model"],
        )

    async def _call_with_retry(
        self,
        messages: list[ChatCompletionMessageParam],
        model: str,
    ) -> tuple[str, dict[str, Any]]:
        """
        1 retry avec backoff. RateLimitError est une sous-classe d'APIStatusError
        — l'ordre des except est intentionnel (plus spécifique en premier).
        """
        last_exc: Exception = RuntimeError("no attempt made")
        for attempt in range(2):
            try:
                return await self._call(messages, model)
            except RateLimitError as e:
                last_exc = e
                if attempt == 0:
                    logger.warning("Rate limit — attente %ss", settings.llm_rate_limit_delay)
                    await asyncio.sleep(settings.llm_rate_limit_delay)
            except APITimeoutError as e:
                last_exc = e
                if attempt == 0:
                    logger.warning("Timeout LLM — retry dans %ss", settings.llm_retry_delay)
                    await asyncio.sleep(settings.llm_retry_delay)
            except APIStatusError as e:
                last_exc = e
                if attempt == 0 and e.status_code >= 500:
                    logger.warning("Erreur serveur LLM %s — retry", e.status_code)
                    await asyncio.sleep(settings.llm_retry_delay)
                else:
                    logger.error("Erreur LLM non-retriable: %s", e)
                    raise LLMTransportError(str(e)) from e

        logger.error("LLM échec après %d tentatives: %s", 2, last_exc)
        raise LLMTransportError(str(last_exc)) from last_exc

    async def _call(
        self,
        messages: list[ChatCompletionMessageParam],
        model: str,
    ) -> tuple[str, dict[str, Any]]:
        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout,
        )

        if not response.choices:
            raise LLMParseError("LLM returned empty choices list")

        choice = response.choices[0]

        if choice.finish_reason == "length":
            raise LLMParseError(
                f"Réponse tronquée (finish_reason=length) — augmenter llm_max_tokens "
                f"(actuel: {settings.llm_max_tokens})"
            )

        content = choice.message.content
        if content is None:
            raise LLMParseError(
                f"LLM returned no content (finish_reason={choice.finish_reason!r})"
            )

        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "model": model,
            "prompt_version": PROMPT_VERSION,
        }
        return content, usage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> str:
    """
    Extrait le premier objet JSON {...} d'un texte pouvant contenir des fences
    Markdown ou du texte de prose (comportement observé avec Gemini via OpenRouter).
    """
    # Strip fences ```json ... ``` ou ``` ... ```
    text = re.sub(r"```(?:json)?\s*", "", text).strip()

    start = text.find("{")
    if start == -1:
        raise LLMParseError(f"Aucun objet JSON trouvé dans la réponse LLM: {text[:200]!r}")

    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    raise LLMParseError(f"Objet JSON non fermé dans la réponse LLM: {text[:200]!r}")


def _log_tokens(usage: dict[str, Any], operation: str) -> None:
    logger.info(
        "LLM %s — prompt=%d completion=%d model=%s version=%s",
        operation,
        usage["prompt_tokens"],
        usage["completion_tokens"],
        usage["model"],
        usage["prompt_version"],
    )


# ---------------------------------------------------------------------------
# Singleton — thread-safe via lru_cache (GIL)
# TODO: migrer vers app.state + lifespan FastAPI quand main.py est créé,
#       et exposer via Depends(get_llm_client) pour l'injection en tests.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    return LLMClient()
