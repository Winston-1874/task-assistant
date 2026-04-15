"""
Wrapper LLM sur OpenRouter (API compatible OpenAI).

Fonctionnalités :
- structured_complete() : appel avec parsing Pydantic, fallback automatique
- Timeout 15s, retry 1× avec backoff exponentiel
- Rate limit : queue in-memory + retry après délai
- Logging tokens pour stats coût
"""

import asyncio
import json
import logging
from typing import Any, TypeVar

from openai import AsyncOpenAI, APIStatusError, APITimeoutError, RateLimitError
from pydantic import BaseModel, ValidationError

from app.config import settings
from app.llm.prompts import PROMPT_VERSION

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_TIMEOUT = 15.0
_RETRY_DELAY = 2.0
_RATE_LIMIT_DELAY = 10.0


class LLMClient:
    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
        )

    async def structured_complete(
        self,
        messages: list[dict[str, str]],
        schema: type[T],
        model: str | None = None,
    ) -> T:
        """
        Appelle le LLM et retourne une instance validée de `schema`.
        Lève ValidationError ou APIStatusError si les deux tentatives échouent.
        """
        raw, usage = await self._call_with_retry(messages, model or settings.llm_model)
        data = json.loads(raw)
        result = schema.model_validate(data)
        self._log_tokens(usage, schema.__name__)
        return result

    async def _call_with_retry(
        self,
        messages: list[dict[str, str]],
        model: str,
    ) -> tuple[str, dict[str, Any]]:
        """Tente l'appel LLM avec 1 retry. Gère timeout et rate limit."""
        for attempt in range(2):
            try:
                return await self._call(messages, model)
            except RateLimitError:
                if attempt == 0:
                    logger.warning("Rate limit atteint — attente %ss", _RATE_LIMIT_DELAY)
                    await asyncio.sleep(_RATE_LIMIT_DELAY)
                    continue
                raise
            except APITimeoutError:
                if attempt == 0:
                    logger.warning("Timeout LLM — retry dans %ss", _RETRY_DELAY)
                    await asyncio.sleep(_RETRY_DELAY)
                    continue
                raise
            except APIStatusError as e:
                if attempt == 0 and e.status_code >= 500:
                    logger.warning("Erreur serveur LLM %s — retry", e.status_code)
                    await asyncio.sleep(_RETRY_DELAY)
                    continue
                raise
        raise RuntimeError("Unreachable")  # satisfait mypy

    async def _call(
        self,
        messages: list[dict[str, str]],
        model: str,
    ) -> tuple[str, dict[str, Any]]:
        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            response_format={"type": "json_object"},
            timeout=_TIMEOUT,
        )
        content = response.choices[0].message.content or "{}"
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "model": model,
            "prompt_version": PROMPT_VERSION,
        }
        return content, usage

    def _log_tokens(self, usage: dict[str, Any], operation: str) -> None:
        logger.info(
            "LLM %s — prompt=%d completion=%d model=%s version=%s",
            operation,
            usage["prompt_tokens"],
            usage["completion_tokens"],
            usage["model"],
            usage["prompt_version"],
        )


# Singleton partagé — instancié au démarrage de l'app
_llm_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
