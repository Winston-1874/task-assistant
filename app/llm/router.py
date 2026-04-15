"""
Étage 1 — Routeur d'intention.

Stateless (pas de DB). Lève LLMTransportError si réseau mort.
Confidence < 0.6 → fallback new_task avec texte brut dans payload.
"""

import logging

from openai.types.chat import ChatCompletionMessageParam

from app.llm.client import LLMClient, LLMParseError, LLMTransportError, get_llm_client
from app.llm.prompts import INTENT_ROUTER_SYSTEM, intent_router_user
from app.schemas import Intent

logger = logging.getLogger(__name__)

_CONFIDENCE_THRESHOLD = 0.6


async def route_intent(message: str, llm: LLMClient | None = None) -> Intent:
    """
    Classe le message utilisateur dans une intention.

    - Lève LLMTransportError si le réseau est mort (pas de fallback silencieux).
    - LLMParseError → fallback new_task (JSON invalide, pas une panne réseau).
    - Confidence < 0.6 → fallback new_task avec texte brut dans payload.
    """
    if llm is None:
        llm = get_llm_client()

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": INTENT_ROUTER_SYSTEM},
        {"role": "user", "content": intent_router_user(message)},
    ]

    # Contrairement à classifier.py, LLMTransportError n'est pas avalée ici :
    # sans réseau, un message de routage ne peut pas être qualifié du tout.
    try:
        result = await llm.structured_complete(messages=messages, schema=Intent)
    except LLMParseError as e:
        logger.warning("Intent parse error — fallback new_task: %s", e)
        return _new_task_fallback(message)

    intent = result.data
    if intent.confidence < _CONFIDENCE_THRESHOLD:
        logger.info(
            "Intent confidence %.2f < %.2f — fallback new_task",
            intent.confidence,
            _CONFIDENCE_THRESHOLD,
        )
        return _new_task_fallback(message)

    logger.info("Intent routed — kind=%s confidence=%.2f", intent.kind, intent.confidence)
    return intent


def _new_task_fallback(message: str) -> Intent:
    return Intent(
        kind="new_task",
        confidence=0.0,
        payload={"raw": message},
    )
