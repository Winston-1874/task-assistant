"""Tests de app/llm/classifier.py — mock LLM + DB en mémoire."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db import Base
from app.llm.classifier import classify_task
from app.llm.client import LLMParseError, LLMResult, LLMTransportError
from app.models import Category, CategoryNote, Context
from app.schemas import Classification


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def db() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    session = session_factory()
    try:
        yield session
    finally:
        await session.close()
        await engine.dispose()


@pytest.fixture
async def db_with_data(db: AsyncSession) -> AsyncSession:
    db.add(Category(name="client_urgent", description="Deadline fiscale imminente"))
    db.add(Category(name="admin_cabinet", description="Gestion interne"))
    db.add(Context(id=1, name="AGRIWAN SRL", kind="srl", aliases='["AGRIWAN"]'))
    await db.commit()
    return db


def make_llm_result(
    category: str = "client_urgent",
    confidence: float = 0.9,
) -> LLMResult[Classification]:
    data = Classification(
        category=category,
        urgency="haute",
        confidence=confidence,
        reasoning="Test",
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


def make_mock_llm(
    result: LLMResult[Classification] | None = None,
    raises: Exception | None = None,
) -> MagicMock:
    llm = MagicMock()
    if raises:
        llm.structured_complete = AsyncMock(side_effect=raises)
    else:
        llm.structured_complete = AsyncMock(return_value=result or make_llm_result())
    return llm


# ---------------------------------------------------------------------------
# Tests nominaux
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_classify_returns_classification_and_result(db_with_data: AsyncSession) -> None:
    llm = make_mock_llm()
    classification, llm_result = await classify_task("TVA AGRIWAN", None, db_with_data, llm=llm)

    assert classification.category == "client_urgent"
    assert llm_result is not None
    assert llm_result.raw_json
    assert llm_result.prompt_version == "1.0.0"


@pytest.mark.asyncio
async def test_classify_passes_categories_to_prompt(db_with_data: AsyncSession) -> None:
    llm = make_mock_llm()
    await classify_task("Facturation interne", None, db_with_data, llm=llm)

    messages = llm.structured_complete.call_args.kwargs["messages"]
    user_content = messages[1]["content"]

    assert "client_urgent" in user_content
    assert "admin_cabinet" in user_content


@pytest.mark.asyncio
async def test_classify_passes_contexts_to_prompt(db_with_data: AsyncSession) -> None:
    llm = make_mock_llm()
    await classify_task("Bilan AGRIWAN", None, db_with_data, llm=llm)

    messages = llm.structured_complete.call_args.kwargs["messages"]
    user_content = messages[1]["content"]

    assert "AGRIWAN" in user_content


@pytest.mark.asyncio
async def test_classify_empty_db_uses_no_category_guard(db: AsyncSession) -> None:
    """Sans catégories — le guard 'AUCUNE CATÉGORIE DÉFINIE' doit apparaître dans le prompt."""
    llm = make_mock_llm()
    await classify_task("Tâche quelconque", None, db, llm=llm)

    messages = llm.structured_complete.call_args.kwargs["messages"]
    user_content = messages[1]["content"]

    assert "AUCUNE CATÉGORIE" in user_content


@pytest.mark.asyncio
async def test_classify_with_category_notes(db: AsyncSession) -> None:
    db.add(Category(name="client_urgent", description="Urgences client"))
    db.add(CategoryNote(category="client_urgent", note="TVA = toujours haute"))
    await db.commit()

    llm = make_mock_llm()
    await classify_task("TVA AGRIWAN", None, db, llm=llm)

    messages = llm.structured_complete.call_args.kwargs["messages"]
    user_content = messages[1]["content"]

    assert "TVA = toujours haute" in user_content


# ---------------------------------------------------------------------------
# Tests fallback inbox
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fallback_on_llm_parse_error(db_with_data: AsyncSession) -> None:
    llm = make_mock_llm(raises=LLMParseError("JSON invalide"))
    classification, llm_result = await classify_task("Tâche", None, db_with_data, llm=llm)

    assert classification.category is None          # était "inbox"
    assert classification.confidence == 0.0
    assert "LLMParseError" in classification.reasoning
    assert llm_result is None


@pytest.mark.asyncio
async def test_fallback_on_llm_transport_error(db_with_data: AsyncSession) -> None:
    llm = make_mock_llm(raises=LLMTransportError("Timeout"))
    classification, llm_result = await classify_task("Tâche", None, db_with_data, llm=llm)

    assert classification.category is None          # était "inbox"
    assert "LLMTransportError" in classification.reasoning
    assert llm_result is None


# ---------------------------------------------------------------------------
# Test confidence faible
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_low_confidence_classification(db_with_data: AsyncSession) -> None:
    llm = make_mock_llm(result=make_llm_result(confidence=0.4))
    classification, llm_result = await classify_task("Tâche ambiguë", None, db_with_data, llm=llm)

    assert classification.confidence == 0.4
    assert llm_result is not None
