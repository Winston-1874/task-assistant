# Bugfixes LLM + Optimistic UI + Feedback loop raisonnement — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Corriger les violations FK et erreurs de parsing LLM, afficher les cartes de tâches immédiatement avec un spinner pendant la classification async, permettre à l'utilisateur de corriger le raisonnement de l'IA, et offrir un écran de configuration simple du modèle LLM.

**Architecture:** FastAPI `BackgroundTasks` pour la classification async ; validators Pydantic `mode="before"` pour normaliser les réponses LLM non conformes ; HTMX polling côté client pour rafraîchir la carte ; table `Setting` existante pour persister le modèle LLM courant, overridé en mémoire au lifespan.

**Tech Stack:** FastAPI, SQLAlchemy async (aiosqlite), Pydantic v2, HTMX 2, Alpine.js 3, Alembic, pytest-asyncio

---

## Fichiers modifiés

| Fichier | Rôle |
|---|---|
| `app/schemas.py` | Validators urgency / confidence / Intent.payload |
| `app/llm/classifier.py` | Fallback `category=None` |
| `app/llm/prompts.py` | Contraintes enum FR explicites dans TASK_CLASSIFIER_SYSTEM |
| `app/models.py` | Champ `llm_pending: bool` sur Task |
| `migrations/versions/xxxx_add_llm_pending.py` | Migration Alembic |
| `app/routes/conversation.py` | BackgroundTasks + optimistic save + helper `_resolve_category` |
| `app/routes/fragments.py` | BackgroundTasks + poll endpoint + reasoning correction |
| `app/routes/settings.py` | Nouveau — page config + endpoints test/save modèle |
| `app/templates/fragments/task_card.html` | Mode classifying + toggle reasoning + bouton Corriger IA |
| `app/templates/settings.html` | Nouveau — page configuration modèle |
| `app/main.py` | Override llm_model depuis DB au lifespan + inclure router settings |
| `app/memory.py` | Inclure corrections `field="reasoning"` dans few-shots |
| `tests/test_classifier.py` | Mise à jour assertions fallback (inbox → None) |
| `tests/test_schemas.py` | Nouveau fichier — tests validators |
| `tests/test_routes.py` | Tests optimistic UI + poll endpoint |
| `tests/test_settings_route.py` | Nouveau — tests page config modèle |

---

## Task 0 : Configuration modèle LLM (indépendante)

**Files:**
- Create: `app/routes/settings.py`
- Create: `app/templates/settings.html`
- Modify: `app/main.py`
- Modify: `app/templates/base.html`
- Create: `tests/test_settings_route.py`

### Contexte technique

`settings.llm_model` est un attribut du singleton Pydantic `Settings` (défini dans `app/config.py`). Le client LLM lit `settings.llm_model` à chaque appel (`app/llm/client.py:95`), pas dans son constructeur. Changer `settings.llm_model` en mémoire suffit — pas besoin de recréer le client.

La table `settings` existe déjà (`app/models.py:167`) : `key TEXT PRIMARY KEY, value TEXT NOT NULL`. On y stocke `key="llm_model"`.

- [ ] **Step 1 : Écrire les tests qui échouent**

Créer `tests/test_settings_route.py` :

```python
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

    # Vérifier que settings.llm_model a été mis à jour en mémoire
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
    # Doit contenir une indication d'erreur visible
    assert "erreur" in response.text.lower() or "error" in response.text.lower() or "échec" in response.text.lower()
```

- [ ] **Step 2 : Lancer les tests pour voir les échecs**

```bash
cd /home/ambiorix/task-assistant && python -m pytest tests/test_settings_route.py -v 2>&1 | head -30
```

Attendu : `FAILED` (routes inexistantes).

- [ ] **Step 3 : Créer `app/routes/settings.py`**

```python
"""
Route configuration — GET /settings, POST /fragments/settings/model/test|save.

Permet de changer le modèle LLM OpenRouter sans redémarrer l'app.
"""

import html
import logging

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import require_auth
from app.config import settings
from app.db import get_db
from app.llm.client import LLMError, get_llm_client
from app.models import Setting
from app.templates_config import templates

logger = logging.getLogger(__name__)
router = APIRouter(dependencies=[Depends(require_auth)])


@router.get("/settings")
async def settings_page(request: Request):
    return templates.TemplateResponse(
        request,
        "settings.html",
        {"current_model": settings.llm_model},
    )


@router.post("/fragments/settings/model/test", response_class=HTMLResponse)
async def test_model(model: str = Form(...)):
    """Envoie un appel LLM minimal pour vérifier que le modèle répond."""
    model = model.strip()
    if not model:
        return _result_html(ok=False, message="Le nom du modèle est vide.")

    class _Ping(BaseModel):
        ok: bool = True

    llm = get_llm_client()
    try:
        await llm.structured_complete(
            messages=[{"role": "user", "content": 'Réponds uniquement avec {"ok": true}'}],
            schema=_Ping,
            model=model,
        )
        return _result_html(ok=True, message=f"Modèle « {html.escape(model)} » répond correctement.")
    except LLMError as e:
        logger.warning("Test modèle %r échoué : %s", model, e)
        return _result_html(ok=False, message=f"Échec : {html.escape(str(e)[:200])}")


@router.post("/fragments/settings/model/save", response_class=HTMLResponse)
async def save_model(model: str = Form(...), db: AsyncSession = Depends(get_db)):
    """Persiste le modèle dans la table settings et met à jour le singleton en mémoire."""
    model = model.strip()
    if not model:
        return _result_html(ok=False, message="Le nom du modèle est vide.")

    # Upsert dans la table settings
    from sqlalchemy import select
    result = await db.execute(select(Setting).where(Setting.key == "llm_model"))
    row = result.scalar_one_or_none()
    if row is None:
        db.add(Setting(key="llm_model", value=model))
    else:
        row.value = model
    await db.commit()

    # Override en mémoire — le client LLM lira la nouvelle valeur au prochain appel
    settings.llm_model = model
    logger.info("Modèle LLM changé → %s", model)

    return _result_html(ok=True, message=f"Modèle enregistré : « {html.escape(model)} »")


def _result_html(ok: bool, message: str) -> HTMLResponse:
    color = "text-green-700 bg-green-50 border-green-200" if ok else "text-red-700 bg-red-50 border-red-200"
    icon = "✓" if ok else "✕"
    return HTMLResponse(
        f'<p class="text-sm px-3 py-2 rounded border {color}">{icon} {message}</p>'
    )
```

- [ ] **Step 4 : Créer `app/templates/settings.html`**

```html
{% extends "base.html" %}

{% block title %}Configuration — Tasks{% endblock %}

{% block content %}
<div class="flex-1 overflow-y-auto">
  <div class="max-w-lg mx-auto px-4 py-10 space-y-8">

    <div>
      <a href="/" class="text-xs text-gray-500 hover:text-gray-700 transition-colors">← Retour</a>
      <h1 class="mt-3 text-lg font-semibold text-gray-800">Configuration</h1>
    </div>

    <section class="bg-white rounded-lg border border-gray-200 p-5 space-y-4">
      <h2 class="text-sm font-semibold text-gray-700">Modèle LLM (OpenRouter)</h2>
      <p class="text-xs text-gray-500">
        Identifiant du modèle tel qu'affiché sur
        <span class="font-mono text-gray-600">openrouter.ai/models</span>
        (ex. <span class="font-mono text-gray-600">google/gemini-2.5-flash</span>).
      </p>

      <div x-data="{ model: '{{ current_model | e }}' }" class="space-y-3">
        <input
          type="text"
          x-model="model"
          placeholder="google/gemini-2.5-flash"
          class="w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm text-gray-800
                 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-400
                 focus:border-transparent transition-colors"
        />

        <div class="flex gap-2">
          <button
            type="button"
            hx-post="/fragments/settings/model/test"
            hx-include="[name='model']"
            hx-target="#model-result"
            hx-swap="innerHTML"
            :disabled="!model.trim()"
            class="rounded-lg border border-gray-300 bg-white px-3 py-1.5 text-sm text-gray-700
                   hover:bg-gray-50 hover:border-gray-400 transition-colors disabled:opacity-40"
          >
            <input type="hidden" name="model" :value="model">
            Tester
          </button>
          <button
            type="button"
            hx-post="/fragments/settings/model/save"
            hx-include="[name='model']"
            hx-target="#model-result"
            hx-swap="innerHTML"
            :disabled="!model.trim()"
            class="rounded-lg bg-blue-600 px-3 py-1.5 text-sm text-white
                   hover:bg-blue-700 transition-colors disabled:opacity-40"
          >
            Enregistrer
          </button>
        </div>

        <div id="model-result"></div>
      </div>
    </section>

  </div>
</div>
{% endblock %}

{% block conversation %}{% endblock %}
```

- [ ] **Step 5 : Enregistrer le router dans `app/main.py`**

Ajouter l'import après les autres imports de routers :

```python
from app.routes import settings as settings_router
```

Ajouter l'inclusion après les autres routers :

```python
app.include_router(settings_router.router)   # /settings, /fragments/settings/* — protégé
```

- [ ] **Step 6 : Charger l'override DB au démarrage dans `app/main.py`**

Dans la fonction `lifespan`, après `await seed_initial_data()`, ajouter :

```python
    # Charger l'override de modèle LLM depuis la DB si présent
    from app.db import get_db_session
    from app.models import Setting as SettingModel
    from sqlalchemy import select as sa_select
    async with get_db_session() as _db:
        _r = await _db.execute(sa_select(SettingModel).where(SettingModel.key == "llm_model"))
        _row = _r.scalar_one_or_none()
        if _row:
            from app.config import settings as _settings
            _settings.llm_model = _row.value
            logger.info("Modèle LLM overridé depuis DB → %s", _row.value)
```

- [ ] **Step 7 : Ajouter le lien ⚙ dans la nav (`app/templates/base.html`)**

Dans la nav, remplacer :

```html
    <div class="ml-auto flex items-center gap-3 text-xs">
      <span class="htmx-indicator text-gray-300 text-xs animate-pulse">●</span>
      <span class="text-gray-200">|</span>
      <form method="post" action="/logout" class="inline">
        <button type="submit" class="text-gray-400 hover:text-gray-700 transition-colors">Déconnexion</button>
      </form>
    </div>
```

par :

```html
    <div class="ml-auto flex items-center gap-3 text-xs">
      <span class="htmx-indicator text-gray-300 text-xs animate-pulse">●</span>
      <span class="text-gray-200">|</span>
      <a href="/settings" title="Configuration" class="text-gray-500 hover:text-gray-700 transition-colors">⚙</a>
      <span class="text-gray-200">|</span>
      <form method="post" action="/logout" class="inline">
        <button type="submit" class="text-gray-500 hover:text-gray-700 transition-colors">Déconnexion</button>
      </form>
    </div>
```

- [ ] **Step 8 : Lancer les tests**

```bash
cd /home/ambiorix/task-assistant && python -m pytest tests/test_settings_route.py -v
```

Attendu : tous `PASSED`.

- [ ] **Step 9 : Lancer la suite complète**

```bash
cd /home/ambiorix/task-assistant && python -m pytest tests/ -v 2>&1 | tail -20
```

Attendu : tous `PASSED`.

- [ ] **Step 10 : Commit**

```bash
cd /home/ambiorix/task-assistant && git add app/routes/settings.py app/templates/settings.html app/main.py app/templates/base.html tests/test_settings_route.py
git commit -m "feat: add LLM model configuration page with connectivity test"
```

---

## Task 1 : Validators LLM dans `app/schemas.py`

**Files:**
- Modify: `app/schemas.py`
- Create: `tests/test_schemas.py`

- [ ] **Step 1 : Écrire les tests qui échouent**

Créer `tests/test_schemas.py` :

```python
"""Tests des validators de normalisation LLM dans app/schemas.py."""
import pytest
from app.schemas import Classification, Intent


# ---------------------------------------------------------------------------
# Classification.urgency
# ---------------------------------------------------------------------------

def test_urgency_alias_high():
    c = Classification(category="x", urgency="high", confidence=0.8, reasoning="r", needs_due_date=False, tags=[])
    assert c.urgency == "haute"

def test_urgency_alias_urgent():
    c = Classification(category="x", urgency="urgent", confidence=0.8, reasoning="r", needs_due_date=False, tags=[])
    assert c.urgency == "critique"

def test_urgency_alias_low():
    c = Classification(category="x", urgency="low", confidence=0.8, reasoning="r", needs_due_date=False, tags=[])
    assert c.urgency == "basse"

def test_urgency_alias_medium():
    c = Classification(category="x", urgency="medium", confidence=0.8, reasoning="r", needs_due_date=False, tags=[])
    assert c.urgency == "normale"

def test_urgency_fr_passthrough():
    c = Classification(category="x", urgency="critique", confidence=0.8, reasoning="r", needs_due_date=False, tags=[])
    assert c.urgency == "critique"

def test_urgency_unknown_raises():
    with pytest.raises(Exception):
        Classification(category="x", urgency="blocker", confidence=0.8, reasoning="r", needs_due_date=False, tags=[])


# ---------------------------------------------------------------------------
# Classification.confidence
# ---------------------------------------------------------------------------

def test_confidence_string_high():
    c = Classification(category="x", urgency="normale", confidence="high", reasoning="r", needs_due_date=False, tags=[])
    assert c.confidence == 0.9

def test_confidence_string_medium():
    c = Classification(category="x", urgency="normale", confidence="medium", reasoning="r", needs_due_date=False, tags=[])
    assert c.confidence == 0.6

def test_confidence_string_low():
    c = Classification(category="x", urgency="normale", confidence="low", reasoning="r", needs_due_date=False, tags=[])
    assert c.confidence == 0.3

def test_confidence_numeric_string():
    c = Classification(category="x", urgency="normale", confidence="0.75", reasoning="r", needs_due_date=False, tags=[])
    assert c.confidence == 0.75

def test_confidence_float_passthrough():
    c = Classification(category="x", urgency="normale", confidence=0.85, reasoning="r", needs_due_date=False, tags=[])
    assert c.confidence == 0.85


# ---------------------------------------------------------------------------
# Intent.payload
# ---------------------------------------------------------------------------

def test_intent_payload_dict_passthrough():
    intent = Intent(kind="new_task", confidence=0.9, payload={"title": "foo"})
    assert intent.payload == {"title": "foo"}

def test_intent_payload_scalar_wrapped():
    intent = Intent(kind="new_task", confidence=0.9, payload="texte brut")
    assert intent.payload == {"raw": "texte brut"}

def test_intent_payload_int_wrapped():
    intent = Intent(kind="query", confidence=0.9, payload=42)
    assert intent.payload == {"raw": 42}
```

- [ ] **Step 2 : Vérifier que les tests échouent**

```bash
cd /home/ambiorix/task-assistant && python -m pytest tests/test_schemas.py -v 2>&1 | head -40
```

Attendu : plusieurs `FAILED` ou `ERROR`.

- [ ] **Step 3 : Implémenter les validators dans `app/schemas.py`**

Remplacer le contenu de `app/schemas.py` :

```python
from datetime import date
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator

_URGENCY_ALIASES: dict[str, str] = {
    "urgent": "critique",
    "critical": "critique",
    "high": "haute",
    "medium": "normale",
    "normal": "normale",
    "low": "basse",
}

_CONFIDENCE_ALIASES: dict[str, float] = {
    "high": 0.9,
    "medium": 0.6,
    "low": 0.3,
}


class Intent(BaseModel):
    kind: Literal["new_task", "command", "query", "update_context"]
    confidence: float = Field(ge=0.0, le=1.0)
    payload: dict[str, Any]

    @field_validator("payload", mode="before")
    @classmethod
    def _normalize_payload(cls, v: Any) -> dict[str, Any]:
        if isinstance(v, dict):
            return v
        return {"raw": v}


class Classification(BaseModel):
    category: str | None = None  # nom existant, "NEW:<nom>", ou None (fallback)
    context_id: int | None = None
    context_suggestion: str | None = None
    urgency: Literal["critique", "haute", "normale", "basse"]
    due_date: date | None = None
    needs_due_date: bool = False
    estimated_minutes: int | None = None
    tags: Annotated[list[str], Field(max_length=5)] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str

    @field_validator("urgency", mode="before")
    @classmethod
    def _normalize_urgency(cls, v: Any) -> str:
        if isinstance(v, str):
            normalized = v.lower().strip()
            return _URGENCY_ALIASES.get(normalized, normalized)
        return v

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, v: Any) -> float:
        if isinstance(v, str):
            lower = v.lower().strip()
            if lower in _CONFIDENCE_ALIASES:
                return _CONFIDENCE_ALIASES[lower]
            return float(lower)
        return v


# ---------------------------------------------------------------------------
# Étage 3 — Dialogues proactifs
# ---------------------------------------------------------------------------


class ProactiveMessage(BaseModel):
    """Réponse LLM pour ask_due_date et zombie_check."""

    message: str


class SignalPriority(BaseModel):
    """Une entrée dans la liste priorisée de la colonne Signal."""

    task_title: str
    reason: str


class SignalResponse(BaseModel):
    """Réponse LLM pour generate_signal — wrappée pour json_object."""

    priorities: list[SignalPriority] = Field(default_factory=list)


class DigestContent(BaseModel):
    """Réponse LLM pour le digest matinal."""

    summary: str
    top_tasks: list[str] = Field(default_factory=list)
    alert: str | None = None
```

- [ ] **Step 4 : Lancer les tests**

```bash
cd /home/ambiorix/task-assistant && python -m pytest tests/test_schemas.py -v
```

Attendu : tous `PASSED`.

- [ ] **Step 5 : Vérifier que les tests existants passent encore**

```bash
cd /home/ambiorix/task-assistant && python -m pytest tests/ -v 2>&1 | tail -20
```

Note : `test_fallback_on_llm_parse_error` et `test_fallback_on_llm_transport_error` dans `test_classifier.py` vont maintenant échouer car ils vérifient `category == "inbox"`. C'est attendu — ils seront corrigés en Task 2.

- [ ] **Step 6 : Commit**

```bash
cd /home/ambiorix/task-assistant && git add app/schemas.py tests/test_schemas.py
git commit -m "fix: normalize LLM enum/confidence/payload via Pydantic field_validators"
```

---

## Task 2 : Corriger le fallback dans `app/llm/classifier.py`

**Files:**
- Modify: `app/llm/classifier.py`
- Modify: `tests/test_classifier.py`

- [ ] **Step 1 : Mettre à jour les tests du fallback**

Dans `tests/test_classifier.py`, remplacer les deux tests de fallback :

```python
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
```

- [ ] **Step 2 : Vérifier que les tests échouent**

```bash
cd /home/ambiorix/task-assistant && python -m pytest tests/test_classifier.py::test_fallback_on_llm_parse_error tests/test_classifier.py::test_fallback_on_llm_transport_error -v
```

Attendu : `FAILED` (classification.category est encore `"inbox"`).

- [ ] **Step 3 : Corriger le fallback dans `classifier.py`**

À la ligne 66-73, remplacer la construction du fallback :

```python
    except (LLMParseError, LLMTransportError) as e:
        logger.warning("Classification fallback: %s", e)
        fallback = Classification(
            category=None,
            urgency="normale",
            confidence=0.0,
            reasoning=f"Fallback automatique : {type(e).__name__}",
            needs_due_date=False,
            tags=[],
        )
        return fallback, None
```

- [ ] **Step 4 : Lancer les tests classifier**

```bash
cd /home/ambiorix/task-assistant && python -m pytest tests/test_classifier.py -v
```

Attendu : tous `PASSED`.

- [ ] **Step 5 : Commit**

```bash
cd /home/ambiorix/task-assistant && git add app/llm/classifier.py tests/test_classifier.py
git commit -m "fix: fallback classification uses category=None instead of 'inbox' to avoid FK violation"
```

---

## Task 3 : Durcissement prompt + helper `_resolve_category`

**Files:**
- Modify: `app/llm/prompts.py`
- Modify: `app/routes/conversation.py`
- Modify: `app/routes/fragments.py`

- [ ] **Step 1 : Ajouter les contraintes au prompt système dans `app/llm/prompts.py`**

Dans `TASK_CLASSIFIER_SYSTEM`, après la ligne `- reasoning : 1-2 phrases maximum, factuel, en français.`, ajouter :

```python
TASK_CLASSIFIER_SYSTEM = """\
Tu es un assistant de classification de tâches pour un expert-comptable fiscaliste belge.
Tu reçois une tâche à classer et tu dois retourner un objet JSON structuré.

Règles :
- Choisis la catégorie la plus précise parmi celles fournies.
- Si aucune ne convient, propose "NEW:<nom_suggéré>".
- urgency DOIT être exactement l'un de : "critique" | "haute" | "normale" | "basse" — jamais "high", "low", "urgent" ni autre valeur.
- confidence DOIT être un nombre décimal entre 0.0 et 1.0 (ex. 0.85) — jamais "high", "low" ni "medium".
- due_date : extrais les dates explicites ou implicites ("fin du mois", "avant la déclaration TVA").
  Format ISO 8601 (YYYY-MM-DD) ou null.
- needs_due_date : true si la tâche semble avoir une échéance mais qu'elle n'est pas exprimée.
- reasoning : 1-2 phrases maximum, factuel, en français.
- tags : mots-clés utiles, 5 maximum.

Réponds UNIQUEMENT avec un objet JSON valide. Aucun texte avant ou après.
"""
```

- [ ] **Step 2 : Ajouter `_resolve_category` dans `app/routes/conversation.py`**

En bas du fichier `app/routes/conversation.py`, ajouter la fonction helper :

```python
def _resolve_category(category: str | None) -> str | None:
    """Retourne None si la catégorie est un préfixe NEW: (LLM hallucination) ou déjà None."""
    if category and category.startswith("NEW:"):
        return None
    return category
```

Puis dans `_handle_new_task`, remplacer `category=classification.category` par :

```python
resolved_category = _resolve_category(classification.category)
task = Task(
    title=message[:200],
    category=resolved_category,
    context_id=classification.context_id,
    urgency=classification.urgency,
    due_date=classification.due_date,
    estimated_minutes=classification.estimated_minutes,
    tags=json.dumps(classification.tags) if classification.tags else None,
    needs_review=1 if (classification.confidence < 0.7 or resolved_category is None) else 0,
    llm_confidence=classification.confidence,
    llm_reasoning=classification.reasoning,
    llm_raw_response=llm_result.raw_json if llm_result else None,
)
```

- [ ] **Step 3 : Même helper dans `app/routes/fragments.py`**

En bas du fichier `app/routes/fragments.py`, ajouter la même fonction :

```python
def _resolve_category(category: str | None) -> str | None:
    """Retourne None si la catégorie est un préfixe NEW: (LLM hallucination) ou déjà None."""
    if category and category.startswith("NEW:"):
        return None
    return category
```

Dans `create_task`, remplacer `category=classification.category` par :

```python
resolved_category = _resolve_category(classification.category)
task = Task(
    title=title,
    description=description or None,
    category=resolved_category,
    context_id=classification.context_id,
    urgency=classification.urgency,
    due_date=classification.due_date,
    estimated_minutes=classification.estimated_minutes,
    tags=json.dumps(classification.tags) if classification.tags else None,
    needs_review=1 if (classification.confidence < 0.7 or resolved_category is None) else 0,
    llm_confidence=classification.confidence,
    llm_reasoning=classification.reasoning,
    llm_raw_response=llm_result.raw_json if llm_result else None,
)
```

- [ ] **Step 4 : Lancer la suite de tests**

```bash
cd /home/ambiorix/task-assistant && python -m pytest tests/ -v 2>&1 | tail -20
```

Attendu : tous `PASSED`.

- [ ] **Step 5 : Commit**

```bash
cd /home/ambiorix/task-assistant && git add app/llm/prompts.py app/routes/conversation.py app/routes/fragments.py
git commit -m "fix: strip NEW: category prefix, harden prompt with explicit FR enum constraints"
```

---

## Task 4 : Migration DB — champ `llm_pending`

**Files:**
- Modify: `app/models.py`
- Create: `migrations/versions/XXXX_add_llm_pending_to_tasks.py`

- [ ] **Step 1 : Ajouter le champ au modèle `Task` dans `app/models.py`**

Après la ligne `needs_review: Mapped[int] = mapped_column(Integer, default=0)` (ligne ~69), ajouter :

```python
    llm_pending: Mapped[bool] = mapped_column(Boolean, default=False)
```

Vérifier que `Boolean` est bien dans les imports SQLAlchemy en haut du fichier (il y est déjà).

- [ ] **Step 2 : Générer la migration Alembic**

```bash
cd /home/ambiorix/task-assistant && python -m alembic revision --autogenerate -m "add_llm_pending_to_tasks"
```

Vérifier le fichier généré dans `migrations/versions/` — il doit contenir un `op.add_column` pour `llm_pending`.

- [ ] **Step 3 : Appliquer la migration sur la DB de développement**

```bash
cd /home/ambiorix/task-assistant && python -m alembic upgrade head
```

Attendu : `Running upgrade ... -> XXXX`.

- [ ] **Step 4 : Commit**

```bash
cd /home/ambiorix/task-assistant && git add app/models.py migrations/
git commit -m "feat: add llm_pending bool column to tasks for optimistic UI"
```

---

## Task 5 : Optimistic UI — routes backend

**Files:**
- Modify: `app/routes/conversation.py`
- Modify: `app/routes/fragments.py`

La `BackgroundTask` a besoin d'ouvrir sa propre session DB. `app/db.py` exporte déjà `get_db_session()` (context manager async) utilisé par le scheduler — on le réutilise.

- [ ] **Step 1 : Écrire les tests routes pour le comportement optimistic**

Dans `tests/test_routes.py`, ajouter après les imports existants et les fixtures :

```python
@pytest.mark.asyncio
async def test_conversation_parse_returns_card_immediately(client, auth_cookies, db_session, monkeypatch):
    """La route retourne une carte (llm_pending) sans attendre le LLM."""
    # On bloque volontairement la classification LLM pour vérifier que la réponse est immédiate
    async def slow_classify(*args, **kwargs):
        import asyncio
        await asyncio.sleep(100)  # ne sera jamais atteint car BackgroundTask

    monkeypatch.setattr("app.routes.conversation.classify_task", slow_classify)

    response = await client.post(
        "/conversation/parse",
        data={"message": "TVA AGRIWAN à envoyer"},
        cookies=auth_cookies,
    )
    assert response.status_code == 200
    html = response.text
    assert "TVA AGRIWAN" in html


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
    # Carte complète ne contient pas le trigger de polling
    assert 'hx-trigger="every 2s"' not in response.text
```

- [ ] **Step 2 : Lancer les tests pour voir les échecs**

```bash
cd /home/ambiorix/task-assistant && python -m pytest tests/test_routes.py::test_poll_endpoint_returns_spinner_while_pending tests/test_routes.py::test_poll_endpoint_returns_full_card_when_ready -v
```

Attendu : `FAILED` (endpoint inexistant).

- [ ] **Step 3 : Refactoriser `app/routes/conversation.py` pour optimistic UI**

Remplacer le contenu complet de `app/routes/conversation.py` :

```python
"""
Route conversationnelle — POST /conversation/parse.

Reçoit un message libre, passe par le routeur d'intention, puis exécute l'action.
Retourne un fragment HTML inséré dans le toast de réponse.
"""

import html
import json
import logging
from datetime import date

from fastapi import APIRouter, BackgroundTasks, Depends, Form, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import require_auth
from app.db import get_db, get_db_session
from app.llm.classifier import classify_task
from app.llm.client import LLMTransportError
from app.llm.proactive import ask_due_date
from app.llm.router import route_intent
from app.models import Task
from app.templates_config import templates

logger = logging.getLogger(__name__)
router = APIRouter(dependencies=[Depends(require_auth)])


@router.post("/conversation/parse", response_class=HTMLResponse)
async def parse_message(
    request: Request,
    background_tasks: BackgroundTasks,
    message: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    try:
        intent = await route_intent(message)
    except LLMTransportError as e:
        logger.error("Réseau LLM mort lors du routage: %s", e)
        return _error_html("Service LLM indisponible. Réessaie dans quelques instants.")

    if intent.kind == "new_task":
        return await _handle_new_task(request, background_tasks, message, db)

    labels = {
        "command": "Commande détectée — fonctionnalité à venir.",
        "query": "Question sur le backlog — fonctionnalité à venir.",
        "update_context": "Info contexte notée — fonctionnalité à venir.",
    }
    text = html.escape(labels.get(intent.kind, "Message reçu."))
    return HTMLResponse(f'<div class="toast toast-info">{text}</div>')


async def _handle_new_task(
    request: Request,
    background_tasks: BackgroundTasks,
    message: str,
    db: AsyncSession,
) -> HTMLResponse:
    task = Task(
        title=message[:200],
        llm_pending=True,
        status="open",
    )
    db.add(task)
    await db.commit()
    await db.refresh(task)

    background_tasks.add_task(_classify_and_update, task.id, message, None)

    return templates.TemplateResponse(
        request,
        "fragments/task_card.html",
        {"task": task, "pending": None, "today": date.today()},
    )


async def _classify_and_update(task_id: int, title: str, description: str | None) -> None:
    """Classifie la tâche en arrière-plan et met à jour la DB."""
    async with get_db_session() as db:
        from sqlalchemy import select
        result = await db.execute(select(Task).where(Task.id == task_id))
        task = result.scalar_one_or_none()
        if task is None:
            logger.warning("_classify_and_update: tâche %d introuvable", task_id)
            return

        try:
            classification, llm_result = await classify_task(title, description, db)
        except Exception as e:
            logger.error("_classify_and_update: erreur classification tâche %d: %s", task_id, e)
            task.llm_pending = False
            task.needs_review = 1
            await db.commit()
            return

        resolved_category = _resolve_category(classification.category)
        task.category = resolved_category
        task.context_id = classification.context_id
        task.urgency = classification.urgency
        task.due_date = classification.due_date
        task.estimated_minutes = classification.estimated_minutes
        task.tags = json.dumps(classification.tags) if classification.tags else None
        task.needs_review = 1 if (classification.confidence < 0.7 or resolved_category is None) else 0
        task.llm_confidence = classification.confidence
        task.llm_reasoning = classification.reasoning
        task.llm_raw_response = llm_result.raw_json if llm_result else None
        task.llm_pending = False

        if classification.needs_due_date and task.due_date is None:
            await ask_due_date(task, db)

        await db.commit()


def _resolve_category(category: str | None) -> str | None:
    """Retourne None si la catégorie est un préfixe NEW: (LLM hallucination) ou déjà None."""
    if category and category.startswith("NEW:"):
        return None
    return category


def _error_html(message: str) -> HTMLResponse:
    return HTMLResponse(
        f'<div class="toast toast-error" role="alert">{html.escape(message)}</div>'
    )
```

- [ ] **Step 4 : Ajouter le poll endpoint dans `app/routes/fragments.py`**

Après la route `delete_task` (vers ligne 120), insérer :

```python
# ---------------------------------------------------------------------------
# Poll statut classification (optimistic UI)
# ---------------------------------------------------------------------------


@router.get("/tasks/{task_id}/status", response_class=HTMLResponse)
async def task_status(
    request: Request,
    task_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Retourne la carte en mode 'classifying' si llm_pending=True,
    ou la carte complète si la classification est terminée.
    HTMX polling s'arrête automatiquement quand la carte sans hx-trigger est retournée.
    """
    task = await _get_task_or_404(db, task_id)
    today = date.today()

    pending_prompt = None
    if not task.llm_pending:
        from sqlalchemy import select as sa_select
        pp_result = await db.execute(
            sa_select(PendingPrompt).where(
                PendingPrompt.task_id == task_id,
                PendingPrompt.resolved_at.is_(None),
            )
        )
        pending_prompt = pp_result.scalar_one_or_none()

    return templates.TemplateResponse(
        request,
        "fragments/task_card.html",
        {"task": task, "pending": pending_prompt, "today": today},
    )
```

Mettre à jour la route `create_task` dans `fragments.py` pour optimistic UI (même pattern que `conversation.py`) :

```python
@router.post("/tasks", response_class=HTMLResponse)
async def create_task(
    request: Request,
    background_tasks: BackgroundTasks,
    title: str = Form(...),
    description: str = Form(""),
    db: AsyncSession = Depends(get_db),
):
    task = Task(
        title=title,
        description=description or None,
        llm_pending=True,
        status="open",
    )
    db.add(task)
    await db.commit()
    await db.refresh(task)

    from app.routes.conversation import _classify_and_update
    background_tasks.add_task(_classify_and_update, task.id, title, description or None)

    return templates.TemplateResponse(
        request,
        "fragments/task_card.html",
        {"task": task, "pending": None, "today": date.today()},
    )
```

Ajouter `BackgroundTasks` aux imports de `fragments.py` :
```python
from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException, Request
```

- [ ] **Step 5 : Lancer les tests**

```bash
cd /home/ambiorix/task-assistant && python -m pytest tests/test_routes.py -v 2>&1 | tail -30
```

Attendu : les nouveaux tests `test_poll_endpoint_*` passent. Les anciens tests peuvent nécessiter une mise à jour si ils vérifiaient la classification synchrone — les ajuster si besoin pour mocker `_classify_and_update`.

- [ ] **Step 6 : Commit**

```bash
cd /home/ambiorix/task-assistant && git add app/routes/conversation.py app/routes/fragments.py tests/test_routes.py
git commit -m "feat: optimistic task creation with BackgroundTasks classification and HTMX polling"
```

---

## Task 6 : Frontend — carte classifying + toggle reasoning + bouton Corriger IA

**Files:**
- Modify: `app/templates/fragments/task_card.html`

Règles de contraste à respecter sur tous les nouveaux éléments :
- Texte sur fond clair : minimum `text-gray-600` (pas `text-gray-300` ni `text-gray-400` pour du contenu lisible)
- Spinner : `text-blue-500` sur fond blanc
- Bouton Corriger IA : `text-gray-600 hover:text-blue-600` — visible sans hover
- Textarea correction : `border-gray-300`, placeholder `text-gray-400`, texte `text-gray-800`

- [ ] **Step 1 : Remplacer le contenu complet de `task_card.html`**

```html
{# Fragment réutilisable — inclus dans index.html et retourné par les endpoints HTMX #}
{# Variables attendues : task, pending (PendingPrompt|None), today (date) #}

{% set urgency_border = {
  "critique": "border-l-red-500",
  "haute":    "border-l-orange-400",
  "normale":  "border-l-blue-400",
  "basse":    "border-l-gray-300",
}.get(task.urgency, "border-l-gray-200") %}

{% set urgency_label = {
  "critique": ("Critique", "text-red-700 bg-red-50"),
  "haute":    ("Haute",    "text-orange-700 bg-orange-50"),
  "normale":  ("Normale",  "text-blue-700 bg-blue-50"),
  "basse":    ("Basse",    "text-gray-600 bg-gray-100"),
}.get(task.urgency, ("—", "text-gray-500 bg-gray-50")) %}

{# Mode classifying : carte temporaire avec spinner HTMX polling #}
{% if task.llm_pending %}
<div
  id="task-{{ task.id }}"
  hx-get="/fragments/tasks/{{ task.id }}/status"
  hx-trigger="every 2s"
  hx-swap="outerHTML"
  class="bg-white rounded-lg border border-gray-200 border-l-4 border-l-blue-300 shadow-sm"
>
  <div class="flex items-center gap-3 px-4 py-3">
    <div class="shrink-0 w-4 h-4 rounded-full border-2 border-blue-400 border-t-transparent animate-spin"></div>
    <p class="text-sm font-medium text-gray-800 flex-1">{{ task.title }}</p>
    <span class="text-xs text-gray-500 italic">Classification…</span>
  </div>
</div>
{% else %}

<div
  id="task-{{ task.id }}"
  data-title="{{ task.title | truncate(40, true, '…') | e }}"
  x-data="{ visible: true, submitting: false, reasoningExpanded: false, correctingReasoning: false }"
  x-show="visible"
  x-transition:leave="transition duration-150"
  x-transition:leave-start="opacity-100"
  x-transition:leave-end="opacity-0"
  class="group bg-white rounded-lg border border-gray-200 border-l-4 {{ urgency_border }} shadow-sm hover:shadow transition-shadow"
>

  {# Bandeau pending prompt #}
  {% if pending and pending.task_id == task.id %}
  <div class="px-4 pt-3 pb-2 border-b border-amber-100 bg-amber-50 rounded-t-lg">
    <p class="text-xs font-medium text-amber-800 mb-2">{{ pending.message }}</p>
    <form
      hx-post="/fragments/tasks/{{ task.id }}/prompt/{{ pending.id }}/answer"
      hx-target="#task-{{ task.id }}"
      hx-swap="outerHTML"
      class="flex flex-wrap gap-1"
    >
      {% set quick_dates = [
        ("Aujourd'hui", today.isoformat()),
        ("Demain",      (today | dateadd(1)).isoformat()),
        ("Cette semaine",(today | dateadd(7)).isoformat()),
        ("+14j",        (today | dateadd(14)).isoformat()),
        ("Pas de deadline", "no_deadline"),
      ] %}
      {% for label, val in quick_dates %}
      <button
        type="submit"
        name="answer"
        value="{{ val }}"
        class="rounded-full border border-amber-300 bg-white px-2.5 py-0.5 text-xs text-amber-800 hover:bg-amber-100 transition-colors"
      >{{ label }}</button>
      {% endfor %}
    </form>
  </div>
  {% endif %}

  {# Corps de la card #}
  <div class="flex items-start gap-3 px-4 py-3">

    {# Checkbox done #}
    {% if task.status == "done" %}
    <form
      hx-post="/fragments/tasks/{{ task.id }}/undo/done"
      hx-target="#task-{{ task.id }}"
      hx-swap="outerHTML"
      class="mt-0.5 shrink-0"
    >
      <button
        type="submit"
        title="Annuler done"
        class="w-4 h-4 rounded border-2 border-blue-400 bg-blue-400 flex items-center justify-center text-white text-xs"
      >✓</button>
    </form>
    {% else %}
    <button
      type="button"
      title="Marquer comme fait"
      class="mt-0.5 shrink-0 w-4 h-4 rounded border-2 border-gray-300 hover:border-blue-400 flex items-center justify-center transition-colors disabled:opacity-40"
      x-bind:disabled="submitting"
      x-on:click="
        if (submitting) return;
        submitting = true;
        htmx.ajax('POST', '/fragments/tasks/{{ task.id }}/done', {
          target: '#task-{{ task.id }}',
          swap: 'outerHTML',
        });
        $store.undo.push(
          '« ' + $el.closest('[id^=task-]').dataset.title + ' » marquée faite',
          null,
          () => htmx.ajax('POST', '/fragments/tasks/{{ task.id }}/undo/done', {
            target: '#task-{{ task.id }}',
            swap: 'outerHTML',
          })
        );
      "
    ></button>
    {% endif %}

    {# Contenu #}
    <div class="flex-1 min-w-0">

      {# Titre #}
      <p class="text-sm font-medium text-gray-800 leading-snug
        {% if task.status == 'done' %}line-through text-gray-400{% endif %}">
        {{ task.title }}
      </p>

      {# Meta ligne #}
      <div class="flex flex-wrap items-center gap-x-2 gap-y-0.5 mt-1">

        {% if task.category %}
        <span class="text-xs text-gray-500 truncate max-w-[12rem]">{{ task.category | truncate(30, true, '…') }}</span>
        {% endif %}

        {% if task.due_date %}
        <span class="text-xs {% if task.due_date <= today %}text-red-600 font-medium{% else %}text-gray-500{% endif %}">
          {{ task.due_date.strftime('%d/%m') }}
        </span>
        {% endif %}

        {% if task.estimated_minutes %}
        <span class="text-xs text-gray-500">~{{ task.estimated_minutes }}min</span>
        {% endif %}

        {% if task.urgency and task.urgency != 'normale' %}
        <span class="text-xs px-1.5 py-0 rounded-full {{ urgency_label[1] }}">{{ urgency_label[0] }}</span>
        {% endif %}

        {% if task.needs_review %}
        <span class="text-xs text-amber-600 font-medium">⚠ à revoir</span>
        {% endif %}

      </div>

      {# Raisonnement LLM avec toggle #}
      {% if task.llm_reasoning %}
      <div class="mt-1" x-data="{ expanded: false }">
        <p
          class="text-xs text-gray-500 italic leading-relaxed"
          :class="expanded ? '' : 'line-clamp-2'"
        >{{ task.llm_reasoning }}</p>
        {% if task.llm_reasoning | length > 120 %}
        <button
          type="button"
          class="text-xs text-blue-600 hover:text-blue-800 mt-0.5 transition-colors"
          x-on:click="expanded = !expanded"
          x-text="expanded ? 'voir moins' : 'voir plus'"
        ></button>
        {% endif %}
      </div>
      {% endif %}

      {# Zone correction raisonnement IA #}
      <div x-show="correctingReasoning" x-cloak class="mt-2">
        <form
          hx-post="/fragments/tasks/{{ task.id }}/correct"
          hx-target="#task-{{ task.id }}"
          hx-swap="outerHTML"
          x-on:htmx:after-request="correctingReasoning = false"
          class="flex flex-col gap-1"
        >
          <input type="hidden" name="field" value="reasoning">
          <textarea
            name="new_value"
            rows="2"
            placeholder="Décris ce qui est incorrect dans ce raisonnement…"
            class="w-full resize-none rounded border border-gray-300 bg-white px-2 py-1.5 text-xs text-gray-800 placeholder:text-gray-400 focus:outline-none focus:ring-1 focus:ring-blue-400"
          ></textarea>
          <div class="flex gap-2 justify-end">
            <button
              type="button"
              class="text-xs text-gray-500 hover:text-gray-700 transition-colors"
              x-on:click="correctingReasoning = false"
            >Annuler</button>
            <button
              type="submit"
              class="text-xs bg-blue-600 text-white px-2 py-1 rounded hover:bg-blue-700 transition-colors"
            >Envoyer</button>
          </div>
        </form>
      </div>

    </div>

    {# Actions (visibles au hover) #}
    <div class="shrink-0 flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">

      {# Bouton Corriger IA — visible seulement si raisonnement présent #}
      {% if task.llm_reasoning %}
      <button
        type="button"
        title="Corriger le raisonnement IA"
        class="p-1 rounded text-gray-500 hover:text-blue-600 hover:bg-blue-50 transition-colors text-xs"
        x-on:click="correctingReasoning = !correctingReasoning"
      >🧠</button>
      {% endif %}

      <button
        type="button"
        title="Corriger"
        class="p-1 rounded text-gray-500 hover:text-gray-700 hover:bg-gray-100 transition-colors text-xs"
        x-data
        x-on:click="$dispatch('open-correction', { taskId: {{ task.id }} })"
      >✎</button>

      <button
        type="button"
        title="Supprimer"
        class="p-1 rounded text-gray-500 hover:text-red-600 hover:bg-red-50 transition-colors text-xs"
        x-on:click="
          visible = false;
          $store.undo.push(
            '« ' + $el.closest('[id^=task-]').dataset.title + ' » supprimée',
            () => htmx.ajax('DELETE', '/fragments/tasks/{{ task.id }}', {
              target: '#task-{{ task.id }}',
              swap: 'outerHTML',
            }),
            () => { visible = true; }
          );
        "
      >✕</button>

    </div>

  </div>
</div>
{% endif %}
```

- [ ] **Step 2 : Vérifier visuellement dans le navigateur**

Démarrer l'app :
```bash
cd /home/ambiorix/task-assistant && uvicorn app.main:app --reload --port 8000
```

Tester :
1. Saisir une tâche → la carte doit apparaître immédiatement avec le spinner "Classification…"
2. Après ~2s, la carte doit se mettre à jour avec le raisonnement et les métadonnées
3. Sur une carte avec raisonnement long, "voir plus" doit afficher le texte complet
4. Au hover, le bouton 🧠 doit être visible avec un contraste suffisant (`text-gray-500`)
5. Cliquer 🧠 → textarea apparaît avec bordure visible et placeholder lisible

- [ ] **Step 3 : Commit**

```bash
cd /home/ambiorix/task-assistant && git add app/templates/fragments/task_card.html
git commit -m "feat: add classifying spinner, reasoning toggle, and AI correction button to task card"
```

---

## Task 7 : Feedback loop — correction raisonnement + injection few-shots

**Files:**
- Modify: `app/routes/fragments.py`
- Modify: `app/memory.py`
- Modify: `app/llm/prompts.py`

- [ ] **Step 1 : Étendre les champs autorisés dans `fragments.py`**

Dans `correct_task`, remplacer :

```python
_ALLOWED_CORRECTION_FIELDS = {"urgency", "category", "due_date", "context_id"}
```

par :

```python
_ALLOWED_CORRECTION_FIELDS = {"urgency", "category", "due_date", "context_id", "reasoning"}
```

Ajouter le cas `reasoning` dans la logique de mise à jour, après le bloc `elif field == "context_id":` :

```python
    elif field == "reasoning":
        task.llm_reasoning = new_value
```

- [ ] **Step 2 : Inclure les corrections reasoning dans `app/memory.py`**

La fonction `select_few_shots` est déjà générique (pas de filtre sur `field`). Aucune modification de la requête nécessaire — les corrections `field="reasoning"` seront automatiquement incluses.

Modifier uniquement `_to_few_shot_dicts` pour formater différemment les corrections reasoning (sinon le prompt affiche `field: reasoning, old: ..., new: ...` ce qui est peu lisible) :

```python
def _to_few_shot_dicts(corrections: list[Correction]) -> list[FewShotDict]:
    return [
        FewShotDict(
            task_title=c.task_title or "",
            task_description=c.task_description,
            field=c.field,
            old_value=c.old_value,
            new_value=c.new_value,
        )
        for c in corrections
    ]
```

Pas de changement ici — le formatage est géré dans `prompts.py`.

- [ ] **Step 3 : Adapter le formatage few-shots dans `app/llm/prompts.py`**

Dans la fonction `task_classifier_user`, remplacer le bloc few-shots :

```python
    if few_shots:
        parts.append("\n=== EXEMPLES DE CORRECTIONS PASSÉES (apprends de ces patterns) ===")
        for fs in few_shots:
            old = fs["old_value"] or "—"
            new = fs["new_value"] or "—"
            desc = f" ({fs['task_description'][:80]})" if fs.get("task_description") else ""
            if fs["field"] == "reasoning":
                parts.append(
                    f'- "{fs["task_title"]}"{desc}\n'
                    f"  Raisonnement corrigé : {new!r}"
                )
            else:
                parts.append(
                    f'- "{fs["task_title"]}"{desc}\n'
                    f"  Correction : {fs['field']} était {old!r}, doit être {new!r}"
                )
```

- [ ] **Step 4 : Écrire un test pour la correction reasoning**

Dans `tests/test_routes.py`, ajouter :

```python
@pytest.mark.asyncio
async def test_correct_reasoning_updates_task(client, auth_cookies, db_session):
    """POST /fragments/tasks/{id}/correct avec field=reasoning met à jour llm_reasoning."""
    from app.models import Task
    task = Task(title="Test reasoning", urgency="normale", status="open",
                llm_reasoning="Raisonnement initial incorrect", llm_pending=False)
    db_session.add(task)
    await db_session.commit()
    await db_session.refresh(task)

    response = await client.post(
        f"/fragments/tasks/{task.id}/correct",
        data={"field": "reasoning", "new_value": "En fait c'est urgent car deadline fiscale"},
        cookies=auth_cookies,
    )
    assert response.status_code == 200

    await db_session.refresh(task)
    assert task.llm_reasoning == "En fait c'est urgent car deadline fiscale"
    assert task.was_corrected == 1
```

- [ ] **Step 5 : Lancer les tests**

```bash
cd /home/ambiorix/task-assistant && python -m pytest tests/test_routes.py::test_correct_reasoning_updates_task -v
```

Attendu : `PASSED`.

- [ ] **Step 6 : Lancer la suite complète**

```bash
cd /home/ambiorix/task-assistant && python -m pytest tests/ -v 2>&1 | tail -20
```

Attendu : tous `PASSED`.

- [ ] **Step 7 : Commit**

```bash
cd /home/ambiorix/task-assistant && git add app/routes/fragments.py app/memory.py app/llm/prompts.py tests/test_routes.py
git commit -m "feat: reasoning correction loop — correct endpoint, few-shots injection, prompt formatting"
```

---

## Self-Review

### Couverture spec

| Spec | Task |
|---|---|
| Page config modèle LLM + test connectivité | Task 0 |
| FK fallback `category=None` | Task 2 |
| Validators urgency/confidence/payload | Task 1 |
| Latent `NEW:` prefix → None | Task 3 |
| Prompt durcissement FR enums | Task 3 |
| Migration `llm_pending` | Task 4 |
| Optimistic UI BackgroundTasks | Task 5 |
| Carte mode classifying + polling | Task 6 |
| Toggle reasoning (plus de troncature) | Task 6 |
| Bouton Corriger IA + textarea | Task 6 |
| Endpoint correction reasoning | Task 7 |
| Injection few-shots reasoning | Task 7 |

### Cohérence types

- `Setting` model importé dans `settings.py` et `main.py` — même classe `app.models.Setting` ✓
- `settings.llm_model` muté directement (Pydantic Settings est un objet mutable en mémoire) ✓
- `Task.llm_pending: Mapped[bool]` défini en Task 4, utilisé en Task 5 et 6 ✓
- `_resolve_category` définie en Task 3 dans `conversation.py`, importée en Task 5 dans `fragments.py` ✓
- `_classify_and_update` définie en Task 5 dans `conversation.py`, importée dans `fragments.py` Task 5 ✓
- `Classification.category` rendu `str | None` en Task 1 — compatible avec `_resolve_category` Task 3 ✓

### Contraste UI (note utilisateur)

Task 6 applique :
- `text-gray-500` sur métadonnées (was `text-gray-400` / `text-gray-300`)
- `text-gray-600` sur badges urgency basse (was `text-gray-500`)
- `text-red-600` / `text-orange-700` / `text-blue-700` sur labels urgence (was `*-600`)
- `text-amber-600` sur badge "à revoir" (was `text-amber-500`)
- Boutons action : `text-gray-500` (was `text-gray-300`)
