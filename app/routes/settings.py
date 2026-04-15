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
        return _result_html(ok=True, message=f"OK — Modèle « {html.escape(model)} » répond correctement.")
    except LLMError as e:
        logger.warning("Test modèle %r échoué : %s", model, e)
        return _result_html(ok=False, message=f"Échec : {html.escape(str(e)[:200])}")


@router.post("/fragments/settings/model/save", response_class=HTMLResponse)
async def save_model(model: str = Form(...), db: AsyncSession = Depends(get_db)):
    """Persiste le modèle dans la table settings et met à jour le singleton en mémoire."""
    model = model.strip()
    if not model:
        return _result_html(ok=False, message="Le nom du modèle est vide.")

    from sqlalchemy import select
    result = await db.execute(select(Setting).where(Setting.key == "llm_model"))
    row = result.scalar_one_or_none()
    if row is None:
        db.add(Setting(key="llm_model", value=model))
    else:
        row.value = model
    await db.commit()

    settings.llm_model = model
    logger.info("Modèle LLM changé → %s", model)

    return _result_html(ok=True, message=f"Modèle enregistré : « {html.escape(model)} »")


def _result_html(ok: bool, message: str) -> HTMLResponse:
    color = "text-green-700 bg-green-50 border-green-200" if ok else "text-red-700 bg-red-50 border-red-200"
    icon = "✓" if ok else "✕"
    return HTMLResponse(
        f'<p class="text-sm px-3 py-2 rounded border {color}">{icon} {message}</p>'
    )
