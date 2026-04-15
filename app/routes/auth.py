"""Routes d'authentification — GET/POST /login, POST /logout."""

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse

from app.auth import delete_session_cookie, is_authenticated, set_session_cookie, verify_password
from app.config import settings
from app.templates_config import templates

router = APIRouter()


@router.get("/login")
async def login_page(request: Request):
    if is_authenticated(request):
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse(request, "login.html", {"error": None})


@router.post("/login")
async def login_submit(
    request: Request,
    password: str = Form(...),
):
    if verify_password(password, settings.app_password_hash):
        response = RedirectResponse(url="/", status_code=303)
        set_session_cookie(response)
        return response
    return templates.TemplateResponse(
        request,
        "login.html",
        {"error": "Mot de passe incorrect"},
        status_code=401,
    )


@router.post("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=303)
    delete_session_cookie(response)
    return response
