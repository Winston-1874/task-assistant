"""
Authentification mono-utilisateur.

Session cookie HttpOnly signé par itsdangerous (URLSafeTimedSerializer).
Stateless : le cookie signé suffit, pas de table de sessions en DB.
Le hash bcrypt du mot de passe est dans settings.app_password_hash.

Pattern d'utilisation dans les routes :
    router = APIRouter(dependencies=[Depends(require_auth)])
Exception NotAuthenticatedException → handler dans main.py → redirect /login.
"""

import bcrypt
from fastapi import Depends, Request
from fastapi.responses import RedirectResponse
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from app.config import settings

_COOKIE_NAME = "session"
_COOKIE_MAX_AGE = 60 * 60 * 24 * 30  # 30 jours
_SESSION_PAYLOAD = "authenticated"


class NotAuthenticatedException(Exception):
    """Levée par require_auth() — interceptée par le handler dans main.py."""


def _serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(settings.secret_key, salt="session")


def verify_password(plain: str, hashed: str) -> bool:
    """Vérifie un mot de passe en clair contre un hash bcrypt.
    Retourne False si le hash est vide ou invalide (plutôt que lever ValueError).
    """
    if not hashed or not hashed.startswith("$2"):
        return False
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def create_session_cookie() -> str:
    """Retourne un token signé à stocker dans le cookie de session."""
    return _serializer().dumps(_SESSION_PAYLOAD)


def is_valid_session(token: str) -> bool:
    """Vérifie signature et expiration du token de session."""
    try:
        value = _serializer().loads(token, max_age=_COOKIE_MAX_AGE)
        return value == _SESSION_PAYLOAD
    except (BadSignature, SignatureExpired):
        return False


def is_authenticated(request: Request) -> bool:
    """True si la requête porte une session valide."""
    token = request.cookies.get(_COOKIE_NAME)
    return token is not None and is_valid_session(token)


async def require_auth(request: Request) -> None:
    """
    Dépendance FastAPI pour les routes protégées.
    Lève NotAuthenticatedException si non authentifié.
    """
    if not is_authenticated(request):
        raise NotAuthenticatedException()


def set_session_cookie(response: RedirectResponse) -> None:
    """Pose le cookie de session signé sur une réponse."""
    response.set_cookie(
        _COOKIE_NAME,
        create_session_cookie(),
        max_age=_COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=settings.environment == "prod",
    )


def delete_session_cookie(response: RedirectResponse) -> None:
    """Supprime le cookie de session."""
    response.delete_cookie(
        _COOKIE_NAME,
        httponly=True,
        samesite="lax",
        secure=settings.environment == "prod",
    )
