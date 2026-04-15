"""
Application FastAPI principale.

Lifespan : création du répertoire data/ si absent.
Exception handler : NotAuthenticatedException → redirect /login.
Routers : auth (public), tasks (protégé), fragments (protégé), conversation (protégé).
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

_STATIC_DIR = Path(__file__).parent / "static"

from app.auth import NotAuthenticatedException
from app.routes import auth as auth_router
from app.routes import conversation as conversation_router
from app.routes import fragments as fragments_router
from app.routes import tasks as tasks_router
from app.scheduler import create_scheduler
from app.seed import seed_initial_data

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    Path("data").mkdir(exist_ok=True)
    await seed_initial_data()
    scheduler = create_scheduler()
    scheduler.start()
    logger.info("task-assistant démarré (scheduler actif : %d jobs)", len(scheduler.get_jobs()))
    yield
    scheduler.shutdown(wait=False)
    logger.info("task-assistant arrêté")


app = FastAPI(title="task-assistant", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(NotAuthenticatedException)
async def auth_redirect(_: Request, __: NotAuthenticatedException) -> RedirectResponse:
    return RedirectResponse(url="/login", status_code=303)


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(auth_router.router)          # /login, /logout — public
app.include_router(tasks_router.router)          # /, /week, /waiting, /inbox — protégé
app.include_router(fragments_router.router)      # /fragments/* — protégé
app.include_router(conversation_router.router)   # /conversation/* — protégé
