"""
Seed initial — catégories par défaut et contextes d'exemple.

Idempotent : utilise INSERT OR IGNORE sur categories (PK = name)
et vérifie l'existence par name sur contexts avant d'insérer.

Appelé une fois au démarrage depuis le lifespan FastAPI.
Les données reflètent un cabinet comptable belge généraliste.
"""

import json
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text

from app.db import get_db_session
from app.models import Category, Context

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Données de référence
# ---------------------------------------------------------------------------

_DEFAULT_CATEGORIES: list[dict] = [
    {
        "name": "client_urgent",
        "description": "Demande client avec deadline fiscale ou légale imminente",
        "color": "#dc2626",  # red-600
    },
    {
        "name": "client_standard",
        "description": "Travail client sans urgence immédiate",
        "color": "#2563eb",  # blue-600
    },
    {
        "name": "admin_cabinet",
        "description": "Gestion interne du cabinet (facturation, RH, IT)",
        "color": "#7c3aed",  # violet-600
    },
    {
        "name": "compta_interne",
        "description": "Comptabilité du cabinet lui-même",
        "color": "#059669",  # emerald-600
    },
    {
        "name": "veille_fiscale",
        "description": "Lecture, formation, circulaires, jurisprudence fiscale",
        "color": "#d97706",  # amber-600
    },
    {
        "name": "it_infra",
        "description": "Infrastructure technique (VPS, self-hosted, logiciels)",
        "color": "#0891b2",  # cyan-600
    },
    {
        "name": "asbl_maison_blanche",
        "description": "Rôle de secrétaire ASBL Maison Blanche",
        "color": "#16a34a",  # green-600
    },
    {
        "name": "perso",
        "description": "Personnel — hors cabinet",
        "color": "#9ca3af",  # gray-400
    },
]

_EXAMPLE_CONTEXTS: list[dict] = [
    {
        "name": "Cabinet REBC",
        "kind": "interne",
        "notes": "Cabinet comptable — tâches internes et administration",
        "aliases": json.dumps(["REBC", "cabinet", "interne"]),
    },
    {
        "name": "ASBL Maison Blanche",
        "kind": "asbl",
        "notes": "Association — rôle de secrétaire bénévole",
        "aliases": json.dumps(["Maison Blanche", "MB", "ASBL MB"]),
    },
    {
        "name": "Exemple SRL",
        "kind": "srl",
        "notes": "Contexte d'exemple — remplacer par vos vrais clients",
        "aliases": json.dumps(["Exemple", "SRL exemple"]),
    },
]


# ---------------------------------------------------------------------------
# Fonction principale
# ---------------------------------------------------------------------------


async def seed_initial_data() -> None:
    """
    Insère les catégories et contextes par défaut si absents.
    Idempotent — peut être appelé à chaque démarrage sans effet de bord.
    """
    async with get_db_session() as db:
        cat_count = await _seed_categories(db)
        ctx_count = await _seed_contexts(db)
        await db.commit()

    if cat_count or ctx_count:
        logger.info(
            "seed_initial_data : %d catégorie(s) + %d contexte(s) insérés",
            cat_count,
            ctx_count,
        )
    else:
        logger.debug("seed_initial_data : base déjà peuplée — aucun insert")


async def _seed_categories(db: AsyncSession) -> int:
    """Insère les catégories manquantes. Retourne le nombre d'insertions."""
    inserted = 0
    for cat_data in _DEFAULT_CATEGORIES:
        result = await db.execute(
            text("INSERT OR IGNORE INTO categories (name, description, color) VALUES (:name, :description, :color)"),
            cat_data,
        )
        inserted += result.rowcount
    return inserted


async def _seed_contexts(db: AsyncSession) -> int:
    """Insère les contextes manquants (filtre par name). Retourne le nombre d'insertions."""
    inserted = 0
    for ctx_data in _EXAMPLE_CONTEXTS:
        existing = await db.execute(
            select(Context).where(Context.name == ctx_data["name"])
        )
        if existing.scalar_one_or_none() is None:
            db.add(Context(**ctx_data))
            await db.flush()
            inserted += 1
    return inserted
