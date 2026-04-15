"""
Mémoire apprenante — sélection des few-shots pour le prompt de classification.

Algorithme :
1. 10 corrections les plus récentes (toutes catégories)
2. 10 corrections les plus similaires via TF-IDF sur title+description
3. Déduplication par Correction.id (PK), max 20 résultats

TF-IDF via sklearn, ré-entraîné à la volée (volume borné à 500 corrections).
Si < 5 corrections en base, on skip la similarité (pas assez de signal).
"""

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.llm.prompts import FewShotDict
from app.models import Correction

logger = logging.getLogger(__name__)

_MIN_CORRECTIONS_FOR_TFIDF = 5
_MAX_CANDIDATE_ROWS = 500


async def select_few_shots(
    title: str,
    description: str | None,
    db: AsyncSession,
) -> list[FewShotDict]:
    """
    Retourne jusqu'à 20 corrections pertinentes pour enrichir le prompt de classification.
    """
    recent = await _get_recent_corrections(db, limit=10)

    if len(recent) < _MIN_CORRECTIONS_FOR_TFIDF:
        return _to_few_shot_dicts(recent)

    recent_ids = {c.id for c in recent}
    similar = await _get_similar_corrections(
        db,
        query=f"{title} {description or ''}".strip(),
        exclude_correction_ids=recent_ids,
        limit=10,
    )

    seen: set[int] = set()
    result: list[Correction] = []
    for correction in recent + similar:
        if correction.id not in seen:
            result.append(correction)
            seen.add(correction.id)

    return _to_few_shot_dicts(result[:20])


async def record_correction(
    db: AsyncSession,
    task_id: int,
    task_title: str,
    task_description: str | None,
    field: str,
    old_value: str | None,
    new_value: str | None,
) -> Correction:
    """
    Prépare et flush une correction manuelle dans la session courante.
    Le commit est à la charge du caller (convention FastAPI : une transaction par requête).
    """
    correction = Correction(
        task_id=task_id,
        task_title=task_title,
        task_description=task_description,
        field=field,
        old_value=str(old_value) if old_value is not None else None,
        new_value=str(new_value) if new_value is not None else None,
    )
    db.add(correction)
    await db.flush()
    return correction


# ---------------------------------------------------------------------------
# Requêtes DB
# ---------------------------------------------------------------------------


async def _get_recent_corrections(db: AsyncSession, limit: int) -> list[Correction]:
    result = await db.execute(
        select(Correction)
        .order_by(Correction.corrected_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


async def _get_similar_corrections(
    db: AsyncSession,
    query: str,
    exclude_correction_ids: set[int],
    limit: int,
) -> list[Correction]:
    """
    Récupère les corrections similaires via TF-IDF sur task_title + task_description.
    Borné à _MAX_CANDIDATE_ROWS pour éviter de charger toute la table.
    """
    result = await db.execute(
        select(Correction)
        .where(Correction.task_title.is_not(None))
        .order_by(Correction.corrected_at.desc())
        .limit(_MAX_CANDIDATE_ROWS)
    )
    all_corrections = list(result.scalars().all())

    candidates = [c for c in all_corrections if c.id not in exclude_correction_ids]
    if not candidates:
        return []

    try:
        return _rank_by_tfidf(query, candidates, limit)
    except ValueError as e:
        logger.warning("TF-IDF échoué (%s) — fallback sans similarité", e)
        return candidates[:limit]


# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------


def _rank_by_tfidf(
    query: str,
    corrections: list[Correction],
    limit: int,
) -> list[Correction]:
    from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: PLC0415
    from sklearn.metrics.pairwise import cosine_similarity  # noqa: PLC0415

    if not query.strip():
        return corrections[:limit]

    corpus = [
        f"{c.task_title or ''} {c.task_description or ''}".strip()
        for c in corrections
    ]
    non_empty = [(i, doc) for i, doc in enumerate(corpus) if doc]
    if not non_empty:
        return corrections[:limit]

    indices, docs = zip(*non_empty)
    all_docs = [query] + list(docs)

    vectorizer = TfidfVectorizer(min_df=1)
    tfidf_matrix = vectorizer.fit_transform(all_docs)

    scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    ranked = sorted(zip(indices, scores), key=lambda x: x[1], reverse=True)
    top_indices = [i for i, _ in ranked[:limit]]

    return [corrections[i] for i in top_indices]


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


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
