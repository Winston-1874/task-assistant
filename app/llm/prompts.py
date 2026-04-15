"""
Prompt templates versionnés pour les trois étages LLM.
PROMPT_VERSION est stocké dans llm_raw_response pour audit.
"""

PROMPT_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Étage 1 — Routeur d'intention
# ---------------------------------------------------------------------------

INTENT_ROUTER_SYSTEM = """\
Tu es un assistant de gestion de tâches pour un expert-comptable fiscaliste belge.
Analyse le message utilisateur et classe-le dans une des intentions suivantes :

- new_task : création d'une nouvelle tâche (ex : "faire la TVA AGRIWAN", "rappeler Dupont")
- command  : action sur une tâche existante (ex : "marque la TVA comme faite", "reporte à lundi")
- query    : question sur le backlog (ex : "qu'est-ce qui traîne ?", "combien d'urgences ?")
- update_context : information à retenir sur un client/dossier (ex : "AGRIWAN change de régime TVA")

Réponds UNIQUEMENT avec un objet JSON valide respectant ce schéma :
{
  "kind": "new_task" | "command" | "query" | "update_context",
  "confidence": <float 0.0-1.0>,
  "payload": <objet libre adapté au kind>
}

Si tu n'es pas sûr (confidence < 0.6), utilise kind="new_task" avec le texte brut dans payload.
"""


def intent_router_user(message: str) -> str:
    return f"Message utilisateur :\n{message}"


# ---------------------------------------------------------------------------
# Étage 2 — Classification de tâche
# ---------------------------------------------------------------------------

TASK_CLASSIFIER_SYSTEM = """\
Tu es un assistant de classification de tâches pour un expert-comptable fiscaliste belge.
Tu reçois une tâche à classer et tu dois retourner un objet JSON structuré.

Règles :
- Choisis la catégorie la plus précise parmi celles fournies.
- Si aucune ne convient, propose "NEW:<nom_suggéré>".
- urgency doit refléter l'urgence fiscale/légale réelle, pas la difficulté.
- due_date : extrais les dates explicites ou implicites ("fin du mois", "avant la déclaration TVA").
- needs_due_date : true si la tâche a clairement une échéance mais qu'elle n'est pas exprimée.
- reasoning : 1-2 phrases maximum, factuel, en français.
- tags : mots-clés utiles, 5 maximum.

Réponds UNIQUEMENT avec un objet JSON valide. Aucun texte avant ou après.
"""


def task_classifier_user(
    title: str,
    description: str | None,
    categories: list[dict],
    contexts: list[dict],
    few_shots: list[dict],
) -> str:
    parts = []

    parts.append("=== CATÉGORIES DISPONIBLES ===")
    for cat in categories:
        notes = f" — Notes : {cat['notes']}" if cat.get("notes") else ""
        parts.append(f"- {cat['name']} : {cat['description']}{notes}")

    if contexts:
        parts.append("\n=== CONTEXTES / CLIENTS CONNUS ===")
        for ctx in contexts:
            aliases = f" (alias : {ctx['aliases']})" if ctx.get("aliases") else ""
            parts.append(f"- [{ctx['id']}] {ctx['name']}{aliases} — {ctx.get('kind', '')}")

    if few_shots:
        parts.append("\n=== EXEMPLES DE CORRECTIONS PASSÉES ===")
        for fs in few_shots:
            parts.append(
                f'- "{fs["task_title"]}" : {fs["field"]} {fs["old_value"]} → {fs["new_value"]}'
            )

    parts.append(f"\n=== TÂCHE À CLASSER ===")
    parts.append(f"Titre : {title}")
    if description:
        parts.append(f"Description : {description}")

    parts.append(
        "\nRetourne un JSON avec : category, context_id, context_suggestion, urgency, "
        "due_date (ISO ou null), needs_due_date, estimated_minutes, tags, confidence, reasoning."
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Étage 3 — Dialogues proactifs
# ---------------------------------------------------------------------------

PROACTIVE_SYSTEM = """\
Tu es un assistant de gestion de tâches pour un expert-comptable fiscaliste belge.
Tu génères des messages courts, directs et professionnels en français.
Pas de formules de politesse inutiles. Va droit au but.
"""


def proactive_ask_due_date_user(task_title: str) -> str:
    return (
        f'La tâche "{task_title}" semble avoir une échéance implicite.\n'
        f"Génère un message court (1-2 phrases max) pour demander à l'utilisateur pour quand "
        f"il prévoit de la terminer. Commence directement par la question, sans introduction."
    )


def proactive_zombie_user(task_title: str, days_idle: int, postponed_count: int) -> str:
    return (
        f'La tâche "{task_title}" n\'a pas été touchée depuis {days_idle} jours '
        f"et a été reportée {postponed_count} fois.\n"
        f"Génère un message court et direct pour demander si l'utilisateur va la faire cette "
        f"semaine, la déléguer, ou l'annuler. 2-3 phrases maximum."
    )


def proactive_signal_user(tasks: list[dict], capacity_minutes: int) -> str:
    task_list = "\n".join(
        f"- [{t['urgency']}] {t['title']} (due: {t.get('due_date', 'pas de date')})"
        for t in tasks
    )
    return (
        f"Voici les tâches ouvertes du backlog :\n{task_list}\n\n"
        f"Capacité journalière : {capacity_minutes} minutes.\n"
        f"Identifie les 3 à 5 tâches prioritaires aujourd'hui avec une justification courte "
        f"(1 phrase par tâche). Réponds avec un JSON : "
        f'[{{"task_title": ..., "reason": ...}}]'
    )
