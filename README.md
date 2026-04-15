# task-assistant

Gestionnaire de tâches personnel avec classification LLM pour cabinet comptable.
Saisie conversationnelle libre → classification automatique → priorisation quotidienne.

## Fonctionnalités

- **Saisie libre** : tape "appeler AGRIWAN pour la TVA avant vendredi" → LLM catégorise, détecte l'urgence, propose une échéance
- **Sections** : NOW / CETTE SEMAINE / EN ATTENTE / VEILLE / INBOX
- **Digest matinal** à 7h00 : résumé du jour généré par LLM
- **Détection zombies** : tâches qui traînent → invite à décider
- **Undo** : annuler un "done" ou une suppression pendant 5 secondes
- **Raccourcis clavier** : `n` nouvelle tâche, `j/k` naviguer, `x` done, `g+t/w/i` sauter section
- **Archivage automatique** : tâches done > 90 jours déplacées dans `tasks_archive`

## Stack

Python 3.11+ · FastAPI · SQLAlchemy 2.0 async · aiosqlite · Alembic
HTMX · Alpine.js · Tailwind CSS · APScheduler · OpenRouter (Gemini 2.5 Flash)

## Démarrage rapide (dev)

### Prérequis

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (`pip install uv` ou `brew install uv`)
- Clé API [OpenRouter](https://openrouter.ai/)

### Installation

```bash
git clone <repo>
cd task-assistant
uv sync --extra dev

cp .env.example .env
# Éditer .env : OPENROUTER_API_KEY, APP_PASSWORD_HASH
```

### Générer le hash du mot de passe

```bash
uv run python -c "import bcrypt; print(bcrypt.hashpw(b'monmotdepasse', bcrypt.gensalt()).decode())"
```

Coller la valeur dans `.env` → `APP_PASSWORD_HASH=...`

### Initialiser la base de données

```bash
uv run alembic upgrade head
```

### Lancer

```bash
uv run uvicorn app.main:app --reload
```

Ouvrir [http://localhost:8000](http://localhost:8000).

## Configuration (`.env`)

| Variable | Défaut | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | *(requis)* | Clé API OpenRouter |
| `LLM_MODEL` | `google/gemini-2.5-flash` | Modèle LLM principal |
| `SECRET_KEY` | `changeme` | Clé de signature des cookies *(à changer en prod)* |
| `APP_PASSWORD_HASH` | *(requis)* | Hash bcrypt du mot de passe unique |
| `DAILY_CAPACITY_MINUTES` | `420` | Capacité journalière (7h) |
| `DIGEST_ENABLED` | `true` | Activer le digest matinal |
| `ZOMBIE_THRESHOLD_DAYS` | `21` | Jours d'inactivité → zombie |
| `ENVIRONMENT` | `dev` | `dev` ou `prod` |

## Tests

```bash
uv run --extra dev pytest tests/ -v
```

## Déploiement (VPS Docker)

Voir [DEPLOY.md](DEPLOY.md).
