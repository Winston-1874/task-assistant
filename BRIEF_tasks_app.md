# Brief — Gestionnaire de tâches web avec classification LLM apprenante et proactive

## Objectif général

Application web mono-utilisateur de gestion de tâches pour un expert-comptable fiscaliste (cabinet belge). Le système :

1. **Classe** automatiquement les tâches entrantes via Gemini 2.5 Flash.
2. **Apprend** des corrections manuelles (mémoire apprenante).
3. **Dialogue** proactivement : demande les échéances manquantes, signale les tâches qui traînent.
4. **Priorise** en fonction du contexte (client, dossier, urgence fiscale/légale).
5. **Accepte** la saisie conversationnelle libre (zero friction).

Déploiement local (dev) + VPS Docker (prod) accessible mobile/desktop.

---

## Stack imposée

- **Python 3.11+**
- **FastAPI** (backend + serving HTML server-rendered)
- **HTMX** + **Alpine.js** (interactions, pas de SPA)
- **Jinja2** (templates)
- **Tailwind CSS** (CDN en dev, build en prod)
- **SQLAlchemy 2.0 async** + **aiosqlite** + **Alembic**
- **Pydantic v2** (validation + settings)
- **google-genai** SDK (Gemini 2.5 Flash)
- **APScheduler** (tâches planifiées : digest, zombies, archivage)
- **uvicorn** (ASGI)
- **python-dotenv**

---

## Architecture du projet

```
tasks-app/
├── pyproject.toml
├── .env.example
├── README.md
├── DEPLOY.md
├── alembic.ini
├── migrations/
├── Dockerfile
├── docker-compose.yml
├── app/
│   ├── __init__.py
│   ├── main.py                       # FastAPI app, middlewares, routers
│   ├── config.py                     # Pydantic Settings
│   ├── db.py                         # engine + session async
│   ├── models.py                     # SQLAlchemy models
│   ├── schemas.py                    # Pydantic schemas I/O
│   ├── auth.py                       # session cookie, login
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py                 # wrapper Gemini + gestion erreurs
│   │   ├── classifier.py             # classification tâches
│   │   ├── router.py                 # routeur d'intention (saisie libre)
│   │   ├── proactive.py              # dialogues proactifs (échéance, zombies)
│   │   └── prompts.py                # templates prompts versionnés
│   ├── memory.py                     # sélection few-shots, apprentissage
│   ├── scheduler.py                  # APScheduler jobs
│   ├── routes/
│   │   ├── tasks.py
│   │   ├── categories.py
│   │   ├── contexts.py               # dossiers/clients
│   │   ├── fragments.py              # partials HTMX
│   │   ├── stats.py
│   │   ├── digest.py
│   │   └── conversation.py           # endpoint saisie libre
│   ├── templates/
│   │   ├── base.html
│   │   ├── index.html                # vue principale
│   │   ├── login.html
│   │   ├── categories.html
│   │   ├── contexts.html
│   │   ├── stats.html
│   │   ├── settings.html
│   │   ├── digest.html
│   │   └── fragments/
│   │       ├── task_card.html
│   │       ├── task_row.html
│   │       ├── correction_modal.html
│   │       ├── proactive_prompt.html # modal "pour quand est-ce ?"
│   │       ├── undo_toast.html
│   │       └── ...
│   └── static/
│       ├── css/
│       └── js/
│           ├── htmx.min.js
│           ├── alpine.min.js
│           └── keyboard.js           # raccourcis
└── tests/
    ├── test_classifier.py
    ├── test_memory.py
    ├── test_router.py
    ├── test_proactive.py
    └── test_routes.py
```

---

## Schéma de données

```sql
-- Catégories définies par l'utilisateur
CREATE TABLE categories (
  name TEXT PRIMARY KEY,
  description TEXT NOT NULL,
  color TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Contextes / dossiers clients (SRL, ASBL, particulier, projet interne)
CREATE TABLE contexts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,           -- ex : "AGRIWAN SRL", "La Maison Blanche"
  kind TEXT,                           -- 'srl', 'asbl', 'independant', 'interne', 'perso'
  notes TEXT,                          -- contexte libre pour le LLM
  aliases TEXT,                        -- JSON: ["AGRIWAN", "Agriwan SRL", "AW"]
  archived INTEGER DEFAULT 0,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tâches
CREATE TABLE tasks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  title TEXT NOT NULL,
  description TEXT,
  category TEXT REFERENCES categories(name),
  context_id INTEGER REFERENCES contexts(id),
  urgency TEXT CHECK(urgency IN ('critique','haute','normale','basse')),
  due_date DATE,
  estimated_minutes INTEGER,           -- V1.1 : estimation LLM, affinée par corrections
  actual_minutes INTEGER,              -- V1.1 : saisi optionnellement au done
  tags TEXT,                           -- JSON array
  status TEXT DEFAULT 'open' CHECK(status IN ('open','doing','waiting','done','cancelled')),
  waiting_reason TEXT,                 -- si status=waiting
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  completed_at TIMESTAMP,
  touched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- dernière modif
  llm_raw_response TEXT,
  llm_confidence REAL,
  llm_reasoning TEXT,                  -- extrait pour affichage inline
  was_corrected INTEGER DEFAULT 0,
  postponed_count INTEGER DEFAULT 0,   -- nb de fois où due_date a été repoussée
  needs_review INTEGER DEFAULT 0       -- flag inbox/confidence faible
);

CREATE INDEX idx_tasks_status ON tasks(status);
CREATE INDEX idx_tasks_due ON tasks(due_date);
CREATE INDEX idx_tasks_context ON tasks(context_id);
CREATE INDEX idx_tasks_touched ON tasks(touched_at);

-- Recherche plein texte (détection doublons + filtre search)
CREATE VIRTUAL TABLE tasks_fts USING fts5(
  title, description, content=tasks, content_rowid=id
);

-- Archives (tâches done > 90j)
CREATE TABLE tasks_archive AS SELECT * FROM tasks WHERE 0;

-- Corrections (cœur du système apprenant)
CREATE TABLE corrections (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id INTEGER REFERENCES tasks(id),
  field TEXT NOT NULL,                 -- 'category', 'urgency', 'due_date', 'context_id'
  old_value TEXT,
  new_value TEXT,
  task_title TEXT,                     -- dupliqué pour few-shot sans JOIN
  task_description TEXT,
  corrected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_corrections_recent ON corrections(corrected_at DESC);

-- Notes métier par catégorie (règles explicites injectées dans prompt)
CREATE TABLE category_notes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  category TEXT REFERENCES categories(name),
  note TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Dialogues proactifs en attente de réponse user
CREATE TABLE pending_prompts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id INTEGER REFERENCES tasks(id),
  kind TEXT NOT NULL,                  -- 'ask_due_date', 'zombie_check', 'estimate_confirm'
  message TEXT NOT NULL,               -- texte généré par LLM
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  resolved_at TIMESTAMP,
  resolution TEXT                      -- JSON : réponse user
);

-- Digest matinaux générés
CREATE TABLE digests (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  date DATE UNIQUE NOT NULL,
  content_html TEXT NOT NULL,
  content_text TEXT NOT NULL,          -- version plain pour email éventuel
  task_ids TEXT                        -- JSON array des tâches citées
);

-- Objectifs (V1.1 scaffoldé)
CREATE TABLE goals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  label TEXT NOT NULL,
  description TEXT,
  starts_at DATE,
  ends_at DATE,
  active INTEGER DEFAULT 1
);

-- Estimations de durée — calibration (V1.1 scaffoldé)
CREATE TABLE duration_calibration (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  category TEXT,
  context_id INTEGER REFERENCES contexts(id),
  keyword TEXT,                        -- pattern détecté
  llm_estimate INTEGER,
  actual_minutes INTEGER,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Settings clé-valeur
CREATE TABLE settings (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
-- contient : auth_password_hash, daily_capacity_minutes, digest_enabled,
--            zombie_threshold_days, coaching_tone (concise|encouraging|off)
```

---

## Authentification

- **Mono-utilisateur V1**, password bcrypt stocké dans `settings`
- Session cookie HttpOnly / SameSite=Lax / Secure (prod)
- Middleware FastAPI → redirect `/login`
- Pas de JWT, pas d'OAuth

---

## Système LLM — trois étages

### Étage 1 : routeur d'intention (saisie conversationnelle)

Endpoint `POST /conversation/parse` appelé depuis l'input unique en bas de page.

Prompt système : classifier le message utilisateur en une intention parmi :
- `new_task` : création tâche → passer à l'étage 2
- `command` : action sur tâche existante (« marque Dupont comme fait », « reporte la TVA AGRIWAN à lundi »)
- `query` : question sur le backlog (« qu'est-ce qui traîne ? », « combien d'urgences cette semaine ? »)
- `update_context` : info à retenir (« AGRIWAN change de régime TVA en janvier »)

Sortie Pydantic :

```python
class Intent(BaseModel):
    kind: Literal['new_task','command','query','update_context']
    confidence: float
    payload: dict   # structure dépendante du kind
```

Si confidence < 0.6 → fallback : créer une tâche en inbox avec le texte brut.

### Étage 2 : classification de tâche

Prompt construit avec 5 couches :

1. **Catégories + descriptions** (table `categories`)
2. **Notes métier** par catégorie (table `category_notes`)
3. **Contextes connus** : liste `contexts` avec aliases → aide au rattachement
4. **Few-shots corrections** : 20 max (10 récentes + 10 similaires TF-IDF sur title/description)
5. **Tâche à classer** + contexte conversationnel éventuel

Sortie Gemini forcée via `response_schema` :

```python
class Classification(BaseModel):
    category: str                       # ∈ catégories ou "NEW:<nom>"
    context_id: Optional[int]           # si reconnu dans aliases
    context_suggestion: Optional[str]   # si pattern mais pas de match
    urgency: Literal['critique','haute','normale','basse']
    due_date: Optional[date]
    needs_due_date: bool                # True si tâche semble avoir une échéance implicite non exprimée
    estimated_minutes: Optional[int]    # V1.1 utilisable
    tags: list[str]                     # max 5
    confidence: float
    reasoning: str                      # 1-2 phrases, affiché inline
```

### Étage 3 : dialogues proactifs

Déclenchés par APScheduler ou à la création.

**a. Demande d'échéance à la création**
Si `needs_due_date=True` ET `due_date=None` → créer une entrée `pending_prompts` avec kind=`ask_due_date` et message LLM type :
> « "Bilan NewCo" — ça sent la deadline. Pour quand tu le vois bouclé ? »

Affiché en bandeau sur la card de la tâche, avec boutons rapides : `Aujourd'hui | Demain | Cette semaine | +7j | +14j | Fin du mois | Custom | Pas de deadline`.

**b. Détection zombies**
Tâche `open` ET (`postponed_count >= 3` OU `touched_at < now - 21 jours`) → prompt LLM :
> « Cette tâche traîne depuis 3 semaines et a été reportée 2 fois. Tu la fais cette semaine, tu la délègues, ou on la tue ? »

Actions rapides : `Planifier cette semaine | Déléguer (à qui ?) | Annuler | Laisser tranquille 7j`.

**c. Confirmation d'estimation** (V1.1)
Si la somme des `estimated_minutes` des tâches `due_date=today` dépasse `daily_capacity_minutes` :
> « Ta journée est pleine à 120% (8h20 vs 7h cible). Tu veux reprioriser ? »

### Gestion erreurs LLM

- Timeout 15s
- Retry 1× avec backoff
- Rate limit → queue in-memory, réessai
- Réponse non-parsable → tâche créée avec `category='inbox'`, `urgency='normale'`, `needs_review=1`, badge visuel ⚠️
- Log tokens dans `llm_raw_response` pour stats coût

---

## Mémoire apprenante — détail

`app/memory.py` :

```python
async def select_few_shots(task_title: str, task_description: str, db) -> list[Correction]:
    # 1. Les 10 corrections les plus récentes (toutes catégories)
    recent = await get_recent_corrections(db, limit=10)

    # 2. Les 10 corrections les plus similaires (TF-IDF sur title+description)
    similar = await get_similar_corrections(
        db, query=f"{task_title} {task_description}", limit=10
    )

    # 3. Déduplication par task_id
    seen = set()
    result = []
    for c in recent + similar:
        if c.task_id not in seen:
            result.append(c)
            seen.add(c.task_id)
    return result[:20]
```

TF-IDF via `sklearn.feature_extraction.text.TfidfVectorizer` ré-entraîné à chaque appel (volume faible, OK). Si volume dépasse 500 corrections, passer à un index persistant.

---

## Interface web

### Structure générale

```
┌─────────────────────────────────────────────────────────────────┐
│ Tasks     [Today] [Week] [Waiting] [Veille] [Inbox]  [Q] [⚙]  │
├─────────────────┬───────────────────────────────────────────────┤
│                 │                                               │
│ SIGNAL          │  ● NOW                          3 active      │
│ (colonne        │  ┌────────────────────┬────────────────────┐ │
│ priorisation    │  │ AGRIWAN — Valider… │ Dupont — Relance… │ │
│ LLM, opt-in)    │  │ 🔴 due today       │ 🟠 due demain     │ │
│                 │  │ [✓][✎][↻]          │ [✓][✎][↻]         │ │
│ - Top tâche 1   │  └────────────────────┴────────────────────┘ │
│ - Top tâche 2   │                                               │
│ - Suggestion    │  ● THIS WEEK                    7 active      │
│                 │  ...                                          │
│                 │                                               │
│                 │  ● WAITING                      2 active      │
│                 │  ...                                          │
│                 │                                               │
│                 │  ● VEILLE / LOW                 5 active      │
│                 │  ...                                          │
├─────────────────┴───────────────────────────────────────────────┤
│ 💬 Dis-moi ce que tu as en tête...              [🎤] [→ send]  │
└─────────────────────────────────────────────────────────────────┘
```

### Sections dynamiques (non configurables V1)

- **NOW** : `due_date <= today` OR `urgency='critique'`
- **THIS WEEK** : `due_date <= today+7j`
- **WAITING** : `status='waiting'`
- **VEILLE / LOW** : `urgency='basse'` AND pas de due_date
- **INBOX** : `needs_review=1` OR `confidence < 0.7`

Filtres transversaux : catégorie, contexte, recherche FTS5.

### Card de tâche

```
┌──────────────────────────────────────────────────┐
│ [□] AGRIWAN — Valider immobilisés amortissements │
│     ↳ AGRIWAN SRL · admin_cabinet · 🔴 today     │
│     💡 Raisonnement (expand)                     │
│ [✓ done] [✎ edit] [↻ reclassify] [🧠 corriger]   │
└──────────────────────────────────────────────────┘
```

- Checkbox → `done`
- Clic titre → panneau latéral détail/édition
- "Corriger" → modal avec catégorie/urgence/contexte → crée `correction`
- Si `pending_prompt` actif → bandeau haut de card avec question + boutons rapides

### Input conversationnel (bas, fixe)

- Textarea auto-resize, Enter = send, Shift+Enter = nouvelle ligne
- Bouton micro (Web Speech API, V1.1)
- Réponse LLM affichée en toast court + action visible (nouvelle tâche insérée, tâche mise à jour, réponse à query)

### Colonne SIGNAL (gauche)

- Liste des 3-5 tâches prioritaires selon le LLM, avec justification courte
- Tone réglable : `concise | encouraging | off` (setting)
- Régénérée 1×/jour ou sur demande (bouton refresh)

### Undo

Toute action mutante (done, delete, cancel, reclassify) → toast en bas-droit avec **Annuler (5s)**. HTMX `hx-swap="none"` + push vers queue undo client-side (Alpine store).

### Raccourcis clavier

```
n         focus input nouvelle tâche
/         recherche
j/k       navigation up/down dans les cards
x         toggle done sur card sélectionnée
c         corriger classification
r         reclassifier
e         éditer
g t       aller à Today
g w       aller à Week
g i       aller à Inbox
?         overlay aide
```

Implémenté via `static/js/keyboard.js`, overlay d'aide déclenché par `?`.

---

## Endpoints FastAPI

```
# Auth
GET  /login
POST /login
POST /logout

# Pages
GET  /                                # Today par défaut
GET  /week
GET  /waiting
GET  /inbox
GET  /categories
GET  /contexts
GET  /stats
GET  /settings
GET  /digest/today
GET  /digest/{date}

# Conversation
POST /conversation/parse              # routeur intention

# Fragments tâches
GET    /fragments/tasks               # listing filtré (query params)
POST   /fragments/tasks               # création + classif
PUT    /fragments/tasks/{id}
DELETE /fragments/tasks/{id}
POST   /fragments/tasks/{id}/done
POST   /fragments/tasks/{id}/undo/{action_id}
POST   /fragments/tasks/{id}/correct
POST   /fragments/tasks/{id}/reclassify
POST   /fragments/tasks/{id}/prompt/{prompt_id}/answer
POST   /fragments/tasks/reclassify-stale

# Fragments catégories / contextes
POST   /fragments/categories
PUT    /fragments/categories/{name}
DELETE /fragments/categories/{name}
POST   /fragments/categories/{name}/notes
POST   /fragments/contexts
PUT    /fragments/contexts/{id}
DELETE /fragments/contexts/{id}

# Signal
GET  /fragments/signal                # colonne priorisation

# Digest / Stats
POST /fragments/digest/generate
GET  /fragments/stats/summary
```

---

## Comportements d'autonomie

1. **Inférence due_date** : extraction par LLM depuis description
2. **Dialogue échéance** : si `needs_due_date=True` → `pending_prompt` type `ask_due_date`
3. **Détection doublons** : FTS5 sur 30j, warning HTMX avant insertion
4. **Reclassification batch** : `tasks reclassify-stale` sur tâches avec classif > 30j OU catégorie ayant reçu corrections récentes (drift)
5. **Détection zombies** : APScheduler quotidien → création prompts zombie
6. **Auto-archivage** : APScheduler hebdo → `done > 90j` vers `tasks_archive`
7. **Digest matinal** : APScheduler 7h00 → génère HTML + texte, affiché sur page d'accueil matin (optionnel email V1.1)
8. **Fallback LLM** : tout échec → inbox + flag

---

## Déploiement

### Dev local

```bash
cp .env.example .env    # GEMINI_API_KEY, APP_PASSWORD, DAILY_CAPACITY_MINUTES=420
uv sync
alembic upgrade head
uvicorn app.main:app --reload
```

### Prod VPS (Docker + Traefik)

```yaml
# docker-compose.yml
services:
  tasks-app:
    build: .
    restart: unless-stopped
    volumes:
      - ./data:/data
    environment:
      - DATABASE_URL=sqlite+aiosqlite:///data/tasks.db
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - APP_PASSWORD_HASH=${APP_PASSWORD_HASH}
      - DAILY_CAPACITY_MINUTES=420
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.tasks.rule=Host(`tasks.<domain>`)"
      - "traefik.http.routers.tasks.entrypoints=websecure"
      - "traefik.http.routers.tasks.tls.certresolver=letsencrypt"
      - "traefik.http.services.tasks.loadbalancer.server.port=8000"
```

`DEPLOY.md` couvre :
- Build et run
- Backup SQLite (`sqlite3 tasks.db ".backup /backup/tasks-$(date +%F).db"` en cron)
- Restauration
- Rotation logs
- Upgrade procedure

---

## Points techniques

- **Async partout** : SQLAlchemy async, httpx pour Gemini, aiosqlite
- **HTMX discipline** : pas de `hx-boost` global, ciblage précis de `hx-target`, `hx-swap` explicite
- **CSRF** : token dans forms, middleware FastAPI
- **Logs structurés** JSON → stdout (Docker)
- **Tests** : pytest + `httpx.AsyncClient`, mocks Gemini via `respx` ou fixture
- **Versioning prompts** : tous les prompts dans `app/llm/prompts.py` avec constante `PROMPT_VERSION`, stockée dans `llm_raw_response` pour audit
- **Coût** : tokens loggés par appel, dashboard `/stats` affiche cumul mensuel

---

## Livrables

1. Repo installable (`uv sync`)
2. `README.md` (install dev, architecture, modèle mental)
3. `DEPLOY.md` (VPS Docker Traefik)
4. Migrations Alembic fonctionnelles
5. `EXAMPLES.md` : 5 scénarios documentés
   - Création simple via input conversationnel
   - Correction d'une classif → apprentissage vérifié sur tâche similaire
   - Tâche sans échéance → dialogue proactif → résolution
   - Tâche zombie → prompt → action
   - Query sur le backlog (« que dois-je faire aujourd'hui ? »)
6. Tests unitaires : `classifier`, `memory`, `router`, `proactive`
7. Tests d'intégration sur routes principales
8. `.env.example` commenté
9. Script seed initial : catégories par défaut + contextes d'exemple

### Catégories par défaut proposées

- `client_urgent` — demande client avec deadline fiscale/légale imminente
- `client_standard` — travail client sans urgence immédiate
- `admin_cabinet` — gestion interne (facturation, RH, IT cabinet)
- `compta_interne` — compta du cabinet lui-même
- `veille_fiscale` — lecture, formation, circulaires, jurisprudence
- `it_infra` — infrastructure technique (VPS, self-hosted)
- `asbl_maison_blanche` — rôle de secrétaire ASBL
- `perso` — hors cabinet

(Modifiables au premier lancement.)

---

## V1.1 — scaffoldé mais pas UI

- **Objectifs / north star** : table `goals` créée, injectable dans prompt prioritisation
- **Estimation durée + capacité** : colonnes DB en place, calibration via `duration_calibration` quand user saisit `actual_minutes` au done. UI d'affichage à ajouter V1.1.
- **Commandes vocales** : bouton micro Web Speech API dans input (fallback gracieux)
- **Digest par email** : générer le digest est fait, envoi SMTP à ajouter

---

## Hors scope total

- Connecteur Todoist / n8n (futur, via API REST de l'app)
- Multi-utilisateur
- Mobile app native (web responsive suffit)
- PWA (possible itération rapide après V1)
- Embeddings pour sélection few-shots (TF-IDF suffit < 500 corrections)

---

## Principes directeurs pour l'agent Claude Code

1. **Commencer par le schéma DB** + migrations Alembic, vérifier que tout tourne
2. **Puis le wrapper Gemini** avec mocks pour tests sans appel réseau
3. **Puis la classification pure** (sans UI) testée en unitaire
4. **Puis le routeur d'intention**
5. **Puis les dialogues proactifs**
6. **Puis les routes FastAPI** + templates minimalistes
7. **Puis HTMX interactions** + undo
8. **Puis le styling Tailwind** (inspiré screenshot Lumin mais sobre)
9. **Puis APScheduler** (zombies, digest, archivage)
10. **Enfin doc + déploiement Docker**

À chaque étape : tests verts avant de passer à la suivante. Commits atomiques. Pas de grosse PR.
