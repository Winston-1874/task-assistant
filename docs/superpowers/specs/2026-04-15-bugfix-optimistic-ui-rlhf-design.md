# Design : Bugfixes LLM + Optimistic UI + Feedback loop raisonnement

**Date :** 2026-04-15  
**Branche cible :** main  
**Statut :** Approuvé

---

## Contexte

Trois groupes de travaux sur l'application task-assistant (FastAPI + HTMX + SQLite + OpenRouter/Gemini) :

1. Correctifs de bugs LLM causant des violations FK et des erreurs de parsing
2. UI optimiste pour la création de tâche (carte immédiate + spinner pendant classification LLM)
3. Feedback loop raisonnement (correction du raisonnement LLM, injection comme few-shots)

---

## 1. Correctifs bugs

### 1a. Fallback FK — `app/llm/classifier.py:67`

**Problème :** Fallback `category="inbox"` provoque une FK violation car `"inbox"` n'existe pas toujours dans la table `categories`.

**Fix :** Remplacer `category="inbox"` par `category=None` dans le fallback de `classify_task`.

### 1b. Validators LLM — `app/schemas.py`

**Problème :** Gemini 2.5 Flash via OpenRouter renvoie :
- `urgency`: `"high"` / `"low"` / `"urgent"` / entier au lieu de `"critique"/"haute"/"normale"/"basse"`
- `confidence`: `"high"` / `"medium"` / `"low"` au lieu d'un float 0–1
- `Intent.payload`: scalaire brut au lieu d'un dict

**Fix :** Ajouter `@field_validator(mode="before")` sur `Classification` et `Intent` :

```python
URGENCY_ALIASES = {
    "urgent": "critique", "critical": "critique", "high": "haute",
    "medium": "normale", "normal": "normale", "low": "basse",
}

CONFIDENCE_ALIASES = {"high": 0.9, "medium": 0.6, "low": 0.3}
```

- `urgency` : normalise via `URGENCY_ALIASES`, passe les valeurs FR valides telles quelles
- `confidence` : `str` dans `CONFIDENCE_ALIASES` → float ; string numérique → `float(v)` ; déjà float → pass
- `Intent.payload` : si scalaire (non-dict) → `{"raw": v}`

### 1c. Préfixe `NEW:` — `app/routes/conversation.py` + `app/routes/fragments.py`

**Problème :** Le LLM peut renvoyer `category="NEW:<nom>"` → FK violation.

**Fix :** Helper partagée `_resolve_category(classification) -> str | None` :
- Si `classification.category` commence par `"NEW:"` → retourne `None` et force `needs_review=1`
- Sinon → retourne `classification.category`

Appelée dans `_handle_new_task` (conversation.py) et `create_task` (fragments.py).

### 1d. Durcissement prompt — `app/llm/prompts.py`

Ajouter dans `TASK_CLASSIFIER_SYSTEM` les contraintes explicites :

```
- urgency DOIT être exactement l'un de : "critique" | "haute" | "normale" | "basse"
- confidence DOIT être un float entre 0.0 et 1.0 (ex. 0.85), jamais "high"/"low"/"medium"
```

---

## 2. Optimistic UI

### 2a. Migration DB

Nouveau champ sur `Task` :

```python
llm_pending: Mapped[bool] = mapped_column(Boolean, default=False)
```

Migration Alembic : `add_llm_pending_to_tasks`.

### 2b. Flow de création

`POST /conversation/parse` et `POST /fragments/tasks` :

1. Créer la tâche immédiatement avec `llm_pending=True`, `category=None`, `urgency=None`
2. Enregistrer `classify_and_update` dans `BackgroundTasks`
3. Retourner `task_card.html` en mode "classifying"

Signature de la fonction de fond :
```python
async def classify_and_update(task_id: int, title: str, description: str | None) -> None
```
Ouvre sa propre session DB (la session HTTP est déjà fermée au moment de l'exécution).

### 2c. Fragment carte — mode classifying

Si `task.llm_pending=True`, la carte affiche :
- Titre de la tâche
- Spinner latéral (`.htmx-indicator` animé)
- Polling HTMX `hx-get="/fragments/tasks/{id}/status"` `hx-trigger="every 2s"` `hx-swap="outerHTML"`
- Pas de méta (urgence, catégorie) ni d'actions (done, corriger, supprimer)

### 2d. Endpoint poll

```
GET /fragments/tasks/{id}/status
```

- `llm_pending=True` → retourne la même carte polling (arrête pas le polling)
- `llm_pending=False` → retourne `task_card.html` normal (sans `hx-trigger`) → polling s'arrête naturellement

### 2e. Background task — gestion d'erreur

- `LLMTransportError` ou `LLMParseError` → `llm_pending=False`, `needs_review=1`, `category=None`
- La carte redevient visible même en cas d'erreur LLM

---

## 3. Affichage raisonnement LLM

### 3a. Suppression troncature

Dans `task_card.html` ligne 138 :
- Retirer `line-clamp-2` et `truncate(120, ...)`
- Remplacer par toggle Alpine `x-data="{ expanded: false }"` :
  - Texte complet affiché, clamp à 2 lignes par défaut
  - Lien "voir plus / voir moins" si texte dépasse

### 3b. Token max (info)

`llm_max_tokens=1024` dans `app/config.py` est suffisant pour le raisonnement (1-2 phrases demandées dans le prompt). Pas de changement nécessaire.

---

## 4. Feedback loop raisonnement (mini-RLHF)

### 4a. Modèle DB

La table `corrections` existe déjà (`app/models.py:92`). Le champ `field` accepte actuellement `category|urgency|due_date|context_id`. Étendre pour accepter `"reasoning"` aussi — pas de migration DB nécessaire (colonne `Text` libre).

Mettre à jour la constante dans `fragments.py` :
```python
_ALLOWED_CORRECTION_FIELDS = {"urgency", "category", "due_date", "context_id", "reasoning"}
```

### 4b. Endpoint

Réutiliser l'endpoint existant `POST /fragments/tasks/{id}/correct` avec `field="reasoning"`. Côté logique : enregistre `old_value=task.llm_reasoning`, `new_value=<texte utilisateur>`, met à jour `task.llm_reasoning` et `task.was_corrected=1`.

### 4c. UI — bouton + champ

Dans `task_card.html`, sous le raisonnement LLM :
- Bouton "Corriger IA" (inline, discret) → toggle Alpine affiche un `<textarea>` + bouton submit
- `hx-post="/fragments/tasks/{id}/correct"` avec `field=reasoning` et `new_value=<textarea>`
- Masqué si `task.llm_pending=True`

### 4d. Injection dans le prompt

Dans `app/memory.py`, `select_few_shots` sélectionne déjà les corrections récentes. Étendre la requête pour inclure `field="reasoning"` dans les few-shots, avec formatage adapté :

```
- "Titre tâche" : raisonnement corrigé de "<old>" à "<new>"
```

Dans `prompts.py`, adapter le formatage few-shots pour afficher les corrections de type `reasoning`.

---

## 5. Configuration modèle LLM

### 5a. Persistance

La table `settings` (clé/valeur) existe déjà. Le modèle courant est stocké sous la clé `"llm_model"`. Au démarrage (lifespan), si la clé existe en DB → override `settings.llm_model` (le singleton Pydantic). Le client LLM lit `settings.llm_model` à chaque appel (pas dans le constructeur) : pas besoin de recréer le client.

### 5b. Endpoint

`GET /settings` — page de configuration (lien dans la nav).

`POST /fragments/settings/model/test` — test de connectivité avec le modèle fourni :
- Appel LLM minimal : `{"role": "user", "content": "Réponds juste {\"ok\": true}"}` avec `max_tokens=20`
- Succès → fragment HTML vert "Modèle OK"
- Échec → fragment HTML rouge avec message d'erreur

`POST /fragments/settings/model/save` — enregistre le modèle :
- Upsert dans `Setting(key="llm_model")` + override `settings.llm_model` en mémoire
- Retourne confirmation inline

### 5c. UI

Page `/settings` minimale :
- Titre "Configuration"
- Label + input text avec valeur courante (`settings.llm_model`), placeholder `google/gemini-2.5-flash`
- Bouton "Tester" → `hx-post="/fragments/settings/model/test"` → résultat inline (vert/rouge)
- Bouton "Enregistrer" → `hx-post="/fragments/settings/model/save"` → confirmation inline
- Lien "← Retour" vers `/`

Lien discret "⚙" dans la nav (à droite, avant "Déconnexion").

---

## Fichiers modifiés

| Fichier | Changement |
|---|---|
| `app/schemas.py` | Validators urgency, confidence, Intent.payload |
| `app/llm/classifier.py` | Fallback category=None |
| `app/llm/prompts.py` | Contraintes enum FR dans TASK_CLASSIFIER_SYSTEM |
| `app/models.py` | Champ `llm_pending: bool` |
| `app/routes/conversation.py` | Optimistic UI + BackgroundTasks + helper `_resolve_category` |
| `app/routes/fragments.py` | Optimistic UI + `_resolve_category` + field reasoning + poll endpoint |
| `app/templates/fragments/task_card.html` | Mode classifying + toggle reasoning + bouton corriger IA |
| `migrations/versions/xxxx_add_llm_pending.py` | Nouvelle migration |
| `app/memory.py` | Inclure corrections reasoning dans few-shots |
| `app/routes/settings.py` | Nouveau — page config + endpoints test/save modèle |
| `app/templates/settings.html` | Nouveau — page configuration modèle |
| `app/main.py` | Charger override DB au lifespan + inclure router settings |

---

## Hors scope

- Interface d'administration des catégories
- Auto-création de nouvelles catégories depuis `NEW:` (risque hallucinations)
- Vrai worker async (Celery/ARQ) — `BackgroundTasks` FastAPI suffisant à cette échelle
- Liste des modèles disponibles OpenRouter (pas d'appel API de discovery)
