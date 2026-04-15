# Déploiement sur VPS (Docker)

## Prérequis

- VPS Linux avec Docker + Docker Compose installés
- Nom de domaine pointant vers le VPS (pour HTTPS via Caddy ou Nginx)
- Clé API OpenRouter

## Étapes

### 1. Cloner le repo sur le VPS

```bash
git clone <repo> /opt/task-assistant
cd /opt/task-assistant
```

### 2. Créer le fichier `.env`

```bash
cp .env.example .env
nano .env
```

Variables obligatoires en prod :

```env
ENVIRONMENT=prod
OPENROUTER_API_KEY=sk-or-...
SECRET_KEY=<chaine-aleatoire-longue>
APP_PASSWORD_HASH=<hash-bcrypt>
```

Générer `SECRET_KEY` :
```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

Générer `APP_PASSWORD_HASH` :
```bash
python3 -c "import bcrypt; print(bcrypt.hashpw(b'monmotdepasse', bcrypt.gensalt()).decode())"
```

### 3. Créer le répertoire de données

```bash
mkdir -p data
```

### 4. Lancer

```bash
docker compose up -d
```

### 5. Initialiser la base de données

```bash
docker compose exec app alembic upgrade head
```

Le seed des catégories et contextes par défaut s'effectue automatiquement au premier démarrage.

### 6. Vérifier

```bash
docker compose logs -f app
```

L'app est accessible sur le port 8000. Placer un reverse proxy (Caddy, Nginx, Traefik) devant pour HTTPS.

---

## Exemple Caddy (HTTPS automatique)

```caddyfile
tasks.exemple.com {
    reverse_proxy localhost:8000
}
```

---

## Mise à jour

```bash
git pull
docker compose build
docker compose up -d
docker compose exec app alembic upgrade head
```

---

## Sauvegardes

La base de données est dans `./data/tasks.db` (volume monté).

Sauvegarde simple :

```bash
# Sur le VPS
cp data/tasks.db data/tasks.db.bak-$(date +%Y%m%d)

# Ou vers machine locale
rsync -avz user@vps:/opt/task-assistant/data/tasks.db ./backups/
```

---

## Variables d'environnement complètes

| Variable | Défaut | Description |
|---|---|---|
| `ENVIRONMENT` | `dev` | `prod` active les validations de sécurité |
| `OPENROUTER_API_KEY` | *(requis)* | Clé API OpenRouter |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | URL de l'API |
| `LLM_MODEL` | `google/gemini-2.5-flash` | Modèle principal |
| `LLM_TIMEOUT` | `15.0` | Timeout LLM en secondes |
| `DATABASE_URL` | `sqlite+aiosqlite:///./data/tasks.db` | URL SQLAlchemy |
| `SECRET_KEY` | `changeme` | Clé cookie (obligatoire en prod) |
| `APP_PASSWORD_HASH` | *(requis)* | Hash bcrypt |
| `DAILY_CAPACITY_MINUTES` | `420` | Capacité journalière (minutes) |
| `DIGEST_ENABLED` | `true` | Digest matinal 7h00 |
| `ZOMBIE_THRESHOLD_DAYS` | `21` | Seuil zombie (jours) |
