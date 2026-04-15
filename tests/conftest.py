"""
Configuration pytest globale.

Les variables d'environnement doivent être posées AVANT tout import de app.*
car app.config.settings est instancié au niveau module.
"""

import os

# Valeurs minimales pour que Settings() ne lève pas à la collection
os.environ.setdefault("OPENROUTER_API_KEY", "sk-test-dummy")
os.environ.setdefault("APP_PASSWORD_HASH", "$2b$04$UCgIWmqmpBHxtAKVoP81..6Z2IkEY4GLQl4mZpaqNnzzSBj2rcPU2")
os.environ.setdefault("SECRET_KEY", "test-secret-key-not-for-production")
os.environ.setdefault("ENVIRONMENT", "dev")
