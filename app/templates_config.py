"""
Configuration centralisée des templates Jinja2.

Instance unique avec :
- filtre dateadd (utilisé dans task_card.html pour les boutons de date rapide)
- chemins absolus (robuste quel que soit le répertoire de lancement)
"""

from datetime import timedelta
from pathlib import Path

from fastapi.templating import Jinja2Templates

_TEMPLATES_DIR = Path(__file__).parent / "templates"

templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
templates.env.filters["dateadd"] = lambda d, days: d + timedelta(days=days)
