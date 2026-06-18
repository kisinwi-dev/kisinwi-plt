"""Общие пути пакета и загрузка .env."""
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def load_env() -> None:
    """Загрузить .env из корня benchmarks. override=False: переменные окружения важнее."""
    load_dotenv(ROOT / ".env")
