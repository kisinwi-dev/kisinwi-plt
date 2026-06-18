"""Загрузка сырых Kaggle-датасетов (real vs deepfake).

Опциональный пред-шаг для master_kaggle (иначе скачается автоматически). Нужен
Kaggle API token в .env (см. .env.example). Точка входа - download_kaggle();
управление через `benchmark.py download`.
"""

from pathlib import Path

from bench import load_env, DATA_DIR

load_env()

from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402 - импорт после load_env

# Те же 6 датасетов, что и в референс-ноутбуке; bench.prepare.kaggle собирает из них набор.
from bench.config import KAGGLE_REFS  # noqa: E402 - импорт после load_env


def download_kaggle(target_dir: Path = DATA_DIR / "datasets", unzip: bool = False) -> None:
    """Скачать 6 Kaggle-датасетов в target_dir (нужна Kaggle-аутентификация).

    Ошибка одного датасета не роняет остальные. unzip=True - распаковать сразу.
    """
    api = KaggleApi()
    api.authenticate()

    failed = []
    for i, ref in enumerate(KAGGLE_REFS, 1):
        dest = target_dir / ref.split("/")[-1].strip("-")
        # уже скачан, если папка не пуста
        if dest.exists() and any(dest.iterdir()):
            print(f"\n[{i}/{len(KAGGLE_REFS)}] {ref} -> {dest} - уже есть, пропуск")
            continue
        dest.mkdir(parents=True, exist_ok=True)
        print(f"\n[{i}/{len(KAGGLE_REFS)}] {ref} -> {dest}")

        try:
            api.dataset_download_files(ref, path=str(dest), unzip=unzip, quiet=False)
        except Exception as e:
            failed.append(ref)
            if not any(dest.iterdir()):
                dest.rmdir()  # пустую папку убираем, иначе примем за уже скачанную
            print(f"  не скачан: {e}")
            print(f"  ссылка: https://www.kaggle.com/datasets/{ref}")

    ok = len(KAGGLE_REFS) - len(failed)
    print(f"\nГотово: {ok}/{len(KAGGLE_REFS)} в {target_dir}/")
    if failed:
        print("Не скачаны:", ", ".join(failed))
