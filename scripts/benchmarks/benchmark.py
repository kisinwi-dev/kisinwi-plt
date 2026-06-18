"""Единая точка входа benchmark-прогона.

Этапы: загрузка датасетов, конвертация в формат платформы, прогон через полный
пайплайн, ссылки на итог (дискуссия и модель) в results.json.

Примеры:
    # весь набор: подготовить недостающее и прогнать всё
    python benchmark.py all

    # один датасет
    python benchmark.py beans

    # несколько конкретных
    python benchmark.py cifar10 flowers102

    # только прогнать (датасеты уже загружены)
    python benchmark.py all --skip-prepare

    # Kaggle fake-пул
    python benchmark.py master_kaggle

    # скачать данные для датасета fake из Kaggle заранее
    python benchmark.py download

Ключи датасетов: см. config.DATASETS
"""
import argparse
import sys

from bench import DATA_DIR, load_env

load_env()

from bench.config import DATASETS, DEFAULT_SUITE
from bench.common import load_registry, dataset_exists


def resolve_model(arg_model: str | None) -> str | None:
    """LLM-модель агентов: флаг --model, иначе интерактивный ввод.

    Неинтерактивный запуск или пустой ввод - дефолт платформы (llm_model не уходит в payload).
    """
    if arg_model:
        return arg_model.strip() or None

    if not sys.stdin.isatty():
        return None

    raw = input("\nВведите название используемой LLM (Enter - дефолт платформы): ").strip()
    return raw or None


def prepare_and_run(keys, *, skip_prepare, model, out) -> str:
    """Подготовить и прогнать набор. Возвращает файл результатов."""
    if not skip_prepare:
        reg = load_registry()
        for key in keys:
            entry = reg.get(key)
            
            if entry and dataset_exists(entry["dataset_id"]):
                print(f"[{key}] уже загружен в платформу. Пропуск подготовки")
                continue
            
            if entry:
                print(f"[{key}] есть в реестре, но в платформе отсутствует "
                      f"(id={entry['dataset_id']}) - подготовка заново")
            
            # тяжёлый datasets тянется только при HF-подготовке
            if DATASETS[key]["source"] == "kaggle":
                # master_kaggle - один набор
                from bench.prepare.kaggle import prepare_kaggle
                prepare_kaggle()
            else:
                from bench.prepare.hf import prepare_hf
                prepare_hf(key)

    from bench.run import run_keys
    run_keys(keys, model=model, out=out)
    return out


def resolve_keys(tokens: list[str]) -> list[str]:
    if not tokens or tokens == ["all"]:
        return list(DEFAULT_SUITE)
    bad = [t for t in tokens if t not in DATASETS]
    if bad:
        raise SystemExit(f"неизвестные датасеты: {bad}. Доступны: {list(DATASETS)} или 'all'")
    return tokens


def main():
    ap = argparse.ArgumentParser(description="Benchmark прогон платформы")
    ap.add_argument("datasets", nargs="*",
                    help="ключи датасетов, 'all' (по умолчанию) или 'download'")
    ap.add_argument("--skip-prepare", action="store_true",
                    help="не готовить датасеты (уже загружены в платформу)")
    ap.add_argument("--out", default=str(DATA_DIR / "results.json"),
                    help="файл со ссылками на итог прогона (по умолчанию data/results.json)")
    ap.add_argument("--model", default=None,
                    help="LLM-модель агентов (id из каталога /settings/llm). "
                         "Не задан - интерактивный выбор.")
    ap.add_argument("--unzip", action="store_true",
                    help="для 'download': распаковать сразу (иначе оставить .zip)")
    args = ap.parse_args()

    # download - спец-токен (загрузка данных из Kaggle для сбора датасета),
    if args.datasets[:1] == ["download"]:
        from bench.prepare.download import download_kaggle
        download_kaggle(unzip=args.unzip)
        return

    keys = resolve_keys(args.datasets)
    model = resolve_model(args.model)

    print(f"Датасеты: {keys}")
    print(f"LLM-модель агентов: {model or 'дефолт платформы'}")

    out = prepare_and_run(keys, skip_prepare=args.skip_prepare, model=model, out=args.out)
    print(f"\nГотово. Ссылки на дискуссии и модели - в {out}")


if __name__ == "__main__":
    main()
