"""Единая точка входа benchmark-прогона платформы KiSinWi.

Делает только суть: подготовка датасета(ов) (скачать с HF -> конвертировать ->
загрузить в платформу) и прогон через полный пайплайн. На выходе — ссылки на
итог обучения (дискуссия и модель) в results.json.

Примеры:
    # весь набор: подготовить недостающее и прогнать всё
    python benchmark.py all

    # один датасет
    python benchmark.py beans

    # несколько конкретных
    python benchmark.py cifar10 flowers102

    # только прогнать (датасеты уже загружены)
    python benchmark.py all --skip-prepare

Ключи датасетов: см. config.DATASETS. Спецслово `all` = config.DEFAULT_SUITE.
"""
import argparse
import subprocess
import sys
from pathlib import Path

from config import DATASETS, DEFAULT_SUITE, DEFAULT_MAX_ITER
from common import load_registry, dataset_exists

HERE = Path(__file__).parent
PY = sys.executable


def _run(cmd: list[str]):
    print(f"\n$ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        raise SystemExit(f"шаг упал: {' '.join(cmd)}")


def resolve_model(arg_model: str | None) -> str | None:
    """Определить LLM-модель агентов для прогона.

    Флаг --model имеет приоритет. Если флаг не задан и запуск интерактивный
    (stdin — терминал), спрашиваем id модели свободным вводом. В неинтерактивном
    запуске (CI, пайп) молча берём дефолт платформы, чтобы не блокироваться на
    вводе. Пустой ввод (или None) означает «дефолт платформы»: поле llm_model
    в payload не уходит.
    """
    if arg_model:
        return arg_model.strip() or None

    if not sys.stdin.isatty():
        return None

    raw = input("\nВведит название используемой LLM (Enter — дефолт платформы): ").strip()
    return raw or None


def prepare_and_run(keys, *, max_iter, skip_prepare, model, out) -> str:
    """Подготовить (опц.) и прогнать набор. Возвращает файл результатов."""
    if not skip_prepare:
        reg = load_registry()
        for key in keys:
            entry = reg.get(key)
            if entry and dataset_exists(entry["dataset_id"]):
                print(f"[{key}] уже загружен в платформу — пропуск подготовки")
                continue
            if entry:
                print(f"[{key}] есть в реестре, но в платформе отсутствует "
                      f"(id={entry['dataset_id']}) — готовлю заново")
            _run([PY, "prepare_dataset.py", key])

    run_cmd = [PY, "run_benchmark.py", *keys, "--max-iter", str(max_iter), "--out", out]
    if model:
        run_cmd += ["--model", model]
    _run(run_cmd)
    return out


def resolve_keys(tokens: list[str]) -> list[str]:
    if not tokens or tokens == ["all"]:
        return list(DEFAULT_SUITE)
    bad = [t for t in tokens if t not in DATASETS]
    if bad:
        raise SystemExit(f"неизвестные датасеты: {bad}. Доступны: {list(DATASETS)} или 'all'")
    return tokens


def main():
    ap = argparse.ArgumentParser(description="Benchmark-прогон платформы KiSinWi")
    ap.add_argument("datasets", nargs="*",
                    help="ключи датасетов или 'all' (по умолчанию all)")
    ap.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER,
                    help="итераций пайплайна (recovery при неудачном выборе архитектуры)")
    ap.add_argument("--skip-prepare", action="store_true",
                    help="не готовить датасеты (уже загружены в платформу)")
    ap.add_argument("--out", default="results.json",
                    help="файл со ссылками на итог прогона (по умолчанию results.json)")
    ap.add_argument("--model", default=None,
                    help="LLM-модель агентов (id из каталога /settings/llm). "
                         "Не задан — интерактивный выбор.")
    args = ap.parse_args()

    keys = resolve_keys(args.datasets)
    model = resolve_model(args.model)

    print(f"Датасеты: {keys}")
    print(f"max_iter={args.max_iter}")
    print(f"LLM-модель агентов: {model or 'дефолт платформы'}")

    out = prepare_and_run(keys, max_iter=args.max_iter,
                          skip_prepare=args.skip_prepare, model=model, out=args.out)
    print(f"\nГотово. Ссылки на дискуссии и модели — в {out}")


if __name__ == "__main__":
    main()
