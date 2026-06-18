"""Прогон датасетов через полный пайплайн платформы.

Для каждого датасета: старт пайплайна, ожидание задачи обучения в tasker, затем
ожидание финального статуса дискуссии. В results.json пишутся ссылки на итог
(discussion_id/model_id, URL фронтенда) и статусы.

Точка входа - run_keys(); управление через benchmark.py.
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from bench import DATA_DIR, load_env

load_env()  # .env до импорта config: иначе override BENCH_*_URL не подхватится

from bench.config import DATASETS, DEFAULT_SUITE, FRONTEND_URL
from bench.common import (
    load_registry, start_development, wait_for_task, wait_for_discussion,
    get_tasks_by_discussion, dataset_exists,
)


def run_one(key: str, entry: dict, llm_model: str | None = None) -> dict:
    cfg = DATASETS[key]
    dataset_id = entry["dataset_id"]
    version_id = entry["version_id"]

    started = time.perf_counter()
    print(f"\n=== {key} | {cfg['name']} (dataset_id={dataset_id}) ===")
    payload = {
        "dataset_id": dataset_id,
        "version_id": version_id,
        "model_name": f"bench_{key}",
        "business_requirements": f"accuracy >= {cfg['baseline']:.2f}",
        "title": f"Benchmark {cfg['name']}",
        "tags": ["benchmark"],
    }
    if llm_model:
        payload["llm_model"] = llm_model
    discussion_id = start_development(payload)
    print(f"  discussion_id={discussion_id}")

    task = wait_for_task(discussion_id)

    # Обучение завершилось, но пайплайн агентов ещё нет (дальше анализ метрик и
    # отчёт). Ждём дискуссию, иначе следующий датасет стартует поверх неё.
    meta = wait_for_discussion(discussion_id)

    # Задач обучения может быть несколько (агенты переобучают) - берём последнюю.
    tasks = get_tasks_by_discussion(discussion_id)
    if tasks:
        task = tasks[-1]
    model_id = task.get("model_id")
    # Число задач обучения - сигнал качества агентов.
    n_train_tasks = len(tasks)

    result = {
        "key": key,
        "dataset_id": dataset_id,
        "dataset_name": cfg["name"],
        "discussion_id": discussion_id,
        "discussion_url": f"{FRONTEND_URL}/agents/discussion/{discussion_id}",
        "model_id": model_id,
        "model_url": f"{FRONTEND_URL}/models/{model_id}" if model_id else None,
        "baseline": cfg["baseline"],
        "discussion_status": meta.get("status") if meta else None,
        "task_status": task["status"],
        "task_error": task.get("error_message"),
        "num_classes": entry.get("num_classes"),
        "llm_model": llm_model,
        "n_train_tasks": n_train_tasks,
        "duration_sec": round(time.perf_counter() - started, 1),
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }

    if task["status"] != "completed":
        print(f"  задача не completed: {task['status']} / {task.get('error_message')}")
    else:
        print(f"  workflow завершён (discussion={result['discussion_status']})")
        print(f"  дискуссия: {result['discussion_url']}")
        print(f"  модель:    {result['model_url']}")
    return result


def run_keys(keys: list[str] | None = None, model: str | None = None,
             out: str = str(DATA_DIR / "results.json")) -> dict:
    """Прогнать ключи через пайплайн, мерж результатов в out по key.

    Возвращает {key: result}. Без keys используется DEFAULT_SUITE.
    """
    keys = keys or DEFAULT_SUITE
    reg = load_registry()

    # Мерж с существующими результатами по key: можно прогонять порциями.
    by_key = {}
    if Path(out).exists():
        for r in json.loads(Path(out).read_text(encoding="utf-8")):
            by_key[r.get("key")] = r

    def dump():
        with open(out, "w", encoding="utf-8") as f:
            json.dump(list(by_key.values()), f, ensure_ascii=False, indent=2)

    for key in keys:
        entry = reg.get(key)
        if not entry:
            print(f"!! {key} не в реестре - пропуск (сначала подготовь через benchmark.py)")
            continue
        # Реестр мог запомнить id датасета, которого уже нет в платформе.
        if not dataset_exists(entry["dataset_id"]):
            print(f"!! {key} есть в реестре (id={entry['dataset_id']}), но в платформе "
                  f"отсутствует - пропуск (подготовь заново через benchmark.py)")
            by_key[key] = {"key": key, "dataset_id": entry["dataset_id"],
                           "error": "dataset not found in platform"}
            dump()
            continue
        try:
            by_key[key] = run_one(key, entry, model)
        except Exception as e:
            print(f"!! {key} упал: {e}")
            by_key[key] = {"key": key, "dataset_id": entry.get("dataset_id"), "error": str(e)}
        dump()
        print(f"  -> сохранено в {out} (всего {len(by_key)})")

    print(f"\nГотово. Результатов: {len(by_key)} -> {out}")
    return by_key
