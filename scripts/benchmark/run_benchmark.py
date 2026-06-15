"""Прогон датасетов через полный пайплайн платформы.

Для каждого датасета:
  1) POST /development/start -> discussion_id
  2) ждём задачу в tasker (discussion_id -> task_id, model_id) и её финальный статус
  3) ждём финальный статус всей дискуссии агентов (после обучения идут анализ
     метрик и отчёт) — только тогда workflow действительно завершён

В results.json пишутся ссылки на итог обучения: discussion_id/model_id и готовые
URL фронтенда (discussion_url, model_url), плюс статусы прогона. Сами метрики и
история не дублируются — источник правды это сервисы и UI платформы.

Запуск:
    python run_benchmark.py [keys...] [--max-iter N] [--out results.json]
Без keys используется DEFAULT_SUITE из config.
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from config import DATASETS, DEFAULT_SUITE, DEFAULT_MAX_ITER, FRONTEND_URL
from common import (
    load_registry, start_development, wait_for_task, wait_for_discussion,
    get_tasks_by_discussion,
)


def run_one(key: str, entry: dict, max_iter: int, llm_model: str | None = None) -> dict:
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
        "max_iter": max_iter,
        "title": f"Benchmark {cfg['name']}",
        "tags": ["benchmark"],
    }
    if llm_model:
        payload["llm_model"] = llm_model
    discussion_id = start_development(payload)
    print(f"  discussion_id={discussion_id}")

    task = wait_for_task(discussion_id)

    # Задача обучения завершилась, но весь пайплайн агентов ещё нет: после
    # обучения идут анализ метрик и отчёт. Ждём финальный статус всей дискуссии,
    # иначе следующий датасет стартует поверх недоделанного workflow.
    meta = wait_for_discussion(discussion_id)

    # При recovery (max_iter) задач обучения могло быть несколько — берём
    # последнюю как финальную модель прогона.
    tasks = get_tasks_by_discussion(discussion_id)
    if tasks:
        task = tasks[-1]
    model_id = task.get("model_id")
    # Сколько задач обучения было в дискуссии: при recovery (max_iter) агенты
    # могли промахнуться с архитектурой и переобучать заново. Сигнал качества
    # агентов, которого по одной модели в UI не видно.
    n_train_tasks = len(tasks)

    result = {
        "key": key,
        "dataset_id": dataset_id,
        "dataset_name": cfg["name"],
        "discussion_id": discussion_id,
        "discussion_url": f"{FRONTEND_URL}/agents/discussion/{discussion_id}",
        "model_id": model_id,
        "model_url": f"{FRONTEND_URL}/models/{model_id}" if model_id else None,
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
        print(f"  ⚠ задача не completed: {task['status']} / {task.get('error_message')}")
    else:
        print(f"  workflow завершён (discussion={result['discussion_status']})")
        print(f"  дискуссия: {result['discussion_url']}")
        print(f"  модель:    {result['model_url']}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="*", default=None)
    ap.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--model", default=None,
                    help="LLM-модель агентов (id). Не задан — дефолт платформы")
    args = ap.parse_args()

    keys = args.keys or DEFAULT_SUITE
    reg = load_registry()

    # Мержим с существующими результатами по key — можно прогонять порциями.
    by_key = {}
    if Path(args.out).exists():
        for r in json.loads(Path(args.out).read_text(encoding="utf-8")):
            by_key[r.get("key")] = r

    def dump():
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(list(by_key.values()), f, ensure_ascii=False, indent=2)

    for key in keys:
        entry = reg.get(key)
        if not entry:
            print(f"!! {key} не в реестре — пропуск (сначала prepare_dataset.py)")
            continue
        try:
            by_key[key] = run_one(key, entry, args.max_iter, args.model)
        except Exception as e:
            print(f"!! {key} упал: {e}")
            by_key[key] = {"key": key, "dataset_id": entry.get("dataset_id"), "error": str(e)}
        dump()
        print(f"  -> сохранено в {args.out} (всего {len(by_key)})")

    print(f"\nГотово. Результатов: {len(by_key)} -> {args.out}")


if __name__ == "__main__":
    main()
