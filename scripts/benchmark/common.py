"""Общие HTTP-хелперы для работы с API платформы KiSinWi."""
import json
import time
from pathlib import Path

import requests

from config import (
    DATASETS_URL, AGENTS_URL, TASKER_URL, AGENT_HISTORY_URL,
)

FINAL_TASK_STATUSES = {"completed", "failed", "cancelled"}
FINAL_DISCUSSION_STATUSES = {"completed", "failed", "cancelled"}

# Платформа генерирует UUID для dataset_id, поэтому ведём локальный реестр
# key -> {dataset_id, version_id, name}, чтобы прогон знал реальные id.
REGISTRY_PATH = Path(__file__).with_name("datasets_registry.json")


def load_registry() -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return {}


def save_registry(reg: dict) -> None:
    REGISTRY_PATH.write_text(json.dumps(reg, ensure_ascii=False, indent=2),
                             encoding="utf-8")


# --- datasets (6500) ---

def upload_archive(id_data: str, zip_path: str) -> None:
    """POST /upload — загрузка zip во временное хранилище."""
    with open(zip_path, "rb") as f:
        resp = requests.post(
            f"{DATASETS_URL}/upload",
            data={"id_data": id_data},
            files={"file": (f"{id_data}.zip", f, "application/zip")},
            timeout=600,
        )
    resp.raise_for_status()


def create_dataset(payload: dict) -> None:
    """POST /datasets/new — создание датасета и версии из загруженных данных."""
    resp = requests.post(f"{DATASETS_URL}/datasets/new", json=payload, timeout=600)
    resp.raise_for_status()


def list_datasets() -> list[dict]:
    """GET /datasets/ — все датасеты платформы."""
    resp = requests.get(f"{DATASETS_URL}/datasets/", timeout=30)
    resp.raise_for_status()
    return resp.json()


def find_dataset_by_name(name: str) -> dict | None:
    """Самый свежий датасет с указанным именем (платформа выдаёт UUID id)."""
    matches = [d for d in list_datasets() if d.get("name") == name]
    if not matches:
        return None
    return sorted(matches, key=lambda d: d.get("created_at", ""))[-1]


def dataset_exists(dataset_id: str) -> bool:
    """Реально ли датасет с таким id есть в платформе сейчас.

    Локальный реестр может помнить id датасетов, которых уже нет (тома db/
    пересоздавались) — тогда подготовку надо запускать заново.
    """
    try:
        resp = requests.get(f"{DATASETS_URL}/datasets/{dataset_id}", timeout=30)
        return resp.status_code == 200
    except requests.RequestException:
        return False


# --- agents (6400) ---

def start_development(payload: dict) -> str:
    """POST /development/start — асинхронный запуск пайплайна. Возвращает discussion_id."""
    resp = requests.post(f"{AGENTS_URL}/development/start", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["discussion_id"]


# --- tasker (6110) ---

def get_tasks_by_discussion(discussion_id: str) -> list[dict]:
    """GET /tasks?discussion_id=... — задачи дискуссии (связка discussion -> model_id/task_id).

    Сортируем по created_at: порядок выдачи API не гарантирован, а вызывающий код
    берёт tasks[-1] как финальную (последнюю) задачу прогона.
    """
    resp = requests.get(
        f"{TASKER_URL}/tasks", params={"discussion_id": discussion_id}, timeout=30
    )
    resp.raise_for_status()
    tasks = resp.json().get("tasks", [])
    return sorted(tasks, key=lambda t: t.get("created_at") or "")


def get_task(task_id: str) -> dict:
    resp = requests.get(f"{TASKER_URL}/tasks/{task_id}", timeout=30)
    resp.raise_for_status()
    return resp.json()


# --- agent_history (6410) ---

def get_discussion_meta(discussion_id: str) -> dict | None:
    """Метаданные дискуссии (status всего пайплайна).

    Пробуем оба варианта пути (`/discussions/` и `/discussion/`): имя ресурса
    между инсталляциями расходится.
    """
    for base in (f"/discussions/{discussion_id}/meta",
                 f"/discussion/{discussion_id}/meta"):
        try:
            resp = requests.get(f"{AGENT_HISTORY_URL}{base}", timeout=30)
            if resp.status_code == 200:
                return resp.json()
        except requests.RequestException:
            continue
    return None


# --- высокоуровневые ожидания ---

# Сколько HTTP-ошибок подряд терпим в полл-циклах, прежде чем сдаться. За
# многочасовой прогон одиночный 5xx/таймаут не должен ронять весь датасет.
MAX_CONSECUTIVE_HTTP_FAILS = 10


def wait_for_task(discussion_id: str, poll: int = 10, timeout: int = 14400,
                  log=print) -> dict:
    """
    Ждёт появления задачи обучения у дискуссии и её финального статуса.

    Возвращает финальную запись задачи (с model_id, status, error_message).
    Бросает TimeoutError при превышении timeout. Одиночные сетевые ошибки в
    цикле логируются и игнорируются; падаем лишь после серии подряд.
    """
    start = time.time()
    task_id = None
    fails = 0

    # 1) дождаться, пока агенты создадут задачу обучения
    while task_id is None:
        if time.time() - start > timeout:
            raise TimeoutError("Задача обучения не появилась за отведённое время")
        try:
            tasks = get_tasks_by_discussion(discussion_id)
            fails = 0
        except requests.RequestException as e:
            fails += 1
            log(f"  ⚠ опрос задач не удался ({fails}/{MAX_CONSECUTIVE_HTTP_FAILS}): {e}")
            if fails >= MAX_CONSECUTIVE_HTTP_FAILS:
                raise
            time.sleep(poll)
            continue
        if tasks:
            task = tasks[-1]
            task_id = task["id"]
            log(f"  задача создана: task_id={task_id} model_id={task.get('model_id')}")
        # Задачи ещё нет. Если дискуссия уже завершилась (failed/cancelled или даже
        # completed без обучения) — задача обучения не появится, нет смысла ждать
        # до timeout. Если meta недоступна (None) — ведём себя как раньше.
        meta = get_discussion_meta(discussion_id)
        if meta and meta.get("status") in FINAL_DISCUSSION_STATUSES:
            raise RuntimeError(
                f"Дискуссия {discussion_id} завершилась "
                f"(status={meta.get('status')}) без задачи обучения"
            )
        time.sleep(poll)

    # 2) дождаться финального статуса
    last_pct = -1
    while True:
        if time.time() - start > timeout:
            raise TimeoutError(f"Задача {task_id} не завершилась за отведённое время")
        try:
            task = get_task(task_id)
            fails = 0
        except requests.RequestException as e:
            fails += 1
            log(f"  ⚠ опрос статуса не удался ({fails}/{MAX_CONSECUTIVE_HTTP_FAILS}): {e}")
            if fails >= MAX_CONSECUTIVE_HTTP_FAILS:
                raise
            time.sleep(poll)
            continue
        status = task["status"]
        pct = task.get("percentages") or 0
        if pct != last_pct and status == "running":
            log(f"  обучение: {pct}%")
            last_pct = pct
        if status in FINAL_TASK_STATUSES:
            log(f"  финальный статус: {status}")
            return task
        time.sleep(poll)


def wait_for_discussion(discussion_id: str, poll: int = 10, timeout: int = 14400,
                        max_empty: int = 30, log=print) -> dict | None:
    """
    Ждёт финального статуса всей дискуссии агентов.

    Задача обучения в tasker завершается раньше всего пайплайна: после обучения
    агенты ещё анализируют метрики и пишут отчёт, и лишь затем дискуссия получает
    статус completed/failed/cancelled. Ждать надо именно его, иначе следующий
    датасет стартует, пока предыдущий workflow ещё доделывается.

    Возвращает финальную meta дискуссии. Бросает TimeoutError при превышении timeout.
    Если meta стабильно недоступна (agent_history не отдаёт /meta), мягко сдаёмся
    после max_empty пустых ответов подряд и возвращаем None — не висим весь timeout.
    """
    start = time.time()
    last_status = None
    empty = 0
    while True:
        if time.time() - start > timeout:
            raise TimeoutError(
                f"Дискуссия {discussion_id} не завершилась за отведённое время"
            )
        meta = get_discussion_meta(discussion_id)
        if meta:
            empty = 0
            status = meta.get("status")
            if status != last_status:
                log(f"  статус дискуссии: {status}")
                last_status = status
            if status in FINAL_DISCUSSION_STATUSES:
                return meta
        else:
            empty += 1
            if empty >= max_empty:
                log(f"  ⚠ meta дискуссии недоступна {empty} раз подряд — "
                    f"пропускаю ожидание дискуссии")
                return None
        time.sleep(poll)
