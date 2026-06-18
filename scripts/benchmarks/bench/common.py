"""HTTP-хелперы к API платформы и упаковка датасетов в zip.

upload_archive стримит zip без чтения в память: master_kaggle +десятки ГБ,
а requests files= читает файл целиком и падает с MemoryError.
"""
import json
import os
import time
import zipfile
from pathlib import Path

import requests
from requests_toolbelt import MultipartEncoder

from bench import DATA_DIR
from bench.config import (
    DATASETS_URL, AGENTS_URL, TASKER_URL, AGENT_HISTORY_URL,
)

FINAL_STATUSES = {"completed", "failed", "cancelled"}

# Локальный реестр key -> {dataset_id, version_id, name}: id датасетов платформа
# выдаёт как UUID, прогону нужно знать реальные.
REGISTRY_PATH = DATA_DIR / "datasets_registry.json"


def load_registry() -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return {}


def save_registry(reg: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(reg, ensure_ascii=False, indent=2),
                             encoding="utf-8")


# --- упаковка ---

def zip_folder(folder: Path) -> Path:
    """Запаковать папку в одноимённый zip (ZIP_STORED - картинки уже сжаты)."""
    zip_path = folder.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for root, _, files in os.walk(folder):
            for fn in files:
                fp = Path(root) / fn
                zf.write(fp, fp.relative_to(folder))
    size = zip_path.stat().st_size
    human = f"{size / 1e9:.2f} GB" if size >= 1e9 else f"{size / 1e6:.1f} MB"
    print(f"[zip] {zip_path.name}: {human}")
    return zip_path


# --- datasets (6500) ---

def upload_archive(id_data: str, zip_path: str) -> None:
    """POST /upload - стриминговая загрузка zip.

    MultipartEncoder даёт file-like с известной длиной: requests ставит
    Content-Length и не переходит на chunked (uvicorn его отвергает).
    """
    with open(zip_path, "rb") as fp:
        m = MultipartEncoder(fields={
            "id_data": id_data,
            "file": (f"{id_data}.zip", fp, "application/zip"),
        })
        resp = requests.post(f"{DATASETS_URL}/upload", data=m,
                             headers={"Content-Type": m.content_type}, timeout=3600)
    resp.raise_for_status()


def create_dataset(payload: dict) -> None:
    """POST /datasets/new - создание датасета и версии из загруженных данных."""
    resp = requests.post(f"{DATASETS_URL}/datasets/new", json=payload, timeout=600)
    resp.raise_for_status()


def list_datasets() -> list[dict]:
    """GET /datasets/ - все датасеты платформы."""
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
    """Есть ли датасет с таким id в платформе сейчас.

    Реестр может помнить id, которых уже нет (тома db/ пересоздавались).
    """
    try:
        resp = requests.get(f"{DATASETS_URL}/datasets/{dataset_id}", timeout=30)
        return resp.status_code == 200
    except requests.RequestException:
        return False


# --- agents (6400) ---

def start_development(payload: dict) -> str:
    """POST /development/start - асинхронный запуск пайплайна. Возвращает discussion_id."""
    resp = requests.post(f"{AGENTS_URL}/development/start", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["discussion_id"]


# --- tasker (6110) ---

def get_tasks_by_discussion(discussion_id: str) -> list[dict]:
    """GET /tasks?discussion_id=... - задачи дискуссии, отсортированные по created_at.

    Порядок выдачи API не гарантирован, а вызывающий код берёт tasks[-1] как финальную.
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
    """Метаданные дискуссии (status всего пайплайна)."""
    try:
        resp = requests.get(
            f"{AGENT_HISTORY_URL}/discussions/{discussion_id}/meta", timeout=30
        )
        if resp.status_code == 200:
            return resp.json()
    except requests.RequestException:
        pass
    return None


# --- высокоуровневые ожидания ---

# Порог HTTP-ошибок подряд: за многочасовой прогон одиночный 5xx/таймаут
# не должен ронять датасет.
MAX_CONSECUTIVE_HTTP_FAILS = 10


def wait_for_task(discussion_id: str, poll: int = 10, timeout: int = 14400) -> dict:
    """Ждёт появления задачи обучения и её финального статуса.

    Возвращает финальную запись задачи. TimeoutError при превышении timeout.
    Одиночные сетевые ошибки игнорируются, падаем после серии подряд.
    """
    start = time.time()
    task_id = None
    fails = 0

    # ждём создания задачи обучения агентами
    while task_id is None:
        if time.time() - start > timeout:
            raise TimeoutError("Задача обучения не появилась за отведённое время")
        try:
            tasks = get_tasks_by_discussion(discussion_id)
            fails = 0
        except requests.RequestException as e:
            fails += 1
            print(f"  опрос задач не удался ({fails}/{MAX_CONSECUTIVE_HTTP_FAILS}): {e}")
            if fails >= MAX_CONSECUTIVE_HTTP_FAILS:
                raise
            time.sleep(poll)
            continue
        if tasks:
            task = tasks[-1]
            task_id = task["id"]
            print(f"  задача создана: task_id={task_id} model_id={task.get('model_id')}")
            break
        # Задачи нет. Если дискуссия уже завершилась - обучения не будет.
        meta = get_discussion_meta(discussion_id)
        if meta and meta.get("status") in FINAL_STATUSES:
            raise RuntimeError(
                f"Дискуссия {discussion_id} завершилась "
                f"(status={meta.get('status')}) без задачи обучения"
            )
        time.sleep(poll)

    # ждём финального статуса
    last_pct = -1
    while True:
        if time.time() - start > timeout:
            raise TimeoutError(f"Задача {task_id} не завершилась за отведённое время")
        try:
            task = get_task(task_id)
            fails = 0
        except requests.RequestException as e:
            fails += 1
            print(f"  опрос статуса не удался ({fails}/{MAX_CONSECUTIVE_HTTP_FAILS}): {e}")
            if fails >= MAX_CONSECUTIVE_HTTP_FAILS:
                raise
            time.sleep(poll)
            continue
        status = task["status"]
        pct = task.get("percentages") or 0
        if pct != last_pct and status == "running":
            print(f"  обучение: {pct}%")
            last_pct = pct
        if status in FINAL_STATUSES:
            print(f"  финальный статус: {status}")
            return task
        time.sleep(poll)


def wait_for_discussion(discussion_id: str, poll: int = 10, timeout: int = 14400,
                        max_empty: int = 30) -> dict | None:
    """Ждёт финального статуса всей дискуссии агентов.

    Дискуссия завершается позже задачи обучения (после неё идут анализ метрик и
    отчёт), поэтому ждём именно её. Возвращает финальную meta или None, если meta
    недоступна max_empty раз подряд. TimeoutError при превышении timeout.
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
                print(f"  статус дискуссии: {status}")
                last_status = status
            if status in FINAL_STATUSES:
                return meta
        else:
            empty += 1
            if empty >= max_empty:
                print(f"  meta дискуссии недоступна {empty} раз подряд - ожидание пропущено")
                return None
        time.sleep(poll)
