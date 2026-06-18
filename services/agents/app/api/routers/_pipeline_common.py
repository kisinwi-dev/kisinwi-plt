import multiprocessing
import threading
from uuid import uuid4
from typing import Any, Callable, Dict, List, Optional

from fastapi import HTTPException
from pydantic import BaseModel, Field, model_validator

from app.core.memory import (
    discussion_context, llm_model_context, id_alias_context, dataset_context,
)
from app.core.cancellation import process_registry
from app.core.llm import get_current_model
from app.services.agent_history import agent_history_client
from app.services.datasets import get_dataset_details, get_dataset_version_details
from app.services.ml_models import ml_models_client
from app.services.tasker import tasker_client
from app.logs import get_logger

logger = get_logger(__name__)

# Пайплайн выполняется в отдельном процессе, чтобы отмену делать через kill.
_mp_ctx = multiprocessing.get_context("forkserver")

# Сколько ждать мягкого завершения перед SIGKILL.
_TERMINATE_GRACE_SEC = 5


class BasePipelineRequest(BaseModel):
    """Общие поля запуска пайплайна (quick и development)."""
    dataset_id: str = Field(..., description="ID датасета для обучения")
    version_id: str = Field(..., description="ID версии датасета")
    model_name: Optional[str] = Field(None, description="Имя модели (обязательно, если model_id не указан)")
    model_id: Optional[str] = Field(None, description="ID существующей модели — новые версии создаются под ней")
    deployment_constraints: Optional[str] = Field(None, description="Технические возможности прода. Если не указано — агенты сами минимизируют затраты")
    business_requirements: Optional[str] = Field(None, description="Описание бизнес требований. Если не указано — агенты сами максимизируют качество")
    title: Optional[str] = Field(None, description="Название запуска (опционально)")
    tags: List[str] = Field(default_factory=list, description="Теги запуска (опционально)")
    llm_model: Optional[str] = Field(None, description="Модель LLM на этот запуск (override глобальной настройки)")

    @model_validator(mode="after")
    def check_model_target(self):
        if not self.model_id and not (self.model_name and self.model_name.strip()):
            raise ValueError("Нужно указать model_name или model_id")
        return self


class StartResponse(BaseModel):
    discussion_id: str = Field(..., description="ID созданной дискуссии для отслеживания прогресса")
    status: str = Field(..., description="Статус запуска")


def resolve_model_name(model_id: Optional[str], model_name: Optional[str]) -> str:
    """
    Резолв имени модели для тайтлов/тегов и пайплайна. При заданном model_id
    имя берём из реестра ml_models (авторитетное); модель не найдена — 404
    до создания дискуссии (fail fast).
    """
    if model_id:
        model = ml_models_client.get_model(model_id)
        if model is None:
            raise HTTPException(status_code=404, detail=f"Модель с ID {model_id} не найдена")
        return model["name"]
    return model_name.strip()


def _resolve_dataset_names(dataset_id: str, version_id: str) -> tuple[str, str]:
    """Имена датасета и версии для тегов; fallback на id, если сервис недоступен."""
    dataset = get_dataset_details(dataset_id)
    version = get_dataset_version_details(dataset_id, version_id)
    dataset_name = dataset.get("name") if isinstance(dataset, dict) else None
    version_name = version.get("name") if isinstance(version, dict) else None
    return dataset_name or dataset_id, version_name or version_id


def _pipeline_entry(
    discussion_id: str,
    pipeline_label: str,
    dataset_id: str,
    version_id: str,
    llm_model: str,
    id_aliases: dict,
    pipeline_func: Callable[..., object],
    pipeline_kwargs: Dict[str, Any],
) -> None:
    """
    Тело дочернего процесса пайплайна.

    Процесс свежий (forkserver) — contextvars родителя не наследуются, поэтому
    выставляем их из явно переданных значений. Финальный статус дискуссии
    (`completed`/`failed`) пишет сам процесс. Отмена — это kill снаружи (SIGTERM
    завершает процесс без Python-исключения), поэтому ветки cancelled здесь нет:
    статус `cancelled` ставит родитель в stop_pipeline.
    """
    discussion_context.set(discussion_id)
    dataset_context.set(dataset_id, version_id)
    llm_model_context.set(llm_model)
    id_alias_context.set_aliases(id_aliases)
    try:
        result = pipeline_func(**pipeline_kwargs)
        agent_history_client.update_discussion_meta(
            discussion_id, "completed" if result is not None else "failed"
        )
    except Exception as e:
        logger.error(f"Пайплайн {pipeline_label} упал (discussion_id={discussion_id}): {e}")
        agent_history_client.update_discussion_meta(discussion_id, "failed")


def _reap(discussion_id: str, proc: multiprocessing.Process) -> None:
    """Дождаться завершения процесса (сжать zombie) и убрать его из реестра.

    # ponytail: поток-жнец на пайплайн; пайплайнов одновременно единицы.
    # Периодический sweep — если станет много параллельных запусков.
    """
    proc.join()
    process_registry.discard(discussion_id)


def start_pipeline(
    *,
    req: BasePipelineRequest,
    pipeline_name: str,
    agent_roles: List[str],
    title: str,
    model_name: str,
    pipeline_func: Callable[..., object],
    pipeline_kwargs: Dict[str, Any],
) -> StartResponse:
    """
    Общий каркас асинхронного старта пайплайна: создаёт дискуссию синхронно
    (чтобы фронт сразу мог её открыть) и запускает пайплайн в отдельном процессе,
    который при отмене можно убить. pipeline_func + pipeline_kwargs должны быть
    picklable (forkserver) — module-level функция и примитивы, не лямбда.
    """
    discussion_id = str(uuid4())
    # Авто-теги из параметров + пользовательские, без дублей, с сохранением порядка.
    dataset_name, version_name = _resolve_dataset_names(req.dataset_id, req.version_id)
    tags = list(dict.fromkeys([
        *req.tags,
        f"Модель: {model_name}",
        f"Датасет: {dataset_name}",
        f"Версия: {version_name}",
    ]))

    # Карта UUID → читаемое имя, чтобы скраб клиента истории не пропускал сырые
    # идентификаторы датасета/версии/модели в историю агентов.
    id_aliases = {
        req.dataset_id: f"датасет «{dataset_name}»",
        req.version_id: f"версия «{version_name}»",
    }
    if req.model_id:
        id_aliases[req.model_id] = f"модель «{model_name}»"

    agent_history_client.create_discussion(
        discussion_id=discussion_id,
        pipeline=pipeline_name,
        agent_roles=agent_roles,
        title=title,
        tags=tags,
    )

    # Глобально выбранная модель (set_current_model) живёт в родителе и не видна
    # свежему дочернему процессу — резолвим здесь и передаём явно.
    llm_model = req.llm_model or get_current_model()

    proc = _mp_ctx.Process(
        target=_pipeline_entry,
        args=(
            discussion_id, pipeline_name, req.dataset_id, req.version_id,
            llm_model, id_aliases, pipeline_func, pipeline_kwargs,
        ),
        daemon=True,
    )
    proc.start()
    process_registry.register(discussion_id, proc)
    threading.Thread(target=_reap, args=(discussion_id, proc), daemon=True).start()

    return StartResponse(discussion_id=discussion_id, status="started")


def stop_pipeline(discussion_id: str) -> StartResponse:
    """
    Остановить пайплайн, убив его процесс.

    Kill мгновенен: рвутся сокеты текущего LLM-запроса, никаких ретраев и
    последующих вызовов. После kill родитель финализирует состояние (родитель
    знает только discussion_id): отменяет активную задачу обучения в tasker и
    переводит дискуссию в `cancelled` (agent_history каскадом помечает зависшие
    IN_PROGRESS события агента как CANCELLED).
    """
    proc = process_registry.get(discussion_id)
    if proc is None or not proc.is_alive():
        raise HTTPException(
            status_code=404,
            detail=f"Активный пайплайн для дискуссии {discussion_id} не найден",
        )

    proc.terminate()
    proc.join(timeout=_TERMINATE_GRACE_SEC)
    if proc.is_alive():
        logger.warning(f"Процесс пайплайна {discussion_id} не завершился по SIGTERM — SIGKILL")
        proc.kill()
    process_registry.discard(discussion_id)

    # Отменить активное обучение, иначе trainer продолжит жечь ресурсы.
    tasker_client.cancel_discussion_tasks(discussion_id)
    agent_history_client.warning(
        "Работа агентов остановлена пользователем.", discussion_id=discussion_id
    )
    agent_history_client.update_discussion_meta(discussion_id, "cancelled")

    return StartResponse(discussion_id=discussion_id, status="stopping")
