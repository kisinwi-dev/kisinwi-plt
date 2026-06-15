from uuid import uuid4
from typing import Callable, List, Optional

from fastapi import BackgroundTasks, HTTPException
from pydantic import BaseModel, Field, model_validator

from app.core.memory import (
    models_context, discussion_context, llm_model_context, id_alias_context,
    dataset_context,
)
from app.core.cancellation import cancellation_registry, PipelineCancelled
from app.services.agent_history import agent_history_client
from app.services.datasets import get_dataset_details, get_dataset_version_details
from app.services.ml_models import ml_models_client
from app.logs import get_logger

logger = get_logger(__name__)


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


def run_pipeline_background(
    discussion_id: str,
    pipeline: Callable[[], object],
    pipeline_label: str,
    dataset_id: str,
    version_id: str,
    llm_model: Optional[str] = None,
    id_aliases: Optional[dict] = None,
) -> None:
    """Фоновое выполнение пайплайна с финализацией статуса дискуссии."""
    discussion_context.set(discussion_id)
    dataset_context.set(dataset_id, version_id)
    if llm_model:
        llm_model_context.set(llm_model)
    if id_aliases:
        id_alias_context.set_aliases(id_aliases)
    try:
        result = pipeline()
        agent_history_client.update_discussion_meta(
            discussion_id, "completed" if result is not None else "failed"
        )
    except PipelineCancelled:
        logger.warning(f"Пайплайн {pipeline_label} остановлен пользователем (discussion_id={discussion_id})")
        agent_history_client.warning("Работа агентов остановлена пользователем.")
        agent_history_client.update_discussion_meta(discussion_id, "cancelled")
    except Exception as e:
        logger.error(f"Пайплайн {pipeline_label} упал (discussion_id={discussion_id}): {e}")
        agent_history_client.update_discussion_meta(discussion_id, "failed")
    finally:
        cancellation_registry.discard(discussion_id)
        discussion_context.clear()
        dataset_context.clear()
        models_context.clear()
        llm_model_context.clear()
        id_alias_context.clear()


def start_pipeline(
    *,
    req: BasePipelineRequest,
    background_tasks: BackgroundTasks,
    pipeline_name: str,
    agent_roles: List[str],
    title: str,
    model_name: str,
    pipeline: Callable[[], object],
) -> StartResponse:
    """
    Общий каркас асинхронного старта пайплайна: создаёт дискуссию синхронно
    (чтобы фронт сразу мог её открыть) и ставит выполнение пайплайна в фон.
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

    cancellation_registry.register(discussion_id)
    background_tasks.add_task(
        run_pipeline_background, discussion_id, pipeline, pipeline_name,
        req.dataset_id, req.version_id, req.llm_model, id_aliases
    )
    return StartResponse(discussion_id=discussion_id, status="started")


def stop_pipeline(discussion_id: str) -> StartResponse:
    """
    Запросить кооперативную остановку запущенного пайплайна.

    Остановка срабатывает в ближайшей безопасной точке (между крю/итерациями
    или в polling-цикле обучения, который дополнительно отменяет задачу в tasker).
    """
    if not cancellation_registry.request_stop(discussion_id):
        raise HTTPException(
            status_code=404,
            detail=f"Активный пайплайн для дискуссии {discussion_id} не найден",
        )
    return StartResponse(discussion_id=discussion_id, status="stopping")
