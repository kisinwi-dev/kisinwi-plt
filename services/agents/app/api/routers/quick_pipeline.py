from uuid import uuid4
from typing import List, Optional
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel, Field, model_validator

from app.core.quick_pipeline import quick_training_models
from app.core.memory import models_context, discussion_context
from app.services.agent_history import agent_history_client
from app.core.crews.ml_engeneer import AGENT_ROLE as ML_ENGINEER_ROLE
from app.core.crews.metrics_analyst import AGENT_ROLE as METRICS_ANALYST_ROLE
from app.logs import get_logger
from .full_pipeline import StartResponse, resolve_model_name

logger = get_logger(__name__)

routers = APIRouter(
    tags=['pipeline']
)

_QUICK_AGENT_ROLES = [
    ML_ENGINEER_ROLE,
    METRICS_ANALYST_ROLE,
]


class QuickRequest(BaseModel):
    dataset_id: str = Field(..., description="ID датасета для обучения")
    version_id: str = Field(..., description="ID версии датасета")
    model_name: Optional[str] = Field(None, description="Имя модели (обязательно, если model_id не указан)")
    model_id: Optional[str] = Field(None, description="ID существующей модели — новые версии создаются под ней")
    deployment_constraints: Optional[str] = Field(None, description="Технические возможности прода. Если не указано — агенты сами минимизируют затраты")
    business_requirements: Optional[str] = Field(None, description="Описание бизнес требований. Если не указано — агенты сами максимизируют качество")
    title: Optional[str] = Field(None, description="Название запуска (опционально)")
    tags: List[str] = Field(default_factory=list, description="Теги запуска (опционально)")

    @model_validator(mode="after")
    def check_model_target(self):
        if not self.model_id and not (self.model_name and self.model_name.strip()):
            raise ValueError("Нужно указать model_name или model_id")
        return self


def _run_quick_background(discussion_id: str, req: QuickRequest, model_name: str) -> None:
    """Фоновое выполнение быстрого пайплайна с финализацией статуса дискуссии."""
    discussion_context.set(discussion_id)
    try:
        result = quick_training_models(
            dataset_id=req.dataset_id,
            dataset_version_id=req.version_id,
            model_name=model_name,
            model_id=req.model_id,
            deployment_constraints=req.deployment_constraints,
            business_requirements=req.business_requirements,
        )
        agent_history_client.update_discussion_meta(
            discussion_id, "completed" if result is not None else "failed"
        )
    except Exception as e:
        logger.error(f"Пайплайн quick_training упал (discussion_id={discussion_id}): {e}")
        agent_history_client.update_discussion_meta(discussion_id, "failed")
    finally:
        discussion_context.clear()
        models_context.clear()


@routers.post(
    "/quick/start",
    status_code=202,
    response_model=StartResponse,
    description="Асинхронный запуск быстрого пайплайна (ML-инженер + аналитик метрик). "
                "Один проход без итераций: конфигурация обучения, обучение, анализ метрик. "
                "Возвращает discussion_id сразу, пайплайн выполняется в фоне."
)
def start_quick_training(req: QuickRequest, background_tasks: BackgroundTasks):
    model_name = resolve_model_name(req.model_id, req.model_name)

    discussion_id = str(uuid4())
    title = req.title or f"Быстрое обучение «{model_name}»"
    # Авто-теги из параметров + пользовательские, без дублей, с сохранением порядка.
    tags = list(dict.fromkeys([*req.tags, model_name, req.dataset_id]))

    # Создаём дискуссию СИНХРОННО, чтобы фронт сразу мог её открыть (active),
    # уже с осмысленными метаданными.
    agent_history_client.create_discussion(
        discussion_id=discussion_id,
        pipeline="quick_training",
        agent_roles=_QUICK_AGENT_ROLES,
        title=title,
        tags=tags,
    )

    background_tasks.add_task(_run_quick_background, discussion_id, req, model_name)
    return StartResponse(discussion_id=discussion_id, status="started")
