from uuid import uuid4
from typing import List, Optional
from fastapi import APIRouter, Query, HTTPException, BackgroundTasks
from pydantic import Field

from app.core import development_models
from app.core.memory import models_context, llm_model_context
from app.services.agent_history import track_discussion
from app.core.crews.dataset_analyst import AGENT_ROLE as DATASET_ANALYST_ROLE
from app.core.crews.researcher import AGENT_ROLE as RESEARCHER_ROLE
from app.core.crews.ml_engeneer import AGENT_ROLE as ML_ENGINEER_ROLE
from app.core.crews.ml_debuger import AGENT_ROLE as ML_DEBUGER_ROLE
from app.core.crews.reporter import AGENT_ROLE as REPORTER_ROLE
from app.logs import get_logger
from ._pipeline_common import (
    BasePipelineRequest, StartResponse, resolve_model_name, start_pipeline,
    stop_pipeline,
)

logger = get_logger(__name__)

routers = APIRouter(
    tags=['pipeline']
)

_DEVELOPMENT_AGENT_ROLES = [
    DATASET_ANALYST_ROLE,
    RESEARCHER_ROLE,
    ML_ENGINEER_ROLE,
    ML_DEBUGER_ROLE,
    REPORTER_ROLE,
]


class DevelopmentRequest(BasePipelineRequest):
    denied_hypotheses_info: List[str] = Field(default_factory=list, description="Гипотезы и практики, которые нужно избегать")
    max_iter: int = Field(0, ge=0, description="Количество попыток обучения. 0 — агент определяет сам")


@routers.get(
        "/development",
        description="Одна полная итерация. Анализ, рассуждение и запуск тренировки"
)
def development(
    dataset_id: str = Query(..., description="ID датасета для анализа"),
    version_id: str = Query(..., description="ID версии датасета"),
    model_name: str = Query(..., description="Имя модели"),
    deployment_constraints: Optional[str] = Query(None, description="Технические возможности прода. Если не указано — агенты сами минимизируют затраты"),
    business_requirements: Optional[str] = Query(None, description="Описание бизнес требований. Если не указано — агенты сами максимизируют качество"),
    denied_hypotheses_info: List[str] = Query(default_factory=list, description="Гипотезы и практики, которые нужно избегать"),
    max_iter: int = Query(0, ge=0, description="Количество попыток обучения. 0 — агент определяет сам"),
    model_id: Optional[str] = Query(None, description="ID существующей модели — новые версии создаются под ней"),
    llm_model: Optional[str] = Query(None, description="Модель LLM на этот запуск (override глобальной настройки)")
):
    discussion_id = str(uuid4())
    if llm_model:
        llm_model_context.set(llm_model)
    try:
        with track_discussion(discussion_id, "development", "Разработка модели", _DEVELOPMENT_AGENT_ROLES):
            result = development_models(
                dataset_id=dataset_id,
                dataset_version_id=version_id,
                model_name=model_name,
                model_id=model_id,
                deployment_constraints=deployment_constraints,
                business_requirements=business_requirements,
                denied_hypotheses_info=denied_hypotheses_info,
                max_iter=max_iter,
            )
            if result is None:
                raise HTTPException(status_code=422, detail="Пайплайн завершился без результата")
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при выполнении: {str(e)}"
        )
    finally:
        models_context.clear()
        llm_model_context.clear()


@routers.post(
    "/development/start",
    status_code=202,
    response_model=StartResponse,
    description="Асинхронный запуск полного пайплайна. Возвращает discussion_id сразу, "
                "пайплайн выполняется в фоне. Прогресс отслеживается через agent_history."
)
def start_development(req: DevelopmentRequest, background_tasks: BackgroundTasks):
    model_name = resolve_model_name(req.model_id, req.model_name)
    title = req.title or f"Разработка модели «{model_name}»"

    return start_pipeline(
        req=req,
        background_tasks=background_tasks,
        pipeline_name="development",
        agent_roles=_DEVELOPMENT_AGENT_ROLES,
        title=title,
        model_name=model_name,
        pipeline=lambda: development_models(
            dataset_id=req.dataset_id,
            dataset_version_id=req.version_id,
            model_name=model_name,
            model_id=req.model_id,
            deployment_constraints=req.deployment_constraints,
            business_requirements=req.business_requirements,
            denied_hypotheses_info=req.denied_hypotheses_info,
            max_iter=req.max_iter,
        ),
    )


@routers.post(
    "/pipeline/{discussion_id}/stop",
    status_code=202,
    response_model=StartResponse,
    description="Остановить работу агентов в запущенном пайплайне (development или quick). "
                "Остановка кооперативная: срабатывает в ближайшей безопасной точке; активная "
                "задача обучения отменяется в tasker.",
)
def stop_pipeline_endpoint(discussion_id: str):
    return stop_pipeline(discussion_id)
