from typing import List
from fastapi import APIRouter
from pydantic import Field

from app.core import development_models
from app.core.crews.dataset_analyst import AGENT_ROLE as DATASET_ANALYST_ROLE
from app.core.crews.researcher import AGENT_ROLE as RESEARCHER_ROLE
from app.core.crews.ml_engeneer import AGENT_ROLE as ML_ENGINEER_ROLE
from app.core.crews.ml_debuger import AGENT_ROLE as ML_DEBUGER_ROLE
from app.core.crews.reporter import AGENT_ROLE as REPORTER_ROLE
from ._pipeline_common import (
    BasePipelineRequest, StartResponse, resolve_model_name, start_pipeline,
    stop_pipeline,
)

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
    skip_dataset_check: bool = Field(False, description="Обучать даже если аналитик данных забраковал датасет")


@routers.post(
    "/development/start",
    status_code=202,
    response_model=StartResponse,
    description="Асинхронный запуск полного пайплайна. Возвращает discussion_id сразу, "
                "пайплайн выполняется в фоне. Прогресс отслеживается через agent_history."
)
def start_development(req: DevelopmentRequest):
    model_name = resolve_model_name(req.model_id, req.model_name)
    title = req.title or f"Разработка модели «{model_name}»"

    return start_pipeline(
        req=req,
        pipeline_name="development",
        agent_roles=_DEVELOPMENT_AGENT_ROLES,
        title=title,
        model_name=model_name,
        pipeline_func=development_models,
        pipeline_kwargs=dict(
            dataset_id=req.dataset_id,
            dataset_version_id=req.version_id,
            model_name=model_name,
            model_id=req.model_id,
            deployment_constraints=req.deployment_constraints,
            business_requirements=req.business_requirements,
            denied_hypotheses_info=req.denied_hypotheses_info,
            max_iter=req.max_iter,
            skip_dataset_check=req.skip_dataset_check,
        ),
    )


@routers.post(
    "/pipeline/{discussion_id}/stop",
    status_code=202,
    response_model=StartResponse,
    description="Остановить работу агентов в запущенном пайплайне (development или quick). "
                "Процесс пайплайна убивается немедленно; активная задача обучения "
                "отменяется в tasker, дискуссия помечается cancelled.",
)
def stop_pipeline_endpoint(discussion_id: str):
    return stop_pipeline(discussion_id)
