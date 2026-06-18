from fastapi import APIRouter

from app.core.quick_pipeline import quick_training_models
from app.core.crews.ml_engeneer_quick import AGENT_ROLE as ML_ENGINEER_ROLE
from app.core.crews.metrics_analyst import AGENT_ROLE as METRICS_ANALYST_ROLE
from app.logs import get_logger
from ._pipeline_common import (
    BasePipelineRequest, StartResponse, resolve_model_name, start_pipeline,
)

logger = get_logger(__name__)

routers = APIRouter(
    tags=['pipeline']
)

_QUICK_AGENT_ROLES = [
    ML_ENGINEER_ROLE,
    METRICS_ANALYST_ROLE,
]


class QuickRequest(BasePipelineRequest):
    pass


@routers.post(
    "/quick/start",
    status_code=202,
    response_model=StartResponse,
    description="Асинхронный запуск быстрого пайплайна (ML-инженер + аналитик метрик). "
                "Один проход без итераций: конфигурация обучения, обучение, анализ метрик. "
                "Возвращает discussion_id сразу, пайплайн выполняется в фоне."
)
def start_quick_training(req: QuickRequest):
    model_name = resolve_model_name(req.model_id, req.model_name)
    title = req.title or f"Быстрое обучение «{model_name}»"

    return start_pipeline(
        req=req,
        pipeline_name="quick_training",
        agent_roles=_QUICK_AGENT_ROLES,
        title=title,
        model_name=model_name,
        pipeline_func=quick_training_models,
        pipeline_kwargs=dict(
            dataset_id=req.dataset_id,
            dataset_version_id=req.version_id,
            model_name=model_name,
            model_id=req.model_id,
            deployment_constraints=req.deployment_constraints,
            business_requirements=req.business_requirements,
        ),
    )
