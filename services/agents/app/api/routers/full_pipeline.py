from uuid import uuid4
from typing import List, Optional
from fastapi import APIRouter, Query, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from app.core import development_models
from app.core.memory import models_context, discussion_context
from app.services.agent_history import track_discussion, agent_history_client
from app.core.crews.dataset_analyst import AGENT_ROLE as DATASET_ANALYST_ROLE
from app.core.crews.researcher import AGENT_ROLE as RESEARCHER_ROLE
from app.core.crews.ml_engeneer import AGENT_ROLE as ML_ENGINEER_ROLE
from app.core.crews.ml_debuger import AGENT_ROLE as ML_DEBUGER_ROLE
from app.core.crews.reporter import AGENT_ROLE as REPORTER_ROLE
from app.logs import get_logger

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


class DevelopmentRequest(BaseModel):
    dataset_id: str = Field(..., description="ID датасета для анализа")
    version_id: str = Field(..., description="ID версии датасета")
    model_name: str = Field(..., description="Имя модели")
    deployment_constraints: str = Field(..., description="Технические возможности прода")
    business_requirements: str = Field(..., description="Описание бизнес требований")
    denied_hypotheses_info: List[str] = Field(default_factory=list, description="Гипотезы и практики, которые нужно избегать")
    max_iter: int = Field(2, description="Количество попыток обучения")
    title: Optional[str] = Field(None, description="Название запуска (опционально)")
    tags: List[str] = Field(default_factory=list, description="Теги запуска (опционально)")


class StartResponse(BaseModel):
    discussion_id: str = Field(..., description="ID созданной дискуссии для отслеживания прогресса")
    status: str = Field(..., description="Статус запуска")


def _run_development_background(discussion_id: str, req: DevelopmentRequest) -> None:
    """Фоновое выполнение пайплайна development с финализацией статуса дискуссии."""
    discussion_context.set(discussion_id)
    try:
        result = development_models(
            dataset_id=req.dataset_id,
            dataset_version_id=req.version_id,
            model_name=req.model_name,
            deployment_constraints=req.deployment_constraints,
            business_requirements=req.business_requirements,
            denied_hypotheses_info=req.denied_hypotheses_info,
            max_iter=req.max_iter,
        )
        agent_history_client.update_discussion_meta(
            discussion_id, "completed" if result is not None else "failed"
        )
    except Exception as e:
        logger.error(f"Пайплайн development упал (discussion_id={discussion_id}): {e}")
        agent_history_client.update_discussion_meta(discussion_id, "failed")
    finally:
        discussion_context.clear()
        models_context.clear()

@routers.get(
        "/development",
        description="Одна полная итерация. Анализ, рассуждение и запуск тренировки"
)
def development(
    dataset_id: str = Query(..., description="ID датасета для анализа"),
    version_id: str = Query(..., description="ID версии датасета"),
    model_name: str = Query(..., description="Имя модели"),
    deployment_constraints: str = Query(..., description="Технические возможности прода"),
    business_requirements: str = Query(..., description="Описание бизнес требований"),
    denied_hypotheses_info: List[str] = Query(default_factory=list, description="Гипотезы и практики, которые нужно избегать"),
    max_iter: int = Query(2, description="Количество попыток обучения")
):
    discussion_id = str(uuid4())
    try:
        with track_discussion(discussion_id, "development", "Разработка модели", _DEVELOPMENT_AGENT_ROLES):
            result = development_models(
                dataset_id=dataset_id,
                dataset_version_id=version_id,
                model_name=model_name,
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


@routers.post(
    "/development/start",
    status_code=202,
    response_model=StartResponse,
    description="Асинхронный запуск полного пайплайна. Возвращает discussion_id сразу, "
                "пайплайн выполняется в фоне. Прогресс отслеживается через agent_history."
)
def start_development(req: DevelopmentRequest, background_tasks: BackgroundTasks):
    discussion_id = str(uuid4())
    title = req.title or f"Разработка модели «{req.model_name}»"
    # Авто-теги из параметров + пользовательские, без дублей, с сохранением порядка.
    tags = list(dict.fromkeys([*req.tags, req.model_name, req.dataset_id]))

    # Создаём дискуссию СИНХРОННО, чтобы фронт сразу мог её открыть (active),
    # уже с осмысленными метаданными.
    agent_history_client.create_discussion(
        discussion_id=discussion_id,
        pipeline="development",
        agent_roles=_DEVELOPMENT_AGENT_ROLES,
        title=title,
        tags=tags,
    )

    background_tasks.add_task(_run_development_background, discussion_id, req)
    return StartResponse(discussion_id=discussion_id, status="started")