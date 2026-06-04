from uuid import uuid4
from typing import List
from fastapi import APIRouter, Query, HTTPException

from app.core import development_models
from app.core.memory import models_context
from app.services.agent_history import track_discussion
from app.core.crews.dataset_analyst import AGENT_ROLE as DATASET_ANALYST_ROLE
from app.core.crews.researcher import AGENT_ROLE as RESEARCHER_ROLE
from app.core.crews.ml_engeneer import AGENT_ROLE as ML_ENGINEER_ROLE
from app.core.crews.ml_debuger import AGENT_ROLE as ML_DEBUGER_ROLE
from app.core.crews.reporter import AGENT_ROLE as REPORTER_ROLE

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