from fastapi import APIRouter, Query, HTTPException

from app.core.crews.ml_engeneer import run_ml_engineering, AGENT_ROLE as ML_ENGINEER_ROLE
from app.core.crews.ml_debuger import run_ml_debug, AGENT_ROLE as ML_DEBUGER_ROLE
from app.services.agent_history import track_discussion

routers = APIRouter(
    tags=['engineering']
)

@routers.get(
        "/ml_engineer",
        description="Рассуждения агентов ML-инженеров"
)
def run_ml_engineer(
    discussion_id: str = Query(description="ID дискуссии"),
    business_requirements: str = Query(description="Бизнес требования к модели"),
    deployment_constraints: str = Query(description="Технические требования к модели"),
    dataset_info: str = Query(description="Информация о датасете"),
    researcher_proposals: str = Query(description="Рекомендации от исследователя"),
    dataset_id: str = Query(description="ID датасета"),
    dataset_version_id: str = Query(description="ID версии датасета")
):
    try:

        discussion_context.set(discussion_id)
        agent_history_client.create_discussion(discussion_id, pipeline="ml_engineer", agent_roles=[ML_ENGINEER_ROLE])

        result = run_ml_engineering(
            dataset_info=dataset_info,
            business_requirements=business_requirements,
            deployment_constraints=deployment_constraints,
            researcher_proposals=researcher_proposals,
            dataset_id=dataset_id,
            dataset_version_id=dataset_version_id,
            verbose=True
        )

        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при выполнении: {str(e)}"
        )

@routers.get(
        "/ml_debug",
        description="Агент исправляет ошибку в конфиге запуска обучения мл моделей"
)
def run_ml_debugger(
    discussion_id: str = Query(description="ID дискуссии"),
    error: str = Query(description="Ошибка полученная при обучении"),
    config: str = Query(description="Конфигурации обучения")
):
    try:
        with track_discussion(discussion_id, "ml_debuger", "Отладка обучения", [ML_DEBUGER_ROLE]):
            result = run_ml_debug(
                error=error,
                config=config,
                verbose=True
            )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при выполнении: {str(e)}"
        )
