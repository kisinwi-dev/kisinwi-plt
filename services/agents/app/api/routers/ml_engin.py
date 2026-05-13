from fastapi import APIRouter, Query, HTTPException

from app.core.crews.ml_engeneer import run_ml_engineering
from app.core.discussion import discussion_context

routers = APIRouter(
    tags=['engineering']
)

@routers.get(
        "/ml_engineer",
        description="Рассуждения агентов ML-инженеров"
)
def run_etp(
    discussion_id: str = Query(description="ID дискуссии"),
    business_requirements: str = Query(description="Бизнес требования к модели"),
    deployment_constraints: str = Query(description="Технические требования к модели"),
    dataset_info: str = Query(description="Информация о датасете"),
    researcher_proposals: str = Query(description="Рекомендации от исследователя")
):
    try:
        
        discussion_context.set(discussion_id)

        result = run_ml_engineering(
            dataset_info=dataset_info,
            business_requirements=business_requirements,
            deployment_constraints=deployment_constraints,
            researcher_proposals=researcher_proposals,
            verbose=True
        )

        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при выполнении: {str(e)}"
        )
    finally:
        discussion_context.clear()
