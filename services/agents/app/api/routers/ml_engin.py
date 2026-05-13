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
    training_objective: str = Query(description="Цель обучения"),
    researcher_proposals: str = Query(description="Количество инженеров")
):
    try:
        
        discussion_context.set(discussion_id)

        result = run_ml_engineering(
            business_requirements=business_requirements,
            deployment_constraints=deployment_constraints,
            training_objective=training_objective,
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
