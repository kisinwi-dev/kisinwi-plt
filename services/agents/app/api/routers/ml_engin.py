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
    business_goal: str = Query(description="Бизнес требования к модели"),
    technical_goal: str = Query(description="Технические требования к модели"),
    training_goal: str = Query(description="Цель обучения"),
    researcher_info: str = Query(description="Количество инженеров")
):
    try:
        
        discussion_context.set(discussion_id)

        result = run_ml_engineering(
            business_goal=business_goal,
            technical_goal=technical_goal,
            training_goal=training_goal,
            researcher_info=researcher_info,
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
