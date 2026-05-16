from uuid import uuid4
from typing import List
from fastapi import APIRouter, Query, HTTPException

from app.core import development_models
from app.core.memory import discussion_context, models_context

routers = APIRouter(
    tags=['pipeline']
)

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
    max_iter: int = Query("", description="Количество попыток обучения")
):
    try:
        discussion_context.set(str(uuid4()))

        return development_models(
            dataset_id=dataset_id,
            dataset_version_id=version_id,
            model_name=model_name,
            deployment_constraints=deployment_constraints,
            business_requirements=business_requirements,
            denied_hypotheses_info=denied_hypotheses_info,
            max_iter=max_iter,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при выполнении: {str(e)}"
        )
    finally:
        discussion_context.clear()
        models_context.clear()