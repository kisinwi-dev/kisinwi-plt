from fastapi import APIRouter, Query, HTTPException

from app.core.crews.metrics_analyst import run_metrics_analyst
from app.core.crews.dataset_analyst import run_dataset_analyst

routers = APIRouter(
    prefix='/analytics',
    tags=['analytics']
)

@routers.get(
        "/datasets",
        description="Анализ датасета"
)
def analyze_datasets(
    discussion_id: str = Query(..., description="ID диалога"),
    dataset_id: str = Query(..., description="ID датасета"),
    dataset_version_id: str = Query(..., description="ID версии датасета"),
):
    try:
        result = run_dataset_analyst(
            discussion_id=discussion_id,
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
        "/ml_metrics",
        description="Анализ метрик"
)
def analyze_ml_metric(
    discussion_id: str = Query(..., description="ID диалога"),
    business_goal: str = Query(..., description="Требования бизнеса"),
    model_id: str = Query(..., description="ID модели")
):
    try:
        result = run_metrics_analyst(
            discussion_id=discussion_id,
            business_goal=business_goal,
            model_id=model_id,
            verbose=True
        )
        
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при выполнении: {str(e)}"
        )
