from fastapi import APIRouter, Query

from app.core.crews.analytics import run_analysis

routers = APIRouter(
    tags=['analytics']
)

@routers.get(
        "/analytic",
        description="Анализ датасета"
)
def analyze_dataset(
    dataset_id: str = Query(..., description="ID датасета"),
    version_id: str | None = Query(None, description="ID версии")
):
    # __WARNING__ НУЖНО ДОБАВИТЬ ОБРАБОТЧИКИ ОШИБОК
    result, metrics = run_analysis(dataset_id, version_id)
    
    return {
        "dataset_id": dataset_id,
        "version_id": version_id,
        "analysis": result,
        "metrics": metrics
    }
