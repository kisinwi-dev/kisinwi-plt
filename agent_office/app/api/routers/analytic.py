from fastapi import APIRouter, Query

from app.core.analytic.crews import run_analysis

routers = APIRouter(
    tags=['analytics']
)

@routers.get(
        "/analytic",
        description="Анализ имеющихся данных и вывод по ним"
)
def analytic_data_for_datasets_service(
    dataset_id: str = Query(..., description="ID датасета для анализа"),
    version_id: str = Query(None, description="ID версии датасета")
):
    """
    Анализ датасета.
    
    Параметры:
    - dataset_id: ID датасета
    - version_id: ID версии
    """
    
    result, metrics = run_analysis(dataset_id, version_id)
    
    return {
        "dataset_id": dataset_id,
        "version_id": version_id,
        "analysis": result.raw,
        "metrics": metrics
    }
