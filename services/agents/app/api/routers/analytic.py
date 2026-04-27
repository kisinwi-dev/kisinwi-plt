from fastapi import APIRouter, Query, HTTPException

from app.core.crews.analytics.metrics import run_analysis as metrics_analyse
from app.core.crews.analytics.datasets import run_analysis as data_analyse

routers = APIRouter(
    prefix='/analytics',
    tags=['analytics']
)

@routers.get(
        "/metrics",
        description="Анализ метрик"
)
def analyze_metrics(
    task_id: str = Query(..., description="ID задачи"),
    dataset_id: str = Query(..., description="ID датасета"),
    version_id: str = Query(None, description="ID версии"),
    bus_req: str = Query(None, description="Бизне требования к модели")
):
    try:
        result, metrics = metrics_analyse(
            task_id,
            dataset_id=dataset_id,
            version_id=version_id,
            bus_req=bus_req
        )
        
        return {
            "task_id": task_id,
            "dataset_id": dataset_id,
            "version_id": version_id,
            "analysis": result,
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при выполнении: {str(e)}"
        )


@routers.get(
        "/data",
        description="Анализ датасета"
)
def analyze_dataset(
    dataset_id: str = Query(..., description="ID датасета"),
    version_id: str | None = Query(None, description="ID версии")
):
    try:
        result, metrics = data_analyse(dataset_id, version_id)
        
        return {
            "dataset_id": dataset_id,
            "version_id": version_id,
            "analysis": result,
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при выполнении: {str(e)}"
        )
