from fastapi import APIRouter, HTTPException, status, Query
from typing import List, Optional

from .storage import MetricsStorage
from .schemas import *

api_routers = APIRouter()

# Инициализируем хранилище
metrics_storage = MetricsStorage(storage_path="ml_metrics_storage")


@api_routers.post("/metrics/update", response_model=bool, status_code=status.HTTP_201_CREATED)
async def update_metric(metric_update: MetricUpdate):
    """
    Обновить или добавить метрику для задачи
    
    - **task_id**: ID задачи обучения
    - **metric_name**: Имя метрики (например, loss, accuracy)
    - **value**: Значение метрики (число или список чисел)
    - **step**: Номер шага/эпохи
    """
    try:
        metrics_storage.update_metric(metric_update)
        return True
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update metric: {str(e)}"
        )


@api_routers.get("/metrics/task/{task_id}", response_model=MetricsResponse)
async def get_task_metrics(task_id: str):
    """
    Получить все метрики для конкретной задачи
    
    - **task_id**: ID задачи обучения
    """
    task_metrics = metrics_storage.get_task_metrics(task_id)
    
    if not task_metrics:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )
    
    return MetricsResponse(
        task_id=task_metrics.task_id,
        metrics=task_metrics.metrics
    )

