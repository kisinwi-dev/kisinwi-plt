from fastapi import APIRouter, HTTPException, status, Query
from typing import List, Optional

from .storage import MetricsStorage
from .models import *

api_routers = APIRouter()

# Инициализируем хранилище
metrics_storage = MetricsStorage(storage_path="ml_metrics_storage")


@api_routers.post("/metrics/update", response_model=TaskMetrics, status_code=status.HTTP_201_CREATED)
async def update_metric(metric_update: MetricUpdate):
    """
    Обновить или добавить метрику для задачи
    
    - **task_id**: ID задачи обучения
    - **metric_name**: Имя метрики (например, loss, accuracy)
    - **value**: Значение метрики (число или список чисел)
    - **step**: Номер шага/эпохи
    """
    try:
        task_metrics = metrics_storage.update_metric(metric_update)
        return task_metrics
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


@api_routers.get("/metrics/task/{task_id}/{metric_name}", response_model=MetricData)
async def get_metric_data(
    task_id: str, 
    metric_name: str,
    start_step: Optional[int] = Query(None, description="Начальный шаг"),
    end_step: Optional[int] = Query(None, description="Конечный шаг")
):
    """
    Получить данные конкретной метрики для задачи с возможностью фильтрации по шагам
    
    - **task_id**: ID задачи обучения
    - **metric_name**: Имя метрики
    - **start_step**: Начальный шаг для фильтрации (опционально)
    - **end_step**: Конечный шаг для фильтрации (опционально)
    """
    metric_data = metrics_storage.get_metric_data(task_id, metric_name)
    
    if not metric_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Metric {metric_name} not found for task {task_id}"
        )
    
    # Фильтрация по шагам
    if start_step is not None or end_step is not None:
        filtered_steps = []
        filtered_values = []
        
        for step, value in zip(metric_data.steps, metric_data.values):
            if start_step is not None and step < start_step:
                continue
            if end_step is not None and step > end_step:
                continue
            filtered_steps.append(step)
            filtered_values.append(value)
        
        return MetricData(steps=filtered_steps, values=filtered_values)
    
    return metric_data


@api_routers.get("/metrics/tasks")
async def list_tasks():
    """
    Получить список всех задач с метриками
    """
    tasks = metrics_storage.get_all_tasks()
    
    # Добавляем информацию о количестве метрик для каждой задачи
    tasks_info = []
    for task_id in tasks:
        task_metrics = metrics_storage.get_task_metrics(task_id)
        if task_metrics:
            tasks_info.append({
                "task_id": task_id,
                "metrics_count": len(task_metrics.metrics),
                "metrics_names": list(task_metrics.metrics.keys())
            })
    
    return {"tasks": tasks_info}


@api_routers.delete("/metrics/task/{task_id}")
async def delete_task_metrics(task_id: str):
    """
    Удалить все метрики для задачи
    
    - **task_id**: ID задачи обучения
    """
    success = metrics_storage.delete_task_metrics(task_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete metrics for task {task_id}"
        )
    
    return {"message": f"Successfully deleted metrics for task {task_id}"}


@api_routers.delete("/metrics/task/{task_id}/{metric_name}")
async def delete_metric(task_id: str, metric_name: str):
    """
    Удалить конкретную метрику из задачи
    
    - **task_id**: ID задачи обучения
    - **metric_name**: Имя метрики
    """
    success = metrics_storage.delete_metric(task_id, metric_name)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Metric {metric_name} not found for task {task_id}"
        )
    
    return {"message": f"Successfully deleted metric {metric_name} for task {task_id}"}


@api_routers.post("/metrics/backup")
async def backup_metrics(backup_path: str = Query(..., description="Путь для резервной копии")):
    """
    Создать резервную копию всех метрик
    """
    success = metrics_storage.backup_metrics(backup_path)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create backup"
        )
    
    return {"message": f"Successfully created backup at {backup_path}"}


@api_routers.get("/metrics/health")
async def health_check():
    """
    Проверка здоровья сервиса
    """
    return {
        "status": "healthy",
        "storage_path": str(metrics_storage.storage_path),
        "tasks_count": len(metrics_storage.get_all_tasks())
    }


# Дополнительный эндпоинт для массового обновления метрик
@api_routers.post("/metrics/batch-update")
async def batch_update_metrics(updates: List[MetricUpdate]):
    """
    Массовое обновление нескольких метрик
    
    - **updates**: Список обновлений метрик
    """
    results = []
    for update in updates:
        try:
            task_metrics = metrics_storage.update_metric(update)
            results.append({
                "task_id": update.task_id,
                "metric_name": update.metric_name,
                "status": "success"
            })
        except Exception as e:
            results.append({
                "task_id": update.task_id,
                "metric_name": update.metric_name,
                "status": "failed",
                "error": str(e)
            })
    
    return {"results": results}